import json
import re
from collections.abc import AsyncIterator
from types import ModuleType

import httpx
import structlog
from langchain_core.messages import AIMessage, ToolMessage

from agent_orchestrator.messages import to_langchain_messages
from agent_orchestrator.schemas.turn import RunTurnRequest

log = structlog.get_logger()

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_content(content: object) -> object:
    """Qwen3.5 served via vLLM without a --reasoning-parser flag leaks raw reasoning into the
    ordinary content field instead of a separate reasoning_content field. Confirmed live in two
    shapes: a well-formed <think>...</think> block, and -- more often -- just a lone </think>
    with no opening tag, because vLLM's chat template pre-fills "<think>\n" as a prompt
    continuation the completion never repeats, so the model's own generated text is
    "reasoning...</think>\n\nactual answer". Stripped here, at the boundary where this untrusted
    model output is turned into a message a user or Rails will see -- not upstream, and not by
    asking every future caller to remember to do it.
    """
    if not isinstance(content, str):
        return content
    content = _THINK_BLOCK.sub("", content)
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1]
    return content.strip()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _tool_result_text(content: object) -> str:
    """langchain-mcp-adapters' ToolMessage.content is the raw MCP content-part array
    ([{"type": "text", "text": "...", "id": "..."}, ...]), not just the tool's own result text.
    Confirmed live: without this, ChatMessage#content (and what the model sees on its next
    turn) ends up holding that whole wrapper serialized as a string, so the frontend's tool
    result card parses it as a one-element array instead of the actual {snowfall_report_id:...}
    object it expects. Every McpTools::* adapter in Operator-Portal returns exactly one text
    part (MCP::Tool::Response.new([{type: "text", text: result.to_json}])), so unwrapping to
    that first part's text is correct for every tool this app has today.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
        return content[0]["text"]
    return json.dumps(content)


async def stream_turn(app_module: ModuleType, request: RunTurnRequest) -> AsyncIterator[str]:
    """Runs one turn and yields already-SSE-formatted strings. Each finalized message (assistant
    or tool) is yielded as its own event the moment the graph produces it -- Rails persists a
    ChatMessage per event as it streams in, preserving the same "nothing runs twice on a mid-loop
    failure" property Chat::TurnHandler's synchronous persist-immediately loop had (see that
    class's own comment) now that the loop runs in a separate process.
    """
    log.info("turn.start", app_id=request.app_id, conversation_id=request.conversation_id, turn_id=request.turn_id)

    # except* forbids return/break/continue directly in its block (PEP 654) -- a build failure
    # is recorded here and turned into a yield + early return once we're back in plain code.
    build_failure: tuple[str, str] | None = None
    try:
        agent, provider_map = await app_module.build_agent(
            identity_token=request.identity_token,
            model_overrides=request.model_overrides.model_dump(exclude_none=True),
        )
    except* httpx.HTTPStatusError as eg:
        # The MCP client's transport runs inside an anyio task group, so a 401 from Rails'
        # McpController arrives wrapped in an ExceptionGroup rather than as a bare
        # HTTPStatusError -- except* unwraps that (and matches a bare one too, per PEP 654).
        if any(e.response.status_code == 401 for e in eg.exceptions):
            build_failure = ("identity_rejected", "identity token rejected by the application")
        else:
            build_failure = ("internal_error", str(eg))
    except* Exception as eg:  # noqa: BLE001 -- last-resort boundary; a bad request must not 500 the process
        build_failure = ("internal_error", str(eg))

    if build_failure is not None:
        error_type, detail = build_failure
        log.warning("turn.build_agent_failed", turn_id=request.turn_id, error_type=error_type, detail=detail)
        yield _sse("error", {"error_type": error_type, "detail": detail})
        return

    roundtrips_used = 0
    try:
        async for chunk in agent.astream(
            {"messages": to_langchain_messages(request.messages)},
            stream_mode="updates",
            config={"recursion_limit": 2 * app_module.MAX_TOOL_ROUNDTRIPS + 4},
        ):
            for delta in chunk.values():
                if not isinstance(delta, dict) or "messages" not in delta:
                    continue

                for m in delta["messages"]:
                    if isinstance(m, AIMessage):
                        roundtrips_used += 1
                        model_used = m.response_metadata.get("model_name")
                        yield _sse(
                            "message",
                            {
                                "role": "assistant",
                                "content": _clean_content(m.content),
                                # Only the primary vLLM model is wrapped with
                                # ReasoningPreservingChatOpenAI (see apps/operator_portal.py) --
                                # the OpenAI/Gemini fallbacks never populate this key, so it's
                                # None for them, same as any other reasoning-less turn.
                                "reasoning_content": m.additional_kwargs.get("reasoning_content"),
                                "tool_calls": [
                                    {"id": tc["id"], "name": tc["name"], "arguments": tc["args"]}
                                    for tc in (m.tool_calls or [])
                                ],
                                "provider_used": provider_map.get(model_used, "unknown"),
                                "model_used": model_used,
                            },
                        )
                    elif isinstance(m, ToolMessage):
                        content = _tool_result_text(m.content)
                        yield _sse(
                            "message",
                            {
                                "role": "tool",
                                "tool_call_id": m.tool_call_id,
                                # "name", not "tool_name" -- matches Operator-Portal's existing
                                # normalized transcript wire format (see schemas/turn.py).
                                "name": m.name,
                                "content": content,
                            },
                        )
    except Exception as exc:  # noqa: BLE001 -- everything reachable here is "all providers/tools exhausted"
        log.error("turn.failed", turn_id=request.turn_id, error=str(exc))
        yield _sse("error", {"error_type": "all_providers_failed", "detail": str(exc)})
        return

    log.info("turn.done", turn_id=request.turn_id, roundtrips_used=roundtrips_used)
    yield _sse("done", {"roundtrips_used": roundtrips_used})
