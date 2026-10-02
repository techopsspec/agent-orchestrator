"""Hard guard on Operator-Portal's write tools: nothing is written until the user has said yes
to a summary of exactly what will be written.

A snowfall reading reprices real pay for every snow stop in that zone on that date, and
testing (2026-10-02) showed the model will happily write on the same turn it was asked --
including a back-dated 12" reading in a real zone, and a relative "tomorrow" resolved to the
wrong date. The system prompt asks for a confirmation step, but a 4B model doesn't reliably
follow prompts, so this middleware enforces it in code: a write tool call only reaches Rails
when

  1. the latest user message is an affirmative reply ("yes", "confirm", "go ahead", ...),
  2. the assistant message just before it restated the zone, inches and date being written
     (so "yes" can't approve something the user was never shown), and
  3. no write has already run since that "yes" (one confirmation, one write).

Anything else comes back to the model as an error tool result telling it to ask first -- the
model then asks, and the user's next "yes" lets the same call through.
"""

import re
from datetime import date

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

WRITE_TOOLS = frozenset({"report_snowfall_reading"})

_AFFIRMATIVE = re.compile(
    r"^(yes|y|yep|yeah|yup|confirm|confirmed|correct|that'?s (right|correct)|go ahead|"
    r"do it|ok|okay|sure|please do|record it|save it)\b"
)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def is_affirmative(text: str) -> bool:
    normalized = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    return bool(_AFFIRMATIVE.match(normalized))


def _date_forms(raw: str) -> list[str]:
    try:
        d = date.fromisoformat(str(raw))
    except ValueError:
        return [str(raw).lower()]
    return [d.isoformat(), f"{d:%B} {d.day}, {d.year}".lower(), f"{d:%b} {d.day}, {d.year}".lower()]


def summary_matches(summary: str, args: dict) -> bool:
    """True when the assistant's summary names this call's zone, inches and date."""
    text = summary.lower()
    zone = str(args.get("zone_name", "")).strip().lower()
    if not zone or zone not in text:
        return False

    try:
        inches = float(args.get("inches"))
    except (TypeError, ValueError):
        return False
    if not any(float(n) == inches for n in re.findall(r"\d+(?:\.\d+)?", text)):
        return False

    return any(form in text for form in _date_forms(args.get("reported_on", "")))


def check_write_allowed(messages: list[BaseMessage], args: dict) -> str | None:
    """None when the write may proceed, otherwise the reason it may not."""
    last_human = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None)
    if last_human is None or not is_affirmative(_text(messages[last_human])):
        return "the user has not confirmed this write yet"

    if any(isinstance(m, ToolMessage) and m.name in WRITE_TOOLS for m in messages[last_human + 1 :]):
        return "one confirmation allows one write, and a write already ran since the user's last confirmation"

    summary = next(
        (_text(m) for m in reversed(messages[:last_human]) if isinstance(m, AIMessage) and _text(m).strip()),
        "",
    )
    if not summary_matches(summary, args):
        return "the zone, inches and date being written don't match what the user was asked to confirm"
    return None


def _refusal(request, reason: str) -> ToolMessage:
    call = request.tool_call
    return ToolMessage(
        content=(
            f"NOT RECORDED: {reason}. Before recording, reply to the user with one short message "
            "that states the zone name, the inches, and the date as YYYY-MM-DD with its weekday, "
            "and ask them to reply yes to confirm. Do not call this tool again until they do."
        ),
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
    )


class WriteConfirmationMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in WRITE_TOOLS:
            return await handler(request)

        reason = check_write_allowed(request.state["messages"], request.tool_call.get("args") or {})
        if reason is not None:
            return _refusal(request, reason)
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in WRITE_TOOLS:
            return handler(request)

        reason = check_write_allowed(request.state["messages"], request.tool_call.get("args") or {})
        if reason is not None:
            return _refusal(request, reason)
        return handler(request)
