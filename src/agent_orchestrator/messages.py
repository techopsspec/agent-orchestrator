from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent_orchestrator.schemas.turn import MessageIn


def to_langchain_messages(messages: list[MessageIn]) -> list[BaseMessage]:
    """Rails sends the full transcript so far on every turn (1:1 with ChatMessage#to_llm_message)
    -- this is the other half of that mapping, into LangChain's message types."""
    result: list[BaseMessage] = []
    for m in messages:
        if m.role == "user":
            result.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            tool_calls = [{"id": tc.id, "name": tc.name, "args": tc.arguments} for tc in m.tool_calls]
            result.append(AIMessage(content=m.content or "", tool_calls=tool_calls))
        elif m.role == "tool":
            result.append(ToolMessage(content=m.content or "", tool_call_id=m.tool_call_id, name=m.tool_name))
        else:
            raise ValueError(f"unknown message role: {m.role!r}")
    return result
