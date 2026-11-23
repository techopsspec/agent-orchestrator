"""apps/recruiting_sms.py: prompt construction, FAQ unwrapping, and the real agent wiring
driven by a scripted model (same approach as test_write_confirmation_agent.py)."""

import asyncio
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent_orchestrator.apps import recruiting_sms
from agent_orchestrator.apps.recruiting_sms import faq_text, make_agent, system_prompt
from agent_orchestrator.apps.registry import APPS

FAQ = "Lawn operators are paid per stop. Season runs April to October."


class ScriptedModel(BaseChatModel):
    """Returns queued AIMessages in order and records the prompts it was given."""

    script: list
    seen: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def run(script, text, locale="en", faq=FAQ, handoffs=None):
    handoffs = [] if handoffs is None else handoffs

    @tool
    def request_human_handoff(reason: str) -> str:
        """Hand the conversation to a person."""
        handoffs.append(reason)
        return '{"ok": true}'

    model = ScriptedModel(script=script, seen=[])
    agent = make_agent(model, [request_human_handoff], [], locale=locale, faq=faq)
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(text)]}, config={"recursion_limit": 10}))
    return result["messages"], model.seen, handoffs


def test_registered():
    assert APPS["recruiting_sms"] is recruiting_sms


def test_prompt_names_the_language_and_carries_the_faq():
    es = system_prompt("es", FAQ)
    assert "Reply only in Spanish" in es
    assert FAQ in es
    assert "request_human_handoff" in es
    assert "Reply only in English" in system_prompt("en", FAQ)


def test_unknown_locale_falls_back_to_english():
    assert "Reply only in English" in system_prompt("fr", FAQ)


def test_empty_faq_tells_the_model_to_hand_off_everything():
    assert "hand off every question" in system_prompt("en", "   ")


def test_faq_text_unwraps_every_shape_the_mcp_adapter_returns():
    payload = json.dumps({"locale": "en", "faq": FAQ})
    assert faq_text(payload) == FAQ
    assert faq_text([{"type": "text", "text": payload}]) == FAQ
    assert faq_text((payload, None)) == FAQ
    assert faq_text("plain text faq") == "plain text faq"
    assert faq_text(json.dumps({"locale": "en", "faq": None})) == ""
    assert faq_text(None) == ""


def test_the_faq_reaches_the_model_as_the_system_prompt():
    out, seen, _ = run([AIMessage("Lawn operators are paid per stop.")], "How does pay work?", locale="es")
    system = seen[0][0]
    assert isinstance(system, SystemMessage)
    assert FAQ in system.content
    assert "Spanish" in system.content
    assert out[-1].content == "Lawn operators are paid per stop."


def test_handoff_tool_runs_and_its_result_is_in_the_transcript():
    script = [
        AIMessage("", tool_calls=[{"id": "h1", "name": "request_human_handoff", "args": {"reason": "wants to apply"}}]),
        AIMessage(""),
    ]
    out, _, handoffs = run(script, "I want to apply today")
    assert handoffs == ["wants to apply"]
    assert any(isinstance(m, ToolMessage) and m.name == "request_human_handoff" for m in out)
