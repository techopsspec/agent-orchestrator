"""End-to-end through the real deep agent (make_agent): a scripted model plays the LLM, fake
tools stand in for the MCP ones, and we check what actually reaches the write tool."""

import asyncio

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent_orchestrator.apps.operator_portal import make_agent

ARGS = {"zone_name": "SBX test - 6th & 2nd Ave", "inches": 4.5, "reported_on": "2026-10-15"}
SUMMARY = "To confirm: 4.5 inches in SBX test - 6th & 2nd Ave on 2026-10-15 (Thursday). Reply yes to record it."


class ScriptedModel(BaseChatModel):
    """Returns the queued AIMessages in order, whoever asks (main agent or subagent)."""

    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def call(name, args, call_id="c1"):
    return AIMessage("", tool_calls=[{"id": call_id, "name": name, "args": args}])


@pytest.fixture
def writes():
    return []


@pytest.fixture
def tools(writes):
    @tool
    def report_snowfall_reading(zone_name: str, inches: float, reported_on: str) -> str:
        """Record a snowfall reading."""
        writes.append({"zone_name": zone_name, "inches": inches, "reported_on": reported_on})
        return '{"snowfall_report_id": 99}'

    @tool
    def list_snowfall_zones() -> str:
        """List zones."""
        return '{"zones": []}'

    return [report_snowfall_reading, list_snowfall_zones]


def run(tools, script, messages):
    """The orchestrator only ever runs the agent async (agent.astream), so do the same here --
    the guard's awrap_tool_call is the path that matters."""
    model = ScriptedModel(script=script)
    agent = make_agent(model, tools, [])
    result = asyncio.run(agent.ainvoke({"messages": messages}, config={"recursion_limit": 30}))
    return result["messages"]


def test_a_write_asked_for_on_this_turn_is_refused(tools, writes):
    out = run(
        tools,
        [call("report_snowfall_reading", ARGS), AIMessage(SUMMARY)],
        [HumanMessage("Report 4.5 inches in SBX test - 6th & 2nd Ave for 2026-10-15")],
    )
    assert writes == []
    refusal = next(m for m in out if isinstance(m, ToolMessage))
    assert refusal.content.startswith("NOT RECORDED")


def test_the_write_goes_through_after_yes(tools, writes):
    run(
        tools,
        [call("report_snowfall_reading", ARGS), AIMessage("Recorded.")],
        [HumanMessage("Report 4.5 inches ..."), AIMessage(SUMMARY), HumanMessage("yes")],
    )
    assert writes == [ARGS]


def test_read_tools_are_never_held_up(tools, writes):
    out = run(tools, [call("list_snowfall_zones", {}), AIMessage("No zones.")], [HumanMessage("zones?")])
    assert any(isinstance(m, ToolMessage) and m.name == "list_snowfall_zones" and "zones" in m.content for m in out)


def test_the_subagent_cannot_be_used_to_get_around_the_guard(tools, writes):
    run(
        tools,
        [
            # main agent hands the write to the general-purpose subagent...
            call("task", {"description": "Record 4.5 inches in SBX test - 6th & 2nd Ave on 2026-10-15",
                          "subagent_type": "general-purpose"}),
            # ...which tries to call the write tool directly
            call("report_snowfall_reading", ARGS, call_id="sub1"),
            AIMessage("I couldn't record it."),
            AIMessage("It wasn't recorded."),
        ],
        [HumanMessage("Report 4.5 inches in SBX test - 6th & 2nd Ave for 2026-10-15")],
    )
    assert writes == []
