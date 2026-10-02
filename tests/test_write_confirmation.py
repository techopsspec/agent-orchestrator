from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_orchestrator.apps.operator_portal import system_prompt
from agent_orchestrator.apps.write_confirmation import (
    check_write_allowed,
    is_affirmative,
    summary_matches,
)

ARGS = {"zone_name": "SBX test - 6th & 2nd Ave", "inches": 4.5, "reported_on": "2026-10-15"}
SUMMARY = "To confirm: 4.5 inches in SBX test - 6th & 2nd Ave on 2026-10-15 (Thursday). Reply yes to record it."


def convo(*messages):
    return list(messages)


# --- is_affirmative ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["yes", "Yes!", "y", "yep", "Yeah, go ahead", "confirm", "Correct.", "go ahead",
                                  "that's right", "OK", "sure", "do it"])
def test_affirmative(text):
    assert is_affirmative(text)


@pytest.mark.parametrize("text", ["no", "wait", "Report 3 inches in eagan for today", "not yet", "",
                                  "yesterday it snowed 3 inches", "okaaay maybe"])
def test_not_affirmative(text):
    assert not is_affirmative(text)


# --- summary_matches --------------------------------------------------------------------------

def test_summary_matches_iso_date():
    assert summary_matches(SUMMARY, ARGS)


def test_summary_matches_long_date_and_int_inches():
    summary = "Record 3 inches in SBX test - 6th & 2nd Ave on October 15, 2026?"
    assert summary_matches(summary, {**ARGS, "inches": 3})


def test_summary_mismatch_on_inches():
    assert not summary_matches(SUMMARY, {**ARGS, "inches": 5})


def test_summary_mismatch_on_date():
    assert not summary_matches(SUMMARY, {**ARGS, "reported_on": "2026-10-16"})


def test_summary_mismatch_on_zone():
    assert not summary_matches(SUMMARY, {**ARGS, "zone_name": "eagan"})


# --- check_write_allowed ----------------------------------------------------------------------

def test_blocks_a_write_on_the_same_turn_it_was_asked():
    messages = convo(HumanMessage("Report 4.5 inches in SBX test - 6th & 2nd Ave for 2026-10-15"))
    assert check_write_allowed(messages, ARGS) == "the user has not confirmed this write yet"


def test_allows_a_write_after_yes_to_a_matching_summary():
    messages = convo(HumanMessage("Report 4.5 inches ..."), AIMessage(SUMMARY), HumanMessage("yes"))
    assert check_write_allowed(messages, ARGS) is None


def test_blocks_when_yes_approved_different_values():
    messages = convo(HumanMessage("Report ..."), AIMessage(SUMMARY), HumanMessage("yes"))
    assert check_write_allowed(messages, {**ARGS, "zone_name": "eagan"}) is not None


def test_blocks_yes_with_no_summary_before_it():
    messages = convo(HumanMessage("yes"))
    assert check_write_allowed(messages, ARGS) is not None


def test_one_yes_allows_one_write():
    messages = convo(
        HumanMessage("Report ..."), AIMessage(SUMMARY), HumanMessage("yes"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "report_snowfall_reading", "args": ARGS}]),
        ToolMessage('{"snowfall_report_id": 7}', tool_call_id="c1", name="report_snowfall_reading"),
    )
    assert "already ran" in check_write_allowed(messages, ARGS)


def test_the_summary_is_the_latest_assistant_text_before_the_yes():
    # An older summary for different values must not be what a later "yes" approves.
    old = "Record 1 inch in SBX test - 6th & 2nd Ave on 2026-10-15? Reply yes."
    messages = convo(AIMessage(old), HumanMessage("no, 4.5"), AIMessage(SUMMARY), HumanMessage("yes"))
    assert check_write_allowed(messages, ARGS) is None
    assert check_write_allowed(messages, {**ARGS, "inches": 1}) is not None


# --- system_prompt date -----------------------------------------------------------------------

def test_system_prompt_states_todays_central_date():
    now = datetime(2026, 10, 2, 14, 0, tzinfo=ZoneInfo("America/Chicago"))
    prompt = system_prompt(now)
    assert "Today is Friday, 2026-10-02" in prompt
    assert "Yesterday was Thursday, 2026-10-01" in prompt
    assert "Tomorrow is Saturday, 2026-10-03" in prompt


def test_system_prompt_uses_central_not_utc_late_in_the_evening():
    # 10pm Central on Oct 2 is already Oct 3 in UTC.
    now = datetime(2026, 10, 3, 3, 0, tzinfo=ZoneInfo("UTC"))
    assert "Today is Friday, 2026-10-02" in system_prompt(now)
