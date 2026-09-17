from agent_orchestrator.streaming import _clean_content, _tool_result_text


def test_passes_through_plain_content():
    assert _clean_content("Here are the zones.") == "Here are the zones."


def test_strips_well_formed_think_block():
    assert _clean_content("<think>reasoning here</think>\n\nThe answer.") == "The answer."


def test_strips_dangling_close_tag_with_no_opening_tag():
    # The shape actually observed live against vLLM/Qwen3.5: the chat template pre-fills the
    # opening <think> as a prompt continuation, so the completion only ever contains the close.
    content = "Okay, I got the zones, let me list them.\n</think>\n\nHere are the zones:\n- Uptown"
    assert _clean_content(content) == "Here are the zones:\n- Uptown"


def test_leaves_non_string_content_untouched():
    payload = [{"type": "text", "text": "hi"}]
    assert _clean_content(payload) is payload


def test_tool_result_text_unwraps_the_mcp_content_part_array():
    # The actual shape observed live from langchain-mcp-adapters, confirmed against
    # Operator-Portal's real MCP endpoint.
    content = [{"type": "text", "text": '{"snowfall_report_id": 404}', "id": "lc_abc123"}]
    assert _tool_result_text(content) == '{"snowfall_report_id": 404}'


def test_tool_result_text_passes_through_a_plain_string():
    assert _tool_result_text('{"already": "unwrapped"}') == '{"already": "unwrapped"}'


def test_tool_result_text_falls_back_to_json_dump_for_anything_unexpected():
    assert _tool_result_text([1, 2, 3]) == "[1, 2, 3]"
