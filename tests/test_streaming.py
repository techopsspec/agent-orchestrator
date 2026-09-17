from agent_orchestrator.streaming import _clean_content


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
