from agent_orchestrator.reasoning import ReasoningPreservingChatOpenAI


def _model() -> ReasoningPreservingChatOpenAI:
    # No real network call happens in these tests -- _create_chat_result only parses an
    # already-fetched response dict.
    return ReasoningPreservingChatOpenAI(api_key="test", model="test-model")


def test_preserves_reasoning_content_from_the_raw_response():
    # Shape confirmed live against ai-infra-lab's vLLM (RedHatAI/Qwen3.5-4B-FP8-dynamic,
    # --reasoning-parser qwen3) on 2026-09-21.
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 408.",
                    "reasoning": "17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    result = _model()._create_chat_result(response)
    message = result.generations[0].message

    assert message.content == "The answer is 408."
    assert message.additional_kwargs["reasoning_content"] == "17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408."


def test_no_reasoning_key_added_when_the_response_has_none():
    # The OpenAI/Gemini fallback models (and any vLLM response where the model chose not to
    # reason) never populate `reasoning` -- additional_kwargs must stay exactly as ChatOpenAI's
    # own parsing produces it, not gain a stray None/empty entry.
    response = {
        "id": "chatcmpl-test2",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    result = _model()._create_chat_result(response)

    assert "reasoning_content" not in result.generations[0].message.additional_kwargs


def test_reasoning_survives_alongside_a_tool_call():
    # The actual shape this app produces most often: the model reasons about which tool to
    # call, then calls it -- content is empty/None on a tool-only turn (see ChatMessage's own
    # role-shape comment), but reasoning should still come through.
    response = {
        "id": "chatcmpl-test4",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "The user wants the Uptown reading recorded -- call report_snowfall_reading.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "report_snowfall_reading",
                                "arguments": '{"zone": "Uptown", "inches": 3.5}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 25, "total_tokens": 37},
    }

    result = _model()._create_chat_result(response)
    message = result.generations[0].message

    assert message.additional_kwargs["reasoning_content"] == (
        "The user wants the Uptown reading recorded -- call report_snowfall_reading."
    )
    assert message.tool_calls[0]["name"] == "report_snowfall_reading"
    assert message.tool_calls[0]["args"] == {"zone": "Uptown", "inches": 3.5}


def test_empty_reasoning_string_is_not_treated_as_present():
    response = {
        "id": "chatcmpl-test3",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi.", "reasoning": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    result = _model()._create_chat_result(response)

    assert "reasoning_content" not in result.generations[0].message.additional_kwargs
