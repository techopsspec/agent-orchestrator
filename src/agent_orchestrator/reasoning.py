from typing import Any

from langchain_openai import ChatOpenAI


class ReasoningPreservingChatOpenAI(ChatOpenAI):
    """`ChatOpenAI` silently drops any `reasoning`/`reasoning_content` field an OpenAI-compatible
    backend returns -- this is a known, still-open langchain-openai limitation, stated in that
    class's own module docstring: "Non-standard response fields added by third-party providers
    (e.g., reasoning_content, reasoning_details) are not extracted or preserved. If you are
    pointing base_url at a provider such as OpenRouter, vLLM, or DeepSeek, use the corresponding
    provider-specific LangChain package instead." There is no such package for vLLM/Qwen3, so
    this preserves it ourselves.

    ai-infra-lab's vLLM now runs with --reasoning-parser qwen3 (alongside the pre-existing
    --tool-call-parser qwen3_xml -- the combination is confirmed not to interfere with either
    field, see vllm-project/vllm#51679), so the raw completion response has reasoning in its own
    `message.reasoning` field, not leaked into `content` the way _clean_content's docstring
    describes. openai-python's response models use `extra="allow"` (confirmed in its own
    _models.py), so that field survives onto the parsed response object this class receives --
    LangChain's own parsing just doesn't look for it by name. Rather than reimplementing
    ChatOpenAI's message-parsing internals (a private, version-fragile surface), this re-reads
    the same raw `response` _create_chat_result already parsed and stitches the field onto the
    resulting message's additional_kwargs, which every downstream consumer (this repo's
    streaming.py, LangGraph itself) already treats as the standard place for provider-specific
    extras.

    Only wrap the primary vLLM model with this -- the OpenAI/Gemini fallback models use their
    own native reasoning conventions (OpenAI's `reasoning` param family, Gemini's own thinking
    config), out of scope here and not needed today since neither fallback model is a reasoning
    model.
    """

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> Any:
        result = super()._create_chat_result(response, generation_info)

        response_dict = response if isinstance(response, dict) else response.model_dump(warnings=False)
        choices = response_dict.get("choices") or []
        for generation, choice in zip(result.generations, choices, strict=False):
            reasoning = (choice.get("message") or {}).get("reasoning")
            if reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning

        return result
