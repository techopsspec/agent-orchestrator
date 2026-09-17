"""Deep agent wiring for Operator-Portal (Andrews Lawn & Snow). One app = one module like this
one; onboarding a future app means adding a sibling module here plus a registry entry
(apps/registry.py), not touching graph/model/tool-loop code (see the orchestration-layer plan,
"Onboarding a future second application")."""

from deepagents import create_deep_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ToolCallLimitMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from agent_orchestrator.config import settings

SYSTEM_PROMPT = (
    "You are the Operator Portal chat assistant for Andrews Lawn & Snow, a lawn-care and "
    "snow-removal field service company. You help staff report and look up snowfall readings, "
    "and check what a snow visit at an address would be paid. Use the available tools rather "
    "than guessing at data you could look up. If a tool returns an error, tell the user what "
    "went wrong instead of silently retrying the same call unchanged."
)

# Ruby's Chat::TurnHandler capped a turn at 5 model calls (MAX_TOOL_ROUNDTRIPS); this is that
# same cap, expressed as "at most 5 tool-calling rounds," with a graceful terminal message
# instead of an exception -- see ToolCallLimitMiddleware's own exit_behavior="end" docs.
MAX_TOOL_ROUNDTRIPS = 5


async def build_agent(identity_token: str, model_overrides: dict | None = None):
    """Builds a fresh deep agent for one turn, plus a {model_name: provider_label} map so the
    caller can label which provider actually answered (Rails persists this as
    ChatMessage#provider_used, same as Integrations::Llm today).

    A fresh MultiServerMCPClient per call is deliberate -- it's what lets this turn's
    identity_token (naming which Operator-Portal user to run tool calls as) be scoped to just
    this call's MCP session, not shared/stale across turns or users.

    model_overrides carries only resolved model *names* from Rails' LlmSettings (admin-
    configurable, e.g. via the Settings page) -- never credentials or base URLs, which stay
    orchestrator-owned deployment facts. A missing/blank override falls back to this
    orchestrator's own env-configured default, mirroring LlmSettings' own `.presence ||` pattern.
    """
    model_overrides = model_overrides or {}
    vllm_model_name = model_overrides.get("vllm_model") or settings.vllm_model
    openai_model_name = model_overrides.get("openai_model") or settings.openai_model
    gemini_model_name = model_overrides.get("gemini_model") or settings.gemini_model

    client = MultiServerMCPClient(
        {
            "operator_portal": {
                "transport": "streamable_http",
                "url": settings.rails_mcp_url,
                "headers": {
                    "Authorization": f"Bearer {settings.mcp_shared_secret}",
                    "X-Operator-Portal-Identity": identity_token,
                },
            }
        }
    )
    tools = await client.get_tools()

    primary_model = ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key or "unused",
        model=vllm_model_name,
    )
    provider_map = {vllm_model_name: "vllm"}

    middleware = [ToolCallLimitMiddleware(run_limit=MAX_TOOL_ROUNDTRIPS, exit_behavior="end")]

    # Only wire in the providers that actually have credentials configured -- a fallback with
    # no API key can never succeed, so listing it would just add a guaranteed-failing hop.
    fallback_models = []
    if settings.openai_api_key:
        fallback_models.append(ChatOpenAI(api_key=settings.openai_api_key, model=openai_model_name))
        provider_map[openai_model_name] = "openai"
    if settings.gemini_api_key:
        fallback_models.append(
            ChatGoogleGenerativeAI(google_api_key=settings.gemini_api_key, model=gemini_model_name)
        )
        provider_map[gemini_model_name] = "gemini"
    if fallback_models:
        # Order matters: ModelFallbackMiddleware tries these in sequence after the primary
        # model (passed as create_deep_agent's model=) fails. Must run before the tool-call
        # limiter so a fallover on round 1 doesn't itself count against the round budget.
        middleware.insert(0, ModelFallbackMiddleware(*fallback_models))

    agent = create_deep_agent(
        model=primary_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
    )
    return agent, provider_map
