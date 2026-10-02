"""Deep agent wiring for Operator-Portal (Andrews Lawn & Snow). One app = one module like this
one; onboarding a future app means adding a sibling module here plus a registry entry
(apps/registry.py), not touching graph/model/tool-loop code (see the orchestration-layer plan,
"Onboarding a future second application")."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from deepagents import create_deep_agent
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents.middleware import ModelFallbackMiddleware, ToolCallLimitMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from agent_orchestrator.apps.write_confirmation import (
    WRITE_TOOLS,
    WriteConfirmationMiddleware,
)
from agent_orchestrator.config import settings
from agent_orchestrator.reasoning import ReasoningPreservingChatOpenAI

SYSTEM_PROMPT = (
    "You are the Operator Portal chat assistant for Andrews Lawn & Snow, a lawn-care and "
    "snow-removal field service company. You help staff report and look up snowfall readings, "
    "and check what a snow visit at an address would be paid. Use the available tools rather "
    "than guessing at data you could look up. If a tool returns an error, tell the user what "
    "went wrong instead of silently retrying the same call unchanged.\n\n"
    "Recording a snowfall reading changes real operator pay, so never record one in the same "
    "reply the user asked in. First reply with one short message that states the zone name, "
    "the inches, and the date as YYYY-MM-DD with its weekday, and ask the user to reply yes. "
    "Only call report_snowfall_reading after they say yes, using exactly those values. If "
    "anything is missing or unclear (zone, inches, or date), ask instead of guessing."
)

# The company works in Central time; every date the user says ("today", "yesterday", "last
# night") is a Central-time date.
BUSINESS_TZ = ZoneInfo("America/Chicago")


def system_prompt(now: datetime | None = None) -> str:
    """SYSTEM_PROMPT plus today's date. Without it the model has no idea what day it is and
    resolves "today"/"tomorrow" by guessing from dates earlier in the conversation -- testing on
    2026-10-02 recorded "tomorrow" as 2026-10-16. Built per turn (build_agent runs once per
    turn), so the date is always current."""
    today = (now or datetime.now(BUSINESS_TZ)).astimezone(BUSINESS_TZ).date()
    yesterday, tomorrow = today - timedelta(days=1), today + timedelta(days=1)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Today is {today:%A}, {today.isoformat()} (Central time). "
        f"Yesterday was {yesterday:%A}, {yesterday.isoformat()}. "
        f"Tomorrow is {tomorrow:%A}, {tomorrow.isoformat()}. "
        "Resolve every relative date the user gives (today, yesterday, last night, Monday) "
        "against this, never against dates mentioned earlier in the conversation."
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

    primary_model = ReasoningPreservingChatOpenAI(
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

    agent = make_agent(primary_model, tools, middleware)
    return agent, provider_map


def make_agent(model, tools, middleware):
    """The deep agent itself, split out of build_agent so tests can build it with a scripted
    model and fake tools and exercise the real wiring (guard + subagent override)."""
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt(),
        # First, so it wraps every tool call -- including one the fallback model makes.
        middleware=[WriteConfirmationMiddleware(), *middleware],
        # deepagents' default general-purpose subagent (reached via its `task` tool) gets every
        # tool but none of our middleware -- it would be a way around WriteConfirmationMiddleware.
        # Same subagent, minus the write tools, so a write can only happen in the main agent,
        # where the confirmation guard runs.
        subagents=[{**GENERAL_PURPOSE_SUBAGENT, "tools": [t for t in tools if t.name not in WRITE_TOOLS]}],
    )
