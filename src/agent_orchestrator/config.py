from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Deployment facts (credentials, base URLs) live here, never in a per-turn request from
    an app -- see the auth summary in the orchestration-layer plan for why."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Inference layer (ai-infra-lab)
    vllm_base_url: str
    vllm_api_key: str | None = None
    vllm_model: str

    # Cloud fallback -- optional; ModelFallbackMiddleware is only wired in when a key is set
    # (see apps/operator_portal.py), since a fallback with no credentials can't ever succeed.
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3-pro"

    # Operator-Portal's MCP server (Api::V1::McpController)
    rails_mcp_url: str
    mcp_shared_secret: str

    # Static shared secret proving an inbound /v1/turns call actually came from Rails.
    # Same trust model as mcp_shared_secret (one trusted service-to-service caller, short-lived
    # per-request use) -- no expiry/JWT needed for the same reason MCP_SHARED_SECRET doesn't
    # have one.
    orchestrator_turn_secret: str


settings = Settings()
