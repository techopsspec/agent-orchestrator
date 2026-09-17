from pydantic import BaseModel, Field


class ToolCallIn(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class MessageIn(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCallIn] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


class ModelOverrides(BaseModel):
    """Resolved model *names* only -- see apps/operator_portal.py's build_agent docstring for
    why credentials/base URLs never travel in a per-turn request."""

    vllm_model: str | None = None
    openai_model: str | None = None
    gemini_model: str | None = None


class RunTurnRequest(BaseModel):
    app_id: str
    conversation_id: str
    turn_id: str
    messages: list[MessageIn]
    identity_token: str
    model_overrides: ModelOverrides = Field(default_factory=ModelOverrides)
