"""app_id -> app module. Each module exposes build_agent(identity_token, model_overrides) and
MAX_TOOL_ROUNDTRIPS -- see apps/operator_portal.py. Onboarding a future app is adding a sibling
module and one entry here, not touching api/, streaming.py, or messages.py."""

from agent_orchestrator.apps import operator_portal

APPS = {
    "operator_portal": operator_portal,
}
