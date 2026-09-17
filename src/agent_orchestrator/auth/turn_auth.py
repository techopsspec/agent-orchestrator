import hmac

from fastapi import Header, HTTPException

from agent_orchestrator.config import settings


def require_service_auth(authorization: str | None = Header(default=None)) -> None:
    """Proves an inbound /v1/turns call came from Rails -- the mirror of Rails' own
    MCP_SHARED_SECRET check on the way back. hmac.compare_digest for the same
    timing-attack-resistance reason Rails uses ActiveSupport::SecurityUtils.secure_compare."""
    provided = (authorization or "").removeprefix("Bearer ")
    if not provided or not hmac.compare_digest(provided, settings.orchestrator_turn_secret):
        raise HTTPException(status_code=401, detail="invalid or missing service credentials")
