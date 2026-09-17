import structlog
from fastapi import FastAPI

from agent_orchestrator.api.v1 import health, turns

structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])

app = FastAPI(title="agent-orchestrator")
app.include_router(health.router)
app.include_router(turns.router)
