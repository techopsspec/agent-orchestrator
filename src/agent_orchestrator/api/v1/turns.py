from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent_orchestrator.apps.registry import APPS
from agent_orchestrator.auth.turn_auth import require_service_auth
from agent_orchestrator.schemas.turn import RunTurnRequest
from agent_orchestrator.streaming import stream_turn

router = APIRouter()


@router.post("/v1/turns", dependencies=[Depends(require_service_auth)])
async def run_turn(request: RunTurnRequest):
    app_module = APPS.get(request.app_id)
    if app_module is None:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {request.app_id!r}")

    return StreamingResponse(stream_turn(app_module, request), media_type="text/event-stream")
