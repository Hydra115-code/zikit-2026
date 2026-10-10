from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.challenge.contracts import TARGET_TENANT
from app.challenge.deployment import verify_deployment_bundle
from app.challenge.explanations import build_challenge_alert_explanation
from app.challenge.ingestion import IngestionError
from app.challenge.live_replay import (
    TERMINAL_STATES,
    ChallengeLiveReplayManager,
    LiveReplayConflict,
    LiveReplayNotFound,
    format_live_replay_sse,
)
from app.challenge.rehearsal import load_rehearsal_artifact
from app.challenge.response import RESPONSE_PLAN_VERSION, response_catalog
from app.challenge.service import ChallengeReplayService
from app.challenge.showcase import load_showcase_artifact
from app.challenge.submission import (
    AGGREGATE_SUBMISSION_COLUMNS,
    DELIVERY_CONTRACT_VERSION,
    SUBMISSION_COLUMNS,
)
from app.config import Settings


router = APIRouter(prefix="/api/challenge", tags=["challenge"])


class ChallengeReplayRequest(BaseModel):
    run_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    )


class ChallengeLiveReplayRequest(ChallengeReplayRequest):
    speed_minutes_per_second: float = Field(default=120, ge=1, le=6000)
    telemetry_interval_minutes: int = Field(default=10, ge=1, le=60)


class ChallengeSubmissionField(BaseModel):
    name: str
    data_type: str
    required: bool
    semantics: str


class ChallengeEndpointContract(BaseModel):
    method: Literal["GET", "POST"]
    path: str
    authentication: str
    purpose: str


class ChallengeDeliveryContract(BaseModel):
    contract_version: str
    scope: str
    tenant: Literal["kaiku"]
    environment: str
    submission_columns: list[str]
    aggregate_audit_columns: list[str]
    partitioning: str
    submission_fields: list[ChallengeSubmissionField]
    ordering: list[str]
    identity: list[str]
    endpoints: list[ChallengeEndpointContract]
    stream_event: str
    stream_state_event: str
    stream_resume_header: str
    response_plan_version: str
    supported_incident_types: list[str]
    limitations: dict[str, bool]


def _service(request: Request) -> ChallengeReplayService:
    return request.app.state.challenge_replay_service


def _live_manager(request: Request) -> ChallengeLiveReplayManager:
    return request.app.state.challenge_live_replay_manager


def _remediation_manager(request: Request):
    return request.app.state.challenge_remediation_manager


def _require_admin(request: Request, token: str | None) -> None:
    settings: Settings = request.app.state.settings
    if not token or token != settings.admin_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "valid X-Admin-Token required",
        )


def _cursor(
    after_id: int,
    last_event_id: str | None,
) -> int:
    if after_id < 0:
        raise HTTPException(422, "after_id must be non-negative")
    cursor = after_id
    if last_event_id:
        try:
            header_cursor = int(last_event_id)
        except ValueError as exc:
            raise HTTPException(422, "Last-Event-ID must be an integer") from exc
        if header_cursor < 0:
            raise HTTPException(422, "Last-Event-ID must be non-negative")
        cursor = max(cursor, header_cursor)
    return cursor


def format_challenge_sse(alert: dict[str, object]) -> str:
    payload = json.dumps(alert, ensure_ascii=False, separators=(",", ":"))
    event_name = str(alert.get("stream_event", "challenge_alert"))
    return f"id: {alert['id']}\nevent: {event_name}\ndata: {payload}\n\n"


@router.get("/contract", response_model=ChallengeDeliveryContract)
async def challenge_delivery_contract() -> dict[str, object]:
    return {
        "contract_version": DELIVERY_CONTRACT_VERSION,
        "scope": "official synthetic Zikit public Kaiku runs",
        "tenant": TARGET_TENANT,
        "environment": "official_synthetic_challenge_replay",
        "submission_columns": list(SUBMISSION_COLUMNS),
        "aggregate_audit_columns": list(AGGREGATE_SUBMISSION_COLUMNS),
        "partitioning": "one evaluator CSV per run_id",
        "submission_fields": [
            {
                "name": "tenant",
                "data_type": "string",
                "required": True,
                "semantics": "always kaiku in this solution",
            },
            {
                "name": "timestamp",
                "data_type": "YYYY-MM-DDTHH:MM:SS",
                "required": True,
                "semantics": "minute when the alert decision became effective",
            },
            {
                "name": "confidence",
                "data_type": "number[0,1]",
                "required": True,
                "semantics": "risk score; not a calibrated incident probability",
            },
            {
                "name": "suspected_incident_type",
                "data_type": "string|null",
                "required": False,
                "semantics": "causal classification when public evidence supports it",
            },
        ],
        "ordering": ["tenant", "timestamp"],
        "identity": ["tenant", "timestamp"],
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/challenge/replay",
                "authentication": "X-Admin-Token",
                "purpose": "process one frozen public feature run idempotently",
            },
            {
                "method": "POST",
                "path": "/api/challenge/live-replay",
                "authentication": "X-Admin-Token",
                "purpose": "start an observable minute-by-minute synthetic replay",
            },
            {
                "method": "GET",
                "path": "/api/challenge/live-replay/stream",
                "authentication": "none",
                "purpose": "stream synthetic observations and causal alerts over SSE",
            },
            {
                "method": "GET",
                "path": "/api/challenge/alerts",
                "authentication": "none",
                "purpose": "read durable alerts with cursor and run filters",
            },
            {
                "method": "GET",
                "path": "/api/challenge/alerts/stream",
                "authentication": "none",
                "purpose": "consume resumable challenge alerts over SSE",
            },
            {
                "method": "GET",
                "path": "/api/challenge/alerts/{event_id}/explanation",
                "authentication": "none",
                "purpose": "read deterministic evidence and response plan",
            },
            {
                "method": "GET",
                "path": "/api/challenge/operations",
                "authentication": "none",
                "purpose": "read durable state and valid timing measurements",
            },
            {
                "method": "GET",
                "path": "/api/challenge/remediations",
                "authentication": "none",
                "purpose": "read the controlled autonomous remediation audit trail",
            },
            {
                "method": "GET",
                "path": "/api/challenge/remediations/audit.csv",
                "authentication": "none",
                "purpose": "download reproducible remediation evidence as CSV",
            },
            {
                "method": "GET",
                "path": "/api/challenge/showcase",
                "authentication": "none",
                "purpose": "read the verified guided Kaiku demonstration",
            },
            {
                "method": "GET",
                "path": "/api/challenge/rehearsal",
                "authentication": "none",
                "purpose": "read the hash-verified live backend rehearsal",
            },
            {
                "method": "GET",
                "path": "/api/challenge/deployment",
                "authentication": "none",
                "purpose": "read the verified deployable runtime manifest",
            },
        ],
        "stream_event": "challenge_alert",
        "stream_state_event": "challenge_alert_acknowledged",
        "stream_resume_header": "Last-Event-ID",
        "response_plan_version": RESPONSE_PLAN_VERSION,
        "supported_incident_types": sorted(
            key
            for key in response_catalog()
            if key not in {"data_quality_alert", "unclassified"}
        ),
        "limitations": {
            "official_ground_truth_used": False,
            "confidence_is_calibrated_probability": False,
            "automatic_remediation": True,
            "gemini_required": False,
        },
    }


@router.post("/replay")
async def replay_run(
    payload: ChallengeReplayRequest,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_admin(request, x_admin_token)
    try:
        return await _service(request).replay(payload.run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _live_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LiveReplayNotFound):
        return HTTPException(404, str(exc))
    if isinstance(exc, LiveReplayConflict):
        return HTTPException(409, str(exc))
    if isinstance(exc, (IngestionError, ValueError)):
        return HTTPException(422, str(exc))
    return HTTPException(500, "live replay failed")


@router.get("/live-replay/runs")
async def challenge_live_replay_runs(request: Request) -> dict[str, object]:
    try:
        runs = await _live_manager(request).available_runs()
    except (IngestionError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "tenant": TARGET_TENANT,
        "environment": "official_synthetic_challenge_live_replay",
        "runs": runs,
        "count": len(runs),
    }


@router.post("/live-replay", status_code=status.HTTP_202_ACCEPTED)
async def challenge_live_replay_start(
    payload: ChallengeLiveReplayRequest,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_admin(request, x_admin_token)
    try:
        return await _live_manager(request).start(
            payload.run_id,
            speed_minutes_per_second=payload.speed_minutes_per_second,
            telemetry_interval_minutes=payload.telemetry_interval_minutes,
        )
    except (LiveReplayConflict, IngestionError, ValueError) as exc:
        raise _live_error(exc) from exc


@router.get("/live-replay")
async def challenge_live_replay_status(
    request: Request,
    session_id: str | None = None,
) -> dict[str, object]:
    try:
        return _live_manager(request).status(session_id)
    except LiveReplayNotFound as exc:
        raise _live_error(exc) from exc


@router.post("/live-replay/{session_id}/pause")
async def challenge_live_replay_pause(
    session_id: str,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_admin(request, x_admin_token)
    try:
        return await _live_manager(request).pause(session_id)
    except (LiveReplayNotFound, LiveReplayConflict) as exc:
        raise _live_error(exc) from exc


@router.post("/live-replay/{session_id}/resume")
async def challenge_live_replay_resume(
    session_id: str,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_admin(request, x_admin_token)
    try:
        return await _live_manager(request).resume(session_id)
    except (LiveReplayNotFound, LiveReplayConflict) as exc:
        raise _live_error(exc) from exc


@router.post("/live-replay/{session_id}/stop")
async def challenge_live_replay_stop(
    session_id: str,
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_admin(request, x_admin_token)
    try:
        return await _live_manager(request).stop(session_id)
    except (LiveReplayNotFound, LiveReplayConflict) as exc:
        raise _live_error(exc) from exc


@router.get("/live-replay/{session_id}/submission")
async def challenge_live_replay_submission(
    session_id: str,
    request: Request,
) -> Response:
    try:
        content = _live_manager(request).submission(session_id)
    except (LiveReplayNotFound, LiveReplayConflict) as exc:
        raise _live_error(exc) from exc
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{TARGET_TENANT}-{session_id}-alerts.csv"'
            )
        },
    )


@router.get("/live-replay/stream")
async def challenge_live_replay_stream(
    request: Request,
    session_id: str,
    after_sequence: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    cursor = after_sequence
    if last_event_id:
        try:
            cursor = max(cursor, int(last_event_id))
        except ValueError as exc:
            raise HTTPException(422, "Last-Event-ID must be an integer") from exc
    manager = _live_manager(request)
    try:
        initial = manager.status(session_id)
    except LiveReplayNotFound as exc:
        raise _live_error(exc) from exc
    if cursor > int(initial["latest_sequence"]):
        raise HTTPException(409, "live replay cursor is ahead of the session")

    settings: Settings = request.app.state.settings
    capacity = request.app.state.event_stream_capacity
    if not await capacity.try_acquire():
        raise HTTPException(503, "event stream connection capacity is exhausted")

    async def generate():
        nonlocal cursor
        try:
            yield f"retry: {settings.event_stream_retry_milliseconds}\n\n"
            last_heartbeat = time.monotonic()
            while not await request.is_disconnected():
                try:
                    events = manager.events_after(
                        session_id,
                        after_sequence=cursor,
                        limit=settings.event_stream_batch_size,
                    )
                    current = manager.status(session_id)
                except LiveReplayNotFound:
                    return
                if events:
                    for event in events:
                        cursor = int(event["sequence"])
                        yield format_live_replay_sse(event)
                    last_heartbeat = time.monotonic()
                elif current["status"] in TERMINAL_STATES:
                    return
                elif (
                    time.monotonic() - last_heartbeat
                    >= settings.event_stream_heartbeat_seconds
                ):
                    yield f'event: heartbeat\ndata: {{"sequence":{cursor}}}\n\n'
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(settings.event_stream_poll_interval_seconds)
        finally:
            await capacity.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Live-Replay-Session": session_id,
            "X-Live-Replay-Environment": (
                "official-synthetic-challenge-live-replay"
            ),
        },
    )


@router.get("/alerts")
async def challenge_alerts(
    request: Request,
    after_id: int = 0,
    limit: int = Query(default=100, ge=1, le=1000),
    run_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    ),
    tenant: str | None = None,
    alert_type: Literal["incident_risk", "data_quality_alert", "all"] = (
        "incident_risk"
    ),
) -> list[dict[str, object]]:
    if after_id < 0:
        raise HTTPException(422, "after_id must be non-negative")
    if tenant is not None and tenant != TARGET_TENANT:
        raise HTTPException(422, "unsupported tenant")
    selected_type = None if alert_type == "all" else alert_type
    return _service(request).store.list(
        after_id=after_id,
        limit=limit,
        run_id=run_id,
        tenant=tenant,
        alert_type=selected_type,
    )


@router.get("/alerts/cursor")
async def challenge_alert_cursor(request: Request) -> dict[str, object]:
    return {
        **_service(request).store.cursor(),
        "storage": "durable_database_phase_7",
        "durable": True,
    }


@router.get("/operations")
async def challenge_operations(request: Request) -> dict[str, object]:
    return _service(request).store.operations_summary()


@router.get("/remediations")
async def challenge_remediations(request: Request) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    records = _remediation_manager(request).list_records()
    return {
        "version": "challenge-controlled-remediation-v1",
        "environment": "controlled_infrastructure_demo",
        "enabled": settings.challenge_remediation_enabled,
        "wait_seconds": settings.challenge_remediation_wait_seconds,
        "verification_minutes": settings.challenge_remediation_verification_minutes,
        "gemini_configured": bool(settings.gemini_api_key),
        "mcp_execution_available": request.app.state.remediation_mcp_server is not None,
        "records": records,
    }


@router.post("/remediations/process")
async def process_challenge_remediations(
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_admin(request, x_admin_token)
    processed = await _remediation_manager(request).process_due_once()
    return {"processed": processed, "records": _remediation_manager(request).list_records()}


@router.get("/remediations/audit.csv")
async def challenge_remediation_audit(request: Request) -> Response:
    content = _remediation_manager(request).audit_csv()
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="challenge-remediation-audit.csv"'
        },
    )


@router.get("/showcase")
async def challenge_showcase(request: Request) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    try:
        return load_showcase_artifact(settings.challenge_showcase_path)
    except FileNotFoundError as exc:
        raise HTTPException(404, "challenge showcase has not been built") from exc
    except IngestionError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/rehearsal")
async def challenge_rehearsal(request: Request) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    try:
        return load_rehearsal_artifact(settings.challenge_rehearsal_path)
    except FileNotFoundError as exc:
        raise HTTPException(404, "challenge rehearsal has not been run") from exc
    except IngestionError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/deployment")
async def challenge_deployment(request: Request) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    if settings.challenge_runtime_path is None:
        raise HTTPException(404, "challenge deployment runtime is not configured")
    try:
        return verify_deployment_bundle(settings.challenge_runtime_path)
    except FileNotFoundError as exc:
        raise HTTPException(404, "challenge deployment runtime is missing") from exc
    except IngestionError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/alerts/{event_id}/explanation")
async def challenge_alert_explanation(event_id: int) -> dict[str, object]:
    if event_id < 1:
        raise HTTPException(422, "event_id must be positive")
    try:
        return build_challenge_alert_explanation(event_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/alerts/stream")
async def challenge_alert_stream(
    request: Request,
    after_id: int = 0,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    cursor = _cursor(after_id, last_event_id)
    service = _service(request)
    window = service.store.cursor()
    if cursor > window["latest_id"]:
        raise HTTPException(
            409,
            {
                "code": "cursor_ahead",
                "message": "challenge cursor is ahead of the durable stream",
                **window,
            },
        )
    settings: Settings = request.app.state.settings
    capacity = request.app.state.event_stream_capacity
    if not await capacity.try_acquire():
        retry_after_seconds = max(
            1, (settings.event_stream_retry_milliseconds + 999) // 1000
        )
        raise HTTPException(
            503,
            {
                "code": "stream_capacity_exceeded",
                "message": "event stream connection capacity is exhausted",
                "max_connections": capacity.max_connections,
            },
            headers={"Retry-After": str(retry_after_seconds)},
        )

    async def generate():
        nonlocal cursor
        try:
            yield f"retry: {settings.event_stream_retry_milliseconds}\n\n"
            last_heartbeat = time.monotonic()
            while not await request.is_disconnected():
                alerts = service.store.stream_updates(
                    after_id=cursor,
                    limit=settings.event_stream_batch_size,
                )
                if alerts:
                    for alert in alerts:
                        cursor = int(alert["id"])
                        yield format_challenge_sse(alert)
                    last_heartbeat = time.monotonic()
                elif (
                    time.monotonic() - last_heartbeat
                    >= settings.event_stream_heartbeat_seconds
                ):
                    yield f'event: heartbeat\ndata: {{"cursor":{cursor}}}\n\n'
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(settings.event_stream_poll_interval_seconds)
        finally:
            await capacity.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Event-Cursor": str(cursor),
            "X-Event-Earliest": str(window["earliest_id"]),
            "X-Event-Latest": str(window["latest_id"]),
            "X-Challenge-Storage": "durable-database-phase-7",
        },
    )
