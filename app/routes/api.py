from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.challenge.deployment import verify_deployment_bundle
from app.challenge.ingestion import IngestionError
from app.challenge.persistence import acknowledge_alert_once
from app.ai.mcp_transport import bearer_token_valid
from app.core.fault_control import FaultAlreadyActiveError
from app.core.explanations import build_incident_explanation
from app.core.evidence_store import (
    EvidenceBundleUnavailable,
    bundle_index,
)
from app.core.replay import (
    aggregate_replay_group,
    build_replay_plan,
    persisted_scenario,
    replay_metadata,
    scenario_fingerprint,
    validate_pair,
)
from app.core.service import new_experiment_id
from app.database import (
    ActionRecord,
    AlertRecord,
    EventRecord,
    ExperimentRecord,
    IncidentRecord,
    InvestigationRecord,
    MetricRecord,
    ResourceMetricRecord,
    ServiceRecord,
    get_session,
    session_factory,
)
from app.ml.model import FEATURES
from app.schemas import ExperimentCreate, HealthResponse, IncidentResponse, ModelScoreRequest, ReplayPlanRequest

router = APIRouter()


def serialize_event(record: EventRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "incident_id": record.incident_id,
        "experiment_id": record.experiment_id,
        "type": record.event_type,
        "timestamp": record.timestamp.isoformat(),
        "details": record.details,
    }


def serialize_alert(alert: AlertRecord, event: EventRecord) -> dict[str, object]:
    return {
        "id": alert.id,
        "event_id": event.id,
        "incident_id": event.incident_id,
        "experiment_id": event.experiment_id,
        "type": event.event_type,
        "priority": alert.priority,
        "state": alert.state,
        "created_at": alert.created_at.isoformat(),
        "acknowledged_at": (
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
        ),
        "details": event.details,
    }


def serialize_experiment(record: ExperimentRecord) -> dict[str, object]:
    metadata = replay_metadata(record)
    return {
        "id": record.id,
        "mode": record.mode,
        "status": record.status,
        "created_at": record.created_at.isoformat(),
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        "replay_group_id": metadata.get("group_id"),
        "repetition": metadata.get("repetition"),
        "scenario_fingerprint": metadata.get("fingerprint") or scenario_fingerprint(record.scenario),
    }


def format_sse(record: EventRecord) -> str:
    payload = json.dumps(serialize_event(record), separators=(",", ":"))
    return f"id: {record.id}\nevent: {record.event_type}\ndata: {payload}\n\n"


def event_cursor_status(session: Session) -> dict[str, int]:
    earliest_id, latest_id, retained_count = session.execute(
        select(
            func.min(EventRecord.id),
            func.max(EventRecord.id),
            func.count(EventRecord.id),
        )
    ).one()
    return {
        "earliest_id": int(earliest_id or 0),
        "latest_id": int(latest_id or 0),
        "retained_count": int(retained_count or 0),
    }


def validate_stream_cursor(
    session: Session,
    cursor: int,
    max_replay_events: int,
) -> dict[str, int]:
    window = event_cursor_status(session)
    earliest_id = window["earliest_id"]
    latest_id = window["latest_id"]
    if cursor > latest_id:
        raise HTTPException(
            409,
            {
                "code": "cursor_ahead",
                "message": "event cursor is ahead of the persisted stream",
                **window,
            },
        )
    if cursor > 0 and earliest_id > 0 and cursor < earliest_id - 1:
        raise HTTPException(
            409,
            {
                "code": "cursor_expired",
                "message": "event cursor predates the retained stream",
                **window,
            },
        )
    pending_count = int(
        session.scalar(
            select(func.count(EventRecord.id)).where(EventRecord.id > cursor)
        )
        or 0
    )
    if pending_count > max_replay_events:
        raise HTTPException(
            409,
            {
                "code": "replay_limit_exceeded",
                "message": "use the dashboard snapshot and reconnect from its latest cursor",
                "pending_count": pending_count,
                "max_replay_events": max_replay_events,
                **window,
            },
        )
    return {**window, "pending_count": pending_count}


async def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> None:
    settings: Settings = request.app.state.settings
    if not x_admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "valid X-Admin-Token required")


async def require_investigation_read(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    settings: Settings = request.app.state.settings
    if not bearer_token_valid(authorization, settings.mcp_read_token):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "valid investigation bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _evidence_snapshot(request: Request):
    settings: Settings = request.app.state.settings
    if not settings.evidence_api_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evidence API is disabled")
    try:
        return request.app.state.evidence_store.load()
    except EvidenceBundleUnavailable as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "verified evidence artifact is unavailable",
        ) from exc


@router.get(
    "/api/evidence",
    dependencies=[Depends(require_investigation_read)],
    tags=["evidence"],
)
async def evidence_index(request: Request) -> dict[str, object]:
    return bundle_index(_evidence_snapshot(request))


@router.get(
    "/api/evidence/cases/{case_id}",
    dependencies=[Depends(require_investigation_read)],
    tags=["evidence"],
)
async def evidence_case(case_id: str, request: Request) -> dict[str, object]:
    snapshot = _evidence_snapshot(request)
    selected = next(
        (item for item in snapshot.payload["cases"] if item["case_id"] == case_id),
        None,
    )
    if selected is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "evidence case not found")
    return selected


@router.get(
    "/api/evidence/bundle",
    dependencies=[Depends(require_investigation_read)],
    tags=["evidence"],
)
async def evidence_bundle(
    request: Request,
    if_none_match: str | None = Header(default=None, alias="If-None-Match"),
) -> Response:
    snapshot = _evidence_snapshot(request)
    etag = f'"{snapshot.transport_sha256}"'
    headers = {
        "Cache-Control": "private, no-cache",
        "ETag": etag,
        "X-Evidence-SHA256": snapshot.transport_sha256,
        "Content-Disposition": 'attachment; filename="zikit-sentinel-evidence.json"',
    }
    if if_none_match == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return Response(
        content=snapshot.content,
        media_type="application/json",
        headers=headers,
    )


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/health/live", tags=["health"])
async def liveness() -> dict[str, str]:
    return {"status": "alive", "version": "0.1.0"}


@router.get("/health/ready", tags=["health"])
async def readiness(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    sentinel = request.app.state.sentinel
    checks: dict[str, str] = {}
    try:
        with session_factory()() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "failed"

    monitor_expected = request.app.state.monitor_requested and settings.monitor_enabled
    monitor_running = sentinel._monitor_task is not None and not sentinel._monitor_task.done()
    checks["monitor"] = "ok" if monitor_running else ("disabled" if not monitor_expected else "failed")

    traffic_expected = request.app.state.monitor_requested and settings.traffic_enabled
    traffic_running = sentinel.traffic_generator.status.running
    checks["traffic"] = "ok" if traffic_running else ("disabled" if not traffic_expected else "failed")
    checks["model"] = "ok" if sentinel.detector.model is not None else "degraded"

    resource_monitor = request.app.state.resource_monitor
    resources_expected = request.app.state.monitor_requested and settings.resource_monitor_enabled
    if not resources_expected:
        checks["resources"] = "disabled"
    elif resource_monitor.status.running and resource_monitor.latest is not None:
        checks["resources"] = "ok"
    else:
        checks["resources"] = "failed"

    if not settings.evidence_api_enabled:
        checks["evidence"] = "disabled"
    else:
        try:
            request.app.state.evidence_store.load()
            checks["evidence"] = "ok"
        except EvidenceBundleUnavailable:
            # The demonstrator may degrade without stopping the monitoring and
            # mitigation path that protects the service.
            checks["evidence"] = "degraded"

    if settings.challenge_runtime_path is None:
        checks["challenge_runtime"] = "disabled"
    else:
        try:
            verify_deployment_bundle(settings.challenge_runtime_path)
            checks["challenge_runtime"] = "ok"
        except (FileNotFoundError, IngestionError, OSError, ValueError):
            checks["challenge_runtime"] = "failed"

    ready = (
        checks["database"] == "ok"
        and checks["monitor"] != "failed"
        and checks["traffic"] != "failed"
        and checks["resources"] != "failed"
        and checks["challenge_runtime"] != "failed"
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@router.get("/api/mcp/status", tags=["ai"])
async def mcp_status(request: Request) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    server = request.app.state.mcp_server
    return {
        "enabled": bool(server is not None and settings.mcp_http_enabled),
        "transport": "streamable-http" if server is not None and settings.mcp_http_enabled else None,
        "path": "/mcp/" if server is not None and settings.mcp_http_enabled else None,
        "authentication_required": settings.mcp_read_token is not None,
        "read_only": True,
        "tool_count": len(request.app.state.tool_registry.functions),
    }


@router.get("/api/resources", tags=["resources"])
async def resource_status(request: Request) -> dict[str, object]:
    return request.app.state.resource_monitor.current()


@router.get("/api/resources/history", tags=["resources"])
async def resource_history(
    limit: int = 100,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    if limit < 1 or limit > 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    records = session.scalars(
        select(ResourceMetricRecord).order_by(ResourceMetricRecord.timestamp.desc()).limit(limit)
    )
    return [
        {
            "source_id": record.source_id,
            "timestamp": record.timestamp,
            "cpu_percent": record.cpu_percent,
            "cpu_count": record.cpu_count,
            "load_1m": record.load_1m,
            "load_5m": record.load_5m,
            "load_15m": record.load_15m,
            "load_per_cpu": record.load_per_cpu,
            "memory_total_bytes": record.memory_total_bytes,
            "memory_available_bytes": record.memory_available_bytes,
            "memory_percent": record.memory_percent,
            "disk_path": record.disk_path,
            "disk_total_bytes": record.disk_total_bytes,
            "disk_free_bytes": record.disk_free_bytes,
            "disk_percent": record.disk_percent,
            "network_rx_bytes": record.network_rx_bytes,
            "network_tx_bytes": record.network_tx_bytes,
            "network_rx_errors": record.network_rx_errors,
            "network_tx_errors": record.network_tx_errors,
            "process_count": record.process_count,
            "containerized": record.containerized,
            "cgroup_memory_current_bytes": record.cgroup_memory_current_bytes,
            "cgroup_memory_limit_bytes": record.cgroup_memory_limit_bytes,
            "cgroup_memory_percent": record.cgroup_memory_percent,
            "cgroup_cpu_quota_cores": record.cgroup_cpu_quota_cores,
            "cgroup_cpu_percent": record.cgroup_cpu_percent,
            "anomalous": record.anomalous,
            "reasons": record.reasons,
        }
        for record in records
    ]


@router.get("/api/resources/events", tags=["resources"])
async def resource_events(
    limit: int = 100,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    if limit < 1 or limit > 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    records = session.scalars(
        select(EventRecord)
        .where(EventRecord.event_type.in_(["RESOURCE_ANOMALY_DETECTED", "RESOURCE_RECOVERED"]))
        .order_by(EventRecord.timestamp.desc())
        .limit(limit)
    )
    return [
        {
            "type": record.event_type,
            "timestamp": record.timestamp,
            "details": record.details,
        }
        for record in records
    ]


@router.get("/api/services", tags=["read"])
async def services(request: Request, session: Session = Depends(get_session)) -> list[dict[str, object]]:
    active = request.app.state.simulator.active_provider
    faulted = {
        name: await request.app.state.sentinel.fault_controller.is_active(name)
        for name in ("primary", "backup")
    }
    persisted = {service.name: service for service in session.scalars(select(ServiceRecord))}
    return [
        {
            "name": name,
            "active": name == active,
            "healthy": persisted[name].healthy if name in persisted else True,
            "updated_at": persisted[name].updated_at if name in persisted else None,
            "faulted": faulted[name],
        }
        for name in ("primary", "backup")
    ]


@router.get("/api/traffic", tags=["read"])
async def traffic_status(request: Request) -> dict[str, object]:
    return request.app.state.sentinel.traffic_generator.status.__dict__


@router.get("/api/probes", tags=["read"])
async def probe_status(request: Request) -> dict[str, object]:
    return request.app.state.sentinel.probes.status.as_public_dict()


@router.get("/api/slo", tags=["read"])
async def slo_status(request: Request) -> dict[str, object]:
    sentinel = request.app.state.sentinel
    settings = request.app.state.settings
    active = request.app.state.simulator.active_provider
    return {
        "active_service": active,
        "objectives": {
            "latency_p95_ms": settings.slo_latency_p95_ms,
            "error_rate": settings.slo_error_rate,
            "at_risk_ratio": settings.slo_at_risk_ratio,
            "breach_persistence_windows": settings.slo_breach_persistence_windows,
            "customer_impact_persistence_windows": settings.customer_impact_persistence_windows,
            "customer_complaint_delay_seconds": settings.customer_complaint_delay_seconds,
        },
        "current_metrics": sentinel.telemetry.snapshot(active).model_dump(mode="json"),
        "state": sentinel.slo.status(active),
    }


@router.get("/api/metrics", tags=["read"])
async def metrics(request: Request) -> list[dict[str, object]]:
    telemetry = request.app.state.sentinel.telemetry
    return [telemetry.snapshot(name).model_dump(mode="json") for name in ("primary", "backup")]


@router.get("/api/metrics/history", tags=["read"])
async def metrics_history(
    service: str | None = None,
    limit: int = 100,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    if limit < 1 or limit > 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    query = select(MetricRecord)
    if service:
        query = query.where(MetricRecord.service == service)
    query = query.order_by(MetricRecord.timestamp.desc()).limit(limit)
    records = list(session.scalars(query))
    return [
        {
            "service": record.service,
            "timestamp": record.timestamp,
            "latency_ms": record.latency_ms,
            "status_code": record.status_code,
            "timed_out": record.timed_out,
            "success": record.success,
        }
        for record in records
    ]


@router.get("/api/events", tags=["operations"])
async def events_feed(
    after_id: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    if after_id < 0:
        raise HTTPException(422, "after_id must be non-negative")
    if limit < 1 or limit > 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    records = session.scalars(
        select(EventRecord)
        .where(EventRecord.id > after_id)
        .order_by(EventRecord.id)
        .limit(limit)
    )
    return [serialize_event(record) for record in records]


@router.get("/api/events/cursor", tags=["operations"])
async def events_cursor(
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    capacity = request.app.state.event_stream_capacity.snapshot()
    return {
        **event_cursor_status(session),
        "replay": {
            "batch_size": settings.event_stream_batch_size,
            "max_events": settings.event_stream_max_replay_events,
        },
        "stream": {
            "active_connections": capacity.active_connections,
            "max_connections": capacity.max_connections,
            "total_opened": capacity.total_opened,
            "total_rejected": capacity.total_rejected,
            "retry_milliseconds": settings.event_stream_retry_milliseconds,
        },
    }


@router.get("/api/events/stream", tags=["operations"])
async def events_stream(
    request: Request,
    after_id: int = 0,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
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
    settings: Settings = request.app.state.settings
    with session_factory()() as session:
        cursor_window = validate_stream_cursor(
            session,
            cursor,
            settings.event_stream_max_replay_events,
        )
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
                with session_factory()() as session:
                    records = list(
                        session.scalars(
                            select(EventRecord)
                            .where(EventRecord.id > cursor)
                            .order_by(EventRecord.id)
                            .limit(settings.event_stream_batch_size)
                        )
                    )
                    rendered = [format_sse(record) for record in records]
                    if records:
                        cursor = records[-1].id
                if rendered:
                    for item in rendered:
                        yield item
                    last_heartbeat = time.monotonic()
                elif time.monotonic() - last_heartbeat >= settings.event_stream_heartbeat_seconds:
                    yield f"event: heartbeat\ndata: {{\"cursor\":{cursor}}}\n\n"
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
            "X-Event-Latest": str(cursor_window["latest_id"]),
        },
    )


@router.get("/api/alerts", tags=["operations"])
async def alerts(
    after_id: int = 0,
    limit: int = 100,
    state: str | None = None,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    if after_id < 0:
        raise HTTPException(422, "after_id must be non-negative")
    if limit < 1 or limit > 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    if state not in {None, "open", "acknowledged"}:
        raise HTTPException(422, "state must be open or acknowledged")
    query = (
        select(AlertRecord, EventRecord)
        .join(EventRecord, EventRecord.id == AlertRecord.event_id)
        .where(AlertRecord.id > after_id)
    )
    if state is not None:
        query = query.where(AlertRecord.state == state)
    rows = session.execute(query.order_by(AlertRecord.id).limit(limit))
    return [serialize_alert(alert, event) for alert, event in rows]


@router.post(
    "/api/admin/alerts/{alert_id}/acknowledge",
    dependencies=[Depends(require_admin)],
    tags=["admin"],
)
async def acknowledge_alert(
    alert_id: int,
    session: Session = Depends(get_session),
) -> dict[str, object]:
    alert, challenge_trace, _ = acknowledge_alert_once(
        session,
        alert_id,
        acknowledged_at=datetime.now(UTC),
    )
    if alert is None:
        session.rollback()
        raise HTTPException(404, "alert not found")
    session.commit()
    session.refresh(alert)
    return {
        "id": alert.id,
        "state": alert.state,
        "acknowledged_at": alert.acknowledged_at,
        "challenge_trace": challenge_trace,
    }


@router.get("/api/dashboard", tags=["operations"])
async def dashboard(request: Request, session: Session = Depends(get_session)) -> dict[str, object]:
    sentinel = request.app.state.sentinel
    active_provider = request.app.state.simulator.active_provider
    status_counts = {
        status_name: count
        for status_name, count in session.execute(
            select(IncidentRecord.status, func.count(IncidentRecord.id)).group_by(
                IncidentRecord.status
            )
        )
    }
    latest_incident = session.scalar(
        select(IncidentRecord).order_by(IncidentRecord.detected_at.desc()).limit(1)
    )
    event_window = event_cursor_status(session)
    stream_capacity = request.app.state.event_stream_capacity.snapshot()
    open_alert_count = session.scalar(
        select(func.count(AlertRecord.id)).where(AlertRecord.state == "open")
    ) or 0
    alert_rows = session.execute(
        select(AlertRecord, EventRecord)
        .join(EventRecord, EventRecord.id == AlertRecord.event_id)
        .order_by(AlertRecord.id.desc())
        .limit(10)
    )
    latest_alerts = [serialize_alert(alert, event) for alert, event in alert_rows]
    latest_alerts.reverse()
    metrics = {
        name: sentinel.telemetry.snapshot(name).model_dump(mode="json")
        for name in ("primary", "backup")
    }
    return {
        "generated_at": datetime.now(UTC),
        "active_provider": active_provider,
        "monitor": {
            "running": (
                sentinel._monitor_task is not None and not sentinel._monitor_task.done()
            ),
            "probe": sentinel.probes.status.as_public_dict(),
            "traffic": sentinel.traffic_generator.status.__dict__,
        },
        "model": {
            "enabled": sentinel.detector.model is not None,
            "persistence_windows": sentinel.detector.persistence,
        },
        "services": metrics,
        "slo": {
            "active_service": active_provider,
            "state": sentinel.slo.status(active_provider),
        },
        "incidents": {
            "counts_by_status": status_counts,
            "latest": (
                IncidentResponse.model_validate(latest_incident).model_dump(mode="json")
                if latest_incident is not None
                else None
            ),
        },
        "events": {
            **event_window,
            "stream": {
                "active_connections": stream_capacity.active_connections,
                "max_connections": stream_capacity.max_connections,
                "total_opened": stream_capacity.total_opened,
                "total_rejected": stream_capacity.total_rejected,
            },
        },
        "alerts": {
            "open_count": open_alert_count,
            "latest": latest_alerts,
        },
        "resources": request.app.state.resource_monitor.current(),
    }


@router.get("/api/model", tags=["model"])
async def model_info(request: Request) -> dict[str, object]:
    detector = request.app.state.sentinel.detector
    model = detector.model
    if model is None:
        return {
            "enabled": False,
            "features": list(FEATURES),
            "persistence_windows": detector.persistence,
            "minimum_samples": detector.minimum_ml_samples,
        }
    return {
        "enabled": True,
        "estimator": type(model.estimator).__name__,
        "feature_version": model.feature_version,
        "features": list(model.feature_names),
        "threshold": model.threshold,
        "training_rows": model.training_rows,
        "persistence_windows": detector.persistence,
        "minimum_samples": detector.minimum_ml_samples,
    }


@router.post("/api/model/score", tags=["model"])
async def score_model(payload: ModelScoreRequest, request: Request) -> dict[str, object]:
    model = request.app.state.sentinel.detector.model
    if model is None:
        raise HTTPException(503, "ML model is not loaded")
    values = payload.model_dump()
    vector = [[float(values[name]) for name in FEATURES]]
    combined, isolation, feature_components = model.score_components(vector)
    contributions = {
        name: float(feature_components[0][index]) for index, name in enumerate(FEATURES)
    }
    dominant_feature = max(contributions, key=contributions.get)
    score = float(combined[0])
    return {
        "score": score,
        "threshold": model.threshold,
        "anomalous": score >= model.threshold,
        "margin": score - model.threshold,
        "isolation_component": float(isolation[0]),
        "feature_contributions": contributions,
        "dominant_feature": dominant_feature,
    }


@router.get("/api/incidents", response_model=list[IncidentResponse], tags=["read"])
async def incidents(session: Session = Depends(get_session)) -> list[IncidentRecord]:
    result = session.execute(select(IncidentRecord).order_by(IncidentRecord.detected_at.desc()))
    return list(result.scalars())


@router.get("/api/incidents/{incident_id}", response_model=IncidentResponse, tags=["read"])
async def incident(incident_id: str, session: Session = Depends(get_session)) -> IncidentRecord:
    record = session.get(IncidentRecord, incident_id)
    if not record:
        raise HTTPException(404, "incident not found")
    return record


@router.get("/api/incidents/{incident_id}/events", tags=["read"])
async def incident_events(incident_id: str, session: Session = Depends(get_session)) -> list[dict[str, object]]:
    result = session.execute(
        select(EventRecord).where(EventRecord.incident_id == incident_id).order_by(EventRecord.monotonic_seconds)
    )
    return [
        {"type": event.event_type, "timestamp": event.timestamp, "details": event.details}
        for event in result.scalars()
    ]


@router.get("/api/incidents/{incident_id}/actions", tags=["read"])
async def incident_actions(incident_id: str, session: Session = Depends(get_session)) -> list[dict[str, object]]:
    if session.get(IncidentRecord, incident_id) is None:
        raise HTTPException(404, "incident not found")
    actions = session.scalars(
        select(ActionRecord).where(ActionRecord.incident_id == incident_id).order_by(ActionRecord.started_at)
    )
    return [
        {
            "id": action.id,
            "runbook": action.runbook,
            "state": action.state,
            "started_at": action.started_at,
            "completed_at": action.completed_at,
            "details": action.details,
        }
        for action in actions
    ]


@router.get("/api/incidents/{incident_id}/explanation", tags=["read"])
async def incident_explanation(incident_id: str) -> dict[str, object]:
    try:
        return build_incident_explanation(incident_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get(
    "/api/incidents/{incident_id}/investigations",
    dependencies=[Depends(require_investigation_read)],
    tags=["ai"],
)
async def incident_investigations(
    incident_id: str,
    limit: int = 20,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    if limit < 1 or limit > 100:
        raise HTTPException(422, "limit must be between 1 and 100")
    if session.get(IncidentRecord, incident_id) is None:
        raise HTTPException(404, "incident not found")
    records = session.scalars(
        select(InvestigationRecord)
        .where(InvestigationRecord.incident_id == incident_id)
        .order_by(InvestigationRecord.created_at.desc(), InvestigationRecord.id.desc())
        .limit(limit)
    )
    return [
        {
            "id": record.id,
            "incident_id": record.incident_id,
            "created_at": record.created_at,
            "provider": record.provider,
            "diagnosis": json.loads(record.diagnosis),
        }
        for record in records
    ]


@router.post("/api/admin/experiments", status_code=202, dependencies=[Depends(require_admin)], tags=["admin"])
async def create_experiment(payload: ExperimentCreate, request: Request, session: Session = Depends(get_session)) -> dict[str, object]:
    sentinel = request.app.state.sentinel
    if payload.replay_group_id is not None:
        for existing in session.scalars(select(ExperimentRecord)):
            metadata = replay_metadata(existing)
            if (
                metadata.get("group_id") == payload.replay_group_id
                and metadata.get("repetition") == payload.repetition
                and existing.mode == payload.mode.value
            ):
                raise HTTPException(409, "this replay group repetition and mode already exists")
    experiment_id = new_experiment_id()
    experiment = ExperimentRecord(
        id=experiment_id,
        mode=payload.mode.value,
        status="running",
        scenario=persisted_scenario(
            payload.scenario,
            replay_group_id=payload.replay_group_id,
            repetition=payload.repetition,
        ),
        created_at=datetime.now(UTC),
    )
    session.add(experiment)
    session.commit()
    try:
        await sentinel.begin_experiment(experiment_id, payload.scenario, payload.mode)
    except FaultAlreadyActiveError as exc:
        session.delete(experiment)
        session.commit()
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        experiment.status = "failed"
        experiment.completed_at = datetime.now(UTC)
        experiment.result = {
            "measurement_valid": False,
            "failure": "fault_control_failed",
            "error_type": type(exc).__name__,
        }
        session.commit()
        raise HTTPException(502, "fault controller could not start the experiment") from exc
    return {
        "id": experiment_id,
        "status": "running",
        "mode": payload.mode,
        "scenario_fingerprint": scenario_fingerprint(payload.scenario),
        "replay_group_id": payload.replay_group_id,
        "repetition": payload.repetition,
    }


@router.get("/api/experiments", tags=["read"])
async def experiments(
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    mode: str | None = None,
    replay_group_id: str | None = None,
    limit: int = 50,
    session: Session = Depends(get_session),
) -> list[dict[str, object]]:
    allowed_statuses = {"running", "completed", "failed", "stopped", "interrupted"}
    if status_filter is not None and status_filter not in allowed_statuses:
        raise HTTPException(422, "unsupported experiment status")
    if mode is not None and mode not in {"automatic", "manual"}:
        raise HTTPException(422, "unsupported experiment mode")
    if replay_group_id is not None and not replay_group_id:
        raise HTTPException(422, "replay_group_id cannot be empty")
    if limit < 1 or limit > 200:
        raise HTTPException(422, "limit must be between 1 and 200")

    statement = select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())
    if status_filter is not None:
        statement = statement.where(ExperimentRecord.status == status_filter)
    if mode is not None:
        statement = statement.where(ExperimentRecord.mode == mode)
    records = list(session.scalars(statement))
    if replay_group_id is not None:
        records = [
            record
            for record in records
            if replay_metadata(record).get("group_id") == replay_group_id
        ]
    return [serialize_experiment(record) for record in records[:limit]]


@router.post("/api/admin/replay-plan", dependencies=[Depends(require_admin)], tags=["admin"])
async def replay_plan(payload: ReplayPlanRequest) -> dict[str, object]:
    return build_replay_plan(payload)


@router.post("/api/admin/lab/reset", dependencies=[Depends(require_admin)], tags=["admin"])
async def reset_lab(request: Request, session: Session = Depends(get_session)) -> dict[str, object]:
    running = session.scalar(select(ExperimentRecord).where(ExperimentRecord.status == "running").limit(1))
    if running is not None:
        raise HTTPException(409, "cannot reset while an experiment is running")
    await request.app.state.sentinel.reset_runtime()
    return {
        "status": "reset",
        "active_provider": request.app.state.simulator.active_provider,
        "telemetry_cleared": True,
        "history_preserved": True,
    }


@router.get("/api/admin/runtime", dependencies=[Depends(require_admin)], tags=["admin"])
async def runtime_identity(request: Request) -> dict[str, object]:
    return {
        "pid": os.getpid(),
        "process_id": request.app.state.sentinel.repository.process_id,
        "monitor_running": (
            request.app.state.sentinel._monitor_task is not None
            and not request.app.state.sentinel._monitor_task.done()
        ),
    }


@router.post("/api/admin/experiments/{experiment_id}/stop", dependencies=[Depends(require_admin)], tags=["admin"])
async def stop_experiment(experiment_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, str]:
    experiment = session.get(ExperimentRecord, experiment_id)
    if not experiment:
        raise HTTPException(404, "experiment not found")
    await request.app.state.sentinel.stop_experiment(
        experiment_id,
        experiment.scenario["service"],
    )
    experiment.status = "stopped"
    experiment.completed_at = datetime.now(UTC)
    session.commit()
    return {"id": experiment_id, "status": "stopped"}


@router.post("/api/incidents/{incident_id}/approve", dependencies=[Depends(require_admin)], tags=["admin"])
async def approve_incident(incident_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, str]:
    incident = session.get(IncidentRecord, incident_id)
    if not incident:
        raise HTTPException(404, "incident not found")
    if incident.status != "awaiting_approval":
        raise HTTPException(409, "incident is not awaiting approval")
    await request.app.state.sentinel.mitigate(incident.id, incident.experiment_id)
    return {"id": incident.id, "status": "mitigation_requested"}


@router.get("/api/experiments/{experiment_id}/results", tags=["read"])
async def experiment_results(experiment_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, object]:
    experiment = session.get(ExperimentRecord, experiment_id)
    if not experiment:
        raise HTTPException(404, "experiment not found")
    result = experiment.result or await request.app.state.sentinel.repository.calculate_result(experiment_id)
    return {"id": experiment.id, "status": experiment.status, "mode": experiment.mode, "result": result}


@router.get("/api/experiments/{experiment_id}/report", tags=["read"])
async def experiment_report(experiment_id: str, request: Request, session: Session = Depends(get_session)) -> dict[str, object]:
    return await experiment_results(experiment_id, request, session)


@router.get("/api/reports/comparison", tags=["read"])
async def comparison_report(automatic_id: str, manual_id: str, session: Session = Depends(get_session)) -> dict[str, object]:
    automatic = session.get(ExperimentRecord, automatic_id)
    manual = session.get(ExperimentRecord, manual_id)
    if not automatic or not manual:
        raise HTTPException(404, "one or both experiments were not found")
    if automatic.mode != "automatic" or manual.mode != "manual":
        raise HTTPException(422, "automatic_id and manual_id must match their respective modes")
    try:
        fingerprint = validate_pair(automatic, manual)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    auto_time = (automatic.result or {}).get("durations", {}).get("first_signal_to_recovery_seconds")
    manual_time = (manual.result or {}).get("durations", {}).get("first_signal_to_recovery_seconds")
    if auto_time is None or manual_time is None or manual_time <= 0:
        raise HTTPException(409, "both experiments need valid measured recovery durations")
    auto_action_time = (automatic.result or {}).get("durations", {}).get(
        "first_signal_to_action_seconds"
    )
    manual_action_time = (manual.result or {}).get("durations", {}).get(
        "first_signal_to_action_seconds"
    )
    action_reduction = None
    if (
        auto_action_time is not None
        and manual_action_time is not None
        and manual_action_time > 0
    ):
        action_reduction = (manual_action_time - auto_action_time) / manual_action_time * 100
    return {
        "automatic_experiment_id": automatic_id,
        "manual_experiment_id": manual_id,
        "scenario_fingerprint": fingerprint,
        "automatic_seconds": auto_time,
        "manual_seconds": manual_time,
        "reduction_percent": (manual_time - auto_time) / manual_time * 100,
        "action_comparison": {
            "automatic_seconds": auto_action_time,
            "manual_seconds": manual_action_time,
            "reduction_percent": action_reduction,
        },
        "impact_comparison": {
            "automatic": (automatic.result or {}).get("impact", {}),
            "manual": (manual.result or {}).get("impact", {}),
            "customer_impact_prevented_vs_manual": (
                bool((manual.result or {}).get("impact", {}).get("customer_impact_observed"))
                and bool((automatic.result or {}).get("impact", {}).get("customer_impact_avoided"))
            ),
            "customer_complaint_prevented_vs_manual": (
                bool((manual.result or {}).get("impact", {}).get("customer_complaint_observed"))
                and bool((automatic.result or {}).get("impact", {}).get("customer_complaint_avoided"))
            ),
        },
    }


@router.get("/api/reports/replay-groups/{group_id}", tags=["read"])
async def replay_group_report(group_id: str, session: Session = Depends(get_session)) -> dict[str, object]:
    records = list(session.scalars(select(ExperimentRecord)))
    try:
        return aggregate_replay_group(records, group_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
