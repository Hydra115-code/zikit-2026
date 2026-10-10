from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.core.incidents import (
    EVENT_ACTION_BLOCKED,
    EVENT_ACTION_COMPLETED,
    EVENT_ACTION_STARTED,
    EVENT_FIRST_SIGNAL,
    EVENT_INCIDENT_DETECTED,
    EVENT_RECOVERY_FAILED,
    EVENT_RECOVERY_VERIFIED,
)
from app.core.slo import EVENT_CUSTOMER_COMPLAINT_SIMULATED, EVENT_CUSTOMER_IMPACT
from app.database import ActionRecord, EventRecord, ExperimentRecord, IncidentRecord, session_factory


_REASON_LABELS = {
    "availability_below_threshold": "la disponibilidad cayó por debajo del umbral",
    "error_rate_exceeded": "la tasa de errores superó el umbral",
    "latency_p95_exceeded": "la latencia p95 superó el umbral",
    "model_score_exceeded": "el puntaje del modelo superó el umbral",
    "timeout_rate_exceeded": "la tasa de timeouts superó el umbral",
}


def _event_map(events: list[EventRecord]) -> dict[str, EventRecord]:
    first: dict[str, EventRecord] = {}
    for event in events:
        first.setdefault(event.event_type, event)
    return first


def _seconds_between(start: EventRecord | None, end: EventRecord | None) -> float | None:
    if start is None or end is None or start.process_id != end.process_id:
        return None
    return round(max(0.0, end.monotonic_seconds - start.monotonic_seconds), 3)


def _reason_text(details: dict[str, Any]) -> str:
    reasons = details.get("reasons") or []
    normalized: list[str] = []
    for reason in reasons:
        key = reason.get("reason") if isinstance(reason, dict) else str(reason)
        if key:
            normalized.append(_REASON_LABELS.get(key, key.replace("_", " ")))
    if normalized:
        return ", ".join(normalized)
    if details.get("reason"):
        return str(details["reason"])
    return "la telemetría persistente cruzó los umbrales configurados"


def _verification_metrics(event: EventRecord | None) -> dict[str, Any]:
    if event is None:
        return {}
    allowed = {
        "availability",
        "error_rate",
        "p95_latency_ms",
        "samples",
        "consecutive_successes",
        "stable",
        "provider_consistent",
        "verification_duration_seconds",
    }
    return {key: value for key, value in event.details.items() if key in allowed}


def build_incident_explanation(incident_id: str) -> dict[str, Any]:
    """Build a deterministic, read-only explanation from persisted evidence."""
    with session_factory()() as session:
        incident = session.get(IncidentRecord, incident_id)
        if incident is None:
            raise LookupError("incident not found")
        if incident.evidence.get("source") == "zikit_official_synthetic_dataset":
            from app.challenge.explanations import build_challenge_incident_explanation

            return build_challenge_incident_explanation(incident_id)
        event_scope = (
            EventRecord.experiment_id == incident.experiment_id
            if incident.experiment_id
            else EventRecord.incident_id == incident_id
        )
        events = list(
            session.scalars(
                select(EventRecord)
                .where(event_scope)
                .order_by(EventRecord.monotonic_seconds, EventRecord.id)
            )
        )
        actions = list(
            session.scalars(
                select(ActionRecord)
                .where(ActionRecord.incident_id == incident_id)
                .order_by(ActionRecord.started_at, ActionRecord.id)
            )
        )
        experiment = (
            session.get(ExperimentRecord, incident.experiment_id)
            if incident.experiment_id
            else None
        )

    first = _event_map(events)
    signal = first.get(EVENT_FIRST_SIGNAL) or first.get(EVENT_INCIDENT_DETECTED)
    started = first.get(EVENT_ACTION_STARTED)
    completed = first.get(EVENT_ACTION_COMPLETED)
    blocked = first.get(EVENT_ACTION_BLOCKED)
    verified = first.get(EVENT_RECOVERY_VERIFIED)
    failed = first.get(EVENT_RECOVERY_FAILED)
    customer_impact = first.get(EVENT_CUSTOMER_IMPACT)
    complaint = first.get(EVENT_CUSTOMER_COMPLAINT_SIMULATED)
    action = actions[0] if actions else None

    stored_result = experiment.result if experiment and experiment.result else {}
    stored_durations = stored_result.get("durations") or {}
    signal_to_action = stored_durations.get("first_signal_to_action_seconds")
    if signal_to_action is None:
        signal_to_action = _seconds_between(signal, started)
    signal_to_recovery = stored_durations.get("first_signal_to_recovery_seconds")
    if signal_to_recovery is None:
        signal_to_recovery = _seconds_between(signal, verified)
    measurement_valid = bool(
        stored_result.get("measurement_valid")
        if stored_result
        else verified is not None and signal_to_recovery is not None
    )

    signal_reason = _reason_text(signal.details if signal else incident.evidence)
    signal_detail = f"En {incident.service} se observó que {signal_reason}."

    if blocked is not None:
        decision_status = "blocked"
        decision_detail = (
            "La política de seguridad bloqueó la mitigación: "
            f"{blocked.details.get('reason', 'no se cumplieron las precondiciones')}"
        )
    elif started is not None:
        decision_status = "authorized"
        decision_detail = (
            "La política validó las precondiciones y autorizó solamente el runbook permitido."
        )
    elif incident.status == "awaiting_approval":
        decision_status = "awaiting_approval"
        decision_detail = "El modo manual exige aprobación humana antes de ejecutar el runbook."
    else:
        decision_status = "evaluating"
        decision_detail = "Sentinel está evaluando las precondiciones de una mitigación segura."

    runbook = action.runbook if action else (started.details.get("runbook") if started else None)
    if blocked is not None:
        action_status = "blocked"
        action_detail = "No se ejecutó ningún cambio porque la política lo impidió."
    elif completed is not None and action is not None:
        action_status = action.state
        action_detail = f"El runbook {runbook} terminó con estado {action.state}."
    elif started is not None:
        action_status = "running"
        action_detail = f"El runbook {runbook or 'permitido'} está en ejecución."
    else:
        action_status = "not_started"
        action_detail = "Todavía no se ha ejecutado una acción."

    if verified is not None:
        verification_status = "verified"
        verification_detail = (
            "La telemetría posterior confirmó estabilidad y consistencia del proveedor activo."
        )
    elif failed is not None:
        verification_status = "failed"
        verification_detail = (
            "La verificación posterior no confirmó recuperación: "
            f"{failed.details.get('reason', 'la recuperación no cumplió los criterios')}"
        )
    else:
        verification_status = "pending"
        verification_detail = "Aún no existe evidencia suficiente para confirmar la recuperación."

    if verified is not None and customer_impact is None:
        impact_label = "impact_avoided_in_run"
        impact_detail = (
            "En esta corrida controlada se verificó la recuperación antes de observar impacto al cliente."
        )
    elif customer_impact is not None:
        impact_label = "impact_observed"
        impact_detail = "La corrida registró impacto al cliente antes de cerrar la recuperación."
    else:
        impact_label = "not_yet_determined"
        impact_detail = "La evidencia actual no permite determinar todavía el resultado sobre el cliente."

    if verified is not None:
        headline = f"{incident.service}: recuperación verificada"
        next_step = "Conservar la evidencia y comparar esta corrida con su escenario control equivalente."
    elif failed is not None or blocked is not None:
        headline = f"{incident.service}: mitigación sin recuperación verificada"
        next_step = "Revisar la precondición fallida antes de intentar otra acción."
    elif incident.status == "awaiting_approval":
        headline = f"{incident.service}: acción segura pendiente de aprobación"
        next_step = "Un operador debe revisar la evidencia y aprobar o rechazar el runbook permitido."
    else:
        headline = f"{incident.service}: anomalía detectada, evaluación en curso"
        next_step = "Esperar la decisión de política y la verificación posterior."

    controlled_lab = experiment is not None
    environment = "controlled_local_lab" if controlled_lab else "observed_runtime"
    limitations = (
        [
            "Escenario de laboratorio local con falla inyectada; no representa infraestructura remota real.",
            "El impacto evitado describe esta corrida observada, no un contrafactual universal.",
        ]
        if controlled_lab
        else ["La explicación describe evidencia observada y no afirma una causa raíz no demostrada."]
    )

    return {
        "version": "1",
        "source": "deterministic-local",
        "incident_id": incident.id,
        "experiment_id": incident.experiment_id,
        "service": incident.service,
        "severity": incident.severity,
        "status": incident.status,
        "environment": environment,
        "headline": headline,
        "summary": f"{signal_detail} {decision_detail} {verification_detail}",
        "signal": {
            "title": "Primera señal",
            "status": "observed" if signal else "missing",
            "detail": signal_detail,
            "observed_at": signal.timestamp.isoformat() if signal else incident.first_signal_at.isoformat(),
        },
        "decision": {
            "title": "Decisión segura",
            "status": decision_status,
            "detail": decision_detail,
        },
        "action": {
            "title": "Acción efectiva",
            "status": action_status,
            "detail": action_detail,
            "runbook": runbook,
            "started_at": action.started_at.isoformat() if action else (started.timestamp.isoformat() if started else None),
            "completed_at": action.completed_at.isoformat() if action and action.completed_at else None,
        },
        "verification": {
            "title": "Verificación posterior",
            "status": verification_status,
            "detail": verification_detail,
            "metrics": _verification_metrics(verified or failed),
        },
        "timing": {
            "measurement_valid": measurement_valid,
            "first_signal_to_action_seconds": signal_to_action,
            "first_signal_to_recovery_seconds": signal_to_recovery,
        },
        "impact": {
            "label": impact_label,
            "customer_impact_observed": customer_impact is not None,
            "customer_complaint_simulated": complaint is not None,
            "detail": impact_detail,
        },
        "next_step": next_step,
        "evidence_event_ids": [event.id for event in events],
        "limitations": limitations,
    }
