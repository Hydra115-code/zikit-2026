from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.challenge.business_impact import kaiku_potential_impact
from app.challenge.persistence import (
    CHALLENGE_SOURCE,
    EVENT_CHALLENGE_ALERT_ACKNOWLEDGED,
)
from app.challenge.response import response_plan_for
from app.database import (
    ChallengePredictionRecord,
    ChallengeRemediationRecord,
    EventRecord,
    IncidentRecord,
    session_factory,
)


_SIGNAL_LABELS = {
    "cpu_past_robust_z_1440m": "CPU fuera de su comportamiento histórico",
    "ram_past_robust_z_1440m": "RAM fuera de su comportamiento histórico",
    "disco_past_robust_z_1440m": "disco fuera de su comportamiento histórico",
    "latencia_ms_past_robust_z_1440m": "latencia fuera de su comportamiento histórico",
    "errores_past_robust_z_1440m": "errores fuera de su comportamiento histórico",
    "risk_40m": "riesgo predictivo a 40 minutos",
    "risk_120m": "riesgo predictivo a 120 minutos",
    "risk_360m": "riesgo predictivo a 6 horas",
    "risk_1440m": "riesgo predictivo a 24 horas",
    "isolation_risk": "desviación multivariable",
    "normal_behavior_risk": "desviación frente al patrón normal de Kaiku para ese horario y contexto",
}


def _top_signals(evidence: object) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    result: list[dict[str, Any]] = []
    for item in evidence[:3]:
        if not isinstance(item, dict) or not item.get("signal"):
            continue
        signal = str(item["signal"])
        result.append(
            {
                "signal": signal,
                "label": _SIGNAL_LABELS.get(signal, signal.replace("_", " ")),
                "value": item.get("value"),
            }
        )
    return result


def _signal_detail(tenant: str, signals: list[dict[str, Any]]) -> str:
    if not signals:
        return f"En {tenant} persistió una combinación anómala de métricas causales."
    rendered = ", ".join(
        f"{item['label']}={item['value']}" for item in signals
    )
    return f"En {tenant} destacaron estas señales: {rendered}."


def _operator_response(events: list[EventRecord]) -> dict[str, Any]:
    acknowledged = next(
        (
            event
            for event in reversed(events)
            if event.event_type == EVENT_CHALLENGE_ALERT_ACKNOWLEDGED
        ),
        None,
    )
    if acknowledged is None:
        return {
            "status": "pending",
            "acknowledged_at": None,
            "delivery_to_acknowledgement_seconds": None,
            "clock_domain": "operator_wall_clock",
            "mitigation_executed": False,
        }
    details = acknowledged.details if isinstance(acknowledged.details, dict) else {}
    return {
        "status": "acknowledged",
        "acknowledged_at": acknowledged.timestamp.isoformat(),
        "delivery_to_acknowledgement_seconds": details.get(
            "delivery_to_acknowledgement_seconds"
        ),
        "clock_domain": "operator_wall_clock",
        "mitigation_executed": False,
    }


def _remediation(incident_id: str) -> ChallengeRemediationRecord | None:
    with session_factory()() as session:
        return session.scalar(
            select(ChallengeRemediationRecord).where(
                ChallengeRemediationRecord.incident_id == incident_id
            )
        )


def _load_by_incident(
    incident_id: str,
) -> tuple[IncidentRecord, ChallengePredictionRecord, list[EventRecord]]:
    with session_factory()() as session:
        incident = session.get(IncidentRecord, incident_id)
        if incident is None:
            raise LookupError("incident not found")
        prediction = session.scalar(
            select(ChallengePredictionRecord).where(
                ChallengePredictionRecord.incident_id == incident_id
            )
        )
        if prediction is None:
            raise LookupError("challenge prediction not found")
        events = list(
            session.scalars(
                select(EventRecord)
                .where(EventRecord.incident_id == incident_id)
                .order_by(EventRecord.monotonic_seconds, EventRecord.id)
            )
        )
    return incident, prediction, events


def build_challenge_incident_explanation(incident_id: str) -> dict[str, Any]:
    """Explain a prediction without turning a hypothesis into a demonstrated cause."""
    incident, prediction, events = _load_by_incident(incident_id)
    signals = _top_signals(prediction.evidence)
    signal_detail = _signal_detail(prediction.tenant, signals)
    detection_seconds = max(
        0.0,
        (prediction.alert_timestamp - prediction.signal_timestamp).total_seconds(),
    )
    incident_type = prediction.suspected_incident_type or "incidente no clasificado"
    response_plan = response_plan_for(
        prediction.alert_type,
        prediction.suspected_incident_type,
    )
    remediation = _remediation(incident_id)
    decision_detail = (
        f"La política emitió una alerta con confianza {prediction.risk_score:.1%} y "
        f"clasificación probable “{incident_type}”; no demuestra una causa raíz."
    )
    action_executed = remediation is not None and remediation.action_completed_at is not None
    if remediation is None:
        action_status = "not_started"
        action_detail = (
            "La alerta espera respuesta durante el plazo configurado. Si permanece abierta, "
            "Gemini solo podrá seleccionar un runbook preautorizado y reversible."
        )
        verification_status = "pending"
        verification_detail = "La verificación comenzará únicamente después de una acción MCP."
    elif remediation.status == "recovered":
        action_status = "completed"
        action_detail = (
            f"MCP ejecutó {remediation.runbook_id} en infraestructura controlada."
        )
        verification_status = "verified"
        verification_detail = (
            "Sentinel verificó la recuperación durante la ventana configurada de la "
            "infraestructura controlada."
        )
    elif remediation.status == "escalated":
        action_status = "failed" if action_executed else "blocked"
        action_detail = (
            "La remediación se escaló al operador"
            + (f" después de ejecutar {remediation.runbook_id}." if action_executed else " sin ejecutar cambios.")
        )
        verification_status = "failed" if action_executed else "pending"
        verification_detail = (
            "La verificación no superó el contrato y se intentó rollback."
            if action_executed
            else "No hubo acción que verificar."
        )
    else:
        action_status = "running"
        action_detail = f"Remediación controlada en estado {remediation.status}."
        verification_status = "pending"
        verification_detail = "La ventana de verificación todavía no concluye."
    operator_response = _operator_response(events)
    operator_response["mitigation_executed"] = action_executed
    return {
        "version": "2",
        "source": "deterministic-local",
        "incident_id": incident.id,
        "experiment_id": None,
        "service": incident.service,
        "tenant": prediction.tenant,
        "run_id": prediction.run_id,
        "severity": incident.severity,
        "status": incident.status,
        "environment": "official_synthetic_challenge_replay",
        "headline": f"{prediction.tenant}: riesgo de incidente detectado anticipadamente",
        "summary": f"{signal_detail} {decision_detail} {action_detail}",
        "signal": {
            "title": "Primera señal causal",
            "status": "observed",
            "detail": signal_detail,
            "observed_at": prediction.signal_timestamp.isoformat(),
            "top_signals": signals,
        },
        "decision": {
            "title": "Decisión de alerta",
            "status": "recommendation_ready",
            "detail": decision_detail,
            "confidence": prediction.risk_score,
            "suspected_incident_type": prediction.suspected_incident_type,
        },
        "action": {
            "title": "Acción recomendada",
            "status": action_status,
            "detail": action_detail,
            "runbook": remediation.runbook_id if remediation and remediation.runbook_id else response_plan["runbook_id"],
            "response_plan": response_plan,
            "execution_authorized": bool(remediation and remediation.runbook_id),
            "approval_required": False,
            "started_at": remediation.action_started_at.isoformat() if remediation and remediation.action_started_at else None,
            "completed_at": remediation.action_completed_at.isoformat() if remediation and remediation.action_completed_at else None,
            "environment": "controlled_infrastructure_demo",
        },
        "operator_response": operator_response,
        "verification": {
            "title": "Verificación posterior",
            "status": verification_status,
            "detail": verification_detail,
            "metrics": {},
            "contract": response_plan["verification"],
        },
        "timing": {
            "measurement_valid": False,
            "signal_to_detection_measurement_valid": True,
            "first_signal_to_detection_seconds": detection_seconds,
            "first_signal_to_action_seconds": None,
            "first_signal_to_recovery_seconds": None,
            "delivery_to_acknowledgement_seconds": operator_response[
                "delivery_to_acknowledgement_seconds"
            ],
            "clock_domains": {
                "signal_to_detection": "synthetic_replay_time",
                "delivery_to_acknowledgement": "operator_wall_clock",
                "cross_domain_duration_valid": False,
            },
        },
        "impact": {
            "label": "potential_not_observed",
            "customer_impact_observed": False,
            "customer_complaint_simulated": False,
            **kaiku_potential_impact(prediction.suspected_incident_type),
        },
        "next_step": prediction.recommended_action,
        "evidence_event_ids": [event.id for event in events],
        "limitations": [
            "Datos sintéticos oficiales reproducidos minuto a minuto; no son máquinas remotas reales.",
            "La explicación traduce evidencia persistida y no modifica el score del modelo.",
            "La acción y recuperación pertenecen a infraestructura controlada; no demuestran impacto evitado en producción.",
        ],
        "provenance": {
            "source": CHALLENGE_SOURCE,
            "model_version": prediction.model_version,
            "model_sha256": prediction.model_sha256,
            "policy_version": prediction.policy_version,
            "policy_sha256": prediction.policy_sha256,
        },
    }


def build_challenge_alert_explanation(event_id: int) -> dict[str, Any]:
    with session_factory()() as session:
        prediction = session.scalar(
            select(ChallengePredictionRecord).where(
                ChallengePredictionRecord.event_id == event_id
            )
        )
        if prediction is None:
            raise LookupError("challenge alert not found")
        incident_id = prediction.incident_id
        if incident_id is None:
            event = session.get(EventRecord, event_id)
            related_events = []
            if event is not None:
                related_events = list(
                    session.scalars(
                        select(EventRecord)
                        .where(EventRecord.process_id == event.process_id)
                        .order_by(EventRecord.id)
                    )
                )
                related_events = [
                    item
                    for item in related_events
                    if item.id == event.id
                    or (
                        item.event_type == EVENT_CHALLENGE_ALERT_ACKNOWLEDGED
                        and isinstance(item.details, dict)
                        and item.details.get("alert_event_id") == event.id
                    )
                ]
            event_ids = [item.id for item in related_events]
            operator_response = _operator_response(related_events)
            evidence = prediction.evidence
            response_plan = response_plan_for(
                prediction.alert_type,
                prediction.suspected_incident_type,
            )
            return {
                "version": "2",
                "source": "deterministic-local",
                "incident_id": None,
                "tenant": prediction.tenant,
                "run_id": prediction.run_id,
                "status": "data_quality_alert",
                "environment": "official_synthetic_challenge_replay",
                "headline": f"{prediction.tenant}: telemetría incompleta",
                "summary": (
                    "La ausencia persistente de datos impide evaluar el riesgo con "
                    "confianza; se separó de los incidentes operativos."
                ),
                "signal": {
                    "title": "Señal de calidad",
                    "status": "observed",
                    "detail": f"Calidad observada: {prediction.data_quality}.",
                    "observed_at": prediction.signal_timestamp.isoformat(),
                    "top_signals": evidence if isinstance(evidence, dict) else {},
                },
                "decision": {
                    "title": "Decisión de calidad",
                    "status": "telemetry_unreliable",
                    "detail": "Se alertó sobre la fuente sin inventar un incidente de TI.",
                },
                "action": {
                    "title": "Acción recomendada",
                    "status": "not_started",
                    "detail": prediction.recommended_action,
                    "runbook": response_plan["runbook_id"],
                    "response_plan": response_plan,
                    "execution_authorized": False,
                    "approval_required": True,
                    "started_at": None,
                    "completed_at": None,
                },
                "operator_response": operator_response,
                "verification": {
                    "title": "Verificación posterior",
                    "status": "pending",
                    "detail": "Debe confirmarse que la telemetría volvió a ser continua.",
                    "metrics": {},
                    "contract": response_plan["verification"],
                },
                "timing": {
                    "measurement_valid": False,
                    "signal_to_detection_measurement_valid": True,
                    "first_signal_to_detection_seconds": max(
                        0.0,
                        (
                            prediction.alert_timestamp
                            - prediction.signal_timestamp
                        ).total_seconds(),
                    ),
                    "first_signal_to_action_seconds": None,
                    "first_signal_to_recovery_seconds": None,
                    "delivery_to_acknowledgement_seconds": operator_response[
                        "delivery_to_acknowledgement_seconds"
                    ],
                    "clock_domains": {
                        "signal_to_detection": "synthetic_replay_time",
                        "delivery_to_acknowledgement": "operator_wall_clock",
                        "cross_domain_duration_valid": False,
                    },
                },
                "impact": {
                    "label": "not_applicable",
                    "customer_impact_observed": False,
                    "customer_complaint_simulated": False,
                    "detail": "Una interrupción de telemetría no se presenta como falla del negocio.",
                },
                "next_step": prediction.recommended_action,
                "evidence_event_ids": event_ids,
                "limitations": [
                    "Datos sintéticos oficiales; la continuidad debe verificarse en filas posteriores."
                ],
            }
    return build_challenge_incident_explanation(incident_id)
