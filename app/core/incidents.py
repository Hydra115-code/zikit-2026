from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.database import (
    ActionRecord,
    AlertRecord,
    EventRecord,
    ExperimentRecord,
    IncidentRecord,
    MetricRecord,
    ServiceRecord,
    session_factory,
)
from app.schemas import TelemetrySample
from app.core.slo import (
    EVENT_CUSTOMER_COMPLAINT_SIMULATED,
    EVENT_CUSTOMER_IMPACT,
    EVENT_SLO_AT_RISK,
    EVENT_SLO_BREACHED,
)


EVENT_FAULT_INJECTED = "FAULT_INJECTED"
EVENT_FIRST_SIGNAL = "FIRST_SIGNAL"
EVENT_INCIDENT_DETECTED = "INCIDENT_DETECTED"
EVENT_ACTION_STARTED = "ACTION_STARTED"
EVENT_ACTION_COMPLETED = "ACTION_COMPLETED"
EVENT_ACTION_BLOCKED = "ACTION_BLOCKED"
EVENT_RECOVERY_VERIFIED = "RECOVERY_VERIFIED"
EVENT_RECOVERY_FAILED = "RECOVERY_FAILED"
PROCESS_ID = str(uuid4())

ALERT_EVENT_PRIORITIES: dict[str, str] = {
    EVENT_FIRST_SIGNAL: "warning",
    EVENT_INCIDENT_DETECTED: "critical",
    EVENT_ACTION_BLOCKED: "critical",
    EVENT_RECOVERY_FAILED: "critical",
    EVENT_RECOVERY_VERIFIED: "resolved",
    EVENT_CUSTOMER_IMPACT: "critical",
    EVENT_CUSTOMER_COMPLAINT_SIMULATED: "critical",
}

INCIDENT_TRANSITIONS: dict[str, set[str]] = {
    "detected": {"awaiting_approval", "mitigating", "mitigation_blocked"},
    "awaiting_approval": {"mitigating", "mitigation_blocked"},
    "mitigating": {"recovered", "recovery_failed"},
    "mitigation_blocked": set(),
    "recovery_failed": set(),
    "recovered": set(),
}


class IncidentRepository:
    def __init__(self) -> None:
        self.process_id = PROCESS_ID

    async def event(
        self,
        event_type: str,
        *,
        incident_id: str | None = None,
        experiment_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> EventRecord:
        record = EventRecord(
            incident_id=incident_id,
            experiment_id=experiment_id,
            event_type=event_type,
            process_id=self.process_id,
            timestamp=datetime.now(UTC),
            monotonic_seconds=time.monotonic(),
            details=details or {},
        )
        with session_factory()() as session:
            session.add(record)
            session.flush()
            priority = ALERT_EVENT_PRIORITIES.get(event_type)
            if priority is not None:
                session.add(
                    AlertRecord(
                        event_id=record.id,
                        priority=priority,
                        created_at=record.timestamp,
                    )
                )
            session.commit()
            session.refresh(record)
        return record

    async def create_incident(
        self,
        service: str,
        severity: str,
        first_signal_at: datetime,
        evidence: dict[str, Any],
        experiment_id: str | None,
    ) -> IncidentRecord:
        record = IncidentRecord(
            id=str(uuid4()),
            service=service,
            severity=severity,
            status="detected",
            first_signal_at=first_signal_at,
            detected_at=datetime.now(UTC),
            evidence=evidence,
            experiment_id=experiment_id,
        )
        with session_factory()() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        await self._link_first_signal(record.id, service, experiment_id)
        await self.event(EVENT_INCIDENT_DETECTED, incident_id=record.id, experiment_id=experiment_id, details=evidence)
        return record

    async def _link_first_signal(
        self,
        incident_id: str,
        service: str,
        experiment_id: str | None,
    ) -> None:
        """Attach the triggering signal to its incident's auditable event chain."""
        with session_factory()() as session:
            query = select(EventRecord).where(
                EventRecord.event_type == EVENT_FIRST_SIGNAL,
                EventRecord.incident_id.is_(None),
                EventRecord.process_id == self.process_id,
            )
            if experiment_id is None:
                query = query.where(EventRecord.experiment_id.is_(None))
            else:
                query = query.where(EventRecord.experiment_id == experiment_id)
            candidates = session.scalars(
                query.order_by(EventRecord.monotonic_seconds.desc()).limit(20)
            )
            signal = next(
                (event for event in candidates if event.details.get("service") == service),
                None,
            )
            if signal is None:
                raise RuntimeError("incident has no matching first-signal event")
            signal.incident_id = incident_id
            session.commit()

    async def active_incident(
        self,
        service: str,
        *,
        since: datetime | None = None,
    ) -> IncidentRecord | None:
        with session_factory()() as session:
            query = select(IncidentRecord).where(
                IncidentRecord.service == service,
                IncidentRecord.status.in_(
                    [
                        "detected",
                        "mitigating",
                        "awaiting_approval",
                        "mitigation_blocked",
                        "recovery_failed",
                    ]
                ),
            )
            if since is not None:
                query = query.where(IncidentRecord.detected_at >= since)
            result = session.execute(query.order_by(IncidentRecord.detected_at.desc()))
            return result.scalars().first()

    async def set_status(self, incident_id: str, status: str) -> None:
        with session_factory()() as session:
            incident = session.get(IncidentRecord, incident_id)
            if incident:
                if incident.status == status:
                    return
                allowed = INCIDENT_TRANSITIONS.get(incident.status, set())
                if status not in allowed:
                    raise ValueError(f"invalid incident transition: {incident.status} -> {status}")
                incident.status = status
                if status == "recovered":
                    incident.recovered_at = datetime.now(UTC)
                session.commit()

    async def incident_status(self, incident_id: str) -> str | None:
        with session_factory()() as session:
            incident = session.get(IncidentRecord, incident_id)
            return incident.status if incident is not None else None

    async def record_sample(self, sample: TelemetrySample, *, active: bool) -> None:
        with session_factory()() as session:
            session.add(
                MetricRecord(
                    service=sample.service,
                    timestamp=sample.timestamp,
                    latency_ms=sample.latency_ms,
                    status_code=sample.status_code,
                    timed_out=sample.timed_out,
                    success=sample.success,
                )
            )
            service = session.get(ServiceRecord, sample.service)
            if service is None:
                service = ServiceRecord(name=sample.service)
                session.add(service)
            service.active = active
            service.healthy = sample.success
            service.updated_at = sample.timestamp
            session.commit()

    async def set_active_provider(self, provider: str) -> None:
        with session_factory()() as session:
            for name in ("primary", "backup"):
                service = session.get(ServiceRecord, name)
                if service is None:
                    service = ServiceRecord(name=name, healthy=True)
                    session.add(service)
                service.active = name == provider
                service.updated_at = datetime.now(UTC)
            session.commit()

    async def action_started(self, incident_id: str, runbook: str) -> int:
        with session_factory()() as session:
            action = ActionRecord(
                incident_id=incident_id,
                runbook=runbook,
                state="running",
            )
            session.add(action)
            session.commit()
            session.refresh(action)
            return action.id

    async def action_finished(self, action_id: int, state: str, details: dict[str, Any]) -> None:
        with session_factory()() as session:
            action = session.get(ActionRecord, action_id)
            if action is None:
                raise LookupError("action not found")
            action.state = state
            action.completed_at = datetime.now(UTC)
            action.details = details
            session.commit()

    async def reconcile_interrupted_state(self) -> dict[str, int]:
        """Close work that cannot safely continue after a process restart."""
        now = datetime.now(UTC)
        interrupted_incidents: list[tuple[str, str | None, str]] = []
        interrupted_experiments: list[str] = []
        action_count = 0
        with session_factory()() as session:
            for action in session.scalars(select(ActionRecord).where(ActionRecord.state == "running")):
                action.state = "interrupted"
                action.completed_at = now
                action.details = {
                    **(action.details or {}),
                    "reason": "process_restarted",
                    "recoverable": False,
                }
                action_count += 1

            for incident in session.scalars(
                select(IncidentRecord).where(
                    IncidentRecord.status.in_(["detected", "awaiting_approval", "mitigating"])
                )
            ):
                previous_status = incident.status
                incident.status = (
                    "recovery_failed" if previous_status == "mitigating" else "mitigation_blocked"
                )
                interrupted_incidents.append(
                    (incident.id, incident.experiment_id, previous_status)
                )

            for experiment in session.scalars(
                select(ExperimentRecord).where(ExperimentRecord.status == "running")
            ):
                experiment.status = "interrupted"
                experiment.completed_at = now
                interrupted_experiments.append(experiment.id)
            session.commit()

        for incident_id, experiment_id, previous_status in interrupted_incidents:
            await self.event(
                EVENT_RECOVERY_FAILED,
                incident_id=incident_id,
                experiment_id=experiment_id,
                details={
                    "reason": "process_restarted",
                    "previous_status": previous_status,
                },
            )

        for experiment_id in interrupted_experiments:
            result = await self.calculate_result(experiment_id)
            with session_factory()() as session:
                experiment = session.get(ExperimentRecord, experiment_id)
                if experiment is not None:
                    experiment.result = result
                    session.commit()

        return {
            "actions": action_count,
            "incidents": len(interrupted_incidents),
            "experiments": len(interrupted_experiments),
        }

    async def calculate_result(self, experiment_id: str) -> dict[str, Any]:
        with session_factory()() as session:
            result = session.execute(
                select(EventRecord).where(EventRecord.experiment_id == experiment_id).order_by(EventRecord.monotonic_seconds)
            )
            events = list(result.scalars())
        first_by_type: dict[str, EventRecord] = {}
        for event in events:
            first_by_type.setdefault(event.event_type, event)
        durations: dict[str, float | None] = {
            "fault_to_first_signal_seconds": None,
            "time_to_detection_seconds": None,
            "time_to_action_seconds": None,
            "first_signal_to_action_seconds": None,
            "action_duration_seconds": None,
            "first_signal_to_recovery_seconds": None,
            "first_signal_to_slo_breach_seconds": None,
            "first_signal_to_customer_impact_seconds": None,
            "first_signal_to_customer_complaint_seconds": None,
            "action_lead_before_customer_impact_seconds": None,
            "customer_impact_to_action_seconds": None,
            "recovery_lead_before_customer_complaint_seconds": None,
            "customer_impact_to_recovery_seconds": None,
            "customer_complaint_to_recovery_seconds": None,
        }
        fault = first_by_type.get(EVENT_FAULT_INJECTED)
        signal = first_by_type.get(EVENT_FIRST_SIGNAL)
        detected = first_by_type.get(EVENT_INCIDENT_DETECTED)
        started = first_by_type.get(EVENT_ACTION_STARTED)
        completed = first_by_type.get(EVENT_ACTION_COMPLETED)
        recovered = first_by_type.get(EVENT_RECOVERY_VERIFIED)
        at_risk = first_by_type.get(EVENT_SLO_AT_RISK)
        slo_breach = first_by_type.get(EVENT_SLO_BREACHED)
        customer_impact = first_by_type.get(EVENT_CUSTOMER_IMPACT)
        customer_complaint = first_by_type.get(EVENT_CUSTOMER_COMPLAINT_SIMULATED)
        if fault and signal and fault.process_id == signal.process_id:
            durations["fault_to_first_signal_seconds"] = signal.monotonic_seconds - fault.monotonic_seconds
        if fault and detected and fault.process_id == detected.process_id:
            durations["time_to_detection_seconds"] = detected.monotonic_seconds - fault.monotonic_seconds
        if detected and started and detected.process_id == started.process_id:
            durations["time_to_action_seconds"] = started.monotonic_seconds - detected.monotonic_seconds
        if signal and started and signal.process_id == started.process_id:
            durations["first_signal_to_action_seconds"] = (
                started.monotonic_seconds - signal.monotonic_seconds
            )
        if started and completed and started.process_id == completed.process_id:
            durations["action_duration_seconds"] = completed.monotonic_seconds - started.monotonic_seconds
        if signal and recovered and signal.process_id == recovered.process_id:
            durations["first_signal_to_recovery_seconds"] = recovered.monotonic_seconds - signal.monotonic_seconds
        if signal and slo_breach and signal.process_id == slo_breach.process_id:
            durations["first_signal_to_slo_breach_seconds"] = slo_breach.monotonic_seconds - signal.monotonic_seconds
        if signal and customer_impact and signal.process_id == customer_impact.process_id:
            durations["first_signal_to_customer_impact_seconds"] = customer_impact.monotonic_seconds - signal.monotonic_seconds
        if signal and customer_complaint and signal.process_id == customer_complaint.process_id:
            durations["first_signal_to_customer_complaint_seconds"] = customer_complaint.monotonic_seconds - signal.monotonic_seconds
        if started and customer_impact and started.process_id == customer_impact.process_id:
            difference = customer_impact.monotonic_seconds - started.monotonic_seconds
            if difference >= 0:
                durations["action_lead_before_customer_impact_seconds"] = difference
            else:
                durations["customer_impact_to_action_seconds"] = -difference
        if recovered and customer_complaint and recovered.process_id == customer_complaint.process_id:
            difference = customer_complaint.monotonic_seconds - recovered.monotonic_seconds
            if difference >= 0:
                durations["recovery_lead_before_customer_complaint_seconds"] = difference
        if customer_impact and recovered and customer_impact.process_id == recovered.process_id:
            durations["customer_impact_to_recovery_seconds"] = recovered.monotonic_seconds - customer_impact.monotonic_seconds
        if customer_complaint and recovered and customer_complaint.process_id == recovered.process_id:
            durations["customer_complaint_to_recovery_seconds"] = recovered.monotonic_seconds - customer_complaint.monotonic_seconds
        impact = {
            "slo_at_risk": at_risk is not None,
            "slo_breached": slo_breach is not None,
            "customer_impact_observed": customer_impact is not None,
            "customer_complaint_observed": customer_complaint is not None,
            "customer_impact_avoided": recovered is not None and customer_impact is None,
            "customer_complaint_avoided": recovered is not None and customer_complaint is None,
            "action_before_customer_impact": (
                started is not None
                and (
                    (customer_impact is None and recovered is not None)
                    or (
                        customer_impact is not None
                        and started.process_id == customer_impact.process_id
                        and started.monotonic_seconds <= customer_impact.monotonic_seconds
                    )
                )
            ),
        }
        return {
            "experiment_id": experiment_id,
            "measurement_valid": recovered is not None and durations["first_signal_to_recovery_seconds"] is not None,
            "recovered": recovered is not None,
            "durations": durations,
            "impact": impact,
            "event_count": len(events),
            "evidence": [
                {
                    "type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "process_id": event.process_id,
                    "details": event.details,
                }
                for event in events
            ],
        }
