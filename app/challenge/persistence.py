from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.challenge.policy import POLICY_VERSION
from app.challenge.response import response_plan_for
from app.database import (
    AlertRecord,
    ChallengePredictionRecord,
    EventRecord,
    IncidentRecord,
    session_factory,
)


EVENT_CHALLENGE_FIRST_SIGNAL = "CHALLENGE_FIRST_SIGNAL"
EVENT_CHALLENGE_INCIDENT_PREDICTED = "CHALLENGE_INCIDENT_PREDICTED"
EVENT_CHALLENGE_DATA_QUALITY_ALERT = "CHALLENGE_DATA_QUALITY_ALERT"
EVENT_CHALLENGE_ALERT_ACKNOWLEDGED = "CHALLENGE_ALERT_ACKNOWLEDGED"
CHALLENGE_EVENT_TYPES = (
    EVENT_CHALLENGE_INCIDENT_PREDICTED,
    EVENT_CHALLENGE_DATA_QUALITY_ALERT,
    EVENT_CHALLENGE_ALERT_ACKNOWLEDGED,
)
CHALLENGE_SOURCE = "zikit_official_synthetic_dataset"


def _utc_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _json_evidence(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("challenge evidence is not valid JSON") from exc
    if isinstance(value, (list, dict)):
        return value
    raise ValueError("challenge evidence must be a JSON object or array")


def _incident_id(payload: dict[str, Any]) -> str:
    identity = "|".join(
        (
            CHALLENGE_SOURCE,
            payload["run_id"],
            payload["tenant"],
            payload["timestamp"].isoformat(),
            payload["alert_type"],
            payload["model_version"],
            payload["policy_version"],
        )
    )
    return str(uuid5(NAMESPACE_URL, identity))


def _process_id(run_id: str, tenant: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{CHALLENGE_SOURCE}|{run_id}|{tenant}"))


def _timeline_position(timestamp: datetime) -> float:
    return max(0.0, timestamp.timestamp())


def _aware_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _seconds_between(later: datetime, earlier: datetime) -> float:
    return max(0.0, (_aware_utc(later) - _aware_utc(earlier)).total_seconds())


def record_challenge_acknowledgement(
    session: Any,
    alert: AlertRecord,
    *,
    acknowledged_at: datetime,
    create_event: bool,
) -> dict[str, Any] | None:
    """Attach an idempotent operator acknowledgement to a challenge prediction."""
    prediction = session.scalar(
        select(ChallengePredictionRecord).where(
            ChallengePredictionRecord.event_id == alert.event_id
        )
    )
    if prediction is None:
        return None
    latency_seconds = _seconds_between(acknowledged_at, prediction.created_at)
    trace = {
        "source": CHALLENGE_SOURCE,
        "run_id": prediction.run_id,
        "tenant": prediction.tenant,
        "alert_event_id": prediction.event_id,
        "outbox_alert_id": alert.id,
        "clock_domain": "operator_wall_clock",
        "delivered_at": _aware_utc(prediction.created_at).isoformat(),
        "acknowledged_at": _aware_utc(acknowledged_at).isoformat(),
        "delivery_to_acknowledgement_seconds": latency_seconds,
        "operator_action": "alert_acknowledged",
        "mitigation_executed": False,
    }
    if create_event:
        source_event = session.get(EventRecord, prediction.event_id)
        if source_event is None:
            raise ValueError("challenge prediction source event not found")
        session.add(
            EventRecord(
                incident_id=prediction.incident_id,
                event_type=EVENT_CHALLENGE_ALERT_ACKNOWLEDGED,
                process_id=source_event.process_id,
                timestamp=acknowledged_at,
                monotonic_seconds=_timeline_position(acknowledged_at),
                details=trace,
            )
        )
        if prediction.incident_id is not None:
            incident = session.get(IncidentRecord, prediction.incident_id)
            if incident is not None:
                incident.status = "prediction_acknowledged"
    return trace


def acknowledge_alert_once(
    session: Any,
    alert_id: int,
    *,
    acknowledged_at: datetime,
) -> tuple[AlertRecord | None, dict[str, Any] | None, bool]:
    """Atomically claim an open alert and create at most one acknowledgement."""
    claimed = session.execute(
        update(AlertRecord)
        .where(AlertRecord.id == alert_id, AlertRecord.state == "open")
        .values(state="acknowledged", acknowledged_at=acknowledged_at)
    ).rowcount == 1
    session.flush()
    alert = session.get(AlertRecord, alert_id)
    if alert is None:
        return None, None, False
    session.refresh(alert)
    trace = None
    if claimed:
        trace = record_challenge_acknowledgement(
            session,
            alert,
            acknowledged_at=acknowledged_at,
            create_event=True,
        )
    elif alert.state == "acknowledged" and alert.acknowledged_at is not None:
        trace = record_challenge_acknowledgement(
            session,
            alert,
            acknowledged_at=alert.acknowledged_at,
            create_event=False,
        )
    return alert, trace, claimed


class ChallengePredictionRepository:
    """Durable, idempotent bridge from causal predictions to the operations feed."""

    @staticmethod
    def _payload(row: pd.Series) -> dict[str, Any]:
        suspected = row["suspected_incident_type"]
        return {
            "run_id": str(row["run_id"]),
            "tenant": str(row["tenant"]),
            "signal_timestamp": _utc_datetime(row["signal_timestamp"]),
            "timestamp": _utc_datetime(row["timestamp"]),
            "decision_latency_minutes": float(row["decision_latency_minutes"]),
            "alert_type": str(row["alert_type"]),
            "confidence": float(row["risk_score"]),
            "suspected_incident_type": (
                None if pd.isna(suspected) else str(suspected)
            ),
            "model_version": str(row["model_version"]),
            "policy_version": POLICY_VERSION,
            "data_quality": str(row["data_quality"]),
            "context_known_event": bool(row["context_known_event"]),
            "evidence": _json_evidence(row["evidence"]),
            "recommended_action": str(row["recommended_action"]),
        }

    @staticmethod
    def _existing(session: Any, payload: dict[str, Any]) -> ChallengePredictionRecord | None:
        return session.scalar(
            select(ChallengePredictionRecord).where(
                ChallengePredictionRecord.run_id == payload["run_id"],
                ChallengePredictionRecord.tenant == payload["tenant"],
                ChallengePredictionRecord.alert_timestamp == payload["timestamp"],
                ChallengePredictionRecord.alert_type == payload["alert_type"],
                ChallengePredictionRecord.model_version == payload["model_version"],
                ChallengePredictionRecord.policy_version == payload["policy_version"],
            )
        )

    def append(
        self,
        frame: pd.DataFrame,
        *,
        model_sha256: str,
        policy_sha256: str,
    ) -> dict[str, int]:
        created = 0
        existing = 0
        with session_factory()() as session:
            for _, row in frame.iterrows():
                payload = self._payload(row)
                if self._existing(session, payload) is not None:
                    existing += 1
                    continue
                try:
                    with session.begin_nested():
                        self._insert(
                            session,
                            payload,
                            model_sha256=model_sha256,
                            policy_sha256=policy_sha256,
                        )
                        session.flush()
                except IntegrityError:
                    existing += 1
                else:
                    created += 1
            session.commit()
            latest_id = session.scalar(
                select(func.max(EventRecord.id)).where(
                    EventRecord.event_type.in_(CHALLENGE_EVENT_TYPES)
                )
            )
        return {
            "created": created,
            "existing": existing,
            "latest_id": int(latest_id or 0),
        }

    @staticmethod
    def _insert(
        session: Any,
        payload: dict[str, Any],
        *,
        model_sha256: str,
        policy_sha256: str,
    ) -> None:
        incident_id: str | None = None
        delivered_at = datetime.now(UTC)
        process_id = _process_id(payload["run_id"], payload["tenant"])
        event_details = {
            **payload,
            "signal_timestamp": payload["signal_timestamp"].isoformat(),
            "timestamp": payload["timestamp"].isoformat(),
            "source": CHALLENGE_SOURCE,
            "model_sha256": model_sha256,
            "policy_sha256": policy_sha256,
            "action_executed": False,
        }

        if payload["alert_type"] == "incident_risk":
            incident_id = _incident_id(payload)
            severity = "critical" if payload["confidence"] >= 0.85 else "warning"
            incident = IncidentRecord(
                id=incident_id,
                service=f"challenge:{payload['tenant']}",
                severity=severity,
                status="prediction_open",
                first_signal_at=payload["signal_timestamp"],
                detected_at=payload["timestamp"],
                evidence=event_details,
                experiment_id=None,
            )
            session.add(incident)
            session.flush()
            session.add(
                EventRecord(
                    incident_id=incident_id,
                    event_type=EVENT_CHALLENGE_FIRST_SIGNAL,
                    process_id=process_id,
                    timestamp=payload["signal_timestamp"],
                    monotonic_seconds=_timeline_position(payload["signal_timestamp"]),
                    details={
                        "source": CHALLENGE_SOURCE,
                        "run_id": payload["run_id"],
                        "tenant": payload["tenant"],
                        "evidence": payload["evidence"],
                        "data_quality": payload["data_quality"],
                    },
                )
            )
            event_type = EVENT_CHALLENGE_INCIDENT_PREDICTED
            priority = severity
        elif payload["alert_type"] == "data_quality_alert":
            event_type = EVENT_CHALLENGE_DATA_QUALITY_ALERT
            priority = "warning"
        else:
            raise ValueError(f"unsupported challenge alert type: {payload['alert_type']}")

        event = EventRecord(
            incident_id=incident_id,
            event_type=event_type,
            process_id=process_id,
            timestamp=payload["timestamp"],
            monotonic_seconds=_timeline_position(payload["timestamp"]),
            details=event_details,
        )
        session.add(event)
        session.flush()
        outbox = AlertRecord(
            event_id=event.id,
            priority=priority,
            created_at=delivered_at,
        )
        session.add(outbox)
        session.flush()
        session.add(
            ChallengePredictionRecord(
                run_id=payload["run_id"],
                tenant=payload["tenant"],
                signal_timestamp=payload["signal_timestamp"],
                alert_timestamp=payload["timestamp"],
                alert_type=payload["alert_type"],
                risk_score=payload["confidence"],
                suspected_incident_type=payload["suspected_incident_type"],
                model_version=payload["model_version"],
                policy_version=payload["policy_version"],
                model_sha256=model_sha256,
                policy_sha256=policy_sha256,
                data_quality=payload["data_quality"],
                context_known_event=payload["context_known_event"],
                evidence=payload["evidence"],
                recommended_action=payload["recommended_action"],
                incident_id=incident_id,
                event_id=event.id,
                created_at=delivered_at,
            )
        )

    @staticmethod
    def _serialize(
        prediction: ChallengePredictionRecord,
        event: EventRecord,
        alert: AlertRecord,
    ) -> dict[str, Any]:
        latency = max(
            0.0,
            (prediction.alert_timestamp - prediction.signal_timestamp).total_seconds()
            / 60,
        )
        response_plan = response_plan_for(
            prediction.alert_type,
            prediction.suspected_incident_type,
        )
        return {
            "id": event.id,
            "prediction_id": prediction.id,
            "event_id": event.id,
            "outbox_alert_id": alert.id,
            "incident_id": prediction.incident_id,
            "run_id": prediction.run_id,
            "tenant": prediction.tenant,
            "signal_timestamp": prediction.signal_timestamp.isoformat(),
            "timestamp": prediction.alert_timestamp.isoformat(),
            "decision_latency_minutes": latency,
            "alert_type": prediction.alert_type,
            "confidence": prediction.risk_score,
            "suspected_incident_type": prediction.suspected_incident_type,
            "model_version": prediction.model_version,
            "policy_version": prediction.policy_version,
            "model_sha256": prediction.model_sha256,
            "policy_sha256": prediction.policy_sha256,
            "data_quality": prediction.data_quality,
            "context_known_event": prediction.context_known_event,
            "evidence": prediction.evidence,
            "recommended_action": prediction.recommended_action,
            "response_plan": response_plan,
            "runbook_id": response_plan["runbook_id"],
            "action_executed": False,
            "delivered_at": _aware_utc(prediction.created_at).isoformat(),
            "priority": alert.priority,
            "state": alert.state,
            "acknowledged_at": (
                alert.acknowledged_at.isoformat()
                if alert.acknowledged_at is not None
                else None
            ),
            "delivery_to_acknowledgement_seconds": (
                _seconds_between(alert.acknowledged_at, prediction.created_at)
                if alert.acknowledged_at is not None
                else None
            ),
            "clock_domains": {
                "signal_to_detection": "synthetic_replay_time",
                "delivery_to_acknowledgement": "operator_wall_clock",
                "cross_domain_duration_valid": False,
            },
        }

    def list(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
        run_id: str | None = None,
        tenant: str | None = None,
        alert_type: str | None = "incident_risk",
    ) -> list[dict[str, Any]]:
        query = (
            select(ChallengePredictionRecord, EventRecord, AlertRecord)
            .join(EventRecord, EventRecord.id == ChallengePredictionRecord.event_id)
            .join(AlertRecord, AlertRecord.event_id == EventRecord.id)
            .where(EventRecord.id > after_id)
        )
        if run_id is not None:
            query = query.where(ChallengePredictionRecord.run_id == run_id)
        if tenant is not None:
            query = query.where(ChallengePredictionRecord.tenant == tenant)
        if alert_type is not None:
            query = query.where(ChallengePredictionRecord.alert_type == alert_type)
        with session_factory()() as session:
            rows = session.execute(query.order_by(EventRecord.id).limit(limit))
            return [self._serialize(*row) for row in rows]

    def get_by_event_id(self, event_id: int) -> dict[str, Any] | None:
        with session_factory()() as session:
            row = session.execute(
                select(ChallengePredictionRecord, EventRecord, AlertRecord)
                .join(EventRecord, EventRecord.id == ChallengePredictionRecord.event_id)
                .join(AlertRecord, AlertRecord.event_id == EventRecord.id)
                .where(EventRecord.id == event_id)
            ).one_or_none()
            return self._serialize(*row) if row is not None else None

    def stream_updates(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with session_factory()() as session:
            events = list(
                session.scalars(
                    select(EventRecord)
                    .where(
                        EventRecord.id > after_id,
                        EventRecord.event_type.in_(CHALLENGE_EVENT_TYPES),
                    )
                    .order_by(EventRecord.id)
                    .limit(limit)
                )
            )
            updates: list[dict[str, Any]] = []
            for event in events:
                if event.event_type == EVENT_CHALLENGE_ALERT_ACKNOWLEDGED:
                    details = event.details if isinstance(event.details, dict) else {}
                    updates.append(
                        {
                            "id": event.id,
                            "event_id": event.id,
                            "outbox_alert_id": details.get("outbox_alert_id"),
                            "incident_id": event.incident_id,
                            "run_id": details.get("run_id"),
                            "tenant": details.get("tenant"),
                            "timestamp": event.timestamp.isoformat(),
                            "alert_type": "operator_acknowledgement",
                            "state": "acknowledged",
                            "action_executed": False,
                            "delivery_to_acknowledgement_seconds": details.get(
                                "delivery_to_acknowledgement_seconds"
                            ),
                            "clock_domain": "operator_wall_clock",
                            "stream_event": "challenge_alert_acknowledged",
                        }
                    )
                    continue
                row = session.execute(
                    select(ChallengePredictionRecord, EventRecord, AlertRecord)
                    .join(
                        EventRecord,
                        EventRecord.id == ChallengePredictionRecord.event_id,
                    )
                    .join(AlertRecord, AlertRecord.event_id == EventRecord.id)
                    .where(EventRecord.id == event.id)
                ).one_or_none()
                if row is None:
                    raise ValueError("challenge stream event has no durable prediction")
                updates.append(
                    {**self._serialize(*row), "stream_event": "challenge_alert"}
                )
            return updates

    def cursor(self) -> dict[str, int]:
        with session_factory()() as session:
            earliest_id, latest_id, retained_count = session.execute(
                select(
                    func.min(EventRecord.id),
                    func.max(EventRecord.id),
                    func.count(EventRecord.id),
                ).where(EventRecord.event_type.in_(CHALLENGE_EVENT_TYPES))
            ).one()
        return {
            "earliest_id": int(earliest_id or 0),
            "latest_id": int(latest_id or 0),
            "retained_count": int(retained_count or 0),
        }

    def operations_summary(self) -> dict[str, Any]:
        with session_factory()() as session:
            prediction_count = int(
                session.scalar(select(func.count(ChallengePredictionRecord.id))) or 0
            )
            rows = list(
                session.execute(
                    select(ChallengePredictionRecord, AlertRecord)
                    .join(
                        AlertRecord,
                        AlertRecord.event_id == ChallengePredictionRecord.event_id,
                    )
                    .order_by(ChallengePredictionRecord.event_id)
                )
            )
            first_signal_incident_ids = list(
                session.scalars(
                    select(EventRecord.incident_id).where(
                        EventRecord.event_type == EVENT_CHALLENGE_FIRST_SIGNAL
                    )
                )
            )
            acknowledgement_event_rows = list(
                session.scalars(
                    select(EventRecord).where(
                        EventRecord.event_type
                        == EVENT_CHALLENGE_ALERT_ACKNOWLEDGED
                    )
                )
            )

        def distribution(values: list[float]) -> dict[str, float | int | None]:
            if not values:
                return {"count": 0, "min": None, "median": None, "max": None}
            series = pd.Series(values, dtype="float64")
            return {
                "count": len(values),
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }

        detection_seconds = [
            _seconds_between(prediction.alert_timestamp, prediction.signal_timestamp)
            for prediction, _ in rows
        ]
        acknowledgement_seconds = [
            _seconds_between(alert.acknowledged_at, prediction.created_at)
            for prediction, alert in rows
            if alert.acknowledged_at is not None
        ]
        incident_risk_count = sum(
            prediction.alert_type == "incident_risk" for prediction, _ in rows
        )
        incident_ids = [
            prediction.incident_id
            for prediction, _ in rows
            if prediction.alert_type == "incident_risk"
        ]
        acknowledged_count = sum(
            alert.state == "acknowledged" for _, alert in rows
        )
        acknowledged_alert_ids = {
            alert.id for _, alert in rows if alert.state == "acknowledged"
        }
        acknowledgement_event_alert_ids = [
            event.details.get("outbox_alert_id")
            for event in acknowledgement_event_rows
            if isinstance(event.details, dict)
        ]
        delivery_timestamps_match = all(
            _aware_utc(alert.created_at) == _aware_utc(prediction.created_at)
            for prediction, alert in rows
        )
        acknowledgement_order_valid = all(
            alert.acknowledged_at is None
            or _aware_utc(alert.acknowledged_at) >= _aware_utc(prediction.created_at)
            for prediction, alert in rows
        )
        integrity = {
            "predictions_have_event_and_outbox": prediction_count == len(rows),
            "incident_predictions_have_first_signal": (
                Counter(incident_ids) == Counter(first_signal_incident_ids)
            ),
            "acknowledgements_have_single_event": (
                acknowledged_count == len(acknowledgement_event_rows)
                and len(acknowledgement_event_alert_ids)
                == len(set(acknowledgement_event_alert_ids))
                and set(acknowledgement_event_alert_ids) == acknowledged_alert_ids
            ),
            "delivery_timestamps_share_wall_clock": delivery_timestamps_match,
            "acknowledgement_not_before_delivery": acknowledgement_order_valid,
        }
        return {
            "version": "challenge-kaiku-operations-v2",
            "environment": "official_synthetic_challenge_replay",
            "alerts": {
                "total": len(rows),
                "incident_risk": incident_risk_count,
                "data_quality": sum(
                    prediction.alert_type == "data_quality_alert"
                    for prediction, _ in rows
                ),
                "open": sum(alert.state == "open" for _, alert in rows),
                "acknowledged": acknowledged_count,
            },
            "timing": {
                "signal_to_detection_seconds": {
                    "clock_domain": "synthetic_replay_time",
                    **distribution(detection_seconds),
                },
                "delivery_to_acknowledgement_seconds": {
                    "clock_domain": "operator_wall_clock",
                    **distribution(acknowledgement_seconds),
                },
                "signal_to_acknowledgement_seconds": {
                    "valid": False,
                    "value": None,
                    "reason": "synthetic replay time and operator wall clock cannot be mixed",
                },
            },
            "cursor": self.cursor(),
            "integrity": {"passed": all(integrity.values()), **integrity},
            "limitations": {
                "mitigation_executed": False,
                "recovery_verified": False,
                "remote_infrastructure_observed": False,
            },
        }
