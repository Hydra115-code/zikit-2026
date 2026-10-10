from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.challenge.mcp_evidence import KaikuEvidenceReader
from app.challenge.remediation_catalog import ControlledRunbook, RemediationCatalog
from app.challenge.remediation_mcp import MCPRemediationClient
from app.challenge.remediation_events import (
    EVENT_REMEDIATION_ACTION_COMPLETED,
    EVENT_REMEDIATION_ACTION_STARTED,
    EVENT_REMEDIATION_CANCELLED,
    EVENT_REMEDIATION_DECIDED,
    EVENT_REMEDIATION_ESCALATED,
    EVENT_REMEDIATION_RECOVERY_VERIFIED,
    EVENT_REMEDIATION_ROLLED_BACK,
    EVENT_REMEDIATION_SCHEDULED,
    EVENT_REMEDIATION_VERIFICATION_STARTED,
)
from app.config import Settings
from app.database import (
    AlertRecord,
    ChallengePredictionRecord,
    ChallengeRemediationRecord,
    EventRecord,
    IncidentRecord,
    session_factory,
)


logger = logging.getLogger(__name__)
class GeminiDecisionUnavailable(RuntimeError):
    pass


class GeminiRunbookDecision(BaseModel):
    runbook_id: str = Field(min_length=3, max_length=140)
    explanation: str = Field(min_length=3, max_length=800)


class RunbookSelector(Protocol):
    async def select(
        self,
        *,
        incident_id: str,
        evidence: dict[str, Any],
        candidates: list[ControlledRunbook],
    ) -> tuple[GeminiRunbookDecision, str]: ...


class ExecutionClient(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class GeminiRunbookSelector:
    """Let Gemini choose only among server-provided safe runbook IDs."""

    def __init__(self, settings: Settings, client_factory: Callable[..., Any] | None = None) -> None:
        self.settings = settings
        self.client_factory = client_factory

    async def select(
        self,
        *,
        incident_id: str,
        evidence: dict[str, Any],
        candidates: list[ControlledRunbook],
    ) -> tuple[GeminiRunbookDecision, str]:
        if not self.settings.gemini_api_key:
            raise GeminiDecisionUnavailable("GeminiNotConfigured")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiDecisionUnavailable("GeminiSDKUnavailable") from exc
        factory = self.client_factory or genai.Client
        client = factory(api_key=self.settings.gemini_api_key)
        prompt = json.dumps(
            {
                "task": "Selecciona exactamente un runbook seguro para la alerta crítica.",
                "incident_id": incident_id,
                "environment": "controlled_infrastructure_demo",
                "evidence_causal": evidence,
                "allowed_runbooks": [
                    {
                        "runbook_id": item.id,
                        "title": item.title,
                        "incident_types": item.incident_types,
                        "verification": [rule.model_dump() for rule in item.verification],
                    }
                    for item in candidates
                ],
                "rules": [
                    "Devuelve un ID exactamente como aparece en allowed_runbooks.",
                    "No propongas comandos, parámetros, scripts ni un runbook nuevo.",
                    "No cambies ni reinterpretas el score del detector.",
                    "Separa evidencia de hipótesis y explica la selección brevemente.",
                ],
            },
            ensure_ascii=False,
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiRunbookDecision,
            temperature=0,
        )
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.settings.gemini_timeout_seconds,
            )
        except Exception as exc:
            raise GeminiDecisionUnavailable(type(exc).__name__) from exc
        decision = response.parsed
        if not isinstance(decision, GeminiRunbookDecision):
            decision = GeminiRunbookDecision.model_validate_json(response.text or "{}")
        allowed = {item.id for item in candidates}
        if decision.runbook_id not in allowed:
            raise GeminiDecisionUnavailable("GeminiSelectedOutsideAllowlist")
        return decision, self.settings.gemini_model


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ChallengeRemediationManager:
    def __init__(
        self,
        settings: Settings,
        catalog: RemediationCatalog,
        catalog_sha256: str,
        *,
        selector: RunbookSelector | None = None,
        execution_client: ExecutionClient | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.catalog_sha256 = catalog_sha256
        self.selector = selector or GeminiRunbookSelector(settings)
        self.execution = execution_client or MCPRemediationClient(settings)
        self.evidence = KaikuEvidenceReader(settings.challenge_features_path)
        self.sleeper = sleeper
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self.settings.challenge_remediation_enabled or self._task is not None:
            return
        await self.reconcile_interrupted()
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="challenge-remediation")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.process_due_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("challenge remediation poll failed safely")
            await self.sleeper(self.settings.challenge_remediation_poll_seconds)

    async def reconcile_interrupted(self) -> int:
        active_states = {"deciding", "decision_ready", "executing", "verifying"}
        with session_factory()() as session:
            rows = list(
                session.execute(
                    select(ChallengeRemediationRecord, EventRecord.process_id)
                    .join(
                        ChallengePredictionRecord,
                        ChallengePredictionRecord.incident_id
                        == ChallengeRemediationRecord.incident_id,
                    )
                    .join(
                        EventRecord,
                        EventRecord.id == ChallengePredictionRecord.event_id,
                    )
                    .where(ChallengeRemediationRecord.status.in_(active_states))
                )
            )
        for record, process_id in rows:
            execution_exists = isinstance((record.audit or {}).get("mcp_execution"), dict)
            if execution_exists and record.runbook_id:
                try:
                    rollback = await self.execution.call(
                        "rollback_challenge_runbook",
                        {
                            "remediation_id": record.id,
                            "incident_id": record.incident_id,
                            "runbook_id": record.runbook_id,
                            "idempotency_key": f"{record.id}:restart-rollback",
                        },
                    )
                    self._transition(
                        record.id,
                        process_id,
                        "reverted",
                        EVENT_REMEDIATION_ROLLED_BACK,
                        details={
                            "failure_code": "RuntimeRestartInterrupted",
                            "mcp_result": rollback,
                        },
                        failure_code="RuntimeRestartInterrupted",
                        verification_completed=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "interrupted remediation rollback failed safely: %s",
                        type(exc).__name__,
                    )
            self._escalate(record.id, process_id, "RuntimeRestartInterrupted")
        return len(rows)

    async def process_due_once(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        cutoff = current - timedelta(seconds=self.settings.challenge_remediation_wait_seconds)
        with session_factory()() as session:
            rows = list(
                session.execute(
                    select(AlertRecord, ChallengePredictionRecord, EventRecord)
                    .join(
                        ChallengePredictionRecord,
                        ChallengePredictionRecord.event_id == AlertRecord.event_id,
                    )
                    .join(EventRecord, EventRecord.id == AlertRecord.event_id)
                    .outerjoin(
                        ChallengeRemediationRecord,
                        ChallengeRemediationRecord.incident_id
                        == ChallengePredictionRecord.incident_id,
                    )
                    .where(
                        AlertRecord.state == "open",
                        AlertRecord.priority == "critical",
                        AlertRecord.created_at <= cutoff,
                        ChallengePredictionRecord.alert_type == "incident_risk",
                        ChallengePredictionRecord.incident_id.is_not(None),
                        ChallengeRemediationRecord.id.is_(None),
                    )
                    .order_by(AlertRecord.created_at, AlertRecord.id)
                )
            )
        processed = 0
        for alert, prediction, source_event in rows:
            if await self._claim_and_run(alert, prediction, source_event, current):
                processed += 1
        return processed

    async def _claim_and_run(
        self,
        alert: AlertRecord,
        prediction: ChallengePredictionRecord,
        source_event: EventRecord,
        now: datetime,
    ) -> bool:
        assert prediction.incident_id is not None
        remediation_id = str(uuid5(NAMESPACE_URL, f"challenge-remediation|{prediction.incident_id}"))
        eligible_at = _aware(alert.created_at) + timedelta(
            seconds=self.settings.challenge_remediation_wait_seconds
        )
        record = ChallengeRemediationRecord(
            id=remediation_id,
            incident_id=prediction.incident_id,
            alert_id=alert.id,
            status="deciding",
            eligible_at=eligible_at,
            audit={
                "environment": "controlled_infrastructure_demo",
                "catalog_version": self.catalog.version,
                "catalog_sha256": self.catalog_sha256,
                "wait_seconds": self.settings.challenge_remediation_wait_seconds,
                "delivered_at": _aware(alert.created_at).isoformat(),
                "eligible_at": eligible_at.isoformat(),
            },
        )
        try:
            with session_factory()() as session:
                session.add(record)
                session.flush()
                self._event(
                    session,
                    record,
                    source_event.process_id,
                    EVENT_REMEDIATION_SCHEDULED,
                    {"wait_seconds": self.settings.challenge_remediation_wait_seconds},
                )
                session.commit()
        except IntegrityError:
            return False
        await self._run(record.id, source_event.process_id)
        return True

    async def _run(self, remediation_id: str, process_id: str) -> None:
        record, prediction, alert = self._load(remediation_id)
        if alert.state != "open":
            self._finish(record.id, process_id, "cancelled", EVENT_REMEDIATION_CANCELLED)
            return
        candidates = self.catalog.eligible(prediction.suspected_incident_type)
        if not candidates:
            self._escalate(record.id, process_id, "NoAllowlistedRunbook")
            return
        metric_window = self.evidence.metric_window(record.incident_id)
        evidence = {
            "run_id": prediction.run_id,
            "tenant": prediction.tenant,
            "alert_timestamp": prediction.alert_timestamp.isoformat(),
            "risk_score": prediction.risk_score,
            "suspected_incident_type": prediction.suspected_incident_type,
            "persisted_evidence": prediction.evidence,
            "metric_summary": metric_window["summary"],
            "future_rows_exposed": metric_window["future_rows_exposed"],
        }
        try:
            decision, provider = await self.selector.select(
                incident_id=record.incident_id,
                evidence=evidence,
                candidates=candidates,
            )
        except GeminiDecisionUnavailable as exc:
            self._escalate(record.id, process_id, str(exc), provider="gemini-unavailable")
            return
        selected = self.catalog.require(decision.runbook_id)
        self._decision(record.id, process_id, decision, provider)
        record, _, alert = self._load(record.id)
        if alert.state != "open":
            self._finish(record.id, process_id, "cancelled", EVENT_REMEDIATION_CANCELLED)
            return
        try:
            self._transition(
                record.id,
                process_id,
                "executing",
                EVENT_REMEDIATION_ACTION_STARTED,
                action_started=True,
            )
            execution = await self.execution.call(
                "execute_challenge_runbook",
                {
                    "remediation_id": record.id,
                    "incident_id": record.incident_id,
                    "runbook_id": selected.id,
                    "idempotency_key": f"{record.id}:execute",
                },
            )
            if execution.get("state") != "completed":
                raise RuntimeError("MCPExecutionNotCompleted")
            self._transition(
                record.id,
                process_id,
                "verifying",
                EVENT_REMEDIATION_ACTION_COMPLETED,
                details={"mcp_result": execution},
                action_completed=True,
            )
            self._transition(
                record.id,
                process_id,
                "verifying",
                EVENT_REMEDIATION_VERIFICATION_STARTED,
                details={
                    "window_minutes": self.settings.challenge_remediation_verification_minutes,
                    "clock_domain": "accelerated_controlled_infrastructure_time",
                },
            )
            samples = await self._verify(record.id, record.incident_id, selected)
        except Exception as exc:
            await self._rollback_or_escalate(
                record.id,
                record.incident_id,
                process_id,
                selected,
                type(exc).__name__,
            )
            return
        self._recover(record.id, process_id, samples)

    async def _verify(
        self,
        remediation_id: str,
        incident_id: str,
        runbook: ControlledRunbook,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for minute in range(1, self.settings.challenge_remediation_verification_minutes + 1):
            sample = await self.execution.call(
                "observe_challenge_controlled_state",
                {"remediation_id": remediation_id, "incident_id": incident_id},
            )
            sample["verification_minute"] = minute
            samples.append(sample)
            if minute < self.settings.challenge_remediation_verification_minutes:
                await self.sleeper(self.settings.challenge_remediation_minute_seconds)
        record, _, _ = self._load(remediation_id)
        execution = (record.audit or {}).get("mcp_execution", {})
        baseline = execution.get("baseline", {}) if isinstance(execution, dict) else {}
        for sample in samples:
            metrics = sample.get("metrics", {})
            for rule in runbook.verification:
                original = baseline.get(rule.signal)
                observed = metrics.get(rule.signal)
                if original is None or observed is None:
                    raise RuntimeError("VerificationMetricUnavailable")
                if float(original) == 0:
                    if float(observed) > 0:
                        raise RuntimeError("VerificationThresholdFailed")
                elif float(observed) / float(original) > rule.maximum_ratio:
                    raise RuntimeError("VerificationThresholdFailed")
        return samples

    async def _rollback_or_escalate(
        self,
        remediation_id: str,
        incident_id: str,
        process_id: str,
        runbook: ControlledRunbook,
        failure_code: str,
    ) -> None:
        record, _, _ = self._load(remediation_id)
        execution_exists = isinstance((record.audit or {}).get("mcp_execution"), dict)
        if execution_exists:
            try:
                rollback = await self.execution.call(
                    "rollback_challenge_runbook",
                    {
                        "remediation_id": remediation_id,
                        "incident_id": incident_id,
                        "runbook_id": runbook.id,
                        "idempotency_key": f"{remediation_id}:rollback",
                    },
                )
                self._transition(
                    remediation_id,
                    process_id,
                    "reverted",
                    EVENT_REMEDIATION_ROLLED_BACK,
                    details={"failure_code": failure_code, "mcp_result": rollback},
                    failure_code=failure_code,
                    verification_completed=True,
                )
            except Exception as rollback_error:
                failure_code = f"{failure_code}+Rollback{type(rollback_error).__name__}"
        self._escalate(remediation_id, process_id, failure_code)

    def _load(
        self, remediation_id: str
    ) -> tuple[ChallengeRemediationRecord, ChallengePredictionRecord, AlertRecord]:
        with session_factory()() as session:
            row = session.execute(
                select(
                    ChallengeRemediationRecord,
                    ChallengePredictionRecord,
                    AlertRecord,
                )
                .join(
                    ChallengePredictionRecord,
                    ChallengePredictionRecord.incident_id
                    == ChallengeRemediationRecord.incident_id,
                )
                .join(AlertRecord, AlertRecord.id == ChallengeRemediationRecord.alert_id)
                .where(ChallengeRemediationRecord.id == remediation_id)
            ).one_or_none()
        if row is None:
            raise LookupError("remediation not found")
        return row

    @staticmethod
    def _event(
        session: Any,
        record: ChallengeRemediationRecord,
        process_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(UTC)
        session.add(
            EventRecord(
                incident_id=record.incident_id,
                event_type=event_type,
                process_id=process_id,
                timestamp=now,
                monotonic_seconds=max(0.0, now.timestamp()),
                details={
                    "remediation_id": record.id,
                    "alert_id": record.alert_id,
                    "status": record.status,
                    "environment": "controlled_infrastructure_demo",
                    **(details or {}),
                },
            )
        )

    def _decision(
        self,
        remediation_id: str,
        process_id: str,
        decision: GeminiRunbookDecision,
        provider: str,
    ) -> None:
        with session_factory()() as session:
            record = session.get(ChallengeRemediationRecord, remediation_id)
            assert record is not None
            record.status = "decision_ready"
            record.decided_at = datetime.now(UTC)
            record.runbook_id = decision.runbook_id
            record.decision_provider = provider
            record.decision_reason = decision.explanation
            record.updated_at = datetime.now(UTC)
            self._event(
                session,
                record,
                process_id,
                EVENT_REMEDIATION_DECIDED,
                {
                    "runbook_id": decision.runbook_id,
                    "decision_provider": provider,
                    "decision_reason": decision.explanation,
                },
            )
            session.commit()

    def _transition(
        self,
        remediation_id: str,
        process_id: str,
        status: str,
        event_type: str,
        *,
        details: dict[str, Any] | None = None,
        failure_code: str | None = None,
        action_started: bool = False,
        action_completed: bool = False,
        verification_completed: bool = False,
    ) -> None:
        with session_factory()() as session:
            record = session.get(ChallengeRemediationRecord, remediation_id)
            assert record is not None
            now = datetime.now(UTC)
            record.status = status
            record.updated_at = now
            record.failure_code = failure_code or record.failure_code
            if action_started:
                record.action_started_at = now
            if action_completed:
                record.action_completed_at = now
            if verification_completed:
                record.verification_completed_at = now
            audit = dict(record.audit or {})
            if details:
                audit.setdefault("transitions", []).append(
                    {"event": event_type, "timestamp": now.isoformat(), **details}
                )
                record.audit = audit
            self._event(session, record, process_id, event_type, details)
            session.commit()

    def _recover(
        self,
        remediation_id: str,
        process_id: str,
        samples: list[dict[str, Any]],
    ) -> None:
        self._transition(
            remediation_id,
            process_id,
            "recovered",
            EVENT_REMEDIATION_RECOVERY_VERIFIED,
            details={
                "samples": samples,
                "window_minutes": self.settings.challenge_remediation_verification_minutes,
            },
            verification_completed=True,
        )
        with session_factory()() as session:
            record = session.get(ChallengeRemediationRecord, remediation_id)
            assert record is not None
            incident = session.get(IncidentRecord, record.incident_id)
            if incident is not None:
                incident.status = "controlled_recovery_verified"
                incident.recovered_at = datetime.now(UTC)
            session.commit()

    def _finish(
        self,
        remediation_id: str,
        process_id: str,
        status: str,
        event_type: str,
    ) -> None:
        self._transition(remediation_id, process_id, status, event_type)

    def _escalate(
        self,
        remediation_id: str,
        process_id: str,
        failure_code: str,
        *,
        provider: str | None = None,
    ) -> None:
        with session_factory()() as session:
            record = session.get(ChallengeRemediationRecord, remediation_id)
            assert record is not None
            record.status = "escalated"
            record.failure_code = failure_code
            if provider:
                record.decision_provider = provider
            record.updated_at = datetime.now(UTC)
            self._event(
                session,
                record,
                process_id,
                EVENT_REMEDIATION_ESCALATED,
                {"failure_code": failure_code, "operator_required": True},
            )
            incident = session.get(IncidentRecord, record.incident_id)
            if incident is not None:
                incident.status = "remediation_escalated"
            session.commit()

    def list_records(self) -> list[dict[str, Any]]:
        with session_factory()() as session:
            rows = list(
                session.scalars(
                    select(ChallengeRemediationRecord).order_by(
                        ChallengeRemediationRecord.created_at,
                        ChallengeRemediationRecord.id,
                    )
                )
            )
        return [serialize_remediation(row) for row in rows]

    def audit_csv(self) -> str:
        fields = [
            "remediation_id",
            "incident_id",
            "alert_id",
            "status",
            "eligible_at",
            "decision_provider",
            "runbook_id",
            "action_started_at",
            "action_completed_at",
            "verification_completed_at",
            "failure_code",
            "environment",
            "catalog_sha256",
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in self.list_records():
            writer.writerow({field: item.get(field) for field in fields})
        return buffer.getvalue()


def serialize_remediation(record: ChallengeRemediationRecord) -> dict[str, Any]:
    audit = record.audit or {}
    return {
        "remediation_id": record.id,
        "incident_id": record.incident_id,
        "alert_id": record.alert_id,
        "status": record.status,
        "eligible_at": _aware(record.eligible_at).isoformat(),
        "decision_provider": record.decision_provider,
        "decision_reason": record.decision_reason,
        "runbook_id": record.runbook_id,
        "action_started_at": (
            _aware(record.action_started_at).isoformat()
            if record.action_started_at is not None
            else None
        ),
        "action_completed_at": (
            _aware(record.action_completed_at).isoformat()
            if record.action_completed_at is not None
            else None
        ),
        "verification_completed_at": (
            _aware(record.verification_completed_at).isoformat()
            if record.verification_completed_at is not None
            else None
        ),
        "failure_code": record.failure_code,
        "environment": audit.get("environment"),
        "catalog_version": audit.get("catalog_version"),
        "catalog_sha256": audit.get("catalog_sha256"),
        "wait_seconds": audit.get("wait_seconds"),
        "audit": audit,
    }
