from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
import io
import json
import math
from time import monotonic
from typing import Any
from uuid import uuid4

import pandas as pd

from app.challenge.service import ChallengeReplayExecution, ChallengeReplayService
from app.challenge.submission import SUBMISSION_COLUMNS, validate_submission_frame


LIVE_REPLAY_VERSION = "challenge-kaiku-live-replay-v1"
LIVE_REPLAY_ENVIRONMENT = "official_synthetic_challenge_live_replay"
TERMINAL_STATES = frozenset({"completed", "stopped", "failed"})
ACTIVE_STATES = frozenset({"preparing", "running", "paused"})
METRIC_COLUMNS = ("cpu", "ram", "disco", "latencia_ms", "errores")
LOG_SIGNAL_COLUMNS = (
    "log_checkout_503_count_60m",
    "log_checkout_timeout_count_180m",
    "log_db_pool_count_1440m",
    "log_deadlock_count_1440m",
    "log_cpu_sustained_count_180m",
)


class LiveReplayConflict(RuntimeError):
    """Raised when a live replay transition is not valid."""


class LiveReplayNotFound(LookupError):
    """Raised when no matching live replay session exists."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _json_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def _submission_csv(alerts: pd.DataFrame) -> str:
    incidents = alerts.loc[alerts["alert_type"].eq("incident_risk")].copy()
    submission = incidents.loc[
        :, ["tenant", "timestamp", "risk_score", "suspected_incident_type"]
    ].rename(columns={"risk_score": "confidence"})
    submission = submission.loc[:, SUBMISSION_COLUMNS].reset_index(drop=True)
    validate_submission_frame(submission)
    buffer = io.StringIO()
    submission.to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.8f",
        lineterminator="\n",
    )
    return buffer.getvalue()


def _alert_payload(row: pd.Series, storage: dict[str, int]) -> dict[str, Any]:
    suspected = row["suspected_incident_type"]
    return {
        "run_id": str(row["run_id"]),
        "tenant": str(row["tenant"]),
        "signal_timestamp": _timestamp(row["signal_timestamp"]),
        "timestamp": _timestamp(row["timestamp"]),
        "signal_to_detection_minutes": float(row["decision_latency_minutes"]),
        "alert_type": str(row["alert_type"]),
        "confidence": float(row["risk_score"]),
        "confidence_semantics": "risk_score_not_calibrated_probability",
        "suspected_incident_type": (
            None if pd.isna(suspected) else str(suspected)
        ),
        "data_quality": str(row["data_quality"]),
        "context_known_event": bool(row["context_known_event"]),
        "evidence": json.loads(str(row["evidence"])),
        "recommended_action": str(row["recommended_action"]),
        "action_executed": False,
        "storage": storage,
    }


def _observation_payload(
    row: pd.Series,
    *,
    position: int,
    total_rows: int,
) -> dict[str, Any]:
    metrics = {
        name: _json_value(row.get(name))
        for name in METRIC_COLUMNS
        if name in row.index
    }
    log_signals = {
        name: _json_value(row.get(name))
        for name in LOG_SIGNAL_COLUMNS
        if name in row.index and _json_value(row.get(name)) not in {None, 0, 0.0}
    }
    return {
        "run_id": str(row["run_id"]),
        "tenant": str(row["tenant"]),
        "timestamp": _timestamp(row["timestamp"]),
        "position": position,
        "total_rows": total_rows,
        "progress": round(position / total_rows, 6),
        "metrics": metrics,
        "log_signals": log_signals,
        "data_quality": str(row["data_quality"]),
        "context_known_event": bool(row["context_known_event"]),
        "clock_domain": "synthetic_replay_time",
    }


@dataclass
class _LiveSession:
    session_id: str
    run_id: str
    speed_minutes_per_second: float
    telemetry_interval_minutes: int
    status: str = "preparing"
    rows_processed: int = 0
    total_rows: int = 0
    current_timestamp: str | None = None
    incident_alerts: int = 0
    data_quality_alerts: int = 0
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    elapsed_wall_seconds: float | None = None
    error: str | None = None
    latest_observation: dict[str, Any] | None = None
    submission_csv: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    next_sequence: int = 1


class ChallengeLiveReplayManager:
    """Run one causal Kaiku replay on an observable accelerated clock."""

    def __init__(self, service: ChallengeReplayService) -> None:
        self.service = service
        self._session: _LiveSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._play = asyncio.Event()
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zikit-live-replay",
        )

    @staticmethod
    def _snapshot(session: _LiveSession) -> dict[str, Any]:
        return {
            "version": LIVE_REPLAY_VERSION,
            "environment": LIVE_REPLAY_ENVIRONMENT,
            "session_id": session.session_id,
            "run_id": session.run_id,
            "tenant": "kaiku",
            "status": session.status,
            "speed_minutes_per_second": session.speed_minutes_per_second,
            "telemetry_interval_minutes": session.telemetry_interval_minutes,
            "rows_processed": session.rows_processed,
            "total_rows": session.total_rows,
            "progress": (
                round(session.rows_processed / session.total_rows, 6)
                if session.total_rows
                else 0.0
            ),
            "current_timestamp": session.current_timestamp,
            "alerts": {
                "incident_risk": session.incident_alerts,
                "data_quality": session.data_quality_alerts,
            },
            "started_at": session.started_at,
            "finished_at": session.finished_at,
            "elapsed_wall_seconds": session.elapsed_wall_seconds,
            "latest_sequence": session.next_sequence - 1,
            "latest_observation": session.latest_observation,
            "error": session.error,
            "clock_domains": {
                "observations_and_detection": "synthetic_replay_time",
                "session_runtime": "operator_wall_clock",
                "cross_domain_duration_valid": False,
            },
            "claims": {
                "official_ground_truth_used": False,
                "remote_infrastructure_observed": False,
                "automatic_action_executed": False,
                "model_or_policy_retrained": False,
            },
        }

    @staticmethod
    def _emit(
        session: _LiveSession,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        event = {
            "sequence": session.next_sequence,
            "event": event_type,
            "emitted_at": _utc_now(),
            "session_id": session.session_id,
            "data": data,
        }
        session.next_sequence += 1
        session.events.append(event)

    def _matching_session(self, session_id: str | None = None) -> _LiveSession:
        session = self._session
        if session is None or (
            session_id is not None and session.session_id != session_id
        ):
            raise LiveReplayNotFound("live replay session not found")
        return session

    async def available_runs(self) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._worker, self.service.available_runs)

    async def start(
        self,
        run_id: str,
        *,
        speed_minutes_per_second: float,
        telemetry_interval_minutes: int,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._session is not None and self._session.status in ACTIVE_STATES:
                raise LiveReplayConflict("a live replay session is already active")
            session = _LiveSession(
                session_id=str(uuid4()),
                run_id=run_id,
                speed_minutes_per_second=speed_minutes_per_second,
                telemetry_interval_minutes=telemetry_interval_minutes,
            )
            self._session = session
            self._play.set()
            self._emit(
                session,
                "live_replay_started",
                {
                    "run_id": run_id,
                    "tenant": "kaiku",
                    "status": "preparing",
                    "dataset": "official_synthetic",
                },
            )
            self._task = asyncio.create_task(self._run(session))
            return self._snapshot(session)

    async def _run(self, session: _LiveSession) -> None:
        started = monotonic()
        try:
            loop = asyncio.get_running_loop()
            execution = await loop.run_in_executor(
                self._worker,
                self.service.prepare,
                session.run_id,
            )
            if session.status == "stopped":
                return
            timeline = execution.timeline
            if timeline is None or timeline.empty:
                raise ValueError("live replay execution has no causal timeline")
            timeline = timeline.sort_values("timestamp", kind="stable").reset_index(
                drop=True
            )
            alerts = execution.alerts.sort_values(
                ["timestamp", "alert_type"], kind="stable"
            ).reset_index(drop=True)
            alerts_by_timestamp: dict[pd.Timestamp, list[pd.Series]] = {}
            for _, alert in alerts.iterrows():
                key = pd.Timestamp(alert["timestamp"])
                alerts_by_timestamp.setdefault(key, []).append(alert)

            session.total_rows = len(timeline)
            session.status = "running"
            self._emit(
                session,
                "live_replay_ready",
                {
                    "rows": session.total_rows,
                    "first_timestamp": _timestamp(timeline.iloc[0]["timestamp"]),
                    "last_timestamp": _timestamp(timeline.iloc[-1]["timestamp"]),
                    "causal_features": True,
                    "model_or_policy_retrained": False,
                },
            )

            delay = 1.0 / session.speed_minutes_per_second
            for index, row in timeline.iterrows():
                await self._play.wait()
                if session.status == "stopped":
                    break
                if delay:
                    await asyncio.sleep(delay)
                position = index + 1
                timestamp = pd.Timestamp(row["timestamp"])
                observation = _observation_payload(
                    row,
                    position=position,
                    total_rows=session.total_rows,
                )
                session.rows_processed = position
                session.current_timestamp = observation["timestamp"]
                session.latest_observation = observation

                alerts_now = alerts_by_timestamp.get(timestamp, [])
                if (
                    position == 1
                    or position == session.total_rows
                    or position % session.telemetry_interval_minutes == 0
                    or alerts_now
                    or observation["data_quality"] not in {"ok", "complete"}
                ):
                    self._emit(session, "live_replay_observation", observation)

                for alert in alerts_now:
                    alert_frame = pd.DataFrame([alert], columns=alerts.columns)
                    storage = self.service.store.append(
                        alert_frame,
                        model_sha256=execution.model_sha256,
                        policy_sha256=execution.policy_sha256,
                    )
                    alert_type = str(alert["alert_type"])
                    if alert_type == "incident_risk":
                        session.incident_alerts += 1
                    else:
                        session.data_quality_alerts += 1
                    self._emit(
                        session,
                        "live_replay_alert",
                        _alert_payload(alert, storage),
                    )

            if session.status != "stopped":
                session.status = "completed"
                session.submission_csv = _submission_csv(alerts)
                self._emit(
                    session,
                    "live_replay_completed",
                    {
                        "rows_processed": session.rows_processed,
                        "incident_alerts": session.incident_alerts,
                        "data_quality_alerts": session.data_quality_alerts,
                        "submission_ready": True,
                        "action_executed": False,
                    },
                )
        except asyncio.CancelledError:
            if session.status != "stopped":
                session.status = "stopped"
                self._emit(
                    session,
                    "live_replay_stopped",
                    {"rows_processed": session.rows_processed},
                )
            raise
        except Exception as exc:  # pragma: no cover - exact library errors vary
            session.status = "failed"
            session.error = str(exc)
            self._emit(session, "live_replay_failed", {"error": str(exc)})
        finally:
            session.finished_at = _utc_now()
            session.elapsed_wall_seconds = round(monotonic() - started, 6)

    async def pause(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            session = self._matching_session(session_id)
            if session.status != "running":
                raise LiveReplayConflict("only a running live replay can be paused")
            session.status = "paused"
            self._play.clear()
            self._emit(
                session,
                "live_replay_paused",
                {"rows_processed": session.rows_processed},
            )
            return self._snapshot(session)

    async def resume(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            session = self._matching_session(session_id)
            if session.status != "paused":
                raise LiveReplayConflict("only a paused live replay can be resumed")
            session.status = "running"
            self._play.set()
            self._emit(
                session,
                "live_replay_resumed",
                {"rows_processed": session.rows_processed},
            )
            return self._snapshot(session)

    async def stop(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            session = self._matching_session(session_id)
            if session.status not in ACTIVE_STATES:
                raise LiveReplayConflict("live replay is not active")
            session.status = "stopped"
            session.finished_at = _utc_now()
            self._play.set()
            self._emit(
                session,
                "live_replay_stopped",
                {"rows_processed": session.rows_processed},
            )
            return self._snapshot(session)

    def status(self, session_id: str | None = None) -> dict[str, Any]:
        return self._snapshot(self._matching_session(session_id))

    def events_after(
        self,
        session_id: str,
        *,
        after_sequence: int,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        session = self._matching_session(session_id)
        return [
            event
            for event in session.events
            if int(event["sequence"]) > after_sequence
        ][:limit]

    def submission(self, session_id: str) -> str:
        session = self._matching_session(session_id)
        if session.status != "completed" or session.submission_csv is None:
            raise LiveReplayConflict("submission is available only after completion")
        return session.submission_csv

    async def shutdown(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._worker.shutdown(wait=True, cancel_futures=True)


def format_live_replay_sse(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return (
        f"id: {event['sequence']}\n"
        f"event: {event['event']}\n"
        f"data: {payload}\n\n"
    )
