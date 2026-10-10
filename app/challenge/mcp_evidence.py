from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.challenge.business_impact import kaiku_potential_impact
from app.challenge.explanations import build_challenge_incident_explanation
from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.training import discover_feature_files
from app.database import ChallengePredictionRecord, session_factory


METRICS = ("cpu", "ram", "disco", "latencia_ms", "errores")
CONTEXT_FLAGS = (
    "context_staffed",
    "context_critical",
    "context_known_event",
    "context_backup",
    "context_campaign",
    "context_load_event",
    "context_maintenance",
    "context_reindex",
    "context_operational_peak",
    "context_weekend",
)
LOG_SIGNAL = re.compile(r"^log_(?P<category>.+)_count_\d+m$")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp


def _json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _metric_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        available = values.dropna()
        latest = values.iloc[-1] if not values.empty else float("nan")
        result[metric] = {
            "minimum": float(available.min()) if not available.empty else None,
            "maximum": float(available.max()) if not available.empty else None,
            "mean": float(available.mean()) if not available.empty else None,
            "latest": None if pd.isna(latest) else float(latest),
        }
    return result


class KaikuEvidenceReader:
    """Read causal Kaiku evidence for one persisted prediction."""

    def __init__(self, feature_directory: Path | None) -> None:
        self.feature_directory = (
            assert_allowed_path(feature_directory)
            if feature_directory is not None
            else None
        )
        self._feature_files: dict[str, Path] | None = None
        self._manifest: dict[str, Any] | None = None

    def _prediction(self, incident_id: str) -> ChallengePredictionRecord:
        with session_factory()() as session:
            prediction = session.scalar(
                select(ChallengePredictionRecord).where(
                    ChallengePredictionRecord.incident_id == incident_id
                )
            )
        if prediction is None:
            raise LookupError("Kaiku challenge prediction not found")
        return prediction

    def _index(self) -> tuple[dict[str, Path], dict[str, Any]]:
        if self.feature_directory is None:
            raise LookupError("Kaiku feature evidence is not configured")
        if self._feature_files is None or self._manifest is None:
            files, _, manifest = discover_feature_files(self.feature_directory)
            self._feature_files = files
            self._manifest = manifest
        return self._feature_files, self._manifest

    def _frame(
        self, prediction: ChallengePredictionRecord, columns: list[str]
    ) -> pd.DataFrame:
        files, _ = self._index()
        path = files.get(prediction.run_id)
        if path is None:
            raise LookupError(f"Kaiku feature run is unavailable: {prediction.run_id}")
        requested = list(dict.fromkeys(["timestamp", "tenant", *columns]))
        frame = pd.read_parquet(path, columns=requested)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.loc[frame["tenant"].astype(str).eq(prediction.tenant)].copy()
        if frame.empty:
            raise LookupError("Kaiku feature evidence is empty")
        return frame.sort_values("timestamp", kind="stable")

    def metric_window(self, incident_id: str) -> dict[str, Any]:
        prediction = self._prediction(incident_id)
        frame = self._frame(prediction, ["data_quality", *METRICS])
        signal = _timestamp(prediction.signal_timestamp)
        alert = _timestamp(prediction.alert_timestamp)
        start = max(signal - pd.Timedelta(minutes=15), alert - pd.Timedelta(minutes=180))
        window = frame.loc[
            frame["timestamp"].between(start, alert, inclusive="both")
        ].copy()
        if window.empty:
            raise LookupError("Kaiku metric window is unavailable")
        sampled = window.iloc[::5].copy()
        if sampled.index[-1] != window.index[-1]:
            sampled = pd.concat([sampled, window.iloc[[-1]]])
        samples = [
            {
                "timestamp": row.timestamp.isoformat(),
                "data_quality": str(row.data_quality),
                **{
                    metric: _json_value(getattr(row, metric))
                    for metric in METRICS
                },
            }
            for row in sampled.itertuples(index=False)
        ]
        return {
            "tenant": prediction.tenant,
            "run_id": prediction.run_id,
            "incident_id": incident_id,
            "window_start": window["timestamp"].iloc[0].isoformat(),
            "window_end": window["timestamp"].iloc[-1].isoformat(),
            "alert_cutoff": alert.isoformat(),
            "minutes": int(
                (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0])
                .total_seconds()
                // 60
            ),
            "rows": len(window),
            "sampling": "every_5_minutes_plus_final_row",
            "summary": _metric_summary(window),
            "samples": samples,
            "future_rows_exposed": False,
            "environment": "official_synthetic_challenge_replay",
        }

    def correlated_logs(self, incident_id: str) -> dict[str, Any]:
        prediction = self._prediction(incident_id)
        _, manifest = self._index()
        if self.feature_directory is None:
            raise LookupError("Kaiku feature evidence is not configured")
        record = manifest.get("log_events")
        if not isinstance(record, dict):
            raise LookupError("Kaiku causal log evidence is not packaged")
        path = assert_allowed_path(
            self.feature_directory / str(record.get("path", ""))
        )
        if not path.is_file() or record.get("sha256") != _sha256_file(path):
            raise IngestionError("Kaiku log evidence hash mismatch")
        logs = pd.read_parquet(path)
        logs["available_timestamp"] = pd.to_datetime(logs["available_timestamp"])
        alert = _timestamp(prediction.alert_timestamp)
        start = alert - pd.Timedelta(minutes=1440)
        logs = logs.loc[
            logs["run_id"].astype(str).eq(prediction.run_id)
            & logs["tenant"].astype(str).eq(prediction.tenant)
            & logs["available_timestamp"].between(start, alert, inclusive="both")
        ].copy()
        categories: set[str] = set()
        evidence = prediction.evidence
        if isinstance(evidence, list):
            for item in evidence:
                signal = str(item.get("signal", "")) if isinstance(item, dict) else ""
                match = LOG_SIGNAL.fullmatch(signal)
                if match:
                    categories.add(match.group("category"))
        if categories:
            logs = logs.loc[logs["category"].astype(str).isin(categories)]
            selection = "categories referenced by the persisted alert evidence"
        else:
            logs = logs.loc[logs["category"].notna()]
            selection = "classified causal log indicators before the alert"
        logs = logs.sort_values("available_timestamp", kind="stable").tail(50)
        events = [
            {
                "event_timestamp": _timestamp(row.event_timestamp).isoformat(),
                "available_timestamp": _timestamp(row.available_timestamp).isoformat(),
                "source": str(row.source),
                "level": str(row.level),
                "category": None if pd.isna(row.category) else str(row.category),
                "source_line": int(row.source_line),
                "message_sha256": str(row.message_sha256),
            }
            for row in logs.itertuples(index=False)
        ]
        return {
            "tenant": prediction.tenant,
            "run_id": prediction.run_id,
            "incident_id": incident_id,
            "window_start": start.isoformat(),
            "window_end": alert.isoformat(),
            "alert_cutoff": alert.isoformat(),
            "selection": selection,
            "evidence_categories": sorted(categories),
            "events": events,
            "raw_messages_stored": False,
            "future_events_exposed": False,
            "interpretation": (
                "These are causally available indicators, not proof of root cause."
            ),
        }

    def business_context(self, incident_id: str) -> dict[str, Any]:
        prediction = self._prediction(incident_id)
        frame = self._frame(prediction, list(CONTEXT_FLAGS))
        alert = _timestamp(prediction.alert_timestamp)
        available = frame.loc[frame["timestamp"].le(alert)]
        if available.empty:
            raise LookupError("Kaiku business context is unavailable")
        row = available.iloc[-1]
        return {
            "tenant": "kaiku",
            "run_id": prediction.run_id,
            "incident_id": incident_id,
            "evaluated_at": _timestamp(row["timestamp"]).isoformat(),
            "company": {
                "business": "comercio electrónico de café y accesorios",
                "employees": 18,
                "online_store": "24/7",
                "support_hours": "lunes a sábado 09:00–19:00",
                "critical_systems": [
                    "base de datos de pedidos e inventario",
                    "catálogo, carrito y checkout",
                    "firewall y WAF básico",
                ],
                "known_risk": "checkout sin límite de conexiones por cliente",
            },
            "published_known_events": [
                "campaña Otoño de café el 9 de octubre de 12:00 a 14:00",
                "reindexado de catálogo los domingos de 04:00 a 05:30",
                "respaldo diario de base de datos de 03:00 a 03:30",
            ],
            "flags_at_alert": {
                key.removeprefix("context_"): bool(row[key])
                for key in CONTEXT_FLAGS
            },
            "potential_business_impact": kaiku_potential_impact(
                prediction.suspected_incident_type
            ),
            "future_context_exposed": False,
            "source": "public Kaiku contexto.md encoded before model inference",
        }

    def allowed_runbooks(self, incident_id: str) -> dict[str, Any]:
        explanation = build_challenge_incident_explanation(incident_id)
        plan = explanation["action"]["response_plan"]
        return {
            "tenant": "kaiku",
            "incident_id": incident_id,
            "runbooks": [plan],
            "recommended_runbook": plan["runbook_id"],
            "execution_authorized": False,
            "approval_required": True,
            "tool_is_read_only": True,
        }

    def verification(self, incident_id: str) -> dict[str, Any]:
        explanation = build_challenge_incident_explanation(incident_id)
        return {
            "tenant": "kaiku",
            "incident_id": incident_id,
            "verification": explanation["verification"],
            "operator_response": explanation["operator_response"],
            "timing": explanation["timing"],
            "mitigation_executed": False,
            "recovery_verified": False,
            "tool_is_read_only": True,
        }
