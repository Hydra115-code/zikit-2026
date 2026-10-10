from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.response import response_plan_for


POLICY_VERSION = "challenge-alert-policy-v6"


class PolicyState(StrEnum):
    NORMAL = "NORMAL"
    OBSERVATION = "OBSERVATION"
    CANDIDATE = "CANDIDATE"
    ALERT = "ALERT"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class PolicyConfig:
    observation_threshold: float = 0.35
    candidate_threshold: float = 0.55
    alert_threshold: float = 0.70
    recovery_threshold: float = 0.25
    observation_minutes: int = 3
    candidate_minutes: int = 3
    cooldown_minutes: int = 360
    recovery_minutes: int = 3
    context_risk_multiplier: float = 0.30
    critical_40m_threshold: float = 0.976
    critical_isolation_threshold: float = 0.85
    data_quality_persistence_minutes: int = 3

    def __post_init__(self) -> None:
        thresholds = (
            self.recovery_threshold,
            self.observation_threshold,
            self.candidate_threshold,
            self.alert_threshold,
        )
        if not all(0 <= value <= 1 for value in thresholds):
            raise ValueError("policy thresholds must be between zero and one")
        if thresholds != tuple(sorted(thresholds)) or len(set(thresholds)) != 4:
            raise ValueError(
                "policy thresholds must increase from recovery to alert"
            )
        if self.observation_minutes < 1 or self.candidate_minutes < 1:
            raise ValueError("policy persistence must be positive")
        if self.cooldown_minutes < 1:
            raise ValueError("policy cooldown must be positive")
        if self.recovery_minutes < 1:
            raise ValueError("policy recovery persistence must be positive")
        if not 0 <= self.context_risk_multiplier <= 1:
            raise ValueError("context risk multiplier must be between zero and one")
        if not 0 <= self.critical_40m_threshold <= 1:
            raise ValueError("critical 40m threshold must be between zero and one")
        if not 0 <= self.critical_isolation_threshold <= 1:
            raise ValueError("critical isolation threshold must be between zero and one")
        if self.data_quality_persistence_minutes < 1:
            raise ValueError("data-quality persistence must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyConfig:
        return cls(**payload)


def save_policy_artifact(
    config: PolicyConfig,
    path: Path,
    *,
    model_version: str,
    model_sha256: str,
    calibration: dict[str, Any],
) -> str:
    payload = {
        "policy_version": POLICY_VERSION,
        "model_version": model_version,
        "model_sha256": model_sha256,
        "config": config.to_dict(),
        "calibration": calibration,
    }
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    digest = sha256(content).hexdigest()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest = digest_path.with_name(f".{digest_path.name}.tmp")
    temporary_digest.write_text(f"{digest}\n", encoding="utf-8")
    os.replace(temporary_digest, digest_path)
    return digest


def load_policy_artifact(
    path: Path,
    *,
    trusted_root: Path | None = None,
    expected_model_sha256: str | None = None,
) -> tuple[PolicyConfig, dict[str, Any]]:
    resolved = path.resolve()
    if trusted_root is not None:
        root = trusted_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("policy path is outside the trusted model directory")
    if path.suffix != ".json" or not path.is_file():
        raise ValueError("policy must be a JSON file")
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise ValueError("policy digest is missing")
    content = path.read_bytes()
    expected = digest_path.read_text(encoding="utf-8").strip()
    if sha256(content).hexdigest() != expected:
        raise ValueError("policy digest mismatch")
    payload = json.loads(content)
    if payload.get("policy_version") != POLICY_VERSION:
        raise ValueError("incompatible challenge policy")
    if expected_model_sha256 is not None and payload.get("model_sha256") != expected_model_sha256:
        raise ValueError("policy was calibrated for a different model")
    return PolicyConfig.from_dict(payload.get("config", {})), payload


@dataclass
class _Runtime:
    state: PolicyState = PolicyState.NORMAL
    observation_streak: int = 0
    candidate_streak: int = 0
    first_signal_timestamp: pd.Timestamp | None = None
    cooldown_until: pd.Timestamp | None = None
    requires_recovery: bool = False
    recovery_streak: int = 0
    outage_streak: int = 0
    outage_started_at: pd.Timestamp | None = None
    outage_alerted: bool = False
    previous_timestamp: pd.Timestamp | None = None

    def reset_incident(self) -> None:
        self.state = PolicyState.NORMAL
        self.observation_streak = 0
        self.candidate_streak = 0
        self.first_signal_timestamp = None
        self.cooldown_until = None
        self.requires_recovery = False
        self.recovery_streak = 0


ALERT_COLUMNS = (
    "run_id",
    "tenant",
    "signal_timestamp",
    "timestamp",
    "decision_latency_minutes",
    "alert_type",
    "risk_score",
    "suspected_incident_type",
    "model_version",
    "policy_state",
    "data_quality",
    "context_known_event",
    "evidence",
    "recommended_action",
)

REQUIRED_SCORE_COLUMNS = (
    "risk_score",
    "risk_40m",
    "risk_120m",
    "risk_360m",
    "risk_1440m",
    "isolation_risk",
    "baseline_risk",
    "change_risk",
    "normal_behavior_risk",
)


def _number(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _incident_type(row: pd.Series) -> str | None:
    checkout_503 = _number(row, "log_checkout_503_count_60m")
    checkout_timeout = _number(row, "log_checkout_timeout_count_180m")
    db_pool = _number(row, "log_db_pool_count_1440m")
    deadlock = _number(row, "log_deadlock_count_1440m")
    cpu_sustained = _number(row, "log_cpu_sustained_count_180m")
    disk_headroom = _number(row, "disk_headroom")
    disk_z = _number(row, "disco_past_robust_z_1440m")
    disk_trend = _number(row, "disco_trend_per_min_60m")
    latency_z = _number(row, "latencia_ms_past_robust_z_1440m")
    latency_trend = _number(row, "latencia_ms_trend_per_min_60m")
    error_z = _number(row, "errores_past_robust_z_1440m")
    cpu_z = _number(row, "cpu_past_robust_z_1440m")
    ram_z = _number(row, "ram_past_robust_z_1440m")

    if (checkout_503 or 0) > 0 or (checkout_timeout or 0) > 0:
        return "checkout_degradation"
    if (db_pool or 0) > 0:
        return "database_connection_pressure"
    if (deadlock or 0) > 0:
        return "database_contention"
    if (cpu_sustained or 0) > 0:
        return "cpu_saturation"
    if (disk_headroom is not None and disk_headroom <= 5) or (
        disk_z is not None
        and disk_z >= 3
        and disk_trend is not None
        and disk_trend > 0
    ):
        return "storage_capacity_risk"
    if (
        error_z is not None
        and error_z >= 3
        and latency_z is not None
        and latency_z >= 2
    ):
        return "application_errors"
    if (
        cpu_z is not None
        and cpu_z >= 2
        and latency_trend is not None
        and latency_trend > 0
    ):
        return "cpu_saturation"
    if (
        ram_z is not None
        and ram_z >= 2
        and latency_trend is not None
        and latency_trend > 0
    ):
        return "memory_pressure"
    return None


def _recommended_action(incident_type: str | None) -> str:
    return str(response_plan_for("incident_risk", incident_type)["summary"])


def _evidence(row: pd.Series) -> str:
    candidates: list[dict[str, Any]] = []
    log_signals = (
        ("log_checkout_503_count_60m", "log_checkout_503_minutes_since"),
        ("log_checkout_timeout_count_180m", "log_checkout_timeout_minutes_since"),
        ("log_db_pool_count_1440m", "log_db_pool_minutes_since"),
        ("log_deadlock_count_1440m", "log_deadlock_minutes_since"),
        ("log_cpu_sustained_count_180m", "log_cpu_sustained_minutes_since"),
    )
    for count_name, age_name in log_signals:
        count = _number(row, count_name)
        age = _number(row, age_name)
        if count is None or count <= 0:
            continue
        recency = 0.0 if age is None else max(0.0, 1.0 - min(age, 1440) / 1440)
        candidates.append(
            {
                "signal": count_name,
                "value": round(count, 4),
                "rank": 8.0 + recency,
            }
        )
    for metric in ("cpu", "ram", "disco", "latencia_ms", "errores"):
        robust_name = f"{metric}_past_robust_z_1440m"
        robust_value = _number(row, robust_name)
        if robust_value is not None:
            candidates.append(
                {
                    "signal": robust_name,
                    "value": round(robust_value, 4),
                    "rank": abs(robust_value),
                }
            )
    for name in (
        "risk_40m",
        "risk_120m",
        "normal_behavior_risk",
        "isolation_risk",
    ):
        value = _number(row, name)
        if value is not None:
            candidates.append(
                {"signal": name, "value": round(value, 4), "rank": value * 4}
            )
    selected = sorted(candidates, key=lambda item: item["rank"], reverse=True)[:3]
    return json.dumps(
        [{"signal": item["signal"], "value": item["value"]} for item in selected],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _alert_record(
    row: pd.Series,
    *,
    signal_timestamp: pd.Timestamp,
    alert_type: str,
    risk_score: float,
    incident_type: str | None,
    model_version: str,
    recommended_action: str,
    evidence: str,
) -> dict[str, Any]:
    timestamp = pd.Timestamp(row["timestamp"])
    return {
        "run_id": str(row["run_id"]),
        "tenant": str(row["tenant"]),
        "signal_timestamp": signal_timestamp,
        "timestamp": timestamp,
        "decision_latency_minutes": float(
            (timestamp - signal_timestamp).total_seconds() / 60
        ),
        "alert_type": alert_type,
        "risk_score": float(np.clip(risk_score, 0, 1)),
        "suspected_incident_type": incident_type,
        "model_version": model_version,
        "policy_state": PolicyState.ALERT.value,
        "data_quality": str(row["data_quality"]),
        "context_known_event": bool(row["context_known_event"]),
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def replay_policy_for_tenant(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    config: PolicyConfig,
    *,
    model_version: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    if frame["run_id"].nunique() != 1 or frame["tenant"].nunique() != 1:
        raise ValueError("policy replay requires exactly one run and tenant")
    missing_scores = sorted(set(REQUIRED_SCORE_COLUMNS) - set(scores.columns))
    if missing_scores:
        raise ValueError(f"policy scores are missing columns: {missing_scores}")
    if len(frame) != len(scores):
        raise ValueError("feature rows and score rows must have equal length")
    numeric_scores = scores.loc[:, list(REQUIRED_SCORE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric_scores.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("policy scores must be finite and between zero and one")

    combined = pd.concat(
        [
            frame.reset_index(drop=True),
            scores.loc[:, list(REQUIRED_SCORE_COLUMNS)].reset_index(drop=True),
        ],
        axis=1,
    ).sort_values("timestamp", kind="stable", ignore_index=True)
    runtime = _Runtime()
    alerts: list[dict[str, Any]] = []
    timestamps = pd.to_datetime(combined["timestamp"], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("policy replay timestamps must be unique per run and tenant")
    data_quality = combined["data_quality"].astype(str).to_numpy()
    known_events = combined["context_known_event"].astype(bool).to_numpy()
    risk_score = combined["risk_score"].to_numpy(dtype=float)
    risk_40m = combined["risk_40m"].to_numpy(dtype=float)
    isolation_risk = combined["isolation_risk"].to_numpy(dtype=float)

    for position in range(len(combined)):
        timestamp = timestamps.iloc[position]
        if runtime.previous_timestamp is not None:
            gap = (timestamp - runtime.previous_timestamp).total_seconds() / 60
            if gap > 1 and not runtime.requires_recovery:
                runtime.reset_incident()
            elif gap > 1:
                runtime.recovery_streak = 0
        runtime.previous_timestamp = timestamp

        if runtime.state == PolicyState.ALERT:
            runtime.state = PolicyState.COOLDOWN

        outage = data_quality[position] == "outage"
        if outage:
            if runtime.outage_streak == 0:
                runtime.outage_started_at = timestamp
            runtime.outage_streak += 1
            if not runtime.requires_recovery:
                runtime.state = PolicyState.NORMAL
                runtime.observation_streak = 0
                runtime.candidate_streak = 0
                runtime.first_signal_timestamp = None
            if (
                runtime.outage_streak >= config.data_quality_persistence_minutes
                and not runtime.outage_alerted
                and runtime.outage_started_at is not None
            ):
                row = combined.iloc[position]
                alerts.append(
                    _alert_record(
                        row,
                        signal_timestamp=runtime.outage_started_at,
                        alert_type="data_quality_alert",
                        risk_score=0.0,
                        incident_type=None,
                        model_version=model_version,
                        recommended_action=(
                            response_plan_for("data_quality_alert", None)["summary"]
                        ),
                        evidence=json.dumps(
                            {
                                "outage_streak_minutes": runtime.outage_streak,
                                "observed_metrics": 0,
                            },
                            separators=(",", ":"),
                        ),
                    )
                )
                runtime.outage_alerted = True
            continue

        runtime.outage_streak = 0
        runtime.outage_started_at = None
        runtime.outage_alerted = False
        raw_risk = risk_score[position]
        known_event = known_events[position]
        critical_override = (
            risk_40m[position] >= config.critical_40m_threshold
            and isolation_risk[position] >= config.critical_isolation_threshold
        )
        effective_risk = (
            raw_risk * config.context_risk_multiplier
            if known_event and not critical_override
            else raw_risk
        )

        if runtime.state == PolicyState.COOLDOWN:
            cooldown_active = (
                runtime.cooldown_until is not None
                and timestamp < runtime.cooldown_until
            )
            if effective_risk <= config.recovery_threshold:
                runtime.recovery_streak += 1
            else:
                runtime.recovery_streak = 0
            if (
                cooldown_active
                or runtime.recovery_streak < config.recovery_minutes
            ):
                continue
            runtime.reset_incident()

        if runtime.state == PolicyState.NORMAL:
            if effective_risk >= config.observation_threshold:
                runtime.state = PolicyState.OBSERVATION
                runtime.observation_streak = 1
                runtime.first_signal_timestamp = timestamp
            continue

        if runtime.state == PolicyState.OBSERVATION:
            if effective_risk < config.observation_threshold:
                runtime.reset_incident()
                continue
            runtime.observation_streak += 1
            if (
                runtime.observation_streak >= config.observation_minutes
                and effective_risk >= config.candidate_threshold
            ):
                runtime.state = PolicyState.CANDIDATE
                runtime.candidate_streak = int(
                    effective_risk >= config.alert_threshold
                )
            continue

        if runtime.state == PolicyState.CANDIDATE:
            if effective_risk < config.observation_threshold:
                runtime.reset_incident()
                continue
            if effective_risk < config.candidate_threshold:
                runtime.state = PolicyState.OBSERVATION
                runtime.observation_streak = 1
                runtime.candidate_streak = 0
                continue
            runtime.candidate_streak = (
                runtime.candidate_streak + 1
                if effective_risk >= config.alert_threshold
                else 0
            )
            if runtime.candidate_streak < config.candidate_minutes:
                continue

            signal_timestamp = runtime.first_signal_timestamp or timestamp
            row = combined.iloc[position]
            incident_type = _incident_type(row)
            alerts.append(
                _alert_record(
                    row,
                    signal_timestamp=signal_timestamp,
                    alert_type="incident_risk",
                    risk_score=raw_risk,
                    incident_type=incident_type,
                    model_version=model_version,
                    recommended_action=_recommended_action(incident_type),
                    evidence=_evidence(row),
                )
            )
            runtime.state = PolicyState.ALERT
            runtime.cooldown_until = timestamp + pd.Timedelta(
                minutes=config.cooldown_minutes
            )
            runtime.requires_recovery = True

    return pd.DataFrame(alerts, columns=ALERT_COLUMNS)


def replay_policy(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    config: PolicyConfig,
    *,
    model_version: str,
) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise ValueError("feature rows and score rows must have equal length")
    combined = frame.copy().reset_index(drop=True)
    scored = scores.copy().reset_index(drop=True)
    outputs: list[pd.DataFrame] = []
    for _, indices in combined.groupby(["run_id", "tenant"], sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        outputs.append(
            replay_policy_for_tenant(
                combined.iloc[positions].reset_index(drop=True),
                scored.iloc[positions].reset_index(drop=True),
                config,
                model_version=model_version,
            )
        )
    if not outputs:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["run_id", "tenant", "timestamp", "alert_type"],
        kind="stable",
        ignore_index=True,
    )
