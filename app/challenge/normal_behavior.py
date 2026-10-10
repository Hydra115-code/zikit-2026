from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from app.challenge.business_context import context_features
from app.challenge.contracts import METRIC_COLUMNS, TARGET_TENANT


NORMAL_BEHAVIOR_VERSION = "challenge-kaiku-normal-behavior-v1"
NORMAL_TIME_BUCKET_MINUTES = 30

_CONTEXT_COLUMNS = (
    "context_known_event",
    "context_backup",
    "context_campaign",
    "context_load_event",
    "context_maintenance",
    "context_reindex",
    "context_operational_peak",
    "context_weekend",
    "context_staffed",
)

_MINIMUM_SCALES = {
    "cpu": 1.0,
    "ram": 1.0,
    "disco": 0.25,
    "latencia_ms": 5.0,
    "errores": 0.5,
}


def _context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise ValueError("normal behavior input is missing timestamp")
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    missing = [name for name in _CONTEXT_COLUMNS if name not in frame.columns]
    if not missing:
        return frame.loc[:, _CONTEXT_COLUMNS].astype(bool)
    derived = context_features(TARGET_TENANT, timestamp)
    result = pd.DataFrame(index=frame.index)
    for name in _CONTEXT_COLUMNS:
        source = frame[name] if name in frame.columns else derived[name]
        result[name] = source.astype(bool)
    return result


def _event_kind(context: pd.DataFrame) -> np.ndarray:
    kind = np.full(len(context), "ordinary", dtype=object)
    known = context["context_known_event"].to_numpy(dtype=bool)
    kind[known] = "known_other"
    for column, label in (
        ("context_operational_peak", "operational_peak"),
        ("context_maintenance", "maintenance"),
        ("context_load_event", "load_event"),
        ("context_reindex", "reindex"),
        ("context_backup", "backup"),
        ("context_campaign", "campaign"),
    ):
        kind[context[column].to_numpy(dtype=bool)] = label
    return kind


def _profile_keys(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    context = _context_frame(frame)
    bucket = (
        (timestamp.dt.hour * 60 + timestamp.dt.minute)
        // NORMAL_TIME_BUCKET_MINUTES
    ).to_numpy(dtype=int)
    weekend = context["context_weekend"].to_numpy(dtype=np.int8)
    staffed = context["context_staffed"].to_numpy(dtype=np.int8)
    event_kind = _event_kind(context)
    exact = np.asarray(
        [
            f"{kind}|{int(is_weekend)}|{int(is_staffed)}|{int(time_bucket):02d}"
            for kind, is_weekend, is_staffed, time_bucket in zip(
                event_kind, weekend, staffed, bucket, strict=True
            )
        ],
        dtype=object,
    )
    return exact, event_kind


def _robust_stats(
    frame: pd.DataFrame,
    global_scales: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
        if not values.size:
            continue
        median = float(np.median(values))
        mad_scale = float(1.4826 * np.median(np.abs(values - median)))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)
        inherited = 0.0 if global_scales is None else global_scales.get(metric, 0.0) * 0.10
        scale = max(
            mad_scale,
            iqr_scale * 0.25,
            inherited,
            _MINIMUM_SCALES[metric],
        )
        stats[metric] = (median, float(scale))
    return stats


@dataclass(frozen=True)
class NormalBehaviorModel:
    exact_profiles: dict[str, dict[str, tuple[float, float]]]
    event_profiles: dict[str, dict[str, tuple[float, float]]]
    global_profile: dict[str, tuple[float, float]]
    profile_counts: dict[str, int]
    training_rows: int
    training_runs: int
    version: str = NORMAL_BEHAVIOR_VERSION
    bucket_minutes: int = NORMAL_TIME_BUCKET_MINUTES

    def raw_deviation(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [metric for metric in METRIC_COLUMNS if metric not in frame.columns]
        if missing:
            raise ValueError(f"normal behavior input is missing metrics: {missing}")
        tenants = set(frame["tenant"].astype(str).unique())
        if tenants != {TARGET_TENANT}:
            raise ValueError(f"normal behavior input must contain only Kaiku: {sorted(tenants)}")

        exact_keys, event_kinds = _profile_keys(frame)
        metric_values = {
            metric: pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
            for metric in METRIC_COLUMNS
        }
        deviations = np.zeros((len(frame), len(METRIC_COLUMNS)), dtype="float32")
        for key in np.unique(exact_keys):
            positions = np.flatnonzero(exact_keys == key)
            event_kind = str(event_kinds[positions[0]])
            profile = self.exact_profiles.get(str(key))
            if profile is None:
                profile = self.event_profiles.get(event_kind, self.global_profile)
            for metric_index, metric in enumerate(METRIC_COLUMNS):
                median, scale = profile.get(metric, self.global_profile[metric])
                values = metric_values[metric][positions]
                valid = np.isfinite(values)
                deviations[positions[valid], metric_index] = np.maximum(
                    (values[valid] - median) / max(scale, 1e-9),
                    0.0,
                )
        ranked = np.sort(deviations, axis=1)
        return (ranked[:, -1] + 0.35 * ranked[:, -2]).astype("float32")


def fit_normal_behavior_model(
    frames: Iterable[pd.DataFrame],
    *,
    training_runs: int,
) -> NormalBehaviorModel:
    selected: list[pd.DataFrame] = []
    required = ["tenant", "timestamp", "data_quality", *METRIC_COLUMNS]
    for frame in frames:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"normal behavior training input is missing columns: {missing}")
        usable = frame.loc[frame["data_quality"].ne("outage")].copy()
        if not usable.empty:
            selected.append(usable)
    if not selected:
        raise ValueError("normal behavior training has no usable rows")
    training = pd.concat(selected, ignore_index=True)
    tenants = set(training["tenant"].astype(str).unique())
    if tenants != {TARGET_TENANT}:
        raise ValueError(f"normal behavior training must contain only Kaiku: {sorted(tenants)}")

    exact_keys, event_kinds = _profile_keys(training)
    training["_normal_exact_key"] = exact_keys
    training["_normal_event_kind"] = event_kinds
    global_profile = _robust_stats(training)
    if set(global_profile) != set(METRIC_COLUMNS):
        raise ValueError("normal behavior training has incomplete metric coverage")
    global_scales = {metric: stats[1] for metric, stats in global_profile.items()}

    exact_profiles = {
        str(key): _robust_stats(group, global_scales)
        for key, group in training.groupby("_normal_exact_key", sort=True)
    }
    event_profiles = {
        str(key): _robust_stats(group, global_scales)
        for key, group in training.groupby("_normal_event_kind", sort=True)
    }
    profile_counts = {
        str(key): int(count)
        for key, count in training["_normal_event_kind"].value_counts().sort_index().items()
    }
    return NormalBehaviorModel(
        exact_profiles=exact_profiles,
        event_profiles=event_profiles,
        global_profile=global_profile,
        profile_counts=profile_counts,
        training_rows=int(len(training)),
        training_runs=int(training_runs),
    )
