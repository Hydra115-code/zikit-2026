from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.business_context import CONTEXT_SCHEMA_VERSION, context_features
from app.challenge.contracts import (
    CHALLENGE_SCHEMA_VERSION,
    CLEAN_COLUMNS,
    METRIC_COLUMNS,
    TARGET_TENANT,
)
from app.challenge.ingestion import IngestionError, assert_allowed_path


FEATURE_SCHEMA_VERSION = "challenge-kaiku-features-v2"
ROLLING_WINDOWS_MINUTES = (5, 15, 30, 60, 180, 720, 1440)
TREND_WINDOWS_MINUTES = (15, 60, 180)
LAG_MINUTES = (5, 15, 60)
EWMA_SPANS = (15, 60)
PERCENTILE_METRICS = ("latencia_ms", "errores")
EPSILON = 1e-6

IDENTITY_OUTPUT_COLUMNS = (
    "feature_version",
    "run_id",
    "tenant",
    "timestamp",
    "source_name",
    "source_line",
    "data_quality",
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except ImportError as exc:
        temporary.unlink(missing_ok=True)
        raise IngestionError("pyarrow is required; install the project with the lab extra") from exc
    os.replace(temporary, path)


def _clean_input_path(input_path: Path) -> tuple[Path, dict[str, Any] | None]:
    resolved = assert_allowed_path(input_path)
    if resolved.is_dir():
        metrics_path = assert_allowed_path(resolved / "metrics.clean.parquet")
        manifest_path = resolved / "manifest.json"
    else:
        metrics_path = resolved
        manifest_path = resolved.parent / "manifest.json"
    if metrics_path.name != "metrics.clean.parquet" or not metrics_path.is_file():
        raise IngestionError("phase 2 requires metrics.clean.parquet from phase 1")

    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CHALLENGE_SCHEMA_VERSION:
            raise IngestionError("incompatible phase-1 manifest schema")
        expected = manifest.get("outputs", {}).get(metrics_path.name, {}).get("sha256")
        if expected and expected != _sha256_file(metrics_path):
            raise IngestionError("phase-1 metrics digest mismatch")
    return metrics_path, manifest


def load_clean_metrics(input_path: Path) -> tuple[pd.DataFrame, Path, dict[str, Any] | None]:
    metrics_path, manifest = _clean_input_path(input_path)
    frame = pd.read_parquet(metrics_path)
    missing = sorted(set(CLEAN_COLUMNS) - set(frame.columns))
    if missing:
        raise IngestionError(f"clean metrics are missing columns: {missing}")
    frame = frame.loc[:, CLEAN_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    tenants = set(frame["tenant"].astype(str).unique())
    if tenants != {TARGET_TENANT}:
        raise IngestionError(
            f"clean metrics must contain only {TARGET_TENANT}: {sorted(tenants)}"
        )
    if manifest is not None:
        scope = manifest.get("scope", {})
        if scope.get("target_tenant") != TARGET_TENANT:
            raise IngestionError("phase-1 manifest is not scoped to Kaiku")
    if frame.duplicated(["run_id", "tenant", "timestamp"]).any():
        raise IngestionError("clean metrics contain duplicate identities")
    return frame, metrics_path, manifest


def _minimum_observations(window: int) -> int:
    return max(3, int(np.ceil(window * 0.20)))


def _valid_shift(series: pd.Series, timestamp: pd.Series, minutes: int) -> pd.Series:
    shifted = series.shift(minutes)
    elapsed = timestamp.diff(minutes).dt.total_seconds().div(60)
    return shifted.where(np.isclose(elapsed, minutes))


def _outage_streak(unavailable: pd.Series) -> pd.Series:
    groups = unavailable.ne(unavailable.shift(fill_value=False)).cumsum()
    streak = unavailable.groupby(groups).cumcount().add(1)
    return streak.where(unavailable, 0).astype("int32")


def _time_of_day_baseline(series: pd.Series, timestamp: pd.Series) -> pd.Series:
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    baseline = series.groupby(minute_of_day, sort=False).transform(
        lambda values: values.expanding(min_periods=1).median().shift(1)
    )
    return baseline.astype(float)


def build_features_for_tenant(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features for one run/tenant using the current and past rows only."""
    if frame.empty:
        raise IngestionError("cannot build features from an empty tenant run")
    if frame["run_id"].nunique() != 1 or frame["tenant"].nunique() != 1:
        raise IngestionError("feature input must contain exactly one run and tenant")

    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    timestamp = pd.to_datetime(ordered["timestamp"], errors="raise")
    if timestamp.duplicated().any():
        raise IngestionError("tenant run contains duplicate timestamps")
    tenant = str(ordered.at[0, "tenant"])
    if tenant != TARGET_TENANT:
        raise IngestionError("feature construction is scoped exclusively to Kaiku")

    identity_and_raw = ordered.loc[
        :,
        [
            "run_id",
            "tenant",
            "timestamp",
            "source_name",
            "source_line",
            "data_quality",
            *METRIC_COLUMNS,
            *(f"missing_{metric}" for metric in METRIC_COLUMNS),
            *(f"invalid_{metric}" for metric in METRIC_COLUMNS),
        ],
    ].copy()
    identity_and_raw.insert(0, "feature_version", FEATURE_SCHEMA_VERSION)
    feature_values: dict[str, np.ndarray] = {}

    metrics = ordered.loc[:, METRIC_COLUMNS].astype(float)
    indexed_metrics = metrics.set_index(pd.DatetimeIndex(timestamp))
    elapsed_from_start = timestamp.sub(timestamp.iloc[0]).dt.total_seconds().div(60).add(1)
    row_present = pd.Series(1.0, index=indexed_metrics.index)
    observed_metric_count = indexed_metrics.notna().sum(axis=1).astype(float)

    for window in ROLLING_WINDOWS_MINUTES:
        interval = f"{window}min"
        minimum = _minimum_observations(window)
        rolling = indexed_metrics.rolling(interval, min_periods=minimum, closed="right")
        means = rolling.mean()
        maxima = rolling.max()
        for metric in METRIC_COLUMNS:
            feature_values[f"{metric}_mean_{window}m"] = means[metric].to_numpy(
                dtype="float32"
            )
            feature_values[f"{metric}_max_{window}m"] = maxima[metric].to_numpy(
                dtype="float32"
            )
        for metric in PERCENTILE_METRICS:
            feature_values[f"{metric}_p95_{window}m"] = (
                indexed_metrics[metric]
                .rolling(interval, min_periods=minimum, closed="right")
                .quantile(0.95)
                .to_numpy(dtype="float32")
            )

        expected_rows = np.minimum(window, elapsed_from_start.to_numpy(dtype=float))
        present_rows = (
            row_present.rolling(interval, min_periods=1, closed="right").sum().to_numpy()
        )
        observed_values = (
            observed_metric_count.rolling(interval, min_periods=1, closed="right")
            .sum()
            .to_numpy()
        )
        feature_values[f"row_coverage_{window}m"] = (
            present_rows / expected_rows
        ).astype("float32")
        feature_values[f"telemetry_coverage_{window}m"] = (
            observed_values / (expected_rows * len(METRIC_COLUMNS))
        ).astype("float32")

    for minutes in LAG_MINUTES:
        for metric in METRIC_COLUMNS:
            lagged = _valid_shift(metrics[metric], timestamp, minutes)
            delta = metrics[metric] - lagged
            relative = delta / lagged.abs().clip(lower=EPSILON)
            feature_values[f"{metric}_delta_{minutes}m"] = delta.to_numpy(
                dtype="float32"
            )
            feature_values[f"{metric}_relative_change_{minutes}m"] = relative.to_numpy(
                dtype="float32"
            )

    for window in TREND_WINDOWS_MINUTES:
        for metric in METRIC_COLUMNS:
            lagged = _valid_shift(metrics[metric], timestamp, window - 1)
            trend = (metrics[metric] - lagged) / (window - 1)
            feature_values[f"{metric}_trend_per_min_{window}m"] = trend.to_numpy(
                dtype="float32"
            )
            if metric in {"cpu", "latencia_ms", "errores"}:
                previous_trend = _valid_shift(trend, timestamp, window)
                acceleration = (trend - previous_trend) / window
                feature_values[f"{metric}_acceleration_{window}m"] = (
                    acceleration.to_numpy(dtype="float32")
                )

    for span in EWMA_SPANS:
        for metric in METRIC_COLUMNS:
            feature_values[f"{metric}_ewma_{span}m"] = (
                metrics[metric]
                .ewm(span=span, adjust=False, ignore_na=True, min_periods=3)
                .mean()
                .to_numpy(dtype="float32")
            )

    for metric in METRIC_COLUMNS:
        series = indexed_metrics[metric]
        baseline = series.rolling("1440min", min_periods=60, closed="left").median()
        deviation = (series - baseline).abs()
        mad = deviation.rolling("1440min", min_periods=60, closed="left").median()
        robust_z = (series - baseline) / (1.4826 * mad.clip(lower=EPSILON))
        tod_baseline = _time_of_day_baseline(metrics[metric], timestamp)
        tod_residual = (metrics[metric] - tod_baseline) / tod_baseline.abs().clip(lower=EPSILON)
        feature_values[f"{metric}_past_median_1440m"] = baseline.to_numpy(
            dtype="float32"
        )
        feature_values[f"{metric}_past_robust_z_1440m"] = robust_z.to_numpy(
            dtype="float32"
        )
        feature_values[f"{metric}_past_time_of_day_median"] = tod_baseline.to_numpy(
            dtype="float32"
        )
        feature_values[f"{metric}_past_time_of_day_residual"] = tod_residual.to_numpy(
            dtype="float32"
        )

    unavailable = metrics.isna().all(axis=1)
    last_observed_at = timestamp.where(~unavailable).ffill()
    feature_values["outage_streak_rows"] = _outage_streak(unavailable).to_numpy(
        dtype="int32"
    )
    feature_values["minutes_since_last_observation"] = (
        timestamp.sub(last_observed_at)
        .dt.total_seconds()
        .div(60)
        .to_numpy(dtype="float32")
    )
    feature_values["gap_before_minutes"] = (
        timestamp.diff()
        .dt.total_seconds()
        .div(60)
        .sub(1)
        .clip(lower=0)
        .fillna(0)
        .to_numpy(dtype="float32")
    )

    feature_values["resource_pressure"] = metrics[["cpu", "ram", "disco"]].max(
        axis=1
    ).to_numpy(dtype="float32")
    feature_values["disk_headroom"] = (100.0 - metrics["disco"]).to_numpy(
        dtype="float32"
    )
    feature_values["cpu_latency_pressure"] = (
        metrics["cpu"] * metrics["latencia_ms"] / 100.0
    ).to_numpy(dtype="float32")
    feature_values["ram_latency_pressure"] = (
        metrics["ram"] * metrics["latencia_ms"] / 100.0
    ).to_numpy(dtype="float32")
    feature_values["latency_error_pressure"] = (
        np.log1p(metrics["latencia_ms"]) * np.log1p(metrics["errores"])
    ).to_numpy(dtype="float32")

    context = context_features(tenant, timestamp)
    for column in context.columns:
        feature_values[column] = context[column].to_numpy()

    features = pd.DataFrame(feature_values, index=identity_and_raw.index)
    output = pd.concat([identity_and_raw, features], axis=1)
    numeric_columns = output.select_dtypes(include=["float64"]).columns
    output = output.astype({column: "float32" for column in numeric_columns})
    return output


def build_feature_dataset(
    input_path: Path,
    output_directory: Path,
    *,
    expected_runs: int | None = None,
    require_all_tenants: bool = False,
) -> dict[str, Any]:
    clean, metrics_path, phase_one_manifest = load_clean_metrics(input_path)
    run_ids = sorted(str(value) for value in clean["run_id"].unique())
    if expected_runs is not None and len(run_ids) != expected_runs:
        raise IngestionError(f"expected {expected_runs} runs, found {len(run_ids)}")
    if require_all_tenants and not clean["tenant"].eq(TARGET_TENANT).all():
        raise IngestionError("phase 2 accepts only complete Kaiku runs")

    output = output_directory.expanduser().resolve(strict=False)
    features_directory = output / "features"
    feature_files: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None
    total_rows = 0

    for run_id in run_ids:
        run = clean.loc[clean["run_id"].astype(str).eq(run_id)].copy()
        tenant_frames = [
            build_features_for_tenant(group)
            for _, group in run.groupby("tenant", sort=True)
        ]
        featured = pd.concat(tenant_frames, ignore_index=True)
        featured = featured.sort_values(["tenant", "timestamp"], kind="stable").reset_index(
            drop=True
        )
        if len(featured) != len(run):
            raise IngestionError(f"feature row conservation failed for {run_id}")
        current_features = [
            column
            for column in featured.columns
            if column not in IDENTITY_OUTPUT_COLUMNS
        ]
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise IngestionError(f"feature schema drift detected in {run_id}")

        path = features_directory / f"{run_id}.parquet"
        _atomic_write_parquet(featured, path)
        digest = _sha256_file(path)
        feature_files.append(
            {"run_id": run_id, "path": f"features/{path.name}", "sha256": digest, "rows": len(featured)}
        )
        run_reports.append(
            {
                "run_id": run_id,
                "rows": len(featured),
                "tenants": {TARGET_TENANT: int(len(featured))},
                "outage_rows": int(featured["data_quality"].eq("outage").sum()),
                "partial_rows": int(featured["data_quality"].eq("partial").sum()),
            }
        )
        total_rows += len(featured)

    assert feature_columns is not None
    report = {
        "feature_version": FEATURE_SCHEMA_VERSION,
        "context_version": CONTEXT_SCHEMA_VERSION,
        "source_schema_version": CHALLENGE_SCHEMA_VERSION,
        "scope": {"target_tenant": TARGET_TENANT, "kaiku_only": True},
        "run_count": len(run_ids),
        "tenant_count": int(clean["tenant"].nunique()),
        "rows_input": int(len(clean)),
        "rows_output": int(total_rows),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "rolling_windows_minutes": list(ROLLING_WINDOWS_MINUTES),
        "trend_windows_minutes": list(TREND_WINDOWS_MINUTES),
        "lag_minutes": list(LAG_MINUTES),
        "ewma_spans": list(EWMA_SPANS),
        "causality": {
            "current_sample_allowed": True,
            "future_rows_used": False,
            "past_baselines_exclude_current_sample": True,
            "absolute_timestamp_feature": False,
        },
        "logs": {
            "included": False,
            "reason": (
                "public logs are traceable per run but remain excluded from the "
                "phase-2 metrics feature contract"
            ),
        },
        "integrity": {
            "target_tenant": TARGET_TENANT,
            "kaiku_only": True,
            "row_conservation": total_rows == len(clean),
            "identity_unique": not clean.duplicated(["run_id", "tenant", "timestamp"]).any(),
        },
        "per_run": run_reports,
    }
    report_path = output / "feature-report.json"
    _atomic_write_json(report, report_path)
    manifest = {
        "feature_version": FEATURE_SCHEMA_VERSION,
        "context_version": CONTEXT_SCHEMA_VERSION,
        "scope": {"target_tenant": TARGET_TENANT, "kaiku_only": True},
        "source": {
            "path": metrics_path.name,
            "sha256": _sha256_file(metrics_path),
            "phase_one_manifest_verified": phase_one_manifest is not None,
        },
        "feature_files": feature_files,
        "report": {
            "path": report_path.name,
            "sha256": _sha256_file(report_path),
        },
    }
    manifest_path = output / "manifest.json"
    _atomic_write_json(manifest, manifest_path)
    return {**report, "output_directory": str(output), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build past-only features from cleaned Zikit challenge metrics"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument("--require-all-tenants", action="store_true")
    args = parser.parse_args()
    if args.expected_runs is not None and args.expected_runs < 1:
        parser.error("--expected-runs must be positive")
    report = build_feature_dataset(
        args.input,
        args.output,
        expected_runs=args.expected_runs,
        require_all_tenants=args.require_all_tenants,
    )
    print(
        json.dumps(
            {
                "feature_version": report["feature_version"],
                "run_count": report["run_count"],
                "tenant_count": report["tenant_count"],
                "rows_input": report["rows_input"],
                "rows_output": report["rows_output"],
                "feature_count": report["feature_count"],
                "output_directory": report["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
