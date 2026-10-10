from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.database import MetricRecord, configure_database, session_factory
from app.ml.model import FEATURES


RAW_COLUMNS = ("service", "timestamp", "latency_ms", "status_code", "timed_out", "success")


def samples_to_features(samples: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """Build version-1 features from observed samples using past-only baselines."""
    missing = set(RAW_COLUMNS) - set(samples.columns)
    if missing:
        raise ValueError(f"missing telemetry columns: {sorted(missing)}")
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    frame = samples.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values(["service", "timestamp"], kind="stable")
    outputs: list[pd.DataFrame] = []
    for service, group in frame.groupby("service", sort=False):
        group = group.copy().reset_index(drop=True)
        latency = pd.to_numeric(group["latency_ms"], errors="coerce")
        failure = ~group["success"].astype(bool)
        timeout = group["timed_out"].astype(bool)
        rolling = latency.rolling(window_size, min_periods=3)
        p95 = rolling.quantile(0.95)
        error_rate = failure.astype(float).rolling(window_size, min_periods=3).mean()
        timeout_rate = timeout.astype(float).rolling(window_size, min_periods=3).mean()
        availability = 1 - error_rate
        elapsed = group["timestamp"].diff(window_size - 1).dt.total_seconds()
        request_rate = window_size / elapsed.replace(0, np.nan)
        latency_baseline = p95.expanding(min_periods=1).median().shift(1)
        rate_baseline = request_rate.expanding(min_periods=1).median().shift(1)
        result = pd.DataFrame(
            {
                "service": service,
                "timestamp": group["timestamp"],
                "latency_ratio": p95 / latency_baseline.clip(lower=1e-6),
                "error_rate": error_rate,
                "timeout_rate": timeout_rate,
                "request_rate_ratio": request_rate / rate_baseline.clip(lower=1e-6),
                "availability": availability,
            }
        )
        outputs.append(result.dropna(subset=list(FEATURES)))
    if not outputs:
        return pd.DataFrame(columns=["service", "timestamp", *FEATURES])
    return pd.concat(outputs, ignore_index=True)


def load_persisted_samples(database_url: str) -> pd.DataFrame:
    configure_database(database_url)
    with session_factory()() as session:
        records = list(session.scalars(select(MetricRecord).order_by(MetricRecord.timestamp)))
    return pd.DataFrame(
        [
            {
                "service": record.service,
                "timestamp": record.timestamp,
                "latency_ms": record.latency_ms,
                "status_code": record.status_code,
                "timed_out": record.timed_out,
                "success": record.success,
            }
            for record in records
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert observed Sentinel telemetry to ML features")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--database-url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=20)
    args = parser.parse_args()
    if args.input:
        samples = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    else:
        samples = load_persisted_samples(args.database_url)
    features = samples_to_features(samples, args.window_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"saved {len(features)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
