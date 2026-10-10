from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import pandas as pd

from app.ml.model import FEATURES, FEATURE_VERSION

LAB_SCHEMA_VERSION: Final[str] = "1"
FAULT_SCENARIOS: Final[tuple[str, ...]] = (
    "constant_latency",
    "progressive_latency",
    "http_error",
    "timeout",
    "intermittent",
    "unavailable",
    "traffic_surge",
    "traffic_drop",
    "correlated_degradation",
)
NORMAL_SCENARIOS: Final[tuple[str, ...]] = (
    "normal",
    "legitimate_spike",
    "baseline_drift",
)
ALL_SCENARIOS: Final[tuple[str, ...]] = NORMAL_SCENARIOS + FAULT_SCENARIOS


@dataclass(frozen=True)
class GenerationConfig:
    normal_runs: int = 500
    fault_runs: int = 1_000
    run_length: int = 120
    seed: int = 42
    train_fraction: float = 0.6
    validation_fraction: float = 0.2

    def validate(self) -> None:
        if self.normal_runs < 3:
            raise ValueError("normal_runs must be at least 3")
        if self.fault_runs < len(FAULT_SCENARIOS) * 3:
            raise ValueError(f"fault_runs must be at least {len(FAULT_SCENARIOS) * 3}")
        if self.run_length < 30:
            raise ValueError("run_length must be at least 30")
        if not 0.5 <= self.train_fraction < 1:
            raise ValueError("train_fraction must be in [0.5, 1)")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")


def _allocate(total: int, names: Iterable[str]) -> dict[str, int]:
    ordered = tuple(names)
    quotient, remainder = divmod(total, len(ordered))
    return {name: quotient + (index < remainder) for index, name in enumerate(ordered)}


def _split_labels(count: int, rng: np.random.Generator, train: float, validation: float) -> list[str]:
    train_count = max(1, int(count * train))
    validation_count = max(1, int(count * validation))
    if train_count + validation_count >= count:
        train_count = max(1, count - 2)
        validation_count = 1
    labels = ["train"] * train_count
    labels.extend(["validation"] * validation_count)
    labels.extend(["test"] * (count - len(labels)))
    rng.shuffle(labels)
    return labels


def _ar_noise(rng: np.random.Generator, length: int, sigma: float) -> np.ndarray:
    values = np.zeros(length, dtype=float)
    shocks = rng.normal(0, sigma, length)
    for index in range(1, length):
        values[index] = 0.72 * values[index - 1] + shocks[index]
    return values


def _normal_features(rng: np.random.Generator, length: int) -> dict[str, np.ndarray]:
    latency = np.clip(1 + _ar_noise(rng, length, 0.045), 0.72, 1.35)
    request_rate = np.clip(1 + _ar_noise(rng, length, 0.065), 0.58, 1.52)
    error_rate = np.clip(rng.beta(0.45, 38, length), 0, 0.08)
    error_rate[rng.random(length) < 0.72] = 0
    timeout_rate = error_rate * rng.uniform(0, 0.22, length)
    return {
        "latency_ratio": latency,
        "error_rate": error_rate,
        "timeout_rate": timeout_rate,
        "request_rate_ratio": request_rate,
        "availability": 1 - error_rate,
    }


def _fault_bounds(rng: np.random.Generator, length: int) -> tuple[int, int]:
    start = int(rng.integers(max(5, int(length * 0.20)), max(6, int(length * 0.36))))
    end = int(rng.integers(max(start + 6, int(length * 0.68)), max(start + 7, int(length * 0.84))))
    return start, min(end, length - 3)


def _apply_normal_challenge(
    features: dict[str, np.ndarray], scenario: str, rng: np.random.Generator
) -> None:
    length = len(features["latency_ratio"])
    if scenario == "legitimate_spike":
        center = int(rng.integers(5, length - 5))
        width = int(rng.integers(1, 3))
        target = slice(center, center + width)
        if rng.random() < 0.5:
            features["latency_ratio"][target] += rng.uniform(0.35, 0.75)
        else:
            features["request_rate_ratio"][target] *= rng.uniform(1.5, 2.1)
    elif scenario == "baseline_drift":
        drift = np.linspace(-rng.uniform(0.05, 0.14), rng.uniform(0.08, 0.22), length)
        features["latency_ratio"] = np.clip(features["latency_ratio"] + drift, 0.65, 1.5)
        rate_drift = np.linspace(rng.uniform(0.08, 0.20), -rng.uniform(0.05, 0.16), length)
        features["request_rate_ratio"] = np.clip(features["request_rate_ratio"] + rate_drift, 0.5, 1.6)


def _apply_fault(
    features: dict[str, np.ndarray], scenario: str, rng: np.random.Generator, start: int, end: int
) -> np.ndarray:
    active = np.zeros(len(features["latency_ratio"]), dtype=bool)
    active[start:end] = True
    count = end - start
    latency = features["latency_ratio"]
    error = features["error_rate"]
    timeout = features["timeout_rate"]
    rate = features["request_rate_ratio"]

    if scenario == "constant_latency":
        latency[start:end] += rng.uniform(1.4, 7.0) * rng.uniform(0.88, 1.12, count)
    elif scenario == "progressive_latency":
        latency[start:end] += np.linspace(0.05, rng.uniform(3.0, 9.0), count)
    elif scenario == "http_error":
        error[start:end] = np.clip(rng.normal(rng.uniform(0.28, 0.82), 0.07, count), 0.12, 1)
    elif scenario == "timeout":
        timeout[start:end] = np.clip(rng.normal(rng.uniform(0.25, 0.82), 0.07, count), 0.10, 1)
        error[start:end] = np.maximum(error[start:end], timeout[start:end])
        latency[start:end] += rng.uniform(3.0, 12.0)
    elif scenario == "intermittent":
        # Bursts exceed the production persistence window, with healthy gaps in between.
        burst = (np.arange(count) // int(rng.integers(4, 8))) % 2 == 0
        burst_error = np.clip(rng.normal(rng.uniform(0.42, 0.82), 0.08, count), 0.18, 1)
        error[start:end] = np.where(burst, burst_error, error[start:end])
        latency[start:end] += np.where(burst, rng.uniform(0.8, 3.5), 0)
    elif scenario == "unavailable":
        error[start:end] = 1
        timeout[start:end] = rng.uniform(0.25, 0.85, count)
        latency[start:end] += rng.uniform(1.5, 8.0)
    elif scenario == "traffic_surge":
        rate[start:end] *= rng.uniform(2.5, 10.0)
        latency[start:end] += np.linspace(0.15, rng.uniform(0.8, 2.5), count)
    elif scenario == "traffic_drop":
        rate[start:end] *= rng.uniform(0.01, 0.18)
    elif scenario == "correlated_degradation":
        progress = np.linspace(0.15, 1, count)
        latency[start:end] += rng.uniform(2.0, 7.0) * progress
        error[start:end] = np.maximum(error[start:end], rng.uniform(0.18, 0.65) * progress)
        timeout[start:end] = np.maximum(timeout[start:end], error[start:end] * rng.uniform(0.1, 0.55))
        rate[start:end] *= np.linspace(1, rng.uniform(0.15, 0.55), count)
    else:
        raise ValueError(f"unknown fault scenario: {scenario}")

    features["error_rate"] = np.clip(error, 0, 1)
    features["timeout_rate"] = np.minimum(np.clip(timeout, 0, 1), features["error_rate"])
    features["availability"] = 1 - features["error_rate"]
    features["latency_ratio"] = np.clip(latency, 0.01, None)
    features["request_rate_ratio"] = np.clip(rate, 0.001, None)
    return active


def _generate_run(
    scenario: str,
    run_id: str,
    split: str,
    length: int,
    rng: np.random.Generator,
    started_at: datetime,
) -> pd.DataFrame:
    features = _normal_features(rng, length)
    is_anomaly = np.zeros(length, dtype=bool)
    fault_start: int | None = None
    fault_end: int | None = None
    if scenario in NORMAL_SCENARIOS:
        _apply_normal_challenge(features, scenario, rng)
    else:
        fault_start, fault_end = _fault_bounds(rng, length)
        is_anomaly = _apply_fault(features, scenario, rng, fault_start, fault_end)

    phase = np.full(length, "normal", dtype=object)
    if fault_start is not None and fault_end is not None:
        phase[fault_start:fault_end] = "fault"
        phase[fault_end:] = "recovery"
    service = "backup" if rng.random() < 0.15 else "primary"
    return pd.DataFrame(
        {
            "lab_schema_version": LAB_SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "run_id": run_id,
            "split": split,
            "scenario": scenario,
            "service": service,
            "step": np.arange(length, dtype=int),
            "timestamp": [started_at + timedelta(seconds=step) for step in range(length)],
            "phase": phase,
            "is_anomaly": is_anomaly,
            **features,
        }
    )


def generate_dataset(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    """Generate deterministic, labeled feature sequences without external services or LLMs."""
    config.validate()
    root_rng = np.random.default_rng(config.seed)
    allocations = {
        **_allocate(config.normal_runs, NORMAL_SCENARIOS),
        **_allocate(config.fault_runs, FAULT_SCENARIOS),
    }
    frames: list[pd.DataFrame] = []
    base = datetime(2026, 1, 1, tzinfo=UTC)
    run_number = 0
    for scenario in ALL_SCENARIOS:
        count = allocations[scenario]
        labels = _split_labels(count, root_rng, config.train_fraction, config.validation_fraction)
        for scenario_index, split in enumerate(labels):
            seed = int(root_rng.integers(0, np.iinfo(np.uint32).max))
            run_rng = np.random.default_rng(seed)
            run_id = f"{scenario}-{scenario_index:06d}"
            frames.append(
                _generate_run(
                    scenario,
                    run_id,
                    split,
                    config.run_length,
                    run_rng,
                    base + timedelta(seconds=run_number * (config.run_length + 10)),
                )
            )
            run_number += 1
    dataset = pd.concat(frames, ignore_index=True)
    if not np.isfinite(dataset.loc[:, FEATURES].to_numpy(dtype=float)).all():
        raise RuntimeError("generator produced non-finite features")
    return dataset
