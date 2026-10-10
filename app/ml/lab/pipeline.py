from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.evaluate import binary_metrics
from app.ml.lab.scenarios import ALL_SCENARIOS, LAB_SCHEMA_VERSION, GenerationConfig
from app.ml.model import FEATURES, FEATURE_VERSION, ModelBundle, load_bundle, save_bundle
from app.ml.train import train, validate_features

REQUIRED_COLUMNS = {
    "lab_schema_version",
    "feature_version",
    "run_id",
    "split",
    "scenario",
    "step",
    "timestamp",
    "is_anomaly",
    *FEATURES,
}


def _dataset_path(path: Path) -> Path:
    return path / "dataset.parquet" if path.is_dir() or not path.suffix else path


def load_dataset(path: Path) -> pd.DataFrame:
    resolved = _dataset_path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    manifest_path = path / "manifest.json" if path.is_dir() else resolved.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_checksum = manifest.get("dataset_sha256")
        if expected_checksum and sha256(resolved.read_bytes()).hexdigest() != expected_checksum:
            raise ValueError("lab dataset digest mismatch")
    frame = pd.read_parquet(resolved) if resolved.suffix == ".parquet" else pd.read_csv(resolved)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"lab dataset is missing columns: {sorted(missing)}")
    if set(frame["lab_schema_version"].astype(str)) != {LAB_SCHEMA_VERSION}:
        raise ValueError("incompatible lab schema version")
    if set(frame["feature_version"].astype(str)) != {FEATURE_VERSION}:
        raise ValueError("incompatible feature version")
    if not set(frame["split"].unique()).issubset({"train", "validation", "test"}):
        raise ValueError("invalid split label")
    validate_features(frame)
    duplicated = frame.duplicated(["run_id", "step"])
    if duplicated.any():
        raise ValueError("duplicate run_id/step rows")
    split_counts = frame.groupby("run_id")["split"].nunique()
    if (split_counts != 1).any():
        raise ValueError("a run crosses dataset splits")
    return frame


def write_dataset(frame: pd.DataFrame, output: Path, config: GenerationConfig) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "dataset.parquet"
    temporary = output / ".dataset.parquet.tmp"
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(dataset_path)
    checksum = sha256(dataset_path.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "lab_schema_version": LAB_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "seed": config.seed,
        "normal_runs": config.normal_runs,
        "fault_runs": config.fault_runs,
        "run_length": config.run_length,
        "rows": len(frame),
        "runs": int(frame["run_id"].nunique()),
        "splits": {str(key): int(value) for key, value in frame.groupby("split")["run_id"].nunique().items()},
        "scenarios": {
            str(key): int(value) for key, value in frame.groupby("scenario")["run_id"].nunique().items()
        },
        "features": list(FEATURES),
        "dataset_sha256": checksum,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def train_lab_model(
    frame: pd.DataFrame,
    output: Path,
    *,
    target_false_positive_rate: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0 < target_false_positive_rate <= 0.2:
        raise ValueError("target_false_positive_rate must be in (0, 0.2]")
    training = frame.loc[(frame["split"] == "train") & ~frame["is_anomaly"].astype(bool)].copy()
    validation = frame.loc[(frame["split"] == "validation") & ~frame["is_anomaly"].astype(bool)].copy()
    if training.empty or validation.empty:
        raise ValueError("training and validation require normal rows")
    training_runs = set(training["run_id"])
    validation_runs = set(validation["run_id"])
    if training_runs & validation_runs:
        raise ValueError("training and validation runs overlap")

    bundle = train(training, contamination=target_false_positive_rate, random_state=seed)
    validation_scores = bundle.score(validation.loc[:, FEATURES].to_numpy(dtype=float))
    bundle.threshold = float(np.quantile(validation_scores, 1 - target_false_positive_rate, method="higher"))
    save_bundle(bundle, output)
    validation_predictions = validation_scores >= bundle.threshold
    report: dict[str, Any] = {
        "model": str(output),
        "feature_version": bundle.feature_version,
        "features": list(bundle.feature_names),
        "seed": seed,
        "target_false_positive_rate": target_false_positive_rate,
        "calibrated_threshold": bundle.threshold,
        "training_rows": len(training),
        "training_runs": len(training_runs),
        "validation_normal_rows": len(validation),
        "validation_runs": len(validation_runs),
        "observed_validation_false_positive_rate": float(validation_predictions.mean()),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _persistence_predictions(raw: np.ndarray, run_ids: pd.Series, windows: int) -> np.ndarray:
    if windows < 1:
        raise ValueError("persistence_windows must be positive")
    confirmed = np.zeros(len(raw), dtype=bool)
    previous_run: str | None = None
    streak = 0
    for index, (prediction, run_id) in enumerate(zip(raw, run_ids, strict=True)):
        current_run = str(run_id)
        if current_run != previous_run:
            previous_run = current_run
            streak = 0
        streak = streak + 1 if prediction else 0
        confirmed[index] = streak >= windows
    return confirmed


def _delay_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    delays: list[int] = []
    fault_runs = 0
    detected_runs = 0
    false_alert_runs = 0
    healthy_runs = 0
    for _, run in frame.groupby("run_id", sort=False):
        labels = run["is_anomaly"].to_numpy(dtype=bool)
        predictions = run[prediction_column].to_numpy(dtype=bool)
        if labels.any():
            fault_runs += 1
            first_fault = int(np.flatnonzero(labels)[0])
            valid_detections = np.flatnonzero(predictions & labels)
            if valid_detections.size:
                detected_runs += 1
                delays.append(max(0, int(valid_detections[0]) - first_fault))
        else:
            healthy_runs += 1
            false_alert_runs += int(predictions.any())
    return {
        "fault_runs": fault_runs,
        "detected_fault_runs": detected_runs,
        "detection_rate": float(detected_runs / fault_runs) if fault_runs else None,
        "median_detection_delay_steps": float(np.median(delays)) if delays else None,
        "p95_detection_delay_steps": float(np.quantile(delays, 0.95)) if delays else None,
        "healthy_runs": healthy_runs,
        "false_alert_runs": false_alert_runs,
        "false_alert_run_rate": float(false_alert_runs / healthy_runs) if healthy_runs else None,
    }


def _metrics_for(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    labels = frame["is_anomaly"].to_numpy(dtype=bool)
    predictions = frame[prediction_column].to_numpy(dtype=bool)
    values: dict[str, Any] = binary_metrics(labels, predictions)
    values.update(_delay_metrics(frame, prediction_column))
    values["rows"] = len(frame)
    values["anomalous_rows"] = int(labels.sum())
    return values


def benchmark_model(
    frame: pd.DataFrame,
    model_path: Path,
    *,
    persistence_windows: int = 3,
    split: str = "test",
    minimum_recall: float = 0.80,
    maximum_false_positive_rate: float = 0.03,
    maximum_p95_delay_steps: float = 12,
    maximum_false_alert_run_rate: float = 0.10,
    minimum_scenario_detection_rate: float = 0.90,
) -> dict[str, Any]:
    evaluation = frame.loc[frame["split"] == split].copy()
    if evaluation.empty:
        raise ValueError(f"split has no rows: {split}")
    evaluation = evaluation.sort_values(["run_id", "step"], kind="stable").reset_index(drop=True)
    bundle = load_bundle(model_path, trusted_root=model_path.parent)
    if bundle is None:
        raise FileNotFoundError(model_path)
    scores = bundle.score(evaluation.loc[:, FEATURES].to_numpy(dtype=float))
    raw = scores >= bundle.threshold
    evaluation["score"] = scores
    evaluation["raw_prediction"] = raw
    evaluation["confirmed_prediction"] = _persistence_predictions(
        raw, evaluation["run_id"], persistence_windows
    )

    overall = _metrics_for(evaluation, "confirmed_prediction")
    raw_metrics = _metrics_for(evaluation, "raw_prediction")
    per_scenario = {
        scenario: _metrics_for(group, "confirmed_prediction")
        for scenario, group in evaluation.groupby("scenario", sort=True)
    }
    p95_delay = overall["p95_detection_delay_steps"]
    scenario_detection_rates = [
        metrics["detection_rate"]
        for metrics in per_scenario.values()
        if metrics["detection_rate"] is not None
    ]
    worst_scenario_detection_rate = min(scenario_detection_rates) if scenario_detection_rates else None
    gates = {
        "minimum_recall": {
            "target": minimum_recall,
            "actual": overall["recall"],
            "passed": overall["recall"] >= minimum_recall,
        },
        "maximum_false_positive_rate": {
            "target": maximum_false_positive_rate,
            "actual": overall["false_positive_rate"],
            "passed": overall["false_positive_rate"] <= maximum_false_positive_rate,
        },
        "maximum_p95_detection_delay_steps": {
            "target": maximum_p95_delay_steps,
            "actual": p95_delay,
            "passed": p95_delay is not None and p95_delay <= maximum_p95_delay_steps,
        },
        "maximum_false_alert_run_rate": {
            "target": maximum_false_alert_run_rate,
            "actual": overall["false_alert_run_rate"],
            "passed": (
                overall["false_alert_run_rate"] is not None
                and overall["false_alert_run_rate"] <= maximum_false_alert_run_rate
            ),
        },
        "minimum_scenario_detection_rate": {
            "target": minimum_scenario_detection_rate,
            "actual": worst_scenario_detection_rate,
            "passed": (
                worst_scenario_detection_rate is not None
                and worst_scenario_detection_rate >= minimum_scenario_detection_rate
            ),
        },
    }
    missing_scenarios = sorted(set(ALL_SCENARIOS) - set(per_scenario))
    return {
        "model": str(model_path),
        "split": split,
        "threshold": bundle.threshold,
        "persistence_windows": persistence_windows,
        "overall": overall,
        "raw_without_persistence": raw_metrics,
        "per_scenario": per_scenario,
        "missing_scenarios": missing_scenarios,
        "quality_gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()) and not missing_scenarios,
    }
