from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.challenge.contracts import METRIC_COLUMNS, TARGET_TENANT
from app.challenge.features import FEATURE_SCHEMA_VERSION
from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.model import (
    CHALLENGE_MODEL_VERSION,
    COMPONENT_WEIGHTS,
    ChallengeModelBundle,
    save_challenge_bundle,
)
from app.challenge.normal_behavior import fit_normal_behavior_model


HORIZONS_MINUTES = (40, 120, 360, 1440)
DEFAULT_SPLIT_COUNTS = {"train": 36, "validation": 9, "test": 9}
PROXY_SMOOTHING_MINUTES = 15
PROXY_SEVERITY_QUANTILE = 0.98
PROXY_DEFAULT_SEVERITY_THRESHOLD = 1.30
PROXY_EPISODE_GAP_MINUTES = 60
PROXY_MIN_EPISODE_POINTS = 5
CALIBRATION_ROWS_PER_RUN = 5_000
MAX_POSITIVE_ROWS_PER_HORIZON = 80_000
NEGATIVE_TO_POSITIVE_RATIO = 2
MIN_NEGATIVE_ROWS_PER_RUN = 500
MODEL_ARCHITECTURE_VERSION = "challenge-kaiku-early-warning-architecture-v1"
FORBIDDEN_INFERENCE_FEATURE_TOKENS = (
    "future",
    "target",
    "proxy_anchor",
    "minutes_to_proxy",
    "lead_time",
)

NON_MODEL_COLUMNS = frozenset(
    {
        "feature_version",
        "run_id",
        "tenant",
        "timestamp",
        "source_name",
        "source_line",
        "data_quality",
        "log_feature_version",
    }
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_for(seed: int, *parts: object) -> int:
    value = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:8], "big")


def assign_run_splits(
    run_ids: list[str],
    *,
    seed: int = 42,
    counts: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    requested = dict(DEFAULT_SPLIT_COUNTS if counts is None else counts)
    if set(requested) != {"train", "validation", "test"}:
        raise ValueError("split counts must define train, validation and test")
    if any(value < 1 for value in requested.values()):
        raise ValueError("every split must contain at least one run")
    unique = sorted(set(run_ids))
    if len(unique) != len(run_ids):
        raise ValueError("run ids must be unique")
    if sum(requested.values()) != len(unique):
        raise ValueError("split counts must equal the available run count")
    ordered = sorted(
        unique, key=lambda run_id: sha256(f"{seed}:{run_id}".encode("utf-8")).hexdigest()
    )
    train_end = requested["train"]
    validation_end = train_end + requested["validation"]
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def discover_feature_files(
    input_directory: Path, *, expected_runs: int | None = None
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    root = assert_allowed_path(input_directory)
    if not root.is_dir():
        raise IngestionError("phase 3 requires the phase-2 output directory")
    manifest_path = assert_allowed_path(root / "manifest.json")
    report_path = assert_allowed_path(root / "feature-report.json")
    if not manifest_path.is_file() or not report_path.is_file():
        raise IngestionError("phase-2 manifest and feature report are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("feature_version") != FEATURE_SCHEMA_VERSION:
        raise IngestionError("incompatible phase-2 manifest")
    if report.get("feature_version") != FEATURE_SCHEMA_VERSION:
        raise IngestionError("incompatible phase-2 feature report")
    if manifest.get("scope", {}).get("target_tenant") != TARGET_TENANT:
        raise IngestionError("phase-2 manifest is not scoped to Kaiku")
    if report.get("scope", {}).get("target_tenant") != TARGET_TENANT:
        raise IngestionError("phase-2 feature report is not scoped to Kaiku")
    expected_report_digest = manifest.get("report", {}).get("sha256")
    if expected_report_digest != _sha256_file(report_path):
        raise IngestionError("phase-2 feature report digest mismatch")

    files: dict[str, Path] = {}
    for record in manifest.get("feature_files", []):
        run_id = str(record.get("run_id", ""))
        candidate = assert_allowed_path(root / str(record.get("path", "")))
        if root != candidate and root not in candidate.parents:
            raise IngestionError("feature manifest points outside its output directory")
        if run_id in files:
            raise IngestionError(f"duplicate feature run in manifest: {run_id}")
        if not candidate.is_file():
            raise IngestionError(f"feature file is missing for {run_id}")
        if record.get("sha256") != _sha256_file(candidate):
            raise IngestionError(f"feature digest mismatch for {run_id}")
        files[run_id] = candidate
    if expected_runs is not None and len(files) != expected_runs:
        raise IngestionError(f"expected {expected_runs} feature runs, found {len(files)}")
    if not files:
        raise IngestionError("phase-2 manifest contains no feature files")
    return files, report, manifest


def _read_kaiku_features(
    path: Path, *, columns: list[str] | None = None
) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    if "tenant" not in frame.columns:
        raise IngestionError(f"feature file is missing tenant: {path.name}")
    tenants = set(frame["tenant"].astype(str).unique())
    if tenants != {TARGET_TENANT}:
        raise IngestionError(
            f"feature file must contain only {TARGET_TENANT}: {path.name} {sorted(tenants)}"
        )
    return frame


def _raw_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    columns = [
        pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype="float32")
        for feature in feature_names
    ]
    return np.column_stack(columns).astype("float32", copy=False)


def _impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    unavailable = ~np.isfinite(result)
    if unavailable.any():
        result[unavailable] = np.take(medians, np.nonzero(unavailable)[1])
    return result


def _finite_medians(matrix: np.ndarray) -> np.ndarray:
    medians = np.zeros(matrix.shape[1], dtype="float32")
    for index in range(matrix.shape[1]):
        finite = matrix[:, index][np.isfinite(matrix[:, index])]
        medians[index] = np.median(finite).astype("float32") if finite.size else 0.0
    return medians


def fit_proxy_reference(
    files: dict[str, Path], train_run_ids: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    columns = ["tenant", *METRIC_COLUMNS]
    calibration = pd.concat(
        [_read_kaiku_features(files[run_id], columns=columns) for run_id in train_run_ids],
        ignore_index=True,
    )
    reference: dict[str, dict[str, dict[str, float]]] = {}
    for tenant, group in calibration.groupby("tenant", sort=True):
        reference[str(tenant)] = {}
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                raise IngestionError(f"training has no usable {metric} values for {tenant}")
            reference[str(tenant)][metric] = {
                "median": float(values.median()),
                "q99": float(values.quantile(0.99)),
            }
    if set(reference) != {TARGET_TENANT}:
        raise IngestionError("training split does not contain only Kaiku")
    return reference


def _proxy_severity_for_tenant(
    frame: pd.DataFrame,
    reference: dict[str, dict[str, dict[str, float]]],
) -> tuple[pd.DataFrame, pd.Series]:
    if frame["tenant"].nunique() != 1:
        raise ValueError("proxy severity input must contain one tenant")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    tenant = str(ordered.at[0, "tenant"])
    if tenant not in reference:
        raise ValueError(f"proxy reference is missing tenant: {tenant}")
    normalized: list[np.ndarray] = []
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(ordered[metric], errors="coerce").to_numpy(dtype=float)
        metric_reference = reference[tenant][metric]
        scale = max(metric_reference["q99"] - metric_reference["median"], 1e-6)
        normalized.append(np.maximum((values - metric_reference["median"]) / scale, 0))
    matrix = np.column_stack(normalized)
    ranked = np.sort(np.where(np.isfinite(matrix), matrix, 0), axis=1)
    eligible = (
        ~ordered["context_known_event"].astype(bool)
        & ordered["data_quality"].ne("outage")
        & ordered.loc[:, METRIC_COLUMNS].notna().sum(axis=1).ge(3)
    )
    severity = pd.Series(ranked[:, -1] + 0.35 * ranked[:, -2]).where(eligible)
    severity = severity.rolling(
        PROXY_SMOOTHING_MINUTES, min_periods=5, closed="right"
    ).mean()
    return ordered, severity.where(eligible)


def fit_proxy_severity_thresholds(
    files: dict[str, Path],
    train_run_ids: list[str],
    reference: dict[str, dict[str, dict[str, float]]],
    *,
    quantile: float = PROXY_SEVERITY_QUANTILE,
) -> dict[str, float]:
    if not 0 < quantile < 1:
        raise ValueError("proxy severity quantile must be between zero and one")
    values_by_tenant: dict[str, list[np.ndarray]] = {TARGET_TENANT: []}
    for run_id in train_run_ids:
        frame = _read_kaiku_features(files[run_id])
        for tenant, tenant_frame in frame.groupby("tenant", sort=True):
            _, severity = _proxy_severity_for_tenant(tenant_frame, reference)
            finite = severity.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                values_by_tenant[str(tenant)].append(finite)
    thresholds: dict[str, float] = {}
    for tenant, chunks in values_by_tenant.items():
        if not chunks:
            raise IngestionError(
                f"training has no eligible severity values for {tenant}"
            )
        thresholds[tenant] = float(np.quantile(np.concatenate(chunks), quantile))
    return thresholds


def proxy_anchors_for_tenant(
    frame: pd.DataFrame,
    reference: dict[str, dict[str, dict[str, float]]],
    *,
    severity_threshold: float = PROXY_DEFAULT_SEVERITY_THRESHOLD,
    episode_gap_minutes: int = PROXY_EPISODE_GAP_MINUTES,
    minimum_episode_points: int = PROXY_MIN_EPISODE_POINTS,
) -> pd.Series:
    if not np.isfinite(severity_threshold) or severity_threshold <= 0:
        raise ValueError("proxy severity threshold must be finite and positive")
    if episode_gap_minutes < 1 or minimum_episode_points < 1:
        raise ValueError("proxy episode controls must be positive")
    ordered, severity = _proxy_severity_for_tenant(frame, reference)
    timestamps = pd.to_datetime(ordered["timestamp"], errors="raise")
    qualifying = np.flatnonzero(severity.ge(severity_threshold).fillna(False).to_numpy())
    episodes: list[list[int]] = []
    for index in qualifying:
        if not episodes:
            episodes.append([int(index)])
            continue
        gap = (timestamps.iloc[index] - timestamps.iloc[episodes[-1][-1]]).total_seconds()
        if gap > episode_gap_minutes * 60:
            episodes.append([int(index)])
        else:
            episodes[-1].append(int(index))
    selected = [
        int(severity.iloc[episode].idxmax())
        for episode in episodes
        if len(episode) >= minimum_episode_points
    ]
    anchors = pd.Series(False, index=ordered.index)
    anchors.loc[selected] = True
    return anchors


def proxy_labels_for_run(
    frame: pd.DataFrame,
    reference: dict[str, dict[str, dict[str, float]]],
    *,
    horizons: tuple[int, ...] = HORIZONS_MINUTES,
    severity_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for _, tenant_frame in frame.groupby("tenant", sort=True):
        ordered = tenant_frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        timestamp = pd.to_datetime(ordered["timestamp"], errors="raise")
        tenant = str(ordered.at[0, "tenant"])
        threshold = (
            PROXY_DEFAULT_SEVERITY_THRESHOLD
            if severity_thresholds is None
            else severity_thresholds.get(tenant)
        )
        if threshold is None:
            raise ValueError(f"proxy severity threshold is missing tenant: {tenant}")
        anchors = proxy_anchors_for_tenant(
            ordered,
            reference,
            severity_threshold=float(threshold),
        )
        next_anchor = timestamp.where(anchors).bfill()
        lead_minutes = next_anchor.sub(timestamp).dt.total_seconds().div(60)
        remaining_minutes = (timestamp.iloc[-1] - timestamp).dt.total_seconds().div(60)
        labels = ordered.loc[:, ["run_id", "tenant", "timestamp"]].copy()
        labels["proxy_anchor"] = anchors.to_numpy(dtype=bool)
        labels["minutes_to_proxy_anchor"] = lead_minutes.astype("float32")
        for horizon in horizons:
            valid = remaining_minutes.ge(horizon)
            labels[f"valid_{horizon}m"] = valid.to_numpy(dtype=bool)
            labels[f"target_{horizon}m"] = (
                lead_minutes.gt(0) & lead_minutes.le(horizon) & valid
            ).to_numpy(dtype=bool)
        outputs.append(labels)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["tenant", "timestamp"], kind="stable", ignore_index=True
    )


def _sample_indices(
    indices: np.ndarray, count: int, *, seed: int
) -> np.ndarray:
    if len(indices) <= count:
        return indices
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(indices, size=count, replace=False))


def _select_feature_names(report: dict[str, Any], sample_path: Path) -> tuple[str, ...]:
    declared = [
        str(column)
        for column in report.get("feature_columns", [])
        if str(column) not in NON_MODEL_COLUMNS
    ]
    sample = _read_kaiku_features(sample_path)
    numeric = set(sample.select_dtypes(include=["number", "bool"]).columns)
    selected = [column for column in declared if column in numeric]
    if not selected:
        raise IngestionError("phase-2 output contains no numeric model features")
    if not any(name.endswith("_past_robust_z_1440m") for name in selected):
        raise IngestionError("phase-2 output is missing causal baseline features")
    feature_names = tuple(selected)
    _assert_causal_feature_names(feature_names)
    return feature_names


def _assert_causal_feature_names(feature_names: tuple[str, ...]) -> None:
    forbidden = sorted(
        name
        for name in feature_names
        if name == "timestamp"
        or any(
            token in name.casefold()
            for token in FORBIDDEN_INFERENCE_FEATURE_TOKENS
        )
    )
    if forbidden:
        raise IngestionError(
            f"model feature set contains future or target leakage: {forbidden}"
        )


def _isolation_feature_names(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(
        name
        for name in feature_names
        if (
            name.endswith("_past_robust_z_1440m")
            or name.endswith("_past_time_of_day_residual")
            or "_trend_per_min_" in name
            or ("_relative_change_" in name and not name.startswith("errores_"))
            or name.startswith("telemetry_coverage_")
        )
    )
    if not selected:
        raise IngestionError("no residual features are available for Isolation Forest")
    return selected


def _component_raw_values(
    matrix: np.ndarray, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    baseline_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.endswith("_past_robust_z_1440m") and not name.startswith("errores_")
    ]
    change_indices = [
        index
        for index, name in enumerate(feature_names)
        if "_trend_per_min_" in name
        or ("_relative_change_" in name and not name.startswith("errores_"))
    ]
    baseline = np.maximum(matrix[:, baseline_indices], 0).max(axis=1)
    change = np.maximum(matrix[:, change_indices], 0).max(axis=1)
    return baseline, change


def _fit_calibration(
    files: dict[str, Path],
    train_run_ids: list[str],
    feature_names: tuple[str, ...],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    samples: list[pd.DataFrame] = []
    for run_id in train_run_ids:
        frame = _read_kaiku_features(files[run_id])
        indices = _sample_indices(
            np.arange(len(frame)),
            min(CALIBRATION_ROWS_PER_RUN, len(frame)),
            seed=_seed_for(seed, run_id, "calibration"),
        )
        samples.append(frame.iloc[indices].copy())
    calibration = pd.concat(samples, ignore_index=True)
    raw = _raw_matrix(calibration, feature_names)
    medians = _finite_medians(raw)
    return medians, _impute(raw, medians), calibration


def _fit_classifiers(
    files: dict[str, Path],
    train_run_ids: list[str],
    feature_names: tuple[str, ...],
    medians: np.ndarray,
    reference: dict[str, dict[str, dict[str, float]]],
    severity_thresholds: dict[str, float],
    *,
    seed: int,
) -> tuple[dict[int, HistGradientBoostingClassifier], dict[str, Any]]:
    matrices: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS_MINUTES}
    targets: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS_MINUTES}
    maximum_positive_per_run = max(1, MAX_POSITIVE_ROWS_PER_HORIZON // len(train_run_ids))

    for run_id in train_run_ids:
        frame = _read_kaiku_features(files[run_id]).sort_values(
            ["tenant", "timestamp"], kind="stable", ignore_index=True
        )
        labels = proxy_labels_for_run(
            frame,
            reference,
            severity_thresholds=severity_thresholds,
        )
        matrix = _impute(_raw_matrix(frame, feature_names), medians)
        for horizon in HORIZONS_MINUTES:
            valid = labels[f"valid_{horizon}m"].to_numpy(dtype=bool)
            target = labels[f"target_{horizon}m"].to_numpy(dtype=bool)
            positive = np.flatnonzero(valid & target)
            negative = np.flatnonzero(valid & ~target)
            positive = _sample_indices(
                positive,
                maximum_positive_per_run,
                seed=_seed_for(seed, run_id, horizon, "positive"),
            )
            negative_count = min(
                len(negative),
                max(MIN_NEGATIVE_ROWS_PER_RUN, len(positive) * NEGATIVE_TO_POSITIVE_RATIO),
            )
            negative = _sample_indices(
                negative,
                negative_count,
                seed=_seed_for(seed, run_id, horizon, "negative"),
            )
            selected = np.sort(np.concatenate([positive, negative]))
            matrices[horizon].append(matrix[selected])
            targets[horizon].append(target[selected])

    classifiers: dict[int, HistGradientBoostingClassifier] = {}
    sample_report: dict[str, Any] = {}
    for horizon in HORIZONS_MINUTES:
        matrix = np.concatenate(matrices[horizon])
        target = np.concatenate(targets[horizon])
        if np.unique(target).size != 2:
            raise IngestionError(f"proxy target {horizon}m does not contain both classes")
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=80,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=10,
            class_weight="balanced",
            random_state=seed + horizon,
        )
        estimator.fit(matrix, target)
        classifiers[horizon] = estimator
        sample_report[str(horizon)] = {
            "rows": int(len(target)),
            "positive_rows": int(target.sum()),
            "negative_rows": int((~target).sum()),
            "iterations": int(estimator.n_iter_),
        }
    return classifiers, sample_report


def _choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    beta_squared = 0.25
    denominator = beta_squared * precision[:-1] + recall[:-1]
    f_beta = np.divide(
        (1 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f_beta))])


def _binary_report(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[False, True]).ravel()
    return {
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def _score_split(
    files: dict[str, Path],
    run_ids: list[str],
    bundle: ChallengeModelBundle,
    reference: dict[str, dict[str, dict[str, float]]],
    severity_thresholds: dict[str, float],
    *,
    thresholds: dict[int, float] | None = None,
) -> tuple[dict[str, Any], dict[int, float]]:
    labels_by_horizon: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    scores_by_horizon: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    anchors_by_tenant = {TARGET_TENANT: 0}
    for run_id in run_ids:
        frame = _read_kaiku_features(files[run_id]).sort_values(
            ["tenant", "timestamp"], kind="stable", ignore_index=True
        )
        labels = proxy_labels_for_run(
            frame,
            reference,
            severity_thresholds=severity_thresholds,
        )
        scores = bundle.score_frame(frame)
        anchors_by_tenant[TARGET_TENANT] += int(labels["proxy_anchor"].sum())
        for horizon in HORIZONS_MINUTES:
            valid = labels[f"valid_{horizon}m"].to_numpy(dtype=bool)
            labels_by_horizon[horizon].append(
                labels.loc[valid, f"target_{horizon}m"].to_numpy(dtype=bool)
            )
            scores_by_horizon[horizon].append(
                scores.loc[valid, f"risk_{horizon}m"].to_numpy(dtype=float)
            )

    chosen: dict[int, float] = {}
    metrics: dict[str, Any] = {}
    for horizon in HORIZONS_MINUTES:
        labels = np.concatenate(labels_by_horizon[horizon])
        scores = np.concatenate(scores_by_horizon[horizon])
        threshold = (
            _choose_threshold(labels, scores)
            if thresholds is None
            else float(thresholds[horizon])
        )
        chosen[horizon] = threshold
        metrics[str(horizon)] = _binary_report(labels, scores, threshold=threshold)
    return {"anchors_by_tenant": anchors_by_tenant, "horizons": metrics}, chosen


def train_challenge_model(
    input_directory: Path,
    output_directory: Path,
    model_path: Path,
    *,
    seed: int = 42,
    expected_runs: int = 54,
    split_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    files, feature_report, feature_manifest = discover_feature_files(
        input_directory, expected_runs=expected_runs
    )
    splits = assign_run_splits(list(files), seed=seed, counts=split_counts)
    split_sets = {name: set(values) for name, values in splits.items()}
    overlap = (
        split_sets["train"] & split_sets["validation"]
        | split_sets["train"] & split_sets["test"]
        | split_sets["validation"] & split_sets["test"]
    )
    if overlap:
        raise IngestionError(f"runs cross dataset splits: {sorted(overlap)}")

    first_path = files[sorted(files)[0]]
    feature_names = _select_feature_names(feature_report, first_path)
    isolation_names = _isolation_feature_names(feature_names)
    reference = fit_proxy_reference(files, splits["train"])
    severity_thresholds = fit_proxy_severity_thresholds(
        files,
        splits["train"],
        reference,
    )
    medians, calibration_matrix, calibration_frame = _fit_calibration(
        files, splits["train"], feature_names, seed=seed
    )
    normal_behavior = fit_normal_behavior_model(
        (_read_kaiku_features(files[run_id]) for run_id in splits["train"]),
        training_runs=len(splits["train"]),
    )
    indices = {name: index for index, name in enumerate(feature_names)}
    isolation_indices = [indices[name] for name in isolation_names]
    normal = calibration_frame["data_quality"].eq("ok").to_numpy(dtype=bool)
    isolation_training = calibration_matrix[normal][:, isolation_indices]
    isolation = IsolationForest(
        n_estimators=160,
        max_samples=min(2048, len(isolation_training)),
        contamination="auto",
        n_jobs=-1,
        random_state=seed,
    )
    isolation.fit(isolation_training)
    isolation_raw = -isolation.score_samples(calibration_matrix[:, isolation_indices])
    baseline_raw, change_raw = _component_raw_values(calibration_matrix, feature_names)
    normal_behavior_raw = normal_behavior.raw_deviation(calibration_frame)
    component_ranges = {
        "isolation": (
            float(np.quantile(isolation_raw, 0.50)),
            float(np.quantile(isolation_raw, 0.995)),
        ),
        "baseline": (
            float(np.quantile(baseline_raw, 0.50)),
            float(np.quantile(baseline_raw, 0.995)),
        ),
        "change": (
            float(np.quantile(change_raw, 0.50)),
            float(np.quantile(change_raw, 0.995)),
        ),
        "normal_behavior": (
            float(np.quantile(normal_behavior_raw, 0.50)),
            float(np.quantile(normal_behavior_raw, 0.995)),
        ),
    }
    classifiers, training_samples = _fit_classifiers(
        files,
        splits["train"],
        feature_names,
        medians,
        reference,
        severity_thresholds,
        seed=seed,
    )
    horizon_weights = {40: 1.0, 120: 0.90, 360: 0.75, 1440: 0.60}
    log_feature_version = feature_report.get("log_feature_version")
    if log_feature_version is not None and not isinstance(log_feature_version, str):
        raise IngestionError("invalid log feature version in feature report")
    bundle = ChallengeModelBundle(
        classifiers=classifiers,
        isolation_estimator=isolation,
        feature_names=feature_names,
        isolation_feature_names=isolation_names,
        imputation_values=tuple(float(value) for value in medians),
        component_ranges=component_ranges,
        horizon_weights=horizon_weights,
        normal_behavior_model=normal_behavior,
        seed=seed,
        training_rows=sum(record["rows"] for record in training_samples.values()),
        log_feature_version=log_feature_version,
    )

    validation_metrics, selected_thresholds = _score_split(
        files,
        splits["validation"],
        bundle,
        reference,
        severity_thresholds,
    )
    test_metrics, _ = _score_split(
        files,
        splits["test"],
        bundle,
        reference,
        severity_thresholds,
        thresholds=selected_thresholds,
    )
    model = model_path.expanduser().resolve(strict=False)
    digest = save_challenge_bundle(bundle, model)
    output = output_directory.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    split_report = {
        "seed": seed,
        "counts": {name: len(values) for name, values in splits.items()},
        "runs": splits,
        "overlap": sorted(overlap),
    }
    split_path = output / "splits.json"
    _atomic_json(split_report, split_path)
    report: dict[str, Any] = {
        "model_version": CHALLENGE_MODEL_VERSION,
        "feature_version": FEATURE_SCHEMA_VERSION,
        "log_feature_version": log_feature_version,
        "scope": {"target_tenant": TARGET_TENANT, "kaiku_only": True},
        "seed": seed,
        "model": {"filename": model.name, "sha256": digest},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source": {
            "phase_two_manifest_sha256": _sha256_file(
                assert_allowed_path(input_directory) / "manifest.json"
            ),
            "phase_two_feature_files": len(feature_manifest["feature_files"]),
        },
        "splits": split_report,
        "feature_count": len(feature_names),
        "isolation_feature_count": len(isolation_names),
        "horizons_minutes": list(HORIZONS_MINUTES),
        "proxy_target": {
            "official_ground_truth": False,
            "selection_mode": "train_quantile_thresholded_episodes",
            "fixed_anchors_per_run": False,
            "severity_quantile": PROXY_SEVERITY_QUANTILE,
            "severity_thresholds": severity_thresholds,
            "episode_gap_minutes": PROXY_EPISODE_GAP_MINUTES,
            "minimum_episode_points": PROXY_MIN_EPISODE_POINTS,
            "known_business_events_excluded": True,
            "future_used_for_training_labels_only": True,
            "future_used_for_inference_features": False,
            "reference_fitted_on_split": "train",
            "reference": reference,
            "limitation": "proxy metrics do not estimate official incident recall or lead time",
        },
        "training_samples": training_samples,
        "model_architecture": {
            "version": MODEL_ARCHITECTURE_VERSION,
            "seasonal_baseline": "train-only context-conditioned robust profiles",
            "change_detection": "causal trends and relative changes",
            "isolation_forest_inputs": "residuals, causal changes and telemetry coverage only",
            "supervised_estimator": "HistGradientBoostingClassifier",
            "supervised_horizons_minutes": list(HORIZONS_MINUTES),
            "absolute_timestamp_used": False,
            "future_inference_features_used": False,
            "proxy_targets_used_as_inference_features": False,
            "isolation_raw_pressure_features_used": False,
        },
        "model_gates": {
            "split_isolation": not bool(overlap),
            "expected_horizons_present": set(classifiers) == set(HORIZONS_MINUTES),
            "absolute_timestamp_excluded": "timestamp" not in feature_names,
            "future_and_target_features_excluded": not any(
                token in name.casefold()
                for name in feature_names
                for token in FORBIDDEN_INFERENCE_FEATURE_TOKENS
            ),
            "isolation_uses_residual_inputs_only": all(
                name.endswith("_past_robust_z_1440m")
                or name.endswith("_past_time_of_day_residual")
                or "_trend_per_min_" in name
                or "_relative_change_" in name
                or name.startswith("telemetry_coverage_")
                for name in isolation_names
            ),
            "normal_behavior_fitted_on_train_only": (
                normal_behavior.training_runs == len(splits["train"])
            ),
        },
        "normal_behavior": {
            "version": normal_behavior.version,
            "fitted_on_split": "train",
            "training_runs": normal_behavior.training_runs,
            "training_rows": normal_behavior.training_rows,
            "time_bucket_minutes": normal_behavior.bucket_minutes,
            "exact_profile_count": len(normal_behavior.exact_profiles),
            "event_profile_counts": normal_behavior.profile_counts,
            "known_business_events_learned_as_normal_regimes": True,
            "official_ground_truth_used": False,
            "estimator": "context-conditioned robust median and scale",
        },
        "component_weights": COMPONENT_WEIGHTS,
        "component_ranges": component_ranges,
        "selected_validation_thresholds": {
            str(key): value for key, value in selected_thresholds.items()
        },
        "validation_proxy_metrics": validation_metrics,
        "test_proxy_metrics": test_metrics,
    }
    if not all(report["model_gates"].values()):
        raise IngestionError("early-warning model failed one or more architecture gates")
    report_path = output / "training-report.json"
    _atomic_json(report, report_path)
    metadata = {
        "model_version": CHALLENGE_MODEL_VERSION,
        "feature_version": FEATURE_SCHEMA_VERSION,
        "log_feature_version": log_feature_version,
        "target_tenant": TARGET_TENANT,
        "sha256": digest,
        "feature_count": len(feature_names),
        "horizons_minutes": list(HORIZONS_MINUTES),
        "training_runs": len(splits["train"]),
        "validation_runs": len(splits["validation"]),
        "test_runs": len(splits["test"]),
        "evaluation_scope": (
            "variable proxy episodes are diagnostic only; official labels were not accessed"
        ),
        "architecture_version": MODEL_ARCHITECTURE_VERSION,
    }
    metadata_path = model.with_suffix(model.suffix + ".metadata.json")
    _atomic_json(metadata, metadata_path)
    manifest = {
        "model_version": CHALLENGE_MODEL_VERSION,
        "outputs": {
            "splits.json": _sha256_file(split_path),
            "training-report.json": _sha256_file(report_path),
            model.name: digest,
            model.with_suffix(model.suffix + ".sha256").name: _sha256_file(
                model.with_suffix(model.suffix + ".sha256")
            ),
            metadata_path.name: _sha256_file(metadata_path),
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    return {**report, "output_directory": str(output), "model_path": str(model)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the causal Zikit early-warning model on phase-2 features"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-runs", type=int, default=54)
    args = parser.parse_args()
    report = train_challenge_model(
        args.input,
        args.output,
        args.model,
        seed=args.seed,
        expected_runs=args.expected_runs,
    )
    print(
        json.dumps(
            {
                "model_version": report["model_version"],
                "feature_count": report["feature_count"],
                "splits": report["splits"]["counts"],
                "model_path": report["model_path"],
                "output_directory": report["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
