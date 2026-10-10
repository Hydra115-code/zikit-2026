from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from app.ml.model import FEATURES, load_bundle
from app.ml.train import validate_features


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(labels).astype(bool)
    y_pred = np.asarray(predictions).astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
    }


def detection_delay_seconds(
    timestamps: pd.Series,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float | None:
    times = pd.to_datetime(timestamps, utc=True, errors="raise")
    y_true = np.asarray(labels).astype(bool)
    y_pred = np.asarray(predictions).astype(bool)
    anomaly_indexes = np.flatnonzero(y_true)
    if anomaly_indexes.size == 0:
        return None
    start = int(anomaly_indexes[0])
    detections = np.flatnonzero(y_pred & (np.arange(len(y_pred)) >= start))
    if detections.size == 0:
        return None
    return max(0.0, float((times.iloc[int(detections[0])] - times.iloc[start]).total_seconds()))


def evaluate(
    frame: pd.DataFrame,
    labels: np.ndarray,
    model_path: Path,
    *,
    timestamp_column: str | None = None,
) -> dict[str, float | None]:
    bundle = load_bundle(model_path)
    if bundle is None:
        raise FileNotFoundError(model_path)
    clean = validate_features(frame)
    y_true = np.asarray(labels)[: len(clean)].astype(bool)
    y_pred = bundle.predict(clean.loc[:, FEATURES].to_numpy(dtype=float))
    result: dict[str, float | None] = binary_metrics(y_true, y_pred)
    if timestamp_column:
        result["detection_delay_seconds"] = detection_delay_seconds(
            frame.loc[clean.index, timestamp_column], y_true, y_pred
        )
    return result


def compare_detectors(
    labels: np.ndarray,
    statistical_predictions: np.ndarray,
    hybrid_predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        "statistical": binary_metrics(labels, statistical_predictions),
        "hybrid": binary_metrics(labels, hybrid_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/isolation_forest.joblib"))
    parser.add_argument("--label", default="is_anomaly")
    parser.add_argument("--timestamp-column")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    print(
        json.dumps(
            evaluate(
                frame,
                frame[args.label].to_numpy(),
                args.model,
                timestamp_column=args.timestamp_column,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
