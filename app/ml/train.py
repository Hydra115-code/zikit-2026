from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.ml.model import FEATURES, ROBUST_SCALE_FLOORS, ModelBundle, save_bundle


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    clean = frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("no valid training rows")
    return clean


def train(frame: pd.DataFrame, contamination: float = 0.03, random_state: int = 42) -> ModelBundle:
    clean = validate_features(frame)
    values = clean.to_numpy(dtype=float)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    model.fit(values)
    centers = np.median(values, axis=0)
    lower_quantiles = np.quantile(values, 0.01, axis=0)
    upper_quantiles = np.quantile(values, 0.99, axis=0)
    floors = np.asarray(ROBUST_SCALE_FLOORS, dtype=float)
    lower_scales = np.maximum(centers - lower_quantiles, floors)
    upper_scales = np.maximum(upper_quantiles - centers, floors)
    isolation_scores = -model.score_samples(values)
    isolation_center = float(np.median(isolation_scores))
    isolation_upper_scale = float(
        max(np.quantile(isolation_scores, 0.99) - isolation_center, 1e-3)
    )
    bundle = ModelBundle(
        model,
        threshold=0,
        training_rows=len(clean),
        robust_center=tuple(float(value) for value in centers),
        robust_lower_scale=tuple(float(value) for value in lower_scales),
        robust_upper_scale=tuple(float(value) for value in upper_scales),
        isolation_center=isolation_center,
        isolation_upper_scale=isolation_upper_scale,
    )
    scores = bundle.score(values)
    threshold = float(np.quantile(scores, 1 - contamination))
    bundle.threshold = threshold
    return bundle


def temporal_split(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if timestamp_column not in frame:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1)")
    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], utc=True, errors="raise")
    ordered = ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    if cut < 2 or len(ordered) - cut < 1:
        raise ValueError("not enough rows for a temporal split")
    train_frame = ordered.iloc[:cut].copy()
    evaluation_frame = ordered.iloc[cut:].copy()
    if train_frame[timestamp_column].max() > evaluation_frame[timestamp_column].min():
        raise ValueError("temporal split overlaps")
    return train_frame, evaluation_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/isolation_forest.joblib"))
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--timestamp-column")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--normal-label", help="Optional boolean anomaly label; only normal training rows are used")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    training_frame = frame
    if args.timestamp_column:
        training_frame, _ = temporal_split(
            frame,
            timestamp_column=args.timestamp_column,
            train_fraction=args.train_fraction,
        )
    if args.normal_label:
        training_frame = training_frame.loc[~training_frame[args.normal_label].astype(bool)]
    bundle = train(training_frame, args.contamination)
    save_bundle(bundle, args.output)
    print(f"saved {args.output} with {bundle.training_rows} training rows")


if __name__ == "__main__":
    main()
