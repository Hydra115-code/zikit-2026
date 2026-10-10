from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from app.challenge.contracts import TARGET_TENANT
from app.challenge.features import FEATURE_SCHEMA_VERSION
from app.challenge.normal_behavior import (
    NORMAL_BEHAVIOR_VERSION,
    NormalBehaviorModel,
)


CHALLENGE_MODEL_VERSION = "challenge-kaiku-early-warning-v6"

COMPONENT_WEIGHTS = {
    "supervised": 0.50,
    "normal_behavior": 0.25,
    "isolation": 0.12,
    "baseline": 0.08,
    "change": 0.05,
}


def _scaled(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    width = max(float(upper) - float(lower), 1e-9)
    return np.clip((values - float(lower)) / width, 0.0, 1.0)


@dataclass
class ChallengeModelBundle:
    classifiers: dict[int, HistGradientBoostingClassifier]
    isolation_estimator: IsolationForest
    feature_names: tuple[str, ...]
    isolation_feature_names: tuple[str, ...]
    imputation_values: tuple[float, ...]
    component_ranges: dict[str, tuple[float, float]]
    horizon_weights: dict[int, float]
    normal_behavior_model: NormalBehaviorModel
    model_version: str = CHALLENGE_MODEL_VERSION
    feature_version: str = FEATURE_SCHEMA_VERSION
    seed: int = 42
    training_rows: int = 0
    target_tenant: str = TARGET_TENANT
    log_feature_version: str | None = None

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(feature for feature in self.feature_names if feature not in frame.columns)
        if missing:
            raise ValueError(f"model input is missing features: {missing}")
        if "tenant" not in frame.columns:
            raise ValueError("model input is missing tenant")
        tenants = set(frame["tenant"].astype(str).unique())
        if tenants != {self.target_tenant}:
            raise ValueError(
                f"model input must contain only {self.target_tenant}: {sorted(tenants)}"
            )
        if self.log_feature_version is not None:
            if "log_feature_version" not in frame.columns:
                raise ValueError("model input is missing log_feature_version")
            versions = set(frame["log_feature_version"].astype(str).unique())
            if versions != {self.log_feature_version}:
                raise ValueError(
                    "model input has an incompatible log feature version: "
                    f"{sorted(versions)}"
                )

        columns = [
            pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype="float32")
            for feature in self.feature_names
        ]
        matrix = np.column_stack(columns).astype("float32", copy=False)
        medians = np.asarray(self.imputation_values, dtype="float32")
        unavailable = ~np.isfinite(matrix)
        if unavailable.any():
            matrix[unavailable] = np.take(medians, np.nonzero(unavailable)[1])
        return matrix

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = self._matrix(frame)
        indices = {name: index for index, name in enumerate(self.feature_names)}
        isolation_indices = [indices[name] for name in self.isolation_feature_names]
        isolation_raw = -self.isolation_estimator.score_samples(matrix[:, isolation_indices])
        isolation_risk = _scaled(
            isolation_raw, *self.component_ranges["isolation"]
        )

        baseline_indices = [
            index
            for index, name in enumerate(self.feature_names)
            if name.endswith("_past_robust_z_1440m")
            and not name.startswith("errores_")
        ]
        change_indices = [
            index
            for index, name in enumerate(self.feature_names)
            if "_trend_per_min_" in name
            or ("_relative_change_" in name and not name.startswith("errores_"))
        ]
        baseline_raw = np.maximum(matrix[:, baseline_indices], 0).max(axis=1)
        change_raw = np.maximum(matrix[:, change_indices], 0).max(axis=1)
        baseline_risk = _scaled(baseline_raw, *self.component_ranges["baseline"])
        change_risk = _scaled(change_raw, *self.component_ranges["change"])
        normal_behavior_raw = self.normal_behavior_model.raw_deviation(frame)
        normal_behavior_risk = _scaled(
            normal_behavior_raw, *self.component_ranges["normal_behavior"]
        )

        result: dict[str, np.ndarray] = {
            "isolation_risk": isolation_risk.astype("float32"),
            "baseline_risk": baseline_risk.astype("float32"),
            "change_risk": change_risk.astype("float32"),
            "normal_behavior_risk": normal_behavior_risk.astype("float32"),
        }
        weighted_horizons: list[np.ndarray] = []
        for horizon in sorted(self.classifiers):
            risk = self.classifiers[horizon].predict_proba(matrix)[:, 1]
            result[f"risk_{horizon}m"] = risk.astype("float32")
            weighted_horizons.append(risk * self.horizon_weights[horizon])
        supervised = np.max(np.column_stack(weighted_horizons), axis=1)
        result["supervised_risk"] = supervised.astype("float32")
        result["risk_score"] = np.clip(
            COMPONENT_WEIGHTS["supervised"] * supervised
            + COMPONENT_WEIGHTS["normal_behavior"] * normal_behavior_risk
            + COMPONENT_WEIGHTS["isolation"] * isolation_risk
            + COMPONENT_WEIGHTS["baseline"] * baseline_risk
            + COMPONENT_WEIGHTS["change"] * change_risk,
            0.0,
            1.0,
        ).astype("float32")
        return pd.DataFrame(result, index=frame.index)


def save_challenge_bundle(bundle: ChallengeModelBundle, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, path)
    digest = sha256(path.read_bytes()).hexdigest()
    digest_path = path.with_suffix(path.suffix + ".sha256")
    temporary_digest = digest_path.with_name(f".{digest_path.name}.tmp")
    temporary_digest.write_text(f"{digest}\n", encoding="utf-8")
    os.replace(temporary_digest, digest_path)
    return digest


def load_challenge_bundle(
    path: Path, *, trusted_root: Path | None = None
) -> ChallengeModelBundle | None:
    if not path.exists():
        return None
    resolved = path.resolve()
    if trusted_root is not None:
        root = trusted_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("model path is outside the trusted model directory")
    if path.suffix != ".joblib" or not path.is_file():
        raise ValueError("model must be a .joblib file")
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise ValueError("model digest is missing")
    expected = digest_path.read_text(encoding="utf-8").strip()
    actual = sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("model digest mismatch")

    bundle: Any = joblib.load(path)
    compatible = (
        isinstance(bundle, ChallengeModelBundle)
        and bundle.model_version == CHALLENGE_MODEL_VERSION
        and bundle.feature_version == FEATURE_SCHEMA_VERSION
        and bundle.target_tenant == TARGET_TENANT
        and (
            bundle.log_feature_version is None
            or isinstance(bundle.log_feature_version, str)
        )
        and len(bundle.feature_names) == len(bundle.imputation_values)
        and set(bundle.classifiers) == set(bundle.horizon_weights)
        and all(
            isinstance(estimator, HistGradientBoostingClassifier)
            for estimator in bundle.classifiers.values()
        )
        and isinstance(bundle.isolation_estimator, IsolationForest)
        and isinstance(bundle.normal_behavior_model, NormalBehaviorModel)
        and bundle.normal_behavior_model.version == NORMAL_BEHAVIOR_VERSION
        and set(bundle.isolation_feature_names).issubset(bundle.feature_names)
        and {"isolation", "baseline", "change", "normal_behavior"}.issubset(
            bundle.component_ranges
        )
    )
    if not compatible:
        raise ValueError("incompatible challenge model bundle")
    return bundle
