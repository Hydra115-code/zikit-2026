from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURE_VERSION = "3"
FEATURES = ("latency_ratio", "error_rate", "timeout_rate", "request_rate_ratio", "availability")
ROBUST_SCALE_FLOORS = (0.03, 0.01, 0.005, 0.04, 0.01)


@dataclass
class ModelBundle:
    estimator: Any
    threshold: float
    feature_names: tuple[str, ...] = FEATURES
    feature_version: str = FEATURE_VERSION
    training_rows: int = 0
    robust_center: tuple[float, ...] | None = None
    robust_lower_scale: tuple[float, ...] | None = None
    robust_upper_scale: tuple[float, ...] | None = None
    isolation_center: float | None = None
    isolation_upper_scale: float | None = None

    def score_components(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.asarray(rows, dtype=float)
        isolation = -self.estimator.score_samples(matrix)
        if (
            self.robust_center is None
            or self.robust_lower_scale is None
            or self.robust_upper_scale is None
            or self.isolation_center is None
            or self.isolation_upper_scale is None
        ):
            empty = np.zeros_like(matrix)
            return isolation, isolation, empty
        isolation_z = (isolation - self.isolation_center) / max(self.isolation_upper_scale, 1e-9)
        centers = np.asarray(self.robust_center, dtype=float)
        lower_scales = np.asarray(self.robust_lower_scale, dtype=float)
        upper_scales = np.asarray(self.robust_upper_scale, dtype=float)
        deviations = matrix - centers
        feature_components = np.where(
            deviations < 0, -deviations / lower_scales, deviations / upper_scales
        )
        feature_distance = np.max(feature_components, axis=1)
        combined = np.maximum(isolation_z, feature_distance)
        return combined, isolation_z, feature_components

    def score(self, rows: np.ndarray) -> np.ndarray:
        combined, _, _ = self.score_components(rows)
        return combined

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return self.score(rows) >= self.threshold


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    digest = sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="utf-8")


def load_bundle(path: Path, *, trusted_root: Path | None = None) -> ModelBundle | None:
    """Load a locally trusted bundle only after path and digest validation.

    Joblib uses pickle internally. Callers must never point this function at an
    uploaded or otherwise untrusted artifact.
    """
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
    if not digest_path.exists():
        raise ValueError("model digest is missing")
    expected = digest_path.read_text(encoding="utf-8").strip()
    actual = sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("model digest mismatch")
    bundle = joblib.load(path)
    if (
        not isinstance(bundle, ModelBundle)
        or bundle.feature_version != FEATURE_VERSION
        or tuple(bundle.feature_names) != FEATURES
        or not isinstance(bundle.estimator, IsolationForest)
        or bundle.robust_center is None
        or bundle.robust_lower_scale is None
        or bundle.robust_upper_scale is None
        or len(bundle.robust_center) != len(FEATURES)
        or len(bundle.robust_lower_scale) != len(FEATURES)
        or len(bundle.robust_upper_scale) != len(FEATURES)
    ):
        raise ValueError("incompatible model bundle")
    return bundle


def snapshot_vector(snapshot: Any) -> np.ndarray:
    return np.asarray([[float(getattr(snapshot, name)) for name in FEATURES]], dtype=float)
