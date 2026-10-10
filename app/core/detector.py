from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.ml.model import ModelBundle, snapshot_vector
from app.schemas import MetricsSnapshot, TelemetrySample


@dataclass(frozen=True)
class Detection:
    anomalous: bool
    actionable: bool
    confirmed: bool
    severity: str
    reasons: tuple[str, ...]
    statistical_anomalous: bool = False
    ml_anomalous: bool = False
    ml_score: float | None = None


class StatisticalDetector:
    """Threshold + EWMA detector with persistence and recovery hysteresis."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ewma_latency: dict[str, float] = {}
        self._streak: dict[str, int] = {}
        self._healthy_streak: dict[str, int] = {}

    def evaluate(
        self,
        metrics: MetricsSnapshot,
        *,
        recent_samples: list[TelemetrySample] | None = None,
    ) -> Detection:
        service = metrics.service
        previous = self._ewma_latency.get(service, metrics.latency_p95_ms)
        ewma = 0.35 * metrics.latency_p95_ms + 0.65 * previous
        self._ewma_latency[service] = ewma
        reasons: list[str] = []
        if metrics.sample_count >= 3 and ewma > self.settings.latency_threshold_ms:
            reasons.append(f"latency_ewma_ms={ewma:.1f}")
        if metrics.error_rate >= self.settings.error_rate_threshold:
            reasons.append(f"error_rate={metrics.error_rate:.3f}")
        if metrics.timeout_rate >= self.settings.timeout_rate_threshold:
            reasons.append(f"timeout_rate={metrics.timeout_rate:.3f}")
        if recent_samples is not None and reasons:
            recent = recent_samples[-max(5, self.settings.anomaly_persistence_windows * 2) :]
            recent_latencies = sorted(sample.latency_ms for sample in recent)
            p95_index = min(
                len(recent_latencies) - 1,
                int(0.95 * len(recent_latencies)),
            )
            recent_p95 = recent_latencies[p95_index] if recent_latencies else 0.0
            recent_error_rate = (
                sum(not sample.success for sample in recent) / len(recent) if recent else 0.0
            )
            recent_timeout_rate = (
                sum(sample.timed_out for sample in recent) / len(recent) if recent else 0.0
            )
            reasons = [
                reason
                for reason in reasons
                if (
                    reason.startswith("latency_ewma_ms=")
                    and recent_p95 > self.settings.latency_threshold_ms
                )
                or (
                    reason.startswith("error_rate=")
                    and recent_error_rate >= self.settings.error_rate_threshold
                )
                or (
                    reason.startswith("timeout_rate=")
                    and recent_timeout_rate >= self.settings.timeout_rate_threshold
                )
            ]
        anomalous = bool(reasons)
        if anomalous:
            self._streak[service] = self._streak.get(service, 0) + 1
            self._healthy_streak[service] = 0
        else:
            self._healthy_streak[service] = self._healthy_streak.get(service, 0) + 1
            if self._healthy_streak[service] >= 2:
                self._streak[service] = 0
        confirmed = self._streak.get(service, 0) >= self.settings.anomaly_persistence_windows
        severity = "critical" if metrics.availability < 0.5 else "warning"
        return Detection(
            anomalous,
            anomalous,
            confirmed,
            severity,
            tuple(reasons),
            statistical_anomalous=anomalous,
        )

    def reset(self, service: str) -> None:
        self._streak.pop(service, None)
        self._healthy_streak.pop(service, None)
        self._ewma_latency.pop(service, None)


class HybridDetector:
    def __init__(self, settings: Settings, model: ModelBundle | None = None) -> None:
        self.statistical = StatisticalDetector(settings)
        self.model = model
        self.persistence = settings.anomaly_persistence_windows
        self.minimum_ml_samples = settings.telemetry_window_size
        self._ml_streak: dict[str, int] = {}

    @property
    def ml_enabled(self) -> bool:
        return self.model is not None

    def evaluate(
        self,
        metrics: MetricsSnapshot,
        *,
        recent_samples: list[TelemetrySample] | None = None,
    ) -> Detection:
        statistical = self.statistical.evaluate(
            metrics,
            recent_samples=recent_samples,
        )
        ml_anomalous = False
        ml_score: float | None = None
        if self.model is not None and metrics.sample_count >= self.minimum_ml_samples:
            vector = snapshot_vector(metrics)
            ml_score = float(self.model.score(vector)[0])
            ml_anomalous = ml_score >= self.model.threshold
        if ml_anomalous:
            self._ml_streak[metrics.service] = self._ml_streak.get(metrics.service, 0) + 1
        else:
            self._ml_streak[metrics.service] = 0
        ml_confirmed = self._ml_streak.get(metrics.service, 0) >= self.persistence
        recent_support = recent_samples is None or statistical.anomalous or (
            metrics.request_rate_ratio
            <= self.statistical.settings.ml_action_max_request_rate_ratio
        )
        ml_actionable = ml_anomalous and recent_support and (
            metrics.latency_ratio >= self.statistical.settings.ml_action_min_latency_ratio
            or metrics.error_rate >= self.statistical.settings.ml_action_min_error_rate
            or metrics.timeout_rate > 0
            or metrics.request_rate_ratio <= self.statistical.settings.ml_action_max_request_rate_ratio
        )
        reasons = list(statistical.reasons)
        if ml_anomalous and ml_score is not None:
            reasons.append(f"isolation_forest_score={ml_score:.6f}")
        return Detection(
            anomalous=statistical.anomalous or ml_anomalous,
            actionable=statistical.anomalous or ml_actionable,
            confirmed=statistical.confirmed or (ml_confirmed and ml_actionable),
            severity=statistical.severity,
            reasons=tuple(reasons),
            statistical_anomalous=statistical.anomalous,
            ml_anomalous=ml_anomalous,
            ml_score=ml_score,
        )

    def reset(self, service: str) -> None:
        self.statistical.reset(service)
        self._ml_streak.pop(service, None)
