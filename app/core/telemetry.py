from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from app.schemas import MetricsSnapshot, TelemetrySample


class TelemetryStore:
    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._samples: dict[str, deque[TelemetrySample]] = defaultdict(lambda: deque(maxlen=window_size))
        self._traffic_arrival_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._baseline_latency: dict[str, float] = {}
        self._baseline_rate: dict[str, float] = {}
        self._last_observed_at: dict[str, float] = {}
        self._sample_arrival_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record(self, sample: TelemetrySample) -> None:
        self._samples[sample.service].append(sample)
        observed_at = time.monotonic()
        self._last_observed_at[sample.service] = observed_at
        self._sample_arrival_times[sample.service].append(observed_at)
        if sample.probe_kind == "traffic":
            self._traffic_arrival_times[sample.service].append(observed_at)
        healthy = sample.success and not sample.timed_out
        if healthy:
            old = self._baseline_latency.get(sample.service)
            self._baseline_latency[sample.service] = (
                sample.latency_ms if old is None else 0.98 * old + 0.02 * sample.latency_ms
            )

    def samples(self, service: str) -> list[TelemetrySample]:
        return list(self._samples[service])

    def sample_age_seconds(self, service: str) -> float | None:
        observed_at = self._last_observed_at.get(service)
        return None if observed_at is None else max(0.0, time.monotonic() - observed_at)

    def recent_sample_count(self, service: str, max_age_seconds: float) -> int:
        now = time.monotonic()
        return sum(
            now - observed_at <= max_age_seconds
            for observed_at in self._sample_arrival_times[service]
        )

    def snapshot(self, service: str) -> MetricsSnapshot:
        samples = self.samples(service)
        if not samples:
            return MetricsSnapshot(
                service=service,
                sample_count=0,
                latency_p50_ms=0,
                latency_p95_ms=0,
                error_rate=0,
                timeout_rate=0,
                request_rate=0,
                availability=1,
                latency_ratio=1,
                request_rate_ratio=1,
            )
        latencies = np.asarray([sample.latency_ms for sample in samples], dtype=float)
        errors = sum(not sample.success for sample in samples)
        timeouts = sum(sample.timed_out for sample in samples)
        # Request rate represents client/gateway traffic only. Mixing health probes
        # and verification requests creates artificial rate spikes and can make an
        # otherwise healthy service look anomalous to the ML model.
        arrivals = self._traffic_arrival_times[service]
        elapsed = max(arrivals[-1] - arrivals[0], 1e-6) if len(arrivals) > 1 else 1.0
        rate = len(arrivals) / elapsed if arrivals else 0.0
        rate_ready = len(arrivals) >= self.window_size
        if math.isfinite(rate) and rate_ready:
            old_rate = self._baseline_rate.get(service)
            self._baseline_rate[service] = rate if old_rate is None else 0.98 * old_rate + 0.02 * rate
        p95 = float(np.percentile(latencies, 95))
        baseline_latency = self._baseline_latency.get(service, p95)
        baseline_rate = self._baseline_rate.get(service, rate)
        return MetricsSnapshot(
            service=service,
            sample_count=len(samples),
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=p95,
            error_rate=errors / len(samples),
            timeout_rate=timeouts / len(samples),
            request_rate=rate,
            availability=1 - errors / len(samples),
            latency_ratio=p95 / max(baseline_latency, 1),
            request_rate_ratio=rate / max(baseline_rate, 1e-6) if rate_ready else 1,
        )

    def clear(self) -> None:
        self._samples.clear()
        self._traffic_arrival_times.clear()
        self._baseline_latency.clear()
        self._baseline_rate.clear()
        self._last_observed_at.clear()
        self._sample_arrival_times.clear()


def make_sample(
    service: str,
    latency_ms: float,
    status_code: int | None,
    timed_out: bool = False,
    probe_kind: Literal["monitor", "traffic", "verification"] = "monitor",
) -> TelemetrySample:
    success = status_code is not None and 200 <= status_code < 400 and not timed_out
    return TelemetrySample(
        service=service,
        timestamp=datetime.now(UTC),
        latency_ms=latency_ms,
        status_code=status_code,
        timed_out=timed_out,
        success=success,
        probe_kind=probe_kind,
    )
