from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.config import Settings
from app.schemas import TelemetrySample


@dataclass(frozen=True)
class VerificationResult:
    recovered: bool
    availability: float
    error_rate: float
    p95_latency_ms: float
    samples: int
    consecutive_successes: int
    stable: bool
    provider_consistent: bool
    duration_seconds: float


class RecoveryVerifier:
    def __init__(
        self,
        settings: Settings,
        probe: Callable[[], Awaitable[TelemetrySample]],
        *,
        expected_service: str | None = None,
    ) -> None:
        self.settings = settings
        self.probe = probe
        self.expected_service = expected_service

    async def verify(self) -> VerificationResult:
        results: list[TelemetrySample] = []
        started = time.monotonic()
        for index in range(self.settings.recovery_verification_requests):
            results.append(await self.probe())
            if index + 1 < self.settings.recovery_verification_requests:
                await asyncio.sleep(self.settings.recovery_verification_interval_seconds)
        ordered = sorted(result.latency_ms for result in results)
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        failures = sum(not result.success for result in results)
        error_rate = failures / len(results)
        p95 = ordered[p95_index]
        consecutive_successes = 0
        for result in reversed(results):
            if not result.success:
                break
            consecutive_successes += 1
        stable = consecutive_successes >= self.settings.recovery_required_consecutive_successes
        provider_consistent = self.expected_service is None or all(
            result.service == self.expected_service for result in results
        )
        recovered = (
            error_rate <= self.settings.recovery_max_error_rate
            and p95 <= self.settings.recovery_max_p95_ms
            and stable
            and provider_consistent
        )
        return VerificationResult(
            recovered,
            1 - error_rate,
            error_rate,
            p95,
            len(results),
            consecutive_successes,
            stable,
            provider_consistent,
            time.monotonic() - started,
        )
