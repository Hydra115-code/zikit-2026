from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import Settings
from app.schemas import MetricsSnapshot


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    runbook: str | None = None


class PolicyEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_action = float("-inf")
        self._action_in_progress = False

    def decide_failover(
        self,
        active_provider: str,
        backup: MetricsSnapshot,
        *,
        backup_sample_age_seconds: float | None,
        backup_recent_sample_count: int,
        automatic: bool = True,
    ) -> PolicyDecision:
        if not self.settings.auto_mitigation_enabled and automatic:
            return PolicyDecision(False, "automatic mitigation disabled")
        if active_provider != "primary":
            return PolicyDecision(False, "primary is not active")
        if self._action_in_progress:
            return PolicyDecision(False, "another mitigation is in progress")
        if time.monotonic() - self._last_action < self.settings.mitigation_cooldown_seconds:
            return PolicyDecision(False, "mitigation cooldown active")
        if (
            backup.sample_count < self.settings.backup_min_samples
            or backup_recent_sample_count < self.settings.backup_min_samples
        ):
            return PolicyDecision(False, "backup has insufficient health samples")
        if (
            backup_sample_age_seconds is None
            or backup_sample_age_seconds > self.settings.backup_max_sample_age_seconds
        ):
            return PolicyDecision(False, "backup health evidence is stale")
        if (
            backup.availability < self.settings.backup_min_availability
            or backup.latency_p95_ms > self.settings.recovery_max_p95_ms
        ):
            return PolicyDecision(False, "backup health check failed")
        return PolicyDecision(True, "persistent degradation and healthy backup", "failover_to_backup")

    def action_started(self) -> None:
        self._action_in_progress = True

    def action_finished(self) -> None:
        self._action_in_progress = False
        self._last_action = time.monotonic()

    def reset(self) -> None:
        self._last_action = float("-inf")
        self._action_in_progress = False
