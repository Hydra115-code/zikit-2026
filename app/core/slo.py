from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.schemas import MetricsSnapshot

EVENT_SLO_AT_RISK = "SLO_AT_RISK"
EVENT_SLO_BREACHED = "SLO_BREACHED"
EVENT_CUSTOMER_IMPACT = "CUSTOMER_IMPACT"
EVENT_CUSTOMER_COMPLAINT_SIMULATED = "CUSTOMER_COMPLAINT_SIMULATED"


@dataclass(frozen=True)
class SloTransition:
    event_type: str
    details: dict[str, Any]


@dataclass
class _ServiceSloState:
    breach_streak: int = 0
    emitted: set[str] = field(default_factory=set)
    impact_started_monotonic: float | None = None


class SloTracker:
    """Translate rolling HTTP metrics into auditable customer-impact milestones."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._states: dict[str, _ServiceSloState] = {}

    def evaluate(
        self,
        metrics: MetricsSnapshot,
        *,
        now_monotonic: float | None = None,
    ) -> list[SloTransition]:
        if metrics.sample_count < self.settings.slo_min_samples:
            return []
        now = time.monotonic() if now_monotonic is None else now_monotonic
        state = self._states.setdefault(metrics.service, _ServiceSloState())
        transitions: list[SloTransition] = []
        risk_reasons = self._risk_reasons(metrics)
        breach_reasons = self._breach_reasons(metrics)

        if risk_reasons and EVENT_SLO_AT_RISK not in state.emitted:
            transitions.append(self._transition(EVENT_SLO_AT_RISK, metrics, risk_reasons))
            state.emitted.add(EVENT_SLO_AT_RISK)

        if breach_reasons:
            state.breach_streak += 1
        else:
            state.breach_streak = 0

        if (
            state.breach_streak >= self.settings.slo_breach_persistence_windows
            and EVENT_SLO_BREACHED not in state.emitted
        ):
            transitions.append(self._transition(EVENT_SLO_BREACHED, metrics, breach_reasons))
            state.emitted.add(EVENT_SLO_BREACHED)

        if (
            state.breach_streak >= self.settings.customer_impact_persistence_windows
            and EVENT_CUSTOMER_IMPACT not in state.emitted
        ):
            transitions.append(self._transition(EVENT_CUSTOMER_IMPACT, metrics, breach_reasons))
            state.emitted.add(EVENT_CUSTOMER_IMPACT)
            state.impact_started_monotonic = now

        if (
            state.impact_started_monotonic is not None
            and now - state.impact_started_monotonic >= self.settings.customer_complaint_delay_seconds
            and EVENT_CUSTOMER_COMPLAINT_SIMULATED not in state.emitted
        ):
            details = self._details(metrics, breach_reasons)
            details["delay_after_customer_impact_seconds"] = self.settings.customer_complaint_delay_seconds
            transitions.append(SloTransition(EVENT_CUSTOMER_COMPLAINT_SIMULATED, details))
            state.emitted.add(EVENT_CUSTOMER_COMPLAINT_SIMULATED)

        return transitions

    def status(self, service: str) -> dict[str, Any]:
        state = self._states.get(service, _ServiceSloState())
        return {
            "service": service,
            "breach_streak": state.breach_streak,
            "at_risk": EVENT_SLO_AT_RISK in state.emitted,
            "breached": EVENT_SLO_BREACHED in state.emitted,
            "customer_impact": EVENT_CUSTOMER_IMPACT in state.emitted,
            "customer_complaint_simulated": EVENT_CUSTOMER_COMPLAINT_SIMULATED in state.emitted,
        }

    def reset(self, service: str | None = None) -> None:
        if service is None:
            self._states.clear()
        else:
            self._states.pop(service, None)

    def _risk_reasons(self, metrics: MetricsSnapshot) -> list[str]:
        ratio = self.settings.slo_at_risk_ratio
        reasons: list[str] = []
        if metrics.latency_p95_ms >= self.settings.slo_latency_p95_ms * ratio:
            reasons.append("latency_p95_near_objective")
        if metrics.error_rate >= self.settings.slo_error_rate * ratio:
            reasons.append("error_rate_near_objective")
        return reasons

    def _breach_reasons(self, metrics: MetricsSnapshot) -> list[str]:
        reasons: list[str] = []
        if metrics.latency_p95_ms >= self.settings.slo_latency_p95_ms:
            reasons.append("latency_p95_exceeded")
        if metrics.error_rate >= self.settings.slo_error_rate:
            reasons.append("error_rate_exceeded")
        return reasons

    def _transition(
        self,
        event_type: str,
        metrics: MetricsSnapshot,
        reasons: list[str],
    ) -> SloTransition:
        return SloTransition(event_type, self._details(metrics, reasons))

    def _details(self, metrics: MetricsSnapshot, reasons: list[str]) -> dict[str, Any]:
        return {
            "service": metrics.service,
            "reasons": reasons,
            "sample_count": metrics.sample_count,
            "latency_p95_ms": metrics.latency_p95_ms,
            "error_rate": metrics.error_rate,
            "availability": metrics.availability,
            "objectives": {
                "latency_p95_ms": self.settings.slo_latency_p95_ms,
                "error_rate": self.settings.slo_error_rate,
            },
        }
