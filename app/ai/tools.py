from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.explanations import build_incident_explanation
from app.challenge.mcp_evidence import KaikuEvidenceReader
from app.core.telemetry import TelemetryStore
from app.database import EventRecord, session_factory
from app.simulator.state import SimulatorState


class SentinelToolRegistry:
    """Read-only tool implementation shared by Gemini and the MCP server."""

    def __init__(
        self,
        simulator: SimulatorState,
        telemetry: TelemetryStore,
        challenge_features_path: Path | None = None,
    ) -> None:
        self.simulator = simulator
        self.telemetry = telemetry
        self.kaiku = KaikuEvidenceReader(challenge_features_path)
        self.call_counts: Counter[str] = Counter()

    def _called(self, name: str) -> None:
        self.call_counts[name] += 1

    def list_services(self) -> list[str]:
        """List every HTTP provider service monitored by Sentinel."""
        self._called("list_services")
        return ["primary", "backup"]

    def get_service_health(self, service: str) -> dict[str, Any]:
        """Return the observed health metrics for one service."""
        self._called("get_service_health")
        return self.telemetry.snapshot(service).model_dump(mode="json")

    def get_service_metrics(self, service: str) -> dict[str, Any]:
        """Return the current rolling telemetry window for one service."""
        self._called("get_service_metrics")
        return self.telemetry.snapshot(service).model_dump(mode="json")

    def get_recent_errors(self, service: str) -> list[dict[str, Any]]:
        """Return failed observations from the current telemetry window."""
        self._called("get_recent_errors")
        return [sample.model_dump(mode="json") for sample in self.telemetry.samples(service) if not sample.success]

    def get_incident_timeline(self, incident_id: str) -> list[dict[str, Any]]:
        """Return the persisted, ordered event timeline for one incident."""
        self._called("get_incident_timeline")
        with session_factory()() as session:
            events = session.scalars(
                select(EventRecord)
                .where(EventRecord.incident_id == incident_id)
                .order_by(EventRecord.monotonic_seconds)
            )
            return [
                {"type": event.event_type, "timestamp": event.timestamp.isoformat(), "details": event.details}
                for event in events
            ]

    def explain_incident(self, incident_id: str) -> dict[str, Any]:
        """Explain an incident in plain language using only persisted evidence."""
        self._called("explain_incident")
        return build_incident_explanation(incident_id)

    def get_active_provider(self) -> str:
        """Return the provider currently receiving gateway traffic."""
        self._called("get_active_provider")
        return self.simulator.active_provider

    def get_available_runbooks(self) -> list[str]:
        """List the safe runbooks allowlisted by the policy engine."""
        self._called("get_available_runbooks")
        return ["failover_to_backup"]

    def get_kaiku_metric_window(self, incident_id: str) -> dict[str, Any]:
        """Return the causal Kaiku metric window ending at the alert timestamp."""
        self._called("get_kaiku_metric_window")
        return self.kaiku.metric_window(incident_id)

    def get_kaiku_correlated_logs(self, incident_id: str) -> dict[str, Any]:
        """Return hash-addressed Kaiku log indicators available before the alert."""
        self._called("get_kaiku_correlated_logs")
        return self.kaiku.correlated_logs(incident_id)

    def get_kaiku_business_context(self, incident_id: str) -> dict[str, Any]:
        """Return public Kaiku business context and flags evaluated at alert time."""
        self._called("get_kaiku_business_context")
        return self.kaiku.business_context(incident_id)

    def get_kaiku_allowed_runbooks(self, incident_id: str) -> dict[str, Any]:
        """Return only the recommendation-only runbook bound to a Kaiku alert."""
        self._called("get_kaiku_allowed_runbooks")
        return self.kaiku.allowed_runbooks(incident_id)

    def get_kaiku_verification(self, incident_id: str) -> dict[str, Any]:
        """Return verification requirements and observed status without executing them."""
        self._called("get_kaiku_verification")
        return self.kaiku.verification(incident_id)

    @property
    def functions(self) -> list[Any]:
        return [
            self.list_services,
            self.get_service_health,
            self.get_service_metrics,
            self.get_recent_errors,
            self.get_incident_timeline,
            self.explain_incident,
            self.get_active_provider,
            self.get_available_runbooks,
            self.get_kaiku_metric_window,
            self.get_kaiku_correlated_logs,
            self.get_kaiku_business_context,
            self.get_kaiku_allowed_runbooks,
            self.get_kaiku_verification,
        ]
