from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.core.detector import HybridDetector
from app.core.executor import RunbookExecutor
from app.core.fault_control import FaultAlreadyActiveError, configured_fault_controller
from app.core.gateway_control import configured_gateway_controller
from app.core.incidents import (
    EVENT_ACTION_COMPLETED,
    EVENT_ACTION_BLOCKED,
    EVENT_ACTION_STARTED,
    EVENT_FIRST_SIGNAL,
    EVENT_FAULT_INJECTED,
    EVENT_RECOVERY_FAILED,
    EVENT_RECOVERY_VERIFIED,
    IncidentRepository,
)
from app.core.policy import PolicyEngine
from app.core.probes import HttpProbeClient, internal_probe_client
from app.core.slo import SloTracker
from app.core.telemetry import TelemetryStore
from app.core.verifier import RecoveryVerifier
from app.database import ExperimentRecord, session_factory
from app.schemas import ExperimentMode, Scenario, TelemetrySample
from app.ml.model import load_bundle
from app.simulator.state import SimulatorState
from app.simulator.traffic import TrafficGenerator

logger = logging.getLogger(__name__)


class SentinelService:
    def __init__(self, settings: Settings, simulator: SimulatorState) -> None:
        self.settings = settings
        self.simulator = simulator
        self.telemetry = TelemetryStore(settings.telemetry_window_size)
        self.probes = internal_probe_client(settings, simulator)
        model = None
        try:
            model = load_bundle(settings.model_path, trusted_root=Path("models"))
        except (OSError, ValueError) as exc:
            logger.warning("ML model unavailable; using statistical detector: %s", exc)
        self.detector = HybridDetector(settings, model)
        self.policy = PolicyEngine(settings)
        self.slo = SloTracker(settings)
        self.repository = IncidentRepository()
        self.fault_controller = configured_fault_controller(settings, simulator)
        self.gateway_controller = configured_gateway_controller(settings, simulator)
        self.executor = RunbookExecutor(self.policy, self.gateway_controller.switch_provider)
        self.verifier = RecoveryVerifier(settings, self.probe_recovery, expected_service="backup")
        self.traffic_generator = TrafficGenerator(self.probe_gateway, settings.traffic_interval_seconds)
        self._monitor_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._runtime_lock = asyncio.Lock()
        self._first_signal: dict[str, datetime] = {}
        self._experiment_by_service: dict[str, str] = {}
        self._mode_by_experiment: dict[str, ExperimentMode] = {}
        self._mitigation_locks: dict[str, asyncio.Lock] = {}
        self._incident_epoch: dict[str, datetime] = {}

    async def start(self) -> None:
        await self.gateway_controller.synchronize()
        if self.settings.traffic_enabled:
            await self.traffic_generator.start()
        if self.settings.monitor_enabled and self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop(), name="sentinel-monitor")

    async def stop(self) -> None:
        self._stop.set()
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        await self.traffic_generator.stop()
        await self.probes.close()
        await self.fault_controller.close()
        await self.gateway_controller.close()

    async def replace_probe_client(self, probes: HttpProbeClient) -> None:
        previous = self.probes
        self.probes = probes
        await previous.close()

    def register_experiment(self, experiment_id: str, service: str, mode: ExperimentMode) -> None:
        self.slo.reset(service)
        self._experiment_by_service[service] = experiment_id
        self._mode_by_experiment[experiment_id] = mode

    async def begin_experiment(
        self,
        experiment_id: str,
        scenario: Scenario,
        mode: ExperimentMode,
    ) -> None:
        async with self._runtime_lock:
            if await self.fault_controller.is_active(scenario.service):
                raise FaultAlreadyActiveError(
                    f"an experiment is already active for {scenario.service}"
                )
            self.register_experiment(experiment_id, scenario.service, mode)
            injected = False
            try:
                await self.fault_controller.inject(scenario)
                injected = True
                await self.repository.event(
                    EVENT_FAULT_INJECTED,
                    experiment_id=experiment_id,
                    details={
                        "service": scenario.service,
                        "fault_type": scenario.fault_type.value,
                    },
                )
            except Exception:
                if injected:
                    try:
                        await self.fault_controller.stop(scenario.service)
                    except Exception:
                        logger.exception(
                            "failed to compensate provider fault after experiment startup error"
                        )
                self._experiment_by_service.pop(scenario.service, None)
                self._mode_by_experiment.pop(experiment_id, None)
                raise

    async def stop_experiment(self, experiment_id: str, service: str) -> None:
        async with self._runtime_lock:
            await self.fault_controller.stop(service)
            self.slo.reset(service)
            self._experiment_by_service.pop(service, None)
            self._mode_by_experiment.pop(experiment_id, None)

    async def probe_provider(self, provider: str) -> TelemetrySample:
        observation = await self.probes.probe_provider(provider, purpose="monitor")
        sample = observation.sample
        self.telemetry.record(sample)
        await self.repository.record_sample(
            sample,
            active=provider == self.simulator.active_provider,
        )
        return sample

    async def probe_gateway(self, *, purpose: str = "traffic") -> TelemetrySample:
        probe_purpose = "verification" if purpose == "verification" else "traffic"
        observation = await self.probes.probe_gateway(purpose=probe_purpose)
        sample = observation.sample
        if sample.service == "unknown":
            sample = sample.model_copy(update={"service": self.simulator.active_provider})
        self.telemetry.record(sample)
        await self.repository.record_sample(
            sample,
            active=sample.service == self.simulator.active_provider,
        )
        return sample

    async def probe_recovery(self) -> TelemetrySample:
        return await self.probe_gateway(purpose="verification")

    async def monitor_once(self) -> None:
        async with self._runtime_lock:
            await self._monitor_once_unlocked()

    async def _monitor_once_unlocked(self) -> None:
        active = self.simulator.active_provider
        await self.probe_provider(active)
        if active == "primary":
            await self.probe_provider("backup")
        metrics = self.telemetry.snapshot(active)
        experiment_id = self._experiment_by_service.get(active)
        detection = self.detector.evaluate(
            metrics,
            recent_samples=self.telemetry.samples(active),
        )
        if detection.actionable and active not in self._first_signal:
            first_signal = datetime.now(UTC)
            self._first_signal[active] = first_signal
            await self.repository.event(
                EVENT_FIRST_SIGNAL,
                experiment_id=experiment_id,
                details={"service": active, "reasons": list(detection.reasons)},
            )
        for transition in self.slo.evaluate(metrics):
            await self.repository.event(
                transition.event_type,
                experiment_id=experiment_id,
                details=transition.details,
            )
        if not detection.actionable:
            # A model-only outlier without current actionable evidence must not
            # preserve an old first-signal timestamp for a later, unrelated fault.
            self._first_signal.pop(active, None)
            return
        if not detection.confirmed or await self.repository.active_incident(
            active,
            since=self._incident_epoch.get(active),
        ):
            return
        evidence = metrics.model_dump()
        evidence["reasons"] = list(detection.reasons)
        incident = await self.repository.create_incident(
            active,
            detection.severity,
            self._first_signal[active],
            evidence,
            experiment_id,
        )
        mode = self._mode_by_experiment.get(experiment_id or "", ExperimentMode.AUTOMATIC)
        if mode == ExperimentMode.MANUAL:
            await self.repository.set_status(incident.id, "awaiting_approval")
            return
        await self.mitigate(incident.id, experiment_id, automatic=True)

    async def mitigate(self, incident_id: str, experiment_id: str | None, *, automatic: bool = False) -> None:
        lock = self._mitigation_locks.setdefault(incident_id, asyncio.Lock())
        async with lock:
            await self._mitigate_once(incident_id, experiment_id, automatic=automatic)

    async def _mitigate_once(
        self,
        incident_id: str,
        experiment_id: str | None,
        *,
        automatic: bool,
    ) -> None:
        status = await self.repository.incident_status(incident_id)
        if status not in {"detected", "awaiting_approval"}:
            return
        backup = await self.probe_provider("backup")
        backup_metrics = self.telemetry.snapshot("backup")
        backup_age = self.telemetry.sample_age_seconds("backup")
        backup_recent_samples = self.telemetry.recent_sample_count(
            "backup",
            self.settings.backup_max_sample_age_seconds,
        )
        decision = self.policy.decide_failover(
            self.simulator.active_provider,
            backup_metrics,
            backup_sample_age_seconds=backup_age,
            backup_recent_sample_count=backup_recent_samples,
            automatic=automatic,
        )
        if not backup.success or not decision.allowed or decision.runbook is None:
            await self.repository.set_status(incident_id, "mitigation_blocked")
            reason = decision.reason if backup.success else "backup probe failed"
            await self.repository.event(
                EVENT_ACTION_BLOCKED,
                incident_id=incident_id,
                experiment_id=experiment_id,
                details={
                    "reason": reason,
                    "backup_sample_count": backup_metrics.sample_count,
                    "backup_recent_sample_count": backup_recent_samples,
                    "backup_sample_age_seconds": backup_age,
                    "backup_availability": backup_metrics.availability,
                    "backup_p95_latency_ms": backup_metrics.latency_p95_ms,
                },
            )
            await self.repository.event(
                EVENT_RECOVERY_FAILED,
                incident_id=incident_id,
                experiment_id=experiment_id,
                details={"reason": reason},
            )
            await self._complete_experiment(experiment_id, "failed")
            return
        await self.repository.set_status(incident_id, "mitigating")
        action_id = await self.repository.action_started(incident_id, decision.runbook)
        await self.repository.event(EVENT_ACTION_STARTED, incident_id=incident_id, experiment_id=experiment_id, details={"runbook": decision.runbook})
        execution = await self.executor.execute(
            decision.runbook,
            timeout_seconds=self.settings.runbook_timeout_seconds,
            idempotency_key=f"{incident_id}:{decision.runbook}",
        )
        await self.repository.action_finished(
            action_id,
            execution.state,
            {
                "details": execution.details,
                "idempotency_key": f"{incident_id}:{decision.runbook}",
                "idempotent_replay": execution.idempotent_replay,
            },
        )
        if execution.state == "completed":
            await self.repository.set_active_provider(self.simulator.active_provider)
        await self.repository.event(
            EVENT_ACTION_COMPLETED,
            incident_id=incident_id,
            experiment_id=experiment_id,
            details={"state": execution.state, "details": execution.details},
        )
        if execution.state != "completed":
            await self.repository.event(
                EVENT_RECOVERY_FAILED,
                incident_id=incident_id,
                experiment_id=experiment_id,
                details={"reason": "runbook did not complete", "state": execution.state},
            )
            await self.repository.set_status(incident_id, "recovery_failed")
            await self._complete_experiment(experiment_id, "failed")
            return
        verification = await self.verifier.verify()
        event_type = EVENT_RECOVERY_VERIFIED if verification.recovered else EVENT_RECOVERY_FAILED
        await self.repository.event(
            event_type,
            incident_id=incident_id,
            experiment_id=experiment_id,
            details={
                "availability": verification.availability,
                "error_rate": verification.error_rate,
                "p95_latency_ms": verification.p95_latency_ms,
                "samples": verification.samples,
                "consecutive_successes": verification.consecutive_successes,
                "stable": verification.stable,
                "provider_consistent": verification.provider_consistent,
                "verification_duration_seconds": verification.duration_seconds,
            },
        )
        status = "recovered" if verification.recovered else "recovery_failed"
        await self.repository.set_status(incident_id, status)
        await self._complete_experiment(experiment_id, "completed" if verification.recovered else "failed")
        if verification.recovered:
            self.detector.reset("primary")

    async def _complete_experiment(self, experiment_id: str | None, status: str) -> None:
        if not experiment_id:
            return
        result = await self.repository.calculate_result(experiment_id)
        with session_factory()() as session:
            experiment = session.get(ExperimentRecord, experiment_id)
            if experiment:
                experiment.status = status
                experiment.completed_at = datetime.now(UTC)
                experiment.result = result
                session.commit()

    async def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("monitor iteration failed")
            await asyncio.sleep(self.settings.monitor_interval_seconds)

    async def reset_runtime(self) -> None:
        traffic_was_running = self.traffic_generator.status.running
        if traffic_was_running:
            await self.traffic_generator.stop()
        async with self._runtime_lock:
            await self.gateway_controller.reset()
            await self.fault_controller.reset()
            await self.simulator.reset()
            self.telemetry.clear()
            self._first_signal.clear()
            self._experiment_by_service.clear()
            self._mode_by_experiment.clear()
            reset_at = datetime.now(UTC)
            self._incident_epoch = {
                "primary": reset_at,
                "backup": reset_at,
            }
            self.slo.reset()
            self.detector.reset("primary")
            self.detector.reset("backup")
            self.policy.reset()
            self.executor.reset()
            self.traffic_generator.reset_counts()
            await self.repository.set_active_provider("primary")
        if traffic_was_running:
            await self.traffic_generator.start()


def new_experiment_id() -> str:
    return str(uuid4())
