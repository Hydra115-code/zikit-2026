from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from app.schemas import FaultType, Scenario


@dataclass
class FaultState:
    scenario: Scenario
    started_monotonic: float = field(default_factory=time.monotonic)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.scenario.seed)

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.started_monotonic >= self.scenario.duration_seconds


class SimulatorState:
    def __init__(self) -> None:
        self.active_provider = "primary"
        self.faults: dict[str, FaultState] = {}
        self._lock = asyncio.Lock()

    async def switch_provider(self, provider: str) -> None:
        if provider not in {"primary", "backup"}:
            raise ValueError("unknown provider")
        async with self._lock:
            self.active_provider = provider

    async def inject(self, scenario: Scenario) -> None:
        async with self._lock:
            self.faults[scenario.service] = FaultState(scenario)

    async def stop_fault(self, service: str) -> None:
        async with self._lock:
            self.faults.pop(service, None)

    async def reset(self) -> None:
        async with self._lock:
            self.active_provider = "primary"
            self.faults.clear()

    def fault_for(self, provider: str) -> FaultState | None:
        fault = self.faults.get(provider)
        if fault and fault.expired:
            self.faults.pop(provider, None)
            return None
        return fault

    async def response(self, provider: str) -> tuple[int, float]:
        base_latency_ms = 20.0 if provider == "primary" else 35.0
        fault = self.fault_for(provider)
        if fault is None or fault.scenario.fault_type == FaultType.NONE:
            await asyncio.sleep(base_latency_ms / 1000)
            return 200, base_latency_ms
        scenario = fault.scenario
        elapsed = time.monotonic() - fault.started_monotonic
        latency = base_latency_ms
        status = 200
        if scenario.fault_type == FaultType.CONSTANT_LATENCY:
            latency += scenario.latency_ms
        elif scenario.fault_type == FaultType.PROGRESSIVE_LATENCY:
            progress = min(1.0, elapsed / scenario.duration_seconds)
            latency += scenario.latency_ms * progress
        elif scenario.fault_type == FaultType.HTTP_ERROR:
            if fault.rng.random() < scenario.error_rate:
                status = scenario.status_code
        elif scenario.fault_type == FaultType.INTERMITTENT:
            if fault.rng.random() < scenario.error_rate:
                status = scenario.status_code
            else:
                latency += scenario.latency_ms
        elif scenario.fault_type == FaultType.UNAVAILABLE:
            status = 503
        elif scenario.fault_type == FaultType.TIMEOUT:
            latency += scenario.latency_ms
        await asyncio.sleep(latency / 1000)
        return status, latency
