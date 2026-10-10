from __future__ import annotations

import hmac
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas import Scenario
from app.simulator.state import SimulatorState


ProviderName = Literal["primary", "backup"]


@dataclass
class ProviderCounters:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class LabProviderRuntime:
    def __init__(self, name: ProviderName, instance_id: str) -> None:
        self.name = name
        self.instance_id = instance_id
        self.simulator = SimulatorState()
        self.counters = ProviderCounters()
        self.started_at = datetime.now(UTC)

    async def serve(self, endpoint: str) -> JSONResponse:
        started = time.monotonic()
        response_status, intended_latency_ms = await self.simulator.response(self.name)
        observed_latency_ms = (time.monotonic() - started) * 1000
        self.counters.total_requests += 1
        if 200 <= response_status < 400:
            self.counters.successful_requests += 1
        else:
            self.counters.failed_requests += 1
        return JSONResponse(
            {
                "provider": self.name,
                "instance_id": self.instance_id,
                "endpoint": endpoint,
                "status": "ok" if response_status == 200 else "error",
                "intended_latency_ms": intended_latency_ms,
                "observed_latency_ms": observed_latency_ms,
                "served_at": datetime.now(UTC).isoformat(),
            },
            status_code=response_status,
        )

    def public_state(self) -> dict[str, object]:
        fault = self.simulator.fault_for(self.name)
        active_fault = None
        if fault is not None:
            active_fault = {
                **fault.scenario.model_dump(mode="json"),
                "remaining_seconds": max(
                    0.0,
                    fault.scenario.duration_seconds
                    - (time.monotonic() - fault.started_monotonic),
                ),
            }
        return {
            "provider": self.name,
            "instance_id": self.instance_id,
            "started_at": self.started_at,
            "pid": os.getpid(),
            "counters": self.counters.__dict__,
            "active_fault": active_fault,
        }


def create_provider_app(
    *,
    provider_name: ProviderName = "primary",
    admin_token: str = "development-lab-token",
    instance_id: str | None = None,
) -> FastAPI:
    if provider_name not in {"primary", "backup"}:
        raise ValueError("provider_name must be primary or backup")
    if len(admin_token) < 12:
        raise ValueError("lab admin token must contain at least 12 characters")
    runtime = LabProviderRuntime(provider_name, instance_id or f"{provider_name}-local")
    app = FastAPI(
        title=f"Zikit Local Lab Provider ({provider_name})",
        version="0.1.0",
        description="Controlled local workload. It is not a remote production machine.",
    )
    app.state.runtime = runtime

    async def require_lab_admin(x_lab_token: str | None = Header(default=None)) -> None:
        if x_lab_token is None or not hmac.compare_digest(x_lab_token, admin_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid X-Lab-Token required",
            )

    @app.get("/live", tags=["health"])
    async def live() -> dict[str, object]:
        return {
            "status": "alive",
            "provider": runtime.name,
            "instance_id": runtime.instance_id,
        }

    @app.get("/health", tags=["workload"])
    async def health() -> JSONResponse:
        return await runtime.serve("health")

    @app.get("/work", tags=["workload"])
    async def work() -> JSONResponse:
        return await runtime.serve("work")

    @app.get("/lab/status", dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def lab_status() -> dict[str, object]:
        return runtime.public_state()

    @app.post("/lab/fault", status_code=202, dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def inject_fault(scenario: Scenario) -> dict[str, object]:
        if scenario.service != runtime.name:
            raise HTTPException(
                status_code=422,
                detail=f"scenario service must be {runtime.name}",
            )
        await runtime.simulator.inject(scenario)
        return {
            "status": "fault_injected",
            "provider": runtime.name,
            "scenario": scenario.model_dump(mode="json"),
            "injected_at": datetime.now(UTC),
        }

    @app.post("/lab/reset", dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def reset_fault() -> dict[str, object]:
        await runtime.simulator.stop_fault(runtime.name)
        return {
            "status": "reset",
            "provider": runtime.name,
            "reset_at": datetime.now(UTC),
        }

    return app


def _from_environment() -> FastAPI:
    provider_name = os.getenv("LAB_PROVIDER_NAME", "primary")
    if provider_name not in {"primary", "backup"}:
        raise RuntimeError("LAB_PROVIDER_NAME must be primary or backup")
    return create_provider_app(
        provider_name=provider_name,
        admin_token=os.getenv("LAB_ADMIN_TOKEN", "development-lab-token"),
        instance_id=os.getenv("LAB_INSTANCE_ID"),
    )


app = _from_environment()
