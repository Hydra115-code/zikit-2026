from __future__ import annotations

from typing import Protocol

import httpx

from app.config import Settings
from app.schemas import Scenario
from app.simulator.state import SimulatorState


class FaultAlreadyActiveError(RuntimeError):
    pass


class FaultController(Protocol):
    async def is_active(self, service: str) -> bool: ...

    async def inject(self, scenario: Scenario) -> None: ...

    async def stop(self, service: str) -> None: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


class LocalFaultController:
    def __init__(self, simulator: SimulatorState) -> None:
        self.simulator = simulator

    async def is_active(self, service: str) -> bool:
        return self.simulator.fault_for(service) is not None

    async def inject(self, scenario: Scenario) -> None:
        await self.simulator.inject(scenario)

    async def stop(self, service: str) -> None:
        await self.simulator.stop_fault(service)

    async def reset(self) -> None:
        for service in ("primary", "backup"):
            await self.simulator.stop_fault(service)

    async def close(self) -> None:
        return None


class HttpFaultController:
    def __init__(
        self,
        primary_url: str,
        backup_url: str,
        token: str,
        *,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_urls = {
            "primary": primary_url.rstrip("/"),
            "backup": backup_url.rstrip("/"),
        }
        self.client = httpx.AsyncClient(
            headers={"X-Lab-Token": token},
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    def _provider_url(self, service: str) -> str:
        try:
            return self.provider_urls[service]
        except KeyError as exc:
            raise ValueError("unknown provider") from exc

    async def is_active(self, service: str) -> bool:
        response = await self.client.get(f"{self._provider_url(service)}/lab/status")
        response.raise_for_status()
        return response.json().get("active_fault") is not None

    async def inject(self, scenario: Scenario) -> None:
        response = await self.client.post(
            f"{self._provider_url(scenario.service)}/lab/fault",
            json=scenario.model_dump(mode="json"),
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "fault_injected" or body.get("provider") != scenario.service:
            raise RuntimeError("provider did not confirm fault injection")

    async def stop(self, service: str) -> None:
        response = await self.client.post(f"{self._provider_url(service)}/lab/reset")
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "reset" or body.get("provider") != service:
            raise RuntimeError("provider did not confirm fault reset")

    async def reset(self) -> None:
        for service in ("primary", "backup"):
            await self.stop(service)

    async def close(self) -> None:
        await self.client.aclose()


def configured_fault_controller(
    settings: Settings,
    simulator: SimulatorState,
) -> FaultController:
    if settings.primary_provider_control_url is None:
        return LocalFaultController(simulator)
    assert settings.backup_provider_control_url is not None
    assert settings.provider_control_token is not None
    return HttpFaultController(
        settings.primary_provider_control_url,
        settings.backup_provider_control_url,
        settings.provider_control_token,
        timeout_seconds=settings.runbook_timeout_seconds,
    )
