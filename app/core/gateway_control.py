from __future__ import annotations

from typing import Protocol

import httpx

from app.config import Settings
from app.simulator.state import SimulatorState


class GatewayController(Protocol):
    async def synchronize(self) -> str: ...

    async def switch_provider(self, provider: str) -> None: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


class LocalGatewayController:
    def __init__(self, simulator: SimulatorState) -> None:
        self.simulator = simulator

    async def synchronize(self) -> str:
        return self.simulator.active_provider

    async def switch_provider(self, provider: str) -> None:
        await self.simulator.switch_provider(provider)

    async def reset(self) -> None:
        await self.simulator.switch_provider("primary")

    async def close(self) -> None:
        return None


class HttpGatewayController:
    def __init__(
        self,
        simulator: SimulatorState,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.simulator = simulator
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-Lab-Token": token},
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    async def synchronize(self) -> str:
        response = await self.client.get("/lab/status")
        response.raise_for_status()
        provider = response.json().get("active_provider")
        if provider not in {"primary", "backup"}:
            raise RuntimeError("gateway returned an invalid active provider")
        await self.simulator.switch_provider(provider)
        return provider

    async def switch_provider(self, provider: str) -> None:
        if provider not in {"primary", "backup"}:
            raise ValueError("unknown provider")
        response = await self.client.post("/lab/switch", json={"provider": provider})
        response.raise_for_status()
        active = response.json().get("active_provider")
        if active != provider:
            raise RuntimeError("gateway did not confirm the requested provider")
        await self.simulator.switch_provider(provider)

    async def reset(self) -> None:
        response = await self.client.post("/lab/reset")
        response.raise_for_status()
        active = response.json().get("active_provider")
        if active != "primary":
            raise RuntimeError("gateway did not confirm reset to primary")
        await self.simulator.switch_provider("primary")

    async def close(self) -> None:
        await self.client.aclose()


def configured_gateway_controller(
    settings: Settings,
    simulator: SimulatorState,
) -> GatewayController:
    if settings.gateway_control_url is None:
        return LocalGatewayController(simulator)
    assert settings.gateway_control_token is not None
    return HttpGatewayController(
        simulator,
        settings.gateway_control_url,
        settings.gateway_control_token,
        timeout_seconds=settings.runbook_timeout_seconds,
    )
