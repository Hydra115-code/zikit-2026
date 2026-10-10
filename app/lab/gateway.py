from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


ProviderName = Literal["primary", "backup"]
GatewayFaultMode = Literal[
    "none",
    "switch_unavailable",
    "backup_traffic_unavailable",
    "switch_delayed_failure",
]


class SwitchRequest(BaseModel):
    provider: ProviderName


class GatewayFaultRequest(BaseModel):
    mode: GatewayFaultMode
    delay_seconds: float = 0


def _validate_provider_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("provider URL must be an HTTP(S) origin without credentials or path")
    return value.rstrip("/")


class LabGatewayRuntime:
    def __init__(
        self,
        primary_url: str,
        backup_url: str,
        *,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_urls = {
            "primary": _validate_provider_url(primary_url),
            "backup": _validate_provider_url(backup_url),
        }
        self.timeout_seconds = timeout_seconds
        self.active_provider: ProviderName = "primary"
        self.switch_count = 0
        self.last_switch_at: datetime | None = None
        self.fault_mode: GatewayFaultMode = "none"
        self.fault_delay_seconds = 0.0
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=timeout_seconds,
        )

    async def switch(self, provider: ProviderName) -> bool:
        async with self._lock:
            changed = provider != self.active_provider
            self.active_provider = provider
            if changed:
                self.switch_count += 1
                self.last_switch_at = datetime.now(UTC)
            return changed

    async def reset(self) -> None:
        self.fault_mode = "none"
        self.fault_delay_seconds = 0.0
        await self.switch("primary")

    async def configure_fault(
        self,
        mode: GatewayFaultMode,
        delay_seconds: float,
    ) -> None:
        if delay_seconds < 0 or delay_seconds > 30:
            raise ValueError("gateway fault delay must be between 0 and 30 seconds")
        if mode != "switch_delayed_failure" and delay_seconds != 0:
            raise ValueError("delay_seconds is only valid for switch_delayed_failure")
        if mode == "switch_delayed_failure" and delay_seconds <= 0:
            raise ValueError("switch_delayed_failure requires a positive delay")
        async with self._lock:
            self.fault_mode = mode
            self.fault_delay_seconds = delay_seconds

    async def forward(self, upstream_path: str) -> JSONResponse:
        provider = self.active_provider
        if (
            self.fault_mode == "backup_traffic_unavailable"
            and provider == "backup"
            and upstream_path == "/work"
        ):
            return JSONResponse(
                {
                    "provider": provider,
                    "gateway_status": "controlled_lab_fault",
                    "error_type": "backup_traffic_unavailable",
                    "served_at": datetime.now(UTC).isoformat(),
                },
                status_code=503,
            )
        url = f"{self.provider_urls[provider]}{upstream_path}"
        try:
            response = await self._client.get(url)
            try:
                upstream = response.json()
            except ValueError:
                upstream = {"error": "invalid_upstream_json"}
            return JSONResponse(
                {
                    "provider": provider,
                    "gateway_status": "ok" if response.status_code < 400 else "upstream_error",
                    "upstream_status_code": response.status_code,
                    "upstream": upstream,
                    "served_at": datetime.now(UTC).isoformat(),
                },
                status_code=response.status_code,
            )
        except (httpx.TimeoutException, TimeoutError):
            error_type = "upstream_timeout"
        except httpx.ConnectError:
            error_type = "upstream_connection_error"
        except httpx.HTTPError:
            error_type = "upstream_http_error"
        return JSONResponse(
            {
                "provider": provider,
                "gateway_status": "upstream_unavailable",
                "error_type": error_type,
                "served_at": datetime.now(UTC).isoformat(),
            },
            status_code=503,
        )

    def status(self) -> dict[str, object]:
        return {
            "active_provider": self.active_provider,
            "switch_count": self.switch_count,
            "last_switch_at": self.last_switch_at,
            "pid": os.getpid(),
            "fault_mode": self.fault_mode,
            "fault_delay_seconds": self.fault_delay_seconds,
        }

    async def close(self) -> None:
        await self._client.aclose()


def create_gateway_app(
    *,
    primary_url: str = "http://127.0.0.1:9101",
    backup_url: str = "http://127.0.0.1:9102",
    admin_token: str = "development-lab-token",
    timeout_seconds: float = 2.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    if len(admin_token) < 12:
        raise ValueError("lab admin token must contain at least 12 characters")
    if timeout_seconds <= 0 or timeout_seconds > 30:
        raise ValueError("gateway timeout must be between 0 and 30 seconds")
    runtime = LabGatewayRuntime(
        primary_url,
        backup_url,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title="Zikit Local Lab Gateway",
        version="0.1.0",
        description="Controlled local gateway. It is not a remote production service.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    async def require_lab_admin(x_lab_token: str | None = Header(default=None)) -> None:
        if x_lab_token is None or not hmac.compare_digest(x_lab_token, admin_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid X-Lab-Token required",
            )

    @app.get("/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "alive", "component": "gateway"}

    @app.get("/gateway", tags=["traffic"])
    async def gateway() -> JSONResponse:
        return await runtime.forward("/work")

    @app.get("/health", tags=["traffic"])
    async def health() -> JSONResponse:
        return await runtime.forward("/health")

    @app.get("/lab/status", dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def gateway_status() -> dict[str, object]:
        return runtime.status()

    @app.post("/lab/switch", dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def switch_provider(payload: SwitchRequest) -> dict[str, object]:
        if runtime.fault_mode == "switch_unavailable":
            raise HTTPException(status_code=503, detail="controlled gateway switch failure")
        if runtime.fault_mode == "switch_delayed_failure":
            await asyncio.sleep(runtime.fault_delay_seconds)
            raise HTTPException(
                status_code=503,
                detail="controlled delayed gateway switch failure",
            )
        changed = await runtime.switch(payload.provider)
        return {
            "status": "switched" if changed else "already_active",
            "active_provider": runtime.active_provider,
            "changed": changed,
            "switched_at": runtime.last_switch_at,
        }

    @app.post("/lab/fault", dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def configure_gateway_fault(payload: GatewayFaultRequest) -> dict[str, object]:
        try:
            await runtime.configure_fault(payload.mode, payload.delay_seconds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "status": "configured",
            "fault_mode": runtime.fault_mode,
            "fault_delay_seconds": runtime.fault_delay_seconds,
        }

    @app.post("/lab/reset", dependencies=[Depends(require_lab_admin)], tags=["lab-admin"])
    async def reset_gateway() -> dict[str, object]:
        await runtime.reset()
        return {"status": "reset", "active_provider": runtime.active_provider}

    return app


def _from_environment() -> FastAPI:
    return create_gateway_app(
        primary_url=os.getenv("LAB_PRIMARY_URL", "http://127.0.0.1:9101"),
        backup_url=os.getenv("LAB_BACKUP_URL", "http://127.0.0.1:9102"),
        admin_token=os.getenv("LAB_ADMIN_TOKEN", "development-lab-token"),
        timeout_seconds=float(os.getenv("LAB_GATEWAY_TIMEOUT_SECONDS", "2")),
    )


app = _from_environment()
