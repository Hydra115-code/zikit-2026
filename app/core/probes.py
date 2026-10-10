from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

import httpx

from app.config import Settings
from app.core.telemetry import make_sample
from app.schemas import TelemetrySample
from app.simulator.state import SimulatorState

logger = logging.getLogger(__name__)

ProbePurpose = Literal["monitor", "traffic", "verification"]


@dataclass(frozen=True)
class ProbeObservation:
    sample: TelemetrySample
    purpose: ProbePurpose
    transport: str
    error_type: str | None = None


@dataclass
class ProbeRuntimeStatus:
    transport: str
    total_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0
    last_probe_at: datetime | None = None
    last_error_type: str | None = None

    def as_public_dict(self) -> dict[str, object]:
        return asdict(self)


class HttpProbeClient:
    """Execute every service observation through HTTPX.

    The local simulator uses an HTTPX transport too. The running FastAPI app replaces
    it with ASGITransport so probes traverse the actual provider and gateway routes.
    Absolute endpoint URLs can be configured later without changing SentinelService.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout_seconds: float,
        primary_url: str,
        backup_url: str,
        gateway_url: str,
        transport: str,
    ) -> None:
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.gateway_url = gateway_url
        self.status = ProbeRuntimeStatus(transport=transport)
        self._closed = False

    async def probe_provider(self, provider: str, *, purpose: ProbePurpose = "monitor") -> ProbeObservation:
        if provider not in {"primary", "backup"}:
            raise ValueError("unknown provider")
        url = self.primary_url if provider == "primary" else self.backup_url
        return await self._probe(url, service=provider, purpose=purpose)

    async def probe_gateway(self, *, purpose: ProbePurpose) -> ProbeObservation:
        return await self._probe(self.gateway_url, service=None, purpose=purpose)

    async def _probe(
        self,
        url: str,
        *,
        service: str | None,
        purpose: ProbePurpose,
    ) -> ProbeObservation:
        loop = asyncio.get_running_loop()
        started = loop.time()
        error_type: str | None = None
        status_code: int | None = None
        timed_out = False
        observed_service = service or "unknown"
        try:
            response = await asyncio.wait_for(
                self.client.get(url),
                timeout=self.timeout_seconds,
            )
            status_code = response.status_code
            if service is None:
                try:
                    provider = response.json().get("provider")
                except (ValueError, AttributeError):
                    provider = None
                if provider not in {"primary", "backup"}:
                    error_type = "invalid_gateway_response"
                    status_code = None
                else:
                    observed_service = provider
            if status_code is not None and status_code >= 400:
                error_type = f"http_status_{status_code}"
        except (TimeoutError, httpx.TimeoutException):
            timed_out = True
            error_type = "timeout"
        except httpx.ConnectError:
            error_type = "connection_error"
        except httpx.HTTPError as exc:
            error_type = type(exc).__name__

        elapsed_ms = (loop.time() - started) * 1000
        sample = make_sample(
            observed_service,
            elapsed_ms,
            status_code,
            timed_out=timed_out,
            probe_kind=purpose,
        )
        self._record_status(sample, error_type)
        if error_type:
            logger.warning(
                "HTTP probe failed",
                extra={
                    "service": observed_service,
                    "purpose": purpose,
                    "transport": self.status.transport,
                    "error_type": error_type,
                },
            )
        return ProbeObservation(
            sample=sample,
            purpose=purpose,
            transport=self.status.transport,
            error_type=error_type,
        )

    def _record_status(self, sample: TelemetrySample, error_type: str | None) -> None:
        self.status.total_requests += 1
        self.status.last_probe_at = sample.timestamp
        if error_type is not None:
            self.status.last_error_type = error_type
        if not sample.success:
            self.status.failed_requests += 1
        if sample.timed_out:
            self.status.timed_out_requests += 1

    async def close(self) -> None:
        if not self._closed:
            await self.client.aclose()
            self._closed = True


def internal_probe_client(settings: Settings, simulator: SimulatorState) -> HttpProbeClient:
    """HTTPX fallback for direct service tests that do not create a FastAPI app."""

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/internal/providers/"):
            provider = path.rsplit("/", 1)[-1]
            if provider not in {"primary", "backup"}:
                return httpx.Response(404, json={"error": "unknown provider"})
        elif path == "/demo/gateway":
            provider = simulator.active_provider
        else:
            return httpx.Response(404, json={"error": "unknown probe endpoint"})
        status, latency = await simulator.response(provider)
        return httpx.Response(status, json={"provider": provider, "latency_ms": latency})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://sentinel.internal",
        follow_redirects=False,
    )
    return HttpProbeClient(
        client,
        timeout_seconds=settings.probe_timeout_seconds,
        primary_url="/internal/providers/primary",
        backup_url="/internal/providers/backup",
        gateway_url="/demo/gateway",
        transport="httpx-mock",
    )


def configured_probe_client(
    settings: Settings,
    *,
    app_transport: httpx.AsyncBaseTransport,
) -> HttpProbeClient:
    external = settings.primary_provider_url is not None
    if external:
        client = httpx.AsyncClient(follow_redirects=False)
    else:
        client = httpx.AsyncClient(
            transport=app_transport,
            base_url="http://sentinel.internal",
            follow_redirects=False,
        )
    return HttpProbeClient(
        client,
        timeout_seconds=settings.probe_timeout_seconds,
        primary_url=settings.primary_provider_url or "/internal/providers/primary",
        backup_url=settings.backup_provider_url or "/internal/providers/backup",
        gateway_url=settings.gateway_url or "/demo/gateway",
        transport="network-http" if external else "asgi-http",
    )
