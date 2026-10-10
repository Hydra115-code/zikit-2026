from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
import logging

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.ai.agent import investigate
from app.ai.mcp_server import create_mcp_server
from app.ai.mcp_transport import MCPBearerTokenMiddleware
from app.ai.tools import SentinelToolRegistry
from app.challenge.api import router as challenge_router
from app.challenge.live_replay import ChallengeLiveReplayManager
from app.challenge.remediation import ChallengeRemediationManager
from app.challenge.remediation_catalog import load_remediation_catalog
from app.challenge.remediation_mcp import (
    ChallengeRemediationToolRegistry,
    create_remediation_mcp_server,
)
from app.challenge.service import ChallengeReplayService
from app.config import get_settings
from app.core.service import SentinelService
from app.core.probes import configured_probe_client
from app.core.operations import EventStreamCapacity
from app.core.evidence_store import EvidenceBundleStore
from app.core.resources import ResourceMonitor
from app.database import configure_database, init_database
from app.logging import configure_logging
from app.routes.api import require_investigation_read, router as api_router
from app.schemas import InvestigationRequest
from app.simulator.routes import router as simulator_router
from app.simulator.state import SimulatorState

logger = logging.getLogger(__name__)


def create_app(*, database_url: str | None = None, start_monitor: bool | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging()
    configure_database(database_url)
    simulator = SimulatorState()
    sentinel = SentinelService(settings, simulator)
    resources = ResourceMonitor(settings, sentinel.repository)
    challenge_replay_service = ChallengeReplayService(
        settings.challenge_features_path,
        settings.challenge_model_path,
        settings.challenge_policy_path,
        require_all_tenants=settings.challenge_require_all_tenants,
        critical_confidence_threshold=settings.challenge_critical_confidence_threshold,
    )
    challenge_live_replay_manager = ChallengeLiveReplayManager(
        challenge_replay_service
    )
    tool_registry = SentinelToolRegistry(
        simulator,
        sentinel.telemetry,
        settings.challenge_features_path,
    )
    remediation_catalog, remediation_catalog_sha256 = load_remediation_catalog(
        settings.challenge_remediation_catalog_path
    )
    remediation_tool_registry = ChallengeRemediationToolRegistry(
        remediation_catalog,
        settings.challenge_features_path,
        minute_seconds=settings.challenge_remediation_minute_seconds,
    )
    try:
        mcp_server = create_mcp_server(tool_registry)
    except ImportError:
        mcp_server = None
        logger.warning("MCP SDK is not installed; monitoring and mitigation remain available")
    mcp_http_app = (
        mcp_server.streamable_http_app()
        if mcp_server is not None and settings.mcp_http_enabled
        else None
    )
    try:
        remediation_mcp_server = create_remediation_mcp_server(
            remediation_tool_registry
        )
    except ImportError:
        remediation_mcp_server = None
        logger.warning(
            "MCP SDK is not installed; challenge detection remains available and "
            "autonomous remediation will escalate safely"
        )
    remediation_mcp_app = (
        remediation_mcp_server.streamable_http_app()
        if remediation_mcp_server is not None
        and settings.challenge_remediation_enabled
        else None
    )
    challenge_remediation_manager = ChallengeRemediationManager(
        settings,
        remediation_catalog,
        remediation_catalog_sha256,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_database()
        reconciled = await sentinel.repository.reconcile_interrupted_state()
        if any(reconciled.values()):
            logger.warning("reconciled interrupted runtime state: %s", reconciled)
        probes = configured_probe_client(
            settings,
            app_transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        )
        await sentinel.replace_probe_client(probes)
        async with AsyncExitStack() as stack:
            if mcp_http_app is not None:
                await stack.enter_async_context(
                    mcp_http_app.router.lifespan_context(mcp_http_app)
                )
            if remediation_mcp_app is not None:
                await stack.enter_async_context(
                    remediation_mcp_app.router.lifespan_context(remediation_mcp_app)
                )
            try:
                if start_monitor is not False:
                    await sentinel.start()
                    await resources.start()
                await challenge_remediation_manager.start()
                yield
            finally:
                await app.state.challenge_live_replay_manager.shutdown()
                await challenge_remediation_manager.stop()
                await resources.stop()
                await sentinel.stop()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.simulator = simulator
    app.state.sentinel = sentinel
    app.state.resource_monitor = resources
    app.state.tool_registry = tool_registry
    app.state.mcp_server = mcp_server
    app.state.remediation_mcp_server = remediation_mcp_server
    app.state.remediation_tool_registry = remediation_tool_registry
    app.state.challenge_remediation_manager = challenge_remediation_manager
    app.state.monitor_requested = start_monitor is not False
    app.state.event_stream_capacity = EventStreamCapacity(
        settings.event_stream_max_connections
    )
    app.state.evidence_store = EvidenceBundleStore(
        settings.evidence_bundle_path,
        settings.evidence_checksum_path,
        max_bytes=settings.evidence_max_bundle_bytes,
    )
    app.state.challenge_replay_service = challenge_replay_service
    app.state.challenge_live_replay_manager = challenge_live_replay_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token", "Last-Event-ID", "Authorization"],
    )
    app.include_router(api_router)
    app.include_router(challenge_router)
    app.include_router(simulator_router)
    if mcp_http_app is not None:
        app.mount(
            "/mcp",
            MCPBearerTokenMiddleware(mcp_http_app, settings.mcp_read_token),
            name="mcp",
        )
    if remediation_mcp_app is not None:
        app.mount(
            "/mcp-execution",
            MCPBearerTokenMiddleware(
                remediation_mcp_app,
                settings.mcp_execution_token,
            ),
            name="mcp-execution",
        )

    @app.post(
        "/api/incidents/{incident_id}/investigate",
        dependencies=[Depends(require_investigation_read)],
        tags=["ai"],
    )
    async def investigate_incident(
        incident_id: str,
        request: Request,
        payload: InvestigationRequest | None = None,
    ):
        try:
            return await investigate(
                incident_id,
                settings,
                request.app.state.tool_registry,
                question=payload.question if payload else InvestigationRequest().question,
                allow_external_ai=payload.allow_external_ai if payload else False,
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app


app = create_app()
