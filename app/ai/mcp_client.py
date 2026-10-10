from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.config import Settings


GEMINI_INVESTIGATION_TOOLS = {
    "explain_incident",
    "get_incident_timeline",
    "get_available_runbooks",
    "get_kaiku_metric_window",
    "get_kaiku_correlated_logs",
    "get_kaiku_business_context",
    "get_kaiku_allowed_runbooks",
    "get_kaiku_verification",
}

INCIDENT_SCOPED_TOOLS = {
    "explain_incident",
    "get_incident_timeline",
    "get_kaiku_metric_window",
    "get_kaiku_correlated_logs",
    "get_kaiku_business_context",
    "get_kaiku_allowed_runbooks",
    "get_kaiku_verification",
}


class TracedMCPClientSession(ClientSession):
    """ClientSession-compatible proxy that records safe MCP call metadata."""

    def __init__(
        self,
        delegate: ClientSession,
        incident_id: str,
        allowed_tools: set[str] | None = None,
    ) -> None:
        # Google GenAI requires an actual ClientSession instance. The proxy delegates
        # the only two methods used by its MCP adapter and deliberately records no
        # arguments, results, bearer tokens, or incident evidence.
        self._delegate = delegate
        self._incident_id = incident_id
        self._allowed_tools = allowed_tools or GEMINI_INVESTIGATION_TOOLS
        self.discovered_tools: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []

    async def list_tools(self, *args: Any, **kwargs: Any):
        result = await self._delegate.list_tools(*args, **kwargs)
        allowed_tools = [
            tool for tool in result.tools if tool.name in self._allowed_tools
        ]
        self.discovered_tools = sorted(tool.name for tool in allowed_tools)
        return result.model_copy(update={"tools": allowed_tools})

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *args: Any, **kwargs: Any):
        started = perf_counter()
        succeeded = False
        incident_scoped = name in INCIDENT_SCOPED_TOOLS
        scope_valid = not incident_scoped or (
            arguments is not None and arguments.get("incident_id") == self._incident_id
        )
        try:
            if name not in self._allowed_tools:
                raise ValueError("MCP tool is outside the Gemini investigation allowlist")
            if not scope_valid:
                raise ValueError("MCP incident tool call escaped the requested incident scope")
            result = await self._delegate.call_tool(name, arguments, *args, **kwargs)
            succeeded = not bool(result.isError)
            return result
        finally:
            self.tool_calls.append(
                {
                    "name": name,
                    "success": succeeded,
                    "scope_valid": scope_valid,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                }
            )


@asynccontextmanager
async def open_mcp_session(settings: Settings) -> AsyncIterator[ClientSession]:
    """Open the authenticated MCP Streamable HTTP endpoint used by Gemini."""

    if not settings.mcp_http_enabled:
        raise RuntimeError("MCP HTTP is disabled")
    headers = (
        {"Authorization": f"Bearer {settings.mcp_read_token}"}
        if settings.mcp_read_token
        else {}
    )
    timeout = httpx.Timeout(settings.mcp_client_timeout_seconds)
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
        async with streamable_http_client(
            settings.mcp_internal_url,
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
