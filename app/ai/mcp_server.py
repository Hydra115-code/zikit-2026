from __future__ import annotations

from app.ai.tools import SentinelToolRegistry


def create_mcp_server(registry: SentinelToolRegistry):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "zikit-sentinel",
        instructions=(
            "Read-only operational evidence for Zikit Sentinel. Observations, timings and "
            "incident explanations come from persisted evidence. Never claim that calling "
            "a tool executed a mitigation."
        ),
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        max_sessions=100,
    )
    for function in registry.functions:
        server.tool()(function)
    return server
