from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy import select

from app.challenge.mcp_evidence import KaikuEvidenceReader
from app.challenge.remediation_catalog import RemediationCatalog
from app.config import Settings
from app.database import ChallengeRemediationRecord, session_factory


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class ChallengeRemediationToolRegistry:
    """MCP mutation surface restricted to configured, reversible demo runbooks."""

    def __init__(
        self,
        catalog: RemediationCatalog,
        feature_directory: Path,
        *,
        minute_seconds: float,
    ) -> None:
        self.catalog = catalog
        self.evidence = KaikuEvidenceReader(feature_directory)
        self.minute_seconds = minute_seconds

    def execute_challenge_runbook(
        self,
        remediation_id: str,
        incident_id: str,
        runbook_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Execute one preauthorized reversible runbook on controlled infrastructure."""
        runbook = self.catalog.require(runbook_id)
        with session_factory()() as session:
            record = session.scalar(
                select(ChallengeRemediationRecord).where(
                    ChallengeRemediationRecord.id == remediation_id,
                    ChallengeRemediationRecord.incident_id == incident_id,
                )
            )
            if record is None:
                raise LookupError("remediation record not found")
            if record.runbook_id != runbook.id:
                raise ValueError("requested runbook does not match the persisted decision")
            audit = dict(record.audit or {})
            previous = audit.get("mcp_execution")
            if isinstance(previous, dict) and previous.get("idempotency_key") == idempotency_key:
                return {**previous, "idempotent_replay": True}

            summary = self.evidence.metric_window(incident_id)["summary"]
            baseline = {
                metric: values.get("latest") if values.get("latest") is not None else values.get("mean")
                for metric, values in summary.items()
            }
            if any(baseline.get(metric) is None for metric in runbook.effects):
                raise ValueError("causal baseline is incomplete for the selected runbook")
            target = {
                metric: round(float(baseline[metric]) * ratio, 6)
                for metric, ratio in runbook.effects.items()
            }
            executed_at = datetime.now(UTC)
            result = {
                "state": "completed",
                "environment": "controlled_infrastructure_demo",
                "remediation_id": remediation_id,
                "incident_id": incident_id,
                "runbook_id": runbook.id,
                "action": runbook.action,
                "rollback_action": runbook.rollback_action,
                "reversible": runbook.reversible,
                "preauthorized": runbook.preauthorized,
                "baseline": baseline,
                "target": target,
                "executed_at": executed_at.isoformat(),
                "idempotency_key": idempotency_key,
                "idempotent_replay": False,
            }
            audit["mcp_execution"] = result
            record.audit = audit
            record.updated_at = executed_at
            session.commit()
            return result

    def observe_challenge_controlled_state(
        self,
        remediation_id: str,
        incident_id: str,
    ) -> dict[str, Any]:
        """Observe the controlled target after an MCP-executed runbook."""
        with session_factory()() as session:
            record = session.scalar(
                select(ChallengeRemediationRecord).where(
                    ChallengeRemediationRecord.id == remediation_id,
                    ChallengeRemediationRecord.incident_id == incident_id,
                )
            )
            if record is None:
                raise LookupError("remediation record not found")
            execution = (record.audit or {}).get("mcp_execution")
            if not isinstance(execution, dict):
                raise LookupError("runbook has not been executed")
            executed_at = datetime.fromisoformat(str(execution["executed_at"]))
            elapsed_seconds = max(0.0, (datetime.now(UTC) - _utc(executed_at)).total_seconds())
            controlled_minute = min(
                10_000,
                int(elapsed_seconds / self.minute_seconds) + 1,
            )
            rolled_back = bool((record.audit or {}).get("mcp_rollback"))
            return {
                "environment": "controlled_infrastructure_demo",
                "remediation_id": remediation_id,
                "incident_id": incident_id,
                "runbook_id": execution["runbook_id"],
                "controlled_minute": controlled_minute,
                "metrics": execution["baseline"] if rolled_back else execution["target"],
                "rolled_back": rolled_back,
                "observed_at": datetime.now(UTC).isoformat(),
                "clock_domain": "accelerated_controlled_infrastructure_time",
            }

    def rollback_challenge_runbook(
        self,
        remediation_id: str,
        incident_id: str,
        runbook_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Reverse one previously executed controlled runbook."""
        runbook = self.catalog.require(runbook_id)
        with session_factory()() as session:
            record = session.scalar(
                select(ChallengeRemediationRecord).where(
                    ChallengeRemediationRecord.id == remediation_id,
                    ChallengeRemediationRecord.incident_id == incident_id,
                )
            )
            if record is None:
                raise LookupError("remediation record not found")
            audit = dict(record.audit or {})
            previous = audit.get("mcp_rollback")
            if isinstance(previous, dict) and previous.get("idempotency_key") == idempotency_key:
                return {**previous, "idempotent_replay": True}
            if not isinstance(audit.get("mcp_execution"), dict):
                raise ValueError("cannot roll back a runbook that was not executed")
            result = {
                "state": "completed",
                "environment": "controlled_infrastructure_demo",
                "remediation_id": remediation_id,
                "incident_id": incident_id,
                "runbook_id": runbook.id,
                "action": runbook.rollback_action,
                "rolled_back_at": datetime.now(UTC).isoformat(),
                "idempotency_key": idempotency_key,
                "idempotent_replay": False,
            }
            audit["mcp_rollback"] = result
            record.audit = audit
            record.updated_at = datetime.now(UTC)
            session.commit()
            return result

    @property
    def functions(self) -> list[Any]:
        return [
            self.execute_challenge_runbook,
            self.observe_challenge_controlled_state,
            self.rollback_challenge_runbook,
        ]


def create_remediation_mcp_server(registry: ChallengeRemediationToolRegistry):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "zikit-sentinel-controlled-remediation",
        instructions=(
            "Controlled demonstration infrastructure only. Execute exactly one configured, "
            "preauthorized and reversible runbook. Never accept commands, scripts, paths, or "
            "runbooks outside the server-side catalog."
        ),
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        max_sessions=25,
    )
    for function in registry.functions:
        server.tool()(function)
    return server


def _tool_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return decoded
    raise RuntimeError("MCP execution tool returned no structured object")


class MCPRemediationClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        headers = (
            {"Authorization": f"Bearer {self.settings.mcp_execution_token}"}
            if self.settings.mcp_execution_token
            else {}
        )
        timeout = httpx.Timeout(self.settings.mcp_client_timeout_seconds)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
            async with streamable_http_client(
                self.settings.mcp_execution_url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    if result.isError:
                        raise RuntimeError(f"MCP execution tool failed: {name}")
                    return _tool_payload(result)


class InProcessRemediationMCPClient:
    """Exercise the same MCP tool boundary without a network listener in audits."""

    def __init__(self, registry: ChallengeRemediationToolRegistry) -> None:
        self.registry = registry

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {function.__name__: function for function in self.registry.functions}
        function = allowed.get(name)
        if function is None:
            raise ValueError("MCP tool is outside the execution allowlist")
        return function(**arguments)
