from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.challenge.persistence import ChallengePredictionRepository, acknowledge_alert_once
from app.challenge.remediation import (
    ChallengeRemediationManager,
    GeminiDecisionUnavailable,
    GeminiRunbookDecision,
)
from app.challenge.remediation_catalog import ControlledRunbook, load_remediation_catalog
from app.challenge.remediation_mcp import (
    ChallengeRemediationToolRegistry,
    InProcessRemediationMCPClient,
)
from app.challenge.service import ChallengeReplayService
from app.config import Settings
from app.database import AlertRecord, Base, ChallengePredictionRecord, configure_database, engine, session_factory


AUDIT_VERSION = "challenge-controlled-remediation-audit-v1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContractGeminiSelector:
    async def select(
        self,
        *,
        incident_id: str,
        evidence: dict[str, Any],
        candidates: list[ControlledRunbook],
    ) -> tuple[GeminiRunbookDecision, str]:
        selected = sorted(candidates, key=lambda item: item.id)[0]
        return (
            GeminiRunbookDecision(
                runbook_id=selected.id,
                explanation=(
                    "Doble contractual de Gemini: eligió un ID exacto de la lista segura "
                    f"para {evidence['suspected_incident_type']}."
                ),
            ),
            "gemini-contract-double",
        )


class UnavailableGeminiSelector:
    async def select(self, **_: Any):
        raise GeminiDecisionUnavailable("GeminiInjectedUnavailable")


class VerificationFailureClient:
    def __init__(self, delegate: InProcessRemediationMCPClient) -> None:
        self.delegate = delegate

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.delegate.call(name, arguments)
        if name == "observe_challenge_controlled_state":
            with session_factory()() as session:
                from app.database import ChallengeRemediationRecord

                record = session.get(ChallengeRemediationRecord, arguments["remediation_id"])
                execution = (record.audit or {}).get("mcp_execution", {}) if record else {}
                baseline = execution.get("baseline", {}) if isinstance(execution, dict) else {}
            result = {**result, "metrics": {key: float(value) * 1.25 for key, value in baseline.items()}}
        return result


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _case_alert(execution: Any, threshold: float) -> pd.DataFrame:
    candidates = execution.alerts.loc[
        execution.alerts["alert_type"].eq("incident_risk")
        & execution.alerts["risk_score"].ge(threshold)
    ].sort_values("risk_score", ascending=False, kind="stable")
    return candidates.head(1).copy()


async def audit_remediation(
    output_directory: Path,
    *,
    requested_runs: list[str] | None = None,
) -> dict[str, Any]:
    base = Settings(
        challenge_remediation_wait_seconds=0,
        challenge_remediation_verification_minutes=10,
        challenge_remediation_minute_seconds=0.05,
    )
    catalog, catalog_sha256 = load_remediation_catalog(base.challenge_remediation_catalog_path)
    service = ChallengeReplayService(
        base.challenge_features_path,
        base.challenge_model_path,
        base.challenge_policy_path,
        require_all_tenants=base.challenge_require_all_tenants,
        critical_confidence_threshold=base.challenge_critical_confidence_threshold,
    )
    run_catalog = requested_runs or service.available_runs()
    prepared: list[tuple[str, Any, pd.DataFrame]] = []
    for run_id in run_catalog:
        execution = service.prepare(run_id)
        alert = _case_alert(execution, base.challenge_critical_confidence_threshold)
        if not alert.empty:
            prepared.append((run_id, execution, alert))
        if requested_runs is None and len(prepared) >= 4:
            break
    if len(prepared) < 3:
        raise ValueError("the configured threshold did not produce three eligible official runs")
    scenarios = ["recovery_verified", "operator_acknowledged", "gemini_unavailable", "verification_failed"]
    prepared = prepared[: min(len(prepared), len(scenarios))]
    original_database_url = engine().url.render_as_string(hide_password=False)
    cases: list[dict[str, Any]] = []
    try:
        with TemporaryDirectory(prefix="zikit-phase4-audit-") as temporary_name:
            temporary = Path(temporary_name)
            for index, (run_id, execution, alert_frame) in enumerate(prepared):
                database_url = f"sqlite:///{temporary / f'{run_id}.db'}"
                configure_database(database_url)
                Base.metadata.create_all(engine())
                store = ChallengePredictionRepository(base.challenge_critical_confidence_threshold)
                stored = store.append(
                    alert_frame,
                    model_sha256=execution.model_sha256,
                    policy_sha256=execution.policy_sha256,
                )
                with session_factory()() as session:
                    prediction = session.scalar(select(ChallengePredictionRecord))
                    if prediction is None:
                        raise RuntimeError("audit prediction was not persisted")
                    alert = session.scalar(
                        select(AlertRecord).where(AlertRecord.event_id == prediction.event_id)
                    )
                    if alert is None:
                        raise RuntimeError("audit alert was not persisted")
                    alert_id = alert.id
                    incident_id = prediction.incident_id
                registry = ChallengeRemediationToolRegistry(
                    catalog,
                    base.challenge_features_path,
                    minute_seconds=base.challenge_remediation_minute_seconds,
                )
                in_process_mcp = InProcessRemediationMCPClient(registry)
                scenario = scenarios[index]
                selector: Any = ContractGeminiSelector()
                client: Any = in_process_mcp
                if scenario == "operator_acknowledged":
                    with session_factory()() as session:
                        acknowledge_alert_once(session, alert_id, acknowledged_at=datetime.now(UTC))
                        session.commit()
                elif scenario == "gemini_unavailable":
                    selector = UnavailableGeminiSelector()
                elif scenario == "verification_failed":
                    client = VerificationFailureClient(in_process_mcp)
                manager = ChallengeRemediationManager(
                    base,
                    catalog,
                    catalog_sha256,
                    selector=selector,
                    execution_client=client,
                    sleeper=lambda _: asyncio.sleep(0),
                )
                processed = await manager.process_due_once(now=datetime.now(UTC))
                records = manager.list_records()
                final = records[0] if records else None
                cases.append(
                    {
                        "scenario": scenario,
                        "run_id": run_id,
                        "incident_id": incident_id,
                        "risk_score": float(alert_frame.iloc[0]["risk_score"]),
                        "suspected_incident_type": str(alert_frame.iloc[0]["suspected_incident_type"]),
                        "source_feature_sha256": _sha256(
                            base.challenge_features_path / "features" / f"{run_id}.parquet"
                        ),
                        "model_sha256": execution.model_sha256,
                        "policy_sha256": execution.policy_sha256,
                        "persisted": stored,
                        "processed": processed,
                        "final": final,
                    }
                )
    finally:
        configure_database(original_database_url)

    expected = {
        "recovery_verified": "recovered",
        "operator_acknowledged": None,
        "gemini_unavailable": "escalated",
        "verification_failed": "escalated",
    }
    gates = {
        "at_least_three_official_runs": len(cases) >= 3,
        "distinct_runs": len({case["run_id"] for case in cases}) == len(cases),
        "success_path_verified": any(
            case["scenario"] == "recovery_verified"
            and case["final"]
            and case["final"]["status"] == "recovered"
            for case in cases
        ),
        "acknowledgement_prevents_execution": any(
            case["scenario"] == "operator_acknowledged"
            and case["processed"] == 0
            and case["final"] is None
            for case in cases
        ),
        "gemini_failure_escalates_safely": any(
            case["scenario"] == "gemini_unavailable"
            and case["final"]
            and case["final"]["status"] == "escalated"
            and case["final"]["action_completed_at"] is None
            for case in cases
        ),
        "failed_verification_rolls_back_and_escalates": any(
            case["scenario"] == "verification_failed"
            and case["final"]
            and case["final"]["status"] == "escalated"
            and isinstance(case["final"]["audit"].get("mcp_rollback"), dict)
            for case in cases
        ),
    }
    report = {
        "audit_version": AUDIT_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "four isolated official synthetic Kaiku runs on controlled infrastructure",
        "implementation": {
            "critical_confidence_threshold": base.challenge_critical_confidence_threshold,
            "wait_seconds_in_audit": base.challenge_remediation_wait_seconds,
            "verification_minutes": base.challenge_remediation_verification_minutes,
            "catalog_sha256": catalog_sha256,
            "gemini_live_used": False,
            "gemini_test_mode": (
                "A strict contract double selects only an allowlisted ID; one case injects "
                "Gemini unavailability. A live Gemini call requires GEMINI_API_KEY."
            ),
            "mcp_test_mode": (
                "The exact MCP registry tools are invoked through the in-process MCP client "
                "to avoid requiring a network listener during the reproducible audit."
            ),
        },
        "gates": gates,
        "expected_terminal_status": expected,
        "cases": cases,
        "claims": {
            "official_ground_truth_used": False,
            "detector_score_modified": False,
            "production_infrastructure_tested": False,
            "controlled_infrastructure_tested": True,
            "live_gemini_tested": False,
        },
    }
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output / "remediation-audit.json")
    return {**report, "output_directory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 4 autonomous remediation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(audit_remediation(args.output, requested_runs=args.runs))
    print(json.dumps({"status": result["status"], "gates": result["gates"], "cases": len(result["cases"])}, indent=2))
    if args.fail_on_gates and result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
