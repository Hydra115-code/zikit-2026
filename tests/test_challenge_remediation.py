from __future__ import annotations

import json

import pytest

from app.challenge.remediation_audit import audit_remediation


@pytest.mark.asyncio
async def test_phase4_audit_uses_distinct_official_runs_and_covers_fail_safe_paths(
    tmp_path,
) -> None:
    result = await audit_remediation(tmp_path)

    assert result["status"] == "passed"
    assert len(result["cases"]) >= 3
    assert len({case["run_id"] for case in result["cases"]}) == len(result["cases"])
    assert result["gates"] == {
        "at_least_three_official_runs": True,
        "distinct_runs": True,
        "success_path_verified": True,
        "acknowledgement_prevents_execution": True,
        "gemini_failure_escalates_safely": True,
        "failed_verification_rolls_back_and_escalates": True,
    }
    assert all(len(case["source_feature_sha256"]) == 64 for case in result["cases"])
    assert result["claims"] == {
        "official_ground_truth_used": False,
        "detector_score_modified": False,
        "production_infrastructure_tested": False,
        "controlled_infrastructure_tested": True,
        "live_gemini_tested": False,
    }

    persisted = json.loads((tmp_path / "remediation-audit.json").read_text())
    assert persisted["status"] == "passed"
    failed = next(
        case for case in persisted["cases"] if case["scenario"] == "verification_failed"
    )
    assert failed["final"]["status"] == "escalated"
    assert failed["final"]["audit"]["mcp_rollback"]["state"] == "completed"
