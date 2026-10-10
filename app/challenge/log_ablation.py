from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.log_features import LOG_FEATURE_SCHEMA_VERSION


LOG_ABLATION_VERSION = "challenge-kaiku-log-ablation-v1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = assert_allowed_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IngestionError(f"expected a JSON object: {resolved.name}")
    return payload, resolved


def compare_log_candidate(
    baseline_training: dict[str, Any],
    candidate_training: dict[str, Any],
    baseline_policy: dict[str, Any],
    candidate_policy: dict[str, Any],
) -> dict[str, Any]:
    if baseline_training.get("splits") != candidate_training.get("splits"):
        raise IngestionError("log ablation requires identical run splits")
    if baseline_training.get("log_feature_version") is not None:
        raise IngestionError("baseline training report already includes logs")
    if candidate_training.get("log_feature_version") != LOG_FEATURE_SCHEMA_VERSION:
        raise IngestionError("candidate training report lacks the expected log features")

    horizons: dict[str, Any] = {}
    for horizon in ("40", "120", "360", "1440"):
        baseline = baseline_training["test_proxy_metrics"]["horizons"][horizon]
        candidate = candidate_training["test_proxy_metrics"]["horizons"][horizon]
        horizons[horizon] = {
            "baseline_average_precision": baseline["average_precision"],
            "candidate_average_precision": candidate["average_precision"],
            "average_precision_delta": (
                candidate["average_precision"] - baseline["average_precision"]
            ),
            "baseline_false_positive_rate": baseline["false_positive_rate"],
            "candidate_false_positive_rate": candidate["false_positive_rate"],
            "false_positive_rate_delta": (
                candidate["false_positive_rate"] - baseline["false_positive_rate"]
            ),
        }

    baseline_test = baseline_policy["test_proxy_metrics"]
    candidate_test = candidate_policy["test_proxy_metrics"]
    baseline_lead = baseline_test.get("median_lead_minutes")
    candidate_lead = candidate_test.get("median_lead_minutes")
    no_detection_loss = (
        candidate_test["detected_proxy_anchors"]
        >= baseline_test["detected_proxy_anchors"]
    )
    no_false_alert_increase = (
        candidate_test["false_incident_alerts"]
        <= baseline_test["false_incident_alerts"]
    )
    no_lead_time_loss = (
        baseline_lead is not None
        and candidate_lead is not None
        and candidate_lead >= baseline_lead
    )
    promote_for_risk = no_detection_loss and no_false_alert_increase and no_lead_time_loss
    return {
        "ablation_version": LOG_ABLATION_VERSION,
        "scope": "held-out proxy anchors; official evaluator labels excluded",
        "same_splits": True,
        "horizons": horizons,
        "policy_test": {
            "baseline": baseline_test,
            "candidate": candidate_test,
            "delta": {
                "detected_proxy_anchors": (
                    candidate_test["detected_proxy_anchors"]
                    - baseline_test["detected_proxy_anchors"]
                ),
                "false_incident_alerts": (
                    candidate_test["false_incident_alerts"]
                    - baseline_test["false_incident_alerts"]
                ),
                "median_lead_minutes": (
                    candidate_lead - baseline_lead
                    if candidate_lead is not None and baseline_lead is not None
                    else None
                ),
            },
        },
        "gates": {
            "no_detection_loss": no_detection_loss,
            "no_false_alert_increase": no_false_alert_increase,
            "no_lead_time_loss": no_lead_time_loss,
        },
        "decision": {
            "promote_log_candidate_for_risk_scoring": promote_for_risk,
            "risk_model": "log_candidate" if promote_for_risk else "metrics_baseline",
            "log_role": (
                "risk_scoring_and_evidence"
                if promote_for_risk
                else "incident typing and evidence only"
            ),
            "reason": (
                "candidate passed detection, false-alert and lead-time gates"
                if promote_for_risk
                else "candidate did not improve all held-out proxy gates"
            ),
        },
        "limitations": {
            "official_ground_truth_used": False,
            "official_precision_or_recall_available": False,
        },
    }


def write_log_ablation(
    baseline_training_path: Path,
    candidate_training_path: Path,
    baseline_policy_path: Path,
    candidate_policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    baseline_training, baseline_training_file = _load(baseline_training_path)
    candidate_training, candidate_training_file = _load(candidate_training_path)
    baseline_policy, baseline_policy_file = _load(baseline_policy_path)
    candidate_policy, candidate_policy_file = _load(candidate_policy_path)
    report = compare_log_candidate(
        baseline_training,
        candidate_training,
        baseline_policy,
        candidate_policy,
    )
    report["lineage"] = {
        "baseline_training_sha256": _sha256_file(baseline_training_file),
        "candidate_training_sha256": _sha256_file(candidate_training_file),
        "baseline_policy_report_sha256": _sha256_file(baseline_policy_file),
        "candidate_policy_report_sha256": _sha256_file(candidate_policy_file),
    }
    output = assert_allowed_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Kaiku metrics-only and log-enriched model candidates"
    )
    parser.add_argument("--baseline-training", type=Path, required=True)
    parser.add_argument("--candidate-training", type=Path, required=True)
    parser.add_argument("--baseline-policy", type=Path, required=True)
    parser.add_argument("--candidate-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_log_ablation(
        args.baseline_training,
        args.candidate_training,
        args.baseline_policy,
        args.candidate_policy,
        args.output,
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
