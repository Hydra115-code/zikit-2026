from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.contracts import TARGET_TENANT
from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.model import load_challenge_bundle
from app.challenge.policy import POLICY_VERSION, PolicyConfig, load_policy_artifact
from app.challenge.policy_calibration import _prepare_runs, evaluate_policy
from app.challenge.replay import REPLAY_VERSION
from app.challenge.submission import (
    SUBMISSION_PACKAGE_MANIFEST,
    validate_aggregate_submission_file,
    validate_submission_package,
)
from app.challenge.training import discover_feature_files


EVALUATION_VERSION = "challenge-kaiku-causal-evaluation-v3"
BASELINE_POLICY_VERSION = "challenge-alert-policy-v2"
MIN_PROXY_EPISODE_COVERAGE = 0.80
MAX_UNMATCHED_PROXY_ALERTS_PER_TENANT_DAY = 0.05
REQUIRED_REPLAY_OUTPUTS = frozenset(
    {
        "alerts.csv",
        "alerts.full.parquet",
        "data-quality-alerts.csv",
        "replay-report.json",
        SUBMISSION_PACKAGE_MANIFEST,
    }
)


def baseline_policy_config(critical_40m_threshold: float) -> PolicyConfig:
    return PolicyConfig(
        observation_threshold=0.30,
        candidate_threshold=0.48,
        alert_threshold=0.64,
        recovery_threshold=0.18,
        observation_minutes=3,
        candidate_minutes=3,
        cooldown_minutes=360,
        context_risk_multiplier=0.25,
        critical_40m_threshold=critical_40m_threshold,
        critical_isolation_threshold=0.85,
        data_quality_persistence_minutes=3,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    resolved = assert_allowed_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IngestionError(f"expected a JSON object: {resolved.name}")
    return payload


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(temporary, path)


def _verify_replay_package(replay_directory: Path) -> dict[str, Any]:
    """Fail closed when replay evidence is incomplete, stale, or modified."""
    root = assert_allowed_path(replay_directory)
    manifest_path = assert_allowed_path(root / "manifest.json")
    if not manifest_path.is_file():
        raise IngestionError("replay manifest is required")
    manifest = _load_json(manifest_path)
    if manifest.get("replay_version") != REPLAY_VERSION:
        raise IngestionError("incompatible replay manifest")

    deterministic = set(manifest.get("deterministic_outputs", []))
    if not REQUIRED_REPLAY_OUTPUTS.issubset(deterministic):
        missing = sorted(REQUIRED_REPLAY_OUTPUTS - deterministic)
        raise IngestionError(f"replay manifest is missing deterministic outputs: {missing}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise IngestionError("replay manifest outputs must be an object")

    verified: dict[str, str] = {}
    for name in sorted(REQUIRED_REPLAY_OUTPUTS):
        if Path(name).name != name:
            raise IngestionError("replay output names must be plain filenames")
        artifact = assert_allowed_path(root / name)
        if not artifact.is_file():
            raise IngestionError(f"replay artifact is missing: {name}")
        actual_digest = _sha256_file(artifact)
        if outputs.get(name) != actual_digest:
            raise IngestionError(f"replay artifact digest mismatch: {name}")
        verified[name] = actual_digest
    return {
        "passed": True,
        "manifest_sha256": _sha256_file(manifest_path),
        "verified_outputs": verified,
    }


def _replay_alert_summary(
    replay_alerts: pd.DataFrame,
    *,
    tenant_days: float,
) -> dict[str, Any]:
    required = {
        "run_id",
        "tenant",
        "signal_timestamp",
        "timestamp",
        "alert_type",
        "risk_score",
        "context_known_event",
    }
    missing = sorted(required - set(replay_alerts.columns))
    if missing:
        raise IngestionError(f"replay alerts are missing columns: {missing}")
    if set(replay_alerts["tenant"].astype(str).unique()) != {TARGET_TENANT}:
        raise IngestionError("replay alerts must contain only Kaiku")
    supported_types = {"incident_risk", "data_quality_alert"}
    alert_types = set(replay_alerts["alert_type"].astype(str).unique())
    if not alert_types.issubset(supported_types):
        raise IngestionError(f"replay contains unsupported alert types: {sorted(alert_types)}")

    identity = ["run_id", "tenant", "timestamp", "alert_type"]
    episode = ["run_id", "tenant", "signal_timestamp", "alert_type"]
    duplicate_alerts = int(replay_alerts.duplicated(identity).sum())
    duplicate_episodes = int(replay_alerts.duplicated(episode).sum())
    incident = replay_alerts.loc[replay_alerts["alert_type"].eq("incident_risk")]
    quality = replay_alerts.loc[
        replay_alerts["alert_type"].eq("data_quality_alert")
    ]
    scores = pd.to_numeric(incident["risk_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise IngestionError("replay incident scores must be finite")
    if not scores.between(0.0, 1.0, inclusive="both").all():
        raise IngestionError("replay incident scores must be between zero and one")
    known_events = pd.to_numeric(
        incident["context_known_event"], errors="coerce"
    )
    if known_events.isna().any() or not known_events.isin([0, 1]).all():
        raise IngestionError("replay context_known_event must be boolean")
    return {
        "incident_alerts": int(len(incident)),
        "data_quality_alerts": int(len(quality)),
        "incident_alerts_per_tenant_day": (
            float(len(incident) / tenant_days) if tenant_days else 0.0
        ),
        "known_event_incident_alerts": int(known_events.sum()),
        "duplicate_alerts": duplicate_alerts,
        "duplicate_episode_alerts": duplicate_episodes,
    }


def _summary_matches_report(
    actual: dict[str, Any], reported: dict[str, Any]
) -> bool:
    integer_fields = (
        "incident_alerts",
        "data_quality_alerts",
        "known_event_incident_alerts",
        "duplicate_alerts",
        "duplicate_episode_alerts",
    )
    if any(int(reported.get(field, -1)) != actual[field] for field in integer_fields):
        return False
    try:
        reported_rate = float(reported["incident_alerts_per_tenant_day"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isclose(
            reported_rate,
            actual["incident_alerts_per_tenant_day"],
            rtol=0,
            atol=1e-12,
        )
    )


def _observed_tenant_days(
    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> float:
    """Measure inclusive observed minutes exactly as the causal replay does."""
    total = 0.0
    for frame, _, _ in prepared.values():
        for _, group in frame.groupby(["run_id", "tenant"], sort=False):
            duration = (
                pd.Timestamp(group["timestamp"].max())
                - pd.Timestamp(group["timestamp"].min())
            ).total_seconds() / 86_400
            total += max(duration + 1 / 1440, 1 / 1440)
    return total


def compare_policy_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        "detected_proxy_anchors": (
            candidate["detected_proxy_anchors"] - baseline["detected_proxy_anchors"]
        ),
        "false_incident_alerts": (
            candidate["false_incident_alerts"] - baseline["false_incident_alerts"]
        ),
        "median_lead_minutes": (
            candidate["median_lead_minutes"] - baseline["median_lead_minutes"]
        ),
        "median_decision_latency_minutes": (
            candidate["median_decision_latency_minutes"]
            - baseline["median_decision_latency_minutes"]
        ),
        "known_event_incident_alerts": (
            candidate["known_event_incident_alerts"]
            - baseline["known_event_incident_alerts"]
        ),
    }
    gates = {
        "no_detection_loss": deltas["detected_proxy_anchors"] >= 0,
        "no_false_alert_increase": deltas["false_incident_alerts"] <= 0,
        "no_lead_time_loss": deltas["median_lead_minutes"] >= 0,
        "decision_is_not_slower": deltas["median_decision_latency_minutes"] <= 0,
        "no_known_event_regression": deltas["known_event_incident_alerts"] <= 0,
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "gates": gates,
        "promoted": all(gates.values()),
    }


def _alert_identity(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "tenant",
        "signal_timestamp",
        "timestamp",
        "alert_type",
        "risk_score",
    ]
    return frame.loc[:, columns].sort_values(
        columns[:-1], kind="stable", ignore_index=True
    )


def evaluate_frozen_policy(
    feature_directory: Path,
    model_path: Path,
    policy_path: Path,
    training_report_path: Path,
    replay_directory: Path,
    output_directory: Path,
    *,
    expected_runs: int = 54,
) -> dict[str, Any]:
    files, feature_report, feature_manifest = discover_feature_files(
        feature_directory, expected_runs=expected_runs
    )
    replay_package = _verify_replay_package(replay_directory)
    training = _load_json(training_report_path)
    replay_report = _load_json(assert_allowed_path(replay_directory) / "replay-report.json")
    replay_alerts = pd.read_parquet(
        assert_allowed_path(replay_directory) / "alerts.full.parquet"
    )

    model = assert_allowed_path(model_path)
    model_digest = _sha256_file(model)
    if training.get("model", {}).get("sha256") != model_digest:
        raise IngestionError("evaluation model does not match the training report")
    bundle = load_challenge_bundle(model, trusted_root=model.parent)
    if bundle is None:
        raise FileNotFoundError(model)
    policy_config, policy_payload = load_policy_artifact(
        policy_path,
        trusted_root=assert_allowed_path(policy_path).parent,
        expected_model_sha256=model_digest,
    )
    if policy_payload.get("model_version") != bundle.model_version:
        raise IngestionError("evaluation policy and model versions do not match")

    feature_manifest_digest = _sha256_file(
        assert_allowed_path(feature_directory) / "manifest.json"
    )
    policy_digest = _sha256_file(assert_allowed_path(policy_path))
    replay_lineage = replay_report.get("lineage", {})
    lineage_matches = (
        replay_report.get("replay_version") == REPLAY_VERSION
        and replay_report.get("model_version") == bundle.model_version
        and replay_report.get("policy_version") == POLICY_VERSION
        and replay_lineage.get("phase_two_manifest_sha256")
        == feature_manifest_digest
        and replay_lineage.get("model_sha256") == model_digest
        and replay_lineage.get("policy_sha256") == policy_digest
    )

    split_runs = training["splits"]["runs"]
    all_run_ids = [run_id for name in ("train", "validation", "test") for run_id in split_runs[name]]
    if len(all_run_ids) != expected_runs or len(set(all_run_ids)) != expected_runs:
        raise IngestionError("evaluation splits do not cover unique expected runs")
    prepared = _prepare_runs(
        all_run_ids,
        files,
        bundle,
        training["proxy_target"]["reference"],
        training["proxy_target"]["severity_thresholds"],
    )
    test_prepared = {
        run_id: prepared[run_id] for run_id in split_runs["test"]
    }
    baseline_config = baseline_policy_config(
        float(training["selected_validation_thresholds"]["40"])
    )
    baseline_test_metrics, _ = evaluate_policy(
        test_prepared,
        baseline_config,
        model_version=bundle.model_version,
    )

    split_metrics: dict[str, Any] = {}
    all_alert_frames: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        subset = {run_id: prepared[run_id] for run_id in split_runs[split_name]}
        metrics, alerts = evaluate_policy(
            subset,
            policy_config,
            model_version=bundle.model_version,
        )
        split_metrics[split_name] = metrics
        all_alert_frames.append(alerts)

    total_tenant_days = _observed_tenant_days(prepared)
    actual_replay_summary = _replay_alert_summary(
        replay_alerts,
        tenant_days=total_tenant_days,
    )
    reported_replay_summary = replay_report.get("alerts", {})
    replay_summary_matches = _summary_matches_report(
        actual_replay_summary,
        reported_replay_summary,
    )
    aggregate_submission_validation = validate_aggregate_submission_file(
        assert_allowed_path(replay_directory) / "alerts.csv"
    )
    submission_validation = validate_submission_package(
        replay_directory,
        expected_runs=expected_runs,
    )

    evaluated_alerts = pd.concat(all_alert_frames, ignore_index=True)
    replay_matches = _alert_identity(evaluated_alerts).equals(
        _alert_identity(replay_alerts)
    )

    per_run_rows: list[dict[str, Any]] = []
    split_by_run = {
        run_id: split_name
        for split_name, run_ids in split_runs.items()
        for run_id in run_ids
    }
    for run_id in sorted(all_run_ids):
        metrics, _ = evaluate_policy(
            {run_id: prepared[run_id]},
            policy_config,
            model_version=bundle.model_version,
        )
        per_run_rows.append(
            {
                "run_id": run_id,
                "split": split_by_run[run_id],
                "proxy_anchors": metrics["proxy_anchor_count"],
                "detected_proxy_anchors": metrics["detected_proxy_anchors"],
                "false_incident_alerts": metrics["false_incident_alerts"],
                "incident_alerts": metrics["incident_alert_count"],
                "known_event_incident_alerts": metrics[
                    "known_event_incident_alerts"
                ],
                "median_lead_minutes": metrics["median_lead_minutes"],
                "median_decision_latency_minutes": metrics[
                    "median_decision_latency_minutes"
                ],
            }
        )
    per_run = pd.DataFrame(per_run_rows)

    comparison = compare_policy_metrics(
        baseline_test_metrics,
        split_metrics["test"],
    )
    replay_alert_summary = actual_replay_summary
    heldout = split_metrics["test"]
    gates = {
        "replay_package_integrity": replay_package,
        "lineage": {
            "passed": lineage_matches,
            "feature_manifest_sha256": feature_manifest_digest,
            "model_sha256": model_digest,
            "policy_sha256": policy_digest,
        },
        "replay_summary_consistency": {
            "passed": replay_summary_matches,
            "actual": actual_replay_summary,
            "reported": {
                key: reported_replay_summary.get(key)
                for key in actual_replay_summary
            },
        },
        "submission_contract": submission_validation,
        "aggregate_audit_contract": aggregate_submission_validation,
        "policy_promotion": {
            "passed": bool(comparison["promoted"]),
            **comparison,
        },
        "heldout_proxy_sanity": {
            "passed": (
                heldout["proxy_recall"] >= MIN_PROXY_EPISODE_COVERAGE
                and heldout["false_alerts_per_tenant_day"]
                <= MAX_UNMATCHED_PROXY_ALERTS_PER_TENANT_DAY
                and heldout["known_event_incident_alerts"] == 0
            ),
            "diagnostic_only": True,
            "minimum_proxy_episode_coverage": MIN_PROXY_EPISODE_COVERAGE,
            "maximum_unmatched_alerts_per_tenant_day": (
                MAX_UNMATCHED_PROXY_ALERTS_PER_TENANT_DAY
            ),
            "proxy_episode_coverage": heldout["proxy_recall"],
            "unmatched_alerts_per_tenant_day": heldout[
                "false_alerts_per_tenant_day"
            ],
            "detected_proxy_anchors": heldout["detected_proxy_anchors"],
            "proxy_anchor_count": heldout["proxy_anchor_count"],
            "false_incident_alerts": heldout["false_incident_alerts"],
            "known_event_incident_alerts": heldout[
                "known_event_incident_alerts"
            ],
        },
        "replay_budget": {
            "passed": (
                replay_alert_summary["known_event_incident_alerts"] == 0
                and replay_alert_summary["duplicate_alerts"] == 0
                and replay_alert_summary["duplicate_episode_alerts"] == 0
                and replay_alert_summary["incident_alerts_per_tenant_day"] <= 0.5
            ),
            "incident_alerts_per_tenant_day": replay_alert_summary[
                "incident_alerts_per_tenant_day"
            ],
            "known_event_incident_alerts": replay_alert_summary[
                "known_event_incident_alerts"
            ],
            "duplicate_alerts": replay_alert_summary["duplicate_alerts"],
            "duplicate_episode_alerts": replay_alert_summary[
                "duplicate_episode_alerts"
            ],
        },
        "causality": {
            "passed": (
                feature_report.get("causality", {}).get("future_rows_used") is False
                and feature_report.get("causality", {}).get("log_future_events_used")
                is False
                and replay_report.get("causality", {}).get("future_rows_used")
                is False
                and replay_report.get("causality", {}).get(
                    "model_or_policy_recalibrated"
                )
                is False
                and replay_report.get("causality", {}).get(
                    "policy_processed_minute_by_minute"
                )
                is True
                and replay_report.get("causality", {}).get(
                    "pointwise_equivalence_checks"
                )
                == expected_runs
                and replay_report.get("causality", {}).get(
                    "max_pointwise_score_delta"
                )
                == 0.0
            ),
            "feature_future_rows_used": feature_report.get("causality", {}).get(
                "future_rows_used"
            ),
            "log_future_events_used": feature_report.get("causality", {}).get(
                "log_future_events_used"
            ),
            "replay_future_rows_used": replay_report.get("causality", {}).get(
                "future_rows_used"
            ),
            "model_or_policy_recalibrated": replay_report.get(
                "causality", {}
            ).get("model_or_policy_recalibrated"),
            "policy_processed_minute_by_minute": replay_report.get(
                "causality", {}
            ).get("policy_processed_minute_by_minute"),
            "pointwise_equivalence_checks": replay_report.get(
                "causality", {}
            ).get("pointwise_equivalence_checks"),
            "max_pointwise_score_delta": replay_report.get(
                "causality", {}
            ).get("max_pointwise_score_delta"),
        },
        "frozen_replay_equivalence": {
            "passed": replay_matches,
            "evaluated_alerts": int(len(evaluated_alerts)),
            "replay_alerts": int(len(replay_alerts)),
        },
        "split_isolation": {
            "passed": training.get("splits", {}).get("overlap") == [],
            "counts": training.get("splits", {}).get("counts"),
            "overlap": training.get("splits", {}).get("overlap"),
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    report: dict[str, Any] = {
        "evaluation_version": EVALUATION_VERSION,
        "status": "passed" if passed else "failed",
        "scope": (
            "public Kaiku runs; variable proxy episodes are diagnostic and are "
            "not official incident truth"
        ),
        "gates": gates,
        "split_proxy_metrics": split_metrics,
        "baseline": {
            "policy_version": BASELINE_POLICY_VERSION,
            "config": baseline_config.to_dict(),
            "test_proxy_metrics": baseline_test_metrics,
        },
        "stability": {
            "runs": int(len(per_run)),
            "runs_with_false_proxy_alerts": int(
                per_run["false_incident_alerts"].gt(0).sum()
            ),
            "runs_with_missed_proxy_anchors": int(
                per_run["detected_proxy_anchors"]
                .lt(per_run["proxy_anchors"])
                .sum()
            ),
            "incident_alerts_per_run": {
                "min": int(per_run["incident_alerts"].min()),
                "median": float(per_run["incident_alerts"].median()),
                "max": int(per_run["incident_alerts"].max()),
            },
        },
        "limitations": {
            "official_ground_truth_used": False,
            "official_precision_recall_available": False,
            "official_lead_time_available": False,
            "test_used_for_policy_selection": False,
            "automatic_remediation": False,
        },
        "lineage": {
            "feature_manifest_sha256": feature_manifest_digest,
            "feature_files": len(feature_manifest["feature_files"]),
            "model_sha256": model_digest,
            "policy_sha256": policy_digest,
            "replay_manifest_sha256": replay_package["manifest_sha256"],
            "training_report_sha256": _sha256_file(
                assert_allowed_path(training_report_path)
            ),
            "replay_report_sha256": _sha256_file(
                assert_allowed_path(replay_directory) / "replay-report.json"
            ),
        },
    }

    output = assert_allowed_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "evaluation-report.json"
    per_run_path = output / "per-run-evaluation.csv"
    _atomic_json(report, report_path)
    _atomic_csv(per_run, per_run_path)
    manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "status": report["status"],
        "outputs": {
            report_path.name: _sha256_file(report_path),
            per_run_path.name: _sha256_file(per_run_path),
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    return {**report, "output_directory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Kaiku policy without private labels"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=54)
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    result = evaluate_frozen_policy(
        args.features,
        args.model,
        args.policy,
        args.training_report,
        args.replay,
        args.output,
        expected_runs=args.expected_runs,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "test_proxy_metrics": result["split_proxy_metrics"]["test"],
                "output_directory": result["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_gates and result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
