from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.model import CHALLENGE_MODEL_VERSION, load_challenge_bundle
from app.challenge.policy import (
    POLICY_VERSION,
    PolicyConfig,
    replay_policy,
    save_policy_artifact,
)
from app.challenge.training import discover_feature_files, proxy_labels_for_run


MAX_PROXY_LEAD_MINUTES = 1440


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_training_report(path: Path) -> dict[str, Any]:
    resolved = assert_allowed_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if report.get("model_version") != CHALLENGE_MODEL_VERSION:
        raise IngestionError("incompatible phase-3 training report")
    if report.get("proxy_target", {}).get("official_ground_truth") is not False:
        raise IngestionError("phase-3 report must identify its targets as proxies")
    return report


def _prepare_runs(
    run_ids: list[str],
    files: dict[str, Path],
    bundle: Any,
    reference: dict[str, dict[str, dict[str, float]]],
    severity_thresholds: dict[str, float],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for run_id in run_ids:
        frame = pd.read_parquet(files[run_id]).sort_values(
            ["tenant", "timestamp"], kind="stable", ignore_index=True
        )
        scores = bundle.score_frame(frame)
        labels = proxy_labels_for_run(
            frame,
            reference,
            severity_thresholds=severity_thresholds,
        )
        anchors = labels.loc[
            labels["proxy_anchor"], ["run_id", "tenant", "timestamp"]
        ].reset_index(drop=True)
        prepared[run_id] = (frame, scores, anchors)
    return prepared


def _match_alerts(
    alerts: pd.DataFrame, anchors: pd.DataFrame
) -> tuple[list[float], int]:
    lead_minutes: list[float] = []
    false_alerts = 0
    for (run_id, tenant), tenant_alerts in alerts.groupby(
        ["run_id", "tenant"], sort=True
    ):
        candidates = anchors.loc[
            anchors["run_id"].eq(run_id) & anchors["tenant"].eq(tenant),
            "timestamp",
        ].sort_values(kind="stable")
        anchor_times = [pd.Timestamp(value) for value in candidates]
        matched: set[int] = set()
        for alert_time_value in tenant_alerts["timestamp"].sort_values(kind="stable"):
            alert_time = pd.Timestamp(alert_time_value)
            match = next(
                (
                    index
                    for index, anchor_time in enumerate(anchor_times)
                    if index not in matched
                    and 0 < (anchor_time - alert_time).total_seconds() / 60
                    <= MAX_PROXY_LEAD_MINUTES
                ),
                None,
            )
            if match is None:
                false_alerts += 1
                continue
            matched.add(match)
            lead_minutes.append(
                (anchor_times[match] - alert_time).total_seconds() / 60
            )
    return lead_minutes, false_alerts


def evaluate_policy(
    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    config: PolicyConfig,
    *,
    model_version: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    alert_frames: list[pd.DataFrame] = []
    anchor_frames: list[pd.DataFrame] = []
    tenant_days = 0.0
    for frame, scores, anchors in prepared.values():
        alert_frames.append(
            replay_policy(frame, scores, config, model_version=model_version)
        )
        anchor_frames.append(anchors)
        for _, group in frame.groupby(["run_id", "tenant"], sort=False):
            duration = (
                pd.Timestamp(group["timestamp"].max())
                - pd.Timestamp(group["timestamp"].min())
            ).total_seconds() / 86_400
            tenant_days += max(duration, 1 / 1440)
    alerts = (
        pd.concat(alert_frames, ignore_index=True)
        if alert_frames
        else pd.DataFrame()
    )
    anchors = pd.concat(anchor_frames, ignore_index=True)
    incident_alerts = alerts.loc[alerts["alert_type"].eq("incident_risk")].copy()
    quality_alerts = alerts.loc[alerts["alert_type"].eq("data_quality_alert")].copy()
    lead_minutes, false_alerts = _match_alerts(incident_alerts, anchors)
    detected = len(lead_minutes)
    alert_count = len(incident_alerts)
    anchor_count = len(anchors)
    recall = detected / anchor_count if anchor_count else 0.0
    precision = detected / alert_count if alert_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alerts_per_tenant_day = false_alerts / tenant_days if tenant_days else 0.0
    known_event_alerts = int(incident_alerts["context_known_event"].sum())
    known_event_rate = known_event_alerts / alert_count if alert_count else 0.0
    median_lead = float(np.median(lead_minutes)) if lead_minutes else None
    median_decision_latency = (
        float(incident_alerts["decision_latency_minutes"].median())
        if alert_count
        else None
    )
    lead_reward = min((median_lead or 0) / MAX_PROXY_LEAD_MINUTES, 1.0)
    decision_latency_penalty = min(
        (median_decision_latency or MAX_PROXY_LEAD_MINUTES)
        / MAX_PROXY_LEAD_MINUTES,
        1.0,
    )
    objective = (
        f1
        + 0.08 * lead_reward
        - 0.10 * false_alerts_per_tenant_day
        - 0.25 * known_event_rate
        - 0.02 * decision_latency_penalty
    )
    metrics: dict[str, Any] = {
        "proxy_anchor_count": anchor_count,
        "incident_alert_count": alert_count,
        "data_quality_alert_count": len(quality_alerts),
        "detected_proxy_anchors": detected,
        "proxy_recall": recall,
        "proxy_precision": precision,
        "proxy_f1": f1,
        "false_incident_alerts": false_alerts,
        "tenant_days": tenant_days,
        "false_alerts_per_tenant_day": false_alerts_per_tenant_day,
        "known_event_incident_alerts": known_event_alerts,
        "median_lead_minutes": median_lead,
        "p10_lead_minutes": (
            float(np.quantile(lead_minutes, 0.10)) if lead_minutes else None
        ),
        "p90_lead_minutes": (
            float(np.quantile(lead_minutes, 0.90)) if lead_minutes else None
        ),
        "median_decision_latency_minutes": median_decision_latency,
        "typed_incident_alerts": int(
            incident_alerts["suspected_incident_type"].notna().sum()
        ),
        "objective": objective,
    }
    return metrics, alerts


def policy_candidates(critical_40m_threshold: float) -> list[PolicyConfig]:
    candidates: list[PolicyConfig] = []
    threshold_profiles = (
        (0.30, 0.44, 0.55),
        (0.42, 0.44, 0.55),
        (0.44, 0.46, 0.55),
        (0.46, 0.48, 0.55),
        (0.48, 0.50, 0.55),
        (0.50, 0.52, 0.55),
        (0.53, 0.54, 0.55),
        (0.30, 0.48, 0.64),
        (0.30, 0.48, 0.72),
        (0.30, 0.56, 0.72),
        (0.38, 0.56, 0.72),
    )
    operating_profiles = (
        (3, 360, 0.25),
        (3, 720, 0.25),
        (5, 360, 0.40),
    )
    for observation, candidate, alert in threshold_profiles:
        for observation_minutes, cooldown, context in operating_profiles:
            candidates.append(
                PolicyConfig(
                    observation_threshold=observation,
                    candidate_threshold=candidate,
                    alert_threshold=alert,
                    recovery_threshold=max(0.05, observation - 0.12),
                    observation_minutes=observation_minutes,
                    candidate_minutes=3,
                    cooldown_minutes=cooldown,
                    recovery_minutes=3,
                    context_risk_multiplier=context,
                    critical_40m_threshold=critical_40m_threshold,
                    critical_isolation_threshold=0.85,
                    data_quality_persistence_minutes=3,
                )
            )
    return candidates


def _selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["metrics"]
    decision_latency = metrics["median_decision_latency_minutes"]
    return (
        float(metrics["objective"]),
        -float(metrics["false_incident_alerts"]),
        float(metrics["median_lead_minutes"] or 0),
        -float(decision_latency if decision_latency is not None else np.inf),
    )


def calibrate_policy(
    input_directory: Path,
    training_report_path: Path,
    model_path: Path,
    output_directory: Path,
    policy_path: Path,
    *,
    expected_runs: int = 54,
) -> dict[str, Any]:
    files, _, feature_manifest = discover_feature_files(
        input_directory, expected_runs=expected_runs
    )
    training_report = _load_training_report(training_report_path)
    model = model_path.expanduser().resolve(strict=False)
    model_digest = _sha256_file(model)
    if model_digest != training_report.get("model", {}).get("sha256"):
        raise IngestionError("model digest does not match the phase-3 report")
    bundle = load_challenge_bundle(model, trusted_root=model.parent)
    if bundle is None:
        raise FileNotFoundError(model)
    splits = training_report["splits"]["runs"]
    reference = training_report["proxy_target"]["reference"]
    severity_thresholds = training_report["proxy_target"]["severity_thresholds"]
    validation = _prepare_runs(
        splits["validation"],
        files,
        bundle,
        reference,
        severity_thresholds,
    )
    critical_40m = float(training_report["selected_validation_thresholds"]["40"])

    grid_results: list[dict[str, Any]] = []
    for config in policy_candidates(critical_40m):
        metrics, _ = evaluate_policy(
            validation, config, model_version=bundle.model_version
        )
        grid_results.append({"config": config.to_dict(), "metrics": metrics})
    selected = max(grid_results, key=_selection_key)
    selected_config = PolicyConfig.from_dict(selected["config"])
    validation_metrics, validation_alerts = evaluate_policy(
        validation, selected_config, model_version=bundle.model_version
    )
    # The test split remains unopened until the policy is fixed on validation.
    test = _prepare_runs(
        splits["test"],
        files,
        bundle,
        reference,
        severity_thresholds,
    )
    test_metrics, test_alerts = evaluate_policy(
        test, selected_config, model_version=bundle.model_version
    )

    policy = policy_path.expanduser().resolve(strict=False)
    policy_digest = save_policy_artifact(
        selected_config,
        policy,
        model_version=bundle.model_version,
        model_sha256=model_digest,
        calibration={
            "split": "validation",
            "official_ground_truth": False,
            "metrics": validation_metrics,
        },
    )
    output = output_directory.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    validation_alerts.to_parquet(
        output / "validation-alerts.parquet", index=False, compression="zstd"
    )
    test_alerts.to_parquet(
        output / "test-alerts.parquet", index=False, compression="zstd"
    )
    top_candidates = sorted(grid_results, key=_selection_key, reverse=True)[:10]
    report: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "model_version": bundle.model_version,
        "model_sha256": model_digest,
        "source": {
            "phase_two_manifest_sha256": _sha256_file(
                assert_allowed_path(input_directory) / "manifest.json"
            ),
            "phase_three_report_sha256": _sha256_file(
                assert_allowed_path(training_report_path)
            ),
            "feature_files": len(feature_manifest["feature_files"]),
        },
        "selection": {
            "split": "validation",
            "candidate_count": len(grid_results),
            "objective": (
                "proxy_f1 + lead_reward - false_alert_rate - known_event_alert_rate; "
                "includes a small signal-to-decision latency penalty; ties prefer lower latency"
            ),
            "config": selected_config.to_dict(),
        },
        "policy_architecture": {
            "state_machine": [
                "NORMAL",
                "OBSERVATION",
                "CANDIDATE",
                "ALERT",
                "COOLDOWN",
            ],
            "first_signal_retained_until_recovery": True,
            "sustained_recovery_required": True,
            "one_alert_per_unrecovered_episode": True,
            "known_event_risk_reduction": True,
            "critical_override_requires_short_horizon_and_isolation": True,
            "outage_alert_separated_from_incident": True,
            "non_finite_scores_rejected": True,
            "official_ground_truth_used": False,
        },
        "validation_proxy_metrics": validation_metrics,
        "test_proxy_metrics": test_metrics,
        "top_validation_candidates": top_candidates,
        "limitations": {
            "official_ground_truth": False,
            "test_used_for_selection": False,
            "metrics_scope": "variable proxy episodes only",
            "automatic_remediation": False,
        },
    }
    report_path = output / "policy-report.json"
    _atomic_json(report, report_path)
    manifest = {
        "policy_version": POLICY_VERSION,
        "outputs": {
            "policy-report.json": _sha256_file(report_path),
            "validation-alerts.parquet": _sha256_file(
                output / "validation-alerts.parquet"
            ),
            "test-alerts.parquet": _sha256_file(output / "test-alerts.parquet"),
            policy.name: policy_digest,
            policy.with_suffix(policy.suffix + ".sha256").name: _sha256_file(
                policy.with_suffix(policy.suffix + ".sha256")
            ),
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    return {**report, "output_directory": str(output), "policy_path": str(policy)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the causal anti-noise alert policy on validation runs"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=54)
    args = parser.parse_args()
    report = calibrate_policy(
        args.input,
        args.training_report,
        args.model,
        args.output,
        args.policy,
        expected_runs=args.expected_runs,
    )
    print(
        json.dumps(
            {
                "policy_version": report["policy_version"],
                "selected_config": report["selection"]["config"],
                "validation_proxy_metrics": report["validation_proxy_metrics"],
                "test_proxy_metrics": report["test_proxy_metrics"],
                "policy_path": report["policy_path"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
