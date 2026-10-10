from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.contracts import TENANTS, TENANT_SET
from app.challenge.features import FEATURE_SCHEMA_VERSION
from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.model import load_challenge_bundle
from app.challenge.policy import (
    ALERT_COLUMNS,
    POLICY_VERSION,
    PolicyConfig,
    REQUIRED_SCORE_COLUMNS,
    load_policy_artifact,
    replay_policy_for_tenant,
)
from app.challenge.response import build_response_audit
from app.challenge.submission import (
    AGGREGATE_SUBMISSION_COLUMNS,
    DELIVERY_CONTRACT_VERSION,
    SUBMISSION_PACKAGE_MANIFEST,
    SUBMISSION_COLUMNS,
    validate_aggregate_submission_file,
    validate_aggregate_submission_frame,
    validate_submission_package,
    validate_submission_file,
    validate_submission_frame,
)
from app.challenge.training import discover_feature_files


REPLAY_VERSION = "challenge-kaiku-causal-replay-v3"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.8f",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _validate_run_frame(
    frame: pd.DataFrame,
    run_id: str,
    *,
    require_all_tenants: bool,
) -> pd.DataFrame:
    required = {
        "feature_version",
        "run_id",
        "tenant",
        "timestamp",
        "data_quality",
        "context_known_event",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise IngestionError(f"replay input is missing columns: {missing}")
    if set(frame["feature_version"].astype(str).unique()) != {FEATURE_SCHEMA_VERSION}:
        raise IngestionError(f"feature version mismatch in {run_id}")
    actual_run_ids = set(frame["run_id"].astype(str).unique())
    if actual_run_ids != {run_id}:
        raise IngestionError(
            f"feature file identity mismatch for {run_id}: {sorted(actual_run_ids)}"
        )
    tenants = set(frame["tenant"].astype(str).unique())
    if not tenants.issubset(TENANT_SET):
        raise IngestionError(f"unsupported tenants in {run_id}: {sorted(tenants)}")
    if require_all_tenants and tenants != TENANT_SET:
        raise IngestionError(
            f"replay run {run_id} does not contain all tenants: "
            f"{sorted(TENANT_SET - tenants)}"
        )
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    ordered = ordered.sort_values(
        ["tenant", "timestamp"], kind="stable", ignore_index=True
    )
    if ordered.duplicated(["run_id", "tenant", "timestamp"]).any():
        raise IngestionError(f"duplicate replay identity in {run_id}")
    return ordered


def _tenant_days(frame: pd.DataFrame) -> float:
    duration = (
        pd.Timestamp(frame["timestamp"].max())
        - pd.Timestamp(frame["timestamp"].min())
    ).total_seconds() / 86_400
    return max(duration + 1 / 1440, 1 / 1440)


def _replay_run(
    frame: pd.DataFrame,
    scorer: Any,
    config: PolicyConfig,
    *,
    model_version: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float | int]]:
    alert_frames: list[pd.DataFrame] = []
    tenant_reports: list[dict[str, Any]] = []
    scoring_seconds = 0.0
    policy_seconds = 0.0
    pointwise_verification_seconds = 0.0
    pointwise_equivalence_checks = 0
    max_pointwise_score_delta = 0.0

    for tenant, tenant_frame in frame.groupby("tenant", sort=True):
        tenant_frame = tenant_frame.reset_index(drop=True)
        started = perf_counter()
        scores = scorer.score_frame(tenant_frame)
        scoring_seconds += perf_counter() - started

        sample_position = len(tenant_frame) // 2
        started = perf_counter()
        pointwise = scorer.score_frame(
            tenant_frame.iloc[[sample_position]].reset_index(drop=True)
        )
        pointwise_verification_seconds += perf_counter() - started
        batch_values = scores.iloc[sample_position].loc[
            list(REQUIRED_SCORE_COLUMNS)
        ].to_numpy(dtype=float)
        pointwise_values = pointwise.iloc[0].loc[
            list(REQUIRED_SCORE_COLUMNS)
        ].to_numpy(dtype=float)
        delta = float(np.max(np.abs(batch_values - pointwise_values)))
        if not np.allclose(batch_values, pointwise_values, rtol=0, atol=1e-7):
            raise IngestionError(
                "batch scoring differs from isolated row scoring; causal replay aborted"
            )
        pointwise_equivalence_checks += 1
        max_pointwise_score_delta = max(max_pointwise_score_delta, delta)

        started = perf_counter()
        alerts = replay_policy_for_tenant(
            tenant_frame,
            scores,
            config,
            model_version=model_version,
        )
        policy_seconds += perf_counter() - started
        alert_frames.append(alerts)

        incident = alerts.loc[alerts["alert_type"].eq("incident_risk")]
        quality = alerts.loc[alerts["alert_type"].eq("data_quality_alert")]
        days = _tenant_days(tenant_frame)
        tenant_reports.append(
            {
                "run_id": str(tenant_frame.at[0, "run_id"]),
                "tenant": str(tenant),
                "rows": int(len(tenant_frame)),
                "tenant_days": days,
                "incident_alerts": int(len(incident)),
                "data_quality_alerts": int(len(quality)),
                "incident_alerts_per_day": float(len(incident) / days),
                "known_event_incident_alerts": int(
                    incident["context_known_event"].sum()
                ),
                "median_risk_score": (
                    float(incident["risk_score"].median())
                    if not incident.empty
                    else None
                ),
                "median_decision_latency_minutes": (
                    float(incident["decision_latency_minutes"].median())
                    if not incident.empty
                    else None
                ),
            }
        )

    alerts = (
        pd.concat(alert_frames, ignore_index=True)
        if alert_frames
        else pd.DataFrame(columns=ALERT_COLUMNS)
    )
    alerts = alerts.sort_values(
        ["run_id", "tenant", "timestamp", "alert_type"],
        kind="stable",
        ignore_index=True,
    )
    return alerts, tenant_reports, {
        "model_scoring_seconds": scoring_seconds,
        "policy_seconds": policy_seconds,
        "pointwise_verification_seconds": pointwise_verification_seconds,
        "pointwise_equivalence_checks": pointwise_equivalence_checks,
        "max_pointwise_score_delta": max_pointwise_score_delta,
    }


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }


def _evidence_summary(incident_alerts: pd.DataFrame) -> dict[str, Any]:
    signal_counts: Counter[str] = Counter()
    valid = 0
    for raw in incident_alerts["evidence"]:
        payload = json.loads(str(raw))
        if not isinstance(payload, list):
            raise IngestionError("incident evidence must be a JSON list")
        valid += 1
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("signal"), str):
                signal_counts[item["signal"]] += 1
    return {
        "alerts_with_valid_evidence": valid,
        "alerts_without_evidence": int(len(incident_alerts) - valid),
        "top_signals": [
            {"signal": signal, "occurrences": count}
            for signal, count in signal_counts.most_common(12)
        ],
    }


def _stability_summary(tenant_reports: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(tenant_reports)
    by_tenant: dict[str, Any] = {}
    for tenant in TENANTS:
        group = frame.loc[frame["tenant"].eq(tenant)]
        by_tenant[tenant] = {
            "runs": int(group["run_id"].nunique()),
            "runs_with_incident_alerts": int(group["incident_alerts"].gt(0).sum()),
            "incident_alerts_per_run": _distribution(group["incident_alerts"]),
            "incident_alerts_per_day": _distribution(
                group["incident_alerts_per_day"]
            ),
            "median_risk_score_across_runs": _distribution(
                group["median_risk_score"]
            ),
        }
    return by_tenant


def replay_feature_dataset(
    input_directory: Path,
    model_path: Path,
    policy_path: Path,
    output_directory: Path,
    *,
    expected_runs: int = 54,
    require_all_tenants: bool = True,
) -> dict[str, Any]:
    total_started = perf_counter()
    files, feature_report, _ = discover_feature_files(
        input_directory, expected_runs=expected_runs
    )
    causality = feature_report.get("causality", {})
    if causality.get("future_rows_used") is not False:
        raise IngestionError("replay requires a past-only feature dataset")
    if causality.get("past_baselines_exclude_current_sample") is not True:
        raise IngestionError("replay requires past-only baselines")

    model = assert_allowed_path(model_path)
    model_digest = _sha256_file(model)
    scorer = load_challenge_bundle(model, trusted_root=model.parent)
    if scorer is None:
        raise FileNotFoundError(model)

    policy = assert_allowed_path(policy_path)
    config, policy_payload = load_policy_artifact(
        policy,
        trusted_root=policy.parent,
        expected_model_sha256=model_digest,
    )
    if policy_payload.get("model_version") != scorer.model_version:
        raise IngestionError("policy and model versions do not match")

    all_alerts: list[pd.DataFrame] = []
    tenant_reports: list[dict[str, Any]] = []
    timing_by_run: list[dict[str, Any]] = []
    rows_processed = 0
    load_seconds = 0.0
    scoring_seconds = 0.0
    policy_seconds = 0.0
    pointwise_verification_seconds = 0.0
    pointwise_equivalence_checks = 0
    max_pointwise_score_delta = 0.0

    for run_id, feature_path in sorted(files.items()):
        run_started = perf_counter()
        started = perf_counter()
        raw = pd.read_parquet(feature_path)
        load_seconds += perf_counter() - started
        frame = _validate_run_frame(
            raw,
            run_id,
            require_all_tenants=require_all_tenants,
        )
        alerts, reports, timing = _replay_run(
            frame,
            scorer,
            config,
            model_version=scorer.model_version,
        )
        all_alerts.append(alerts)
        tenant_reports.extend(reports)
        rows_processed += len(frame)
        scoring_seconds += timing["model_scoring_seconds"]
        policy_seconds += timing["policy_seconds"]
        pointwise_verification_seconds += timing[
            "pointwise_verification_seconds"
        ]
        pointwise_equivalence_checks += int(
            timing["pointwise_equivalence_checks"]
        )
        max_pointwise_score_delta = max(
            max_pointwise_score_delta,
            float(timing["max_pointwise_score_delta"]),
        )
        timing_by_run.append(
            {
                "run_id": run_id,
                "rows": int(len(frame)),
                "wall_seconds": perf_counter() - run_started,
                **timing,
            }
        )

    alerts = pd.concat(all_alerts, ignore_index=True).sort_values(
        ["run_id", "tenant", "timestamp", "alert_type"],
        kind="stable",
        ignore_index=True,
    )
    duplicate_key = ["run_id", "tenant", "timestamp", "alert_type"]
    duplicate_count = int(alerts.duplicated(duplicate_key).sum())
    if duplicate_count:
        raise IngestionError("replay produced duplicate alerts")
    episode_key = ["run_id", "tenant", "signal_timestamp", "alert_type"]
    duplicate_episode_count = int(alerts.duplicated(episode_key).sum())
    if duplicate_episode_count:
        raise IngestionError("replay emitted more than one alert for an episode")

    incident = alerts.loc[alerts["alert_type"].eq("incident_risk")].copy()
    quality = alerts.loc[alerts["alert_type"].eq("data_quality_alert")].copy()
    if not incident["risk_score"].between(0, 1).all():
        raise IngestionError("replay produced risk scores outside zero and one")

    aggregate_submission = incident.loc[
        :, ["run_id", "tenant", "timestamp", "risk_score", "suspected_incident_type"]
    ].rename(columns={"risk_score": "confidence"})
    aggregate_submission = aggregate_submission.loc[:, AGGREGATE_SUBMISSION_COLUMNS]
    aggregate_validation = validate_aggregate_submission_frame(aggregate_submission)
    quality_export = quality.loc[
        :,
        [
            "run_id",
            "tenant",
            "signal_timestamp",
            "timestamp",
            "decision_latency_minutes",
            "data_quality",
            "evidence",
            "recommended_action",
        ],
    ]

    output = assert_allowed_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    alerts_path = output / "alerts.csv"
    full_path = output / "alerts.full.parquet"
    quality_path = output / "data-quality-alerts.csv"
    report_path = output / "replay-report.json"
    benchmark_path = output / "benchmark.json"
    submissions_manifest_path = output / SUBMISSION_PACKAGE_MANIFEST
    _atomic_csv(aggregate_submission, alerts_path)
    written_aggregate_validation = validate_aggregate_submission_file(alerts_path)
    if written_aggregate_validation["rows"] != aggregate_validation["rows"]:
        raise IngestionError("written aggregate alert export changed the row count")

    submission_records: list[dict[str, Any]] = []
    for run_id in sorted(files):
        evaluator_submission = aggregate_submission.loc[
            aggregate_submission["run_id"].eq(run_id), SUBMISSION_COLUMNS
        ].reset_index(drop=True)
        validate_submission_frame(evaluator_submission)
        submission_path = output / "submissions" / run_id / "alerts.csv"
        _atomic_csv(evaluator_submission, submission_path)
        written = validate_submission_file(submission_path)
        submission_records.append(
            {
                "run_id": run_id,
                "path": f"submissions/{run_id}/alerts.csv",
                "rows": written["rows"],
                "sha256": _sha256_file(submission_path),
            }
        )
    _atomic_json(
        {
            "contract_version": DELIVERY_CONTRACT_VERSION,
            "scope": "one evaluator-facing CSV per public Kaiku run",
            "columns": list(SUBMISSION_COLUMNS),
            "run_count": len(submission_records),
            "total_rows": int(sum(record["rows"] for record in submission_records)),
            "submissions": submission_records,
        },
        submissions_manifest_path,
    )
    submission_package_validation = validate_submission_package(
        output,
        expected_runs=expected_runs,
    )
    _atomic_parquet(alerts, full_path)
    _atomic_csv(quality_export, quality_path)

    tenant_frame = pd.DataFrame(tenant_reports)
    total_tenant_days = float(tenant_frame["tenant_days"].sum())
    report: dict[str, Any] = {
        "replay_version": REPLAY_VERSION,
        "feature_version": FEATURE_SCHEMA_VERSION,
        "model_version": scorer.model_version,
        "policy_version": POLICY_VERSION,
        "lineage": {
            "phase_two_manifest_sha256": _sha256_file(
                assert_allowed_path(input_directory) / "manifest.json"
            ),
            "model_sha256": model_digest,
            "policy_sha256": _sha256_file(policy),
        },
        "coverage": {
            "run_count": len(files),
            "tenant_count": int(tenant_frame["tenant"].nunique()),
            "tenant_run_count": int(len(tenant_reports)),
            "rows_processed": int(rows_processed),
            "tenant_days": total_tenant_days,
            "all_tenants_required": require_all_tenants,
        },
        "alerts": {
            "incident_alerts": int(len(incident)),
            "data_quality_alerts": int(len(quality)),
            "incident_alerts_per_tenant_day": (
                float(len(incident) / total_tenant_days)
                if total_tenant_days
                else 0.0
            ),
            "known_event_incident_alerts": int(
                incident["context_known_event"].sum()
            ),
            "duplicate_alerts": duplicate_count,
            "duplicate_episode_alerts": duplicate_episode_count,
            "incident_episode_count": int(len(incident)),
            "typed_incident_alerts": int(
                incident["suspected_incident_type"].notna().sum()
            ),
            "risk_score": _distribution(incident["risk_score"]),
            "decision_latency_minutes": _distribution(
                incident["decision_latency_minutes"]
            ),
        },
        "stability_by_tenant": _stability_summary(tenant_reports),
        "evidence": _evidence_summary(incident),
        "response_readiness": build_response_audit(alerts),
        "contract": {
            "version": DELIVERY_CONTRACT_VERSION,
            "evaluator_package_manifest": submissions_manifest_path.name,
            "evaluator_columns": list(SUBMISSION_COLUMNS),
            "evaluator_sorted_by": ["tenant", "timestamp"],
            "evaluator_validation": submission_package_validation,
            "aggregate_audit_file": alerts_path.name,
            "aggregate_columns": list(AGGREGATE_SUBMISSION_COLUMNS),
            "aggregate_sorted_by": ["run_id", "tenant", "timestamp"],
            "aggregate_validation": written_aggregate_validation,
            "confidence_semantics": (
                "risk_score between 0 and 1; not a calibrated probability of a real incident"
            ),
        },
        "causality": {
            "future_rows_used": False,
            "model_scored_in_batches": True,
            "pointwise_equivalence_checks": pointwise_equivalence_checks,
            "max_pointwise_score_delta": max_pointwise_score_delta,
            "policy_processed_minute_by_minute": True,
            "policy_state_shared_between_tenants": False,
            "model_or_policy_recalibrated": False,
        },
        "limitations": {
            "official_ground_truth_used": False,
            "official_recall_available": False,
            "official_lead_time_available": False,
            "automatic_remediation": False,
            "gemini_required": False,
        },
    }
    _atomic_json(report, report_path)

    total_seconds = perf_counter() - total_started
    benchmark = {
        "replay_version": REPLAY_VERSION,
        "scope": "local wall-clock measurement; intentionally non-deterministic",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "rows_processed": int(rows_processed),
        "wall_seconds": total_seconds,
        "rows_per_second": rows_processed / total_seconds if total_seconds else None,
        "mean_wall_milliseconds_per_row": (
            total_seconds * 1000 / rows_processed if rows_processed else None
        ),
        "feature_load_seconds": load_seconds,
        "model_scoring_seconds": scoring_seconds,
        "pointwise_verification_seconds": pointwise_verification_seconds,
        "policy_seconds": policy_seconds,
        "per_run": timing_by_run,
    }
    _atomic_json(benchmark, benchmark_path)

    manifest = {
        "replay_version": REPLAY_VERSION,
        "deterministic_outputs": [
            "alerts.csv",
            "alerts.full.parquet",
            "data-quality-alerts.csv",
            "replay-report.json",
            SUBMISSION_PACKAGE_MANIFEST,
        ],
        "outputs": {
            path.name: _sha256_file(path)
            for path in (
                alerts_path,
                full_path,
                quality_path,
                report_path,
                submissions_manifest_path,
                benchmark_path,
            )
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    return {
        **report,
        "benchmark": benchmark,
        "output_directory": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen Zikit features through the frozen model and policy"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=54)
    parser.add_argument("--allow-missing-tenants", action="store_true")
    args = parser.parse_args()
    if args.expected_runs < 1:
        parser.error("--expected-runs must be positive")
    result = replay_feature_dataset(
        args.input,
        args.model,
        args.policy,
        args.output,
        expected_runs=args.expected_runs,
        require_all_tenants=not args.allow_missing_tenants,
    )
    print(
        json.dumps(
            {
                "replay_version": result["replay_version"],
                "coverage": result["coverage"],
                "alerts": result["alerts"],
                "benchmark": {
                    "wall_seconds": result["benchmark"]["wall_seconds"],
                    "rows_per_second": result["benchmark"]["rows_per_second"],
                },
                "output_directory": result["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
