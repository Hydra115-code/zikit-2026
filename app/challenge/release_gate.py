from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.contracts import TARGET_TENANT, TENANT_SET
from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.model import load_challenge_bundle
from app.challenge.operations_audit import audit_durable_operations
from app.challenge.policy import load_policy_artifact
from app.challenge.replay import (
    _atomic_json,
    _replay_run,
    _validate_run_frame,
    replay_feature_dataset,
)
from app.challenge.response import build_response_audit
from app.challenge.submission import (
    SUBMISSION_PACKAGE_MANIFEST,
    validate_aggregate_submission_file,
    validate_submission_package,
)
from app.challenge.training import discover_feature_files


RELEASE_GATE_VERSION = "challenge-kaiku-release-gate-v7"
RELEASE_PACKAGE_VERSION = "challenge-kaiku-release-package-v1"
DETERMINISTIC_REPLAY_FILES = (
    "alerts.csv",
    "alerts.full.parquet",
    "data-quality-alerts.csv",
    "replay-report.json",
    SUBMISSION_PACKAGE_MANIFEST,
)
FORBIDDEN_RELEASE_FILENAMES = frozenset(
    {
        ".env",
        ".env.local",
        "etiquetas.csv",
        "generar.py",
        "todos_metricas.csv",
    }
)
FORBIDDEN_RELEASE_SUFFIXES = frozenset(
    {".db", ".key", ".pem", ".sqlite", ".sqlite3"}
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(assert_allowed_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IngestionError(f"expected a JSON object: {path.name}")
    return value


def verify_release_package(
    output_directory: Path,
    *,
    expected_runs: int = 54,
) -> dict[str, Any]:
    """Verify that the release is complete, immutable and contains no extras."""
    output = assert_allowed_path(output_directory)
    manifest_path = output / "manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise IngestionError("release manifest must list package files")

    invalid_paths: list[str] = []
    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    forbidden_files: list[str] = []
    expected_files: set[str] = {"manifest.json"}
    for raw_relative, expected_digest in sorted(files.items()):
        relative = Path(str(raw_relative))
        relative_text = relative.as_posix()
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative_text != str(raw_relative)
        ):
            invalid_paths.append(str(raw_relative))
            continue
        raw_candidate = output / relative
        candidate = raw_candidate.resolve(strict=False)
        try:
            candidate.relative_to(output)
        except ValueError:
            invalid_paths.append(str(raw_relative))
            continue
        expected_files.add(relative_text)
        if (
            {part.casefold() for part in relative.parts}
            & {"_solucion", ".replay-work"}
            or relative.name.casefold() in FORBIDDEN_RELEASE_FILENAMES
            or relative.suffix.casefold() in FORBIDDEN_RELEASE_SUFFIXES
        ):
            forbidden_files.append(relative_text)
        if not candidate.is_file() or raw_candidate.is_symlink():
            missing_files.append(relative_text)
        elif (
            not isinstance(expected_digest, str)
            or _sha256_file(candidate) != expected_digest
        ):
            hash_mismatches.append(relative_text)

    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    unmanifested_files = sorted(actual_files - expected_files)
    omitted_files = sorted(expected_files - actual_files)
    forbidden_files.extend(
        relative
        for relative in sorted(actual_files)
        if (
            {part.casefold() for part in Path(relative).parts}
            & {"_solucion", ".replay-work"}
            or Path(relative).name.casefold() in FORBIDDEN_RELEASE_FILENAMES
            or Path(relative).suffix.casefold() in FORBIDDEN_RELEASE_SUFFIXES
        )
        and relative not in forbidden_files
    )
    try:
        submission_contract = validate_submission_package(
            output / "replay",
            expected_runs=expected_runs,
        )
    except (IngestionError, FileNotFoundError, KeyError, ValueError) as exc:
        submission_contract = {
            "passed": False,
            "error": str(exc),
        }
    checks = {
        "package_version_matches": (
            manifest.get("package_version") == RELEASE_PACKAGE_VERSION
        ),
        "release_gate_version_matches": (
            manifest.get("release_gate_version") == RELEASE_GATE_VERSION
        ),
        "manifest_is_regular_file": (
            manifest_path.is_file() and not manifest_path.is_symlink()
        ),
        "release_status_passed": manifest.get("status") == "passed",
        "manifest_paths_safe": not invalid_paths,
        "all_hashes_match": not missing_files and not hash_mismatches,
        "package_has_no_unmanifested_files": not unmanifested_files,
        "package_has_no_omitted_files": not omitted_files,
        "package_has_no_forbidden_files": not forbidden_files,
        "submission_contract_passed": bool(submission_contract.get("passed")),
    }
    return {
        "version": "challenge-kaiku-release-verification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "files_verified": len(files),
        "invalid_paths": invalid_paths,
        "missing_files": missing_files,
        "hash_mismatches": hash_mismatches,
        "unmanifested_files": unmanifested_files,
        "omitted_files": omitted_files,
        "forbidden_files": sorted(forbidden_files),
        "submission_contract": submission_contract,
    }


def compare_replay_manifests(
    baseline_manifest: dict[str, Any],
    candidate_directory: Path,
) -> dict[str, Any]:
    expected = baseline_manifest.get("outputs", {})
    comparisons: dict[str, Any] = {}
    for filename in DETERMINISTIC_REPLAY_FILES:
        candidate = assert_allowed_path(candidate_directory / filename)
        expected_digest = expected.get(filename)
        actual_digest = _sha256_file(candidate) if candidate.is_file() else None
        comparisons[filename] = {
            "expected_sha256": expected_digest,
            "actual_sha256": actual_digest,
            "matches": bool(expected_digest and expected_digest == actual_digest),
        }
    return {
        "matches": all(item["matches"] for item in comparisons.values()),
        "files": comparisons,
    }


def forbidden_reference_audit(
    phase_one_manifest: dict[str, Any],
    source_files: list[Path],
) -> dict[str, Any]:
    consumed_files = phase_one_manifest.get("input", {}).get("files", [])
    source_names = [
        str(item.get("source_name", ""))
        for item in consumed_files
        if isinstance(item, dict)
    ]
    serialized = json.dumps(source_names, ensure_ascii=False).casefold()
    manifest_forbidden = [
        token
        for token in ("_solucion", "generar.py", "etiquetas.csv", "todos_metricas.csv")
        if token in serialized
    ]
    forbidden_imports: list[str] = []
    for source in source_files:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == "scripts.generar" or module.startswith("scripts.generar.")
                for module in modules
            ):
                forbidden_imports.append(f"{source.name}:{node.lineno}")
    return {
        "passed": not manifest_forbidden and not forbidden_imports,
        "manifest_forbidden_references": manifest_forbidden,
        "forbidden_generator_imports": forbidden_imports,
    }


def _artifact_integrity(model_path: Path, policy_path: Path) -> dict[str, Any]:
    model = assert_allowed_path(model_path)
    policy = assert_allowed_path(policy_path)
    model_digest = _sha256_file(model)
    policy_digest = _sha256_file(policy)
    model_sidecar = model.with_suffix(model.suffix + ".sha256")
    policy_sidecar = policy.with_suffix(policy.suffix + ".sha256")
    metadata_path = model.with_suffix(model.suffix + ".metadata.json")
    metadata = _load_json(metadata_path)
    policy_payload = _load_json(policy)
    checks = {
        "model_sidecar_matches": (
            model_sidecar.is_file()
            and model_sidecar.read_text(encoding="utf-8").strip() == model_digest
        ),
        "policy_sidecar_matches": (
            policy_sidecar.is_file()
            and policy_sidecar.read_text(encoding="utf-8").strip() == policy_digest
        ),
        "metadata_matches_model": metadata.get("sha256") == model_digest,
        "policy_bound_to_model": policy_payload.get("model_sha256") == model_digest,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "model_sha256": model_digest,
        "policy_sha256": policy_digest,
        "model_version": metadata.get("model_version"),
        "policy_version": policy_payload.get("policy_version"),
    }


def _causality_probe(
    feature_path: Path,
    run_id: str,
    scorer: Any,
    policy_config: Any,
) -> dict[str, Any]:
    frame = _validate_run_frame(
        pd.read_parquet(feature_path),
        run_id,
        require_all_tenants=True,
    )
    tenant = TARGET_TENANT
    original = frame.loc[frame["tenant"].eq(tenant)].reset_index(drop=True)
    cutoff = len(original) // 2
    cutoff_timestamp = pd.Timestamp(original.at[cutoff, "timestamp"])
    altered = original.copy()
    for feature in scorer.feature_names:
        if feature not in altered.columns:
            continue
        if not pd.api.types.is_numeric_dtype(altered[feature]):
            continue
        if pd.api.types.is_bool_dtype(altered[feature]):
            altered.loc[cutoff + 1 :, feature] = ~altered.loc[
                cutoff + 1 :, feature
            ].astype(bool)
        else:
            altered.loc[cutoff + 1 :, feature] = (
                pd.to_numeric(altered.loc[cutoff + 1 :, feature], errors="coerce")
                + 1000
            )

    original_scores = scorer.score_frame(original)
    altered_scores = scorer.score_frame(altered)
    score_delta = (
        original_scores.iloc[: cutoff + 1]
        .subtract(altered_scores.iloc[: cutoff + 1])
        .abs()
        .to_numpy()
        .max()
    )
    original_alerts, _, _ = _replay_run(
        original,
        scorer,
        policy_config,
        model_version=scorer.model_version,
    )
    altered_alerts, _, _ = _replay_run(
        altered,
        scorer,
        policy_config,
        model_version=scorer.model_version,
    )
    original_past = original_alerts.loc[
        original_alerts["timestamp"].le(cutoff_timestamp)
    ].reset_index(drop=True)
    altered_past = altered_alerts.loc[
        altered_alerts["timestamp"].le(cutoff_timestamp)
    ].reset_index(drop=True)
    original_semantic = original_past.astype(object).where(
        original_past.notna(), None
    )
    altered_semantic = altered_past.astype(object).where(
        altered_past.notna(), None
    )
    alerts_equal = original_semantic.equals(altered_semantic)
    return {
        "passed": float(score_delta) == 0.0 and alerts_equal,
        "run_id": run_id,
        "tenant": tenant,
        "cutoff_timestamp": cutoff_timestamp.isoformat(),
        "future_rows_modified": int(len(original) - cutoff - 1),
        "max_past_score_delta": float(score_delta),
        "past_alerts_unchanged": alerts_equal,
    }


def _context_ablation(
    feature_directory: Path,
    scorer: Any,
    policy_config: Any,
    normal_alerts: pd.DataFrame,
    *,
    expected_runs: int,
) -> dict[str, Any]:
    files, _, _ = discover_feature_files(
        feature_directory, expected_runs=expected_runs
    )
    ablated_frames: list[pd.DataFrame] = []
    known_event_keys: set[tuple[str, str, pd.Timestamp]] = set()
    context_columns: set[str] = set()
    for run_id, feature_path in sorted(files.items()):
        frame = _validate_run_frame(
            pd.read_parquet(feature_path),
            run_id,
            require_all_tenants=True,
        )
        known = frame.loc[
            frame["context_known_event"].astype(bool),
            ["run_id", "tenant", "timestamp"],
        ]
        known_event_keys.update(
            (str(row.run_id), str(row.tenant), pd.Timestamp(row.timestamp))
            for row in known.itertuples(index=False)
        )
        ablated = frame.copy()
        for column in ablated.columns:
            if column.startswith("context_"):
                context_columns.add(column)
                ablated[column] = False
        alerts, _, _ = _replay_run(
            ablated,
            scorer,
            policy_config,
            model_version=scorer.model_version,
        )
        ablated_frames.append(alerts)

    without_context = pd.concat(ablated_frames, ignore_index=True)
    normal_incident = normal_alerts.loc[
        normal_alerts["alert_type"].eq("incident_risk")
    ].copy()
    ablated_incident = without_context.loc[
        without_context["alert_type"].eq("incident_risk")
    ].copy()

    def identities(frame: pd.DataFrame) -> set[tuple[str, str, pd.Timestamp]]:
        return {
            (str(row.run_id), str(row.tenant), pd.Timestamp(row.timestamp))
            for row in frame.itertuples(index=False)
        }

    normal_keys = identities(normal_incident)
    ablated_keys = identities(ablated_incident)
    ablated_known_event_alerts = len(ablated_keys & known_event_keys)
    normal_known_event_alerts = len(normal_keys & known_event_keys)
    return {
        "status": "completed",
        "scope": "inference and alert-policy ablation; frozen model and thresholds",
        "context_columns_disabled": sorted(context_columns),
        "with_context": {
            "incident_alerts": len(normal_incident),
            "known_event_incident_alerts": normal_known_event_alerts,
        },
        "without_context": {
            "incident_alerts": len(ablated_incident),
            "known_event_incident_alerts": ablated_known_event_alerts,
        },
        "difference": {
            "incident_alerts": len(ablated_incident) - len(normal_incident),
            "known_event_incident_alerts": (
                ablated_known_event_alerts - normal_known_event_alerts
            ),
            "alerts_only_without_context": len(ablated_keys - normal_keys),
            "alerts_only_with_context": len(normal_keys - ablated_keys),
        },
        "interpretation_limit": (
            "This measures alert behavior, not official precision or recall; evaluator "
            "ground truth was not accessed."
        ),
    }


def _log_evidence_ablation(
    feature_directory: Path,
    scorer: Any,
    policy_config: Any,
    normal_alerts: pd.DataFrame,
    *,
    expected_runs: int,
) -> dict[str, Any]:
    files, feature_report, _ = discover_feature_files(
        feature_directory, expected_runs=expected_runs
    )
    log_columns = sorted(
        column
        for column in feature_report.get("feature_columns", [])
        if str(column).startswith("log_")
    )
    if not log_columns:
        return {
            "status": "missing",
            "passed": False,
            "reason": "feature dataset contains no causal log columns",
        }

    ablated_frames: list[pd.DataFrame] = []
    for run_id, feature_path in sorted(files.items()):
        frame = _validate_run_frame(
            pd.read_parquet(feature_path),
            run_id,
            require_all_tenants=True,
        )
        ablated = frame.copy()
        for column in log_columns:
            if column not in ablated.columns:
                continue
            ablated[column] = np.nan if column.endswith("_minutes_since") else 0.0
        alerts, _, _ = _replay_run(
            ablated,
            scorer,
            policy_config,
            model_version=scorer.model_version,
        )
        ablated_frames.append(alerts)

    without_logs = pd.concat(ablated_frames, ignore_index=True)
    normal_incident = normal_alerts.loc[
        normal_alerts["alert_type"].eq("incident_risk")
    ].copy()
    ablated_incident = without_logs.loc[
        without_logs["alert_type"].eq("incident_risk")
    ].copy()
    key_columns = ["run_id", "tenant", "signal_timestamp", "timestamp", "risk_score"]
    normal_keys = normal_incident.loc[:, key_columns].sort_values(
        key_columns[:-1], kind="stable", ignore_index=True
    )
    ablated_keys = ablated_incident.loc[:, key_columns].sort_values(
        key_columns[:-1], kind="stable", ignore_index=True
    )
    risk_decisions_unchanged = normal_keys.equals(ablated_keys)

    def log_evidence_count(frame: pd.DataFrame) -> int:
        count = 0
        for raw in frame["evidence"]:
            payload = json.loads(str(raw))
            if isinstance(payload, list) and any(
                isinstance(item, dict)
                and str(item.get("signal", "")).startswith("log_")
                for item in payload
            ):
                count += 1
        return count

    with_logs_typed = int(normal_incident["suspected_incident_type"].notna().sum())
    without_logs_typed = int(ablated_incident["suspected_incident_type"].notna().sum())
    return {
        "status": "completed",
        "passed": risk_decisions_unchanged and with_logs_typed >= without_logs_typed,
        "log_columns_disabled": len(log_columns),
        "risk_decisions_unchanged": risk_decisions_unchanged,
        "with_logs": {
            "incident_alerts": int(len(normal_incident)),
            "typed_incident_alerts": with_logs_typed,
            "alerts_with_log_evidence": log_evidence_count(normal_incident),
        },
        "without_logs": {
            "incident_alerts": int(len(ablated_incident)),
            "typed_incident_alerts": without_logs_typed,
            "alerts_with_log_evidence": log_evidence_count(ablated_incident),
        },
        "role": "incident typing and evidence only; risk model remains metrics-only",
    }


def _copy_release_artifacts(
    output_directory: Path,
    replay_directory: Path,
    model_path: Path,
    policy_path: Path,
) -> list[Path]:
    copied: list[Path] = []
    for filename in (*DETERMINISTIC_REPLAY_FILES, "benchmark.json"):
        source = replay_directory / filename
        destination = output_directory / "replay" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(destination)
    submissions_manifest = _load_json(
        replay_directory / SUBMISSION_PACKAGE_MANIFEST
    )
    for record in submissions_manifest.get("submissions", []):
        relative = Path(str(record.get("path", "")))
        if relative.parts[:1] != ("submissions",) or ".." in relative.parts:
            raise IngestionError("invalid evaluator submission path in replay manifest")
        source = replay_directory / relative
        destination = output_directory / "replay" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(destination)
    artifact_paths = (
        model_path,
        model_path.with_suffix(model_path.suffix + ".sha256"),
        model_path.with_suffix(model_path.suffix + ".metadata.json"),
        policy_path,
        policy_path.with_suffix(policy_path.suffix + ".sha256"),
    )
    for source in artifact_paths:
        destination = output_directory / "artifacts" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(destination)
    return copied


def build_end_to_end_evidence(
    replay_result: dict[str, Any],
    submission_contract: dict[str, Any],
    response_audit: dict[str, Any],
    durable_operations: dict[str, Any],
) -> dict[str, Any]:
    alerts = replay_result["alerts"]
    evidence = replay_result["evidence"]
    latency = alerts["decision_latency_minutes"]
    durability_passed = durable_operations.get("status") == "passed"
    stages = {
        "first_signal": {
            "status": "observed",
            "evidence_complete": evidence["alerts_without_evidence"] == 0,
            "clock_domain": "synthetic_replay_time",
        },
        "detection": {
            "status": "alert_emitted",
            "incident_alerts": alerts["incident_alerts"],
            "median_signal_to_detection_minutes": latency["median"],
            "duplicates": alerts["duplicate_alerts"],
            "known_event_alerts": alerts["known_event_incident_alerts"],
        },
        "decision": {
            "status": "contract_validated",
            "submission_contract_passed": submission_contract["passed"],
            "response_plans_passed": response_audit["passed"],
            "confidence_is_calibrated_probability": False,
        },
        "operator_response": {
            "status": "controlled_local_probe_passed" if durability_passed else "failed",
            "durable": durability_passed,
            "idempotent": durable_operations.get("gates", {})
            .get("idempotent_replay", {})
            .get("passed", False),
            "resumable": durable_operations.get("gates", {})
            .get("resumable_cursor", {})
            .get("passed", False),
            "mitigation_executed": False,
        },
        "verification": {
            "status": "contract_ready_execution_pending",
            "plans_have_verification_contract": response_audit.get("gates", {}).get(
                "all_plans_have_verification_contract", False
            ),
            "recovery_verified": False,
        },
    }
    passed = (
        stages["first_signal"]["evidence_complete"]
        and stages["detection"]["incident_alerts"] > 0
        and stages["detection"]["duplicates"] == 0
        and stages["detection"]["known_event_alerts"] == 0
        and stages["decision"]["submission_contract_passed"]
        and stages["decision"]["response_plans_passed"]
        and stages["operator_response"]["durable"]
        and stages["operator_response"]["idempotent"]
        and stages["operator_response"]["resumable"]
        and stages["verification"]["plans_have_verification_contract"]
    )
    return {
        "version": "challenge-kaiku-end-to-end-evidence-v1",
        "status": "release_ready_with_explicit_action_boundary" if passed else "failed",
        "passed": passed,
        "scope": "official synthetic Kaiku replay plus controlled local durability probe",
        "chain": stages,
        "claim": (
            "Sentinel detects causal warning signals, emits a validated alert, "
            "delivers an actionable response plan and durably records operator "
            "acknowledgement. It does not claim mitigation or recovery."
        ),
        "limitations": {
            "official_ground_truth_used": False,
            "remote_infrastructure_observed": False,
            "automatic_action_executed": False,
            "recovery_verified": False,
            "cross_clock_signal_to_acknowledgement_reported": False,
        },
    }


def run_challenge_release_gate(
    feature_directory: Path,
    model_path: Path,
    policy_path: Path,
    phase_one_directory: Path,
    baseline_replay_directory: Path,
    training_report_path: Path,
    output_directory: Path,
    *,
    expected_runs: int = 54,
) -> dict[str, Any]:
    features = assert_allowed_path(feature_directory)
    model = assert_allowed_path(model_path)
    policy = assert_allowed_path(policy_path)
    phase_one = assert_allowed_path(phase_one_directory)
    baseline = assert_allowed_path(baseline_replay_directory)
    training_report = assert_allowed_path(training_report_path)
    output = assert_allowed_path(output_directory)
    replay_workspace = TemporaryDirectory(prefix="zikit-kaiku-release-replay-")
    replay_work = Path(replay_workspace.name)

    replay_result = replay_feature_dataset(
        features,
        model,
        policy,
        replay_work,
        expected_runs=expected_runs,
        require_all_tenants=True,
    )
    baseline_manifest = _load_json(baseline / "manifest.json")
    reproducibility = compare_replay_manifests(baseline_manifest, replay_work)
    integrity = _artifact_integrity(model, policy)
    scorer = load_challenge_bundle(model, trusted_root=model.parent)
    if scorer is None:
        raise FileNotFoundError(model)
    policy_config, _ = load_policy_artifact(
        policy,
        trusted_root=policy.parent,
        expected_model_sha256=integrity["model_sha256"],
    )
    files, feature_report, feature_manifest = discover_feature_files(
        features, expected_runs=expected_runs
    )
    first_run_id = sorted(files)[0]
    causality_probe = _causality_probe(
        files[first_run_id], first_run_id, scorer, policy_config
    )
    normal_alerts = pd.read_parquet(replay_work / "alerts.full.parquet")
    aggregate_submission_contract = validate_aggregate_submission_file(
        replay_work / "alerts.csv"
    )
    submission_contract = validate_submission_package(
        replay_work,
        expected_runs=expected_runs,
    )
    response_audit = build_response_audit(normal_alerts)
    durable_operations = audit_durable_operations(replay_work, output)
    context_ablation = _context_ablation(
        features,
        scorer,
        policy_config,
        normal_alerts,
        expected_runs=expected_runs,
    )
    log_ablation = _log_evidence_ablation(
        features,
        scorer,
        policy_config,
        normal_alerts,
        expected_runs=expected_runs,
    )

    phase_one_manifest = _load_json(phase_one / "manifest.json")
    quality_report = _load_json(phase_one / "quality-report.json")
    training = _load_json(training_report)
    source_files = sorted((Path(__file__).parent).glob("*.py"))
    forbidden = forbidden_reference_audit(phase_one_manifest, source_files)
    split_counts = training.get("splits", {}).get("counts", {})
    split_gate = {
        "passed": (
            split_counts == {"train": 36, "validation": 9, "test": 9}
            and training.get("splits", {}).get("overlap") == []
        ),
        "counts": split_counts,
        "overlap": training.get("splits", {}).get("overlap"),
        "seed": training.get("seed"),
    }
    coverage = replay_result["coverage"]
    replay_alerts = replay_result["alerts"]
    coverage_gate = {
        "passed": (
            coverage["run_count"] == expected_runs
            and coverage["tenant_count"] == len(TENANT_SET)
            and coverage["tenant_run_count"] == expected_runs * len(TENANT_SET)
            and coverage["all_tenants_required"] is True
        ),
        **coverage,
    }
    quality_gate = {
        "passed": (
            quality_report.get("integrity", {}).get("row_conservation") is True
            and quality_report.get("integrity", {}).get("identity_unique") is True
            and quality_report.get("data_quality", {}).get("outage", 0) > 0
            and quality_report.get("data_quality", {}).get("partial", 0) > 0
        ),
        "data_quality": quality_report.get("data_quality"),
        "missing_by_metric": quality_report.get("missing_by_metric"),
        "integrity": quality_report.get("integrity"),
    }
    replay_contract_gate = {
        "passed": (
            replay_alerts["duplicate_alerts"] == 0
            and replay_alerts["duplicate_episode_alerts"] == 0
            and replay_result["evidence"]["alerts_without_evidence"] == 0
            and replay_result["causality"]["future_rows_used"] is False
            and replay_result["causality"]["max_pointwise_score_delta"] <= 1e-7
        ),
        "alerts": replay_alerts,
        "causality": replay_result["causality"],
        "evidence": replay_result["evidence"],
    }
    durable_operations_gate = {
        "passed": durable_operations["status"] == "passed",
        "audit_version": durable_operations["audit_version"],
        "gates": durable_operations["gates"],
        "limitations": durable_operations["limitations"],
    }
    end_to_end_evidence = build_end_to_end_evidence(
        replay_result,
        submission_contract,
        response_audit,
        durable_operations,
    )
    gates = {
        "coverage": coverage_gate,
        "data_quality": quality_gate,
        "split_isolation": split_gate,
        "forbidden_data": forbidden,
        "artifact_integrity": integrity,
        "causal_future_mutation": causality_probe,
        "replay_contract": replay_contract_gate,
        "submission_contract": submission_contract,
        "aggregate_audit_contract": aggregate_submission_contract,
        "byte_reproducibility": {"passed": reproducibility["matches"], **reproducibility},
        "context_ablation": {
            "passed": (
                context_ablation["status"] == "completed"
                and context_ablation["with_context"][
                    "known_event_incident_alerts"
                ]
                <= context_ablation["without_context"][
                    "known_event_incident_alerts"
                ]
            ),
            **context_ablation,
        },
        "logs_ablation": log_ablation,
        "response_plans": {
            "passed": bool(response_audit["passed"]),
            **response_audit,
        },
        "durable_operations": durable_operations_gate,
        "end_to_end_evidence": {
            "passed": end_to_end_evidence["passed"],
            "status": end_to_end_evidence["status"],
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    report = {
        "release_gate_version": RELEASE_GATE_VERSION,
        "status": "passed" if passed else "failed",
        "scope": (
            "official synthetic Zikit public Kaiku runs; evaluator ground truth excluded"
        ),
        "gates": gates,
        "summary": {
            "gate_count": len(gates),
            "end_to_end_status": end_to_end_evidence["status"],
            "runs": coverage["run_count"],
            "tenants": coverage["tenant_count"],
            "rows": coverage["rows_processed"],
            "incident_alerts": replay_alerts["incident_alerts"],
            "data_quality_alerts": replay_alerts["data_quality_alerts"],
            "known_event_incident_alerts": replay_alerts[
                "known_event_incident_alerts"
            ],
        },
        "limitations": {
            "official_ground_truth_used": False,
            "official_recall_available": False,
            "official_lead_time_available": False,
            "automatic_action_executed": False,
            "operator_acknowledgement_demonstrated": True,
            "effective_mitigation_demonstrated": False,
            "recovery_verified": False,
            "gemini_required": False,
        },
        "lineage": {
            "phase_one_manifest_sha256": _sha256_file(
                phase_one / "manifest.json"
            ),
            "phase_two_manifest_sha256": _sha256_file(features / "manifest.json"),
            "feature_report_sha256": _sha256_file(features / "feature-report.json"),
            "feature_file_count": len(feature_manifest.get("feature_files", [])),
            "feature_version": feature_report.get("feature_version"),
            "log_feature_version": feature_report.get("log_feature_version"),
            "model_sha256": integrity["model_sha256"],
            "policy_sha256": integrity["policy_sha256"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "release-report.json"
    ablation_path = output / "context-ablation.json"
    log_ablation_path = output / "log-ablation.json"
    response_audit_path = output / "response-plan-audit.json"
    operations_audit_path = output / "operations-audit.json"
    end_to_end_path = output / "end-to-end-evidence.json"
    _atomic_json(report, report_path)
    _atomic_json(context_ablation, ablation_path)
    _atomic_json(log_ablation, log_ablation_path)
    _atomic_json(response_audit, response_audit_path)
    _atomic_json(end_to_end_evidence, end_to_end_path)
    copied = _copy_release_artifacts(output, replay_work, model, policy)
    package_files = [
        report_path,
        ablation_path,
        log_ablation_path,
        response_audit_path,
        operations_audit_path,
        end_to_end_path,
        *copied,
    ]
    package_manifest = {
        "package_version": RELEASE_PACKAGE_VERSION,
        "release_gate_version": RELEASE_GATE_VERSION,
        "status": report["status"],
        "files": {
            str(path.relative_to(output)): _sha256_file(path)
            for path in sorted(package_files)
        },
        "non_deterministic_files": [
            "replay/benchmark.json",
            "operations-audit.json",
        ],
    }
    _atomic_json(package_manifest, output / "manifest.json")
    package_verification = verify_release_package(
        output,
        expected_runs=expected_runs,
    )
    replay_workspace.cleanup()
    if not package_verification["passed"]:
        raise IngestionError(
            "release package verification failed: "
            + json.dumps(package_verification["checks"], sort_keys=True)
        )
    return {
        **report,
        "package_verification": package_verification,
        "output_directory": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and verify the final Zikit challenge backend package"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--phase-one", type=Path, required=True)
    parser.add_argument("--baseline-replay", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=54)
    parser.add_argument("--fail-on-gates", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing --output package without rebuilding it",
    )
    args = parser.parse_args()
    if args.verify_only:
        verification = verify_release_package(
            args.output,
            expected_runs=args.expected_runs,
        )
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        if args.fail_on_gates and not verification["passed"]:
            raise SystemExit(1)
        return
    result = run_challenge_release_gate(
        args.features,
        args.model,
        args.policy,
        args.phase_one,
        args.baseline_replay,
        args.training_report,
        args.output,
        expected_runs=args.expected_runs,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
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
