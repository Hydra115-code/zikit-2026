from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.model import load_challenge_bundle
from app.challenge.policy import load_policy_artifact
from app.challenge.rehearsal import load_rehearsal_artifact
from app.challenge.service import ChallengeReplayService
from app.challenge.showcase import load_showcase_artifact
from app.challenge.training import discover_feature_files


DEPLOYMENT_BUNDLE_VERSION = "challenge-kaiku-deployment-bundle-v1"
DEPLOYMENT_MANIFEST = "deployment-manifest.json"


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


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != DEPLOYMENT_MANIFEST
    )


def verify_deployment_bundle(directory: Path) -> dict[str, Any]:
    root = assert_allowed_path(directory)
    manifest_path = root / DEPLOYMENT_MANIFEST
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise IngestionError("invalid deployment manifest") from exc
    if not isinstance(manifest, dict):
        raise IngestionError("invalid deployment manifest")
    if manifest.get("version") != DEPLOYMENT_BUNDLE_VERSION:
        raise IngestionError("unsupported deployment bundle version")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise IngestionError("deployment bundle has no declared files")
    actual = _relative_files(root)
    if actual != sorted(str(path) for path in declared):
        raise IngestionError("deployment bundle contains missing or unexpected files")
    for relative, expected_digest in declared.items():
        candidate = assert_allowed_path(root / str(relative))
        if root not in candidate.parents or not candidate.is_file():
            raise IngestionError("deployment manifest points outside the bundle")
        if _sha256_file(candidate) != expected_digest:
            raise IngestionError(f"deployment hash mismatch: {relative}")
    if manifest.get("status") != "passed":
        raise IngestionError("deployment bundle did not pass its gates")
    return manifest


def _filtered_feature_report(
    report: dict[str, Any], run_id: str, rows: int
) -> dict[str, Any]:
    filtered = dict(report)
    filtered["run_count"] = 1
    filtered["rows_input"] = rows
    filtered["rows_output"] = rows
    for key in ("per_run", "per_run_logs"):
        records = report.get(key, [])
        filtered[key] = [
            record
            for record in records
            if isinstance(record, dict) and record.get("run_id") == run_id
        ]
    filtered["deployment_subset"] = {
        "purpose": "official synthetic Kaiku guided replay",
        "run_ids": [run_id],
        "full_evaluation_packaged": False,
    }
    return filtered


def build_deployment_bundle(
    *,
    features_path: Path,
    model_path: Path,
    policy_path: Path,
    showcase_path: Path,
    rehearsal_path: Path,
    output_directory: Path,
    run_id: str,
) -> dict[str, Any]:
    features = assert_allowed_path(features_path)
    model = assert_allowed_path(model_path)
    policy = assert_allowed_path(policy_path)
    showcase_source = assert_allowed_path(showcase_path)
    rehearsal_source = assert_allowed_path(rehearsal_path)
    output = assert_allowed_path(output_directory)

    feature_files, feature_report, feature_manifest = discover_feature_files(
        features
    )
    if run_id not in feature_files:
        raise IngestionError(f"deployment run is not available: {run_id}")
    selected_record = next(
        (
            record
            for record in feature_manifest.get("feature_files", [])
            if record.get("run_id") == run_id
        ),
        None,
    )
    if not isinstance(selected_record, dict):
        raise IngestionError("deployment feature record is missing")
    selected_feature = feature_files[run_id]
    if _sha256_file(selected_feature) != selected_record.get("sha256"):
        raise IngestionError("deployment feature source hash mismatch")
    source_log_record = feature_manifest.get("log_events")
    if not isinstance(source_log_record, dict):
        raise IngestionError("deployment causal log evidence is missing")
    source_log_events = assert_allowed_path(
        features / str(source_log_record.get("path", ""))
    )
    if (
        not source_log_events.is_file()
        or _sha256_file(source_log_events) != source_log_record.get("sha256")
    ):
        raise IngestionError("deployment causal log evidence hash mismatch")

    showcase = load_showcase_artifact(showcase_source)
    rehearsal = load_rehearsal_artifact(rehearsal_source)
    if showcase.get("status") != "passed" or rehearsal.get("status") != "passed":
        raise IngestionError("deployment evidence has not passed")
    if showcase.get("representative_case", {}).get("run_id") != run_id:
        raise IngestionError("deployment run does not match the showcase")
    if rehearsal.get("case", {}).get("run_id") != run_id:
        raise IngestionError("deployment run does not match the live rehearsal")

    model_digest = _sha256_file(model)
    scorer = load_challenge_bundle(model, trusted_root=model.parent)
    if scorer is None:
        raise FileNotFoundError(model)
    _, policy_payload = load_policy_artifact(
        policy,
        trusted_root=policy.parent,
        expected_model_sha256=model_digest,
    )
    if policy_payload.get("model_version") != scorer.model_version:
        raise IngestionError("deployment policy and model versions do not match")

    source_model_files = (
        model,
        model.with_suffix(model.suffix + ".sha256"),
        model.with_suffix(model.suffix + ".metadata.json"),
        policy,
        policy.with_suffix(policy.suffix + ".sha256"),
    )
    missing_model_files = [path.name for path in source_model_files if not path.is_file()]
    if missing_model_files:
        raise FileNotFoundError(", ".join(missing_model_files))

    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        selected_feature: output / "features" / "features" / selected_feature.name,
        showcase_source: output / "evidence" / "showcase.json",
        showcase_source.with_name("showcase-manifest.json"): (
            output / "evidence" / "showcase-manifest.json"
        ),
        rehearsal_source: output / "evidence" / "rehearsal-report.json",
        rehearsal_source.with_name("rehearsal-manifest.json"): (
            output / "evidence" / "rehearsal-manifest.json"
        ),
    }
    for source in source_model_files:
        destinations[source] = output / "models" / source.name
    for source, destination in destinations.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        _copy(source, destination)

    runtime_features_directory = output / "features" / "features"
    for stale_feature in runtime_features_directory.glob("*.parquet"):
        if stale_feature != runtime_features_directory / selected_feature.name:
            stale_feature.unlink()

    rows = int(selected_record.get("rows", 0))
    runtime_report = _filtered_feature_report(feature_report, run_id, rows)
    runtime_report_path = output / "features" / "feature-report.json"
    _atomic_json(runtime_report, runtime_report_path)
    runtime_feature_path = output / "features" / "features" / selected_feature.name
    runtime_log_path = output / "features" / "log-events.parquet"
    selected_logs = pd.read_parquet(source_log_events)
    selected_logs = selected_logs.loc[
        selected_logs["run_id"].astype(str).eq(run_id)
        & selected_logs["tenant"].astype(str).eq("kaiku")
    ].copy()
    if selected_logs.empty:
        raise IngestionError("deployment run has no causal log evidence")
    temporary_log_path = runtime_log_path.with_name(
        f".{runtime_log_path.name}.tmp.parquet"
    )
    selected_logs.to_parquet(temporary_log_path, index=False)
    os.replace(temporary_log_path, runtime_log_path)
    runtime_feature_manifest = {
        "feature_version": feature_manifest["feature_version"],
        "scope": feature_manifest["scope"],
        "source": {
            "full_feature_manifest_sha256": _sha256_file(
                features / "manifest.json"
            ),
            "full_feature_report_sha256": _sha256_file(
                features / "feature-report.json"
            ),
            "subset_reason": "guided deployment replay only",
        },
        "feature_files": [
            {
                "path": f"features/{runtime_feature_path.name}",
                "rows": rows,
                "run_id": run_id,
                "sha256": _sha256_file(runtime_feature_path),
            }
        ],
        "report": {
            "path": runtime_report_path.name,
            "sha256": _sha256_file(runtime_report_path),
        },
        "logs": [],
        "log_events": {
            "path": runtime_log_path.name,
            "rows": len(selected_logs),
            "sha256": _sha256_file(runtime_log_path),
            "raw_messages_stored": False,
        },
    }
    _atomic_json(runtime_feature_manifest, output / "features" / "manifest.json")

    runtime_model = output / "models" / model.name
    runtime_policy = output / "models" / policy.name
    execution = ChallengeReplayService(
        output / "features",
        runtime_model,
        runtime_policy,
        require_all_tenants=False,
    )._execute(run_id)
    incident_alerts = execution.alerts.loc[
        execution.alerts["alert_type"].eq("incident_risk")
    ]
    quality_alerts = execution.alerts.loc[
        execution.alerts["alert_type"].eq("data_quality_alert")
    ]
    representative = showcase["representative_case"]
    representative_matches = bool(
        (
            pd.to_datetime(incident_alerts["timestamp"])
            == pd.Timestamp(representative["alert_timestamp"])
        )
        .where(
            incident_alerts["suspected_incident_type"].astype(str).eq(
                str(representative["suspected_incident_type"])
            ),
            False,
        )
        .any()
    )
    gates = {
        "source_integrity": True,
        "model_policy_binding": True,
        "signed_showcase": True,
        "passed_live_rehearsal": True,
        "runtime_replay": len(execution.alerts) > 0,
        "representative_alert_parity": representative_matches,
        "single_declared_run": run_id == str(execution.run_id),
        "no_action_claim": (
            rehearsal["limitations"]["mitigation_executed"] is False
            and rehearsal["limitations"]["recovery_verified"] is False
        ),
    }
    if not all(gates.values()):
        raise IngestionError("deployment bundle failed semantic parity gates")

    expected_files = sorted(
        {
            destination.relative_to(output).as_posix()
            for destination in destinations.values()
        }
        | {
            "features/feature-report.json",
            "features/manifest.json",
            "features/log-events.parquet",
        }
    )
    actual_files = _relative_files(output)
    unexpected = sorted(set(actual_files) - set(expected_files))
    missing = sorted(set(expected_files) - set(actual_files))
    if unexpected or missing:
        raise IngestionError(
            f"deployment output differs from allowlist; missing={missing}, "
            f"unexpected={unexpected}"
        )
    files = {
        relative: _sha256_file(output / relative) for relative in expected_files
    }
    manifest = {
        "version": DEPLOYMENT_BUNDLE_VERSION,
        "status": "passed",
        "scope": "minimal deployable runtime for official synthetic Kaiku replay",
        "run_id": run_id,
        "gates": gates,
        "replay": {
            "rows_processed": execution.rows_processed,
            "incident_alerts": len(incident_alerts),
            "data_quality_alerts": len(quality_alerts),
            "model_sha256": execution.model_sha256,
            "policy_sha256": execution.policy_sha256,
        },
        "runtime_environment": {
            "CHALLENGE_FEATURES_PATH": "/app/runtime/features",
            "CHALLENGE_MODEL_PATH": f"/app/runtime/models/{model.name}",
            "CHALLENGE_POLICY_PATH": f"/app/runtime/models/{policy.name}",
            "CHALLENGE_SHOWCASE_PATH": "/app/runtime/evidence/showcase.json",
            "CHALLENGE_REHEARSAL_PATH": (
                "/app/runtime/evidence/rehearsal-report.json"
            ),
        },
        "limitations": {
            "packaged_runs": 1,
            "full_54_run_evaluation_packaged": False,
            "remote_infrastructure_observed": False,
            "automatic_mitigation": False,
        },
        "files": files,
    }
    _atomic_json(manifest, output / DEPLOYMENT_MANIFEST)
    return verify_deployment_bundle(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the minimal signed Kaiku deployment runtime"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("data/challenge-phase-3-logs")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/challenge_early_warning.joblib")
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("models/challenge_alert_policy.json")
    )
    parser.add_argument(
        "--showcase", type=Path, default=Path("data/challenge-phase-9/showcase.json")
    )
    parser.add_argument(
        "--rehearsal",
        type=Path,
        default=Path("data/challenge-phase-10/rehearsal-report.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("deploy/challenge-runtime")
    )
    parser.add_argument(
        "--run-id",
        help="Run sintético a empaquetar; por defecto usa el caso firmado del showcase",
    )
    args = parser.parse_args()
    run_id = args.run_id
    if run_id is None:
        showcase = load_showcase_artifact(args.showcase)
        run_id = str(showcase["representative_case"]["run_id"])
    result = build_deployment_bundle(
        features_path=args.features,
        model_path=args.model,
        policy_path=args.policy,
        showcase_path=args.showcase,
        rehearsal_path=args.rehearsal,
        output_directory=args.output,
        run_id=run_id,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "gates": result["gates"],
                "replay": result["replay"],
                "files": len(result["files"]),
                "output": str(assert_allowed_path(args.output)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
