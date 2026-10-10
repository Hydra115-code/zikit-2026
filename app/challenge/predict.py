from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from app.challenge.features import build_feature_dataset
from app.challenge.ingestion import IngestionError, assert_allowed_path, ingest_dataset
from app.challenge.log_features import enrich_log_feature_dataset
from app.challenge.replay import replay_feature_dataset
from app.challenge.submission import (
    SUBMISSION_PACKAGE_MANIFEST,
    validate_submission_file,
    validate_submission_package,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def predict_run(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    policy_path: Path,
    *,
    require_all_tenants: bool = False,
) -> dict[str, Any]:
    source = assert_allowed_path(input_path)
    output = assert_allowed_path(output_path)
    model = assert_allowed_path(model_path)
    policy = assert_allowed_path(policy_path)
    if output.suffix.casefold() != ".csv":
        raise IngestionError("prediction output must be a CSV file")
    if source.resolve(strict=False) == output.resolve(strict=False):
        raise IngestionError("prediction output cannot overwrite its input")

    with TemporaryDirectory(prefix="zikit-predict-") as temporary_name:
        temporary = Path(temporary_name)
        phase_one = temporary / "phase-1"
        phase_two = temporary / "phase-2"
        phase_three = temporary / "phase-3"
        phase_five = temporary / "phase-5"
        ingestion = ingest_dataset(
            source,
            phase_one,
            expected_runs=1,
            require_all_tenants=require_all_tenants,
        )
        features = build_feature_dataset(
            phase_one,
            phase_two,
            expected_runs=1,
            require_all_tenants=require_all_tenants,
        )
        enriched = enrich_log_feature_dataset(
            source,
            phase_two,
            phase_one,
            phase_three,
            expected_runs=1,
        )
        replay = replay_feature_dataset(
            phase_three,
            model,
            policy,
            phase_five,
            expected_runs=1,
            require_all_tenants=require_all_tenants,
        )
        package_validation = validate_submission_package(
            phase_five,
            expected_runs=1,
        )
        package_manifest = json.loads(
            (phase_five / SUBMISSION_PACKAGE_MANIFEST).read_text(encoding="utf-8")
        )
        submission_record = package_manifest["submissions"][0]
        _atomic_copy(phase_five / submission_record["path"], output)
        submission_validation = validate_submission_file(output)
        _atomic_copy(
            phase_five / "data-quality-alerts.csv",
            output.with_name(f"{output.stem}.data-quality.csv"),
        )
        _atomic_copy(
            phase_five / "alerts.full.parquet",
            output.with_name(f"{output.stem}.audit.parquet"),
        )
        _atomic_copy(
            phase_five / "replay-report.json",
            output.with_name(f"{output.stem}.replay-report.json"),
        )

    quality_output = output.with_name(f"{output.stem}.data-quality.csv")
    audit_output = output.with_name(f"{output.stem}.audit.parquet")
    replay_report_output = output.with_name(f"{output.stem}.replay-report.json")

    manifest = {
        "command": "python -m app.challenge predict",
        "input": {
            "path": source.name,
            "files": ingestion["manifest"]["input"]["files"],
            "rows_received": ingestion["rows_received"],
            "rows_clean": ingestion["rows_clean"],
            "rows_quarantined": ingestion["rows_quarantined"],
        },
        "features": {
            "version": features["feature_version"],
            "log_version": enriched["log_feature_version"],
            "rows": enriched["rows_output"],
            "future_rows_used": enriched["causality"]["future_rows_used"],
            "log_future_events_used": enriched["causality"][
                "log_future_events_used"
            ],
        },
        "runtime": {
            "model_sha256": _sha256_file(model),
            "policy_sha256": _sha256_file(policy),
            "model_or_policy_recalibrated": False,
            "gemini_required": False,
        },
        "output": {
            "path": output.name,
            "sha256": _sha256_file(output),
            "data_quality_path": quality_output.name,
            "data_quality_sha256": _sha256_file(quality_output),
            "audit_path": audit_output.name,
            "audit_sha256": _sha256_file(audit_output),
            "replay_report_path": replay_report_output.name,
            "replay_report_sha256": _sha256_file(replay_report_output),
            "incident_alerts": replay["alerts"]["incident_alerts"],
            "data_quality_alerts": replay["alerts"]["data_quality_alerts"],
            "confidence_is_calibrated_probability": False,
            "contract_validation": submission_validation,
            "package_validation": package_validation,
            "run_id": submission_record["run_id"],
        },
        "limitations": {
            "official_ground_truth_used": False,
            "automatic_remediation": False,
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_json(manifest, manifest_path)
    return {**manifest, "manifest_path": str(manifest_path), "output_path": str(output)}
