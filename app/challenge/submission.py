from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

from app.challenge.contracts import TARGET_TENANT
from app.challenge.ingestion import IngestionError, RUN_ID_PATTERN, assert_allowed_path


DELIVERY_CONTRACT_VERSION = "challenge-kaiku-delivery-contract-v2"
SUBMISSION_COLUMNS = (
    "tenant",
    "timestamp",
    "confidence",
    "suspected_incident_type",
)
SUBMISSION_IDENTITY_COLUMNS = ("tenant", "timestamp")
AGGREGATE_SUBMISSION_COLUMNS = ("run_id", *SUBMISSION_COLUMNS)
AGGREGATE_SUBMISSION_IDENTITY_COLUMNS = ("run_id", *SUBMISSION_IDENTITY_COLUMNS)
SUBMISSION_PACKAGE_MANIFEST = "submissions-manifest.json"
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
INCIDENT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


def _fail(message: str) -> None:
    raise IngestionError(f"invalid challenge submission: {message}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_common(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...],
    identity_columns: tuple[str, ...],
    aggregate: bool,
) -> dict[str, Any]:
    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != expected_columns:
        _fail(
            "columns must be exactly "
            f"{list(expected_columns)}, received {list(actual_columns)}"
        )

    if frame.empty:
        return {
            "contract_version": DELIVERY_CONTRACT_VERSION,
            "scope": "aggregate_audit" if aggregate else "single_run_evaluator",
            "passed": True,
            "rows": 0,
            "runs": 0 if aggregate else 1,
            "tenant": TARGET_TENANT,
            "duplicates": 0,
            "sorted": True,
            "confidence": {"min": None, "max": None},
        }

    required = frame.loc[:, identity_columns]
    if required.isna().any().any():
        _fail(f"{', '.join(identity_columns)} cannot be null")

    run_ids: pd.Series | None = None
    if aggregate:
        run_ids = frame["run_id"].astype(str)
        invalid_runs = sorted(
            value for value in run_ids.unique() if not RUN_ID_PATTERN.fullmatch(value)
        )
        if invalid_runs:
            _fail(f"invalid run_id values: {invalid_runs[:5]}")

    tenants = sorted(frame["tenant"].astype(str).unique())
    if tenants != [TARGET_TENANT]:
        _fail(f"tenant must be {TARGET_TENANT!r}, received {tenants}")

    timestamp_values = frame["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(timestamp_values):
        timestamps = pd.to_datetime(timestamp_values, errors="coerce")
    else:
        timestamp_text = timestamp_values.astype(str)
        invalid_format = ~timestamp_text.map(
            lambda value: bool(TIMESTAMP_PATTERN.fullmatch(value))
        )
        if invalid_format.any():
            _fail("timestamps must use YYYY-MM-DDTHH:MM:SS without a timezone suffix")
        timestamps = pd.to_datetime(
            timestamp_text,
            format="%Y-%m-%dT%H:%M:%S",
            errors="coerce",
        )
    if timestamps.isna().any():
        _fail("timestamp contains an invalid calendar value")

    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.map(math.isfinite).all():
        _fail("confidence must contain finite numbers")
    if not confidence.between(0.0, 1.0, inclusive="both").all():
        _fail("confidence must be between zero and one")

    incident_types = frame["suspected_incident_type"]
    present_types = incident_types.loc[incident_types.notna()].astype(str)
    invalid_types = sorted(
        value
        for value in present_types.unique()
        if not INCIDENT_TYPE_PATTERN.fullmatch(value)
    )
    if invalid_types:
        _fail(f"invalid suspected_incident_type values: {invalid_types[:5]}")

    normalized_values: dict[str, pd.Series] = {
        "tenant": frame["tenant"].astype(str),
        "timestamp": timestamps,
    }
    if run_ids is not None:
        normalized_values["run_id"] = run_ids
    normalized = pd.DataFrame(normalized_values).loc[:, identity_columns]
    duplicate_count = int(normalized.duplicated(list(identity_columns)).sum())
    if duplicate_count:
        _fail(f"found {duplicate_count} duplicate alert identities")

    expected_order = normalized.sort_values(
        list(identity_columns), kind="stable", ignore_index=True
    )
    sorted_output = normalized.reset_index(drop=True).equals(expected_order)
    if not sorted_output:
        _fail(f"rows must be sorted by {', '.join(identity_columns)}")

    return {
        "contract_version": DELIVERY_CONTRACT_VERSION,
        "scope": "aggregate_audit" if aggregate else "single_run_evaluator",
        "passed": True,
        "rows": int(len(frame)),
        "runs": int(run_ids.nunique()) if run_ids is not None else 1,
        "tenant": TARGET_TENANT,
        "duplicates": duplicate_count,
        "sorted": sorted_output,
        "confidence": {
            "min": float(confidence.min()),
            "max": float(confidence.max()),
        },
    }


def validate_submission_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate one evaluator-facing Kaiku run without internal run metadata."""
    return _validate_common(
        frame,
        expected_columns=SUBMISSION_COLUMNS,
        identity_columns=SUBMISSION_IDENTITY_COLUMNS,
        aggregate=False,
    )


def validate_aggregate_submission_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the multi-run audit export; this is not the evaluator CSV."""
    return _validate_common(
        frame,
        expected_columns=AGGREGATE_SUBMISSION_COLUMNS,
        identity_columns=AGGREGATE_SUBMISSION_IDENTITY_COLUMNS,
        aggregate=True,
    )


def _read_submission(path: Path) -> pd.DataFrame:
    source = assert_allowed_path(path)
    if source.suffix.casefold() != ".csv":
        _fail("submission must be a CSV file")
    try:
        return pd.read_csv(
            source,
            dtype={
                "run_id": "string",
                "tenant": "string",
                "timestamp": "string",
                "suspected_incident_type": "string",
            },
            keep_default_na=True,
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise IngestionError(f"unable to read challenge submission: {exc}") from exc


def validate_submission_file(path: Path) -> dict[str, Any]:
    source = assert_allowed_path(path)
    report = validate_submission_frame(_read_submission(source))
    return {**report, "file": source.name}


def validate_aggregate_submission_file(path: Path) -> dict[str, Any]:
    source = assert_allowed_path(path)
    report = validate_aggregate_submission_frame(_read_submission(source))
    return {**report, "file": source.name}


def validate_submission_package(
    directory: Path,
    *,
    expected_runs: int | None = None,
) -> dict[str, Any]:
    root = assert_allowed_path(directory)
    manifest_path = assert_allowed_path(root / SUBMISSION_PACKAGE_MANIFEST)
    if not manifest_path.is_file():
        _fail(f"missing {SUBMISSION_PACKAGE_MANIFEST}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestionError(f"unable to read submission package: {exc}") from exc
    if not isinstance(manifest, dict):
        _fail("submission package manifest must be an object")
    if manifest.get("contract_version") != DELIVERY_CONTRACT_VERSION:
        _fail("submission package contract version mismatch")
    records = manifest.get("submissions")
    if not isinstance(records, list):
        _fail("submission package entries must be a list")

    seen_runs: set[str] = set()
    total_rows = 0
    for record in records:
        if not isinstance(record, dict):
            _fail("submission package entry must be an object")
        run_id = str(record.get("run_id", ""))
        if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in seen_runs:
            _fail(f"invalid or duplicate package run_id: {run_id!r}")
        expected_relative = f"submissions/{run_id}/alerts.csv"
        if record.get("path") != expected_relative:
            _fail(f"invalid package path for {run_id}")
        artifact = assert_allowed_path(root / expected_relative)
        if root != artifact and root not in artifact.parents:
            _fail("submission package points outside its directory")
        if not artifact.is_file():
            _fail(f"missing evaluator submission for {run_id}")
        if record.get("sha256") != _sha256_file(artifact):
            _fail(f"submission digest mismatch for {run_id}")
        validation = validate_submission_file(artifact)
        if int(record.get("rows", -1)) != validation["rows"]:
            _fail(f"submission row count mismatch for {run_id}")
        seen_runs.add(run_id)
        total_rows += validation["rows"]

    if expected_runs is not None and len(seen_runs) != expected_runs:
        _fail(f"expected {expected_runs} packaged runs, found {len(seen_runs)}")
    if int(manifest.get("run_count", -1)) != len(seen_runs):
        _fail("submission package run count mismatch")
    if int(manifest.get("total_rows", -1)) != total_rows:
        _fail("submission package total row count mismatch")
    return {
        "contract_version": DELIVERY_CONTRACT_VERSION,
        "scope": "partitioned_evaluator_package",
        "passed": True,
        "runs": len(seen_runs),
        "rows": total_rows,
        "tenant": TARGET_TENANT,
        "manifest": SUBMISSION_PACKAGE_MANIFEST,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one evaluator-facing Kaiku alert CSV"
    )
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_submission_file(args.input),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
