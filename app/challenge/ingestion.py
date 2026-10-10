from __future__ import annotations

import argparse
from collections.abc import Iterable
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.contracts import (
    CHALLENGE_SCHEMA_VERSION,
    CLEAN_COLUMNS,
    CLEANING_EVENT_COLUMNS,
    METRIC_COLUMNS,
    METRIC_LIMITS,
    PUBLIC_RUN_FILENAMES,
    QUARANTINE_COLUMNS,
    REQUIRED_COLUMNS,
    RESERVED_LABEL_COLUMNS,
    TARGET_TENANT,
    TENANTS,
)


FORBIDDEN_PATH_PARTS = frozenset({"_solucion"})
FORBIDDEN_FILENAMES = frozenset({"generar.py", "etiquetas.csv", "todos_metricas.csv"})
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class IngestionError(ValueError):
    """Raised when challenge input cannot be consumed without ambiguity."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_allowed_path(path: Path) -> Path:
    """Reject evaluator answers and generator internals before opening a path."""
    resolved = path.expanduser().resolve(strict=False)
    lowered_parts = {part.casefold() for part in resolved.parts}
    if lowered_parts & FORBIDDEN_PATH_PARTS:
        raise IngestionError("paths containing _solucion are forbidden")
    filename = resolved.name.casefold()
    if filename in FORBIDDEN_FILENAMES:
        raise IngestionError(f"reserved challenge file is forbidden: {filename}")
    return resolved


def _run_id_for_metrics(path: Path) -> str:
    if path.parent.name.casefold() != TARGET_TENANT:
        raise IngestionError(
            f"metricas.csv must belong to a {TARGET_TENANT}/ directory"
        )
    candidate = path.parent.parent.name
    run_id = TARGET_TENANT if candidate.casefold() == "data" else candidate
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise IngestionError(f"invalid run id: {run_id!r}")
    return run_id


def _validate_public_bundle(metrics_path: Path) -> dict[str, Path]:
    metrics = assert_allowed_path(metrics_path)
    if metrics.name.casefold() != "metricas.csv":
        raise IngestionError("Kaiku input must be named metricas.csv")
    _run_id_for_metrics(metrics)
    files = {
        filename: assert_allowed_path(metrics.parent / filename)
        for filename in PUBLIC_RUN_FILENAMES
    }
    missing = sorted(name for name, path in files.items() if not path.is_file())
    if missing:
        raise IngestionError(
            f"Kaiku public bundle is incomplete; missing files: {missing}"
        )
    context_header = files["contexto.md"].read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()[:1]
    logs_header = files["logs.md"].read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()[:1]
    if not context_header or context_header[0].strip().casefold() != "# contexto — kaiku":
        raise IngestionError("contexto.md is not the public Kaiku context")
    if not logs_header or logs_header[0].strip().casefold() != "# logs — kaiku":
        raise IngestionError("logs.md is not the public Kaiku log")
    return files


def discover_run_files(input_path: Path) -> list[Path]:
    """Discover only public ``kaiku/metricas.csv`` files and complete bundles."""
    root = assert_allowed_path(input_path)
    if root.is_file():
        _validate_public_bundle(root)
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise IngestionError("input must be a dataset directory or Kaiku metricas.csv")

    direct_metrics = root / "metricas.csv"
    if direct_metrics.is_file():
        _validate_public_bundle(direct_metrics)
        return [direct_metrics]

    candidates = (root / "data" / "corridas", root / "corridas", root)
    runs_root = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir()
            and any(
                child.is_dir()
                and (child / TARGET_TENANT / "metricas.csv").is_file()
                for child in candidate.iterdir()
                if child.name.casefold() not in FORBIDDEN_PATH_PARTS
            )
        ),
        None,
    )
    if runs_root is None:
        nested_metrics = root / TARGET_TENANT / "metricas.csv"
        if nested_metrics.is_file():
            _validate_public_bundle(nested_metrics)
            return [nested_metrics]
        raise IngestionError("no public Kaiku run files were found")

    files: list[Path] = []
    for run_directory in sorted(runs_root.iterdir(), key=lambda item: item.name):
        if not run_directory.is_dir():
            continue
        if run_directory.name.casefold() in FORBIDDEN_PATH_PARTS:
            continue
        run_file = run_directory / TARGET_TENANT / "metricas.csv"
        if run_file.is_file():
            _validate_public_bundle(run_file)
            files.append(assert_allowed_path(run_file))
    if not files:
        raise IngestionError("no public run files were found")
    return files


def _as_json_record(row: pd.Series, columns: Iterable[str]) -> str:
    payload = {
        column: None if pd.isna(row[column]) else str(row[column])
        for column in columns
        if column in row
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event(
    *,
    run_id: str,
    source_name: str,
    source_line: int,
    field: str,
    action: str,
    reason: str,
    original_value: object,
    clean_value: object,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "source_name": source_name,
        "source_line": source_line,
        "field": field,
        "action": action,
        "reason": reason,
        "original_value": None if pd.isna(original_value) else str(original_value),
        "clean_value": None if pd.isna(clean_value) else str(clean_value),
    }


def _clean_run(
    path: Path,
    *,
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_id = _run_id_for_metrics(path)

    raw = pd.read_csv(path, dtype="string", keep_default_na=True)
    reserved_columns = sorted(set(raw.columns) & RESERVED_LABEL_COLUMNS)
    if reserved_columns:
        raise IngestionError(
            f"reserved evaluator label columns are forbidden: {reserved_columns}"
        )
    if "tenant" in raw.columns:
        raise IngestionError(
            "Kaiku metricas.csv must not contain a tenant column; combined datasets are forbidden"
        )
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise IngestionError(f"{source_name} is missing columns: {missing_columns}")
    extra_columns = sorted(set(raw.columns) - set(REQUIRED_COLUMNS))
    raw = raw.loc[:, REQUIRED_COLUMNS].copy()
    raw["source_line"] = np.arange(2, len(raw) + 2, dtype=np.int64)

    clean = pd.DataFrame(index=raw.index)
    clean["run_id"] = run_id
    clean["source_name"] = source_name
    clean["source_line"] = raw["source_line"]
    events: list[dict[str, object]] = []

    clean["tenant"] = TARGET_TENANT

    timestamp_raw = raw["timestamp"].astype("string").str.strip()
    timestamp = pd.to_datetime(timestamp_raw, errors="coerce", format="ISO8601")
    clean["timestamp"] = timestamp
    invalid_timestamp = timestamp.isna()

    invalid_metric_masks: dict[str, pd.Series] = {}
    invalid_reason_counts: dict[str, dict[str, int]] = {}
    missing_metric_masks: dict[str, pd.Series] = {}
    for metric in METRIC_COLUMNS:
        metric_raw = raw[metric].astype("string")
        stripped = metric_raw.str.strip()
        missing = metric_raw.isna() | stripped.eq("")
        parsed = pd.to_numeric(stripped.mask(missing), errors="coerce")
        not_numeric = ~missing & parsed.isna()
        minimum, maximum = METRIC_LIMITS[metric]
        below_minimum = parsed.notna() & (parsed < minimum)
        above_maximum = pd.Series(False, index=raw.index)
        if maximum is not None:
            above_maximum = parsed.notna() & (parsed > maximum)
        fractional_count = pd.Series(False, index=raw.index)
        if metric == "errores":
            fractional_count = parsed.notna() & ~np.isclose(parsed % 1, 0)

        invalid = not_numeric | below_minimum | above_maximum | fractional_count
        reason = pd.Series("", index=raw.index, dtype="string")
        reason.loc[not_numeric] = "not_numeric"
        reason.loc[below_minimum] = "below_minimum"
        reason.loc[above_maximum] = "above_maximum"
        reason.loc[
            fractional_count & ~below_minimum & ~above_maximum
        ] = "fractional_count"

        cleaned_metric = parsed.mask(invalid).astype(float)
        clean[metric] = cleaned_metric
        clean[f"missing_{metric}"] = missing.astype(bool)
        clean[f"invalid_{metric}"] = invalid.astype(bool)
        missing_metric_masks[metric] = missing
        invalid_metric_masks[metric] = invalid
        invalid_reason_counts[metric] = {
            key: int(value)
            for key, value in reason.loc[invalid].value_counts().sort_index().items()
        }
        for index in raw.index[invalid]:
            events.append(
                _event(
                    run_id=run_id,
                    source_name=source_name,
                    source_line=int(raw.at[index, "source_line"]),
                    field=metric,
                    action="invalidated_to_null",
                    reason=str(reason.at[index]),
                    original_value=metric_raw.at[index],
                    clean_value=None,
                )
            )

    valid_identity = ~invalid_timestamp
    duplicate_identity = pd.Series(False, index=raw.index)
    duplicate_identity.loc[valid_identity] = clean.loc[valid_identity].duplicated(
        ["tenant", "timestamp"], keep=False
    )
    fatal = invalid_timestamp | duplicate_identity

    quarantine_rows: list[dict[str, object]] = []
    for index in raw.index[fatal]:
        reasons: list[str] = []
        if invalid_timestamp.at[index]:
            reasons.append("invalid_timestamp")
        if duplicate_identity.at[index]:
            reasons.append("duplicate_tenant_timestamp")
        quarantine_rows.append(
            {
                "run_id": run_id,
                "source_name": source_name,
                "source_line": int(raw.at[index, "source_line"]),
                "reason": ";".join(reasons),
                "tenant": TARGET_TENANT,
                "timestamp": (
                    None if pd.isna(raw.at[index, "timestamp"]) else str(raw.at[index, "timestamp"])
                ),
                "raw_record": _as_json_record(raw.loc[index], REQUIRED_COLUMNS),
            }
        )

    clean = clean.loc[~fatal].copy()
    any_missing = clean[[f"missing_{name}" for name in METRIC_COLUMNS]].any(axis=1)
    any_invalid = clean[[f"invalid_{name}" for name in METRIC_COLUMNS]].any(axis=1)
    all_unavailable = clean.loc[:, METRIC_COLUMNS].isna().all(axis=1)
    clean["data_quality"] = np.select(
        [all_unavailable, any_missing | any_invalid],
        ["outage", "partial"],
        default="ok",
    )
    clean = clean.loc[:, CLEAN_COLUMNS]

    valid_timestamps = pd.to_datetime(clean["timestamp"], errors="raise").sort_values(
        kind="stable"
    )
    timestamp_deltas = valid_timestamps.diff().dropna().dt.total_seconds().div(60)
    positive_gaps = timestamp_deltas.loc[timestamp_deltas.gt(1)]
    unavailable = clean["data_quality"].eq("outage")
    outage_groups = unavailable.ne(unavailable.shift(fill_value=False)).cumsum()
    outage_streaks = unavailable.groupby(outage_groups).sum()

    quarantine = pd.DataFrame(quarantine_rows, columns=QUARANTINE_COLUMNS)
    event_frame = pd.DataFrame(events, columns=CLEANING_EVENT_COLUMNS)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "source_name": source_name,
        "source_sha256": _sha256_file(path),
        "rows_received": int(len(raw)),
        "rows_clean": int(len(clean)),
        "rows_quarantined": int(fatal.sum()),
        "duplicate_rows": int(duplicate_identity.sum()),
        "invalid_tenant_rows": 0,
        "invalid_timestamp_rows": int(invalid_timestamp.sum()),
        "extra_columns_ignored": extra_columns,
        "missing_by_metric": {
            metric: int(missing_metric_masks[metric].loc[~fatal].sum())
            for metric in METRIC_COLUMNS
        },
        "invalidated_by_metric": {
            metric: int(invalid_metric_masks[metric].loc[~fatal].sum())
            for metric in METRIC_COLUMNS
        },
        "invalidated_by_reason": invalid_reason_counts,
        "temporal_integrity": {
            "expected_cadence_minutes": 1,
            "non_monotonic_intervals": int(timestamp_deltas.le(0).sum()),
            "gap_count": int(len(positive_gaps)),
            "missing_minutes_between_rows": int(
                np.rint(positive_gaps.sub(1).clip(lower=0)).sum()
            ),
            "maximum_gap_minutes": (
                float(positive_gaps.max()) if not positive_gaps.empty else 0.0
            ),
            "maximum_outage_streak_rows": (
                int(outage_streaks.max()) if not outage_streaks.empty else 0
            ),
        },
        "data_quality": {
            key: int(value)
            for key, value in clean["data_quality"].value_counts().sort_index().items()
        },
        "tenants": {
            tenant: int((clean["tenant"] == tenant).sum()) for tenant in TENANTS
        },
    }
    return clean, quarantine, event_frame, summary


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_write_bytes(path, content)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write_bytes(path, content)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except ImportError as exc:
        temporary.unlink(missing_ok=True)
        raise IngestionError("pyarrow is required; install the project with the lab extra") from exc
    os.replace(temporary, path)


def _aggregate_report(
    clean: pd.DataFrame,
    quarantine: pd.DataFrame,
    events: pd.DataFrame,
    run_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    per_run_tenant = [
        {
            "run_id": str(run_id),
            "tenant": str(tenant),
            "rows_clean": int(len(group)),
            "outages": int(group["data_quality"].eq("outage").sum()),
            "partial_rows": int(group["data_quality"].eq("partial").sum()),
            "first_timestamp": group["timestamp"].min().isoformat(),
            "last_timestamp": group["timestamp"].max().isoformat(),
            "missing_by_metric": {
                metric: int(group[f"missing_{metric}"].sum()) for metric in METRIC_COLUMNS
            },
            "invalidated_by_metric": {
                metric: int(group[f"invalid_{metric}"].sum()) for metric in METRIC_COLUMNS
            },
        }
        for (run_id, tenant), group in clean.groupby(["run_id", "tenant"], sort=True)
    ]
    return {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "run_count": int(clean["run_id"].nunique()),
        "tenant_count": int(clean["tenant"].nunique()),
        "tenants": sorted(str(value) for value in clean["tenant"].unique()),
        "rows_received": int(sum(item["rows_received"] for item in run_summaries)),
        "rows_clean": int(len(clean)),
        "rows_quarantined": int(len(quarantine)),
        "cleaning_event_count": int(len(events)),
        "data_quality": {
            key: int(value)
            for key, value in clean["data_quality"].value_counts().sort_index().items()
        },
        "missing_by_metric": {
            metric: int(clean[f"missing_{metric}"].sum()) for metric in METRIC_COLUMNS
        },
        "invalidated_by_metric": {
            metric: int(clean[f"invalid_{metric}"].sum()) for metric in METRIC_COLUMNS
        },
        "invalidated_by_reason": {
            str(reason): int(count)
            for reason, count in events["reason"].value_counts().sort_index().items()
        },
        "temporal_integrity": {
            "expected_cadence_minutes": 1,
            "runs_with_gaps": int(
                sum(item["temporal_integrity"]["gap_count"] > 0 for item in run_summaries)
            ),
            "gap_count": int(
                sum(item["temporal_integrity"]["gap_count"] for item in run_summaries)
            ),
            "missing_minutes_between_rows": int(
                sum(
                    item["temporal_integrity"]["missing_minutes_between_rows"]
                    for item in run_summaries
                )
            ),
            "maximum_gap_minutes": float(
                max(
                    (
                        item["temporal_integrity"]["maximum_gap_minutes"]
                        for item in run_summaries
                    ),
                    default=0.0,
                )
            ),
            "maximum_outage_streak_rows": int(
                max(
                    (
                        item["temporal_integrity"]["maximum_outage_streak_rows"]
                        for item in run_summaries
                    ),
                    default=0,
                )
            ),
        },
        "quarantine_reasons": (
            {
                key: int(value)
                for key, value in quarantine["reason"].value_counts().sort_index().items()
            }
            if not quarantine.empty
            else {}
        ),
        "per_run": run_summaries,
        "per_run_tenant": per_run_tenant,
    }


def ingest_dataset(
    input_path: Path,
    output_directory: Path,
    *,
    expected_runs: int | None = None,
    require_all_tenants: bool = False,
) -> dict[str, Any]:
    """Clean public challenge runs and write deterministic, auditable outputs."""
    files = discover_run_files(input_path)
    if expected_runs is not None and len(files) != expected_runs:
        raise IngestionError(f"expected {expected_runs} runs, found {len(files)}")

    source_root = Path(os.path.commonpath([str(path.parent.parent) for path in files]))
    clean_frames: list[pd.DataFrame] = []
    quarantine_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        clean, quarantine, events, summary = _clean_run(path, source_name=relative)
        bundle = _validate_public_bundle(path)
        summary["public_bundle"] = {
            name: {
                "source_name": bundle_path.relative_to(source_root).as_posix(),
                "sha256": _sha256_file(bundle_path),
                "bytes": bundle_path.stat().st_size,
            }
            for name, bundle_path in sorted(bundle.items())
        }
        clean_frames.append(clean)
        quarantine_frames.append(quarantine)
        event_frames.append(events)
        run_summaries.append(summary)

    clean_frame = pd.concat(clean_frames, ignore_index=True)
    clean_frame = clean_frame.sort_values(
        ["run_id", "tenant", "timestamp", "source_line"], kind="stable"
    ).reset_index(drop=True)
    quarantine_frame = pd.concat(quarantine_frames, ignore_index=True)
    if not quarantine_frame.empty:
        quarantine_frame = quarantine_frame.sort_values(
            ["run_id", "source_line"], kind="stable"
        ).reset_index(drop=True)
    event_frame = pd.concat(event_frames, ignore_index=True)
    if not event_frame.empty:
        event_frame = event_frame.sort_values(
            ["run_id", "source_line", "field"], kind="stable"
        ).reset_index(drop=True)

    rows_received = int(sum(item["rows_received"] for item in run_summaries))
    if len(clean_frame) + len(quarantine_frame) != rows_received:
        raise IngestionError("row conservation invariant failed")
    if clean_frame.duplicated(["run_id", "tenant", "timestamp"]).any():
        raise IngestionError("clean output contains duplicate run/tenant/timestamp identities")
    invalid_scope = clean_frame.loc[~clean_frame["tenant"].eq(TARGET_TENANT)]
    if not invalid_scope.empty:
        raise IngestionError("clean output escaped the Kaiku-only ingestion scope")

    report = _aggregate_report(clean_frame, quarantine_frame, event_frame, run_summaries)
    report["integrity"] = {
        "target_tenant": TARGET_TENANT,
        "kaiku_only": True,
        "identity_unique": True,
        "row_conservation": True,
        "one_minute_cadence": report["temporal_integrity"]["gap_count"] == 0,
    }
    output = output_directory.expanduser().resolve(strict=False)
    metrics_path = output / "metrics.clean.parquet"
    quarantine_path = output / "quarantine.csv"
    events_path = output / "cleaning-events.csv"
    report_path = output / "quality-report.json"
    manifest_path = output / "manifest.json"

    _atomic_write_parquet(clean_frame, metrics_path)
    _atomic_write_csv(quarantine_frame, quarantine_path)
    _atomic_write_csv(event_frame, events_path)
    _atomic_write_json(report, report_path)
    manifest = {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "scope": {
            "target_tenant": TARGET_TENANT,
            "allowed_public_filenames": list(PUBLIC_RUN_FILENAMES),
            "reserved_path_parts": sorted(FORBIDDEN_PATH_PARTS),
            "reserved_filenames": sorted(FORBIDDEN_FILENAMES),
            "reserved_label_columns": sorted(RESERVED_LABEL_COLUMNS),
        },
        "input": {
            "run_count": len(files),
            "runs": [
                {
                    "run_id": item["run_id"],
                    "files": item["public_bundle"],
                }
                for item in run_summaries
            ],
            "files": [
                {
                    "run_id": item["run_id"],
                    "role": role,
                    **metadata,
                }
                for item in run_summaries
                for role, metadata in sorted(item["public_bundle"].items())
            ],
        },
        "outputs": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (metrics_path, quarantine_path, events_path, report_path)
        },
    }
    _atomic_write_json(manifest, manifest_path)
    return {**report, "output_directory": str(output), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and audit the public Zikit challenge metric runs"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument("--require-all-tenants", action="store_true")
    args = parser.parse_args()
    if args.expected_runs is not None and args.expected_runs < 1:
        parser.error("--expected-runs must be positive")
    report = ingest_dataset(
        args.input,
        args.output,
        expected_runs=args.expected_runs,
        require_all_tenants=args.require_all_tenants,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "run_count": report["run_count"],
                "tenant_count": report["tenant_count"],
                "rows_received": report["rows_received"],
                "rows_clean": report["rows_clean"],
                "rows_quarantined": report["rows_quarantined"],
                "data_quality": report["data_quality"],
                "output_directory": report["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
