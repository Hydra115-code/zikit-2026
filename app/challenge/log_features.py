from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from app.challenge.contracts import TARGET_TENANT
from app.challenge.features import FEATURE_SCHEMA_VERSION
from app.challenge.ingestion import (
    IngestionError,
    assert_allowed_path,
    discover_run_files,
)
from app.challenge.training import discover_feature_files


LOG_FEATURE_SCHEMA_VERSION = "challenge-kaiku-log-features-v1"
LOG_WINDOWS_MINUTES = (15, 60, 180, 720, 1440)
LOG_LINE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+"
    r"(DEBUG|INFO|WARN|ERROR|CRITICAL)\s+(.+)$"
)
ALLOWED_LOG_SOURCES = frozenset({"servidor", "firewall", "app"})

ACTIONABLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("checkout_503", re.compile(r"checkout:\s*503\b")),
    ("checkout_timeout", re.compile(r"timeout en checkout")),
    ("db_pool", re.compile(r"pool de conexiones")),
    ("deadlock", re.compile(r"deadlock")),
    ("db_lock_wait", re.compile(r"mysqldump: espera de bloqueo")),
    ("cpu_sustained", re.compile(r"cpu de srv-bd-\d+.*sostenido")),
)
BACKGROUND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("campaign_traffic", re.compile(r"pico de conexiones entrantes.*campana")),
    ("firewall_scan", re.compile(r"escaneo de puertos")),
    ("sync_contactos", re.compile(r"sync_contactos")),
    ("webhook_retry", re.compile(r"webhook de notificacion")),
    ("printer_queue", re.compile(r"servicio de impresion")),
    ("email_retry", re.compile(r"reintento de envio de correo")),
)
ACTIONABLE_CATEGORIES = tuple(name for name, _ in ACTIONABLE_PATTERNS)
BACKGROUND_CATEGORIES = tuple(name for name, _ in BACKGROUND_PATTERNS)
OTHER_CATEGORIES = ("other_error", "other_warn")
LOG_CATEGORIES = (*ACTIONABLE_CATEGORIES, *BACKGROUND_CATEGORIES, *OTHER_CATEGORIES)

EVENT_COLUMNS = (
    "run_id",
    "tenant",
    "event_timestamp",
    "available_timestamp",
    "source",
    "level",
    "category",
    "source_line",
    "message_sha256",
)


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


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _classify(level: str, message: str) -> str | None:
    lowered = message.casefold()
    for name, pattern in ACTIONABLE_PATTERNS:
        if pattern.search(lowered):
            return name
    for name, pattern in BACKGROUND_PATTERNS:
        if pattern.search(lowered):
            return name
    if level in {"ERROR", "CRITICAL"}:
        return "other_error"
    if level == "WARN":
        return "other_warn"
    return None


def parse_kaiku_log(path: Path, run_id: str) -> pd.DataFrame:
    resolved = assert_allowed_path(path)
    if resolved.name != "logs.md" or resolved.parent.name.casefold() != TARGET_TENANT:
        raise IngestionError("log input must be a public kaiku/logs.md file")
    source: str | None = None
    events: list[dict[str, object]] = []
    malformed: list[int] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        if line.startswith("## "):
            source = line[3:].strip().casefold()
            continue
        match = LOG_LINE_PATTERN.fullmatch(line)
        if match is None:
            if re.match(r"^\d{4}-\d{2}-\d{2}", line):
                malformed.append(line_number)
            continue
        if source not in ALLOWED_LOG_SOURCES:
            raise IngestionError(
                f"log event outside a supported source section at line {line_number}"
            )
        event_timestamp = pd.Timestamp(match.group(1))
        level = match.group(2)
        message = match.group(3)
        category = _classify(level, message)
        events.append(
            {
                "run_id": run_id,
                "tenant": TARGET_TENANT,
                "event_timestamp": event_timestamp,
                "available_timestamp": event_timestamp.ceil("min"),
                "source": source,
                "level": level,
                "category": category,
                "source_line": line_number,
                "message_sha256": sha256(message.encode("utf-8")).hexdigest(),
            }
        )
    if malformed:
        raise IngestionError(f"malformed timestamped log lines: {malformed[:10]}")
    frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if frame.duplicated(
        ["run_id", "source", "event_timestamp", "level", "message_sha256"]
    ).any():
        raise IngestionError(f"duplicate log events in {run_id}")
    return frame.sort_values(
        ["event_timestamp", "source", "source_line"], kind="stable", ignore_index=True
    )


def _rolling_counts(series: pd.Series, window: int) -> np.ndarray:
    return (
        series.rolling(f"{window}min", min_periods=1, closed="right")
        .sum()
        .to_numpy(dtype="float32")
    )


def build_log_features_for_run(
    base_features: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if base_features.empty:
        raise IngestionError("cannot enrich an empty feature run")
    ordered = base_features.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if set(ordered["tenant"].astype(str).unique()) != {TARGET_TENANT}:
        raise IngestionError("log enrichment is scoped exclusively to Kaiku")
    run_ids = set(ordered["run_id"].astype(str).unique())
    event_run_ids = set(events["run_id"].astype(str).unique())
    if len(run_ids) != 1 or (event_run_ids and event_run_ids != run_ids):
        raise IngestionError("log events and metric features belong to different runs")

    timestamps = pd.DatetimeIndex(pd.to_datetime(ordered["timestamp"], errors="raise"))
    start = timestamps.min()
    end = timestamps.max()
    event_time = pd.to_datetime(events["event_timestamp"], errors="raise")
    available_time = pd.to_datetime(events["available_timestamp"], errors="raise")
    if (available_time < event_time).any():
        raise IngestionError("log availability precedes its source event")

    in_window = available_time.between(start, end, inclusive="both")
    usable = events.loc[in_window].copy()
    usable["available_timestamp"] = available_time.loc[in_window].to_numpy()
    full_index = pd.date_range(start, end, freq="min")
    values: dict[str, np.ndarray] = {}

    category_minute = (
        usable.dropna(subset=["category"])
        .groupby(["available_timestamp", "category"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(full_index, fill_value=0)
    )
    for category in LOG_CATEGORIES:
        counts = category_minute.get(
            category, pd.Series(0, index=full_index, dtype="int64")
        ).astype(float)
        values[f"log_{category}_count_1m"] = counts.reindex(timestamps).to_numpy(
            dtype="float32"
        )
        for window in LOG_WINDOWS_MINUTES:
            rolled = pd.Series(_rolling_counts(counts, window), index=full_index)
            values[f"log_{category}_count_{window}m"] = rolled.reindex(
                timestamps
            ).to_numpy(dtype="float32")
        last_seen = pd.Series(full_index, index=full_index).where(counts.gt(0)).ffill()
        minutes_since = pd.Series(full_index, index=full_index).sub(last_seen)
        values[f"log_{category}_minutes_since"] = (
            minutes_since.dt.total_seconds().div(60).reindex(timestamps).to_numpy(
                dtype="float32"
            )
        )

    classified = usable["category"].notna()
    minute_groups = {
        "events": usable,
        "actionable": usable.loc[usable["category"].isin(ACTIONABLE_CATEGORIES)],
        "background": usable.loc[
            usable["category"].isin((*BACKGROUND_CATEGORIES, *OTHER_CATEGORIES))
        ],
        "classified": usable.loc[classified],
    }
    for name, group in minute_groups.items():
        counts = (
            group.groupby("available_timestamp", sort=True)
            .size()
            .reindex(full_index, fill_value=0)
            .astype(float)
        )
        values[f"log_{name}_count_1m"] = counts.reindex(timestamps).to_numpy(
            dtype="float32"
        )
        for window in LOG_WINDOWS_MINUTES:
            rolled = pd.Series(_rolling_counts(counts, window), index=full_index)
            values[f"log_{name}_count_{window}m"] = rolled.reindex(
                timestamps
            ).to_numpy(dtype="float32")

    enriched = ordered.copy()
    enriched.insert(1, "log_feature_version", LOG_FEATURE_SCHEMA_VERSION)
    enriched = pd.concat([enriched, pd.DataFrame(values)], axis=1)
    delays = (
        available_time.sub(event_time).dt.total_seconds().astype(float)
    )
    category_counts = {
        str(key): int(value)
        for key, value in events["category"].fillna("unclassified").value_counts().sort_index().items()
    }
    report = {
        "run_id": next(iter(run_ids)),
        "events_parsed": int(len(events)),
        "events_used": int(in_window.sum()),
        "events_before_metrics": int((available_time < start).sum()),
        "events_after_metrics": int((available_time > end).sum()),
        "classified_events": int(events["category"].notna().sum()),
        "actionable_events": int(events["category"].isin(ACTIONABLE_CATEGORIES).sum()),
        "background_events": int(
            events["category"].isin((*BACKGROUND_CATEGORIES, *OTHER_CATEGORIES)).sum()
        ),
        "category_counts": category_counts,
        "availability_delay_seconds": {
            "minimum": float(delays.min()) if not delays.empty else None,
            "maximum": float(delays.max()) if not delays.empty else None,
        },
        "future_events_used": False,
    }
    return enriched, report


def _phase_one_log_hashes(phase_one_directory: Path) -> tuple[dict[str, str], Path]:
    root = assert_allowed_path(phase_one_directory)
    manifest_path = assert_allowed_path(root / "manifest.json")
    if not manifest_path.is_file():
        raise IngestionError("phase-1 manifest is required for log provenance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scope", {}).get("target_tenant") != TARGET_TENANT:
        raise IngestionError("phase-1 manifest is not scoped to Kaiku")
    hashes: dict[str, str] = {}
    for item in manifest.get("input", {}).get("files", []):
        if item.get("role") != "logs.md":
            continue
        run_id = str(item.get("run_id", ""))
        if run_id in hashes:
            raise IngestionError(f"duplicate phase-1 log provenance for {run_id}")
        hashes[run_id] = str(item.get("sha256", ""))
    if not hashes:
        raise IngestionError("phase-1 manifest contains no Kaiku log hashes")
    return hashes, manifest_path


def enrich_log_feature_dataset(
    source_path: Path,
    base_feature_directory: Path,
    phase_one_directory: Path,
    output_directory: Path,
    *,
    expected_runs: int | None = None,
) -> dict[str, Any]:
    files, base_report, base_manifest = discover_feature_files(
        base_feature_directory, expected_runs=expected_runs
    )
    if base_report.get("logs", {}).get("included") is not False:
        raise IngestionError("base feature set already includes logs")
    expected_hashes, phase_one_manifest_path = _phase_one_log_hashes(
        phase_one_directory
    )

    metric_paths = discover_run_files(source_path)
    log_paths = {
        metric.parent.parent.name if metric.parent.parent.name != "data" else TARGET_TENANT:
        assert_allowed_path(metric.parent / "logs.md")
        for metric in metric_paths
    }
    if set(log_paths) != set(files) or set(expected_hashes) != set(files):
        raise IngestionError("source logs, phase-1 provenance and feature runs differ")

    output = assert_allowed_path(output_directory)
    output_features = output / "features"
    all_events: list[pd.DataFrame] = []
    feature_records: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None
    rows_output = 0
    for run_id, base_path in sorted(files.items()):
        log_path = log_paths[run_id]
        actual_digest = _sha256_file(log_path)
        if actual_digest != expected_hashes[run_id]:
            raise IngestionError(f"Kaiku log digest differs from phase 1 for {run_id}")
        events = parse_kaiku_log(log_path, run_id)
        base = pd.read_parquet(base_path)
        enriched, run_report = build_log_features_for_run(base, events)
        current_columns = [
            column
            for column in enriched.columns
            if column
            not in {
                "feature_version",
                "log_feature_version",
                "run_id",
                "tenant",
                "timestamp",
                "source_name",
                "source_line",
                "data_quality",
            }
        ]
        if feature_columns is None:
            feature_columns = current_columns
        elif feature_columns != current_columns:
            raise IngestionError(f"log feature schema drift in {run_id}")
        feature_path = output_features / f"{run_id}.parquet"
        _atomic_parquet(enriched, feature_path)
        feature_records.append(
            {
                "run_id": run_id,
                "path": f"features/{feature_path.name}",
                "sha256": _sha256_file(feature_path),
                "rows": int(len(enriched)),
            }
        )
        log_records.append(
            {
                "run_id": run_id,
                "source_name": log_path.relative_to(
                    Path(os.path.commonpath([str(path.parent.parent) for path in metric_paths]))
                ).as_posix(),
                "sha256": actual_digest,
                "events": int(len(events)),
            }
        )
        run_reports.append(run_report)
        all_events.append(events)
        rows_output += len(enriched)

    assert feature_columns is not None
    event_frame = pd.concat(all_events, ignore_index=True).sort_values(
        ["run_id", "event_timestamp", "source", "source_line"],
        kind="stable",
        ignore_index=True,
    )
    events_path = output / "log-events.parquet"
    _atomic_parquet(event_frame, events_path)
    report = {
        **base_report,
        "log_feature_version": LOG_FEATURE_SCHEMA_VERSION,
        "rows_output": int(rows_output),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "logs": {
            "included": True,
            "parser": "strict timestamp-level-message",
            "availability_rule": "ceil event timestamp to the next observable minute",
            "events_parsed": int(len(event_frame)),
            "events_used": int(sum(item["events_used"] for item in run_reports)),
            "events_after_metrics": int(
                sum(item["events_after_metrics"] for item in run_reports)
            ),
            "actionable_categories": list(ACTIONABLE_CATEGORIES),
            "background_categories": list((*BACKGROUND_CATEGORIES, *OTHER_CATEGORIES)),
            "windows_minutes": list(LOG_WINDOWS_MINUTES),
        },
        "causality": {
            **base_report.get("causality", {}),
            "log_future_events_used": False,
            "log_subminute_events_delayed_until_observable": True,
        },
        "per_run_logs": run_reports,
    }
    report_path = output / "feature-report.json"
    _atomic_json(report, report_path)
    manifest = {
        "feature_version": FEATURE_SCHEMA_VERSION,
        "log_feature_version": LOG_FEATURE_SCHEMA_VERSION,
        "scope": {"target_tenant": TARGET_TENANT, "kaiku_only": True},
        "source": {
            "base_feature_manifest_sha256": _sha256_file(
                assert_allowed_path(base_feature_directory) / "manifest.json"
            ),
            "phase_one_manifest_sha256": _sha256_file(phase_one_manifest_path),
        },
        "logs": log_records,
        "log_events": {
            "path": events_path.name,
            "sha256": _sha256_file(events_path),
            "rows": int(len(event_frame)),
            "raw_messages_stored": False,
        },
        "feature_files": feature_records,
        "report": {"path": report_path.name, "sha256": _sha256_file(report_path)},
    }
    _atomic_json(manifest, output / "manifest.json")
    return {**report, "manifest": manifest, "output_directory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add causal, semantically filtered Kaiku log features"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--phase-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=54)
    args = parser.parse_args()
    report = enrich_log_feature_dataset(
        args.source,
        args.features,
        args.phase_one,
        args.output,
        expected_runs=args.expected_runs,
    )
    print(
        json.dumps(
            {
                "log_feature_version": report["log_feature_version"],
                "run_count": report["run_count"],
                "rows_output": report["rows_output"],
                "feature_count": report["feature_count"],
                "events_parsed": report["logs"]["events_parsed"],
                "output_directory": report["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
