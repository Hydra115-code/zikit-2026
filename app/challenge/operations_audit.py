from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from sqlalchemy import func, select

from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.persistence import (
    EVENT_CHALLENGE_ALERT_ACKNOWLEDGED,
    ChallengePredictionRepository,
    acknowledge_alert_once,
)
from app.database import (
    AlertRecord,
    Base,
    ChallengePredictionRecord,
    EventRecord,
    configure_database,
    engine,
    session_factory,
)


OPERATIONS_AUDIT_VERSION = "challenge-kaiku-durable-operations-audit-v2"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def audit_durable_operations(
    replay_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    replay = assert_allowed_path(replay_directory)
    full_path = replay / "alerts.full.parquet"
    report_path = replay / "replay-report.json"
    if not full_path.is_file() or not report_path.is_file():
        raise IngestionError("operations audit requires a complete replay directory")
    replay_report = json.loads(report_path.read_text(encoding="utf-8"))
    alerts = pd.read_parquet(full_path)
    incident = alerts.loc[alerts["alert_type"].eq("incident_risk")].head(1)
    data_quality = alerts.loc[alerts["alert_type"].eq("data_quality_alert")].head(1)
    if incident.empty or data_quality.empty:
        raise IngestionError(
            "operations audit requires one incident and one data-quality alert"
        )
    sample = pd.concat([incident, data_quality], ignore_index=True).sort_values(
        ["run_id", "tenant", "timestamp", "alert_type"],
        kind="stable",
        ignore_index=True,
    )
    lineage = replay_report.get("lineage", {})
    model_sha256 = str(lineage.get("model_sha256", ""))
    policy_sha256 = str(lineage.get("policy_sha256", ""))
    if len(model_sha256) != 64 or len(policy_sha256) != 64:
        raise IngestionError("replay report is missing artifact lineage hashes")

    original_database_url = engine().url.render_as_string(hide_password=False)
    try:
        with TemporaryDirectory(prefix="zikit-phase-7-") as temporary_name:
            database_url = f"sqlite:///{Path(temporary_name) / 'operations.db'}"
            configure_database(database_url)
            Base.metadata.create_all(engine())
            repository = ChallengePredictionRepository()
            first = repository.append(
                sample,
                model_sha256=model_sha256,
                policy_sha256=policy_sha256,
            )
            cursor_before_ack = repository.cursor()
            second = repository.append(
                sample,
                model_sha256=model_sha256,
                policy_sha256=policy_sha256,
            )

            with session_factory()() as session:
                prediction = session.scalar(
                    select(ChallengePredictionRecord).where(
                        ChallengePredictionRecord.alert_type == "incident_risk"
                    )
                )
                if prediction is None:
                    raise IngestionError("incident prediction was not persisted")
                alert = session.scalar(
                    select(AlertRecord).where(
                        AlertRecord.event_id == prediction.event_id
                    )
                )
                if alert is None:
                    raise IngestionError("prediction outbox alert was not persisted")
                alert_id = alert.id
                acknowledged_alert, first_trace, first_claimed = acknowledge_alert_once(
                    session,
                    alert_id,
                    acknowledged_at=datetime.now(UTC),
                )
                if acknowledged_alert is None:
                    raise IngestionError(
                        "prediction alert disappeared during acknowledgement"
                    )
                session.commit()

            with session_factory()() as session:
                repeated_alert, repeated_trace, repeated_claimed = acknowledge_alert_once(
                    session,
                    alert_id,
                    acknowledged_at=datetime.now(UTC),
                )
                if repeated_alert is None:
                    raise IngestionError("acknowledged alert disappeared")
                session.commit()

            configure_database(database_url)
            Base.metadata.create_all(engine())
            restarted_repository = ChallengePredictionRepository()
            replay_after_ack = restarted_repository.append(
                sample,
                model_sha256=model_sha256,
                policy_sha256=policy_sha256,
            )
            summary = restarted_repository.operations_summary()
            resumed_updates = restarted_repository.stream_updates(
                after_id=cursor_before_ack["latest_id"]
            )
            with session_factory()() as session:
                acknowledgement_event_count = int(
                    session.scalar(
                        select(func.count(EventRecord.id)).where(
                            EventRecord.event_type
                            == EVENT_CHALLENGE_ALERT_ACKNOWLEDGED
                        )
                    )
                    or 0
                )
    finally:
        configure_database(original_database_url)

    gates = {
        "transactional_delivery": {
            "passed": first["created"] == 2 and summary["alerts"]["total"] == 2,
            "created": first["created"],
            "persisted_after_restart": summary["alerts"]["total"],
        },
        "idempotent_replay": {
            "passed": second["created"] == 0 and second["existing"] == 2,
            "created_on_repeat": second["created"],
            "existing_on_repeat": second["existing"],
        },
        "acknowledgement_trace": {
            "passed": (
                first_claimed is True
                and repeated_claimed is False
                and first_trace == repeated_trace
                and summary["alerts"]["acknowledged"] == 1
                and acknowledgement_event_count == 1
            ),
            "acknowledged_alerts": summary["alerts"]["acknowledged"],
            "acknowledgement_events": acknowledgement_event_count,
            "first_request_claimed": first_claimed,
            "repeat_request_claimed": repeated_claimed,
            "repeat_returned_same_trace": first_trace == repeated_trace,
            "mitigation_executed": False,
        },
        "cursor_monotonic_after_acknowledgement": {
            "passed": (
                replay_after_ack["created"] == 0
                and replay_after_ack["existing"] == 2
                and replay_after_ack["latest_id"] == summary["cursor"]["latest_id"]
                and replay_after_ack["latest_id"] > cursor_before_ack["latest_id"]
            ),
            "cursor_before_acknowledgement": cursor_before_ack["latest_id"],
            "cursor_after_repeated_replay": replay_after_ack["latest_id"],
            "durable_latest_cursor": summary["cursor"]["latest_id"],
        },
        "resumable_cursor": {
            "passed": (
                len(resumed_updates) == 1
                and resumed_updates[0].get("stream_event")
                == "challenge_alert_acknowledged"
            ),
            "cursor_before_ack": cursor_before_ack,
            "updates_after_cursor": len(resumed_updates),
            "update_types": [
                item.get("stream_event") for item in resumed_updates
            ],
        },
        "relationship_integrity": summary["integrity"],
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    report = {
        "audit_version": OPERATIONS_AUDIT_VERSION,
        "status": "passed" if passed else "failed",
        "scope": "controlled local durability probe using two official synthetic Kaiku alerts",
        "sample": {
            "rows": len(sample),
            "run_ids": sorted(sample["run_id"].astype(str).unique()),
            "alert_types": sorted(sample["alert_type"].astype(str).unique()),
        },
        "gates": gates,
        "operations": summary,
        "limitations": {
            "remote_infrastructure_tested": False,
            "mitigation_executed": False,
            "recovery_verified": False,
            "database_probe": "temporary SQLite database",
        },
    }
    output = assert_allowed_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output / "operations-audit.json")
    return {**report, "output_directory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit durable and resumable operations for Kaiku alerts"
    )
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    result = audit_durable_operations(args.replay, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
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
