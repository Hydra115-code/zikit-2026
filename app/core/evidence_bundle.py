from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from statistics import median
from typing import Any

from sqlalchemy import select

from app.core.explanations import build_incident_explanation
from app.database import (
    ActionRecord,
    EventRecord,
    IncidentRecord,
    InvestigationRecord,
    session_factory,
)
from app.lab.complex_scenarios import COMPLEX_INCIDENT_CASES, case_fingerprint
from app.schemas import InvestigationResponse


BUNDLE_VERSION = "1"
SENSITIVE_KEYS = {
    "access_token",
    "admin_token",
    "api_key",
    "authorization",
    "bearer_token",
    "cookie",
    "credential",
    "credentials",
    "gemini_api_key",
    "lab_admin_token",
    "mcp_read_token",
    "password",
    "refresh_token",
    "secret",
    "set_cookie",
    "token",
}


def _is_sensitive_key(key: str) -> bool:
    return key.lower().replace("-", "_") in SENSITIVE_KEYS


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if not _is_sensitive_key(key)
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _event_payload(
    event: EventRecord,
    first_signal: EventRecord | None,
) -> dict[str, Any]:
    relative_seconds = None
    if first_signal is not None and event.process_id == first_signal.process_id:
        relative_seconds = round(
            event.monotonic_seconds - first_signal.monotonic_seconds,
            6,
        )
    return {
        "id": event.id,
        "type": event.event_type,
        "timestamp": event.timestamp.isoformat(),
        "relative_to_first_signal_seconds": relative_seconds,
        "details": _sanitize(event.details),
    }


def _stage(
    events: list[EventRecord],
    names: tuple[str, ...],
    first_signal: EventRecord | None,
) -> dict[str, Any] | None:
    event = next((item for item in events if item.event_type in names), None)
    if event is None:
        return None
    payload = _event_payload(event, first_signal)
    return {
        "event_id": payload["id"],
        "event_type": payload["type"],
        "timestamp": payload["timestamp"],
        "seconds_from_first_signal": payload["relative_to_first_signal_seconds"],
    }


def _latest_investigation(
    incident_id: str,
    provider: str,
) -> InvestigationResponse | None:
    with session_factory()() as session:
        record = session.scalar(
            select(InvestigationRecord)
            .where(
                InvestigationRecord.incident_id == incident_id,
                InvestigationRecord.provider == provider,
            )
            .order_by(InvestigationRecord.id.desc())
            .limit(1)
        )
    if record is None:
        return None
    return InvestigationResponse.model_validate_json(record.diagnosis)


def _case_evidence(
    case: Any,
    observation: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_id = observation["experiment_id"]
    incident_id = observation.get("incident_id")
    with session_factory()() as session:
        events = list(
            session.scalars(
                select(EventRecord)
                .where(EventRecord.experiment_id == experiment_id)
                .order_by(EventRecord.monotonic_seconds, EventRecord.id)
            )
        )
        incident = session.get(IncidentRecord, incident_id) if incident_id else None
        actions = (
            list(
                session.scalars(
                    select(ActionRecord)
                    .where(ActionRecord.incident_id == incident_id)
                    .order_by(ActionRecord.started_at, ActionRecord.id)
                )
            )
            if incident_id
            else []
        )

    first_signal = next(
        (item for item in events if item.event_type == "FIRST_SIGNAL"),
        None,
    )
    timeline = [_event_payload(item, first_signal) for item in events]
    chain = {
        "first_signal": _stage(events, ("FIRST_SIGNAL",), first_signal),
        "detection": _stage(events, ("INCIDENT_DETECTED",), first_signal),
        "safe_decision": _stage(
            events,
            ("ACTION_STARTED", "ACTION_BLOCKED"),
            first_signal,
        ),
        "action_result": _stage(
            events,
            ("ACTION_COMPLETED", "ACTION_BLOCKED"),
            first_signal,
        ),
        "verification": _stage(
            events,
            ("RECOVERY_VERIFIED", "RECOVERY_FAILED"),
            first_signal,
        ),
    }

    diagnosis = None
    explanation = None
    if incident_id and evaluation is not None:
        provider = str(evaluation["provider"])
        saved = _latest_investigation(incident_id, provider)
        if saved is not None:
            diagnosis = _sanitize(saved.model_dump(mode="json", exclude={"raw"}))
        explanation = build_incident_explanation(incident_id)

    body = {
        "case_id": case.id,
        "title": case.title,
        "classification": case.classification,
        "contract_fingerprint": case_fingerprint(case),
        "contract": {
            "early_signal": case.early_signal,
            "effective_action": case.effective_action,
            "measurement": case.measurement,
            "reproducible_evidence": case.reproducible_evidence,
            "allowed_actions": case.allowed_actions,
            "forbidden_actions": case.forbidden_actions,
            "required_events": case.required_events,
            "forbidden_events": case.forbidden_events,
            "requires_uncertainty": case.requires_uncertainty,
            "root_cause_mode": case.root_cause_mode,
        },
        "scenario": case.scenario.model_dump(mode="json"),
        "observation": {
            "experiment_id": experiment_id,
            "incident_id": incident_id,
            "experiment_status": observation["experiment_status"],
            "incident_status": observation["incident_status"],
            "action_states": observation["action_states"],
            "event_types": observation["event_types"],
            "active_provider": observation["active_provider"],
            "switch_delta": observation["switch_delta"],
            "measurement_valid": observation["measurement_valid"],
            "durations": observation["durations"],
            "passed": observation["passed"],
        },
        "causal_chain": chain,
        "timeline": timeline,
        "actions": [
            {
                "id": action.id,
                "runbook": action.runbook,
                "state": action.state,
                "started_at": action.started_at.isoformat(),
                "completed_at": (
                    action.completed_at.isoformat() if action.completed_at else None
                ),
            }
            for action in actions
        ],
        "incident": (
            {
                "id": incident.id,
                "service": incident.service,
                "severity": incident.severity,
                "status": incident.status,
                "first_signal_at": incident.first_signal_at.isoformat(),
                "detected_at": incident.detected_at.isoformat(),
                "recovered_at": (
                    incident.recovered_at.isoformat() if incident.recovered_at else None
                ),
            }
            if incident
            else None
        ),
        "deterministic_explanation": _sanitize(explanation),
        "operator_investigation": diagnosis,
        "explanation_evaluation": evaluation,
    }
    return {**body, "evidence_sha256": content_sha256(body)}


def _nearest_rank_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return round(ordered[rank - 1], 6)


def _timing_summary(cases: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    values = [
        item["causal_chain"][stage]["seconds_from_first_signal"]
        for item in cases
        if item["causal_chain"][stage] is not None
        and item["causal_chain"][stage]["seconds_from_first_signal"] is not None
        and item["causal_chain"][stage]["seconds_from_first_signal"] >= 0
    ]
    return {
        "sample_count": len(values),
        "min_seconds": round(min(values), 6) if values else None,
        "median_seconds": round(median(values), 6) if values else None,
        "p95_nearest_rank_seconds": _nearest_rank_p95(values),
        "max_seconds": round(max(values), 6) if values else None,
    }


def build_evidence_bundle(
    phase17: dict[str, Any],
    phase18: dict[str, Any],
    *,
    phase17_sha256: str,
    phase18_sha256: str,
) -> dict[str, Any]:
    if phase17.get("all_gates_passed") is not True:
        raise ValueError("phase 17 source report did not pass")
    if phase18.get("all_gates_passed") is not True:
        raise ValueError("phase 18 source report did not pass")

    observations = {item["case_id"]: item for item in phase17["observations"]}
    evaluations = {item["case_id"]: item for item in phase18["evaluations"]}
    cases = [
        _case_evidence(case, observations[case.id], evaluations.get(case.id))
        for case in COMPLEX_INCIDENT_CASES
    ]
    incident_cases = [item for item in cases if item["incident"] is not None]
    no_action_cases = [item for item in cases if item["incident"] is None]
    recovered = [
        item
        for item in incident_cases
        if item["observation"]["incident_status"] == "recovered"
    ]
    blocked = [
        item
        for item in incident_cases
        if item["observation"]["incident_status"] == "mitigation_blocked"
    ]
    recovery_failed = [
        item
        for item in incident_cases
        if item["observation"]["incident_status"] == "recovery_failed"
    ]
    body = {
        "version": BUNDLE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "classification": "controlled_local_lab_not_remote_or_production",
        "claim": (
            "Observed local evidence for the time from first signal to safe decision "
            "and verified outcome; no counterfactual or customer call is invented."
        ),
        "source_artifacts": {
            "phase_17": {
                "sha256": phase17_sha256,
                "all_gates_passed": True,
            },
            "phase_18": {
                "sha256": phase18_sha256,
                "all_gates_passed": True,
                "provider_mode": phase18["provider_mode"],
            },
        },
        "summary": {
            "case_count": len(cases),
            "incident_case_count": len(incident_cases),
            "no_action_case_count": len(no_action_cases),
            "recovered_count": len(recovered),
            "mitigation_blocked_count": len(blocked),
            "recovery_failed_count": len(recovery_failed),
            "false_recovery_count": sum(
                "RECOVERY_VERIFIED" in [event["type"] for event in item["timeline"]]
                for item in incident_cases
                if item["observation"]["incident_status"] != "recovered"
            ),
            "first_signal_to_safe_decision": _timing_summary(
                incident_cases, "safe_decision"
            ),
            "first_signal_to_verified_outcome": _timing_summary(
                recovered, "verification"
            ),
        },
        "cases": cases,
    }
    return {**body, "content_sha256": content_sha256(body)}


def _sensitive_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}"
            if _is_sensitive_key(key):
                found.append(current)
            found.extend(_sensitive_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_sensitive_paths(item, f"{path}[{index}]"))
    return found


def verify_evidence_bundle(bundle: dict[str, Any]) -> dict[str, bool]:
    body = {key: value for key, value in bundle.items() if key != "content_sha256"}
    cases = bundle.get("cases") or []
    case_hashes_valid = all(
        item.get("evidence_sha256")
        == content_sha256(
            {key: value for key, value in item.items() if key != "evidence_sha256"}
        )
        for item in cases
    )
    causal_contracts_valid = True
    no_action_cases_valid = True
    evaluations_valid = True
    for item in cases:
        event_types = [event["type"] for event in item["timeline"]]
        required = item["contract"]["required_events"]
        forbidden = item["contract"]["forbidden_events"]
        positions = [event_types.index(name) for name in required if name in event_types]
        causal_contracts_valid = causal_contracts_valid and all(
            (
                set(required).issubset(event_types),
                not set(forbidden).intersection(event_types),
                positions == sorted(positions),
                event_types == item["observation"]["event_types"],
            )
        )
        if item["incident"] is None:
            no_action_cases_valid = no_action_cases_valid and all(
                (
                    not item["actions"],
                    not item["observation"]["action_states"],
                    item["causal_chain"]["safe_decision"] is None,
                    item["causal_chain"]["verification"] is None,
                    item["operator_investigation"] is None,
                    item["explanation_evaluation"] is None,
                )
            )
        else:
            evaluation = item["explanation_evaluation"] or {}
            evaluations_valid = evaluations_valid and all(
                (
                    evaluation.get("passed") is True,
                    not evaluation.get("hard_gate_failures"),
                    item["operator_investigation"] is not None,
                )
            )

    summary = bundle.get("summary") or {}
    sources = bundle.get("source_artifacts") or {}
    gates = {
        "bundle_hash_valid": bundle.get("content_sha256") == content_sha256(body),
        "all_case_hashes_valid": case_hashes_valid,
        "twelve_unique_cases_present": len(cases) == 12
        and len({item.get("case_id") for item in cases}) == 12,
        "source_reports_passed": set(sources) == {"phase_17", "phase_18"}
        and all(source.get("all_gates_passed") is True for source in sources.values()),
        "causal_contracts_valid": causal_contracts_valid,
        "no_action_cases_are_not_invented": no_action_cases_valid,
        "incident_explanations_passed_phase_18": evaluations_valid,
        "no_sensitive_keys_exported": not _sensitive_paths(bundle),
        "no_false_recovery": summary.get("false_recovery_count") == 0,
    }
    return gates
