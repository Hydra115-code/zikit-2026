from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any
from uuid import uuid4

import numpy as np

from app.database import ExperimentRecord
from app.schemas import ExperimentMode, ReplayPlanRequest, Scenario

REPLAY_METADATA_KEY = "_replay"


def scenario_fingerprint(value: Scenario | dict[str, Any]) -> str:
    scenario = value if isinstance(value, Scenario) else Scenario.model_validate(value)
    canonical = json.dumps(
        scenario.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def persisted_scenario(
    scenario: Scenario,
    *,
    replay_group_id: str | None,
    repetition: int | None,
) -> dict[str, Any]:
    payload = scenario.model_dump(mode="json")
    if replay_group_id is not None:
        payload[REPLAY_METADATA_KEY] = {
            "group_id": replay_group_id,
            "repetition": repetition,
            "fingerprint": scenario_fingerprint(scenario),
        }
    return payload


def replay_metadata(record: ExperimentRecord) -> dict[str, Any]:
    value = (record.scenario or {}).get(REPLAY_METADATA_KEY, {})
    return value if isinstance(value, dict) else {}


def validate_pair(automatic: ExperimentRecord, manual: ExperimentRecord) -> str:
    auto_fingerprint = scenario_fingerprint(automatic.scenario)
    manual_fingerprint = scenario_fingerprint(manual.scenario)
    if auto_fingerprint != manual_fingerprint:
        raise ValueError("automatic and manual experiments do not use the same scenario")
    auto_meta = replay_metadata(automatic)
    manual_meta = replay_metadata(manual)
    if auto_meta or manual_meta:
        if not auto_meta or not manual_meta:
            raise ValueError("both experiments must belong to the same replay pair")
        if (
            auto_meta.get("group_id") != manual_meta.get("group_id")
            or auto_meta.get("repetition") != manual_meta.get("repetition")
        ):
            raise ValueError("automatic and manual experiments belong to different replay pairs")
    return auto_fingerprint


def build_replay_plan(payload: ReplayPlanRequest) -> dict[str, Any]:
    group_id = payload.group_id or str(uuid4())
    runs: list[dict[str, Any]] = []
    for repetition in range(1, payload.repetitions + 1):
        seed = payload.scenario.seed + repetition - 1
        scenario = payload.scenario.model_copy(update={"seed": seed})
        fingerprint = scenario_fingerprint(scenario)
        for mode in (ExperimentMode.AUTOMATIC, ExperimentMode.MANUAL):
            runs.append(
                {
                    "replay_group_id": group_id,
                    "repetition": repetition,
                    "mode": mode.value,
                    "scenario": scenario.model_dump(mode="json"),
                    "scenario_fingerprint": fingerprint,
                }
            )
    return {
        "replay_group_id": group_id,
        "repetitions": payload.repetitions,
        "run_count": len(runs),
        "execution_order": "automatic_then_reset_then_manual_per_repetition",
        "runs": runs,
    }


def aggregate_replay_group(records: list[ExperimentRecord], group_id: str) -> dict[str, Any]:
    grouped: dict[int, dict[str, ExperimentRecord]] = defaultdict(dict)
    for record in records:
        metadata = replay_metadata(record)
        if metadata.get("group_id") != group_id:
            continue
        repetition = metadata.get("repetition")
        if isinstance(repetition, int):
            grouped[repetition][record.mode] = record

    pairs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for repetition, modes in sorted(grouped.items()):
        automatic = modes.get("automatic")
        manual = modes.get("manual")
        if automatic is None or manual is None:
            excluded.append({"repetition": repetition, "reason": "pair_incomplete"})
            continue
        try:
            fingerprint = validate_pair(automatic, manual)
        except ValueError as exc:
            excluded.append({"repetition": repetition, "reason": str(exc)})
            continue
        auto_time = _recovery_time(automatic)
        manual_time = _recovery_time(manual)
        if auto_time is None or manual_time is None or manual_time <= 0:
            excluded.append({"repetition": repetition, "reason": "invalid_measurement"})
            continue
        auto_impact = (automatic.result or {}).get("impact", {})
        manual_impact = (manual.result or {}).get("impact", {})
        auto_action_time = _duration(automatic, "first_signal_to_action_seconds")
        manual_action_time = _duration(manual, "first_signal_to_action_seconds")
        pairs.append(
            {
                "repetition": repetition,
                "scenario_fingerprint": fingerprint,
                "automatic_experiment_id": automatic.id,
                "manual_experiment_id": manual.id,
                "automatic_seconds": auto_time,
                "manual_seconds": manual_time,
                "reduction_percent": (manual_time - auto_time) / manual_time * 100,
                "automatic_action_seconds": auto_action_time,
                "manual_action_seconds": manual_action_time,
                "action_reduction_percent": (
                    (manual_action_time - auto_action_time) / manual_action_time * 100
                    if auto_action_time is not None
                    and manual_action_time is not None
                    and manual_action_time > 0
                    else None
                ),
                "customer_impact_prevented": bool(manual_impact.get("customer_impact_observed"))
                and bool(auto_impact.get("customer_impact_avoided")),
                "customer_complaint_prevented": bool(manual_impact.get("customer_complaint_observed"))
                and bool(auto_impact.get("customer_complaint_avoided")),
            }
        )

    if not pairs:
        raise ValueError("replay group has no complete valid pairs")
    automatic_times = [pair["automatic_seconds"] for pair in pairs]
    manual_times = [pair["manual_seconds"] for pair in pairs]
    reductions = [pair["reduction_percent"] for pair in pairs]
    action_pairs = [
        pair
        for pair in pairs
        if pair["automatic_action_seconds"] is not None
        and pair["manual_action_seconds"] is not None
        and pair["action_reduction_percent"] is not None
    ]
    report = {
        "replay_group_id": group_id,
        "valid_pairs": len(pairs),
        "excluded_pairs": excluded,
        "automatic_seconds": _summary(automatic_times),
        "manual_seconds": _summary(manual_times),
        "reduction_percent": _summary(reductions),
        "customer_impact_prevented_pairs": sum(pair["customer_impact_prevented"] for pair in pairs),
        "customer_complaint_prevented_pairs": sum(
            pair["customer_complaint_prevented"] for pair in pairs
        ),
        "pairs": pairs,
    }
    if action_pairs:
        report["automatic_action_seconds"] = _summary(
            [float(pair["automatic_action_seconds"]) for pair in action_pairs]
        )
        report["manual_action_seconds"] = _summary(
            [float(pair["manual_action_seconds"]) for pair in action_pairs]
        )
        report["action_reduction_percent"] = _summary(
            [float(pair["action_reduction_percent"]) for pair in action_pairs]
        )
    return report


def _recovery_time(record: ExperimentRecord) -> float | None:
    return _duration(record, "first_signal_to_recovery_seconds")


def _duration(record: ExperimentRecord, name: str) -> float | None:
    value = (record.result or {}).get("durations", {}).get(name)
    return float(value) if value is not None else None


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
