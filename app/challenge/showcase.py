from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.response import response_plan_for


SHOWCASE_VERSION = "challenge-kaiku-guided-showcase-v1"
REQUIRED_RELEASE_FILES = frozenset(
    {
        "release-report.json",
        "context-ablation.json",
        "log-ablation.json",
        "end-to-end-evidence.json",
        "replay/alerts.full.parquet",
    }
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
        raise IngestionError(f"expected JSON object: {path.name}")
    return value


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _verify_release_manifest(release: Path) -> dict[str, Any]:
    manifest = _load_json(release / "manifest.json")
    if manifest.get("status") != "passed":
        raise IngestionError("showcase requires a passed release package")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise IngestionError("release manifest does not list package files")
    missing_required = sorted(REQUIRED_RELEASE_FILES - set(files))
    if missing_required:
        raise IngestionError(
            f"release manifest omits showcase dependencies: {missing_required}"
        )
    mismatches: list[str] = []
    for relative, expected_digest in sorted(files.items()):
        candidate = (release / str(relative)).resolve(strict=False)
        try:
            candidate.relative_to(release)
        except ValueError as exc:
            raise IngestionError("release manifest contains path traversal") from exc
        if not candidate.is_file() or _sha256_file(candidate) != expected_digest:
            mismatches.append(str(relative))
    if mismatches:
        raise IngestionError(f"release package hash mismatch: {mismatches[:5]}")
    return {"passed": True, "files_verified": len(files)}


def _parse_evidence(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise IngestionError("showcase alert evidence is not valid JSON") from exc
    if not isinstance(parsed, (list, dict)):
        raise IngestionError("showcase alert evidence must be structured")
    return parsed


def _timestamp(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def build_guided_showcase(
    release_directory: Path,
    training_report_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    release = assert_allowed_path(release_directory)
    manifest_gate = _verify_release_manifest(release)
    release_report = _load_json(release / "release-report.json")
    if release_report.get("status") != "passed":
        raise IngestionError("showcase requires a passed release report")
    training = _load_json(training_report_path)
    split_runs = training.get("splits", {}).get("runs", {})
    test_runs = split_runs.get("test", [])
    train_runs = split_runs.get("train", [])
    validation_runs = split_runs.get("validation", [])
    if (
        not isinstance(test_runs, list)
        or len(test_runs) != 9
        or set(test_runs) & set(train_runs)
        or set(test_runs) & set(validation_runs)
    ):
        raise IngestionError("showcase requires the isolated nine-run test split")

    alerts = pd.read_parquet(release / "replay" / "alerts.full.parquet")
    required = {
        "run_id",
        "tenant",
        "signal_timestamp",
        "timestamp",
        "decision_latency_minutes",
        "alert_type",
        "risk_score",
        "suspected_incident_type",
        "context_known_event",
        "data_quality",
        "evidence",
        "recommended_action",
        "model_version",
    }
    missing = sorted(required - set(alerts.columns))
    if missing:
        raise IngestionError(f"release replay is missing showcase columns: {missing}")
    test_alerts = alerts.loc[alerts["run_id"].astype(str).isin(test_runs)].copy()
    eligible = test_alerts.loc[
        test_alerts["alert_type"].eq("incident_risk")
        & test_alerts["suspected_incident_type"].notna()
        & ~test_alerts["context_known_event"].astype(bool)
    ].copy()
    if eligible.empty:
        raise IngestionError("test split contains no typed showcase alert")
    eligible = eligible.sort_values(
        ["risk_score", "run_id", "timestamp"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )
    selected = eligible.iloc[0]
    selected_type = str(selected["suspected_incident_type"])
    response_plan = response_plan_for("incident_risk", selected_type)
    selected_evidence = _parse_evidence(selected["evidence"])
    if not isinstance(selected_evidence, list) or not selected_evidence:
        raise IngestionError("representative risk alert has no causal evidence")

    quality_candidates = test_alerts.loc[
        test_alerts["alert_type"].eq("data_quality_alert")
    ].sort_values(["run_id", "timestamp"], kind="stable")
    same_run_quality = quality_candidates.loc[
        quality_candidates["run_id"].eq(selected["run_id"])
    ]
    if not same_run_quality.empty:
        quality_selected = same_run_quality.iloc[0]
    elif not quality_candidates.empty:
        quality_selected = quality_candidates.iloc[0]
    else:
        raise IngestionError("test split contains no data-quality showcase alert")

    context = _load_json(release / "context-ablation.json")
    logs = _load_json(release / "log-ablation.json")
    end_to_end = _load_json(release / "end-to-end-evidence.json")
    gates = {
        "release_package_integrity": manifest_gate,
        "isolated_test_case": {
            "passed": str(selected["run_id"]) in set(test_runs),
            "split": "test",
            "test_run_count": len(test_runs),
        },
        "causal_evidence_present": {
            "passed": bool(selected_evidence),
            "signal_count": len(selected_evidence),
        },
        "known_event_excluded": {
            "passed": not bool(selected["context_known_event"]),
        },
        "context_ablation": {
            "passed": (
                context.get("with_context", {}).get("known_event_incident_alerts") == 0
                and context.get("without_context", {}).get(
                    "known_event_incident_alerts", 0
                )
                > 0
            ),
            "with_context_known_event_alerts": context.get("with_context", {}).get(
                "known_event_incident_alerts"
            ),
            "without_context_known_event_alerts": context.get(
                "without_context", {}
            ).get("known_event_incident_alerts"),
        },
        "logs_are_explanatory_only": {
            "passed": logs.get("risk_decisions_unchanged") is True,
            "typed_with_logs": logs.get("with_logs", {}).get(
                "typed_incident_alerts"
            ),
            "typed_without_logs": logs.get("without_logs", {}).get(
                "typed_incident_alerts"
            ),
        },
        "data_quality_separated": {
            "passed": quality_selected["alert_type"] == "data_quality_alert",
        },
        "action_boundary": {
            "passed": (
                response_plan["execution_authorized"] is False
                and end_to_end.get("limitations", {}).get(
                    "automatic_action_executed"
                )
                is False
            ),
            "execution_authorized": response_plan["execution_authorized"],
            "mitigation_executed": False,
            "recovery_verified": False,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    showcase = {
        "version": SHOWCASE_VERSION,
        "status": "passed" if passed else "failed",
        "scope": "demostración guiada del backend sobre el split sintético oficial de prueba de Kaiku",
        "company": {
            "tenant": "kaiku",
            "business": "tienda en línea",
            "critical_flow": "catálogo → carrito → checkout → pago → pedido",
        },
        "selection_rule": (
            "mayor score de riesgo entre alertas tipificadas fuera de eventos "
            "conocidos en el split de prueba congelado; los empates usan run_id y timestamp"
        ),
        "representative_case": {
            "run_id": str(selected["run_id"]),
            "tenant": str(selected["tenant"]),
            "split": "test",
            "signal_timestamp": _timestamp(selected["signal_timestamp"]),
            "alert_timestamp": _timestamp(selected["timestamp"]),
            "signal_to_alert_minutes": float(selected["decision_latency_minutes"]),
            "risk_score": float(selected["risk_score"]),
            "confidence_semantics": "score de riesgo, no probabilidad calibrada de incidente",
            "suspected_incident_type": selected_type,
            "model_version": str(selected["model_version"]),
            "evidence": selected_evidence,
            "recommended_action": str(selected["recommended_action"]),
            "response_plan": response_plan,
            "business_interpretation": (
                "Las señales observadas podrían degradar la capacidad del checkout; "
                "el kit público no demuestra impacto al cliente ni causa raíz."
            ),
        },
        "false_alarm_control": {
            "known_event_alerts_with_context": gates["context_ablation"][
                "with_context_known_event_alerts"
            ],
            "known_event_alerts_without_context": gates["context_ablation"][
                "without_context_known_event_alerts"
            ],
            "interpretation": (
                "La ablación cambia el comportamiento de alertas; no es una métrica oficial de precisión."
            ),
        },
        "telemetry_outage_case": {
            "run_id": str(quality_selected["run_id"]),
            "signal_timestamp": _timestamp(quality_selected["signal_timestamp"]),
            "alert_timestamp": _timestamp(quality_selected["timestamp"]),
            "data_quality": str(quality_selected["data_quality"]),
            "evidence": _parse_evidence(quality_selected["evidence"]),
            "classification": "data_quality_alert",
            "interpretation": (
                "La pérdida de visibilidad se alerta por separado y no se presenta "
                "como incidente del negocio sin evidencia adicional."
            ),
        },
        "demo_api_sequence": [
            {
                "method": "POST",
                "path": "/api/challenge/replay",
                "body": {"run_id": str(selected["run_id"])},
                "authentication": "X-Admin-Token",
            },
            {
                "method": "GET",
                "path": (
                    "/api/challenge/alerts?run_id=" + str(selected["run_id"])
                ),
                "authentication": "none",
            },
            {
                "method": "GET",
                "path": "/api/challenge/alerts/stream",
                "authentication": "none",
            },
        ],
        "gates": gates,
        "claim_boundary": {
            "official_ground_truth_used": False,
            "representative_alert_confirmed_real_incident": False,
            "customer_impact_observed": False,
            "root_cause_demonstrated": False,
            "mitigation_executed": False,
            "recovery_verified": False,
        },
    }
    output = assert_allowed_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    showcase_path = output / "showcase.json"
    _atomic_json(showcase, showcase_path)
    showcase_manifest = {
        "version": SHOWCASE_VERSION,
        "status": showcase["status"],
        "file": showcase_path.name,
        "sha256": _sha256_file(showcase_path),
    }
    _atomic_json(showcase_manifest, output / "showcase-manifest.json")
    return {**showcase, "output_directory": str(output)}


def load_showcase_artifact(path: Path) -> dict[str, Any]:
    source = assert_allowed_path(path)
    manifest = _load_json(source.with_name("showcase-manifest.json"))
    if manifest.get("file") != source.name:
        raise IngestionError("showcase manifest points to a different file")
    if manifest.get("sha256") != _sha256_file(source):
        raise IngestionError("showcase artifact hash mismatch")
    payload = _load_json(source)
    if payload.get("version") != SHOWCASE_VERSION or payload.get("status") != "passed":
        raise IngestionError("showcase artifact is not a passed compatible release")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic guided showcase from the Kaiku release"
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    result = build_guided_showcase(
        args.release,
        args.training_report,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "representative_case": result["representative_case"],
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
