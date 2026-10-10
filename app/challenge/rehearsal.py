from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any, Callable

import httpx

from app.challenge.ingestion import IngestionError, assert_allowed_path
from app.challenge.showcase import load_showcase_artifact


REHEARSAL_VERSION = "challenge-kaiku-live-rehearsal-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def load_rehearsal_artifact(path: Path) -> dict[str, Any]:
    report_path = assert_allowed_path(path)
    manifest_path = report_path.with_name("rehearsal-manifest.json")
    if not report_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(report_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise IngestionError("invalid live rehearsal artifact") from exc
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise IngestionError("invalid live rehearsal artifact")
    if manifest.get("file") != report_path.name:
        raise IngestionError("live rehearsal manifest points to another file")
    if manifest.get("sha256") != _sha256_file(report_path):
        raise IngestionError("live rehearsal hash mismatch")
    if (
        manifest.get("version") != REHEARSAL_VERSION
        or report.get("version") != REHEARSAL_VERSION
    ):
        raise IngestionError("unsupported live rehearsal version")
    if manifest.get("status") != report.get("status"):
        raise IngestionError("live rehearsal status mismatch")
    return report


def allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(
    base_url: str,
    process: subprocess.Popen[Any],
    *,
    timeout_seconds: float,
) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"rehearsal backend exited during startup with code {process.returncode}"
            )
        try:
            response = httpx.get(f"{base_url}/health/ready", timeout=1.0)
            if response.status_code == 200:
                return (time.monotonic() - started) * 1000
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("rehearsal backend did not become ready")


def _stop_process(process: subprocess.Popen[Any]) -> dict[str, Any]:
    forced = False
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            forced = True
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
    return {
        "stopped": process.poll() is not None,
        "exit_code": process.returncode,
        "forced": forced,
    }


def _consume_sse(
    base_url: str,
    *,
    after_id: int,
    last_event_id: int | None,
    predicate: Callable[[str, dict[str, Any]], bool],
    ready: threading.Event,
    result: queue.Queue[tuple[str, object]],
    timeout_seconds: float,
) -> None:
    headers = (
        {"Last-Event-ID": str(last_event_id)}
        if last_event_id is not None
        else {}
    )
    try:
        timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            with client.stream(
                "GET",
                "/api/challenge/alerts/stream",
                params={"after_id": after_id},
                headers=headers,
            ) as response:
                response.raise_for_status()
                ready.set()
                event_name = "message"
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        event_name = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        payload = json.loads(line.removeprefix("data: "))
                        if predicate(event_name, payload):
                            result.put(("event", {"name": event_name, "data": payload}))
                            return
        result.put(("error", "SSE stream ended before the expected event"))
    except Exception as exc:  # result is re-raised in the controlling thread
        ready.set()
        result.put(("error", f"{type(exc).__name__}: {exc}"))


def _start_sse_consumer(
    base_url: str,
    *,
    after_id: int,
    last_event_id: int | None = None,
    predicate: Callable[[str, dict[str, Any]], bool],
    timeout_seconds: float,
) -> tuple[threading.Thread, queue.Queue[tuple[str, object]]]:
    ready = threading.Event()
    result: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    thread = threading.Thread(
        target=_consume_sse,
        kwargs={
            "base_url": base_url,
            "after_id": after_id,
            "last_event_id": last_event_id,
            "predicate": predicate,
            "ready": ready,
            "result": result,
            "timeout_seconds": timeout_seconds,
        },
        name="kaiku-rehearsal-sse",
        daemon=True,
    )
    thread.start()
    if not ready.wait(timeout=10):
        raise RuntimeError("SSE consumer did not connect")
    return thread, result


def _await_sse(
    thread: threading.Thread,
    result: queue.Queue[tuple[str, object]],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise RuntimeError("timed out waiting for the expected SSE event")
    kind, value = result.get_nowait()
    if kind == "error":
        raise RuntimeError(str(value))
    if not isinstance(value, dict):
        raise RuntimeError("SSE consumer returned an invalid result")
    return value


def run_live_rehearsal(
    *,
    features_path: Path,
    model_path: Path,
    policy_path: Path,
    showcase_path: Path,
    output_directory: Path,
    startup_timeout_seconds: float = 30,
    event_timeout_seconds: float = 60,
) -> dict[str, Any]:
    features = assert_allowed_path(features_path)
    model = assert_allowed_path(model_path)
    policy = assert_allowed_path(policy_path)
    showcase_source = assert_allowed_path(showcase_path)
    for required in (features, model, policy, showcase_source):
        if not required.exists():
            raise FileNotFoundError(required)
    showcase = load_showcase_artifact(showcase_source)
    representative = showcase["representative_case"]
    run_id = str(representative["run_id"])
    expected_timestamp = str(representative["alert_timestamp"])
    admin_token = secrets.token_urlsafe(32)
    output = assert_allowed_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    total_started = time.monotonic()

    with TemporaryDirectory(prefix="zikit-phase-10-") as temporary_name:
        temporary = Path(temporary_name)
        database_path = temporary / "rehearsal.db"
        log_path = temporary / "backend.log"
        port = allocate_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment.update(
            {
                "DATABASE_URL": f"sqlite:///{database_path}",
                "ADMIN_TOKEN": admin_token,
                "MONITOR_ENABLED": "false",
                "TRAFFIC_ENABLED": "false",
                "RESOURCE_MONITOR_ENABLED": "false",
                "AUTO_MITIGATION_ENABLED": "false",
                "MCP_HTTP_ENABLED": "false",
                "EVIDENCE_API_ENABLED": "false",
                "CHALLENGE_FEATURES_PATH": str(features),
                "CHALLENGE_MODEL_PATH": str(model),
                "CHALLENGE_POLICY_PATH": str(policy),
                "CHALLENGE_SHOWCASE_PATH": str(showcase_source),
                "CHALLENGE_REQUIRE_ALL_TENANTS": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        with log_path.open("w", encoding="utf-8") as backend_log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=backend_log,
                stderr=subprocess.STDOUT,
            )
            stop_result: dict[str, Any] | None = None
            try:
                startup_ms = _wait_for_health(
                    base_url,
                    process,
                    timeout_seconds=startup_timeout_seconds,
                )
                headers = {"X-Admin-Token": admin_token}
                with httpx.Client(base_url=base_url, timeout=event_timeout_seconds) as client:
                    contract_response = client.get("/api/challenge/contract")
                    contract_response.raise_for_status()
                    showcase_response = client.get("/api/challenge/showcase")
                    showcase_response.raise_for_status()
                    cursor_before = client.get("/api/challenge/alerts/cursor").json()

                    alert_thread, alert_queue = _start_sse_consumer(
                        base_url,
                        after_id=int(cursor_before["latest_id"]),
                        predicate=lambda event, payload: (
                            event == "challenge_alert"
                            and payload.get("run_id") == run_id
                            and payload.get("alert_type") == "incident_risk"
                            and str(payload.get("timestamp", "")).replace("+00:00", "")
                            == expected_timestamp
                        ),
                        timeout_seconds=event_timeout_seconds,
                    )
                    replay_started = time.monotonic()
                    replay_response = client.post(
                        "/api/challenge/replay",
                        headers=headers,
                        json={"run_id": run_id},
                    )
                    replay_ms = (time.monotonic() - replay_started) * 1000
                    replay_response.raise_for_status()
                    replay_payload = replay_response.json()
                    alert_sse = _await_sse(
                        alert_thread,
                        alert_queue,
                        timeout_seconds=event_timeout_seconds,
                    )

                    alerts_response = client.get(
                        "/api/challenge/alerts",
                        params={"run_id": run_id, "alert_type": "all", "limit": 1000},
                    )
                    alerts_response.raise_for_status()
                    alerts = alerts_response.json()
                    selected = next(
                        (
                            item
                            for item in alerts
                            if item["alert_type"] == "incident_risk"
                            and str(item["timestamp"]).replace("+00:00", "")
                            == expected_timestamp
                        ),
                        None,
                    )
                    if selected is None:
                        raise RuntimeError("representative showcase alert was not persisted")
                    explanation_response = client.get(
                        f"/api/challenge/alerts/{selected['event_id']}/explanation"
                    )
                    explanation_response.raise_for_status()
                    explanation = explanation_response.json()
                    cursor_before_ack = client.get(
                        "/api/challenge/alerts/cursor"
                    ).json()

                    acknowledge_response = client.post(
                        f"/api/admin/alerts/{selected['outbox_alert_id']}/acknowledge",
                        headers=headers,
                    )
                    acknowledge_response.raise_for_status()
                    acknowledgement = acknowledge_response.json()
                    ack_thread, ack_queue = _start_sse_consumer(
                        base_url,
                        after_id=0,
                        last_event_id=int(cursor_before_ack["latest_id"]),
                        predicate=lambda event, payload: (
                            event == "challenge_alert_acknowledged"
                            and payload.get("outbox_alert_id")
                            == selected["outbox_alert_id"]
                        ),
                        timeout_seconds=event_timeout_seconds,
                    )
                    acknowledgement_sse = _await_sse(
                        ack_thread,
                        ack_queue,
                        timeout_seconds=event_timeout_seconds,
                    )
                    cursor_after_ack = client.get(
                        "/api/challenge/alerts/cursor"
                    ).json()
                    acknowledged_again = client.post(
                        f"/api/admin/alerts/{selected['outbox_alert_id']}/acknowledge",
                        headers=headers,
                    )
                    acknowledged_again.raise_for_status()
                    cursor_after_repeat_ack = client.get(
                        "/api/challenge/alerts/cursor"
                    ).json()
                    duplicate_response = client.post(
                        "/api/challenge/replay",
                        headers=headers,
                        json={"run_id": run_id},
                    )
                    duplicate_response.raise_for_status()
                    duplicate_payload = duplicate_response.json()
                    operations_response = client.get("/api/challenge/operations")
                    operations_response.raise_for_status()
                    operations = operations_response.json()
            finally:
                stop_result = _stop_process(process)

        if process.returncode not in {0, None}:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"rehearsal backend stopped with code {process.returncode}: {log_tail}"
            )

    gates = {
        "backend_started": {
            "passed": (
                startup_ms > 0
                and stop_result["stopped"]
                and stop_result["exit_code"] == 0
                and stop_result["forced"] is False
            ),
            "startup_milliseconds": startup_ms,
            "clean_shutdown": stop_result["exit_code"] == 0,
            "forced_shutdown": stop_result["forced"],
        },
        "contract_and_showcase": {
            "passed": (
                contract_response.json()["tenant"] == "kaiku"
                and showcase_response.json()["version"] == showcase["version"]
                and showcase_response.json()["representative_case"]["run_id"] == run_id
            ),
        },
        "live_replay": {
            "passed": (
                replay_payload["new_alerts"]
                == replay_payload["incident_alerts"]
                + replay_payload["data_quality_alerts"]
                and replay_payload["new_alerts"] > 0
            ),
            "run_id": run_id,
            "request_milliseconds": replay_ms,
            "incident_alerts": replay_payload["incident_alerts"],
            "data_quality_alerts": replay_payload["data_quality_alerts"],
        },
        "alert_received_by_sse": {
            "passed": (
                alert_sse["name"] == "challenge_alert"
                and alert_sse["data"]["event_id"] == selected["event_id"]
            ),
            "event_id": alert_sse["data"]["event_id"],
        },
        "evidence_and_response": {
            "passed": (
                explanation["signal"]["status"] == "observed"
                and bool(explanation["signal"]["top_signals"])
                and explanation["action"]["status"] == "not_started"
                and explanation["action"]["execution_authorized"] is False
            ),
            "incident_id": selected["incident_id"],
            "runbook": explanation["action"]["runbook"],
        },
        "acknowledgement_resumed_by_cursor": {
            "passed": (
                acknowledgement["state"] == "acknowledged"
                and acknowledgement["challenge_trace"]["mitigation_executed"] is False
                and acknowledgement_sse["name"]
                == "challenge_alert_acknowledged"
                and cursor_after_ack["latest_id"]
                == cursor_after_repeat_ack["latest_id"]
            ),
            "delivery_to_acknowledgement_seconds": acknowledgement[
                "challenge_trace"
            ]["delivery_to_acknowledgement_seconds"],
            "cursor_before_ack": cursor_before_ack["latest_id"],
            "cursor_after_ack": cursor_after_ack["latest_id"],
            "cursor_after_repeat_ack": cursor_after_repeat_ack["latest_id"],
        },
        "idempotent_replay": {
            "passed": (
                duplicate_payload["new_alerts"] == 0
                and duplicate_payload["existing_alerts"]
                == replay_payload["new_alerts"]
            ),
            "new_alerts_on_repeat": duplicate_payload["new_alerts"],
            "existing_alerts_on_repeat": duplicate_payload["existing_alerts"],
        },
        "durable_integrity": {
            "passed": operations["integrity"]["passed"] is True,
            "integrity": operations["integrity"],
        },
        "safety_boundary": {
            "passed": (
                selected["action_executed"] is False
                and explanation["verification"]["status"] == "pending"
                and showcase["claim_boundary"]["mitigation_executed"] is False
            ),
            "gemini_used": False,
            "mcp_required": False,
            "mitigation_executed": False,
            "recovery_verified": False,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    report = {
        "version": REHEARSAL_VERSION,
        "status": "passed" if passed else "failed",
        "scope": "live loopback HTTP rehearsal using official synthetic Kaiku replay",
        "environment": {
            "transport": "real HTTP over 127.0.0.1",
            "database": "temporary SQLite",
            "backend_process": "Uvicorn child process",
            "remote_machine": False,
        },
        "case": {
            "run_id": run_id,
            "signal_timestamp": representative["signal_timestamp"],
            "alert_timestamp": representative["alert_timestamp"],
            "synthetic_signal_to_alert_minutes": representative[
                "signal_to_alert_minutes"
            ],
            "suspected_incident_type": representative[
                "suspected_incident_type"
            ],
            "incident_id": selected["incident_id"],
            "event_id": selected["event_id"],
            "outbox_alert_id": selected["outbox_alert_id"],
        },
        "gates": gates,
        "total_wall_milliseconds": (time.monotonic() - total_started) * 1000,
        "limitations": {
            "official_ground_truth_used": False,
            "representative_alert_confirmed_real_incident": False,
            "remote_infrastructure_observed": False,
            "customer_impact_observed": False,
            "mitigation_executed": False,
            "recovery_verified": False,
        },
    }
    report_path = output / "rehearsal-report.json"
    _atomic_json(report, report_path)
    manifest = {
        "version": REHEARSAL_VERSION,
        "status": report["status"],
        "file": report_path.name,
        "sha256": _sha256_file(report_path),
    }
    _atomic_json(manifest, output / "rehearsal-manifest.json")
    return {**report, "output_directory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the live one-command Kaiku backend rehearsal"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/challenge-phase-3-logs"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/challenge_early_warning.joblib"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("models/challenge_alert_policy.json"),
    )
    parser.add_argument(
        "--showcase",
        type=Path,
        default=Path("data/challenge-phase-9/showcase.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/challenge-phase-10"),
    )
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--event-timeout", type=float, default=60)
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    if args.startup_timeout <= 0 or args.event_timeout <= 0:
        parser.error("timeouts must be positive")
    result = run_live_rehearsal(
        features_path=args.features,
        model_path=args.model,
        policy_path=args.policy,
        showcase_path=args.showcase,
        output_directory=args.output,
        startup_timeout_seconds=args.startup_timeout,
        event_timeout_seconds=args.event_timeout,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "case": result["case"],
                "gates": {
                    name: gate["passed"] for name, gate in result["gates"].items()
                },
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
