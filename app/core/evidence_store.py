from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.core.evidence_bundle import file_sha256, verify_evidence_bundle


class EvidenceBundleUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceBundleSnapshot:
    content: bytes
    payload: dict[str, Any]
    transport_sha256: str
    verification_gates: dict[str, bool]


class EvidenceBundleStore:
    """Load a sealed evidence artifact and refuse to serve invalid content."""

    def __init__(
        self,
        bundle_path: Path,
        checksum_path: Path | None = None,
        *,
        max_bytes: int = 5_000_000,
    ) -> None:
        self.bundle_path = bundle_path
        self.checksum_path = checksum_path or bundle_path.with_suffix(
            bundle_path.suffix + ".sha256"
        )
        self.max_bytes = max_bytes
        self._cache_key: tuple[int, int, int, int] | None = None
        self._snapshot: EvidenceBundleSnapshot | None = None

    def _stat_key(self) -> tuple[int, int, int, int]:
        try:
            bundle_stat = self.bundle_path.stat()
            checksum_stat = self.checksum_path.stat()
        except OSError as exc:
            raise EvidenceBundleUnavailable("evidence artifact is not available") from exc
        return (
            bundle_stat.st_mtime_ns,
            bundle_stat.st_size,
            checksum_stat.st_mtime_ns,
            checksum_stat.st_size,
        )

    def load(self) -> EvidenceBundleSnapshot:
        key = self._stat_key()
        if self._cache_key == key and self._snapshot is not None:
            return self._snapshot
        if key[1] <= 0 or key[1] > self.max_bytes:
            raise EvidenceBundleUnavailable("evidence artifact size is outside policy")
        try:
            content = self.bundle_path.read_bytes()
            payload = json.loads(content)
            expected_checksum = (
                self.checksum_path.read_text(encoding="utf-8").strip().split()[0]
            )
        except (OSError, UnicodeError, json.JSONDecodeError, IndexError) as exc:
            raise EvidenceBundleUnavailable("evidence artifact could not be decoded") from exc
        actual_checksum = file_sha256(content)
        if expected_checksum != actual_checksum:
            raise EvidenceBundleUnavailable("evidence transport checksum does not match")
        gates = verify_evidence_bundle(payload)
        if not gates or not all(gates.values()):
            raise EvidenceBundleUnavailable("evidence artifact failed verification gates")
        snapshot = EvidenceBundleSnapshot(
            content=content,
            payload=payload,
            transport_sha256=actual_checksum,
            verification_gates=gates,
        )
        self._cache_key = key
        self._snapshot = snapshot
        return snapshot


def case_summary(item: dict[str, Any]) -> dict[str, Any]:
    observation = item["observation"]
    decision = item["causal_chain"]["safe_decision"]
    verification = item["causal_chain"]["verification"]
    evaluation = item.get("explanation_evaluation")
    return {
        "case_id": item["case_id"],
        "title": item["title"],
        "classification": item["classification"],
        "incident_id": observation["incident_id"],
        "incident_status": observation["incident_status"],
        "experiment_status": observation["experiment_status"],
        "active_provider": observation["active_provider"],
        "switch_delta": observation["switch_delta"],
        "decision_event": decision["event_type"] if decision else None,
        "seconds_to_safe_decision": (
            decision["seconds_from_first_signal"] if decision else None
        ),
        "verification_event": verification["event_type"] if verification else None,
        "seconds_to_verified_outcome": (
            verification["seconds_from_first_signal"] if verification else None
        ),
        "explanation_score": evaluation["score"] if evaluation else None,
        "explanation_passed": evaluation["passed"] if evaluation else None,
        "evidence_sha256": item["evidence_sha256"],
    }


def bundle_index(snapshot: EvidenceBundleSnapshot) -> dict[str, Any]:
    payload = snapshot.payload
    return {
        "version": payload["version"],
        "generated_at": payload["generated_at"],
        "classification": payload["classification"],
        "claim": payload["claim"],
        "transport_sha256": snapshot.transport_sha256,
        "content_sha256": payload["content_sha256"],
        "summary": payload["summary"],
        "verification_gates": snapshot.verification_gates,
        "cases": [case_summary(item) for item in payload["cases"]],
    }
