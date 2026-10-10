from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from app.challenge.contracts import TARGET_TENANT
from app.challenge.features import FEATURE_SCHEMA_VERSION
from app.challenge.ingestion import IngestionError, RUN_ID_PATTERN, assert_allowed_path
from app.challenge.model import load_challenge_bundle
from app.challenge.persistence import ChallengePredictionRepository
from app.challenge.policy import load_policy_artifact
from app.challenge.replay import _replay_run, _validate_run_frame
from app.challenge.training import discover_feature_files


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ChallengeReplayExecution:
    run_id: str
    rows_processed: int
    tenants: tuple[str, ...]
    alerts: pd.DataFrame
    model_sha256: str
    policy_sha256: str
    elapsed_milliseconds: float
    timeline: pd.DataFrame | None = None


class ChallengeReplayService:
    def __init__(
        self,
        feature_directory: Path,
        model_path: Path,
        policy_path: Path,
        *,
        require_all_tenants: bool = True,
        executor: Callable[[str], ChallengeReplayExecution] | None = None,
    ) -> None:
        self.feature_directory = feature_directory
        self.model_path = model_path
        self.policy_path = policy_path
        self.require_all_tenants = require_all_tenants
        self.store = ChallengePredictionRepository()
        self._replay_lock = asyncio.Lock()
        self._executor = executor or self._execute

    def _execute(self, run_id: str) -> ChallengeReplayExecution:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise IngestionError("invalid replay run id")
        started = perf_counter()
        files, feature_report, _ = discover_feature_files(self.feature_directory)
        if feature_report.get("feature_version") != FEATURE_SCHEMA_VERSION:
            raise IngestionError("incompatible replay feature version")
        causality = feature_report.get("causality", {})
        if causality.get("future_rows_used") is not False:
            raise IngestionError("replay features are not causal")
        if run_id not in files:
            raise FileNotFoundError(f"challenge run not found: {run_id}")

        model = assert_allowed_path(self.model_path)
        model_digest = _sha256_file(model)
        scorer = load_challenge_bundle(model, trusted_root=model.parent)
        if scorer is None:
            raise FileNotFoundError(model)
        policy = assert_allowed_path(self.policy_path)
        config, policy_payload = load_policy_artifact(
            policy,
            trusted_root=policy.parent,
            expected_model_sha256=model_digest,
        )
        if policy_payload.get("model_version") != scorer.model_version:
            raise IngestionError("policy and model versions do not match")

        source_frame = pd.read_parquet(files[run_id])
        if "tenant" not in source_frame.columns:
            raise IngestionError("replay feature input is missing tenant")
        source_frame = source_frame.loc[
            source_frame["tenant"].astype(str).eq(TARGET_TENANT)
        ].copy()
        if source_frame.empty:
            raise IngestionError(f"replay run {run_id} does not contain Kaiku")
        frame = _validate_run_frame(
            source_frame,
            run_id,
            require_all_tenants=self.require_all_tenants,
        )
        alerts, _, _ = _replay_run(
            frame,
            scorer,
            config,
            model_version=scorer.model_version,
        )
        return ChallengeReplayExecution(
            run_id=run_id,
            rows_processed=len(frame),
            tenants=tuple(sorted(str(value) for value in frame["tenant"].unique())),
            alerts=alerts,
            model_sha256=model_digest,
            policy_sha256=_sha256_file(policy),
            elapsed_milliseconds=(perf_counter() - started) * 1000,
            timeline=frame,
        )

    def prepare(self, run_id: str) -> ChallengeReplayExecution:
        """Build one frozen causal execution without persisting its alerts."""
        return self._executor(run_id)

    def available_runs(self) -> list[str]:
        files, feature_report, _ = discover_feature_files(self.feature_directory)
        if feature_report.get("feature_version") != FEATURE_SCHEMA_VERSION:
            raise IngestionError("incompatible replay feature version")
        if feature_report.get("causality", {}).get("future_rows_used") is not False:
            raise IngestionError("replay features are not causal")
        return sorted(files)

    async def replay(self, run_id: str) -> dict[str, Any]:
        async with self._replay_lock:
            execution = self.prepare(run_id)
            stored = self.store.append(
                execution.alerts,
                model_sha256=execution.model_sha256,
                policy_sha256=execution.policy_sha256,
            )
        incident_count = int(
            execution.alerts["alert_type"].eq("incident_risk").sum()
        )
        quality_count = int(
            execution.alerts["alert_type"].eq("data_quality_alert").sum()
        )
        return {
            "run_id": execution.run_id,
            "rows_processed": execution.rows_processed,
            "tenants": list(execution.tenants),
            "incident_alerts": incident_count,
            "data_quality_alerts": quality_count,
            "new_alerts": stored["created"],
            "existing_alerts": stored["existing"],
            "latest_cursor": stored["latest_id"],
            "elapsed_milliseconds": execution.elapsed_milliseconds,
            "model_sha256": execution.model_sha256,
            "policy_sha256": execution.policy_sha256,
            "storage": "durable_database_phase_7",
            "execution": "inline_serialized_phase_7",
        }
