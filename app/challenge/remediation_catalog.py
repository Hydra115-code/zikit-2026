from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.challenge.ingestion import assert_allowed_path


class VerificationRule(BaseModel):
    signal: str = Field(min_length=1, max_length=80)
    maximum_ratio: float = Field(gt=0, le=1)


class ControlledRunbook(BaseModel):
    id: str = Field(min_length=3, max_length=140)
    title: str = Field(min_length=3, max_length=240)
    incident_types: list[str] = Field(min_length=1)
    action: str = Field(min_length=3, max_length=140)
    rollback_action: str = Field(min_length=3, max_length=140)
    reversible: bool
    preauthorized: bool
    effects: dict[str, float]
    verification: list[VerificationRule] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_safe_contract(self) -> "ControlledRunbook":
        if not self.reversible or not self.preauthorized:
            raise ValueError("every autonomous runbook must be reversible and preauthorized")
        if set(self.effects) - {"cpu", "ram", "disco", "latencia_ms", "errores"}:
            raise ValueError("runbook effect contains an unsupported metric")
        if any(value <= 0 or value > 1 for value in self.effects.values()):
            raise ValueError("runbook effects must be ratios in (0, 1]")
        if any(rule.signal not in self.effects for rule in self.verification):
            raise ValueError("verification signals must be affected by the runbook")
        return self


class RemediationCatalog(BaseModel):
    version: str = Field(min_length=1)
    environment: str = Field(pattern=r"^controlled_")
    runbooks: list[ControlledRunbook] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "RemediationCatalog":
        ids = [item.id for item in self.runbooks]
        if len(ids) != len(set(ids)):
            raise ValueError("runbook ids must be unique")
        return self

    def eligible(self, incident_type: str | None) -> list[ControlledRunbook]:
        if not incident_type:
            return []
        return [item for item in self.runbooks if incident_type in item.incident_types]

    def require(self, runbook_id: str) -> ControlledRunbook:
        selected = next((item for item in self.runbooks if item.id == runbook_id), None)
        if selected is None:
            raise ValueError("runbook is outside the configured allowlist")
        return selected


def load_remediation_catalog(path: Path) -> tuple[RemediationCatalog, str]:
    resolved = assert_allowed_path(path)
    content = resolved.read_bytes()
    catalog = RemediationCatalog.model_validate(json.loads(content))
    return catalog, sha256(content).hexdigest()
