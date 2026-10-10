from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FaultType(StrEnum):
    NONE = "none"
    CONSTANT_LATENCY = "constant_latency"
    PROGRESSIVE_LATENCY = "progressive_latency"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    INTERMITTENT = "intermittent"
    UNAVAILABLE = "unavailable"


class ExperimentMode(StrEnum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class Scenario(BaseModel):
    service: Literal["primary", "backup"] = "primary"
    fault_type: FaultType
    duration_seconds: float = Field(default=10, gt=0, le=300)
    latency_ms: float = Field(default=0, ge=0, le=10_000)
    error_rate: float = Field(default=0, ge=0, le=1)
    status_code: Literal[500, 503] = 503
    seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def validate_fault_parameters(self) -> "Scenario":
        if self.fault_type in {FaultType.CONSTANT_LATENCY, FaultType.PROGRESSIVE_LATENCY, FaultType.TIMEOUT} and self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive for a latency or timeout fault")
        if self.fault_type in {FaultType.HTTP_ERROR, FaultType.INTERMITTENT} and self.error_rate <= 0:
            raise ValueError("error_rate must be positive for an error fault")
        return self


class ExperimentCreate(BaseModel):
    mode: ExperimentMode = ExperimentMode.AUTOMATIC
    scenario: Scenario
    replay_group_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    repetition: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_replay_metadata(self) -> "ExperimentCreate":
        if (self.replay_group_id is None) != (self.repetition is None):
            raise ValueError("replay_group_id and repetition must be provided together")
        return self


class ReplayPlanRequest(BaseModel):
    scenario: Scenario
    repetitions: int = Field(default=3, ge=1, le=20)
    group_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @model_validator(mode="after")
    def validate_seed_range(self) -> "ReplayPlanRequest":
        if self.scenario.seed + self.repetitions - 1 > 2_147_483_647:
            raise ValueError("scenario seed range exceeds the supported maximum")
        return self


class TelemetrySample(BaseModel):
    service: str
    timestamp: datetime
    latency_ms: float
    status_code: int | None
    timed_out: bool = False
    success: bool
    probe_kind: Literal["monitor", "traffic", "verification"] = "monitor"


class MetricsSnapshot(BaseModel):
    service: str
    sample_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    error_rate: float
    timeout_rate: float
    request_rate: float
    availability: float
    latency_ratio: float
    request_rate_ratio: float


class ResourceSnapshot(BaseModel):
    source_id: str
    timestamp: datetime
    cpu_percent: float = Field(ge=0, le=100)
    cpu_count: int = Field(ge=1)
    load_1m: float = Field(ge=0)
    load_5m: float = Field(ge=0)
    load_15m: float = Field(ge=0)
    load_per_cpu: float = Field(ge=0)
    memory_total_bytes: int = Field(ge=0)
    memory_available_bytes: int = Field(ge=0)
    memory_percent: float = Field(ge=0, le=100)
    disk_path: str
    disk_total_bytes: int = Field(ge=0)
    disk_free_bytes: int = Field(ge=0)
    disk_percent: float = Field(ge=0, le=100)
    network_rx_bytes: int = Field(ge=0)
    network_tx_bytes: int = Field(ge=0)
    network_rx_errors: int = Field(ge=0)
    network_tx_errors: int = Field(ge=0)
    process_count: int = Field(ge=0)
    containerized: bool
    cgroup_memory_current_bytes: int | None = Field(default=None, ge=0)
    cgroup_memory_limit_bytes: int | None = Field(default=None, ge=0)
    cgroup_memory_percent: float | None = Field(default=None, ge=0)
    cgroup_cpu_quota_cores: float | None = Field(default=None, gt=0)
    cgroup_cpu_percent: float | None = Field(default=None, ge=0, le=100)


class ModelScoreRequest(BaseModel):
    latency_ratio: float = Field(ge=0.001, le=100)
    error_rate: float = Field(ge=0, le=1)
    timeout_rate: float = Field(ge=0, le=1)
    request_rate_ratio: float = Field(ge=0.001, le=100)
    availability: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_consistent_rates(self) -> "ModelScoreRequest":
        if self.timeout_rate > self.error_rate:
            raise ValueError("timeout_rate cannot exceed error_rate")
        if abs(self.availability - (1 - self.error_rate)) > 0.001:
            raise ValueError("availability must equal 1 - error_rate")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class InvestigationRequest(BaseModel):
    question: str = Field(
        default="¿Qué ocurrió y cuál es el siguiente paso seguro?",
        min_length=3,
        max_length=500,
    )
    allow_external_ai: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return normalized


class InvestigationHypothesis(BaseModel):
    statement: str
    confidence: Literal["low", "medium", "high"]
    evidence_event_ids: list[int]


class InvestigationBriefing(BaseModel):
    incident_outcome: Literal[
        "recovered",
        "mitigation_blocked",
        "recovery_failed",
        "awaiting_approval",
        "under_investigation",
    ]
    action_status: Literal[
        "completed", "blocked", "failed", "interrupted", "running", "not_started"
    ]
    verification_status: Literal["verified", "failed", "pending"]
    root_cause_status: Literal["demonstrated", "not_demonstrated"]
    what_happened: str
    customer_impact: str
    sentinel_action: str
    verification: str
    next_safe_step: str
    unknowns: list[str] = Field(default_factory=list)


class InvestigationToolCall(BaseModel):
    name: str
    success: bool
    scope_valid: bool = True
    duration_ms: float = Field(ge=0)


class InvestigationTrace(BaseModel):
    mcp_transport: Literal["streamable-http"] | None = None
    mcp_used: bool = False
    discovered_tools: list[str] = Field(default_factory=list)
    tool_calls: list[InvestigationToolCall] = Field(default_factory=list)
    model: str | None = None
    ai_duration_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    failure_type: str | None = None


class InvestigationResponse(BaseModel):
    incident_id: str
    provider: str
    question: str = "¿Qué ocurrió y cuál es el siguiente paso seguro?"
    answer: str = ""
    answer_kind: Literal["deterministic", "ai-hypothesis"] = "deterministic"
    external_ai_requested: bool = False
    facts: list[str]
    hypotheses: list[str]
    hypothesis_evidence: list[InvestigationHypothesis] = Field(default_factory=list)
    evidence_event_ids: list[int] = Field(default_factory=list)
    suggested_runbook: str | None
    operator_briefing: InvestigationBriefing
    trace: InvestigationTrace = Field(default_factory=InvestigationTrace)
    guardrails: dict[str, bool] = Field(
        default_factory=lambda: {
            "read_only": True,
            "facts_are_deterministic": True,
            "citations_validated": True,
            "action_authority": False,
        }
    )
    raw: str | None = None


class IncidentResponse(BaseModel):
    id: str
    service: str
    severity: str
    status: str
    first_signal_at: datetime
    detected_at: datetime
    recovered_at: datetime | None
    evidence: dict[str, Any]
    experiment_id: str | None

    model_config = {"from_attributes": True}
