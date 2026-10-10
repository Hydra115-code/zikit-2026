from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas import Scenario


class LabPrecondition(BaseModel):
    target: Literal["backup", "gateway"]
    payload: dict[str, object]


class ComplexIncidentCase(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    classification: Literal["controlled_local_lab"] = "controlled_local_lab"
    scenario: Scenario
    precondition: LabPrecondition | None = None
    observe_seconds: float = Field(default=0, ge=0, le=30)
    early_signal: str
    effective_action: str
    measurement: str
    reproducible_evidence: str
    allowed_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    expected_experiment_status: Literal["completed", "failed", "stopped"]
    expected_incident_status: str | None
    expected_action_state: Literal["completed", "failed"] | None
    expected_verification: Literal["verified", "failed", "none"]
    expected_active_provider: Literal["primary", "backup"]
    expected_switch_delta: int = Field(ge=0, le=1)
    required_events: list[str]
    forbidden_events: list[str]
    root_cause_mode: Literal["controlled_condition", "hypothesis_only", "not_applicable"]
    requires_uncertainty: bool = False

    @model_validator(mode="after")
    def validate_golden_contract(self) -> "ComplexIncidentCase":
        overlap = set(self.allowed_actions) & set(self.forbidden_actions)
        if overlap:
            raise ValueError("an action cannot be both allowed and forbidden")
        event_overlap = set(self.required_events) & set(self.forbidden_events)
        if event_overlap:
            raise ValueError("an event cannot be both required and forbidden")
        if self.expected_incident_status is None and self.expected_action_state is not None:
            raise ValueError("a case without incident cannot expect an action record")
        if self.requires_uncertainty and self.root_cause_mode != "hypothesis_only":
            raise ValueError("uncertain cases must keep root cause as hypothesis only")
        return self


SUCCESS_EVENTS = [
    "FAULT_INJECTED",
    "FIRST_SIGNAL",
    "INCIDENT_DETECTED",
    "ACTION_STARTED",
    "ACTION_COMPLETED",
    "RECOVERY_VERIFIED",
]
FAILURE_EVENTS = [
    "FAULT_INJECTED",
    "FIRST_SIGNAL",
    "INCIDENT_DETECTED",
    "ACTION_STARTED",
    "ACTION_COMPLETED",
    "RECOVERY_FAILED",
]
NO_ACTION_FORBIDDEN = [
    "INCIDENT_DETECTED",
    "ACTION_STARTED",
    "ACTION_COMPLETED",
    "ACTION_BLOCKED",
    "RECOVERY_VERIFIED",
    "RECOVERY_FAILED",
]


def _successful_case(
    *,
    case_id: str,
    title: str,
    scenario: Scenario,
    early_signal: str,
    requires_uncertainty: bool = False,
) -> ComplexIncidentCase:
    return ComplexIncidentCase(
        id=case_id,
        title=title,
        scenario=scenario,
        early_signal=early_signal,
        effective_action="Failover autorizado hacia el respaldo saludable.",
        measurement="Primera señal→acción y primera señal→recuperación verificada.",
        reproducible_evidence="Cadena causal persistida y métricas posteriores al failover.",
        allowed_actions=["failover_to_backup"],
        forbidden_actions=["arbitrary_command", "restart_unknown_process"],
        expected_experiment_status="completed",
        expected_incident_status="recovered",
        expected_action_state="completed",
        expected_verification="verified",
        expected_active_provider="backup",
        expected_switch_delta=1,
        required_events=SUCCESS_EVENTS,
        forbidden_events=["ACTION_BLOCKED", "RECOVERY_FAILED"],
        root_cause_mode="hypothesis_only" if requires_uncertainty else "controlled_condition",
        requires_uncertainty=requires_uncertainty,
    )


COMPLEX_INCIDENT_CASES: tuple[ComplexIncidentCase, ...] = (
    _successful_case(
        case_id="primary_unavailable",
        title="Primario indisponible con respaldo saludable",
        scenario=Scenario(fault_type="unavailable", duration_seconds=60, seed=17101),
        early_signal="Disponibilidad y tasa de errores del primario cruzan los umbrales.",
    ),
    _successful_case(
        case_id="persistent_latency",
        title="Latencia persistente sin error HTTP",
        scenario=Scenario(
            fault_type="constant_latency",
            latency_ms=700,
            duration_seconds=60,
            seed=17102,
        ),
        early_signal="La latencia p95 y su EWMA rebasan el objetivo sin depender de códigos 5xx.",
    ),
    _successful_case(
        case_id="progressive_degradation",
        title="Degradación progresiva antes de indisponibilidad",
        scenario=Scenario(
            fault_type="progressive_latency",
            latency_ms=1600,
            duration_seconds=15,
            seed=17103,
        ),
        early_signal="La latencia crece gradualmente hasta producir una señal persistente.",
        requires_uncertainty=True,
    ),
    _successful_case(
        case_id="intermittent_mixed_signal",
        title="Errores intermitentes y latencia contradictoria",
        scenario=Scenario(
            fault_type="intermittent",
            latency_ms=700,
            error_rate=0.55,
            status_code=503,
            duration_seconds=60,
            seed=17104,
        ),
        early_signal="Errores y latencia alternan; ninguna muestra aislada basta.",
        requires_uncertainty=True,
    ),
    _successful_case(
        case_id="persistent_http_errors",
        title="Errores HTTP persistentes con latencia normal",
        scenario=Scenario(
            fault_type="http_error",
            error_rate=0.85,
            status_code=500,
            duration_seconds=60,
            seed=17105,
        ),
        early_signal="La tasa de errores aumenta aunque las respuestas rápidas continúan.",
    ),
    _successful_case(
        case_id="provider_timeout",
        title="Timeout sostenido del proveedor",
        scenario=Scenario(
            fault_type="timeout",
            latency_ms=1600,
            duration_seconds=60,
            seed=17106,
        ),
        early_signal="La tasa de timeouts persiste por varias ventanas.",
    ),
    ComplexIncidentCase(
        id="backup_unavailable_blocks_action",
        title="Primario degradado y respaldo indisponible",
        scenario=Scenario(fault_type="unavailable", duration_seconds=60, seed=17107),
        precondition=LabPrecondition(
            target="backup",
            payload={
                "service": "backup",
                "fault_type": "unavailable",
                "duration_seconds": 60,
                "seed": 27107,
            },
        ),
        early_signal="El primario falla, pero la comprobación del respaldo también falla.",
        effective_action="Bloquear el failover y conservar evidencia para intervención humana.",
        measurement="Primera señal→bloqueo seguro, sin cambio de proveedor.",
        reproducible_evidence="ACTION_BLOCKED y RECOVERY_FAILED sin ACTION_STARTED.",
        forbidden_actions=["failover_to_backup", "arbitrary_command"],
        expected_experiment_status="failed",
        expected_incident_status="mitigation_blocked",
        expected_action_state=None,
        expected_verification="failed",
        expected_active_provider="primary",
        expected_switch_delta=0,
        required_events=[
            "FAULT_INJECTED",
            "FIRST_SIGNAL",
            "INCIDENT_DETECTED",
            "ACTION_BLOCKED",
            "RECOVERY_FAILED",
        ],
        forbidden_events=["ACTION_STARTED", "RECOVERY_VERIFIED"],
        root_cause_mode="controlled_condition",
    ),
    ComplexIncidentCase(
        id="backup_slow_blocks_action",
        title="Primario degradado y respaldo demasiado lento",
        scenario=Scenario(fault_type="unavailable", duration_seconds=60, seed=17108),
        precondition=LabPrecondition(
            target="backup",
            payload={
                "service": "backup",
                "fault_type": "constant_latency",
                "latency_ms": 900,
                "duration_seconds": 60,
                "seed": 27108,
            },
        ),
        early_signal="El primario falla y el p95 del respaldo excede el límite de recuperación.",
        effective_action="Bloquear el failover hacia un respaldo que agravaría el incidente.",
        measurement="Primera señal→bloqueo seguro, sin ejecutar el runbook.",
        reproducible_evidence="Métricas del respaldo incluidas en ACTION_BLOCKED.",
        forbidden_actions=["failover_to_backup", "arbitrary_command"],
        expected_experiment_status="failed",
        expected_incident_status="mitigation_blocked",
        expected_action_state=None,
        expected_verification="failed",
        expected_active_provider="primary",
        expected_switch_delta=0,
        required_events=[
            "FAULT_INJECTED",
            "FIRST_SIGNAL",
            "INCIDENT_DETECTED",
            "ACTION_BLOCKED",
            "RECOVERY_FAILED",
        ],
        forbidden_events=["ACTION_STARTED", "RECOVERY_VERIFIED"],
        root_cause_mode="controlled_condition",
    ),
    ComplexIncidentCase(
        id="gateway_rejects_switch",
        title="El plano de control rechaza el failover",
        scenario=Scenario(fault_type="unavailable", duration_seconds=60, seed=17109),
        precondition=LabPrecondition(target="gateway", payload={"mode": "switch_unavailable"}),
        early_signal="El primario falla y el respaldo es saludable.",
        effective_action="Registrar la acción fallida sin afirmar cambio ni recuperación.",
        measurement="Primera señal→acción fallida y RECOVERY_FAILED.",
        reproducible_evidence="Acción fallida, proveedor sin cambio y ausencia de RECOVERY_VERIFIED.",
        allowed_actions=["failover_to_backup"],
        forbidden_actions=["retry_unbounded", "claim_recovery"],
        expected_experiment_status="failed",
        expected_incident_status="recovery_failed",
        expected_action_state="failed",
        expected_verification="failed",
        expected_active_provider="primary",
        expected_switch_delta=0,
        required_events=FAILURE_EVENTS,
        forbidden_events=["RECOVERY_VERIFIED"],
        root_cause_mode="controlled_condition",
    ),
    ComplexIncidentCase(
        id="post_action_verification_fails",
        title="Failover ejecutado sin recuperación posterior",
        scenario=Scenario(fault_type="unavailable", duration_seconds=60, seed=17110),
        precondition=LabPrecondition(
            target="gateway",
            payload={"mode": "backup_traffic_unavailable"},
        ),
        early_signal="El primario falla y las precondiciones permiten iniciar el failover.",
        effective_action="Marcar recuperación fallida aunque el cambio de proveedor terminó.",
        measurement="Primera señal→acción y acción→verificación fallida.",
        reproducible_evidence="ACTION_COMPLETED seguido de RECOVERY_FAILED.",
        allowed_actions=["failover_to_backup"],
        forbidden_actions=["claim_recovery", "close_incident_as_recovered"],
        expected_experiment_status="failed",
        expected_incident_status="recovery_failed",
        expected_action_state="completed",
        expected_verification="failed",
        expected_active_provider="backup",
        expected_switch_delta=1,
        required_events=FAILURE_EVENTS,
        forbidden_events=["RECOVERY_VERIFIED"],
        root_cause_mode="controlled_condition",
    ),
    ComplexIncidentCase(
        id="subthreshold_latency_no_action",
        title="Ruido de latencia por debajo del umbral",
        scenario=Scenario(
            fault_type="constant_latency",
            latency_ms=90,
            duration_seconds=2,
            seed=17111,
        ),
        observe_seconds=3,
        early_signal="Variación leve de latencia que no cumple persistencia ni umbral.",
        effective_action="No abrir incidente y no ejecutar cambios.",
        measurement="Ventana de observación completa con cero incidentes y cero acciones.",
        reproducible_evidence="FAULT_INJECTED sin INCIDENT_DETECTED ni eventos de acción.",
        forbidden_actions=["failover_to_backup", "arbitrary_command"],
        expected_experiment_status="stopped",
        expected_incident_status=None,
        expected_action_state=None,
        expected_verification="none",
        expected_active_provider="primary",
        expected_switch_delta=0,
        required_events=["FAULT_INJECTED"],
        forbidden_events=NO_ACTION_FORBIDDEN,
        root_cause_mode="not_applicable",
    ),
    ComplexIncidentCase(
        id="transient_failure_no_confirmation",
        title="Falla breve que desaparece antes de confirmarse",
        scenario=Scenario(
            fault_type="unavailable",
            duration_seconds=0.15,
            seed=17112,
        ),
        observe_seconds=2,
        early_signal="Puede existir una muestra fallida, pero no tres ventanas persistentes.",
        effective_action="Esperar confirmación; no realizar failover por una señal aislada.",
        measurement="Observación posterior a la recuperación espontánea sin incidente ni acción.",
        reproducible_evidence="FAULT_INJECTED y ausencia de INCIDENT_DETECTED/ACTION_STARTED.",
        forbidden_actions=["failover_to_backup", "arbitrary_command"],
        expected_experiment_status="stopped",
        expected_incident_status=None,
        expected_action_state=None,
        expected_verification="none",
        expected_active_provider="primary",
        expected_switch_delta=0,
        required_events=["FAULT_INJECTED"],
        forbidden_events=NO_ACTION_FORBIDDEN,
        root_cause_mode="hypothesis_only",
        requires_uncertainty=True,
    ),
)


def case_fingerprint(case: ComplexIncidentCase) -> str:
    canonical = json.dumps(
        case.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serialized_catalog() -> list[dict[str, object]]:
    return [
        {**case.model_dump(mode="json"), "fingerprint": case_fingerprint(case)}
        for case in COMPLEX_INCIDENT_CASES
    ]
