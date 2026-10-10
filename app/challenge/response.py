from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


RESPONSE_PLAN_VERSION = "challenge-kaiku-response-plan-v1"


@dataclass(frozen=True)
class VerificationCheck:
    signal: str
    condition: str
    stable_minutes: int


@dataclass(frozen=True)
class ResponsePlan:
    runbook_id: str
    title: str
    summary: str
    diagnostic_steps: tuple[str, ...]
    controlled_steps: tuple[str, ...]
    verification: tuple[VerificationCheck, ...]
    stop_conditions: tuple[str, ...]


_PLANS: dict[str, ResponsePlan] = {
    "checkout_degradation": ResponsePlan(
        runbook_id="kaiku.checkout.degradation.v1",
        title="Proteger el flujo de compra",
        summary=(
            "Revisar timeouts y respuestas 503 del checkout; validar el pool de "
            "conexiones y reducir carga antes de afectar nuevas compras."
        ),
        diagnostic_steps=(
            "Confirmar la tendencia de latencia, errores y eventos del checkout.",
            "Revisar saturación del pool y dependencias de pago e inventario.",
        ),
        controlled_steps=(
            "Reducir trabajo no crítico o limitar tráfico únicamente con aprobación.",
            "Preparar rollback si la degradación coincide con un cambio reciente.",
        ),
        verification=(
            VerificationCheck("latencia_ms", "regresa al baseline histórico", 5),
            VerificationCheck("errores", "no mantiene crecimiento sostenido", 5),
        ),
        stop_conditions=(
            "No existe una dependencia saludable a la cual desviar carga.",
            "La acción propuesta puede duplicar pedidos o cobros.",
        ),
    ),
    "database_connection_pressure": ResponsePlan(
        runbook_id="kaiku.database.connection-pressure.v1",
        title="Recuperar capacidad de conexiones",
        summary=(
            "Revisar conexiones activas, límites por cliente y consultas retenidas; "
            "proteger capacidad para el checkout."
        ),
        diagnostic_steps=(
            "Inspeccionar uso del pool, conexiones por servicio y consultas lentas.",
            "Separar tráfico de checkout de trabajos no críticos y tareas masivas.",
        ),
        controlled_steps=(
            "Pausar trabajadores no críticos solamente con aprobación.",
            "Ajustar límites o capacidad únicamente después de confirmar headroom.",
        ),
        verification=(
            VerificationCheck("db_pool", "deja de reportar saturación", 10),
            VerificationCheck("latencia_ms", "permanece cerca del baseline", 10),
        ),
        stop_conditions=(
            "No se identificó qué consumidor retiene conexiones.",
            "Aumentar el pool agotaría memoria o conexiones de la base de datos.",
        ),
    ),
    "database_contention": ResponsePlan(
        runbook_id="kaiku.database.contention.v1",
        title="Aislar contención de base de datos",
        summary=(
            "Revisar deadlocks y transacciones sobre pedidos e inventario; identificar "
            "la consulta contendiente antes de aplicar cambios."
        ),
        diagnostic_steps=(
            "Capturar las transacciones, tablas y consultas involucradas.",
            "Correlacionar la contención con jobs, despliegues y tráfico de compra.",
        ),
        controlled_steps=(
            "Pausar el job causante solamente si no afecta integridad de pedidos.",
            "Aplicar rollback o corrección de consulta mediante cambio aprobado.",
        ),
        verification=(
            VerificationCheck("deadlock", "no reaparece", 15),
            VerificationCheck("errores", "regresa al rango histórico", 15),
        ),
        stop_conditions=(
            "No se conoce el impacto de terminar una transacción.",
            "Existe riesgo de perder o duplicar pedidos.",
        ),
    ),
    "cpu_saturation": ResponsePlan(
        runbook_id="kaiku.compute.cpu-saturation.v1",
        title="Contener saturación de CPU",
        summary=(
            "Identificar procesos y tráfico que elevan CPU; limitar carga anómala o "
            "preparar capacidad adicional."
        ),
        diagnostic_steps=(
            "Identificar procesos, endpoints y jobs que concentran CPU.",
            "Descartar campañas y cargas programadas permitidas por el contexto.",
        ),
        controlled_steps=(
            "Limitar trabajo no crítico únicamente con aprobación.",
            "Escalar capacidad solo si hay margen y un procedimiento reversible.",
        ),
        verification=(
            VerificationCheck("cpu", "permanece debajo del baseline de presión", 10),
            VerificationCheck("latencia_ms", "se estabiliza", 10),
        ),
        stop_conditions=(
            "La carga elevada corresponde a un evento legítimo conocido.",
            "La reducción de capacidad afectaría checkout o pagos.",
        ),
    ),
    "memory_pressure": ResponsePlan(
        runbook_id="kaiku.compute.memory-pressure.v1",
        title="Contener presión de memoria",
        summary=(
            "Revisar consumo por proceso, colas y reinicios recientes; preparar una "
            "reducción segura de carga."
        ),
        diagnostic_steps=(
            "Identificar procesos, cachés y colas con crecimiento sostenido.",
            "Revisar reinicios, OOM y cambios recientes de configuración.",
        ),
        controlled_steps=(
            "Drenar carga antes de reiniciar un componente aprobado.",
            "Reducir concurrencia no crítica de forma reversible.",
        ),
        verification=(
            VerificationCheck("ram", "deja de crecer y recupera headroom", 10),
            VerificationCheck("latencia_ms", "permanece estable", 10),
        ),
        stop_conditions=(
            "El componente no puede drenarse sin perder transacciones.",
            "No hay capacidad saludable para absorber la carga.",
        ),
    ),
    "storage_capacity_risk": ResponsePlan(
        runbook_id="kaiku.storage.capacity-risk.v1",
        title="Recuperar margen de almacenamiento",
        summary=(
            "Revisar crecimiento de disco y mayores consumidores; liberar o ampliar "
            "capacidad antes del siguiente periodo crítico."
        ),
        diagnostic_steps=(
            "Identificar volúmenes, tablas, logs y artefactos con mayor crecimiento.",
            "Confirmar política de retención y respaldo antes de eliminar datos.",
        ),
        controlled_steps=(
            "Rotar o archivar únicamente datos cubiertos por política aprobada.",
            "Ampliar capacidad mediante un cambio reversible y auditado.",
        ),
        verification=(
            VerificationCheck("disco", "recupera margen operativo", 15),
            VerificationCheck("errores", "no aparecen fallos de escritura", 15),
        ),
        stop_conditions=(
            "No existe respaldo verificado de los datos candidatos.",
            "El volumen o tabla afectada no está identificado con certeza.",
        ),
    ),
    "application_errors": ResponsePlan(
        runbook_id="kaiku.application.error-burst.v1",
        title="Aislar incremento de errores",
        summary=(
            "Revisar errores recientes, dependencias y latencia; preparar reversión o "
            "aislamiento del componente afectado."
        ),
        diagnostic_steps=(
            "Correlacionar errores con endpoint, dependencia y despliegue reciente.",
            "Confirmar si checkout, pagos o inventario están afectados.",
        ),
        controlled_steps=(
            "Ejecutar rollback aprobado si existe una versión saludable conocida.",
            "Aislar la dependencia solamente si hay una ruta degradada segura.",
        ),
        verification=(
            VerificationCheck("errores", "regresa al rango histórico", 10),
            VerificationCheck("latencia_ms", "permanece estable", 10),
        ),
        stop_conditions=(
            "No existe una versión saludable o rollback probado.",
            "El aislamiento puede perder pedidos o pagos.",
        ),
    ),
    "unclassified": ResponsePlan(
        runbook_id="kaiku.triage.unclassified-risk.v1",
        title="Clasificar riesgo no identificado",
        summary=(
            "Revisar las tres señales principales y confirmar el servicio afectado "
            "antes de actuar."
        ),
        diagnostic_steps=(
            "Revisar evidencia causal, contexto de negocio y calidad de telemetría.",
            "Identificar el servicio y recurso con cambio sostenido.",
        ),
        controlled_steps=(
            "Escalar al operador; no ejecutar una mitigación sin causa operativa.",
        ),
        verification=(
            VerificationCheck("risk_score", "desciende bajo el umbral de recuperación", 10),
        ),
        stop_conditions=(
            "La causa y el alcance siguen sin identificarse.",
        ),
    ),
    "data_quality_alert": ResponsePlan(
        runbook_id="kaiku.telemetry.restore-visibility.v1",
        title="Restablecer visibilidad",
        summary=(
            "Restablecer la telemetría y validar el agente; no tratar el apagón como "
            "incidente del negocio sin evidencia adicional."
        ),
        diagnostic_steps=(
            "Revisar agente, transporte y permisos de lectura de métricas.",
            "Confirmar si el servicio funciona mediante una fuente independiente.",
        ),
        controlled_steps=(
            "Reiniciar el agente únicamente con aprobación y sin tocar la aplicación.",
        ),
        verification=(
            VerificationCheck("data_quality", "permanece en ok o partial", 5),
        ),
        stop_conditions=(
            "No existe una fuente independiente para confirmar el servicio.",
        ),
    ),
}


def response_plan_for(
    alert_type: str,
    suspected_incident_type: str | None,
) -> dict[str, Any]:
    requested_key = (
        "data_quality_alert"
        if alert_type == "data_quality_alert"
        else suspected_incident_type or "unclassified"
    )
    matched_key = requested_key if requested_key in _PLANS else "unclassified"
    plan = _PLANS[matched_key]
    payload = {
        "runbook_id": plan.runbook_id,
        "title": plan.title,
        "summary": plan.summary,
        "diagnostic_steps": list(plan.diagnostic_steps),
        "controlled_steps": list(plan.controlled_steps),
        "verification": [asdict(check) for check in plan.verification],
        "stop_conditions": list(plan.stop_conditions),
    }
    payload.update(
        {
            "version": RESPONSE_PLAN_VERSION,
            "requested_incident_type": requested_key,
            "matched_incident_type": matched_key,
            "execution_mode": "recommendation_only",
            "execution_authorized": False,
            "approval_required": True,
            "verification_status": "not_started",
            "environment": "official_synthetic_challenge_replay",
        }
    )
    return payload


def response_catalog() -> dict[str, Any]:
    return {
        key: response_plan_for(
            "data_quality_alert" if key == "data_quality_alert" else "incident_risk",
            None if key in {"data_quality_alert", "unclassified"} else key,
        )
        for key in sorted(_PLANS)
    }


def build_response_audit(alerts: pd.DataFrame) -> dict[str, Any]:
    required = {
        "alert_type",
        "suspected_incident_type",
        "decision_latency_minutes",
    }
    missing = sorted(required - set(alerts.columns))
    if missing:
        raise ValueError(f"response audit is missing columns: {missing}")

    plans = [
        response_plan_for(str(row.alert_type), row.suspected_incident_type)
        for row in alerts.itertuples(index=False)
    ]
    runbooks = Counter(str(plan["runbook_id"]) for plan in plans)
    def latency_distribution(values: pd.Series) -> dict[str, float | int | None]:
        latency = pd.to_numeric(values, errors="coerce").dropna()
        return {
            "count": int(len(latency)),
            "min": float(latency.min()) if not latency.empty else None,
            "median": float(latency.median()) if not latency.empty else None,
            "p95": float(latency.quantile(0.95)) if not latency.empty else None,
            "max": float(latency.max()) if not latency.empty else None,
        }

    latency = latency_distribution(alerts["decision_latency_minutes"])
    latency_by_type = {
        str(alert_type): latency_distribution(group["decision_latency_minutes"])
        for alert_type, group in alerts.groupby("alert_type", sort=True)
    }
    all_mapped = len(plans) == len(alerts) and all(plan["runbook_id"] for plan in plans)
    execution_blocked = all(plan["execution_authorized"] is False for plan in plans)
    verification_complete = all(bool(plan["verification"]) for plan in plans)
    return {
        "response_plan_version": RESPONSE_PLAN_VERSION,
        "scope": "Kaiku synthetic replay; recommendation and verification contract only",
        "alerts_evaluated": int(len(alerts)),
        "runbook_distribution": dict(sorted(runbooks.items())),
        "signal_to_recommendation_minutes": latency,
        "signal_to_recommendation_by_alert_type": latency_by_type,
        "gates": {
            "all_alerts_have_response_plan": all_mapped,
            "automatic_execution_blocked": execution_blocked,
            "all_plans_have_verification_contract": verification_complete,
        },
        "passed": all_mapped and execution_blocked and verification_complete,
        "limitations": {
            "action_executed": False,
            "recovery_verified": False,
            "reason": (
                "The official kit is a synthetic replay and exposes no controllable "
                "Kaiku infrastructure."
            ),
        },
    }
