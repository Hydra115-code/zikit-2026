from __future__ import annotations

CHALLENGE_SCHEMA_VERSION = "challenge-kaiku-ingestion-v2"

TARGET_TENANT = "kaiku"
TENANTS = (TARGET_TENANT,)
TENANT_SET = frozenset(TENANTS)

IDENTITY_COLUMNS = ("tenant", "timestamp")
METRIC_COLUMNS = ("cpu", "ram", "disco", "latencia_ms", "errores")
PUBLIC_METRIC_COLUMNS = ("timestamp", *METRIC_COLUMNS)
REQUIRED_COLUMNS = PUBLIC_METRIC_COLUMNS
PUBLIC_RUN_FILENAMES = ("contexto.md", "metricas.csv", "logs.md")
RESERVED_LABEL_COLUMNS = frozenset({"inicio_incidente", "tipo", "es_real"})

METRIC_LIMITS: dict[str, tuple[float, float | None]] = {
    "cpu": (0.0, 100.0),
    "ram": (0.0, 100.0),
    "disco": (0.0, 100.0),
    "latencia_ms": (0.0, None),
    "errores": (0.0, None),
}

CLEAN_COLUMNS = (
    "run_id",
    "tenant",
    "timestamp",
    *METRIC_COLUMNS,
    *(f"missing_{name}" for name in METRIC_COLUMNS),
    *(f"invalid_{name}" for name in METRIC_COLUMNS),
    "data_quality",
    "source_name",
    "source_line",
)

QUARANTINE_COLUMNS = (
    "run_id",
    "source_name",
    "source_line",
    "reason",
    "tenant",
    "timestamp",
    "raw_record",
)

CLEANING_EVENT_COLUMNS = (
    "run_id",
    "source_name",
    "source_line",
    "field",
    "action",
    "reason",
    "original_value",
    "clean_value",
)
