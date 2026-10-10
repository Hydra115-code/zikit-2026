from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class IncidentRecord(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    service: Mapped[str] = mapped_column(String(100), index=True)
    severity: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30), index=True)
    first_signal_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recovered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), index=True, nullable=True
    )
    events: Mapped[list["EventRecord"]] = relationship(back_populates="incident", cascade="all, delete-orphan")


class EventRecord(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str | None] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), index=True, nullable=True
    )
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), index=True, nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    process_id: Mapped[str] = mapped_column(String(36), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    monotonic_seconds: Mapped[float] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    incident: Mapped[IncidentRecord | None] = relationship(back_populates="events")


class AlertRecord(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), unique=True, index=True
    )
    state: Mapped[str] = mapped_column(String(20), default="open", index=True)
    priority: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    event: Mapped[EventRecord] = relationship()


class ChallengePredictionRecord(Base):
    __tablename__ = "challenge_predictions"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "tenant",
            "alert_timestamp",
            "alert_type",
            "model_version",
            "policy_version",
            name="uq_challenge_prediction_identity",
        ),
        CheckConstraint("tenant = 'kaiku'", name="ck_challenge_tenant_kaiku"),
        CheckConstraint(
            "risk_score >= 0 and risk_score <= 1",
            name="ck_challenge_risk_score_range",
        ),
        CheckConstraint(
            "alert_timestamp >= signal_timestamp",
            name="ck_challenge_timestamp_order",
        ),
        CheckConstraint(
            "(alert_type = 'incident_risk' and incident_id is not null) or "
            "(alert_type = 'data_quality_alert' and incident_id is null)",
            name="ck_challenge_incident_kind",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant: Mapped[str] = mapped_column(String(30), index=True)
    signal_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    alert_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    alert_type: Mapped[str] = mapped_column(String(30), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    suspected_incident_type: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    model_version: Mapped[str] = mapped_column(String(100))
    policy_version: Mapped[str] = mapped_column(String(100))
    model_sha256: Mapped[str] = mapped_column(String(64))
    policy_sha256: Mapped[str] = mapped_column(String(64))
    data_quality: Mapped[str] = mapped_column(String(30))
    context_known_event: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence: Mapped[object] = mapped_column(JSON)
    recommended_action: Mapped[str] = mapped_column(Text)
    incident_id: Mapped[str | None] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), index=True, nullable=True
    )
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )


class ExperimentRecord(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(30), index=True)
    scenario: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class InvestigationRecord(Base):
    __tablename__ = "investigations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    provider: Mapped[str] = mapped_column(String(30))
    diagnosis: Mapped[str] = mapped_column(Text)


class ServiceRecord(Base):
    __tablename__ = "services"
    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class MetricRecord(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    service: Mapped[str] = mapped_column(ForeignKey("services.name", ondelete="CASCADE"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timed_out: Mapped[bool] = mapped_column(Boolean, default=False)
    success: Mapped[bool] = mapped_column(Boolean, index=True)


class ResourceMetricRecord(Base):
    __tablename__ = "resource_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cpu_percent: Mapped[float] = mapped_column(Float)
    cpu_count: Mapped[int] = mapped_column(Integer)
    load_1m: Mapped[float] = mapped_column(Float)
    load_5m: Mapped[float] = mapped_column(Float)
    load_15m: Mapped[float] = mapped_column(Float)
    load_per_cpu: Mapped[float] = mapped_column(Float)
    memory_total_bytes: Mapped[int] = mapped_column(Integer)
    memory_available_bytes: Mapped[int] = mapped_column(Integer)
    memory_percent: Mapped[float] = mapped_column(Float)
    disk_path: Mapped[str] = mapped_column(String(500))
    disk_total_bytes: Mapped[int] = mapped_column(Integer)
    disk_free_bytes: Mapped[int] = mapped_column(Integer)
    disk_percent: Mapped[float] = mapped_column(Float)
    network_rx_bytes: Mapped[int] = mapped_column(Integer)
    network_tx_bytes: Mapped[int] = mapped_column(Integer)
    network_rx_errors: Mapped[int] = mapped_column(Integer)
    network_tx_errors: Mapped[int] = mapped_column(Integer)
    process_count: Mapped[int] = mapped_column(Integer)
    containerized: Mapped[bool] = mapped_column(Boolean, default=False)
    cgroup_memory_current_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cgroup_memory_limit_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cgroup_memory_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    cgroup_cpu_quota_cores: Mapped[float | None] = mapped_column(Float, nullable=True)
    cgroup_cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomalous: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reasons: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class ActionRecord(Base):
    __tablename__ = "actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    runbook: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(30), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url.replace("sqlite+aiosqlite://", "sqlite://").replace(
        "postgresql+asyncpg://", "postgresql+psycopg://"
    )


def configure_database(url: str | None = None) -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    database_url = normalize_database_url(url or get_settings().database_url)
    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        path = database_url.rsplit("///", 1)[-1]
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        connect_args["check_same_thread"] = False
    engine_options: dict[str, Any] = {"pool_pre_ping": True, "connect_args": connect_args}
    if database_url.startswith("sqlite"):
        engine_options["poolclass"] = NullPool
    else:
        parsed_url = make_url(database_url)
        engine_options.update({"pool_size": 5, "max_overflow": 5, "pool_recycle": 300})
        if parsed_url.port == 6543:
            connect_args["prepare_threshold"] = None
    _engine = create_engine(database_url, **engine_options)
    _session_factory = sessionmaker(_engine, expire_on_commit=False)


def engine() -> Engine:
    if _engine is None:
        configure_database()
    assert _engine is not None
    return _engine


def session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        configure_database()
    assert _session_factory is not None
    return _session_factory


async def init_database() -> None:
    Base.metadata.create_all(engine())


async def get_session() -> AsyncIterator[Session]:
    with session_factory()() as session:
        yield session
