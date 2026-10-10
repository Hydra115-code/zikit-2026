from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Zikit Sentinel"
    environment: str = "development"
    database_url: str = "sqlite:///./data/sentinel.db"
    admin_token: str = "development-only-token"
    cors_origins: list[str] = ["http://localhost:3000"]
    monitor_enabled: bool = True
    monitor_interval_seconds: float = Field(default=0.25, ge=0.05)
    traffic_enabled: bool = True
    traffic_interval_seconds: float = Field(default=0.10, ge=0.01)
    event_stream_poll_interval_seconds: float = Field(default=0.25, ge=0.05, le=10)
    event_stream_heartbeat_seconds: float = Field(default=10, ge=1, le=60)
    event_stream_batch_size: int = Field(default=100, ge=1, le=1000)
    event_stream_max_replay_events: int = Field(default=5000, ge=1, le=100_000)
    event_stream_max_connections: int = Field(default=25, ge=1, le=1000)
    event_stream_retry_milliseconds: int = Field(default=1000, ge=250, le=60_000)
    resource_monitor_enabled: bool = True
    resource_monitor_interval_seconds: float = Field(default=1.0, ge=0.25, le=300)
    resource_source_id: str | None = Field(default=None, min_length=1, max_length=100)
    resource_history_size: int = Field(default=300, ge=10, le=10_000)
    resource_retention_hours: int = Field(default=72, ge=1, le=24 * 365)
    resource_cleanup_every_samples: int = Field(default=300, ge=1, le=100_000)
    resource_disk_path: Path = Path("/")
    resource_cpu_threshold_percent: float = Field(default=90, gt=0, le=100)
    resource_memory_threshold_percent: float = Field(default=90, gt=0, le=100)
    resource_disk_threshold_percent: float = Field(default=90, gt=0, le=100)
    resource_cgroup_memory_threshold_percent: float = Field(default=90, gt=0, le=100)
    resource_load_per_cpu_threshold: float = Field(default=1.5, gt=0, le=100)
    resource_anomaly_persistence_windows: int = Field(default=3, ge=1, le=100)
    resource_recovery_persistence_windows: int = Field(default=3, ge=1, le=100)
    probe_timeout_seconds: float = Field(default=1.0, gt=0, le=30)
    primary_provider_url: str | None = None
    backup_provider_url: str | None = None
    gateway_url: str | None = None
    gateway_control_url: str | None = None
    gateway_control_token: str | None = None
    primary_provider_control_url: str | None = None
    backup_provider_control_url: str | None = None
    provider_control_token: str | None = None
    telemetry_window_size: int = Field(default=50, ge=3)
    anomaly_persistence_windows: int = Field(default=3, ge=1)
    latency_threshold_ms: float = Field(default=400, gt=0)
    error_rate_threshold: float = Field(default=0.25, ge=0, le=1)
    timeout_rate_threshold: float = Field(default=0.20, ge=0, le=1)
    ml_action_min_latency_ratio: float = Field(default=1.75, gt=1)
    ml_action_min_error_rate: float = Field(default=0.05, ge=0, le=1)
    ml_action_max_request_rate_ratio: float = Field(default=0.5, gt=0, lt=1)
    slo_min_samples: int = Field(default=5, ge=1)
    slo_latency_p95_ms: float = Field(default=500, gt=0)
    slo_error_rate: float = Field(default=0.20, gt=0, le=1)
    slo_at_risk_ratio: float = Field(default=0.70, gt=0, lt=1)
    slo_breach_persistence_windows: int = Field(default=2, ge=1)
    customer_impact_persistence_windows: int = Field(default=3, ge=1)
    customer_complaint_delay_seconds: float = Field(default=2.0, ge=0, le=300)
    backup_min_samples: int = Field(default=3, ge=1, le=100)
    backup_max_sample_age_seconds: float = Field(default=2.0, gt=0, le=300)
    backup_min_availability: float = Field(default=0.95, gt=0, le=1)
    runbook_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    recovery_verification_requests: int = Field(default=5, ge=2, le=100)
    recovery_verification_interval_seconds: float = Field(default=0.05, ge=0, le=10)
    recovery_required_consecutive_successes: int = Field(default=3, ge=1, le=100)
    recovery_max_error_rate: float = Field(default=0.1, ge=0, le=1)
    recovery_max_p95_ms: float = Field(default=500, gt=0)
    auto_mitigation_enabled: bool = True
    mitigation_cooldown_seconds: float = Field(default=10, ge=0)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = Field(default=60, gt=0, le=120)
    gemini_max_tool_calls: int = Field(default=7, ge=1, le=20)
    gemini_max_attempts: int = Field(default=3, ge=1, le=5)
    gemini_retry_base_seconds: float = Field(default=0.75, ge=0, le=10)
    mcp_http_enabled: bool = True
    mcp_read_token: str | None = Field(default=None, min_length=12)
    mcp_internal_url: str = "http://127.0.0.1:8000/mcp/"
    mcp_client_timeout_seconds: float = Field(default=10, gt=0, le=60)
    evidence_api_enabled: bool = True
    evidence_bundle_path: Path = Path("reports/lab-phase-19-evidence-bundle.json")
    evidence_checksum_path: Path | None = None
    evidence_max_bundle_bytes: int = Field(default=5_000_000, ge=1_000, le=100_000_000)
    model_path: Path = Path("models/isolation_forest.joblib")
    challenge_features_path: Path = Path("data/challenge-phase-3-logs")
    challenge_model_path: Path = Path("models/challenge_early_warning.joblib")
    challenge_policy_path: Path = Path("models/challenge_alert_policy.json")
    challenge_showcase_path: Path = Path("data/challenge-phase-9/showcase.json")
    challenge_rehearsal_path: Path = Path(
        "data/challenge-phase-10/rehearsal-report.json"
    )
    challenge_runtime_path: Path | None = None
    challenge_require_all_tenants: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "primary_provider_url",
        "backup_provider_url",
        "gateway_url",
        "gateway_control_url",
        "gateway_control_token",
        "primary_provider_control_url",
        "backup_provider_control_url",
        "provider_control_token",
        "mcp_read_token",
        mode="before",
    )
    @classmethod
    def empty_probe_url_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.event_stream_max_replay_events < self.event_stream_batch_size:
            raise ValueError(
                "EVENT_STREAM_MAX_REPLAY_EVENTS must be greater than or equal to "
                "EVENT_STREAM_BATCH_SIZE"
            )
        if self.customer_impact_persistence_windows < self.slo_breach_persistence_windows:
            raise ValueError(
                "CUSTOMER_IMPACT_PERSISTENCE_WINDOWS must be greater than or equal to "
                "SLO_BREACH_PERSISTENCE_WINDOWS"
            )
        if self.recovery_required_consecutive_successes > self.recovery_verification_requests:
            raise ValueError(
                "RECOVERY_REQUIRED_CONSECUTIVE_SUCCESSES cannot exceed "
                "RECOVERY_VERIFICATION_REQUESTS"
            )
        probe_urls = [self.primary_provider_url, self.backup_provider_url, self.gateway_url]
        if any(probe_urls) and not all(probe_urls):
            raise ValueError(
                "PRIMARY_PROVIDER_URL, BACKUP_PROVIDER_URL and GATEWAY_URL must be configured together"
            )
        if (self.gateway_control_url is None) != (self.gateway_control_token is None):
            raise ValueError(
                "GATEWAY_CONTROL_URL and GATEWAY_CONTROL_TOKEN must be configured together"
            )
        if self.gateway_control_url is not None and not all(probe_urls):
            raise ValueError("gateway control requires all external probe URLs")
        if self.gateway_control_token is not None and len(self.gateway_control_token) < 12:
            raise ValueError("GATEWAY_CONTROL_TOKEN must contain at least 12 characters")
        provider_control = [
            self.primary_provider_control_url,
            self.backup_provider_control_url,
            self.provider_control_token,
        ]
        if any(provider_control) and not all(provider_control):
            raise ValueError(
                "PRIMARY_PROVIDER_CONTROL_URL, BACKUP_PROVIDER_CONTROL_URL and "
                "PROVIDER_CONTROL_TOKEN must be configured together"
            )
        if any(provider_control) and (not all(probe_urls) or self.gateway_control_url is None):
            raise ValueError(
                "provider control requires external probes and gateway control"
            )
        if self.provider_control_token is not None and len(self.provider_control_token) < 12:
            raise ValueError("PROVIDER_CONTROL_TOKEN must contain at least 12 characters")
        for value in [*probe_urls, self.gateway_control_url]:
            if value is None:
                continue
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("provider probe URLs must be absolute HTTP(S) URLs")
            if parsed.username or parsed.password:
                raise ValueError("provider probe URLs must not contain credentials")
        mcp_url = urlsplit(self.mcp_internal_url)
        if (
            mcp_url.scheme not in {"http", "https"}
            or not mcp_url.netloc
            or mcp_url.username
            or mcp_url.password
            or mcp_url.query
            or mcp_url.fragment
        ):
            raise ValueError(
                "MCP_INTERNAL_URL must be an absolute HTTP(S) URL without credentials, "
                "query or fragment"
            )
        for value in [self.primary_provider_control_url, self.backup_provider_control_url]:
            if value is None:
                continue
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "provider control URLs must be HTTP(S) origins without credentials, "
                    "path, query or fragment"
                )
        if self.environment.lower() != "production":
            return self
        problems: list[str] = []
        if self.database_url.startswith("sqlite"):
            problems.append("DATABASE_URL must use PostgreSQL in production")
        if self.admin_token == "development-only-token" or len(self.admin_token) < 32:
            problems.append("ADMIN_TOKEN must be a unique secret of at least 32 characters")
        if self.gateway_control_token is not None and len(self.gateway_control_token) < 32:
            problems.append("GATEWAY_CONTROL_TOKEN must contain at least 32 characters in production")
        if self.provider_control_token is not None and len(self.provider_control_token) < 32:
            problems.append("PROVIDER_CONTROL_TOKEN must contain at least 32 characters in production")
        if (self.mcp_http_enabled or self.evidence_api_enabled) and (
            self.mcp_read_token is None or len(self.mcp_read_token) < 32
        ):
            problems.append(
                "MCP_READ_TOKEN must contain at least 32 characters when MCP HTTP or the evidence API is enabled in production"
            )
        if "*" in self.cors_origins or any("localhost" in origin for origin in self.cors_origins):
            problems.append("CORS_ORIGINS must contain only explicit production origins")
        if problems:
            raise ValueError("; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
