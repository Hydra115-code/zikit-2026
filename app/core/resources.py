from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import delete

from app.config import Settings
from app.core.incidents import IncidentRepository
from app.database import ResourceMetricRecord, session_factory
from app.schemas import ResourceSnapshot

logger = logging.getLogger(__name__)

EVENT_RESOURCE_ANOMALY_DETECTED = "RESOURCE_ANOMALY_DETECTED"
EVENT_RESOURCE_RECOVERED = "RESOURCE_RECOVERED"


class ResourceCollector(Protocol):
    def collect(self) -> ResourceSnapshot: ...


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _read_nonnegative_int(path: Path) -> int | None:
    value = _read_text(path)
    if value is None or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


class LinuxResourceCollector:
    """Collect Linux host/container metrics without shelling out or requiring root."""

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        disk_path: Path = Path("/"),
        source_id: str | None = None,
    ) -> None:
        self.proc_root = proc_root
        self.cgroup_root = cgroup_root
        self.disk_path = disk_path
        self.source_id = source_id or socket.gethostname()
        self._previous_cpu: tuple[int, int] | None = None
        self._previous_cgroup_cpu: tuple[int, float] | None = None

    def _cpu(self) -> tuple[float, int]:
        stat = _read_text(self.proc_root / "stat")
        if stat is None:
            raise RuntimeError("/proc/stat is unavailable")
        cpu_line = next((line for line in stat.splitlines() if line.startswith("cpu ")), None)
        if cpu_line is None:
            raise RuntimeError("aggregate CPU counters are unavailable")
        values = [int(value) for value in cpu_line.split()[1:9]]
        total = sum(values)
        idle = values[3] + values[4]
        percent = 0.0
        if self._previous_cpu is not None:
            previous_total, previous_idle = self._previous_cpu
            total_delta = total - previous_total
            idle_delta = idle - previous_idle
            if total_delta > 0:
                percent = max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100))
        self._previous_cpu = (total, idle)
        cpu_count = os.cpu_count() or 1
        return percent, cpu_count

    def _memory(self) -> tuple[int, int, float]:
        contents = _read_text(self.proc_root / "meminfo")
        if contents is None:
            raise RuntimeError("/proc/meminfo is unavailable")
        values: dict[str, int] = {}
        for line in contents.splitlines():
            key, _, raw = line.partition(":")
            fields = raw.strip().split()
            if fields:
                values[key] = int(fields[0]) * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        if total <= 0:
            raise RuntimeError("total memory is unavailable")
        percent = max(0.0, min(100.0, (total - available) / total * 100))
        return total, available, percent

    def _network(self) -> tuple[int, int, int, int]:
        contents = _read_text(self.proc_root / "net" / "dev")
        if contents is None:
            raise RuntimeError("/proc/net/dev is unavailable")
        rx_bytes = tx_bytes = rx_errors = tx_errors = 0
        for line in contents.splitlines()[2:]:
            interface, separator, raw = line.partition(":")
            if not separator or interface.strip() == "lo":
                continue
            fields = raw.split()
            if len(fields) < 16:
                continue
            rx_bytes += int(fields[0])
            rx_errors += int(fields[2])
            tx_bytes += int(fields[8])
            tx_errors += int(fields[10])
        return rx_bytes, tx_bytes, rx_errors, tx_errors

    def _cgroup(
        self,
    ) -> tuple[int | None, int | None, float | None, float | None, int | None]:
        current = _read_nonnegative_int(self.cgroup_root / "memory.current")
        limit = _read_nonnegative_int(self.cgroup_root / "memory.max")
        memory_percent = None
        if current is not None and limit not in {None, 0}:
            memory_percent = current / limit * 100
        cpu_quota = None
        cpu_max = _read_text(self.cgroup_root / "cpu.max")
        if cpu_max:
            quota, _, period = cpu_max.partition(" ")
            if quota != "max":
                try:
                    quota_value = int(quota)
                    period_value = int(period)
                    if quota_value > 0 and period_value > 0:
                        cpu_quota = quota_value / period_value
                except ValueError:
                    pass
        usage_usec = None
        cpu_stat = _read_text(self.cgroup_root / "cpu.stat")
        if cpu_stat:
            for line in cpu_stat.splitlines():
                key, _, raw = line.partition(" ")
                if key == "usage_usec":
                    try:
                        usage_usec = int(raw)
                    except ValueError:
                        pass
                    break
        return current, limit, memory_percent, cpu_quota, usage_usec

    def _cgroup_cpu_percent(
        self,
        usage_usec: int | None,
        quota_cores: float | None,
        cpu_count: int,
    ) -> float | None:
        if usage_usec is None:
            return None
        now = time.monotonic()
        percent = None
        if self._previous_cgroup_cpu is not None:
            previous_usage, previous_time = self._previous_cgroup_cpu
            elapsed = now - previous_time
            usage_delta_seconds = (usage_usec - previous_usage) / 1_000_000
            capacity_cores = quota_cores or float(cpu_count)
            if elapsed > 0 and usage_delta_seconds >= 0:
                percent = max(
                    0.0,
                    min(100.0, usage_delta_seconds / (elapsed * capacity_cores) * 100),
                )
        self._previous_cgroup_cpu = (usage_usec, now)
        return percent

    def _containerized(self) -> bool:
        if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
            return True
        cgroup = _read_text(self.proc_root / "1" / "cgroup") or ""
        return any(marker in cgroup for marker in ("docker", "containerd", "libpod", "kubepods"))

    def collect(self) -> ResourceSnapshot:
        cpu_percent, cpu_count = self._cpu()
        memory_total, memory_available, memory_percent = self._memory()
        disk = shutil.disk_usage(self.disk_path)
        disk_percent = (disk.total - disk.free) / disk.total * 100 if disk.total else 0.0
        load_1m, load_5m, load_15m = os.getloadavg()
        rx_bytes, tx_bytes, rx_errors, tx_errors = self._network()
        process_count = sum(
            1 for entry in self.proc_root.iterdir() if entry.name.isdigit() and entry.is_dir()
        )
        cgroup_current, cgroup_limit, cgroup_percent, cpu_quota, cpu_usage = self._cgroup()
        cgroup_cpu_percent = self._cgroup_cpu_percent(cpu_usage, cpu_quota, cpu_count)
        return ResourceSnapshot(
            source_id=self.source_id,
            timestamp=datetime.now(UTC),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            load_1m=load_1m,
            load_5m=load_5m,
            load_15m=load_15m,
            load_per_cpu=load_1m / cpu_count,
            memory_total_bytes=memory_total,
            memory_available_bytes=memory_available,
            memory_percent=memory_percent,
            disk_path=str(self.disk_path),
            disk_total_bytes=disk.total,
            disk_free_bytes=disk.free,
            disk_percent=disk_percent,
            network_rx_bytes=rx_bytes,
            network_tx_bytes=tx_bytes,
            network_rx_errors=rx_errors,
            network_tx_errors=tx_errors,
            process_count=process_count,
            containerized=self._containerized(),
            cgroup_memory_current_bytes=cgroup_current,
            cgroup_memory_limit_bytes=cgroup_limit,
            cgroup_memory_percent=cgroup_percent,
            cgroup_cpu_quota_cores=cpu_quota,
            cgroup_cpu_percent=cgroup_cpu_percent,
        )


def assess_resources(snapshot: ResourceSnapshot, settings: Settings) -> list[dict[str, Any]]:
    checks = (
        ("cpu_percent", snapshot.cpu_percent, settings.resource_cpu_threshold_percent),
        ("memory_percent", snapshot.memory_percent, settings.resource_memory_threshold_percent),
        ("disk_percent", snapshot.disk_percent, settings.resource_disk_threshold_percent),
        ("load_per_cpu", snapshot.load_per_cpu, settings.resource_load_per_cpu_threshold),
    )
    reasons = [
        {"metric": metric, "value": value, "threshold": threshold}
        for metric, value, threshold in checks
        if value >= threshold
    ]
    if (
        snapshot.cgroup_memory_percent is not None
        and snapshot.cgroup_memory_percent >= settings.resource_cgroup_memory_threshold_percent
    ):
        reasons.append(
            {
                "metric": "cgroup_memory_percent",
                "value": snapshot.cgroup_memory_percent,
                "threshold": settings.resource_cgroup_memory_threshold_percent,
            }
        )
    if (
        snapshot.cgroup_cpu_percent is not None
        and snapshot.cgroup_cpu_percent >= settings.resource_cpu_threshold_percent
    ):
        reasons.append(
            {
                "metric": "cgroup_cpu_percent",
                "value": snapshot.cgroup_cpu_percent,
                "threshold": settings.resource_cpu_threshold_percent,
            }
        )
    return reasons


@dataclass
class ResourceMonitorStatus:
    running: bool = False
    total_samples: int = 0
    failed_samples: int = 0
    anomaly_active: bool = False
    consecutive_anomalous: int = 0
    consecutive_healthy: int = 0
    last_collected_at: datetime | None = None
    last_error_type: str | None = None

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "total_samples": self.total_samples,
            "failed_samples": self.failed_samples,
            "anomaly_active": self.anomaly_active,
            "consecutive_anomalous": self.consecutive_anomalous,
            "consecutive_healthy": self.consecutive_healthy,
            "last_collected_at": self.last_collected_at,
            "last_error_type": self.last_error_type,
        }


class ResourceMonitor:
    def __init__(
        self,
        settings: Settings,
        repository: IncidentRepository,
        collector: ResourceCollector | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.collector = collector or LinuxResourceCollector(
            disk_path=settings.resource_disk_path,
            source_id=settings.resource_source_id,
        )
        self.history: deque[tuple[ResourceSnapshot, list[dict[str, Any]]]] = deque(
            maxlen=settings.resource_history_size
        )
        self.latest: ResourceSnapshot | None = None
        self.latest_reasons: list[dict[str, Any]] = []
        self.status = ResourceMonitorStatus()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self.settings.resource_monitor_enabled or self._task is not None:
            return
        self._stop.clear()
        self.status.running = True
        try:
            await self.sample_once()
        except Exception as exc:
            self.status.failed_samples += 1
            self.status.last_error_type = type(exc).__name__
            logger.exception("initial resource collection failed")
        self._task = asyncio.create_task(self._loop(), name="sentinel-resource-monitor")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.status.running = False

    async def sample_once(self) -> ResourceSnapshot:
        # The collector only performs bounded local procfs/cgroup reads. Keeping
        # it synchronous avoids depending on thread creation in constrained
        # containers while remaining well below the configured sampling period.
        snapshot = self.collector.collect()
        reasons = assess_resources(snapshot, self.settings)
        self.latest = snapshot
        self.latest_reasons = reasons
        self.history.append((snapshot, reasons))
        self.status.total_samples += 1
        self.status.last_collected_at = snapshot.timestamp
        self.status.last_error_type = None
        if reasons:
            self.status.consecutive_anomalous += 1
            self.status.consecutive_healthy = 0
        else:
            self.status.consecutive_anomalous = 0
            self.status.consecutive_healthy += 1
        await self._persist(snapshot, reasons)

        if (
            reasons
            and not self.status.anomaly_active
            and self.status.consecutive_anomalous
            >= self.settings.resource_anomaly_persistence_windows
        ):
            self.status.anomaly_active = True
            await self.repository.event(
                EVENT_RESOURCE_ANOMALY_DETECTED,
                details={
                    "source_id": snapshot.source_id,
                    "reasons": reasons,
                    "snapshot": snapshot.model_dump(mode="json"),
                },
            )
        elif (
            not reasons
            and self.status.anomaly_active
            and self.status.consecutive_healthy
            >= self.settings.resource_recovery_persistence_windows
        ):
            self.status.anomaly_active = False
            await self.repository.event(
                EVENT_RESOURCE_RECOVERED,
                details={
                    "source_id": snapshot.source_id,
                    "snapshot": snapshot.model_dump(mode="json"),
                },
            )
        return snapshot

    async def _persist(
        self,
        snapshot: ResourceSnapshot,
        reasons: list[dict[str, Any]],
    ) -> None:
        values = snapshot.model_dump()
        with session_factory()() as session:
            session.add(ResourceMetricRecord(**values, anomalous=bool(reasons), reasons=reasons))
            if self.status.total_samples % self.settings.resource_cleanup_every_samples == 0:
                cutoff = datetime.now(UTC) - timedelta(hours=self.settings.resource_retention_hours)
                session.execute(
                    delete(ResourceMetricRecord).where(ResourceMetricRecord.timestamp < cutoff)
                )
            session.commit()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.settings.resource_monitor_interval_seconds)
                await self.sample_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status.failed_samples += 1
                self.status.last_error_type = type(exc).__name__
                logger.exception("resource collection failed")

    def current(self) -> dict[str, Any]:
        return {
            "status": self.status.as_public_dict(),
            "snapshot": self.latest.model_dump(mode="json") if self.latest else None,
            "anomalous": bool(self.latest_reasons),
            "reasons": self.latest_reasons,
            "thresholds": {
                "cpu_percent": self.settings.resource_cpu_threshold_percent,
                "memory_percent": self.settings.resource_memory_threshold_percent,
                "disk_percent": self.settings.resource_disk_threshold_percent,
                "cgroup_memory_percent": self.settings.resource_cgroup_memory_threshold_percent,
                "load_per_cpu": self.settings.resource_load_per_cpu_threshold,
                "anomaly_persistence_windows": self.settings.resource_anomaly_persistence_windows,
                "recovery_persistence_windows": self.settings.resource_recovery_persistence_windows,
            },
        }
