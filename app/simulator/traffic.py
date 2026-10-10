from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.schemas import TelemetrySample


@dataclass(frozen=True)
class TrafficGeneratorStatus:
    running: bool
    request_count: int
    failure_count: int


class TrafficGenerator:
    """Continuously exercises the gateway through the same measurable probe path."""

    def __init__(self, probe: Callable[[], Awaitable[TelemetrySample]], interval_seconds: float) -> None:
        self.probe = probe
        self.interval_seconds = interval_seconds
        self.request_count = 0
        self.failure_count = 0
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def status(self) -> TrafficGeneratorStatus:
        return TrafficGeneratorStatus(
            running=self._task is not None and not self._task.done(),
            request_count=self.request_count,
            failure_count=self.failure_count,
        )

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="sentinel-traffic-generator")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def reset_counts(self) -> None:
        self.request_count = 0
        self.failure_count = 0

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = await self.probe()
                self.request_count += 1
                if not sample.success:
                    self.failure_count += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.request_count += 1
                self.failure_count += 1
            await asyncio.sleep(self.interval_seconds)
