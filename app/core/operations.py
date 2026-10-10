from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamCapacitySnapshot:
    active_connections: int
    max_connections: int
    total_opened: int
    total_rejected: int


class EventStreamCapacity:
    """Process-local admission control for long-lived SSE responses."""

    def __init__(self, max_connections: int) -> None:
        self.max_connections = max_connections
        self._active_connections = 0
        self._total_opened = 0
        self._total_rejected = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active_connections >= self.max_connections:
                self._total_rejected += 1
                return False
            self._active_connections += 1
            self._total_opened += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active_connections <= 0:
                raise RuntimeError("event stream capacity released without acquisition")
            self._active_connections -= 1

    def snapshot(self) -> StreamCapacitySnapshot:
        return StreamCapacitySnapshot(
            active_connections=self._active_connections,
            max_connections=self.max_connections,
            total_opened=self._total_opened,
            total_rejected=self._total_rejected,
        )
