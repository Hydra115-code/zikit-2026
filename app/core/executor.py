from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core.policy import PolicyEngine


@dataclass(frozen=True)
class ExecutionResult:
    state: str
    action: str
    details: str
    idempotent_replay: bool = False


class RunbookExecutor:
    def __init__(self, policy: PolicyEngine, switch_provider: Callable[[str], Awaitable[None]]) -> None:
        self.policy = policy
        self.switch_provider = switch_provider
        self._lock = asyncio.Lock()
        self._completed: dict[str, ExecutionResult] = {}

    async def execute(
        self,
        runbook: str,
        timeout_seconds: float = 5,
        *,
        idempotency_key: str | None = None,
    ) -> ExecutionResult:
        if runbook != "failover_to_backup":
            return ExecutionResult("rejected", runbook, "runbook is not allowlisted")
        if idempotency_key and idempotency_key in self._completed:
            previous = self._completed[idempotency_key]
            return ExecutionResult(
                previous.state,
                previous.action,
                previous.details,
                idempotent_replay=True,
            )
        if self._lock.locked():
            return ExecutionResult("rejected", runbook, "another runbook is executing")
        async with self._lock:
            self.policy.action_started()
            try:
                await asyncio.wait_for(self.switch_provider("backup"), timeout=timeout_seconds)
                result = ExecutionResult("completed", runbook, "active provider changed to backup")
                if idempotency_key:
                    self._completed[idempotency_key] = result
                return result
            except TimeoutError:
                return ExecutionResult("failed", runbook, "runbook timed out")
            except Exception as exc:  # audit a safe message, never a generated command
                return ExecutionResult("failed", runbook, f"provider switch failed: {type(exc).__name__}")
            finally:
                self.policy.action_finished()

    def reset(self) -> None:
        self._completed.clear()
