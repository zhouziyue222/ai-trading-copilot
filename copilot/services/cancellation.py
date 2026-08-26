"""Cooperative cancellation primitives for UI runs and tool calls."""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterator


class RunCancelled(Exception):
    """Raised when a run should stop and discard the current result."""


CancellationChecker = Callable[[], bool]
CancelCallback = Callable[[], None]


@dataclass
class ToolExecution:
    execution_id: str
    run_id: str
    tool_name: str
    cancel_callback: CancelCallback | None = None
    cancel_requested: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def request_cancel(self) -> None:
        self.cancel_requested = True
        if self.cancel_callback is not None:
            self.cancel_callback()


class CancellationManager:
    """Tracks active cancellable tool executions by run id."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._executions: dict[str, ToolExecution] = {}
        self._run_index: dict[str, set[str]] = {}

    @contextmanager
    def register(
        self,
        *,
        run_id: str,
        tool_name: str,
        cancel_callback: CancelCallback | None = None,
    ) -> Iterator[ToolExecution]:
        execution = ToolExecution(
            execution_id=secrets.token_hex(8),
            run_id=run_id,
            tool_name=tool_name,
            cancel_callback=cancel_callback,
        )
        with self._lock:
            self._executions[execution.execution_id] = execution
            self._run_index.setdefault(run_id, set()).add(execution.execution_id)
        try:
            yield execution
        finally:
            with self._lock:
                self._executions.pop(execution.execution_id, None)
                run_executions = self._run_index.get(run_id)
                if run_executions is not None:
                    run_executions.discard(execution.execution_id)
                    if not run_executions:
                        self._run_index.pop(run_id, None)

    def request_cancel(self, run_id: str) -> list[str]:
        with self._lock:
            executions = [
                self._executions[execution_id]
                for execution_id in sorted(self._run_index.get(run_id, set()))
                if execution_id in self._executions
            ]
        cancelled_ids = []
        for execution in executions:
            execution.request_cancel()
            cancelled_ids.append(execution.execution_id)
        return cancelled_ids


@dataclass
class CancellationToken:
    run_id: str
    checker: CancellationChecker
    manager: CancellationManager

    def is_cancel_requested(self) -> bool:
        return bool(self.checker())

    def check(self) -> None:
        if self.is_cancel_requested():
            raise RunCancelled("Run cancelled by user.")


_current_token: ContextVar[CancellationToken | None] = ContextVar(
    "copilot_cancellation_token",
    default=None,
)

GLOBAL_CANCELLATION_MANAGER = CancellationManager()


@contextmanager
def activate_cancellation(token: CancellationToken | None) -> Iterator[None]:
    if token is None:
        yield
        return
    context_token = _current_token.set(token)
    try:
        yield
    finally:
        _current_token.reset(context_token)


def get_current_cancellation_token() -> CancellationToken | None:
    return _current_token.get()


def check_cancelled() -> None:
    token = get_current_cancellation_token()
    if token is not None:
        token.check()


def tool_cancel_callback(tool_item: Any, execution_id: str) -> CancelCallback | None:
    cancel_execution = getattr(tool_item, "cancel_execution", None)
    if callable(cancel_execution):
        return lambda: cancel_execution(execution_id)
    cancel = getattr(tool_item, "cancel", None)
    if callable(cancel):
        return lambda: cancel(execution_id=execution_id)
    return None
