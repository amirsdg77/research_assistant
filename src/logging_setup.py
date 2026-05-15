"""Structured logging via structlog + contextvars.

Design:
- JSON renderer to stdout (Docker captures it; easy to grep / pipe to jq).
- `contextvars` integration so `session_id`, `task_id`, `request_id` bind once
  and propagate through every async call beneath them — no manual passing.
- Helpers (`bind_session`, `bind_task`, `bind_request`) return context managers
  that auto-unbind on exit, so binds don't leak across requests/tasks.
- A small set of canonical event names is documented at the bottom so call
  sites stay consistent and dashboards can rely on them.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import UUID

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)

from src.config import settings


_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent. Safe to call from app startup and from CLI entrypoints."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # stdlib logging is the transport; structlog renders into it.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            merge_contextvars,  # pulls session_id/task_id/request_id automatically
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Always pass through `configure_logging` first.

    Modules can do `log = get_logger(__name__)` at import time safely; if
    configuration hasn't happened yet (e.g. in tests), it happens now.
    """
    configure_logging()
    return structlog.get_logger(name)


# --- Context binding helpers --------------------------------------------
#
# These wrap structlog.contextvars so callers don't have to remember the exact
# keys, and so unbinding is paired with binding via context managers — even on
# exceptions the keys are cleared. Critical for long-running servers where a
# stray `session_id` would otherwise pollute unrelated logs.


def _str_uuid(value: UUID | str) -> str:
    return str(value)


@contextmanager
def bind_session(session_id: UUID | str) -> Iterator[None]:
    bind_contextvars(session_id=_str_uuid(session_id))
    try:
        yield
    finally:
        unbind_contextvars("session_id")


@contextmanager
def bind_task(task_id: UUID | str) -> Iterator[None]:
    bind_contextvars(task_id=_str_uuid(task_id))
    try:
        yield
    finally:
        unbind_contextvars("task_id")


@contextmanager
def bind_request(request_id: str) -> Iterator[None]:
    bind_contextvars(request_id=request_id)
    try:
        yield
    finally:
        unbind_contextvars("request_id")


def reset_context() -> None:
    """Clear all bound context. Useful in tests and worker boundaries."""
    clear_contextvars()


# --- Event name catalog --------------------------------------------------
#
# Single source of truth for log event names. Modules import these constants
# rather than typing strings, so a rename here propagates everywhere and the
# UI activity feed / dashboards have a stable contract.

class Events:
    # Session lifecycle
    SESSION_STARTED = "session.started"
    SESSION_PLANNING = "session.planning"
    SESSION_PLAN_READY = "session.plan_ready"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_RESUMED = "session.resumed"

    # Task lifecycle
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_TOOL_ITERATION = "task.tool_iteration"

    # Tool execution
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # LLM
    LLM_CALLED = "llm.called"
    LLM_RESPONDED = "llm.responded"

    # Context / guardrails
    CONTEXT_TRUNCATED = "context.truncated"
    GUARDRAIL_TRIGGERED = "guardrail.triggered"


__all__ = [
    "configure_logging",
    "get_logger",
    "bind_session",
    "bind_task",
    "bind_request",
    "reset_context",
    "Events",
]


def _log_extras_for_test(**kwargs: Any) -> dict[str, Any]:
    """Internal helper used by the smoke test to assert that contextvars
    propagate through an `await` boundary. Not part of the public API."""
    log = get_logger("ctx-test")
    log.info("probe", **kwargs)
    return dict(structlog.contextvars.get_contextvars())
