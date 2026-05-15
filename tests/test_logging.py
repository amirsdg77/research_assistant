"""Smoke tests for src.logging_setup.

The core property we need: bound context (session_id, task_id) survives
across `await` boundaries and concurrent tasks — that's what makes the
agent loop's logging usable without manual key-passing.

Capture strategy: re-configure structlog at test time with a list-appending
final processor in place of JSONRenderer. We keep `merge_contextvars` first
in the chain so bound contextvars actually appear in captured events
(structlog.testing.capture_logs swaps the entire chain and would skip
contextvars merging, which is exactly the property we're trying to verify).
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
import structlog
from structlog.contextvars import merge_contextvars

from src.logging_setup import (
    Events,
    bind_request,
    bind_session,
    bind_task,
    get_logger,
    reset_context,
)


class _ListSink:
    """Final-processor sink: captures the fully-merged event dict."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def __call__(self, _logger, _method_name, event_dict):
        self.entries.append(dict(event_dict))
        return ""  # stdlib logger swallows it; we don't care about stdout here


@pytest.fixture
def capture():
    """Reconfigure structlog with merge_contextvars + list sink for one test.

    Two non-obvious bits:
    1. `reset_defaults()` wipes any cached logger so the new processor chain
       actually takes effect — without it, `cache_logger_on_first_use=True`
       in the app config would pin a stale logger to the old chain.
    2. We flip `_CONFIGURED=True` so `get_logger()` calls inside the test
       short-circuit and don't re-run the production configure_logging(),
       which would clobber our sink with the JSONRenderer chain.
    """
    import src.logging_setup as _ls
    structlog.reset_defaults()
    sink = _ListSink()
    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.processors.add_log_level,
            sink,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _ls._CONFIGURED = True  # prevent get_logger() from re-configuring
    reset_context()
    yield sink
    reset_context()
    structlog.reset_defaults()
    _ls._CONFIGURED = False


async def test_context_propagates_across_await(capture):
    sid = uuid4()
    tid = uuid4()
    log = get_logger("test")

    with bind_session(sid):
        with bind_task(tid):
            log.info(Events.TASK_STARTED)
            await asyncio.sleep(0)  # forces a real await boundary
            log.info(Events.TOOL_INVOKED, tool="web_search")

    cap = capture.entries
    assert len(cap) == 2
    for entry in cap:
        assert entry["session_id"] == str(sid)
        assert entry["task_id"] == str(tid)
    assert cap[0]["event"] == Events.TASK_STARTED
    assert cap[1]["event"] == Events.TOOL_INVOKED
    assert cap[1]["tool"] == "web_search"


async def test_context_unbinds_on_exit(capture):
    log = get_logger("test")
    sid = uuid4()
    with bind_session(sid):
        log.info("inside")
    log.info("outside")

    cap = capture.entries
    assert cap[0]["session_id"] == str(sid)
    assert "session_id" not in cap[1]


async def test_concurrent_tasks_have_isolated_context(capture):
    """If session contexts leaked across asyncio.gather sibling tasks, we'd
    see entries with the wrong session_id. This is the regression test."""
    log = get_logger("test")
    sids = [uuid4() for _ in range(3)]

    async def worker(sid):
        with bind_session(sid):
            await asyncio.sleep(0)
            log.info("work", which=str(sid))
            await asyncio.sleep(0)
            log.info("work_again", which=str(sid))

    await asyncio.gather(*(worker(s) for s in sids))

    cap = capture.entries
    # Every line's session_id must match its own `which` field — no leakage.
    assert len(cap) == 6
    for entry in cap:
        assert entry["session_id"] == entry["which"]


async def test_request_id_binds_independently(capture):
    log = get_logger("test")
    with bind_request("req-123"):
        with bind_session(uuid4()):
            log.info(Events.SESSION_STARTED)
    cap = capture.entries
    assert cap[0]["request_id"] == "req-123"
    assert "session_id" in cap[0]


async def test_nested_bind_does_not_leak_after_inner_exit(capture):
    """bind_task inside bind_session should clean up only the task_id on
    its exit — session_id must persist until its own context closes."""
    log = get_logger("test")
    sid = uuid4()
    tid = uuid4()
    with bind_session(sid):
        with bind_task(tid):
            log.info("inner")
        log.info("after_task_exit")

    cap = capture.entries
    assert cap[0]["session_id"] == str(sid)
    assert cap[0]["task_id"] == str(tid)
    assert cap[1]["session_id"] == str(sid)
    assert "task_id" not in cap[1]
