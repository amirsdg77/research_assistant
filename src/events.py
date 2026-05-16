"""In-process event bus for streaming agent events to SSE clients.

Design:
- One `asyncio.Queue` per (session_id, subscriber) pair. The agent loop
  publishes to *all* subscribers of a session; each SSE handler owns its
  queue so a slow client can't block the loop or other clients.
- Bounded queues with `maxsize=256`. If a subscriber falls behind, we drop
  the oldest event and log it. Dropping > blocking: SSE clients can always
  re-fetch session state via `GET /api/sessions/{id}` to recover; blocking
  the agent loop would freeze the whole session.
- `subscribe()` returns an async context manager so SSE handlers can't
  forget to unregister. Disconnects clean themselves up.
- A sentinel event (`type="__close__"`) is used to break the SSE consumer
  out of its read loop on session completion — saves the client from
  hanging open after `session.completed`.

Why in-process and not Redis pub/sub: the plan says single-app deployment.
Adding Redis here would be premature; it's trivial to swap in later if we
horizontally scale the API.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from src.logging_setup import get_logger


log = get_logger(__name__)


# --- Event types ---------------------------------------------------------
#
# Canonical event type names used across loop, SSE, and UI. The UI's app.js
# dispatches by `event.type` so adding new ones requires touching both sides.
# Kept as plain strings (not an Enum) because they ride across the JSON
# wire — strings are simpler than serialization gymnastics.

class EventTypes:
    PLAN_READY = "plan.ready"
    TASK_STATUS_CHANGED = "task.status_changed"
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    LLM_CALLED = "llm.called"
    REPORT_READY = "report.ready"
    VERIFICATION_READY = "verification.ready"
    SESSION_STARTED = "session.started"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    GUARDRAIL_TRIGGERED = "guardrail.triggered"
    # Internal sentinel; never sent to clients.
    _CLOSE = "__close__"


_CRITICAL_TYPES = frozenset({
    EventTypes.PLAN_READY,
    EventTypes.TASK_STATUS_CHANGED,
    EventTypes.REPORT_READY,
    EventTypes.VERIFICATION_READY,
    EventTypes.SESSION_STARTED,
    EventTypes.SESSION_COMPLETED,
    EventTypes.SESSION_FAILED,
    EventTypes.GUARDRAIL_TRIGGERED,
    EventTypes._CLOSE,
})


@dataclass
class AgentEvent:
    """A single event flowing from the agent loop to subscribers."""

    session_id: UUID
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def critical(self) -> bool:
        return self.type in _CRITICAL_TYPES

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": str(self.session_id),
            "type": self.type,
            "data": self.data,
            "ts": self.ts.isoformat(),
        }


# --- Bus -----------------------------------------------------------------


# Per-subscriber queue capacity. Sized for a typical session emitting
# ~30-100 events; high enough that a normally-paced SSE client never
# evicts. If a queue fills, publish() drops the oldest. Not deployer-tunable
# — change here if event volume per session grows materially.
_QUEUE_MAXSIZE = 256


class EventBus:
    """Process-singleton fan-out hub.

    Subscribers register their own queue per session_id. Publishers don't
    know or care who's listening; they just call `publish(event)` and the
    bus copies it into every matching queue.
    """

    def __init__(self) -> None:
        # session_id -> {subscriber_id: queue}
        self._subs: dict[UUID, dict[UUID, asyncio.Queue[AgentEvent]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event: AgentEvent) -> None:
        """Fan out to all subscribers of the event's session.

        Non-blocking: if a queue is full, drop the oldest and try again.
        We never `await put()` because that would tie loop progress to
        subscriber speed.
        """
        async with self._lock:
            sub_queues = list(self._subs.get(event.session_id, {}).items())

        for sub_id, queue in sub_queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._evict_and_put(queue, event, sub_id)

    def _evict_and_put(
        self,
        queue: asyncio.Queue[AgentEvent],
        event: AgentEvent,
        sub_id: UUID,
    ) -> None:
        """Make room by dropping the oldest non-critical event, then enqueue.

        If every queued event is critical (rare), drop the oldest of those.
        Falling behind is preferable to disconnecting the subscriber.
        """
        buffered: list[AgentEvent] = []
        try:
            while True:
                buffered.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            pass

        drop_idx: int | None = None
        for i, ev in enumerate(buffered):
            if not ev.critical:
                drop_idx = i
                break
        if drop_idx is None and buffered:
            drop_idx = 0
        dropped = buffered.pop(drop_idx) if drop_idx is not None else None

        for ev in buffered:
            try:
                queue.put_nowait(ev)
            except asyncio.QueueFull:  # pragma: no cover
                break
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover
            log.warning(
                "event.dropped_after_evict",
                session_id=str(event.session_id),
                sub_id=str(sub_id),
                type=event.type,
            )
            return
        if dropped is not None:
            log.info(
                "event.evicted",
                session_id=str(event.session_id),
                sub_id=str(sub_id),
                evicted_type=dropped.type,
                evicted_critical=dropped.critical,
            )

    @asynccontextmanager
    async def subscribe(
        self, session_id: UUID
    ) -> AsyncIterator[asyncio.Queue[AgentEvent]]:
        """Register a new subscriber and yield its queue.

        Always use as `async with bus.subscribe(sid) as queue: ...` — the
        context manager guarantees cleanup even if the SSE client drops.
        """
        sub_id = uuid4()
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        async with self._lock:
            self._subs.setdefault(session_id, {})[sub_id] = queue
        log.info(
            "event.subscribed",
            session_id=str(session_id),
            sub_id=str(sub_id),
            subscriber_count=len(self._subs[session_id]),
        )
        try:
            yield queue
        finally:
            async with self._lock:
                session_subs = self._subs.get(session_id, {})
                session_subs.pop(sub_id, None)
                if not session_subs:
                    self._subs.pop(session_id, None)
            log.info(
                "event.unsubscribed",
                session_id=str(session_id),
                sub_id=str(sub_id),
            )

    async def close_session(self, session_id: UUID) -> None:
        """Send the close sentinel to every subscriber so their read loops exit.

        Called from the agent loop after `session.completed` or
        `session.failed` so the SSE response can finalize cleanly.
        """
        sentinel = AgentEvent(session_id=session_id, type=EventTypes._CLOSE)
        await self.publish(sentinel)

    def subscriber_count(self, session_id: UUID) -> int:
        """Diagnostic — used in logs/tests to confirm registration."""
        return len(self._subs.get(session_id, {}))


# Module-level singleton. Importers grab `bus` directly.
bus = EventBus()


__all__ = ["AgentEvent", "EventTypes", "EventBus", "bus"]
