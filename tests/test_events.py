"""Tests for the event bus.

Properties under test:
- Ordered delivery to a single subscriber.
- Fan-out: multiple subscribers on the same session each get every event.
- Cross-session isolation: events for session A don't reach subscribers of B.
- Cleanup: exiting the subscribe() context drops the subscriber.
- Backpressure: a slow subscriber evicts its oldest event rather than
  blocking the publisher or other subscribers.
- Close sentinel: `close_session` reaches all subscribers.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from src.events import AgentEvent, EventBus, EventTypes


async def _drain(queue: asyncio.Queue, n: int, timeout: float = 1.0) -> list:
    """Pull exactly n events from a queue or raise on timeout."""
    out = []
    for _ in range(n):
        out.append(await asyncio.wait_for(queue.get(), timeout=timeout))
    return out


async def test_single_subscriber_receives_in_order():
    bus = EventBus()
    sid = uuid4()
    async with bus.subscribe(sid) as queue:
        for i in range(5):
            await bus.publish(AgentEvent(session_id=sid, type="x", data={"i": i}))
        events = await _drain(queue, 5)

    assert [e.data["i"] for e in events] == [0, 1, 2, 3, 4]


async def test_fanout_to_multiple_subscribers():
    bus = EventBus()
    sid = uuid4()
    async with bus.subscribe(sid) as q1, bus.subscribe(sid) as q2:
        assert bus.subscriber_count(sid) == 2
        await bus.publish(AgentEvent(session_id=sid, type=EventTypes.PLAN_READY))
        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)

    assert e1.type == EventTypes.PLAN_READY
    assert e2.type == EventTypes.PLAN_READY
    # After context exit, both subs are gone.
    assert bus.subscriber_count(sid) == 0


async def test_cross_session_isolation():
    bus = EventBus()
    sid_a, sid_b = uuid4(), uuid4()
    async with bus.subscribe(sid_a) as qa, bus.subscribe(sid_b) as qb:
        await bus.publish(AgentEvent(session_id=sid_a, type="A"))
        await bus.publish(AgentEvent(session_id=sid_b, type="B"))

        ea = await asyncio.wait_for(qa.get(), timeout=1.0)
        eb = await asyncio.wait_for(qb.get(), timeout=1.0)

    assert ea.type == "A"
    assert eb.type == "B"
    # Neither queue saw the other's event.
    assert qa.empty()
    assert qb.empty()


async def test_subscribe_context_manager_cleans_up_on_exception():
    bus = EventBus()
    sid = uuid4()
    with pytest.raises(RuntimeError):
        async with bus.subscribe(sid):
            assert bus.subscriber_count(sid) == 1
            raise RuntimeError("boom")
    assert bus.subscriber_count(sid) == 0


async def test_slow_subscriber_does_not_block_publisher():
    """One subscriber drains, the other doesn't. Publisher must stay
    responsive — i.e. the call to publish() never hangs."""
    bus = EventBus()
    sid = uuid4()
    async with bus.subscribe(sid) as fast, bus.subscribe(sid) as slow:
        # Fill the slow subscriber's queue past its capacity. Publisher
        # must NOT await on the slow queue.
        for i in range(300):
            await bus.publish(AgentEvent(session_id=sid, type="x", data={"i": i}))
        # Fast subscriber can still drain its queue normally.
        # It saw the last 256 events (queue maxsize); we just confirm
        # we can pull something without blocking.
        first = await asyncio.wait_for(fast.get(), timeout=1.0)
        assert first.type == "x"
        # Slow subscriber should also have something pending — we never
        # blocked, just evicted.
        assert slow.qsize() > 0


async def test_close_session_reaches_all_subscribers():
    bus = EventBus()
    sid = uuid4()
    async with bus.subscribe(sid) as q1, bus.subscribe(sid) as q2:
        await bus.close_session(sid)
        e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        e2 = await asyncio.wait_for(q2.get(), timeout=1.0)

    assert e1.type == EventTypes._CLOSE
    assert e2.type == EventTypes._CLOSE


async def test_publish_with_no_subscribers_is_noop():
    """The agent loop shouldn't have to check if anyone is listening."""
    bus = EventBus()
    sid = uuid4()
    # No subscribers, no exception, no log spam.
    await bus.publish(AgentEvent(session_id=sid, type="anyone_home"))
    assert bus.subscriber_count(sid) == 0


def test_event_to_json_shape():
    sid = uuid4()
    e = AgentEvent(session_id=sid, type="task.status_changed", data={"status": "done"})
    payload = e.to_json()
    assert payload["session_id"] == str(sid)
    assert payload["type"] == "task.status_changed"
    assert payload["data"] == {"status": "done"}
    # Timestamp serializes as ISO string.
    assert isinstance(payload["ts"], str)
    assert "T" in payload["ts"]
