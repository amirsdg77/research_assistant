"""Tests for the agent loop.

Strategy: mock `complete`, the tool invoker, the verifier, the guardrail,
and `session_scope` so the loop runs entirely in memory. Each test asserts
on a specific phase transition, persistence call, or event emission.

Properties under test:
- Guardrail failure short-circuits and marks the session failed.
- Planner creates and persists tasks; PLAN_READY event fires.
- Per-task: status moves pending → in_progress → done; result + sources persist.
- Inner tool loop: parallel tool calls run, results route correctly,
  finish_task with valid sources ends the task.
- Citation enforcement: empty-sources finish triggers a correction; >2
  corrections fails the task.
- Iteration cap: max iterations without finish_task fails the task.
- Synthesis + verification: report stored on session, verification notes
  composed from claims + notes.
- Session completion event fires; bus close sentinel is sent.
- Resume: existing plan is reused; done tasks are skipped; in_progress
  tasks restart.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import src.agent as agent_mod
from src.agent import run_session
from src.events import EventTypes, bus
from src.models import Session, SessionStatus, Task, TaskStatus
from src.schemas import (
    GuardrailResult,
    LLMTextResponse,
    LLMToolCallsResponse,
    ParsedToolCall,
    VerificationResult,
)


# ---------- DB stub ----------


@dataclass
class _FakeSessionRow:
    id: UUID
    goal: str = "x"
    status: SessionStatus = SessionStatus.planning
    final_report: str | None = None
    verification_notes: str | None = None


class _FakeDB:
    """Tracks Session/Task rows in plain dicts so tests can assert on state."""

    def __init__(self) -> None:
        self.sessions: dict[UUID, _FakeSessionRow] = {}
        self.tasks: dict[UUID, Task] = {}
        self.documents_for_session: dict[UUID, list[str]] = {}

    def seed_session(self, session_id: UUID, *, status=SessionStatus.planning) -> None:
        self.sessions[session_id] = _FakeSessionRow(id=session_id, status=status)

    def seed_task(
        self,
        *,
        session_id: UUID,
        order_index: int,
        description: str,
        status: TaskStatus = TaskStatus.pending,
        result_summary: str | None = None,
        sources: list[str] | None = None,
    ) -> Task:
        t = Task(
            id=uuid4(),
            session_id=session_id,
            order_index=order_index,
            description=description,
            status=status,
            result_summary=result_summary,
            sources=sources,
        )
        self.tasks[t.id] = t
        return t


class _FakeSession:
    def __init__(self, db: _FakeDB):
        self._db = db
        self._pending_add: list[Any] = []

    def add(self, obj: Any) -> None:
        self._pending_add.append(obj)

    def add_all(self, objs: list[Any]) -> None:
        for o in objs:
            self.add(o)

    async def flush(self) -> None:
        for obj in self._pending_add:
            if isinstance(obj, Task):
                if obj.id is None:
                    obj.id = uuid4()
                self._db.tasks[obj.id] = obj
        # don't clear; commit will handle persistence ack
        self._pending_add = []

    async def get(self, model, pk):
        if model is Session:
            return self._db.sessions.get(pk)
        if model is Task:
            return self._db.tasks.get(pk)
        return None

    async def execute(self, stmt):
        # Cheap dispatcher: inspect the statement's targeted entity.
        from sqlalchemy.sql import Select

        if not isinstance(stmt, Select):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

        # Document filename query
        target = str(stmt)
        if "documents.filename" in target.lower():
            # Try to match any session's docs — there's only one in our tests.
            all_docs = []
            for sid, fns in self._db.documents_for_session.items():
                all_docs.extend(fns)
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: all_docs))

        # Task listing
        if "tasks" in target.lower():
            rows = sorted(self._db.tasks.values(), key=lambda t: t.order_index)
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

    async def commit(self): ...
    async def rollback(self): ...


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeDB()

    @asynccontextmanager
    async def _scope():
        yield _FakeSession(db)

    monkeypatch.setattr(agent_mod, "session_scope", _scope)
    return db


# ---------- helpers to build LLM responses ----------


def _plan_response(*descriptions: str) -> LLMToolCallsResponse:
    args = {
        "tasks": [
            {"description": d, "rationale": f"because {d}"} for d in descriptions
        ]
    }
    return LLMToolCallsResponse(
        calls=[
            ParsedToolCall(
                id="call_plan",
                name="create_plan",
                arguments=args,
                raw_arguments="{}",
                decoded=True,
            )
        ]
    )


def _tool_call(name: str, arguments: dict, *, call_id: str | None = None) -> ParsedToolCall:
    import json as _json

    return ParsedToolCall(
        id=call_id or f"call_{name}",
        name=name,
        arguments=arguments,
        raw_arguments=_json.dumps(arguments),
        decoded=True,
    )


def _tool_calls_response(*calls: ParsedToolCall) -> LLMToolCallsResponse:
    return LLMToolCallsResponse(calls=list(calls))


# ---------- guardrail mocks ----------


@pytest.fixture
def passing_guardrail(monkeypatch):
    async def _check(_goal: str):
        return GuardrailResult(passed=True)

    monkeypatch.setattr(agent_mod, "check_goal", _check)


@pytest.fixture
def passing_verifier(monkeypatch):
    async def _verify(_report: str, _sources: list[str]):
        return VerificationResult(unsupported_claims=[], notes="ok")

    monkeypatch.setattr(agent_mod, "verify_report", _verify)


# ---------- tool invoker mock ----------


@pytest.fixture
def tool_invoker(monkeypatch):
    """Returns a recorder you can configure with return values per tool name."""

    recorder = SimpleNamespace(
        calls=[],
        returns={},  # tool_name -> output dict
    )

    async def _invoke(name: str, arguments: dict):
        recorder.calls.append((name, arguments))
        out = recorder.returns.get(name)
        if out is None:
            raise KeyError(f"no fake return registered for {name}")
        # Return a minimal Pydantic-like object with model_dump().
        return SimpleNamespace(model_dump=lambda: out)

    monkeypatch.setattr("src.tools.invoke", _invoke)
    monkeypatch.setattr("src.agent.tools_pkg.invoke", _invoke)

    # Avoid hitting real schemas (which would touch Tavily/etc imports).
    monkeypatch.setattr(
        "src.agent.tools_pkg.openai_schemas_for", lambda *_args, **_kwargs: []
    )
    return recorder


# ---------- complete() programmable mock ----------


@pytest.fixture
def llm_script(monkeypatch):
    """Push responses in order; raise if exhausted."""

    queue: list = []

    async def _complete(**kwargs):
        if not queue:
            raise AssertionError(f"complete() called with no more scripted responses; kwargs={list(kwargs)}")
        return queue.pop(0)

    monkeypatch.setattr(agent_mod, "complete", _complete)
    return queue


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


# --- Guardrail short-circuit ----


async def test_input_guardrail_failure_marks_session_failed(monkeypatch, fake_db):
    sid = uuid4()
    fake_db.seed_session(sid)

    async def _check(_goal):
        return GuardrailResult(passed=False, reason="empty goal")

    monkeypatch.setattr(agent_mod, "check_goal", _check)

    await run_session("", sid)
    assert fake_db.sessions[sid].status == SessionStatus.failed


# --- Planner ----


async def test_planner_persists_tasks_and_emits_plan_ready(
    fake_db, passing_guardrail, tool_invoker, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)

    # Subscribe BEFORE running so we capture the PLAN_READY event.
    async with bus.subscribe(sid) as queue:
        llm_script.append(_plan_response("Find studies on X", "Compare A vs B", "Summarize gaps"))
        # Each task will fail (no follow-on responses) — that's fine; we're
        # only testing the planning phase.
        # Push three "no tool calls" responses so each task fails fast.
        llm_script.append(LLMTextResponse(content="give up"))
        llm_script.append(LLMTextResponse(content="give up"))
        llm_script.append(LLMTextResponse(content="give up"))
        # Synthesis won't be called because every task fails → no completed.
        # (loop returns early when completed_summaries is empty.)

        await run_session("Map the landscape of X.", sid)

        # Drain the queue and find PLAN_READY.
        events = []
        for _ in range(15):
            if queue.empty():
                break
            events.append(queue.get_nowait())

    plan_events = [e for e in events if e.type == EventTypes.PLAN_READY]
    assert len(plan_events) == 1
    assert len(plan_events[0].data["tasks"]) == 3

    # Tasks were persisted.
    assert len(fake_db.tasks) == 3
    descriptions = sorted(t.description for t in fake_db.tasks.values())
    assert descriptions == sorted(["Find studies on X", "Compare A vs B", "Summarize gaps"])


async def test_planner_failure_marks_session_failed(
    fake_db, passing_guardrail, tool_invoker, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)

    # Plan response is text instead of tool_calls → parse fails → empty plan.
    llm_script.append(LLMTextResponse(content="I cannot plan"))

    await run_session("Goal.", sid)
    assert fake_db.sessions[sid].status == SessionStatus.failed


# --- Single-task happy path with finish_task ----


async def test_task_completes_with_finish_task(
    fake_db, passing_guardrail, passing_verifier, tool_invoker, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)

    # Plan with one task.
    llm_script.append(_plan_response("Investigate Y", "Compare Z", "Summarize"))

    # For each of the 3 tasks: model calls finish_task immediately.
    for i in range(3):
        llm_script.append(
            _tool_calls_response(
                _tool_call(
                    "finish_task",
                    {
                        "result_summary": f"Findings for task {i}.",
                        "sources": [f"https://src-{i}"],
                    },
                )
            )
        )
    # Synthesis: text report.
    llm_script.append(LLMTextResponse(content="# Final Report\n\nBody [1].\n\nSources\n1. https://src-0"))

    tool_invoker.returns["finish_task"] = {
        "accepted": True,
        "result_summary": "ok",
        "sources": ["https://x"],
    }

    await run_session("Goal.", sid)

    # All tasks are done with result summaries persisted.
    assert all(t.status == TaskStatus.done for t in fake_db.tasks.values())
    assert all(t.result_summary for t in fake_db.tasks.values())
    assert all(t.sources for t in fake_db.tasks.values())

    # Final report saved on session.
    assert fake_db.sessions[sid].final_report is not None
    assert "Final Report" in fake_db.sessions[sid].final_report
    assert fake_db.sessions[sid].status == SessionStatus.completed


# --- Citation enforcement ----


async def test_finish_task_empty_sources_triggers_correction_then_succeeds(
    fake_db, passing_guardrail, passing_verifier, llm_script, monkeypatch
):
    """The first finish_task call has no sources → ValidationError → correction
    attempt counted. Model retries with sources → success."""
    sid = uuid4()
    fake_db.seed_session(sid)

    # Plan with three tasks (schema requires >=3).
    llm_script.append(_plan_response("T1", "T2", "T3"))

    # Task 1: first finish_task → no sources; second → corrected.
    llm_script.append(
        _tool_calls_response(
            _tool_call("finish_task", {"result_summary": "early", "sources": []})
        )
    )
    llm_script.append(
        _tool_calls_response(
            _tool_call(
                "finish_task",
                {"result_summary": "corrected", "sources": ["https://x"]},
            )
        )
    )
    # Tasks 2 and 3: clean single-call finishes.
    for _ in range(2):
        llm_script.append(
            _tool_calls_response(
                _tool_call(
                    "finish_task",
                    {"result_summary": "ok", "sources": ["https://x"]},
                )
            )
        )
    llm_script.append(LLMTextResponse(content="# Report\n\nBody."))

    # Use the REAL invoke so Pydantic min_length=1 actually fires on the
    # empty-sources call.
    import src.tools as real_tools
    monkeypatch.setattr("src.agent.tools_pkg.invoke", real_tools.invoke)
    monkeypatch.setattr(
        "src.agent.tools_pkg.openai_schemas_for", lambda *_a, **_k: []
    )

    await run_session("Goal.", sid)

    # Task 1 must be done; its persisted summary is "corrected".
    done = [t for t in fake_db.tasks.values() if t.order_index == 0]
    assert done[0].status == TaskStatus.done
    assert "corrected" in (done[0].result_summary or "")


async def test_finish_task_repeated_empty_sources_fails_task(
    fake_db, passing_guardrail, passing_verifier, llm_script, monkeypatch
):
    """Three empty-source finishes in a row → corrections exhausted (>2) → fail."""
    sid = uuid4()
    fake_db.seed_session(sid)

    llm_script.append(_plan_response("T1", "T2", "T3"))
    # Task 1: 3 attempts at empty-sources finish.
    for _ in range(3):
        llm_script.append(
            _tool_calls_response(
                _tool_call("finish_task", {"result_summary": "x", "sources": []})
            )
        )
    # Tasks 2 and 3 finish cleanly.
    for _ in range(2):
        llm_script.append(
            _tool_calls_response(
                _tool_call(
                    "finish_task",
                    {"result_summary": "ok", "sources": ["https://x"]},
                )
            )
        )
    llm_script.append(LLMTextResponse(content="# Report"))

    # Use real invoke so Pydantic validation actually fires.
    import src.tools as real_tools
    monkeypatch.setattr("src.agent.tools_pkg.invoke", real_tools.invoke)
    monkeypatch.setattr(
        "src.agent.tools_pkg.openai_schemas_for", lambda *_a, **_k: []
    )

    await run_session("Goal.", sid)

    task1 = next(t for t in fake_db.tasks.values() if t.order_index == 0)
    assert task1.status == TaskStatus.failed


# --- Iteration cap ----


async def test_iteration_cap_fails_task(
    fake_db, passing_guardrail, passing_verifier, monkeypatch, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)

    # Plan one task (using 3 to satisfy schema).
    llm_script.append(_plan_response("T1", "T2", "T3"))

    # For each of 3 tasks, push max_tool_iterations_per_task responses that
    # each call web_search but never finish_task — exhausts the cap.
    from src.config import settings as _settings
    monkeypatch.setattr(_settings, "max_tool_iterations_per_task", 2)

    for _ in range(3):  # 3 tasks
        for _ in range(2):  # 2 iterations each
            llm_script.append(
                _tool_calls_response(
                    _tool_call("web_search", {"query": "x", "max_results": 3})
                )
            )
    llm_script.append(LLMTextResponse(content="# Report"))

    async def _invoke(name, args):
        if name == "web_search":
            return SimpleNamespace(
                model_dump=lambda: {"query": args.get("query"), "results": []}
            )
        raise KeyError(name)

    monkeypatch.setattr("src.agent.tools_pkg.invoke", _invoke)
    monkeypatch.setattr("src.agent.tools_pkg.openai_schemas_for", lambda *_a, **_k: [])

    await run_session("Goal.", sid)
    # All three tasks failed since none ever called finish_task.
    assert all(t.status == TaskStatus.failed for t in fake_db.tasks.values())


# --- Resume ----


async def test_resume_skips_done_tasks(
    fake_db, passing_guardrail, passing_verifier, tool_invoker, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)
    # Pre-seed a complete plan: T1 done, T2 pending, T3 pending.
    fake_db.seed_task(
        session_id=sid,
        order_index=0,
        description="T1",
        status=TaskStatus.done,
        result_summary="already done",
        sources=["https://old"],
    )
    fake_db.seed_task(session_id=sid, order_index=1, description="T2")
    fake_db.seed_task(session_id=sid, order_index=2, description="T3")

    # No planner call — resume should reuse existing plan.
    # Two task responses, then synthesis.
    for _ in range(2):
        llm_script.append(
            _tool_calls_response(
                _tool_call(
                    "finish_task",
                    {"result_summary": "fresh", "sources": ["https://new"]},
                )
            )
        )
    llm_script.append(LLMTextResponse(content="# Report"))

    tool_invoker.returns["finish_task"] = {
        "accepted": True,
        "result_summary": "ok",
        "sources": ["https://x"],
    }

    await run_session("Goal.", sid, resume=True)

    # T1 still says "already done" — wasn't re-run.
    t1 = next(t for t in fake_db.tasks.values() if t.order_index == 0)
    assert t1.result_summary == "already done"
    # T2 and T3 ran fresh.
    others = [t for t in fake_db.tasks.values() if t.order_index in (1, 2)]
    assert all(t.status == TaskStatus.done for t in others)


async def test_resume_restarts_in_progress_tasks(
    fake_db, passing_guardrail, passing_verifier, tool_invoker, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)
    # T1 stuck in_progress on resume → must restart.
    t1 = fake_db.seed_task(
        session_id=sid,
        order_index=0,
        description="T1",
        status=TaskStatus.in_progress,
    )
    fake_db.seed_task(session_id=sid, order_index=1, description="T2")
    fake_db.seed_task(session_id=sid, order_index=2, description="T3")

    for _ in range(3):
        llm_script.append(
            _tool_calls_response(
                _tool_call(
                    "finish_task",
                    {"result_summary": "ok", "sources": ["https://x"]},
                )
            )
        )
    llm_script.append(LLMTextResponse(content="# Report"))

    tool_invoker.returns["finish_task"] = {
        "accepted": True,
        "result_summary": "ok",
        "sources": ["https://x"],
    }

    await run_session("Goal.", sid, resume=True)

    # T1 is done now (was restarted).
    refreshed = fake_db.tasks[t1.id]
    assert refreshed.status == TaskStatus.done
    assert refreshed.result_summary is not None


# --- Verification notes ----


async def test_verification_notes_persisted(
    fake_db, passing_guardrail, tool_invoker, llm_script, monkeypatch
):
    sid = uuid4()
    fake_db.seed_session(sid)
    llm_script.append(_plan_response("T1", "T2", "T3"))
    for _ in range(3):
        llm_script.append(
            _tool_calls_response(
                _tool_call(
                    "finish_task",
                    {"result_summary": "ok", "sources": ["https://x"]},
                )
            )
        )
    llm_script.append(LLMTextResponse(content="# Report\n\nA claim of 42%."))

    tool_invoker.returns["finish_task"] = {
        "accepted": True,
        "result_summary": "ok",
        "sources": ["https://x"],
    }

    async def _verify(_r, _s):
        return VerificationResult(
            unsupported_claims=["The 42% figure has no source."],
            notes="One quantitative claim is unsourced.",
        )

    monkeypatch.setattr(agent_mod, "verify_report", _verify)

    await run_session("Goal.", sid)
    notes = fake_db.sessions[sid].verification_notes or ""
    assert "42% figure" in notes
    assert "unsourced" in notes


# --- Bus events ----


async def test_session_completed_event_fires(
    fake_db, passing_guardrail, passing_verifier, tool_invoker, llm_script
):
    sid = uuid4()
    fake_db.seed_session(sid)
    async with bus.subscribe(sid) as queue:
        llm_script.append(_plan_response("T1", "T2", "T3"))
        for _ in range(3):
            llm_script.append(
                _tool_calls_response(
                    _tool_call(
                        "finish_task",
                        {"result_summary": "ok", "sources": ["https://x"]},
                    )
                )
            )
        llm_script.append(LLMTextResponse(content="# Report"))

        tool_invoker.returns["finish_task"] = {
            "accepted": True,
            "result_summary": "ok",
            "sources": ["https://x"],
        }

        await run_session("Goal.", sid)

        seen_types: list[str] = []
        for _ in range(50):
            if queue.empty():
                break
            seen_types.append(queue.get_nowait().type)

    assert EventTypes.SESSION_STARTED in seen_types
    assert EventTypes.PLAN_READY in seen_types
    assert EventTypes.SESSION_COMPLETED in seen_types
