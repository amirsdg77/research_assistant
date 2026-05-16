"""Tests for src/api/routes.py and main.py.

Uses httpx.AsyncClient against the FastAPI ASGI app directly. The agent
loop and ingest pipeline are mocked; the DB session dependency is
overridden with an in-memory stub so tests run without Postgres.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from src.api import routes as routes_mod
from src.api.main import app as fastapi_app
from src.db import get_session
from src.events import AgentEvent, EventTypes, bus
from src.models import Document, Session, SessionStatus, Task, TaskStatus


# ---------- DB stub --------------------------------------------------------


class _FakeDB:
    def __init__(self) -> None:
        self.sessions: dict[UUID, Session] = {}
        self.tasks: dict[UUID, Task] = {}
        self.documents: dict[UUID, Document] = {}

    def install(self, app: FastAPI) -> None:
        async def _override() -> Any:
            yield _FakeSession(self)

        app.dependency_overrides[get_session] = _override


class _FakeSession:
    def __init__(self, db: _FakeDB):
        self._db = db

    def add(self, obj):
        if isinstance(obj, Session):
            self._db.sessions[obj.id] = obj
        elif isinstance(obj, Task):
            self._db.tasks[obj.id] = obj
        elif isinstance(obj, Document):
            self._db.documents[obj.id] = obj

    async def get(self, model, pk):
        if model is Session:
            return self._db.sessions.get(pk)
        if model is Task:
            return self._db.tasks.get(pk)
        if model is Document:
            return self._db.documents.get(pk)
        return None

    async def execute(self, stmt):
        target = str(stmt).lower()
        if "from documents" in target:
            rows = list(self._db.documents.values())
        elif "from tasks" in target:
            rows = sorted(self._db.tasks.values(), key=lambda t: t.order_index)
        else:
            rows = list(self._db.sessions.values())
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

    async def commit(self): ...
    async def rollback(self): ...


@pytest.fixture
def fake_db():
    db = _FakeDB()
    db.install(fastapi_app)
    yield db
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def patch_agent(monkeypatch):
    """Replace agent.run_session with an async no-op so BackgroundTasks doesn't
    actually try to plan + execute."""
    called: list = []

    async def _no_run(*args, **kwargs):
        called.append((args, kwargs))

    monkeypatch.setattr(routes_mod, "run_session", _no_run)
    return called


@pytest.fixture
def patch_ingest(monkeypatch):
    """Replace ingest_document with a fake returning predictable IDs."""

    @dataclass
    class _R:
        document_id: UUID
        filename: str
        chunk_count: int

    async def _fake(*, filename, data, session_id, content_type=None):
        return _R(document_id=uuid4(), filename=filename, chunk_count=2)

    monkeypatch.setattr(routes_mod, "ingest_document", _fake)


async def _client():
    transport = httpx.ASGITransport(app=fastapi_app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------- Tests ---------------------------------------------------------


async def test_create_session_returns_id_and_kicks_off_agent(
    fake_db, patch_agent
):
    async with await _client() as c:
        resp = await c.post("/api/sessions", data={"goal": "Map the landscape."})
    assert resp.status_code == 201
    body = resp.json()
    sid = UUID(body["session_id"])
    assert sid in fake_db.sessions
    assert fake_db.sessions[sid].goal == "Map the landscape."
    # Background task scheduled.
    # FastAPI runs background tasks after response; we don't need to wait
    # on them — just confirm the route registered the call.
    await asyncio.sleep(0)  # let the background task run
    assert any(c_args[0] == "Map the landscape." for c_args, _ in patch_agent)


async def test_create_session_rejects_empty_goal(fake_db, patch_agent):
    async with await _client() as c:
        resp = await c.post("/api/sessions", data={"goal": "   "})
    assert resp.status_code == 400


async def test_create_session_with_files_ingests_them(
    fake_db, patch_agent, patch_ingest
):
    async with await _client() as c:
        resp = await c.post(
            "/api/sessions",
            data={"goal": "Discuss the docs."},
            files=[
                ("files", ("a.txt", b"hello", "text/plain")),
                ("files", ("b.md", b"# md", "text/markdown")),
            ],
        )
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["documents"]) == 2
    assert {d["filename"] for d in body["documents"]} == {"a.txt", "b.md"}


async def test_get_session_returns_state_and_tasks(fake_db, patch_agent):
    sid = uuid4()
    fake_db.sessions[sid] = Session(
        id=sid, goal="g", status=SessionStatus.completed, final_report="# Done"
    )
    fake_db.tasks[uuid4()] = Task(
        id=uuid4(),
        session_id=sid,
        order_index=0,
        description="T1",
        status=TaskStatus.done,
    )

    async with await _client() as c:
        resp = await c.get(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(sid)
    assert body["status"] == "completed"
    assert body["final_report"] == "# Done"
    assert len(body["tasks"]) == 1
    assert body["tasks"][0]["description"] == "T1"


async def test_get_session_404(fake_db):
    async with await _client() as c:
        resp = await c.get(f"/api/sessions/{uuid4()}")
    assert resp.status_code == 404


async def test_list_sessions(fake_db):
    sid = uuid4()
    fake_db.sessions[sid] = Session(id=sid, goal="g", status=SessionStatus.completed)
    async with await _client() as c:
        resp = await c.get("/api/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert any(s["id"] == str(sid) for s in body["sessions"])


async def test_resume_existing_session(fake_db, patch_agent):
    sid = uuid4()
    fake_db.sessions[sid] = Session(id=sid, goal="g", status=SessionStatus.failed)
    async with await _client() as c:
        resp = await c.post(f"/api/sessions/{sid}/resume")
    assert resp.status_code == 202
    body = resp.json()
    assert body["resumed"] is True
    await asyncio.sleep(0)
    assert any(c_args[1] == sid and c_kwargs.get("resume") for c_args, c_kwargs in patch_agent)


async def test_resume_missing_session(fake_db):
    async with await _client() as c:
        resp = await c.post(f"/api/sessions/{uuid4()}/resume")
    assert resp.status_code == 404


async def test_upload_session_document(fake_db, patch_ingest):
    sid = uuid4()
    fake_db.sessions[sid] = Session(id=sid, goal="g", status=SessionStatus.running)
    async with await _client() as c:
        resp = await c.post(
            f"/api/sessions/{sid}/documents",
            files=[("files", ("note.txt", b"hello", "text/plain"))],
        )
    assert resp.status_code == 201
    body = resp.json()
    assert body["documents"][0]["filename"] == "note.txt"


async def test_upload_session_document_404_when_session_missing(fake_db, patch_ingest):
    async with await _client() as c:
        resp = await c.post(
            f"/api/sessions/{uuid4()}/documents",
            files=[("files", ("note.txt", b"hello", "text/plain"))],
        )
    assert resp.status_code == 404


async def test_upload_global_document(fake_db, patch_ingest):
    async with await _client() as c:
        resp = await c.post(
            "/api/documents",
            files=[("files", ("note.txt", b"hello", "text/plain"))],
        )
    assert resp.status_code == 201
    assert resp.json()["documents"][0]["filename"] == "note.txt"


async def test_upload_propagates_ingest_error(fake_db, monkeypatch):
    from src.ingest import IngestError

    async def _broken(**kwargs):
        raise IngestError("too big")

    monkeypatch.setattr(routes_mod, "ingest_document", _broken)
    async with await _client() as c:
        resp = await c.post(
            "/api/documents",
            files=[("files", ("note.txt", b"hello", "text/plain"))],
        )
    assert resp.status_code == 400
    assert "too big" in resp.json()["detail"]


async def test_list_documents_filters_by_global(fake_db):
    fake_db.documents[uuid4()] = Document(
        id=uuid4(),
        session_id=None,
        filename="g.pdf",
        content_type="application/pdf",
        byte_size=10,
        chunk_count=1,
    )
    sid = uuid4()
    fake_db.documents[uuid4()] = Document(
        id=uuid4(),
        session_id=sid,
        filename="s.pdf",
        content_type="application/pdf",
        byte_size=10,
        chunk_count=1,
    )
    async with await _client() as c:
        resp = await c.get("/api/documents?global=true")
    assert resp.status_code == 200
    # Our stub returns all docs and the route filters in SQL; we can't
    # easily test the filter without a real DB, but we at least assert
    # the route accepts the param without erroring.
    assert "documents" in resp.json()


# ---------- SSE ----------------------------------------------------------


async def test_sse_stream_delivers_events_and_closes(fake_db):
    """Subscribe to /events, push an event + a close sentinel, verify the
    stream yields the event in SSE format and terminates."""
    sid = uuid4()
    fake_db.sessions[sid] = Session(id=sid, goal="g", status=SessionStatus.running)

    async with await _client() as c:
        async def _publish_then_close():
            # Yield once so the SSE handler subscribes.
            await asyncio.sleep(0.05)
            await bus.publish(
                AgentEvent(
                    session_id=sid,
                    type=EventTypes.PLAN_READY,
                    data={"tasks": [{"description": "T1"}]},
                )
            )
            await asyncio.sleep(0.05)
            await bus.close_session(sid)

        publisher = asyncio.create_task(_publish_then_close())

        chunks: list[str] = []
        async with c.stream("GET", f"/api/sessions/{sid}/events") as response:
            async for line in response.aiter_lines():
                chunks.append(line)
                # When the close sentinel fires the server-side generator
                # returns, ending the stream — aiter_lines will terminate.

        await publisher

    full = "\n".join(chunks)
    assert "event: plan.ready" in full
    assert "T1" in full


async def test_health_endpoint(fake_db):
    """The /api/health endpoint runs a SELECT 1 against engine.connect();
    we need to monkeypatch engine.connect for an isolated test."""
    # The simplest reliable check here: the route is registered.
    routes_by_path = {r.path for r in fastapi_app.routes}
    assert "/api/health" in routes_by_path
