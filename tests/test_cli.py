"""Tests for src.cli.

Uses Typer's CliRunner. We mock the agent loop, the ingest pipeline, and
session_scope so commands exercise only the CLI's own surface: argument
parsing, output formatting, and dispatch.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from typer.testing import CliRunner

import src.cli as cli_mod


runner = CliRunner()


# ---------- Stubs ----------------------------------------------------------


@pytest.fixture
def quiet_bootstrap(monkeypatch):
    monkeypatch.setattr(cli_mod, "bootstrap_setup", lambda: None)


@pytest.fixture
def fake_db(monkeypatch):
    """Replace session_scope with an in-memory recorder that lets us read
    back what the CLI persisted/queried."""
    state: dict[str, Any] = {
        "sessions": {},
        "tasks": [],
        "documents": [],
        "added": [],
    }

    class _Stub:
        def add(self, obj):
            state["added"].append(obj)
            # If a Session, register it.
            if hasattr(obj, "goal") and hasattr(obj, "status"):
                state["sessions"][obj.id] = obj

        async def get(self, model, pk):
            return state["sessions"].get(pk)

        async def execute(self, stmt):
            target = str(stmt).lower()
            if "tasks" in target:
                rows = state["tasks"]
            elif "documents" in target:
                rows = state["documents"]
            else:
                rows = list(state["sessions"].values())
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

        async def commit(self): ...
        async def rollback(self): ...

    @asynccontextmanager
    async def _scope():
        yield _Stub()

    monkeypatch.setattr(cli_mod, "session_scope", _scope)
    return state


# ---------- Tests ---------------------------------------------------------


def test_help_lists_commands(quiet_bootstrap):
    result = runner.invoke(cli_mod.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["run", "resume", "list-sessions", "show", "ingest", "list-documents"]:
        assert cmd in result.output


def test_run_invokes_agent_with_new_session_id(monkeypatch, quiet_bootstrap, fake_db):
    captured: dict = {}

    async def _fake_run(goal, sid, *, resume):
        captured["goal"] = goal
        captured["sid"] = sid
        captured["resume"] = resume

    monkeypatch.setattr(cli_mod, "run_session", _fake_run)
    monkeypatch.setattr(cli_mod, "_stream_events", AsyncMock())

    result = runner.invoke(cli_mod.app, ["run", "Map the landscape."])
    assert result.exit_code == 0, result.output
    assert captured["goal"] == "Map the landscape."
    assert isinstance(captured["sid"], UUID)
    assert captured["resume"] is False
    assert str(captured["sid"]) in result.output


def test_run_reuses_provided_session_id(monkeypatch, quiet_bootstrap, fake_db):
    captured: dict = {}
    sid = uuid4()

    async def _fake_run(goal, s, *, resume):
        captured["sid"] = s
        captured["resume"] = resume

    monkeypatch.setattr(cli_mod, "run_session", _fake_run)
    monkeypatch.setattr(cli_mod, "_stream_events", AsyncMock())

    result = runner.invoke(cli_mod.app, ["run", "Goal", "--session", str(sid)])
    assert result.exit_code == 0
    assert captured["sid"] == sid


def test_resume_sets_resume_flag(monkeypatch, quiet_bootstrap, fake_db):
    captured: dict = {}
    sid = uuid4()

    async def _fake_run(goal, s, *, resume):
        captured["resume"] = resume
        captured["sid"] = s

    monkeypatch.setattr(cli_mod, "run_session", _fake_run)
    monkeypatch.setattr(cli_mod, "_stream_events", AsyncMock())

    result = runner.invoke(cli_mod.app, ["resume", str(sid)])
    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["sid"] == sid


def test_show_renders_session(monkeypatch, quiet_bootstrap, fake_db):
    sid = uuid4()
    fake_db["sessions"][sid] = SimpleNamespace(
        id=sid,
        goal="Survey X",
        status=SimpleNamespace(value="completed"),
        final_report="# Report\nBody.",
        verification_notes=None,
    )
    fake_db["tasks"] = [
        SimpleNamespace(
            order_index=0,
            description="T1",
            status=SimpleNamespace(value="done"),
            result_summary="found stuff",
            sources=["https://x"],
        )
    ]

    result = runner.invoke(cli_mod.app, ["show", str(sid)])
    assert result.exit_code == 0
    assert "Survey X" in result.output
    assert "T1" in result.output
    assert "Report" in result.output


def test_show_missing_session_errors(monkeypatch, quiet_bootstrap, fake_db):
    sid = uuid4()
    result = runner.invoke(cli_mod.app, ["show", str(sid)])
    assert result.exit_code == 1
    assert "no session" in result.output


def test_list_sessions_renders_table(monkeypatch, quiet_bootstrap, fake_db):
    sid = uuid4()
    fake_db["sessions"][sid] = SimpleNamespace(
        id=sid,
        goal="Survey X" * 20,  # longer than column truncation
        status=SimpleNamespace(value="completed"),
        created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
    )
    result = runner.invoke(cli_mod.app, ["list-sessions"])
    assert result.exit_code == 0
    assert "Sessions" in result.output
    assert "completed" in result.output


def test_ingest_calls_pipeline(monkeypatch, quiet_bootstrap, tmp_path):
    @dataclass
    class _R:
        document_id: UUID
        filename: str
        chunk_count: int

    async def _fake_ingest(*, filename, data, session_id, content_type=None):
        return _R(document_id=uuid4(), filename=filename, chunk_count=3)

    monkeypatch.setattr(cli_mod, "ingest_document", _fake_ingest)

    p = tmp_path / "doc.txt"
    p.write_bytes(b"hello")

    result = runner.invoke(cli_mod.app, ["ingest", str(p), "--global"])
    assert result.exit_code == 0, result.output
    assert "ingested" in result.output
    assert "doc.txt" in result.output
    assert "chunks=3" in result.output


def test_ingest_rejects_conflicting_scope(monkeypatch, quiet_bootstrap, tmp_path):
    """--session and --global together should fail with a clear message."""
    p = tmp_path / "doc.txt"
    p.write_bytes(b"hello")
    result = runner.invoke(
        cli_mod.app,
        ["ingest", str(p), "--session", str(uuid4()), "--global"],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_list_documents_renders_table(monkeypatch, quiet_bootstrap, fake_db):
    fake_db["documents"] = [
        SimpleNamespace(
            id=uuid4(),
            filename="paper.pdf",
            session_id=None,
            chunk_count=12,
            byte_size=4096,
            created_at=None,
        )
    ]
    result = runner.invoke(cli_mod.app, ["list-documents"])
    assert result.exit_code == 0
    assert "paper.pdf" in result.output
    assert "(global)" in result.output
