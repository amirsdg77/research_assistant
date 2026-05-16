"""Tests that the SPA static assets are served correctly.

These tests don't validate UI behavior (that needs a browser). They check:
- The static directory exists with all three expected files.
- The FastAPI app mounts /static and serves /index.html via the root route.
- styles.css and app.js are reachable.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.api.main import app as fastapi_app


STATIC = Path(__file__).resolve().parents[1] / "src" / "api" / "static"


def test_static_files_present():
    assert (STATIC / "index.html").exists()
    assert (STATIC / "styles.css").exists()
    assert (STATIC / "app.js").exists()


def test_index_html_references_expected_endpoints():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    # Critical wiring the SPA depends on.
    assert "/static/styles.css" in html
    assert "/static/app.js" in html
    # Composer, todo panel, activity, report — sanity checks on layout ids.
    for elem_id in [
        "goal-input", "run-button", "file-input", "file-chips",
        "task-list", "todo-count", "activity-feed",
        "report-body", "verification-block",
    ]:
        assert f'id="{elem_id}"' in html, f"missing element id: {elem_id}"


def test_app_js_handles_all_event_types():
    """Every event type emitted by the agent must be wired in the dispatcher,
    otherwise updates fail silently in the browser."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    expected = [
        "session.started",
        "plan.ready",
        "task.status_changed",
        "tool.invoked",
        "tool.completed",
        "tool.failed",
        "report.ready",
        "verification.ready",
        "session.completed",
        "session.failed",
        "guardrail.triggered",
    ]
    for t in expected:
        assert f'"{t}"' in js, f"event {t} not handled in app.js"


def test_styles_define_all_task_status_icons():
    """The TODO panel renders status icons via CSS classes; if a status name
    drifts the icon falls back to invisible. Pin the contract."""
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for status in ["pending", "in_progress", "done", "failed", "skipped"]:
        assert f".task-icon.{status}" in css, f"missing icon style for status: {status}"


async def test_root_serves_index_html():
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()
    assert "research-agent" in resp.text


async def test_static_assets_served():
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        css = await c.get("/static/styles.css")
        js = await c.get("/static/app.js")
    assert css.status_code == 200
    assert "--bg:" in css.text  # design token sanity
    assert js.status_code == 200
    assert "EventSource" in js.text  # SSE wiring sanity
