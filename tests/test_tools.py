"""Tests for src.tools.

Mocks Tavily, httpx, the LLM summarizer, and the memory store so tests run
without network or external services. Verifies the contracts the agent loop
depends on:
- Tool schemas render in OpenAI's function-call shape.
- finish_task requires non-empty sources (citation enforcement).
- fetch_url and search_memory require an active session_id.
- The registry dispatches by name and raises ValidationError on bad args.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

import src.tools as tools_pkg
from src import memory as memory_mod
from src.memory import InMemoryStubStore
from src.tools import invoke, openai_schemas_for
from src.tools.base import ToolError
from src.tools.fetch_url import current_session_id
from src.tools.finish_task import finish_task_spec
from src.tools.web_search import WebSearchInput


# --- schemas -------------------------------------------------------------


def test_all_tools_render_openai_schemas():
    schemas = openai_schemas_for()
    names = {s["function"]["name"] for s in schemas}
    assert names == {
        "web_search",
        "fetch_url",
        "search_memory",
        "search_documents",
        "finish_task",
    }
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]
        assert s["function"]["parameters"]["type"] == "object"


def test_finish_task_schema_requires_min_one_source():
    schema = finish_task_spec.openai_schema()["function"]["parameters"]
    sources = schema["properties"]["sources"]
    # Pydantic emits `minItems` for `min_length` on list fields.
    assert sources.get("minItems") == 1


# --- finish_task ---------------------------------------------------------


async def test_finish_task_accepts_valid_input():
    result = await invoke(
        "finish_task",
        {"result_summary": "Found three studies.", "sources": ["https://x"]},
    )
    assert result.accepted is True
    assert result.sources == ["https://x"]


async def test_finish_task_rejects_empty_sources():
    with pytest.raises(ValidationError):
        await invoke(
            "finish_task",
            {"result_summary": "Found three studies.", "sources": []},
        )


async def test_finish_task_rejects_missing_sources():
    with pytest.raises(ValidationError):
        await invoke("finish_task", {"result_summary": "Found three studies."})


# --- web_search ----------------------------------------------------------


async def test_web_search_calls_tavily_and_normalizes_results(monkeypatch):
    fake = SimpleNamespace(
        search=AsyncMock(
            return_value={
                "results": [
                    {"title": "T1", "url": "https://a", "content": "snippet a"},
                    {"title": "T2", "url": "https://b", "content": "snippet b"},
                ]
            }
        )
    )
    monkeypatch.setattr("src.tools.web_search._get_client", lambda: fake)

    out = await invoke("web_search", {"query": "transformers", "max_results": 2})
    assert len(out.results) == 2
    assert out.results[0].title == "T1"
    assert out.results[0].url == "https://a"
    fake.search.assert_awaited_once()


async def test_web_search_validates_max_results_upper_bound():
    with pytest.raises(ValidationError):
        WebSearchInput(query="x", max_results=99)


async def test_web_search_validates_empty_query():
    with pytest.raises(ValidationError):
        WebSearchInput(query="", max_results=3)


# --- fetch_url + search_memory require session_id -----------------------


async def test_fetch_url_without_session_raises():
    # No session bound in this contextvar → ToolError.
    assert current_session_id.get() is None
    with pytest.raises(ToolError):
        await invoke("fetch_url", {"url": "https://example.com"})


async def test_search_memory_without_session_raises():
    assert current_session_id.get() is None
    with pytest.raises(ToolError):
        await invoke("search_memory", {"query": "anything"})


# --- fetch_url happy path with mocked deps ------------------------------


async def test_fetch_url_end_to_end(monkeypatch):
    # Replace the global memory store with a fresh stub for isolation.
    fresh_store = InMemoryStubStore()
    monkeypatch.setattr(memory_mod, "store", fresh_store)

    async def _fake_get(url):
        return ("This is the main article body.", "Sample Title")

    monkeypatch.setattr("src.tools.fetch_url._http_get", _fake_get)

    async def _fake_summarize(text):
        return "Auto-summary of the article."

    monkeypatch.setattr("src.tools.fetch_url._summarize", _fake_summarize)

    sid = uuid4()
    token = current_session_id.set(sid)
    try:
        out = await invoke("fetch_url", {"url": "https://example.com/x"})
    finally:
        current_session_id.reset(token)

    assert out.url == "https://example.com/x"
    assert out.title == "Sample Title"
    assert out.summary == "Auto-summary of the article."
    assert out.memory_id  # uuid string
    # The page must have been written to the store under the session id.
    hits = await fresh_store.search_pages(session_id=sid, query="main article", k=5)
    assert len(hits) == 1
    assert hits[0].metadata["source_url"] == "https://example.com/x"


async def test_fetch_url_propagates_tool_error(monkeypatch):
    async def _broken_get(url):
        raise ToolError("network down")

    monkeypatch.setattr("src.tools.fetch_url._http_get", _broken_get)

    sid = uuid4()
    token = current_session_id.set(sid)
    try:
        with pytest.raises(ToolError):
            await invoke("fetch_url", {"url": "https://example.com"})
    finally:
        current_session_id.reset(token)


# --- search_memory hits the store --------------------------------------


async def test_search_memory_returns_hits_from_store(monkeypatch):
    fresh_store = InMemoryStubStore()
    monkeypatch.setattr(memory_mod, "store", fresh_store)

    sid = uuid4()
    await fresh_store.add_page(
        session_id=sid,
        url="https://a",
        title="A",
        text="transformers are great models",
    )

    token = current_session_id.set(sid)
    try:
        out = await invoke("search_memory", {"query": "transformers", "k": 3})
    finally:
        current_session_id.reset(token)

    assert len(out.hits) == 1
    assert out.hits[0].source_url == "https://a"
    assert "transformers" in out.hits[0].chunk.lower()


# --- registry ----------------------------------------------------------


async def test_invoke_unknown_tool_raises_keyerror():
    with pytest.raises(KeyError):
        await invoke("does_not_exist", {})


async def test_invoke_validation_error_surfaces():
    """A bad argument shape must raise ValidationError so the agent loop
    can return a tool-error message to the model for self-correction."""
    with pytest.raises(ValidationError):
        await invoke("web_search", {"max_results": 3})  # missing 'query'


def test_registry_lists_expected_tools():
    assert set(tools_pkg.all_tool_names()) == {
        "web_search",
        "fetch_url",
        "search_memory",
        "search_documents",
        "finish_task",
    }
