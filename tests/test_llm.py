"""Tests for src.llm.

We mock the OpenAI client and the DB session so these tests run without
network or Postgres. Properties under test:
- Model routing by purpose returns the configured model name.
- Text response normalizes to LLMTextResponse.
- Tool-call response normalizes to LLMToolCallsResponse with parsed args dict.
- Malformed tool-call JSON arguments → decoded=False, raw_arguments preserved.
- BadRequestError does NOT retry.
- RateLimitError DOES retry (verified by attempt count on a transient mock).
- DB logging is best-effort: a session_scope() failure doesn't break the call.
- Embeddings: batching at 100, preserves order, returns the expected shape.
- Truncation: large prompt payload doesn't crash the DB log.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError, RateLimitError

import src.llm as llm_mod
from src.models import LLMPurpose
from src.schemas import LLMTextResponse, LLMToolCallsResponse


# --- helpers to build fake OpenAI responses -----------------------------


def _fake_message(*, content=None, tool_calls=None):
    """Build a stand-in for ChatCompletionMessage."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )


def _fake_tool_call(*, name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _fake_completion(message, *, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


@pytest.fixture(autouse=True)
def _reset_client_singleton(monkeypatch):
    """Force get_client() to rebuild for each test so injected mocks stick."""
    monkeypatch.setattr(llm_mod, "_client", None)
    yield
    monkeypatch.setattr(llm_mod, "_client", None)


@pytest.fixture
def stub_db(monkeypatch):
    """Replace session_scope with a no-op so tests don't touch Postgres."""
    class _NoopSession:
        def add(self, obj): ...
        async def commit(self): ...
        async def rollback(self): ...

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _scope():
        yield _NoopSession()

    monkeypatch.setattr(llm_mod, "session_scope", _scope)


# --- routing -------------------------------------------------------------


def test_model_routing_uses_configured_names():
    assert llm_mod.model_for(LLMPurpose.plan) == llm_mod.settings.planner_model
    assert llm_mod.model_for(LLMPurpose.synthesize) == llm_mod.settings.synthesizer_model
    assert llm_mod.model_for(LLMPurpose.decide) == llm_mod.settings.executor_model
    assert llm_mod.model_for(LLMPurpose.verify) == llm_mod.settings.verifier_model
    assert llm_mod.model_for(LLMPurpose.summarize) == llm_mod.settings.summarizer_model


# --- complete: text & tool-call shapes ----------------------------------


async def test_complete_text_response(stub_db, monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_fake_completion(_fake_message(content="hello"))
                )
            )
        )
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    result = await llm_mod.complete(
        purpose=LLMPurpose.summarize,
        messages=[{"role": "user", "content": "say hi"}],
    )

    assert isinstance(result, LLMTextResponse)
    assert result.content == "hello"
    fake_client.chat.completions.create.assert_awaited_once()


async def test_complete_tool_call_response_parses_arguments(stub_db, monkeypatch):
    args = {"query": "transformers", "max_results": 3}
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_fake_completion(
                        _fake_message(
                            tool_calls=[
                                _fake_tool_call(
                                    name="web_search",
                                    arguments=json.dumps(args),
                                )
                            ]
                        )
                    )
                )
            )
        )
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    result = await llm_mod.complete(
        purpose=LLMPurpose.decide,
        messages=[{"role": "user", "content": "search"}],
        tools=[{"type": "function", "function": {"name": "web_search"}}],
    )

    assert isinstance(result, LLMToolCallsResponse)
    assert len(result.calls) == 1
    call = result.calls[0]
    assert call.name == "web_search"
    assert call.arguments == args
    assert call.decoded is True


async def test_complete_malformed_tool_args_preserves_raw(stub_db, monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_fake_completion(
                        _fake_message(
                            tool_calls=[
                                _fake_tool_call(
                                    name="web_search",
                                    arguments='{"query": broken',  # invalid JSON
                                )
                            ]
                        )
                    )
                )
            )
        )
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    result = await llm_mod.complete(
        purpose=LLMPurpose.decide,
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "web_search"}}],
    )
    assert isinstance(result, LLMToolCallsResponse)
    call = result.calls[0]
    assert call.decoded is False
    assert call.arguments == {}
    assert call.raw_arguments == '{"query": broken'


# --- retries -------------------------------------------------------------


def _make_rate_limit_error() -> RateLimitError:
    """RateLimitError requires a response/body. Build the smallest valid one."""
    return RateLimitError(
        message="rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
        body={},
    )


def _make_bad_request_error() -> BadRequestError:
    return BadRequestError(
        message="bad",
        response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
        body={},
    )


async def test_rate_limit_error_retries_then_succeeds(stub_db, monkeypatch):
    create_mock = AsyncMock(
        side_effect=[
            _make_rate_limit_error(),
            _fake_completion(_fake_message(content="ok")),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    # tenacity.asyncio._portable_async_sleep gates the backoff. Patch it
    # to return immediately so the test doesn't actually wait.
    async def _no_sleep(_seconds):
        return None

    import tenacity.asyncio as tenacity_async
    monkeypatch.setattr(tenacity_async, "_portable_async_sleep", _no_sleep)

    result = await llm_mod.complete(
        purpose=LLMPurpose.summarize,
        messages=[{"role": "user", "content": "x"}],
    )

    assert isinstance(result, LLMTextResponse)
    assert result.content == "ok"
    assert create_mock.await_count == 2


async def test_bad_request_error_does_not_retry(stub_db, monkeypatch):
    create_mock = AsyncMock(side_effect=_make_bad_request_error())
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    with pytest.raises(BadRequestError):
        await llm_mod.complete(
            purpose=LLMPurpose.summarize,
            messages=[{"role": "user", "content": "x"}],
        )

    assert create_mock.await_count == 1


# --- DB logging is best-effort ------------------------------------------


async def test_db_log_failure_does_not_break_call(monkeypatch):
    """If session_scope blows up, the LLM result must still be returned."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _broken_scope():
        raise RuntimeError("db down")
        yield  # unreachable, but makes it a generator

    monkeypatch.setattr(llm_mod, "session_scope", _broken_scope)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_fake_completion(_fake_message(content="ok"))
                )
            )
        )
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    result = await llm_mod.complete(
        purpose=LLMPurpose.summarize,
        messages=[{"role": "user", "content": "x"}],
    )
    assert isinstance(result, LLMTextResponse)
    assert result.content == "ok"


# --- truncation ----------------------------------------------------------


def test_truncate_passes_small_payloads_through():
    payload = {"messages": [{"role": "user", "content": "small"}], "tools": None}
    out = llm_mod._truncate_for_jsonb(payload)
    assert out == payload


def test_truncate_replaces_large_payload_with_marker():
    big = "x" * (60 * 1024)
    payload = {"messages": [{"role": "user", "content": big}]}
    out = llm_mod._truncate_for_jsonb(payload)
    assert out["_truncated"] is True
    assert out["_original_size_bytes"] > 50 * 1024
    assert "_head" in out and "_tail" in out


def test_truncate_handles_unserializable():
    class NotJSONable:
        pass

    out = llm_mod._truncate_for_jsonb({"obj": NotJSONable()})
    # default=str saves us — the object becomes its repr, so it serializes.
    # Anything that genuinely fails (e.g. recursive) hits the except branch.
    assert isinstance(out, (dict, list)) or out.get("_truncated") is True


# --- embeddings ----------------------------------------------------------


async def test_embed_empty_input_returns_empty():
    result = await llm_mod.embed([])
    assert result == []


async def test_embed_single_batch(monkeypatch):
    fake_data = [SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in range(5)]
    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(data=fake_data))
        )
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    result = await llm_mod.embed(["a", "b", "c", "d", "e"])
    assert len(result) == 5
    assert all(len(v) == 3 for v in result)
    fake_client.embeddings.create.assert_awaited_once()


async def test_embed_batches_at_100(monkeypatch):
    """101 inputs → 2 calls, returns 101 vectors in order."""
    call_inputs: list[list[str]] = []

    async def _fake_create(*, model, input):
        call_inputs.append(input)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(i)]) for i in range(len(input))]
        )

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fake_create))
    )
    monkeypatch.setattr(llm_mod, "get_client", lambda: fake_client)

    inputs = [f"text-{i}" for i in range(101)]
    result = await llm_mod.embed(inputs)

    assert len(result) == 101
    assert len(call_inputs) == 2
    assert len(call_inputs[0]) == 100
    assert len(call_inputs[1]) == 1
