"""Single async wrapper around OpenAI's AsyncOpenAI client.

Why a wrapper:
- Centralizes model routing by `purpose` so the rest of the codebase never
  hard-codes a model name.
- Logs every call to the `llm_calls` table (tokens, latency, prompt+response
  JSONB). The agent loop, CLI, and tests all benefit from the trail.
- Retries on transient errors with tenacity. BadRequestError (e.g. malformed
  schema) is NOT retried — it'd just burn quota.
- Returns a normalized `LLMResponse` so callers don't pick apart
  `message.tool_calls[*].function.arguments` themselves.

Why model routing lives here:
- The plan calls for two-tier routing (gpt-4o for plan/synthesize, gpt-4o-mini
  for everything else). Putting the map next to the callsite means changing
  it is a one-line edit, and every call site already passes `purpose=` for
  the DB log anyway — no duplication.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import UUID

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.db import session_scope
from src.logging_setup import Events, get_logger
from src.models import LLMCall, LLMPurpose
from src.schemas import (
    LLMResponse,
    LLMTextResponse,
    LLMToolCallsResponse,
    ParsedToolCall,
)


log = get_logger(__name__)


# Model routing: purpose -> model name.
# `plan` and `synthesize` get the smarter model since plan quality drives the
# whole session and synthesis is the user-visible artifact. Everything else
# uses the cheaper/faster mini model. Change in one place if economics shift.
_MODEL_BY_PURPOSE: dict[LLMPurpose, str] = {
    LLMPurpose.plan: settings.planner_model,
    LLMPurpose.synthesize: settings.synthesizer_model,
    LLMPurpose.decide: settings.executor_model,
    LLMPurpose.summarize: settings.summarizer_model,
    LLMPurpose.verify: settings.verifier_model,
}


# JSONB columns truncate prompt/response payloads over this size. Avoids
# blowing up the DB on a runaway tool output without losing the LLM result.
# Not a deployer-tunable: the cap is tied to the truncation marker shape
# below; bump both together if you ever need more headroom.
_MAX_JSONB_BYTES = 50 * 1024


# Single client. AsyncOpenAI is safe to share across coroutines.
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Configure it in .env before calling the LLM."
            )
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def shutdown() -> None:
    global _client, _LLM_LOG_TASK, _LLM_LOG_QUEUE
    if _LLM_LOG_TASK is not None:
        _LLM_LOG_TASK.cancel()
        try:
            await _LLM_LOG_TASK
        except (asyncio.CancelledError, Exception):
            pass
        _LLM_LOG_TASK = None
    _LLM_LOG_QUEUE = None
    if _client is not None:
        try:
            await _client.close()
        except Exception:  # pragma: no cover
            pass
        _client = None


def model_for(purpose: LLMPurpose) -> str:
    return _MODEL_BY_PURPOSE[purpose]


# --- Retry policy --------------------------------------------------------
#
# tenacity is configured at function level. Exponential backoff with a hard
# cap of 4 attempts (3 retries). RateLimitError + APIConnectionError +
# APITimeoutError get retried; BadRequestError does NOT — it indicates a
# caller bug (bad schema, bad model name) and retrying just wastes tokens.

_RETRY = retry(
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)


# --- Core completion -----------------------------------------------------


async def complete(
    *,
    purpose: LLMPurpose,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] = "auto",
    parallel_tool_calls: bool = True,
    temperature: float | None = None,
    session_id: UUID | None = None,
    task_id: UUID | None = None,
) -> LLMResponse:
    """Run a chat completion. Returns a normalized LLMResponse.

    Args:
        purpose: routes the call to the right model and tags the DB log.
        messages: OpenAI Chat Completions messages array.
        tools: OpenAI function-calling tools, or None for plain text.
        tool_choice: "auto" / "none" / "required" / specific function dict.
        parallel_tool_calls: when True the model may emit multiple tool calls
            in one turn (the agent's inner loop executes them concurrently).
        temperature: passed through if set; otherwise the model default.
        session_id, task_id: for DB logging and structured-log binding.
    """
    model = model_for(purpose)
    client = get_client()

    log.info(
        Events.LLM_CALLED,
        purpose=purpose.value,
        model=model,
        message_count=len(messages),
        tool_count=len(tools) if tools else 0,
        tool_choice=tool_choice if isinstance(tool_choice, str) else "function",
    )

    @_RETRY
    async def _do_call():
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            # Some routing endpoints reject parallel_tool_calls when only
            # one tool is forced (tool_choice={"type": "function", ...}).
            # Only set it when truly meaningful (auto/required, > 1 tool).
            if tool_choice in ("auto", "required") or (
                isinstance(tool_choice, str) and len(tools) > 1
            ):
                kwargs["parallel_tool_calls"] = parallel_tool_calls
        if temperature is not None:
            kwargs["temperature"] = temperature

        start = time.perf_counter()
        response = await client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return response, latency_ms

    try:
        response, latency_ms = await _do_call()
    except BadRequestError:
        # Don't retry. Surface the error — likely a schema bug.
        raise

    # Normalize result.
    message = response.choices[0].message
    parsed = _normalize_response(message)

    log.info(
        Events.LLM_RESPONDED,
        purpose=purpose.value,
        model=model,
        latency_ms=latency_ms,
        input_tokens=getattr(response.usage, "prompt_tokens", None),
        output_tokens=getattr(response.usage, "completion_tokens", None),
        response_type=parsed.type,
    )

    # DB log is best-effort: an LLM call that succeeded shouldn't fail just
    # because we couldn't write the row. We still log the error.
    await _log_llm_call_safely(
        purpose=purpose,
        model=model,
        messages=messages,
        tools=tools,
        response_message=message,
        usage=response.usage,
        latency_ms=latency_ms,
        session_id=session_id,
        task_id=task_id,
    )

    return parsed


def _normalize_response(message: Any) -> LLMResponse:
    """Turn an OpenAI ChatCompletionMessage into our LLMResponse union."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        parsed_calls: list[ParsedToolCall] = []
        for tc in tool_calls:
            raw = tc.function.arguments or ""
            try:
                args = json.loads(raw) if raw else {}
                decoded = True
            except json.JSONDecodeError:
                args = {}
                decoded = False
                log.warning(
                    "llm.tool_args_decode_failed",
                    tool=tc.function.name,
                    raw_preview=raw[:200],
                )
            parsed_calls.append(
                ParsedToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    raw_arguments=raw,
                    decoded=decoded,
                )
            )
        return LLMToolCallsResponse(calls=parsed_calls)

    return LLMTextResponse(content=message.content or "")


# --- DB logging ----------------------------------------------------------


def _truncate_for_jsonb(payload: Any) -> Any:
    """Truncate a JSON-serializable payload if its serialization > _MAX_JSONB_BYTES.

    Strategy: serialize, check size, if too big replace with a marker dict
    containing a head/tail preview. We keep the row insertable, the trail
    still useful for debugging.
    """
    try:
        serialized = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return {"_truncated": True, "_reason": "unserializable"}

    if len(serialized.encode("utf-8")) <= _MAX_JSONB_BYTES:
        return payload

    head = serialized[:1000]
    tail = serialized[-500:]
    return {
        "_truncated": True,
        "_original_size_bytes": len(serialized.encode("utf-8")),
        "_head": head,
        "_tail": tail,
    }


_LLM_LOG_QUEUE: asyncio.Queue[LLMCall] | None = None
_LLM_LOG_TASK: asyncio.Task | None = None
_LLM_LOG_BATCH_SIZE = 16
_LLM_LOG_FLUSH_INTERVAL = 1.0


def _ensure_llm_log_worker() -> asyncio.Queue[LLMCall]:
    global _LLM_LOG_QUEUE, _LLM_LOG_TASK
    if _LLM_LOG_QUEUE is None:
        _LLM_LOG_QUEUE = asyncio.Queue()
    if _LLM_LOG_TASK is None or _LLM_LOG_TASK.done():
        _LLM_LOG_TASK = asyncio.create_task(_drain_llm_log())
    return _LLM_LOG_QUEUE


async def _drain_llm_log() -> None:
    assert _LLM_LOG_QUEUE is not None
    while True:
        batch: list[LLMCall] = []
        try:
            first = await _LLM_LOG_QUEUE.get()
            batch.append(first)
        except asyncio.CancelledError:
            return
        deadline = asyncio.get_event_loop().time() + _LLM_LOG_FLUSH_INTERVAL
        while len(batch) < _LLM_LOG_BATCH_SIZE:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                row = await asyncio.wait_for(_LLM_LOG_QUEUE.get(), timeout=remaining)
                batch.append(row)
            except asyncio.TimeoutError:
                break
        try:
            async with session_scope() as db:
                db.add_all(batch)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("llm.db_log_failed", error=str(exc), batch_size=len(batch))


async def _log_llm_call_safely(
    *,
    purpose: LLMPurpose,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    response_message: Any,
    usage: Any,
    latency_ms: int,
    session_id: UUID | None,
    task_id: UUID | None,
) -> None:
    try:
        response_payload: dict[str, Any] = {
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (response_message.tool_calls or [])
            ],
        }
        prompt_payload = {"messages": messages, "tools": tools}

        row = LLMCall(
            session_id=session_id,
            task_id=task_id,
            purpose=purpose,
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
            prompt=_truncate_for_jsonb(prompt_payload),
            response=_truncate_for_jsonb(response_payload),
        )
        queue = _ensure_llm_log_worker()
        queue.put_nowait(row)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("llm.db_log_failed", error=str(exc), purpose=purpose.value)


# --- Embeddings ----------------------------------------------------------


# OpenAI's embeddings endpoint accepts up to 2048 inputs per request, but
# we cap at 100 to keep individual calls quick and predictable for logging
# and retry purposes. Not deployer-tunable.
_EMBEDDING_BATCH = 100


@_RETRY
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = get_client()
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [d.embedding for d in response.data]


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batched at 100 inputs per call.

    Returns embeddings in input order.
    """
    if not texts:
        return []

    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBEDDING_BATCH):
        batch = texts[i : i + _EMBEDDING_BATCH]
        start = time.perf_counter()
        vectors = await _embed_batch(batch)
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "llm.embedded",
            model=settings.embedding_model,
            batch_size=len(batch),
            latency_ms=latency_ms,
            dim=len(vectors[0]) if vectors else 0,
        )
        out.extend(vectors)
    return out


__all__ = [
    "complete",
    "embed",
    "get_client",
    "model_for",
]
