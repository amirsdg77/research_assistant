"""fetch_url tool.

Fetch a page, extract main content with `readability-lxml`, store the full
text in memory, then return a
short auto-summary + memory_id to the LLM.

Critical context-strategy rule: the LLM NEVER sees raw page text. It sees
the 150-token summary and the memory_id, and can query the full content
later via `search_memory`.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import httpx
from pydantic import BaseModel, Field
from readability import Document  # type: ignore[import-untyped]

from src import memory as memory_mod
from src.llm import complete
from src.logging_setup import Events, get_logger
from src.models import LLMPurpose
from src.tools.base import ToolError, ToolSpec, current_session_id


log = get_logger(__name__)


_FETCH_TIMEOUT_SEC = 10.0
_MAX_BODY_CHARS = 60_000  # ~15k tokens — cap before chunking
_SUMMARY_TOKEN_TARGET = 150


class FetchUrlInput(BaseModel):
    url: str = Field(
        ..., min_length=8, max_length=2048, description="Absolute URL to fetch."
    )


class FetchUrlOutput(BaseModel):
    url: str
    title: str
    summary: str = Field(..., description="A short auto-summary (~150 tokens).")
    memory_id: str = Field(
        ..., description="Opaque id; use search_memory to retrieve full content."
    )


async def _http_get(url: str) -> tuple[str, str]:
    """Returns (extracted_text, title). Raises ToolError on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_SEC,
            follow_redirects=True,
            headers={"User-Agent": "research-agent/0.1"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise ToolError(f"fetch failed: {exc}") from exc

    try:
        doc = Document(html)
        title = (doc.short_title() or "").strip()
        # readability returns HTML for the main content; strip tags crudely.
        # `summary()` is HTML; we drop tags via lxml's text-only extraction.
        from lxml.html import fromstring  # type: ignore[import-untyped]

        cleaned = doc.summary()
        text = fromstring(cleaned).text_content() if cleaned else ""
    except Exception as exc:
        raise ToolError(f"content extraction failed: {exc}") from exc

    text = " ".join(text.split())  # collapse whitespace
    return text[:_MAX_BODY_CHARS], title


async def _summarize(text: str) -> str:
    """One-shot summary via the summarizer model. Bounded length."""
    if not text:
        return ""
    response = await complete(
        purpose=LLMPurpose.summarize,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following web page content in about "
                    f"{_SUMMARY_TOKEN_TARGET} tokens. Preserve specifics "
                    "(names, numbers, dates). Plain prose, no bullet points."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    if response.type == "text":
        return response.content.strip()
    # If a summarizer model unexpectedly emits tool calls (it shouldn't —
    # we don't pass any tools), fall back to a snippet.
    return text[:500]


def _make_fetch_handler(session_id_var):
    """Factory closure capturing the per-call session_id.

    The agent loop binds the active session_id into a contextvar before
    invoking tools; this resolves it lazily so the tool stays stateless
    at module load time.
    """

    async def handler(args: FetchUrlInput) -> FetchUrlOutput:
        sid: UUID | None = session_id_var.get()
        if sid is None:
            raise ToolError(
                "fetch_url requires an active session; no session_id is bound."
            )

        log.info(Events.TOOL_INVOKED, tool="fetch_url", url=args.url)
        try:
            text, title = await _http_get(args.url)
            summary = await _summarize(text) if text else ""
            memory_id = await memory_mod.store.add_page(
                session_id=sid, url=args.url, title=title, text=text
            )
        except asyncio.CancelledError:
            raise
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"fetch_url failed: {exc}") from exc

        log.info(
            Events.TOOL_COMPLETED,
            tool="fetch_url",
            url=args.url,
            text_chars=len(text),
            summary_chars=len(summary),
            memory_id=str(memory_id),
        )
        return FetchUrlOutput(
            url=args.url,
            title=title,
            summary=summary,
            memory_id=str(memory_id),
        )

    return handler


fetch_url_spec = ToolSpec(
    name="fetch_url",
    description=(
        "Fetch a web page, store its full content in semantic memory, and "
        "return a short summary plus a memory_id. Use this on the 1–3 most "
        "promising results from web_search. Cite the URL in finish_task."
    ),
    input_model=FetchUrlInput,
    output_model=FetchUrlOutput,
    handler=_make_fetch_handler(current_session_id),  # type: ignore[arg-type]
)
