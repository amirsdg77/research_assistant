"""web_search tool — Tavily-backed.

We pass the query straight through; Tavily handles ranking. We cap
`max_results` at 10 to keep the model from asking for floods.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from src.config import settings
from src.logging_setup import Events, get_logger
from src.tools.base import ToolError, ToolSpec


log = get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=400, description="Search query.")
    max_results: int = Field(
        5, ge=1, le=10, description="Max results to return (1–10)."
    )


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    query: str
    results: list[SearchHit]


_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        if not settings.tavily_api_key:
            raise ToolError(
                "TAVILY_API_KEY is not set. Configure it in .env before using web_search."
            )
        _client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    return _client


async def web_search_handler(args: WebSearchInput) -> WebSearchOutput:
    client = _get_client()
    log.info(Events.TOOL_INVOKED, tool="web_search", query=args.query, k=args.max_results)
    try:
        raw: dict[str, Any] = await client.search(
            query=args.query,
            max_results=args.max_results,
            search_depth="basic",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise ToolError(f"Tavily search failed: {exc}") from exc

    results = [
        SearchHit(
            title=item.get("title", "") or "",
            url=item.get("url", "") or "",
            snippet=item.get("content", "") or "",
        )
        for item in raw.get("results", [])
    ]
    log.info(Events.TOOL_COMPLETED, tool="web_search", hit_count=len(results))
    return WebSearchOutput(query=args.query, results=results)


web_search_spec = ToolSpec(
    name="web_search",
    description=(
        "Search the web via Tavily. Returns a list of {title, url, snippet}. "
        "Use this for broad discovery; follow up with fetch_url on the best "
        "results to actually read content."
    ),
    input_model=WebSearchInput,
    output_model=WebSearchOutput,
    handler=web_search_handler,  # type: ignore[arg-type]
)
