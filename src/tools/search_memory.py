"""search_memory tool — semantic query over pages fetched this session.

Returns chunks with their source URL and a memory_id so the model can cite
sources without us re-fetching anything.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src import memory as memory_mod
from src.logging_setup import Events, get_logger
from src.tools.base import ToolError, ToolSpec, current_session_id


log = get_logger(__name__)


class SearchMemoryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=400)
    k: int = Field(5, ge=1, le=10)


class MemoryHit(BaseModel):
    chunk: str
    source_url: str = ""
    score: float


class SearchMemoryOutput(BaseModel):
    query: str
    hits: list[MemoryHit]


async def search_memory_handler(args: SearchMemoryInput) -> SearchMemoryOutput:
    sid = current_session_id.get()
    if sid is None:
        raise ToolError(
            "search_memory requires an active session; no session_id is bound."
        )

    log.info(Events.TOOL_INVOKED, tool="search_memory", query=args.query, k=args.k)
    raw = await memory_mod.store.search_pages(session_id=sid, query=args.query, k=args.k)
    hits = [
        MemoryHit(
            chunk=h.chunk,
            source_url=h.metadata.get("source_url", ""),
            score=h.score,
        )
        for h in raw
    ]
    log.info(Events.TOOL_COMPLETED, tool="search_memory", hit_count=len(hits))
    return SearchMemoryOutput(query=args.query, hits=hits)


search_memory_spec = ToolSpec(
    name="search_memory",
    description=(
        "Semantic search over web pages fetched earlier in this session. "
        "Prefer this over web_search if the topic was already explored — "
        "stays consistent with prior findings and saves a network call."
    ),
    input_model=SearchMemoryInput,
    output_model=SearchMemoryOutput,
    handler=search_memory_handler,  # type: ignore[arg-type]
)
