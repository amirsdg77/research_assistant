"""search_documents tool — semantic query over uploaded documents.

Returns chunks with their filename and (when available) page number so the
executor can cite the document plus a precise locator.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src import memory as memory_mod
from src.logging_setup import Events, get_logger
from src.tools.base import ToolError, ToolSpec, current_session_id


log = get_logger(__name__)


class SearchDocumentsInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=400)
    k: int = Field(5, ge=1, le=10)


class DocumentHit(BaseModel):
    chunk: str
    filename: str = ""
    page: int | None = None
    document_id: str = ""
    score: float


class SearchDocumentsOutput(BaseModel):
    query: str
    hits: list[DocumentHit]


async def search_documents_handler(args: SearchDocumentsInput) -> SearchDocumentsOutput:
    sid = current_session_id.get()
    # Document search supports session_id=None (globals only), but the agent
    # always runs inside a session. Refuse if nothing is bound to surface
    # the misconfiguration loudly rather than silently search only globals.
    if sid is None:
        raise ToolError(
            "search_documents requires an active session; no session_id is bound."
        )

    log.info(Events.TOOL_INVOKED, tool="search_documents", query=args.query, k=args.k)
    raw = await memory_mod.store.search_documents(
        session_id=sid, query=args.query, k=args.k
    )
    hits: list[DocumentHit] = []
    for h in raw:
        meta = h.metadata or {}
        page = meta.get("page")
        hits.append(
            DocumentHit(
                chunk=h.chunk,
                filename=meta.get("filename", ""),
                page=int(page) if isinstance(page, (int, float)) else None,
                document_id=str(meta.get("document_id", "")),
                score=h.score,
            )
        )
    log.info(Events.TOOL_COMPLETED, tool="search_documents", hit_count=len(hits))
    return SearchDocumentsOutput(query=args.query, hits=hits)


search_documents_spec = ToolSpec(
    name="search_documents",
    description=(
        "Semantic search over uploaded user documents. Prefer this when the "
        "task references uploaded materials. Returns chunks with filename "
        "and page (when known) — cite both in finish_task."
    ),
    input_model=SearchDocumentsInput,
    output_model=SearchDocumentsOutput,
    handler=search_documents_handler,  # type: ignore[arg-type]
)
