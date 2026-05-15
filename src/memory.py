"""Memory store interface, chunking utilities, and backends.

Defines the `MemoryStore` Protocol that the agent and tools use, plus two
implementations:

- `ChromaMemoryStore` — production backend. Two persistent collections
  (`pages`, `documents`) with metadata filtering by session_id. Embeddings
  are produced by our OpenAI wrapper so cost/latency stays in one logged
  pipeline.
- `InMemoryStubStore` — substring-match fallback for tests and offline runs.

Both satisfy the same Protocol so tools and the agent loop don't care which
one is active.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol
from uuid import UUID, uuid4

import tiktoken

from src.config import settings
from src.llm import embed
from src.logging_setup import get_logger


log = get_logger(__name__)


# --- Chunking ---------------------------------------------------------------
#
# 800-token chunks with 100-token overlap, counted via cl100k_base. Overlap
# preserves context across boundary sentences so retrieval doesn't fragment
# claims. tiktoken's encoder is loaded once and reused.

_CHUNK_TOKENS = 800
_CHUNK_OVERLAP = 100
_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> list[str]:
    """Split `text` into ~800-token chunks with 100-token overlap.

    Returns chunks in document order. Empty/whitespace-only input → [].
    Each chunk is decoded back to text after slicing the token stream so
    no half-codepoint splits are possible.
    """
    text = (text or "").strip()
    if not text:
        return []

    tokens = _ENCODER.encode(text)
    if not tokens:
        return []

    if len(tokens) <= _CHUNK_TOKENS:
        return [text]

    chunks: list[str] = []
    step = _CHUNK_TOKENS - _CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        end = start + _CHUNK_TOKENS
        slice_tokens = tokens[start:end]
        if not slice_tokens:
            break
        chunks.append(_ENCODER.decode(slice_tokens))
        if end >= len(tokens):
            break
    return chunks


@dataclass
class MemoryChunk:
    """A chunk retrieved from memory."""

    chunk: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    """Abstract semantic memory backing the agent's page and document collections."""

    async def add_page(
        self,
        *,
        session_id: UUID,
        url: str,
        title: str,
        text: str,
    ) -> UUID:
        """Chunk, embed, and store a fetched web page. Returns a memory_id
        the executor can quote in citations."""
        ...

    async def search_pages(
        self,
        *,
        session_id: UUID,
        query: str,
        k: int = 5,
    ) -> list[MemoryChunk]:
        """Semantic search across this session's stored pages."""
        ...

    async def search_documents(
        self,
        *,
        session_id: UUID | None,
        query: str,
        k: int = 5,
    ) -> list[MemoryChunk]:
        """Semantic search across this session's documents + globals."""
        ...


# --- In-process stub --------------------------------------------------------
#
# A trivial substring-match store, NOT semantic. Used as a fallback or in
# environments where Chroma isn't available (e.g. unit tests).


class InMemoryStubStore:
    """No-Chroma fallback. Substring match scored by overlap length."""

    def __init__(self) -> None:
        # session_id -> list[(memory_id, url, title, text)]
        self._pages: dict[UUID, list[tuple[UUID, str, str, str]]] = {}

    async def add_page(
        self, *, session_id: UUID, url: str, title: str, text: str
    ) -> UUID:
        mid = uuid4()
        self._pages.setdefault(session_id, []).append((mid, url, title, text))
        return mid

    async def search_pages(
        self, *, session_id: UUID, query: str, k: int = 5
    ) -> list[MemoryChunk]:
        q = query.lower()
        hits: list[MemoryChunk] = []
        for mid, url, title, text in self._pages.get(session_id, []):
            if not text:
                continue
            score = float(text.lower().count(q))
            if score > 0:
                # Pull a small excerpt around the first match.
                idx = text.lower().find(q)
                start = max(0, idx - 80)
                end = min(len(text), idx + 240)
                hits.append(
                    MemoryChunk(
                        chunk=text[start:end],
                        score=score,
                        metadata={"source_url": url, "title": title, "memory_id": str(mid)},
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    async def search_documents(
        self, *, session_id: UUID | None, query: str, k: int = 5
    ) -> list[MemoryChunk]:
        # The stub doesn't track documents; document ingest writes to the
        # Chroma-backed store.
        return []


# --- Chroma backend ---------------------------------------------------------
#
# Two persistent collections shared across the whole deployment. Per-session
# isolation is enforced via metadata filtering, not per-session collections —
# Chroma doesn't garbage-collect empty collections, and our session counts
# would balloon the metastore otherwise.

_PAGES_COLLECTION = "pages"
_DOCUMENTS_COLLECTION = "documents"


@dataclass
class DocumentChunkSpec:
    """One pre-chunked piece of an uploaded document, ready for ingestion."""

    text: str
    chunk_index: int
    page: int | None = None


class ChromaMemoryStore:
    """Chroma-backed `MemoryStore`. Lazily connects on first use.

    Uses OpenAI embeddings via `src.llm.embed` so cost/latency is logged in
    one place rather than via Chroma's own embedding function.
    """

    def __init__(self) -> None:
        self._client = None  # type: ignore[assignment]
        self._pages = None
        self._documents = None
        self._init_lock = asyncio.Lock()

    async def _ensure(self) -> None:
        if self._client is not None:
            return
        async with self._init_lock:
            if self._client is not None:
                return
            import chromadb  # local import keeps import cost off the hot path

            log.info(
                "memory.chroma.connecting",
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            client = await chromadb.AsyncHttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
            self._client = client
            self._pages = await client.get_or_create_collection(_PAGES_COLLECTION)
            self._documents = await client.get_or_create_collection(_DOCUMENTS_COLLECTION)
            log.info("memory.chroma.ready")

    # ---- pages ----

    async def add_page(
        self, *, session_id: UUID, url: str, title: str, text: str
    ) -> UUID:
        await self._ensure()
        memory_id = uuid4()
        chunks = chunk_text(text)
        if not chunks:
            return memory_id

        vectors = await embed(chunks)
        ids = [f"{memory_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "session_id": str(session_id),
                "memory_id": str(memory_id),
                "url": url,
                "title": title,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        await self._pages.add(  # type: ignore[union-attr]
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=metadatas,
        )
        log.info(
            "memory.page_added",
            session_id=str(session_id),
            url=url,
            chunk_count=len(chunks),
            memory_id=str(memory_id),
        )
        return memory_id

    async def search_pages(
        self, *, session_id: UUID, query: str, k: int = 5
    ) -> list[MemoryChunk]:
        await self._ensure()
        vectors = await embed([query])
        result = await self._pages.query(  # type: ignore[union-attr]
            query_embeddings=vectors,
            n_results=k,
            where={"session_id": str(session_id)},
        )
        return _result_to_chunks(result, source_key="url")

    # ---- documents ----

    async def add_document_chunks(
        self,
        *,
        document_id: UUID,
        session_id: UUID | None,
        filename: str,
        chunks: Iterable[DocumentChunkSpec],
    ) -> int:
        """Ingest pre-chunked document content. Returns the number of chunks stored."""
        await self._ensure()
        chunks_list = list(chunks)
        if not chunks_list:
            return 0

        texts = [c.text for c in chunks_list]
        vectors = await embed(texts)
        ids = [f"{document_id}:{c.chunk_index}" for c in chunks_list]
        metadatas: list[dict[str, Any]] = []
        for c in chunks_list:
            meta: dict[str, Any] = {
                "document_id": str(document_id),
                "filename": filename,
                "chunk_index": c.chunk_index,
                # session_id="" sentinel for globals — Chroma metadata can't be None.
                "session_id": str(session_id) if session_id is not None else "",
            }
            if c.page is not None:
                meta["page"] = c.page
            metadatas.append(meta)
        await self._documents.add(  # type: ignore[union-attr]
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        log.info(
            "memory.document_added",
            document_id=str(document_id),
            session_id=str(session_id) if session_id else None,
            filename=filename,
            chunk_count=len(chunks_list),
        )
        return len(chunks_list)

    async def search_documents(
        self, *, session_id: UUID | None, query: str, k: int = 5
    ) -> list[MemoryChunk]:
        await self._ensure()
        vectors = await embed([query])
        # Match this session's docs OR globals (session_id="" sentinel).
        if session_id is None:
            where: dict[str, Any] = {"session_id": ""}
        else:
            where = {"session_id": {"$in": [str(session_id), ""]}}
        result = await self._documents.query(  # type: ignore[union-attr]
            query_embeddings=vectors,
            n_results=k,
            where=where,
        )
        return _result_to_chunks(result, source_key="filename")


def _result_to_chunks(
    result: dict[str, Any], *, source_key: str
) -> list[MemoryChunk]:
    """Flatten Chroma's per-query result lists into MemoryChunk objects.

    Chroma returns shape `{ids: [[...]], documents: [[...]], ...}` because
    `query` supports a batch of queries; we always pass one query so we
    unwrap the outer list. Distances become 1/(1+d) scores so higher = better.
    """
    docs_outer = result.get("documents") or [[]]
    metas_outer = result.get("metadatas") or [[]]
    dists_outer = result.get("distances") or [[]]
    docs = docs_outer[0] if docs_outer else []
    metas = metas_outer[0] if metas_outer else []
    dists = dists_outer[0] if dists_outer else []

    out: list[MemoryChunk] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else 0.0
        score = 1.0 / (1.0 + float(dist))
        metadata = dict(meta) if meta else {}
        if source_key == "url":
            metadata["source_url"] = metadata.get("url", "")
        elif source_key == "filename":
            metadata["source_url"] = metadata.get("filename", "")
        out.append(MemoryChunk(chunk=doc, score=score, metadata=metadata))
    return out


# --- Singleton --------------------------------------------------------------
#
# Default to the stub so unit tests and CLI smoke checks don't require a
# running Chroma. Application startup swaps in the Chroma-backed store.

store: MemoryStore = InMemoryStubStore()


def use_chroma_store() -> ChromaMemoryStore:
    """Install the Chroma backend as the process-wide store. Idempotent."""
    global store
    if isinstance(store, ChromaMemoryStore):
        return store
    store = ChromaMemoryStore()
    return store


__all__ = [
    "MemoryStore",
    "MemoryChunk",
    "InMemoryStubStore",
    "ChromaMemoryStore",
    "DocumentChunkSpec",
    "chunk_text",
    "store",
    "use_chroma_store",
]
