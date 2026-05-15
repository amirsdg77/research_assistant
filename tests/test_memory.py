"""Tests for src.memory.

Covers:
- chunk_text behavior: empty in/out, single-chunk under threshold, multi-chunk
  with overlap, token-accurate boundaries via cl100k_base.
- ChromaMemoryStore: add_page chunks/embeds/writes with correct metadata,
  search_pages filters by session_id, add_document_chunks ingests with
  filename/page metadata, search_documents includes globals in the where
  filter, distance → score conversion is monotonic.
- _result_to_chunks: unwraps Chroma's batched-query shape and maps the
  source key correctly.

Chroma is mocked end-to-end so these tests run without a running server.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import tiktoken

import src.memory as memory_mod
from src.memory import (
    ChromaMemoryStore,
    DocumentChunkSpec,
    InMemoryStubStore,
    MemoryChunk,
    _result_to_chunks,
    chunk_text,
    use_chroma_store,
)


# --- chunking ----------------------------------------------------------


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\t") == []


def test_chunk_text_short_input_is_single_chunk():
    text = "Just a short paragraph."
    chunks = chunk_text(text)
    assert chunks == [text]


def test_chunk_text_splits_long_input_with_overlap():
    # Build text longer than one chunk (800 tokens). 5,000 "word " tokens
    # ~ 5000 tokens, which produces multiple chunks.
    encoder = tiktoken.get_encoding("cl100k_base")
    text = "word " * 5000
    chunks = chunk_text(text)

    assert len(chunks) > 1
    # Every chunk should be at most 800 tokens.
    for c in chunks:
        assert len(encoder.encode(c)) <= 800

    # Adjacent chunks should overlap. We can verify by token equality at the
    # join: the tail tokens of chunk i should match the head tokens of i+1.
    a_tail = encoder.encode(chunks[0])[-100:]
    b_head = encoder.encode(chunks[1])[:100]
    assert a_tail == b_head


def test_chunk_text_handles_exact_boundary():
    """Input whose token count is a multiple of the step should still
    terminate (no infinite loop / no empty trailing chunk)."""
    encoder = tiktoken.get_encoding("cl100k_base")
    # Build text of exactly 1400 tokens (one chunk of 800 plus one of 700
    # with 100 overlap → 800+700-100 = 1400 step).
    tokens = encoder.encode("word " * 2000)[:1400]
    text = encoder.decode(tokens)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)  # no empty chunks


# --- ChromaMemoryStore -------------------------------------------------


def _fake_collection() -> SimpleNamespace:
    """Stand-in for chromadb.AsyncCollection."""
    return SimpleNamespace(add=AsyncMock(), query=AsyncMock())


@pytest.fixture
def chroma_store(monkeypatch):
    """A ChromaMemoryStore with its `_ensure` no-op'd and fake collections."""
    store = ChromaMemoryStore()
    store._client = SimpleNamespace()
    store._pages = _fake_collection()
    store._documents = _fake_collection()

    async def _noop_ensure():
        return None

    monkeypatch.setattr(store, "_ensure", _noop_ensure)

    async def _fake_embed(texts):
        return [[0.1] * 4 for _ in texts]

    monkeypatch.setattr("src.memory.embed", _fake_embed)
    return store


async def test_chroma_add_page_chunks_and_writes(chroma_store):
    sid = uuid4()
    text = "x" * 50  # short → single chunk
    mid = await chroma_store.add_page(
        session_id=sid, url="https://a", title="T", text=text
    )
    chroma_store._pages.add.assert_awaited_once()
    kwargs = chroma_store._pages.add.await_args.kwargs
    assert len(kwargs["documents"]) == 1
    assert len(kwargs["embeddings"]) == 1
    metadata = kwargs["metadatas"][0]
    assert metadata["session_id"] == str(sid)
    assert metadata["url"] == "https://a"
    assert metadata["memory_id"] == str(mid)
    assert metadata["chunk_index"] == 0


async def test_chroma_add_page_handles_empty_text(chroma_store):
    sid = uuid4()
    mid = await chroma_store.add_page(session_id=sid, url="https://a", title="T", text="")
    # Returns an id, but never writes anything to the collection.
    assert mid is not None
    chroma_store._pages.add.assert_not_awaited()


async def test_chroma_search_pages_filters_by_session(chroma_store):
    sid = uuid4()
    chroma_store._pages.query.return_value = {
        "documents": [["chunk one"]],
        "metadatas": [[{"url": "https://a", "session_id": str(sid)}]],
        "distances": [[0.25]],
    }
    hits = await chroma_store.search_pages(session_id=sid, query="foo", k=3)
    where = chroma_store._pages.query.await_args.kwargs["where"]
    assert where == {"session_id": str(sid)}
    assert len(hits) == 1
    assert hits[0].chunk == "chunk one"
    assert hits[0].metadata["source_url"] == "https://a"
    # Score should be 1/(1+0.25) = 0.8
    assert abs(hits[0].score - 0.8) < 1e-6


async def test_chroma_add_document_chunks_writes_filename_and_page(chroma_store):
    doc_id = uuid4()
    sid = uuid4()
    specs = [
        DocumentChunkSpec(text="alpha", chunk_index=0, page=1),
        DocumentChunkSpec(text="beta", chunk_index=1, page=2),
    ]
    n = await chroma_store.add_document_chunks(
        document_id=doc_id,
        session_id=sid,
        filename="paper.pdf",
        chunks=specs,
    )
    assert n == 2
    kwargs = chroma_store._documents.add.await_args.kwargs
    assert kwargs["documents"] == ["alpha", "beta"]
    metas = kwargs["metadatas"]
    assert metas[0]["filename"] == "paper.pdf"
    assert metas[0]["session_id"] == str(sid)
    assert metas[0]["page"] == 1
    assert metas[1]["page"] == 2


async def test_chroma_add_document_chunks_global_uses_empty_session(chroma_store):
    """Globals store session_id='' so the where filter can union session+globals."""
    n = await chroma_store.add_document_chunks(
        document_id=uuid4(),
        session_id=None,
        filename="g.pdf",
        chunks=[DocumentChunkSpec(text="hello", chunk_index=0)],
    )
    assert n == 1
    metas = chroma_store._documents.add.await_args.kwargs["metadatas"]
    assert metas[0]["session_id"] == ""
    assert "page" not in metas[0]


async def test_chroma_search_documents_unions_session_and_globals(chroma_store):
    sid = uuid4()
    chroma_store._documents.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    await chroma_store.search_documents(session_id=sid, query="foo", k=3)
    where = chroma_store._documents.query.await_args.kwargs["where"]
    assert where == {"session_id": {"$in": [str(sid), ""]}}


async def test_chroma_search_documents_global_only(chroma_store):
    chroma_store._documents.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    await chroma_store.search_documents(session_id=None, query="foo", k=3)
    where = chroma_store._documents.query.await_args.kwargs["where"]
    assert where == {"session_id": ""}


# --- _result_to_chunks helper -----------------------------------------


def test_result_to_chunks_maps_url_source():
    result = {
        "documents": [["A"]],
        "metadatas": [[{"url": "https://x"}]],
        "distances": [[0.0]],
    }
    chunks = _result_to_chunks(result, source_key="url")
    assert chunks[0].metadata["source_url"] == "https://x"
    # distance=0 → score=1
    assert abs(chunks[0].score - 1.0) < 1e-6


def test_result_to_chunks_maps_filename_source():
    result = {
        "documents": [["A"]],
        "metadatas": [[{"filename": "f.pdf"}]],
        "distances": [[1.0]],
    }
    chunks = _result_to_chunks(result, source_key="filename")
    assert chunks[0].metadata["source_url"] == "f.pdf"
    # distance=1 → score=0.5
    assert abs(chunks[0].score - 0.5) < 1e-6


def test_result_to_chunks_handles_empty_result():
    chunks = _result_to_chunks({"documents": [[]], "metadatas": [[]], "distances": [[]]}, source_key="url")
    assert chunks == []


# --- singleton swap ----------------------------------------------------


def test_use_chroma_store_swaps_singleton(monkeypatch):
    # Start from the stub.
    monkeypatch.setattr(memory_mod, "store", InMemoryStubStore())
    new_store = use_chroma_store()
    assert isinstance(memory_mod.store, ChromaMemoryStore)
    assert memory_mod.store is new_store
    # Idempotent.
    again = use_chroma_store()
    assert again is new_store


def test_memory_chunk_dataclass_defaults():
    c = MemoryChunk(chunk="x", score=1.0)
    assert c.metadata == {}
