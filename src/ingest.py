"""Document ingest pipeline.

Parses uploaded files into text, chunks via the shared `chunk_text` helper,
embeds and stores chunks in the documents collection of the memory store,
and writes a `documents` row to Postgres recording the ingestion.

Supported types:
- PDF: per-page text via pypdf.
- DOCX: paragraph text via python-docx.
- TXT/MD: plain UTF-8 read.

Pages metadata is preserved for PDFs so citations can name a specific page.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db import session_scope
from src.logging_setup import get_logger
from src import memory as memory_mod
from src.memory import (
    ChromaMemoryStore,
    DocumentChunkSpec,
    chunk_text,
)
from src.models import Document


log = get_logger(__name__)


# --- Errors -----------------------------------------------------------------


class IngestError(Exception):
    """Operational failure during ingest (parsing, validation, storage)."""


# --- Type detection ---------------------------------------------------------


_EXT_TO_CONTENT_TYPE = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def detect_content_type(filename: str, *, fallback: str | None = None) -> str:
    """Infer content type from filename extension. Falls back if provided
    by the caller (e.g. multipart upload's declared type)."""
    ext = Path(filename).suffix.lower()
    if ext in _EXT_TO_CONTENT_TYPE:
        return _EXT_TO_CONTENT_TYPE[ext]
    if fallback:
        return fallback
    raise IngestError(f"unsupported file type: {filename}")


# --- Parsers ----------------------------------------------------------------


@dataclass
class ParsedPage:
    """One unit of parsed source text. `page` is 1-based for PDFs and None
    for formats without a real page concept."""

    text: str
    page: int | None = None


def _parse_pdf(data: bytes) -> list[ParsedPage]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"PDF parse failed: {exc}") from exc

    pages: list[ParsedPage] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(ParsedPage(text=text, page=i + 1))
    return pages


def _parse_docx(data: bytes) -> list[ParsedPage]:
    from docx import Document as DocxDocument

    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"DOCX parse failed: {exc}") from exc

    # DOCX has no real page boundaries (Word's pagination is renderer-dependent).
    # Concatenate paragraphs into one block; chunking handles the split.
    blob = "\n\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
    blob = blob.strip()
    return [ParsedPage(text=blob)] if blob else []


def _parse_plaintext(data: bytes) -> list[ParsedPage]:
    try:
        text = data.decode("utf-8", errors="replace").strip()
    except Exception as exc:  # pragma: no cover — decode with replace can't raise
        raise IngestError(f"text decode failed: {exc}") from exc
    return [ParsedPage(text=text)] if text else []


def parse(data: bytes, content_type: str) -> list[ParsedPage]:
    """Dispatch to the right parser. Returns ordered ParsedPage list."""
    if content_type == "application/pdf":
        return _parse_pdf(data)
    if content_type == _EXT_TO_CONTENT_TYPE[".docx"]:
        return _parse_docx(data)
    if content_type in ("text/plain", "text/markdown"):
        return _parse_plaintext(data)
    raise IngestError(f"unsupported content type: {content_type}")


# --- Chunking with page metadata --------------------------------------------


def chunks_from_pages(pages: Iterable[ParsedPage]) -> list[DocumentChunkSpec]:
    """Chunk each parsed page individually so the `page` metadata is preserved
    on every chunk. Chunk indices are global across the whole document."""
    specs: list[DocumentChunkSpec] = []
    idx = 0
    for page in pages:
        for chunk in chunk_text(page.text):
            specs.append(
                DocumentChunkSpec(text=chunk, chunk_index=idx, page=page.page)
            )
            idx += 1
    return specs


# --- Validation -------------------------------------------------------------


async def _check_session_quota(db: AsyncSession, session_id: UUID) -> None:
    result = await db.execute(
        select(Document).where(Document.session_id == session_id)
    )
    existing = result.scalars().all()
    if len(existing) >= settings.max_docs_per_session:
        raise IngestError(
            f"session has reached the max docs limit ({settings.max_docs_per_session})"
        )


def _check_size(data: bytes, filename: str) -> None:
    if len(data) > settings.max_upload_bytes:
        raise IngestError(
            f"{filename} is {len(data)} bytes; max is {settings.max_upload_bytes}"
        )


# --- Ingest -----------------------------------------------------------------


@dataclass
class IngestResult:
    document_id: UUID
    filename: str
    chunk_count: int


async def ingest_document(
    *,
    filename: str,
    data: bytes,
    session_id: UUID | None,
    content_type: str | None = None,
) -> IngestResult:
    """Parse, chunk, embed, store, and record one document.

    Args:
        filename: original filename; used for metadata and content-type sniffing.
        data: raw bytes.
        session_id: bind to a session, or None for a global document.
        content_type: declared MIME (e.g. from a multipart upload). If absent
            we sniff from the extension.

    Raises IngestError on validation, parse, or storage failure.
    """
    _check_size(data, filename)
    ctype = detect_content_type(filename, fallback=content_type)

    if session_id is not None:
        async with session_scope() as db:
            await _check_session_quota(db, session_id)

    pages = parse(data, ctype)
    chunks = chunks_from_pages(pages)
    if not chunks:
        raise IngestError(f"{filename} parsed to zero text chunks")

    document_id = uuid4()

    # Chroma write first. If it fails, the documents row never gets created
    # (no orphan DB row). The opposite failure mode — Chroma succeeds but
    # the DB row fails — leaves orphan chunks; rare in practice, but called
    # out as a known limitation in the README.
    if not isinstance(memory_mod.store, ChromaMemoryStore):
        # Stub store doesn't support documents; refuse rather than silently
        # accept an ingest that produces no searchable result.
        raise IngestError(
            "ingest requires the Chroma-backed memory store; "
            "call memory.use_chroma_store() at startup"
        )

    stored = await memory_mod.store.add_document_chunks(
        document_id=document_id,
        session_id=session_id,
        filename=filename,
        chunks=chunks,
    )

    async with session_scope() as db:
        doc = Document(
            id=document_id,
            session_id=session_id,
            filename=filename,
            content_type=ctype,
            byte_size=len(data),
            chunk_count=stored,
        )
        db.add(doc)

    log.info(
        "ingest.completed",
        document_id=str(document_id),
        session_id=str(session_id) if session_id else None,
        filename=filename,
        bytes=len(data),
        chunk_count=stored,
    )
    return IngestResult(document_id=document_id, filename=filename, chunk_count=stored)


__all__ = [
    "IngestError",
    "IngestResult",
    "ParsedPage",
    "detect_content_type",
    "parse",
    "chunks_from_pages",
    "ingest_document",
]
