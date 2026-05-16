"""HTTP routes for the research-agent API.

Endpoints:
- POST   /api/sessions                       — create + start session (multipart, optional files)
- GET    /api/sessions                       — list sessions
- GET    /api/sessions/{id}                  — session state + tasks + report
- POST   /api/sessions/{id}/resume           — resume a session
- GET    /api/sessions/{id}/events           — SSE event stream
- POST   /api/sessions/{id}/documents        — upload a document to a session
- POST   /api/documents                      — upload a global document
- GET    /api/documents                      — list documents

Sessions run in background tasks so the HTTP request returns immediately
with the session id; clients then subscribe via SSE for updates.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent import run_session
from src.db import get_session
from src.events import AgentEvent, EventTypes, bus
from src.ingest import IngestError, ingest_document
from src.logging_setup import get_logger
from src.models import Document, Session, SessionStatus, Task


log = get_logger(__name__)

router = APIRouter(prefix="/api")


# ---------- helpers --------------------------------------------------------


def _session_to_dict(s: Session) -> dict:
    return {
        "id": str(s.id),
        "goal": s.goal,
        "status": s.status.value,
        "final_report": s.final_report,
        "verification_notes": s.verification_notes,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _task_to_dict(t: Task) -> dict:
    return {
        "id": str(t.id),
        "order_index": t.order_index,
        "description": t.description,
        "status": t.status.value,
        "result_summary": t.result_summary,
        "sources": t.sources or [],
    }


def _document_to_dict(d: Document) -> dict:
    return {
        "id": str(d.id),
        "session_id": str(d.session_id) if d.session_id else None,
        "filename": d.filename,
        "content_type": d.content_type,
        "byte_size": d.byte_size,
        "chunk_count": d.chunk_count,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


async def _ingest_uploads(
    files: list[UploadFile], session_id: Optional[UUID]
) -> list[dict]:
    """Ingest a batch of uploaded files. Errors raise HTTPException."""
    out: list[dict] = []
    for f in files:
        try:
            data = await f.read()
            try:
                result = await ingest_document(
                    filename=f.filename or "untitled",
                    data=data,
                    session_id=session_id,
                    content_type=f.content_type,
                )
            except IngestError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{f.filename}: {exc}",
                )
        finally:
            await f.close()
        out.append(
            {
                "document_id": str(result.document_id),
                "filename": result.filename,
                "chunk_count": result.chunk_count,
            }
        )
    return out


# ---------- session endpoints ---------------------------------------------


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    background: BackgroundTasks,
    goal: str = Form(...),
    files: list[UploadFile] | None = File(None),
    db: AsyncSession = Depends(get_session),
) -> dict:
    """Create a session, ingest any attached files, and kick off the agent
    loop in the background. Returns the session id immediately so the
    client can subscribe to /events."""
    if not goal or not goal.strip():
        raise HTTPException(status_code=400, detail="goal is required")

    sid = uuid4()
    sess = Session(id=sid, goal=goal.strip(), status=SessionStatus.planning)
    db.add(sess)
    await db.commit()

    ingested: list[dict] = []
    if files:
        ingested = await _ingest_uploads(files, sid)

    background.add_task(run_session, goal.strip(), sid, resume=False)

    return {"session_id": str(sid), "documents": ingested}


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
) -> dict:
    result = await db.execute(
        select(Session).order_by(Session.created_at.desc()).limit(limit)
    )
    return {"sessions": [_session_to_dict(s) for s in result.scalars().all()]}


@router.get("/sessions/{session_id}")
async def get_session_state(
    session_id: UUID, db: AsyncSession = Depends(get_session)
) -> dict:
    sess = await db.get(Session, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    tasks_res = await db.execute(
        select(Task).where(Task.session_id == session_id).order_by(Task.order_index)
    )
    tasks = [_task_to_dict(t) for t in tasks_res.scalars().all()]
    return {**_session_to_dict(sess), "tasks": tasks}


@router.post("/sessions/{session_id}/resume", status_code=status.HTTP_202_ACCEPTED)
async def resume_session(
    session_id: UUID,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_session),
) -> dict:
    sess = await db.get(Session, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    background.add_task(run_session, sess.goal, session_id, resume=True)
    return {"session_id": str(session_id), "resumed": True}


# ---------- SSE -----------------------------------------------------------


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: UUID, request: Request) -> StreamingResponse:
    """Server-Sent Events stream of AgentEvents for the given session.

    The stream stays open until the agent loop publishes a `__close__`
    sentinel (i.e. session finished) or the client disconnects.
    """

    async def _gen() -> AsyncIterator[bytes]:
        async with bus.subscribe(session_id) as queue:
            while True:
                get_task = asyncio.create_task(queue.get())
                try:
                    done, _pending = await asyncio.wait({get_task}, timeout=15.0)
                except asyncio.CancelledError:
                    get_task.cancel()
                    raise
                if not done:
                    get_task.cancel()
                    try:
                        await get_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    if await request.is_disconnected():
                        return
                    yield b": keepalive\n\n"
                    continue
                event: AgentEvent = get_task.result()

                if event.type == EventTypes._CLOSE:
                    return

                payload = json.dumps(event.to_json(), default=str)
                yield f"event: {event.type}\ndata: {payload}\n\n".encode("utf-8")

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------- documents -----------------------------------------------------


@router.post("/sessions/{session_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_session_documents(
    session_id: UUID,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_session),
) -> dict:
    sess = await db.get(Session, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session not found")
    docs = await _ingest_uploads(files, session_id)
    return {"documents": docs}


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_global_documents(files: list[UploadFile] = File(...)) -> dict:
    docs = await _ingest_uploads(files, None)
    return {"documents": docs}


@router.get("/documents")
async def list_documents(
    session_id: Optional[UUID] = Query(None),
    only_global: bool = Query(False, alias="global"),
    db: AsyncSession = Depends(get_session),
) -> dict:
    stmt = select(Document).order_by(Document.created_at.desc())
    if only_global:
        stmt = stmt.where(Document.session_id.is_(None))
    elif session_id is not None:
        stmt = stmt.where(Document.session_id == session_id)
    result = await db.execute(stmt)
    return {"documents": [_document_to_dict(d) for d in result.scalars().all()]}


__all__ = ["router"]
