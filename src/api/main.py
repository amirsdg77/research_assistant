"""FastAPI app entrypoint.

Configures structured logging on startup and binds a per-request request_id
in middleware so every log line emitted while handling a request carries it.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import text

from src.db import engine
from src.logging_setup import (
    Events,
    bind_request,
    configure_logging,
    get_logger,
)
from src.prompts import ALL_PROMPTS


log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("app.startup", version=app.version)
    # Importing src.prompts already validated all prompts exist (it loads them
    # at import time). Log their sizes so we can spot accidental empties.
    for name, body in ALL_PROMPTS.items():
        log.info("prompt.loaded", name=name, bytes=len(body))
    yield
    log.info("app.shutdown")


app = FastAPI(title="Research Agent", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Bind a request_id for the duration of every request so all logs
    emitted while handling it inherit the id automatically."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    with bind_request(rid):
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Verifies the app is up AND the DB is reachable."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    log.info(Events.SESSION_STARTED, note="health-check-ok")
    return {"status": "ok"}
