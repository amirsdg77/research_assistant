"""FastAPI app entrypoint.

Configures structured logging on startup, runs shared bootstrap (installs
the Chroma-backed memory store), binds a per-request request_id in
middleware, mounts the API router and the SPA static directory.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from src.api.routes import router as api_router
from src.bootstrap import setup as bootstrap_setup
from src.db import engine
from src.logging_setup import bind_request, get_logger
from src.prompts import ALL_PROMPTS


log = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_setup()
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
    return {"status": "ok"}


app.include_router(api_router)


# Static SPA. The `/` route returns index.html so deep links work, and the
# /static path serves the rest of the bundle. The static dir is created in
# the next build step; if absent we just skip mounting rather than crash.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")
