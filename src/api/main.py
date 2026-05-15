"""FastAPI app entrypoint.

Step 2 ships only the health check so `docker compose up` succeeds end-to-end
(migrations run, server boots, healthcheck passes). Real routes get added in
step 13.
"""
from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy import text

from src.db import engine


app = FastAPI(title="Research Agent", version="0.1.0")


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Verifies the app is up AND the DB is reachable."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"status": "ok"}
