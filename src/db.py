"""Async SQLAlchemy engine + session factory.

Single engine for the process. `get_session()` is a FastAPI dependency that
yields an `AsyncSession` and ensures cleanup. Outside FastAPI, use the
`session_scope()` async context manager.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings


engine: AsyncEngine = create_async_engine(
    settings.postgres_url,
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Async context manager that commits on success, rolls back on error."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Caller controls commit/rollback semantics."""
    async with SessionLocal() as session:
        yield session
