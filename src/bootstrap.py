"""Shared startup wiring used by both the CLI and the API server.

Idempotent: callers can invoke `setup()` freely on every entrypoint; the
underlying functions handle their own idempotency.
"""
from __future__ import annotations

from src.logging_setup import configure_logging, get_logger
from src.memory import use_chroma_store


_log = get_logger(__name__)


def setup() -> None:
    """Configure logging and install the Chroma-backed memory store."""
    configure_logging()
    try:
        use_chroma_store()
    except Exception as exc:
        # Don't crash the process if Chroma isn't reachable — surface in logs
        # and let downstream features (ingest, search_documents) fail loudly
        # when actually invoked. CLI commands that don't need Chroma (e.g.
        # list-sessions) should still work.
        _log.warning("bootstrap.chroma_unavailable", error=str(exc))


__all__ = ["setup"]
