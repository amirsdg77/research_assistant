"""Prompt loader.

Prompts live as `.md` files in this directory. We load them at import time
into module-level constants so missing-file errors crash the app at startup
rather than at the first user request.

HTML comments at the top of each .md file serve as design notes — the LLM
ignores them cleanly, but they document intent for future editors.

`load(name)` is cached so re-reads are free. Prefer importing the eager
constants (`PLANNER_SYSTEM`, etc.) over calling `load()` directly so static
analysis catches typos.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Load a prompt by filename stem (e.g. 'planner_system')."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# Eager-loaded constants. Importing this module verifies all four prompts
# exist and are readable. If one is missing, we crash here rather than mid-run.
PLANNER_SYSTEM = load("planner_system")
EXECUTOR_SYSTEM = load("executor_system")
SYNTHESIZER_SYSTEM = load("synthesizer_system")
VERIFIER_SYSTEM = load("verifier_system")


ALL_PROMPTS: dict[str, str] = {
    "planner_system": PLANNER_SYSTEM,
    "executor_system": EXECUTOR_SYSTEM,
    "synthesizer_system": SYNTHESIZER_SYSTEM,
    "verifier_system": VERIFIER_SYSTEM,
}


__all__ = [
    "load",
    "PLANNER_SYSTEM",
    "EXECUTOR_SYSTEM",
    "SYNTHESIZER_SYSTEM",
    "VERIFIER_SYSTEM",
    "ALL_PROMPTS",
]
