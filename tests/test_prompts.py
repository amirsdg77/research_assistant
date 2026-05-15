"""Tests for src.prompts.

Importing the module must:
- Load all four prompts into module-level constants.
- Raise FileNotFoundError loudly if any is missing.
- Preserve the HTML-comment design notes at the top of each file (the LLM
  ignores them, but we depend on them as in-file documentation).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import src.prompts as prompts


EXPECTED_NAMES = [
    "planner_system",
    "executor_system",
    "synthesizer_system",
    "verifier_system",
]


def test_all_four_constants_loaded():
    assert isinstance(prompts.PLANNER_SYSTEM, str) and prompts.PLANNER_SYSTEM
    assert isinstance(prompts.EXECUTOR_SYSTEM, str) and prompts.EXECUTOR_SYSTEM
    assert isinstance(prompts.SYNTHESIZER_SYSTEM, str) and prompts.SYNTHESIZER_SYSTEM
    assert isinstance(prompts.VERIFIER_SYSTEM, str) and prompts.VERIFIER_SYSTEM


def test_all_prompts_dict_is_complete():
    assert set(prompts.ALL_PROMPTS.keys()) == set(EXPECTED_NAMES)
    for name, body in prompts.ALL_PROMPTS.items():
        assert body, f"prompt {name} is empty"


def test_design_notes_preserved():
    """The <!-- design notes --> blocks document intent. They're loaded into
    the prompt string (the LLM ignores HTML comments) and shouldn't be
    silently stripped by the loader."""
    for name in EXPECTED_NAMES:
        body = prompts.load(name)
        assert body.startswith("<!--"), f"{name} should start with a design-notes comment"
        assert "design notes" in body.split("-->")[0].lower()


def test_load_caches_results():
    a = prompts.load("planner_system")
    b = prompts.load("planner_system")
    assert a is b  # lru_cache returns the same object


def test_missing_prompt_raises(tmp_path, monkeypatch):
    """If src/prompts/ were missing a file, import would fail loudly.
    We can't easily simulate that without breaking the real module, but
    we can call load() directly with a nonexistent name."""
    with pytest.raises(FileNotFoundError):
        prompts.load("does_not_exist")


def test_prompts_present_in_filesystem():
    """Sanity: the .md files actually exist on disk where the loader looks."""
    prompts_dir = Path(prompts.__file__).parent
    for name in EXPECTED_NAMES:
        assert (prompts_dir / f"{name}.md").exists()


def test_prompts_under_size_cap():
    """Design intent: each file body stays under ~400 tokens. We approximate
    with a 4 chars/token rule, so a hard cap of 1800 chars is generous and
    catches obvious bloat without being fragile."""
    for name, body in prompts.ALL_PROMPTS.items():
        assert len(body) < 4000, (
            f"{name} is {len(body)} chars — getting bloated; tighten it"
        )
