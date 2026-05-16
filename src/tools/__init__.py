"""Tool registry + dispatch.

Each tool registers a ToolSpec. The agent loop:
1. Builds the OpenAI `tools` list via `openai_schemas_for(...)`.
2. Receives ParsedToolCall objects from the LLM wrapper.
3. Calls `invoke(name, arguments)` which validates input → runs handler →
   returns the Pydantic output (or raises ValidationError / ToolError).

The agent loop turns exceptions into tool-role error messages so the model
can self-correct.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from src.logging_setup import Events, get_logger
from src.tools.base import ToolError, ToolSpec, current_session_id
from src.tools.fetch_url import fetch_url_spec
from src.tools.finish_task import finish_task_spec
from src.tools.search_documents import search_documents_spec
from src.tools.search_memory import search_memory_spec
from src.tools.web_search import web_search_spec


log = get_logger(__name__)


REGISTRY: dict[str, ToolSpec] = {
    web_search_spec.name: web_search_spec,
    fetch_url_spec.name: fetch_url_spec,
    search_memory_spec.name: search_memory_spec,
    search_documents_spec.name: search_documents_spec,
    finish_task_spec.name: finish_task_spec,
}


def all_tool_names() -> list[str]:
    return list(REGISTRY.keys())


def openai_schemas_for(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Render OpenAI function-call schemas for the given tools (or all)."""
    if names is None:
        names = list(REGISTRY.keys())
    return [REGISTRY[n].openai_schema() for n in names]


async def invoke(name: str, arguments: dict[str, Any]) -> BaseModel:
    """Validate arguments and run the named tool.

    Raises:
        KeyError if the tool isn't registered.
        ValidationError if arguments fail the Pydantic schema.
        ToolError if the handler reported an operational failure.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown tool: {name}")
    spec = REGISTRY[name]
    try:
        parsed = spec.input_model.model_validate(arguments)
    except ValidationError:
        # Caller (agent loop) turns this into a tool-role error message.
        raise
    return await spec.handler(parsed)


__all__ = [
    "REGISTRY",
    "all_tool_names",
    "openai_schemas_for",
    "invoke",
    "current_session_id",
    "ToolError",
    "ToolSpec",
]
