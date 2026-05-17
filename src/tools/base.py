"""Tool protocol.

Each tool is three things:
- a Pydantic input model (validates the model's arguments dict),
- an OpenAI-style function schema (`{"type": "function", "function": {...}}`),
- an async handler that takes the parsed input and returns a Pydantic output.

This file defines the protocol + a small base class. Individual tools live in
their own modules and register via `src.tools.__init__`.

Why Pydantic for I/O: when the executor returns a tool_calls response with
malformed arguments, we get a `ValidationError` we can surface back to the
model verbatim — that's what powers the self-correction loop.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID

from pydantic import BaseModel


current_session_id: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

# Per-task set of URLs already fetched. Set by the agent at task entry.
# Tools can check membership to avoid redundant work.
fetched_urls: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "fetched_urls", default=None
)


class ToolError(Exception):
    """Raised by a handler to signal an operational error (network down,
    upstream API failure, etc). The agent loop wraps the message into a
    tool-role message so the model can react."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool: schema + handler + Pydantic input class."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[BaseModel]]

    def openai_schema(self) -> dict[str, Any]:
        """Render the OpenAI function-calling JSON Schema for this tool."""
        
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolHandler(Protocol):
    async def __call__(self, args: BaseModel) -> BaseModel: ...
