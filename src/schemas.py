"""Pydantic schemas used across module boundaries.

Plan, agent events, guardrail results, and LLM-wrapper types live here so
every interface speaks structured Pydantic, never untyped dicts. Tool I/O
schemas live next to their handlers in src/tools/*.
"""
from __future__ import annotations

from typing import Any, Literal, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def openai_function_from_model(
    model: type[BaseModel], *, name: str, description: str
) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


# --- LLM wrapper boundary types -----------------------------------------


class ParsedToolCall(BaseModel):
    """A single tool call extracted from an OpenAI response.

    `arguments` is the JSON-decoded dict (not the raw string). If decoding
    fails, the wrapper records the raw string and `decoded=False` so the
    caller can surface a tool-error message back to the model for self-correction.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""
    decoded: bool = True


class LLMToolCallsResponse(BaseModel):
    type: Literal["tool_calls"] = "tool_calls"
    calls: list[ParsedToolCall]


class LLMTextResponse(BaseModel):
    type: Literal["text"] = "text"
    content: str


LLMResponse = Union[LLMToolCallsResponse, LLMTextResponse]


# --- Plan structure (used by planner function schema) -------------------


class PlanTask(BaseModel):
    description: str = Field(..., max_length=500)
    rationale: str = Field(..., max_length=500)


class Plan(BaseModel):
    tasks: list[PlanTask] = Field(..., min_length=3, max_length=7)


# --- Guardrail results --------------------------------------------------


class GuardrailResult(BaseModel):
    passed: bool
    reason: str | None = None


class VerificationResult(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str = ""


class ReportVerificationArgs(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str = Field("", max_length=1000)


class TaskDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    order_index: int
    description: str
    status: Any
    result_summary: str | None = None
    sources: list[Any] | None = None


__all__ = [
    "ParsedToolCall",
    "LLMToolCallsResponse",
    "LLMTextResponse",
    "LLMResponse",
    "PlanTask",
    "Plan",
    "GuardrailResult",
    "VerificationResult",
    "ReportVerificationArgs",
    "TaskDTO",
    "openai_function_from_model",
]
