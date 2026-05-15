"""finish_task — terminal tool that commits a task's result.

Schema enforces `sources` has at least one entry. If the model calls
finish_task without sources, Pydantic raises ValidationError; the agent
loop surfaces that error back as a tool message and counts a retry. After
2 such retries the task fails (rule lives in the agent loop, not here).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.logging_setup import Events, get_logger
from src.tools.base import ToolSpec


log = get_logger(__name__)


class FinishTaskInput(BaseModel):
    result_summary: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "3–8 sentence summary of findings, dense with specifics "
            "(names, numbers, dates). Every claim must trace to a source."
        ),
    )
    sources: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "URLs or document references backing the summary. "
            "MUST contain at least one entry."
        ),
    )


class FinishTaskOutput(BaseModel):
    accepted: bool = True
    result_summary: str
    sources: list[str]


async def finish_task_handler(args: FinishTaskInput) -> FinishTaskOutput:
    log.info(
        Events.TOOL_COMPLETED,
        tool="finish_task",
        source_count=len(args.sources),
        summary_chars=len(args.result_summary),
    )
    return FinishTaskOutput(
        accepted=True,
        result_summary=args.result_summary,
        sources=args.sources,
    )


finish_task_spec = ToolSpec(
    name="finish_task",
    description=(
        "Terminal tool. Commit your findings for the current task. "
        "MUST include at least one source URL or document reference. "
        "Call this once you have enough evidence — do not over-iterate."
    ),
    input_model=FinishTaskInput,
    output_model=FinishTaskOutput,
    handler=finish_task_handler,  # type: ignore[arg-type]
)
