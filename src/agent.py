"""The agent loop.

run_session(goal, session_id, resume=False) orchestrates the whole flow:

  Phase 0 — input guardrail
  Phase 1 — planning (one LLM call → 3-7 tasks persisted)
  Phase 2 — execution loop (per-task inner tool loop)
  Phase 3 — synthesis (markdown report with inline citations)
  Phase 4 — output guardrail (verifier flags unsupported claims)

Context strategy (the rules this module enforces):
- LLM input budget capped at TOKEN_BUDGET_PER_CALL (counted with cl100k_base).
- Full page contents NEVER enter LLM context; only fetch_url's auto-summary
  + memory_id flow through. Cross-task and cross-source memory lives in
  the vector store, retrieved per-task.
- Completed-task context is summary-only.
- Inner tool loop keeps the last 3 tool exchanges verbatim and summarizes
  older ones into a scratchpad before re-calling the model.

The loop emits structured log events and AgentEvents at every transition
so the SSE channel and the activity feed can render in real time.
"""
from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import tiktoken
from pydantic import ValidationError
from sqlalchemy import select

from src import tools as tools_pkg
from src.config import settings
from src.db import session_scope
from src.events import AgentEvent, EventTypes, bus
from src.guardrails import check_goal, verify_report
from src.llm import complete
from src.logging_setup import Events, bind_session, bind_task, get_logger
from src.models import (
    Document,
    LLMPurpose,
    Session,
    SessionStatus,
    Task,
    TaskStatus,
)
from src.prompts import EXECUTOR_SYSTEM, PLANNER_SYSTEM, SYNTHESIZER_SYSTEM
from src.schemas import Plan, TaskDTO, openai_function_from_model
from src.tools.base import ToolError


log = get_logger(__name__)


_ENCODER = tiktoken.get_encoding("cl100k_base")

_SESSION_SEMAPHORE: asyncio.Semaphore | None = None


def _get_session_semaphore() -> asyncio.Semaphore:
    global _SESSION_SEMAPHORE
    if _SESSION_SEMAPHORE is None:
        _SESSION_SEMAPHORE = asyncio.Semaphore(settings.max_concurrent_sessions)
    return _SESSION_SEMAPHORE


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _count_message_tokens(m: dict[str, Any]) -> int:
    total = 0
    content = m.get("content")
    if isinstance(content, str):
        total += _count_tokens(content)
    total += _count_tokens(str(m.get("role", "")))
    if "tool_calls" in m and m["tool_calls"]:
        total += _count_tokens(str(m["tool_calls"]))
    if "name" in m:
        total += _count_tokens(str(m["name"]))
    return total


# --- Plan schema (forced function) -----------------------------------------


_CREATE_PLAN_FUNCTION = openai_function_from_model(
    Plan,
    name="create_plan",
    description="Emit a structured research plan of 3-7 sub-questions.",
)


# --- Entry point -----------------------------------------------------------


async def run_session(
    goal: str, session_id: UUID, *, resume: bool = False
) -> None:
    """Drive a session from start to verification.

    Always returns; failures are caught and persisted as session status
    `failed` so the UI can render an end state. Exceptions inside individual
    tasks fail that task only, not the whole session.
    """
    with bind_session(session_id):
        log.info(
            Events.SESSION_RESUMED if resume else Events.SESSION_STARTED, goal=goal
        )
        await bus.publish(
            AgentEvent(
                session_id=session_id,
                type=EventTypes.SESSION_STARTED,
                data={"goal": goal, "resume": resume},
            )
        )
        sem = _get_session_semaphore()
        async with sem:
            try:
                await _run_session_inner(goal, session_id, resume=resume)
            except Exception as exc:
                log.exception(Events.SESSION_FAILED, error=str(exc))
                await _set_session_status(session_id, SessionStatus.failed)
                await bus.publish(
                    AgentEvent(
                        session_id=session_id,
                        type=EventTypes.SESSION_FAILED,
                        data={"error": str(exc)},
                    )
                )
            finally:
                await bus.close_session(session_id)


async def _run_session_inner(
    goal: str, session_id: UUID, *, resume: bool
) -> None:
    # --- Phase 0: input guardrail (skip on resume; goal was already accepted)
    if not resume:
        guard = await check_goal(goal)
        if not guard.passed:
            await _set_session_status(session_id, SessionStatus.failed)
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.GUARDRAIL_TRIGGERED,
                    data={"check": "check_goal", "reason": guard.reason},
                )
            )
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.SESSION_FAILED,
                    data={"error": guard.reason or "guardrail rejected goal"},
                )
            )
            return

    # --- Phase 1: planning
    plan_tasks = await _plan_or_load(goal, session_id, resume=resume)
    if not plan_tasks:
        await _set_session_status(session_id, SessionStatus.failed)
        await bus.publish(
            AgentEvent(
                session_id=session_id,
                type=EventTypes.SESSION_FAILED,
                data={"error": "planning failed to produce tasks"},
            )
        )
        return

    # --- Phase 2: execution
    await _set_session_status(session_id, SessionStatus.running)
    completed_summaries: list[tuple[str, str, list[str]]] = []  # (description, summary, sources)
    failed_tasks: list[tuple[str, str]] = []  # (description, failure reason)

    for task_row in plan_tasks:
        # Preload already-completed task summaries so resume picks up cleanly.
        if task_row.status == TaskStatus.done and task_row.result_summary:
            completed_summaries.append(
                (
                    task_row.description,
                    task_row.result_summary,
                    list(task_row.sources or []),
                )
            )
            continue

        # Resume: a previously-failed task stays failed; surface to synthesis.
        if task_row.status == TaskStatus.failed:
            failed_tasks.append((task_row.description, "previously failed"))
            continue

        # Resume: restart anything that was in_progress.
        if task_row.status == TaskStatus.in_progress:
            await _set_task_status(task_row.id, TaskStatus.pending)

        ok = await _execute_task(
            session_id=session_id,
            task_id=task_row.id,
            goal=goal,
            task_description=task_row.description,
            completed=completed_summaries,
        )
        if ok:
            # Reload the freshly-committed summary so synthesis sees it.
            async with session_scope() as db:
                refreshed = await db.get(Task, task_row.id)
                if refreshed and refreshed.result_summary:
                    completed_summaries.append(
                        (
                            refreshed.description,
                            refreshed.result_summary,
                            list(refreshed.sources or []),
                        )
                    )
        else:
            # Task failed — synthesis must know so the Limitations section
            # can name it explicitly rather than silently dropping coverage.
            failed_tasks.append((task_row.description, "execution failed"))

    # --- Phase 3: synthesis
    report = await _synthesize(goal, completed_summaries, failed_tasks, session_id)
    if not report:
        await _set_session_status(session_id, SessionStatus.failed)
        await bus.publish(
            AgentEvent(
                session_id=session_id,
                type=EventTypes.SESSION_FAILED,
                data={"error": "synthesis produced no report"},
            )
        )
        return

    async with session_scope() as db:
        sess = await db.get(Session, session_id)
        if sess:
            sess.final_report = report
    await bus.publish(
        AgentEvent(
            session_id=session_id,
            type=EventTypes.REPORT_READY,
            data={"report": report},
        )
    )

    # --- Phase 4: output guardrail (informational)
    summaries_only = [s for (_d, s, _u) in completed_summaries]
    verification = await verify_report(report, summaries_only)
    async with session_scope() as db:
        sess = await db.get(Session, session_id)
        if sess:
            notes = verification.notes or ""
            if verification.unsupported_claims:
                bullets = "\n".join(f"- {c}" for c in verification.unsupported_claims)
                notes = f"{notes}\n\n{bullets}".strip()
            sess.verification_notes = notes or None
    await bus.publish(
        AgentEvent(
            session_id=session_id,
            type=EventTypes.VERIFICATION_READY,
            data={
                "unsupported_claims": verification.unsupported_claims,
                "notes": verification.notes,
            },
        )
    )

    await _set_session_status(session_id, SessionStatus.completed)
    log.info(Events.SESSION_COMPLETED)
    await bus.publish(
        AgentEvent(
            session_id=session_id, type=EventTypes.SESSION_COMPLETED, data={}
        )
    )


# --- Phase 1: planning -----------------------------------------------------


async def _plan_or_load(
    goal: str, session_id: UUID, *, resume: bool
) -> list[TaskDTO]:
    """Return tasks in order. Plans only if this is a fresh session or resume
    finds an empty plan."""
    if resume:
        existing = await _load_tasks(session_id)
        if existing:
            return existing

    await _set_session_status(session_id, SessionStatus.planning)
    log.info(Events.SESSION_PLANNING)

    doc_filenames = await _list_session_documents(session_id)
    doc_note = (
        f"\n\nUploaded documents available for this session: {', '.join(doc_filenames)}."
        if doc_filenames
        else ""
    )
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Current date: {today}.\n\n"
                f"Research goal:\n\n{goal}{doc_note}"
            ),
        },
    ]

    try:
        response = await complete(
            purpose=LLMPurpose.plan,
            messages=messages,
            tools=[_CREATE_PLAN_FUNCTION],
            tool_choice={"type": "function", "function": {"name": "create_plan"}},
            temperature=0.3,
            session_id=session_id,
        )
    except Exception as exc:
        log.exception("plan.failed", error=str(exc))
        raise RuntimeError(f"planning call failed: {exc}") from exc

    plan = _parse_plan_response(response)
    if not plan or not plan.tasks:
        log.warning("plan.empty")
        return []

    tasks = await _persist_plan(session_id, plan)
    log.info(Events.SESSION_PLAN_READY, task_count=len(tasks))
    await bus.publish(
        AgentEvent(
            session_id=session_id,
            type=EventTypes.PLAN_READY,
            data={
                "tasks": [
                    {
                        "id": str(t.id),
                        "order_index": t.order_index,
                        "description": t.description,
                        "status": t.status.value,
                    }
                    for t in tasks
                ]
            },
        )
    )
    return tasks


def _parse_plan_response(response) -> Plan | None:
    if response.type != "tool_calls" or not response.calls:
        return None
    call = response.calls[0]
    if call.name != "create_plan" or not call.decoded:
        return None
    try:
        return Plan.model_validate(call.arguments)
    except ValidationError:
        return None


async def _persist_plan(session_id: UUID, plan: Plan) -> list[TaskDTO]:
    async with session_scope() as db:
        rows = [
            Task(
                session_id=session_id,
                order_index=i,
                description=t.description,
                status=TaskStatus.pending,
            )
            for i, t in enumerate(plan.tasks)
        ]
        db.add_all(rows)
        await db.flush()
        return [TaskDTO.model_validate(r) for r in rows]


# --- Phase 2: per-task execution -------------------------------------------


_EXECUTOR_TOOL_NAMES = [
    "web_search",
    "fetch_url",
    "search_memory",
    "search_documents",
    "finish_task",
]

# Inner-loop scratchpad: when the conversation grows past 3 tool exchanges,
# older ones are folded into a short text summary stored as a synthetic
# assistant message at the head of the chain.
_TOOL_EXCHANGES_VERBATIM = 3


@dataclass
class _ToolExchange:
    """One model-call → tool-execution pair from the inner loop."""

    assistant_message: dict[str, Any]
    tool_messages: list[dict[str, Any]]


async def _execute_task(
    *,
    session_id: UUID,
    task_id: UUID,
    goal: str,
    task_description: str,
    completed: list[tuple[str, str, list[str]]],
) -> bool:
    """Run the inner tool loop for one task. Returns True on success."""

    # Bind the session contextvar so tools (fetch_url, search_*) can resolve it.
    # Also give fetch_url a fresh per-task set of seen URLs so it can dedup.
    from src.tools.base import fetched_urls as _fetched_urls_var
    token = tools_pkg.current_session_id.set(session_id)
    urls_token = _fetched_urls_var.set(set())
    try:
        with bind_task(task_id):
            await _set_task_status(task_id, TaskStatus.in_progress)
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.TASK_STATUS_CHANGED,
                    data={
                        "task_id": str(task_id),
                        "status": TaskStatus.in_progress.value,
                    },
                )
            )
            log.info(Events.TASK_STARTED, description=task_description)

            try:
                ok = await _inner_tool_loop(
                    session_id=session_id,
                    task_id=task_id,
                    goal=goal,
                    task_description=task_description,
                    completed=completed,
                )
            except Exception as exc:
                log.exception(Events.TASK_FAILED, error=str(exc))
                await _set_task_status(task_id, TaskStatus.failed)
                await bus.publish(
                    AgentEvent(
                        session_id=session_id,
                        type=EventTypes.TASK_STATUS_CHANGED,
                        data={
                            "task_id": str(task_id),
                            "status": TaskStatus.failed.value,
                            "error": str(exc),
                        },
                    )
                )
                return False

            status = TaskStatus.done if ok else TaskStatus.failed
            if not ok:
                await _set_task_status(task_id, status)
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.TASK_STATUS_CHANGED,
                    data={"task_id": str(task_id), "status": status.value},
                )
            )
            log.info(
                Events.TASK_COMPLETED if ok else Events.TASK_FAILED,
                status=status.value,
            )
            return ok
    finally:
        tools_pkg.current_session_id.reset(token)
        _fetched_urls_var.reset(urls_token)


async def _inner_tool_loop(
    *,
    session_id: UUID,
    task_id: UUID,
    goal: str,
    task_description: str,
    completed: list[tuple[str, str, list[str]]],
) -> bool:
    """Drive the tool-call loop until finish_task succeeds or we run out of
    iterations / corrective attempts. Returns True on a successful finish."""

    # Build the initial system + user context. Completed-task summaries are
    # truncated to 200 tokens each to keep the prompt focused.
    completed_block = _format_completed(completed)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    user_block = (
        f"Current date: {today}.\n\n"
        f"OVERALL GOAL:\n{goal}\n\n"
        f"COMPLETED SO FAR:\n{completed_block or '(none)'}\n\n"
        f"YOUR CURRENT TASK:\n{task_description}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {"role": "user", "content": user_block},
    ]
    all_tools = tools_pkg.openai_schemas_for(_EXECUTOR_TOOL_NAMES)
    finish_only_tools = tools_pkg.openai_schemas_for(["finish_task"])

    exchanges: list[_ToolExchange] = []
    scratch_lines: list[str] = []
    correction_attempts = 0
    max_iters = settings.max_tool_iterations_per_task

    for iteration in range(max_iters):
        log.info(Events.TASK_TOOL_ITERATION, iteration=iteration)

        # Budget-aware tool selection:
        # - Last iteration: only finish_task is offered, and forced. Model
        #   commits with whatever evidence it has.
        # - Second-to-last: all tools, but a hint reminds the model this is
        #   its last chance to gather before forced commit.
        is_last = iteration == max_iters - 1
        is_penultimate = iteration == max_iters - 2

        if is_last:
            tools = finish_only_tools
            tool_choice: str | dict = {
                "type": "function",
                "function": {"name": "finish_task"},
            }
        else:
            tools = all_tools
            tool_choice = "auto"

        budget_hint = ""
        if is_penultimate:
            budget_hint = (
                "\n\n[Budget notice] You have one iteration left after this one. "
                "Use this turn to gather final evidence; the next turn will be "
                "restricted to finish_task only."
            )
        elif is_last:
            budget_hint = (
                "\n\n[Budget notice] This is your final iteration. Call finish_task "
                "with the best summary you can produce from the evidence you've "
                "already gathered. Cite the sources you actually saw."
            )

        ctx = _build_context_window(messages, exchanges, scratch_lines)
        if budget_hint:
            ctx = ctx + [{"role": "user", "content": budget_hint.strip()}]

        response = await complete(
            purpose=LLMPurpose.decide,
            messages=ctx,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=not is_last,
            temperature=0.2,
            session_id=session_id,
            task_id=task_id,
        )

        if response.type != "tool_calls" or not response.calls:
            # Model gave up and emitted text; treat as a soft failure of
            # the task — no result to persist.
            log.info("task.no_tool_call", iteration=iteration)
            return False

        # Assistant message that issued the calls (recorded so the model
        # sees its own prior turn in subsequent iterations).
        assistant_msg = _assistant_message_from_calls(response.calls)
        tool_results = await _execute_parallel_tools(
            response.calls, session_id=session_id, task_id=task_id
        )

        # Check for a successful finish_task before recording the exchange:
        # if successful, we can short-circuit and not extend the conversation.
        finish_outcome = _check_finish(response.calls, tool_results)
        if finish_outcome is _FinishOutcome.SUCCESS:
            payload = next(
                tr for tc, tr in zip(response.calls, tool_results) if tc.name == "finish_task"
            )
            output = payload["output"]
            await _persist_task_result(
                task_id,
                result_summary=output["result_summary"],
                sources=list(output.get("sources", [])),
            )
            return True
        if finish_outcome is _FinishOutcome.CORRECTION_REQUIRED:
            # On the forced-finish iteration, salvage the result_summary the
            # model produced rather than wasting it. The schema rejected the
            # call (likely empty sources), but the prose is still useful —
            # better a published report with a synthetic-source note than a
            # silently-discarded summary.
            if is_last:
                salvaged = _salvage_finish_summary(response.calls)
                if salvaged is not None:
                    log.info("task.finish_salvaged_on_forced_finish")
                    await _persist_task_result(
                        task_id,
                        result_summary=salvaged,
                        sources=["(no sources cited; forced-finish salvage)"],
                    )
                    return True
                # No salvageable prose either — fall through to the
                # iterations-exhausted return at loop end.
            else:
                correction_attempts += 1
                if correction_attempts > 2:
                    log.info("task.finish_corrections_exhausted")
                    return False

        exchanges.append(
            _ToolExchange(
                assistant_message=assistant_msg, tool_messages=tool_results_to_messages(tool_results)
            )
        )
        if len(exchanges) > _TOOL_EXCHANGES_VERBATIM:
            aged = exchanges.pop(0)
            for tm in aged.tool_messages:
                name = tm.get("name", "tool")
                content = tm.get("content", "")
                hint = content[:200].replace("\n", " ")
                scratch_lines.append(f"- {name}: {hint}")

    log.info("task.iterations_exhausted", iterations=max_iters)
    return False


# --- Context-window assembly ----------------------------------------------


def _build_context_window(
    base_messages: list[dict[str, Any]],
    exchanges: list[_ToolExchange],
    scratch_lines: list[str],
) -> list[dict[str, Any]]:
    """Stitch base + recent exchanges into a budget-bounded prompt.

    Recent exchanges (capped at _TOOL_EXCHANGES_VERBATIM) are appended verbatim.
    `scratch_lines` (built incrementally as exchanges age out) becomes a single
    synthetic 'scratchpad' assistant message. If the result still exceeds the
    token budget we drop oldest items, but always as full assistant+tool-message
    units so the OpenAI API's tool-pairing requirement is preserved.
    """
    out = list(base_messages)
    scratch_msg: dict[str, Any] | None = None

    if scratch_lines:
        scratch_msg = {
            "role": "assistant",
            "content": "Earlier tool activity (summarized):\n" + "\n".join(scratch_lines),
        }
        out.append(scratch_msg)

    working_exchanges = list(exchanges)
    for ex in working_exchanges:
        out.append(ex.assistant_message)
        out.extend(ex.tool_messages)

    budget = settings.token_budget_per_call

    def _total() -> int:
        return sum(_count_message_tokens(m) for m in out)

    # Drop in priority order: scratchpad first (no tool pairing to worry about),
    # then whole exchanges from the oldest.
    while _total() > budget and scratch_msg is not None and scratch_msg in out:
        out.remove(scratch_msg)
        log.info(
            Events.CONTEXT_TRUNCATED,
            dropped="scratchpad",
            tokens_after=_total(),
        )
        scratch_msg = None

    while _total() > budget and working_exchanges:
        oldest = working_exchanges.pop(0)
        out.remove(oldest.assistant_message)
        for tm in oldest.tool_messages:
            if tm in out:
                out.remove(tm)
        log.info(
            Events.CONTEXT_TRUNCATED,
            dropped="exchange",
            tool_msgs=len(oldest.tool_messages),
            tokens_after=_total(),
        )

    return out


def _format_completed(
    completed: list[tuple[str, str, list[str]]],
) -> str:
    if not completed:
        return ""
    parts: list[str] = []
    for desc, summary, sources in completed:
        truncated = _truncate_to_tokens(summary, 200)
        src_line = f" Sources: {', '.join(sources[:5])}" if sources else ""
        parts.append(f"- {desc}\n  {truncated}{src_line}")
    return "\n".join(parts)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "…"


# --- Tool dispatch helpers -------------------------------------------------


class _FinishOutcome(enum.Enum):
    SUCCESS = "success"
    CORRECTION_REQUIRED = "correction_required"
    NOT_PRESENT = "not_present"


def _check_finish(calls, results) -> _FinishOutcome:
    """Did this turn call finish_task? Was it accepted? Did it need correction?"""
    for call, result in zip(calls, results):
        if call.name != "finish_task":
            continue
        if result.get("error"):
            return _FinishOutcome.CORRECTION_REQUIRED
        return _FinishOutcome.SUCCESS
    return _FinishOutcome.NOT_PRESENT


def _salvage_finish_summary(calls) -> str | None:
    """Extract a usable result_summary from a rejected finish_task call.

    Used only on the forced-finish iteration when the schema rejected the
    model's call (empty sources, etc.). Returns the prose if present and
    non-trivial; None if there's nothing worth saving.
    """
    for call in calls:
        if call.name != "finish_task":
            continue
        summary = (call.arguments or {}).get("result_summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    return None


def _assistant_message_from_calls(calls) -> dict[str, Any]:
    """Reconstruct the OpenAI assistant message that issued these tool calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.raw_arguments or "{}"},
            }
            for c in calls
        ],
    }


async def _execute_parallel_tools(
    calls, *, session_id: UUID, task_id: UUID
) -> list[dict[str, Any]]:
    """Run all requested tool calls concurrently. Returns aligned results in
    call order. Each result is a dict {tool_call_id, output|error}."""

    async def _one(call):
        await bus.publish(
            AgentEvent(
                session_id=session_id,
                type=EventTypes.TOOL_INVOKED,
                data={
                    "task_id": str(task_id),
                    "tool": call.name,
                    "input_preview": _truncate_to_tokens(call.raw_arguments or "", 50),
                },
            )
        )
        try:
            if not call.decoded:
                return {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "error": (
                        "Arguments were not valid JSON. Please retry with a "
                        "valid JSON object."
                    ),
                }
            output_model = await tools_pkg.invoke(call.name, call.arguments)
            output = output_model.model_dump()
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.TOOL_COMPLETED,
                    data={"task_id": str(task_id), "tool": call.name},
                )
            )
            return {
                "tool_call_id": call.id,
                "name": call.name,
                "output": output,
            }
        except ValidationError as exc:
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.TOOL_FAILED,
                    data={
                        "task_id": str(task_id),
                        "tool": call.name,
                        "error": "validation",
                    },
                )
            )
            return {
                "tool_call_id": call.id,
                "name": call.name,
                "error": f"Input validation failed: {exc.errors()}",
            }
        except (KeyError, ToolError) as exc:
            await bus.publish(
                AgentEvent(
                    session_id=session_id,
                    type=EventTypes.TOOL_FAILED,
                    data={
                        "task_id": str(task_id),
                        "tool": call.name,
                        "error": str(exc),
                    },
                )
            )
            return {
                "tool_call_id": call.id,
                "name": call.name,
                "error": str(exc),
            }

    return await asyncio.gather(*(_one(c) for c in calls))


def tool_results_to_messages(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render parallel tool results as a list of `role: tool` messages."""
    import json as _json

    messages: list[dict[str, Any]] = []
    for r in results:
        if "output" in r:
            content = _json.dumps(r["output"], default=str)
        else:
            content = _json.dumps({"error": r.get("error", "tool failed")})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": r["tool_call_id"],
                "name": r["name"],
                "content": content,
            }
        )
    return messages


# --- Phase 3: synthesis ----------------------------------------------------


async def _synthesize(
    goal: str,
    completed: list[tuple[str, str, list[str]]],
    failed: list[tuple[str, str]],
    session_id: UUID,
) -> str:
    """Generate a structured markdown report from the completed-task summaries."""
    if not completed:
        log.warning("synthesize.no_completed_tasks")
        return ""

    task_block_parts = []
    for i, (desc, summary, sources) in enumerate(completed):
        src_block = "\n".join(f"  - {s}" for s in sources) if sources else "  (no sources)"
        task_block_parts.append(
            f"### Task {i + 1}: {desc}\n\n{summary}\n\nSources:\n{src_block}"
        )
    task_block = "\n\n".join(task_block_parts)

    failed_block = ""
    if failed:
        failed_lines = "\n".join(
            f"- {desc} (reason: {reason})" for desc, reason in failed
        )
        failed_block = (
            f"\n\nFAILED TASKS (not in the summaries above; must be acknowledged "
            f"in the Limitations section under a 'Tasks not completed' bullet):\n"
            f"{failed_lines}"
        )

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Current date: {today}.\n\n"
                f"RESEARCH GOAL:\n{goal}\n\n"
                f"COMPLETED TASK SUMMARIES:\n\n{task_block}"
                f"{failed_block}"
            ),
        },
    ]

    try:
        response = await complete(
            purpose=LLMPurpose.synthesize,
            messages=messages,
            temperature=0.3,
            session_id=session_id,
        )
    except Exception as exc:
        log.exception("synthesize.failed", error=str(exc))
        return ""

    if response.type != "text":
        log.warning("synthesize.unexpected_response_type")
        return ""
    return response.content


# --- DB helpers ------------------------------------------------------------


async def _load_tasks(session_id: UUID) -> list[TaskDTO]:
    async with session_scope() as db:
        result = await db.execute(
            select(Task).where(Task.session_id == session_id).order_by(Task.order_index)
        )
        return [TaskDTO.model_validate(r) for r in result.scalars().all()]


async def _list_session_documents(session_id: UUID) -> list[str]:
    """Returns filenames of documents bound to this session (or globals)."""
    async with session_scope() as db:
        result = await db.execute(
            select(Document.filename).where(
                (Document.session_id == session_id) | (Document.session_id.is_(None))
            )
        )
        return list(result.scalars().all())


async def _set_session_status(session_id: UUID, status: SessionStatus) -> None:
    async with session_scope() as db:
        sess = await db.get(Session, session_id)
        if sess is not None:
            sess.status = status


async def _set_task_status(task_id: UUID, status: TaskStatus) -> None:
    async with session_scope() as db:
        t = await db.get(Task, task_id)
        if t is not None:
            t.status = status


async def _persist_task_result(
    task_id: UUID, *, result_summary: str, sources: list[str]
) -> None:
    async with session_scope() as db:
        t = await db.get(Task, task_id)
        if t is not None:
            t.result_summary = result_summary
            t.sources = sources
            t.status = TaskStatus.done


__all__ = ["run_session"]
