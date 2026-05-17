"""Input and output guardrails.

`check_goal` runs before planning: empty/oversize/obvious-injection checks
that are cheap and catch high-value classes of bad input.

`verify_report` runs after synthesis: the verifier model reads the report
plus the source summaries and lists unsupported claims via a forced function
call. The result is informational — the agent loop writes it to
`sessions.verification_notes` and shows it to the user, but does NOT block
publication. Censoring would be worse than surfacing: users decide what to
trust.
"""
from __future__ import annotations

import re
from uuid import UUID

from src.logging_setup import Events, get_logger
from src.models import LLMPurpose
from src.prompts import VERIFIER_SYSTEM
from src.llm import complete
from src.schemas import (
    GuardrailResult,
    ReportVerificationArgs,
    VerificationResult,
    openai_function_from_model,
)


log = get_logger(__name__)


# --- Input guardrail --------------------------------------------------------


_MAX_GOAL_CHARS = 2000

# Short, conservative blocklist. A long list catches more patterns but
# introduces false positives (legitimate research goals discussing AI safety
# would otherwise be flagged). These three target the canonical injection
# vectors that distinguish "research goal" from "prompt-injection payload."
_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all |the )?(?:previous|prior|above) (?:instructions|prompts)", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"disregard (?:all |the )?(?:previous|prior|above) (?:instructions|prompts)", re.I),
]


async def check_goal(goal: str) -> GuardrailResult:
    """Validate a user-supplied research goal before planning.

    Rules:
    - Must be non-empty after stripping whitespace.
    - Must be <= 2000 characters.
    - Must not contain obvious prompt-injection patterns.
    """
    stripped = (goal or "").strip()
    if not stripped:
        result = GuardrailResult(passed=False, reason="goal is empty")
        log.info(
            Events.GUARDRAIL_TRIGGERED,
            check="check_goal",
            passed=False,
            reason=result.reason,
        )
        return result

    if len(stripped) > _MAX_GOAL_CHARS:
        result = GuardrailResult(
            passed=False,
            reason=f"goal exceeds {_MAX_GOAL_CHARS} characters",
        )
        log.info(
            Events.GUARDRAIL_TRIGGERED,
            check="check_goal",
            passed=False,
            reason=result.reason,
            length=len(stripped),
        )
        return result

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            result = GuardrailResult(
                passed=False,
                reason=f"goal contains prompt-injection pattern: {pattern.pattern}",
            )
            log.info(
                Events.GUARDRAIL_TRIGGERED,
                check="check_goal",
                passed=False,
                pattern=pattern.pattern,
            )
            return result

    log.info(
        Events.GUARDRAIL_TRIGGERED,
        check="check_goal",
        passed=True,
        length=len(stripped),
    )
    return GuardrailResult(passed=True)


# --- Output guardrail -------------------------------------------------------
#
# The verifier reads the report and source summaries, returning a structured
# list of unsupported claims via a forced function call. The schema is
# constrained server-side; we still validate the result with Pydantic.


_REPORT_VERIFICATION_FUNCTION = openai_function_from_model(
    ReportVerificationArgs,
    name="report_verification",
    description=(
        "Return a structured verification of the report. List any claims "
        "not supported by the provided source summaries."
    ),
)


async def verify_report(
    report: str,
    source_summaries: list[str],
    *,
    failed_tasks: list[tuple[str, str]] | None = None,
    session_id: UUID | None = None,
) -> VerificationResult:
    """Call the verifier model on a report. Always returns a result.

    Never raises on verifier failure: returns an empty VerificationResult
    with a note explaining the failure so the session can still complete.

    failed_tasks: list of (description, reason) for tasks the agent attempted
    but couldn't complete. Passed to the verifier so it knows the report's
    "Tasks not completed" Limitations entries are structural acknowledgements,
    not claims to be checked against source summaries.
    """
    if not (report or "").strip():
        result = VerificationResult(unsupported_claims=[], notes="empty report")
        log.info(
            Events.GUARDRAIL_TRIGGERED,
            check="verify_report",
            passed=True,
            note="empty report",
        )
        return result

    # We pass the source summaries as a numbered list so the verifier prompt
    # can reference them by index if it wants.
    sources_block = (
        "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(source_summaries))
        if source_summaries
        else "(no source summaries provided)"
    )

    failed_block = ""
    if failed_tasks:
        failed_lines = "\n".join(
            f"- {desc} (reason: {reason})" for desc, reason in failed_tasks
        )
        failed_block = (
            "\n\nFAILED TASKS (the report's 'Tasks not completed' Limitations "
            "entries reference these; treat such entries as structural "
            "acknowledgements, not claims to flag):\n"
            f"{failed_lines}"
        )

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {
            "role": "user",
            "content": (
                "REPORT:\n\n"
                f"{report}\n\n"
                "SOURCE SUMMARIES:\n\n"
                f"{sources_block}"
                f"{failed_block}"
            ),
        },
    ]

    try:
        response = await complete(
            purpose=LLMPurpose.verify,
            messages=messages,
            tools=[_REPORT_VERIFICATION_FUNCTION],
            tool_choice={
                "type": "function",
                "function": {"name": "report_verification"},
            },
            temperature=0.0,
            session_id=session_id,
        )
    except Exception as exc:
        log.warning(
            Events.GUARDRAIL_TRIGGERED,
            check="verify_report",
            passed=False,
            error=str(exc),
        )
        return VerificationResult(
            unsupported_claims=[],
            notes=f"verifier failed: {exc}",
        )

    parsed = _parse_verifier_response(response)
    log.info(
        Events.GUARDRAIL_TRIGGERED,
        check="verify_report",
        passed=True,
        unsupported_claim_count=len(parsed.unsupported_claims),
    )
    return parsed


def _parse_verifier_response(response) -> VerificationResult:
    """Pull the verifier's structured output. Always returns a result; on
    any parse hiccup we fall back to an empty result with a diagnostic note.
    """
    if response.type != "tool_calls" or not response.calls:
        return VerificationResult(
            unsupported_claims=[],
            notes="verifier returned no tool call",
        )

    call = response.calls[0]
    if call.name != "report_verification":
        return VerificationResult(
            unsupported_claims=[],
            notes=f"verifier called unexpected function: {call.name}",
        )

    if not call.decoded:
        return VerificationResult(
            unsupported_claims=[],
            notes="verifier returned malformed JSON arguments",
        )

    args = call.arguments
    raw_claims = args.get("unsupported_claims", []) or []
    # Defensive: coerce to list[str] in case the model emits non-strings.
    claims = [str(c) for c in raw_claims if c]
    notes = str(args.get("notes", "") or "")
    return VerificationResult(unsupported_claims=claims, notes=notes)


__all__ = ["check_goal", "verify_report"]
