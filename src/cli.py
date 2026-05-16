"""Typer-based CLI.

Commands:
- run <goal>            — start a new session, stream events to stdout.
- resume <session_id>   — resume an existing session.
- list-sessions         — show recent sessions and their statuses.
- show <session_id>     — render a session's tasks + final report.
- ingest <path>         — ingest a document (per-session or --global).
- list-documents        — list ingested documents.

The `run` and `resume` commands stream loop events using rich while the
agent runs in a background task.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from src.agent import run_session
from src.bootstrap import setup as bootstrap_setup
from src.db import session_scope
from src.events import AgentEvent, EventTypes, bus
from src.ingest import ingest_document
from src.models import Document, Session, SessionStatus, Task


app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


# --- run / resume ----------------------------------------------------------


def _run_command(goal: str, session_id: UUID, resume: bool) -> None:
    bootstrap_setup()

    async def _main():
        # Ensure the session row exists before run_session begins.
        if not resume:
            async with session_scope() as db:
                db.add(Session(id=session_id, goal=goal, status=SessionStatus.planning))

        # Run agent and event-streaming as two concurrent tasks; cancel the
        # streamer when the agent finishes (it'll also receive the close
        # sentinel via the event bus).
        agent_task = asyncio.create_task(run_session(goal, session_id, resume=resume))
        streamer = asyncio.create_task(_stream_events(session_id))
        try:
            await agent_task
        finally:
            streamer.cancel()
            try:
                await streamer
            except (asyncio.CancelledError, Exception):
                pass

    asyncio.run(_main())


async def _stream_events(session_id: UUID) -> None:
    """Print loop events to the console as they arrive."""
    async with bus.subscribe(session_id) as queue:
        while True:
            event: AgentEvent = await queue.get()
            if event.type == EventTypes._CLOSE:
                return
            _render_event(event)


def _render_event(event: AgentEvent) -> None:
    if event.type == EventTypes.PLAN_READY:
        tasks = event.data.get("tasks", [])
        console.print(f"[bold cyan]plan ready[/bold cyan] ({len(tasks)} tasks):")
        for t in tasks:
            console.print(f"  [dim]{t['order_index'] + 1}.[/dim] {t['description']}")
    elif event.type == EventTypes.TASK_STATUS_CHANGED:
        status = event.data.get("status", "?")
        tid = event.data.get("task_id", "")[:8]
        color = {"in_progress": "yellow", "done": "green", "failed": "red"}.get(
            status, "white"
        )
        console.print(f"  task [dim]{tid}[/dim] → [{color}]{status}[/{color}]")
    elif event.type == EventTypes.TOOL_INVOKED:
        tool = event.data.get("tool", "?")
        preview = event.data.get("input_preview", "")
        console.print(f"    [dim]→[/dim] {tool} [dim]{preview}[/dim]")
    elif event.type == EventTypes.TOOL_COMPLETED:
        console.print(f"    [dim]✓[/dim] {event.data.get('tool', '?')}")
    elif event.type == EventTypes.TOOL_FAILED:
        console.print(
            f"    [red]✗[/red] {event.data.get('tool', '?')}: "
            f"{event.data.get('error', '')}"
        )
    elif event.type == EventTypes.REPORT_READY:
        console.print("[bold green]report ready[/bold green]")
    elif event.type == EventTypes.VERIFICATION_READY:
        claims = event.data.get("unsupported_claims", [])
        if claims:
            console.print(
                f"[yellow]verification[/yellow]: {len(claims)} unsupported claim(s)"
            )
        else:
            console.print("[green]verification[/green]: report is well-grounded")
    elif event.type == EventTypes.SESSION_COMPLETED:
        console.print("[bold green]session completed[/bold green]")
    elif event.type == EventTypes.SESSION_FAILED:
        console.print(f"[bold red]session failed[/bold red]: {event.data.get('error', '')}")


@app.command()
def run(
    goal: str = typer.Argument(..., help="High-level research goal."),
    session: Optional[str] = typer.Option(
        None, "--session", help="Reuse an existing session id (otherwise a new uuid is generated)."
    ),
) -> None:
    """Start a new research session and stream events."""
    sid = UUID(session) if session else uuid4()
    console.print(f"[bold]session:[/bold] {sid}")
    _run_command(goal, sid, resume=False)


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Existing session id to resume."),
) -> None:
    """Resume a paused or failed session."""
    sid = UUID(session_id)
    console.print(f"[bold]resuming session:[/bold] {sid}")
    _run_command(goal="", session_id=sid, resume=True)


# --- read-only commands ----------------------------------------------------


@app.command("list-sessions")
def list_sessions(
    limit: int = typer.Option(20, "--limit", help="Max rows to show."),
) -> None:
    """List recent sessions."""
    bootstrap_setup()

    async def _main():
        async with session_scope() as db:
            result = await db.execute(
                select(Session).order_by(Session.created_at.desc()).limit(limit)
            )
            sessions = result.scalars().all()
            rows = [
                (
                    str(s.id),
                    s.status.value,
                    (s.goal or "")[:80],
                    s.created_at.isoformat() if s.created_at else "",
                )
                for s in sessions
            ]
        return rows

    rows = asyncio.run(_main())
    table = Table(title="Sessions")
    table.add_column("id")
    table.add_column("status")
    table.add_column("goal")
    table.add_column("created_at")
    for r in rows:
        table.add_row(*r)
    console.print(table)


@app.command()
def show(
    session_id: str = typer.Argument(..., help="Session id."),
) -> None:
    """Show a session's tasks and final report."""
    bootstrap_setup()
    sid = UUID(session_id)

    async def _main():
        async with session_scope() as db:
            sess = await db.get(Session, sid)
            if not sess:
                return None, []
            result = await db.execute(
                select(Task).where(Task.session_id == sid).order_by(Task.order_index)
            )
            tasks = list(result.scalars().all())
            # Detach
            sess_data = {
                "id": str(sess.id),
                "goal": sess.goal,
                "status": sess.status.value,
                "final_report": sess.final_report,
                "verification_notes": sess.verification_notes,
            }
            task_data = [
                {
                    "order_index": t.order_index,
                    "description": t.description,
                    "status": t.status.value,
                    "result_summary": t.result_summary,
                    "sources": t.sources or [],
                }
                for t in tasks
            ]
        return sess_data, task_data

    sess, tasks = asyncio.run(_main())
    if sess is None:
        console.print(f"[red]no session with id {sid}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]session[/bold] {sess['id']}  [dim]({sess['status']})[/dim]")
    console.print(f"[bold]goal:[/bold] {sess['goal']}")
    console.print()
    for t in tasks:
        status = t["status"]
        color = {"done": "green", "failed": "red", "in_progress": "yellow"}.get(
            status, "white"
        )
        console.print(
            f"  [{color}]{t['order_index'] + 1}. [{status}][/{color}] {t['description']}"
        )
        if t["result_summary"]:
            console.print(f"     [dim]{t['result_summary'][:200]}[/dim]")
    if sess["final_report"]:
        console.print()
        console.print("[bold]final report[/bold]")
        console.print(sess["final_report"])
    if sess["verification_notes"]:
        console.print()
        console.print("[bold yellow]verification notes[/bold yellow]")
        console.print(sess["verification_notes"])


# --- ingest / list-documents ----------------------------------------------


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the document."),
    session: Optional[str] = typer.Option(
        None, "--session", help="Bind to a session id (omit for a global document)."
    ),
    global_: bool = typer.Option(False, "--global", help="Make the document available across sessions."),
) -> None:
    """Ingest a document into the memory store."""
    bootstrap_setup()
    if session and global_:
        console.print("[red]--session and --global are mutually exclusive[/red]")
        raise typer.Exit(code=2)
    sid: Optional[UUID] = UUID(session) if session else None
    data = path.read_bytes()

    async def _main():
        result = await ingest_document(
            filename=path.name,
            data=data,
            session_id=None if global_ else sid,
        )
        return result

    result = asyncio.run(_main())
    console.print(
        f"[green]ingested[/green] {result.filename} "
        f"(document_id={result.document_id}, chunks={result.chunk_count})"
    )


@app.command("list-documents")
def list_documents(
    session: Optional[str] = typer.Option(
        None, "--session", help="Filter by session id."
    ),
) -> None:
    """List ingested documents."""
    bootstrap_setup()
    sid = UUID(session) if session else None

    async def _main():
        async with session_scope() as db:
            stmt = select(Document).order_by(Document.created_at.desc())
            if sid is not None:
                stmt = stmt.where(Document.session_id == sid)
            result = await db.execute(stmt)
            return [
                (
                    str(d.id),
                    d.filename,
                    "(global)" if d.session_id is None else str(d.session_id)[:8],
                    str(d.chunk_count),
                    f"{d.byte_size} B",
                )
                for d in result.scalars().all()
            ]

    rows = asyncio.run(_main())
    table = Table(title="Documents")
    table.add_column("id")
    table.add_column("filename")
    table.add_column("scope")
    table.add_column("chunks")
    table.add_column("size")
    for r in rows:
        table.add_row(*r)
    console.print(table)


def main() -> None:  # pragma: no cover — typer wires this as the script entrypoint
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
