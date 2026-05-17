"""initial schema: sessions, tasks, llm_calls, tool_calls, documents

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-15

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Enum names are declared once so upgrade/downgrade reference the same objects.
# `create_type=False` on the column-level references prevents SQLAlchemy from
# auto-emitting CREATE TYPE when the table is created — we manage the lifecycle
# explicitly in upgrade()/downgrade() with checkfirst=True.
SESSION_STATUS = sa.Enum(
    "planning",
    "running",
    "completed",
    "failed",
    "paused",
    name="session_status",
)
TASK_STATUS = sa.Enum(
    "pending",
    "in_progress",
    "done",
    "failed",
    "skipped",
    name="task_status",
)
LLM_PURPOSE = sa.Enum(
    "plan",
    "decide",
    "summarize",
    "synthesize",
    "verify",
    name="llm_purpose",
)

SESSION_STATUS_REF = postgresql.ENUM(
    "planning", "running", "completed", "failed", "paused",
    name="session_status", create_type=False,
)
TASK_STATUS_REF = postgresql.ENUM(
    "pending", "in_progress", "done", "failed", "skipped",
    name="task_status", create_type=False,
)
LLM_PURPOSE_REF = postgresql.ENUM(
    "plan", "decide", "summarize", "synthesize", "verify",
    name="llm_purpose", create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    SESSION_STATUS.create(bind, checkfirst=True)
    TASK_STATUS.create(bind, checkfirst=True)
    LLM_PURPOSE.create(bind, checkfirst=True)

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column(
            "status",
            SESSION_STATUS_REF,
            nullable=False,
            server_default="planning",
        ),
        sa.Column("final_report", sa.Text(), nullable=True),
        sa.Column("verification_notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", TASK_STATUS_REF, nullable=False, server_default="pending"),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_tasks_session_id", "tasks", ["session_id"])
    op.create_index(
        "ix_tasks_session_order", "tasks", ["session_id", "order_index"], unique=True
    )

    op.create_table(
        "llm_calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("purpose", LLM_PURPOSE_REF, nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("prompt", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_llm_calls_session_id", "llm_calls", ["session_id"])
    op.create_index("ix_llm_calls_task_id", "llm_calls", ["task_id"])

    op.create_table(
        "tool_calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("input", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_tool_calls_session_id", "tool_calls", ["session_id"])
    op.create_index("ix_tool_calls_task_id", "tool_calls", ["task_id"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_session_id", "documents", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_session_id", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_tool_calls_task_id", table_name="tool_calls")
    op.drop_index("ix_tool_calls_session_id", table_name="tool_calls")
    op.drop_table("tool_calls")

    op.drop_index("ix_llm_calls_task_id", table_name="llm_calls")
    op.drop_index("ix_llm_calls_session_id", table_name="llm_calls")
    op.drop_table("llm_calls")

    op.drop_index("ix_tasks_session_order", table_name="tasks")
    op.drop_index("ix_tasks_session_id", table_name="tasks")
    op.drop_table("tasks")

    op.drop_table("sessions")

    bind = op.get_bind()
    LLM_PURPOSE.drop(bind, checkfirst=True)
    TASK_STATUS.drop(bind, checkfirst=True)
    SESSION_STATUS.drop(bind, checkfirst=True)
