"""Application configuration loaded from environment.

All runtime knobs live here as a single Settings object so modules don't reach
into os.environ directly. Instantiated once at module import (`settings`) and
imported elsewhere — pydantic-settings handles .env loading and type coercion.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API keys ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    tavily_api_key: str = Field(default="", description="Tavily search API key")

    # --- Models ---
    planner_model: str = "gpt-4o"
    executor_model: str = "gpt-4o-mini"
    synthesizer_model: str = "gpt-4o"
    verifier_model: str = "gpt-4o-mini"
    summarizer_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Storage ---
    postgres_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/agent"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # --- Runtime knobs ---
    log_level: str = "INFO"
    max_tasks_per_session: int = 7
    max_tool_iterations_per_task: int = 6
    token_budget_per_call: int = 8000
    max_upload_bytes: int = 10 * 1024 * 1024
    max_docs_per_session: int = 10
    max_concurrent_sessions: int = 4


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
