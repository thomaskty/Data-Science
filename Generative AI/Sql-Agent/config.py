from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model_name: str
    db_path: Path
    project_name: str
    langsmith_endpoint: str | None
    langsmith_tracing_v2: str | None


def load_settings() -> Settings:
    """Load environment config for the agent runtime."""
    load_dotenv()

    model_name = os.getenv("MODEL_NAME", "openai:gpt-5-mini")
    db_path = Path(os.getenv("DB_PATH", "Chinook.db")).expanduser().resolve()
    project_name = os.getenv("LANGSMITH_PROJECT", "project")

    # Keep user-provided endpoint/keys untouched; only mirror required vars.
    langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
    langsmith_tracing_v2 = os.getenv("LANGSMITH_TRACING_V2")

    os.environ["LANGSMITH_PROJECT"] = project_name
    if langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint
    if langsmith_tracing_v2:
        os.environ["LANGSMITH_TRACING_V2"] = langsmith_tracing_v2

    return Settings(
        model_name=model_name,
        db_path=db_path,
        project_name=project_name,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_tracing_v2=langsmith_tracing_v2,
    )
