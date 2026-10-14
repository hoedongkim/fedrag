"""Application settings, read from environment variables and backend/.env."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration; every field can be set by its upper-case environment variable."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: Literal["development", "test", "production"] = "development"
    openai_api_key: SecretStr | None = None
    database_url: SecretStr | None = None
    llm_model: str = "gpt-5.4-mini-2026-03-17"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"  # commit of reranker_model
    # The retriever preset (app/rag/retriever_presets.json, or "phase3") that POST /api/ask,
    # /api/ask/stream, make ask and make smoke use: the one place the production default is set.
    # "tuned" since Phase 7, by the TEST switch rule in docs/rag-tuning.md; "phase3" is
    # the Phase 3 retriever.
    retriever_preset: str = "tuned"
    data_dir: Path = REPO_ROOT / "data"
    chroma_dir: Path = REPO_ROOT / "data" / "chroma"
    index_dir: Path = REPO_ROOT / "data" / "index"
    # A JSON list (["https://a.example"]), one origin, or comma-separated origins: see _origins.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # Load BM25 and the reranker in a background thread at startup, so the first hybrid_rerank
    # question is not ~5 s slower than the rest; the test client turns it off. Never on the
    # public demo, which does not answer questions.
    warmup: bool = True
    # The public demo (PUBLIC_DEMO=true): every feature that would call OpenAI is off, so it
    # needs no OPENAI_API_KEY and costs nothing to run. Ask the Fed answers 403 (GET
    # /api/ask/examples still serves recorded answers), a comparison without a stored analysis
    # reports analysis_status "disabled_on_demo", and startup loads no embeddings, vector store
    # or reranker. Everything that reads stored results works as usual.
    public_demo: bool = False

    @field_validator("openai_api_key", "database_url", mode="before")
    @classmethod
    def _blank_secret_is_unset(cls, value: object) -> object:
        """Treat `OPENAI_API_KEY=` or `DATABASE_URL=` (empty or whitespace) as not set at all."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _origins(cls, value: object) -> object:
        """Accept CORS_ORIGINS as a JSON list, a single origin or comma-separated origins.

        A browser sends its Origin without a trailing slash, so one is stripped here: otherwise
        "https://x.vercel.app/" would be accepted and never match. Blank items are dropped.
        """
        if isinstance(value, str):
            text = value.strip()
            value = json.loads(text) if text.startswith("[") else text.split(",")
        if isinstance(value, list | tuple):
            items = (o.strip().rstrip("/") if isinstance(o, str) else o for o in value)
            return [o for o in items if o != ""]
        return value

    @field_validator("retriever_preset")
    @classmethod
    def _preset_name(cls, value: str) -> str:
        """Whether the preset exists is checked when the app starts (create_app) and whenever a
        retriever is built (app.rag.config.production_config): this module stays free of RAG
        imports."""
        if not value.strip():
            raise ValueError("RETRIEVER_PRESET must name a preset, e.g. tuned or phase3")
        return value.strip()

    @field_validator("data_dir", "chroma_dir", "index_dir")
    @classmethod
    def _relative_to_repo_root(cls, value: Path) -> Path:
        """Resolve relative paths against the repo root so the server works from any cwd."""
        return value if value.is_absolute() else REPO_ROOT / value

    @property
    def llm_configured(self) -> bool:
        return self.openai_api_key is not None

    @property
    def database_configured(self) -> bool:
        return self.database_url is not None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings; override this dependency in tests."""
    return Settings()
