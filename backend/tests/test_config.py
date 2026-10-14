"""Tests for how Settings reads .env files and paths, and for the retriever preset setting."""

from pathlib import Path

import pytest

from app.core.config import REPO_ROOT, Settings
from app.main import create_app

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_empty_key_in_env_file_is_not_configured(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\n")

    assert Settings(_env_file=env_file).llm_configured is False


def test_blank_database_url_in_env_file_is_not_configured(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=   \n")

    settings = Settings(_env_file=env_file)

    assert settings.database_url is None
    assert settings.database_configured is False


def test_database_url_is_kept_secret() -> None:
    url = "postgresql://fedrag:hunter2@ep-example-pooler.aws.neon.tech/neondb?sslmode=require"

    settings = Settings(_env_file=None, database_url=url)

    assert settings.database_configured is True
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == url
    assert "hunter2" not in repr(settings)


def test_env_example_matches_defaults() -> None:
    assert Settings(_env_file=ENV_EXAMPLE) == Settings(_env_file=None)


def test_relative_paths_resolve_against_repo_root() -> None:
    settings = Settings(_env_file=None, data_dir="data", chroma_dir="data/chroma")

    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.chroma_dir == REPO_ROOT / "data" / "chroma"


def test_an_unknown_retriever_preset_stops_the_app_at_startup() -> None:
    with pytest.raises(ValueError, match="RETRIEVER_PRESET: unknown retriever preset 'nope'"):
        create_app(Settings(_env_file=None, retriever_preset="nope", warmup=False))
    assert create_app(Settings(_env_file=None, retriever_preset="phase3", warmup=False))


@pytest.mark.parametrize(
    ("value", "demo"), [("true", True), ("1", True), ("false", False), ("0", False)]
)
def test_public_demo_reads_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch, value: str, demo: bool
) -> None:
    monkeypatch.setenv("PUBLIC_DEMO", value)

    assert Settings(_env_file=None).public_demo is demo


def test_the_public_demo_needs_no_key_and_is_off_by_default() -> None:
    assert Settings(_env_file=None).public_demo is False
    assert create_app(Settings(_env_file=None, public_demo=True, openai_api_key=None))


@pytest.mark.parametrize(
    ("value", "origins"),
    [
        ('["https://fedrag.vercel.app"]', ["https://fedrag.vercel.app"]),
        (
            '["https://fedrag.vercel.app/", " https://b.example "]',
            ["https://fedrag.vercel.app", "https://b.example"],
        ),
        ("https://fedrag.vercel.app", ["https://fedrag.vercel.app"]),
        ("https://fedrag.vercel.app/", ["https://fedrag.vercel.app"]),
        (
            "https://a.vercel.app, https://b.vercel.app,",
            ["https://a.vercel.app", "https://b.vercel.app"],
        ),
    ],
)
def test_cors_origins_accepts_a_json_list_one_origin_or_a_comma_list(
    monkeypatch: pytest.MonkeyPatch, value: str, origins: list[str]
) -> None:
    """The demo's CORS_ORIGINS variable is typed by hand: a plain URL must not stop startup,
    and a trailing slash (which a browser's Origin never has) must not make it never match."""
    monkeypatch.setenv("CORS_ORIGINS", value)

    settings = Settings(_env_file=None)

    assert settings.cors_origins == origins
    assert create_app(Settings(_env_file=None, public_demo=True, warmup=False))
