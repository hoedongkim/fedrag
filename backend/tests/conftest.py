"""Fixtures that build the app on isolated settings, never the developer's .env."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app

MakeClient = Callable[..., TestClient]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "db: needs a throwaway Postgres in TEST_DATABASE_URL; skipped when it is not set "
        "(see tests/db/conftest.py)",
    )
    config.addinivalue_line(
        "markers",
        "live: calls the real OpenAI API with the key in backend/.env (paid); skipped unless "
        "RUN_LIVE_TESTS=1",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="calls the paid OpenAI API: set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's exported variables out of every test."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def make_client(tmp_path: Path) -> MakeClient:
    """Return a factory for clients whose settings ignore .env; keyword args override fields."""

    def _make(**overrides: Any) -> TestClient:
        fields = {
            "environment": "test",
            "warmup": False,  # no background model loading in tests: a test turns it on
            "openai_api_key": None,
            "data_dir": tmp_path,
            "chroma_dir": tmp_path / "chroma",
            "index_dir": tmp_path / "index",
            **overrides,
        }
        settings = Settings(_env_file=None, **fields)
        app = create_app(settings)
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)

    return _make


@pytest.fixture
def client(make_client: MakeClient) -> TestClient:
    return make_client()


@pytest.fixture
def index_ready(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], list[Settings]]:
    """Stand in for app.rag.index.index_ready (ready or not); records the settings it saw."""

    def _install(ready: bool) -> list[Settings]:
        seen: list[Settings] = []

        def fake_index_ready(settings: Settings) -> bool:
            seen.append(settings)
            return ready

        monkeypatch.setattr("app.rag.index.index_ready", fake_index_ready)
        return seen

    return _install
