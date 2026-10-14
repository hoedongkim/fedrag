"""Tests for GET /api/health."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.core.config import Settings, get_settings

MakeClient = Callable[..., TestClient]


def test_health_shape(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "environment": "test",
        "llm_configured": False,
        "database_configured": False,
        "vector_store_ready": False,
        "demo_mode": False,
    }


@pytest.mark.parametrize("demo", [False, True], ids=["local", "public-demo"])
def test_demo_mode_follows_public_demo(make_client: MakeClient, demo: bool) -> None:
    assert make_client(public_demo=demo).get("/api/health").json()["demo_mode"] is demo


@pytest.mark.parametrize(
    ("api_key", "configured"),
    [(None, False), ("", False), ("   ", False), ("sk-test", True)],
)
def test_llm_configured(make_client: MakeClient, api_key: str | None, configured: bool) -> None:
    body = make_client(openai_api_key=api_key).get("/api/health").json()

    assert body["llm_configured"] is configured


@pytest.mark.parametrize(
    ("database_url", "configured"),
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("postgresql://fedrag:secret@127.0.0.1:55432/fedrag", True),
    ],
)
def test_database_configured(
    make_client: MakeClient, database_url: str | None, configured: bool
) -> None:
    response = make_client(database_url=database_url).get("/api/health")

    assert response.json()["database_configured"] is configured
    assert "secret" not in response.text


@pytest.mark.parametrize("ready", [False, True], ids=["not-built", "built"])
def test_vector_store_ready_follows_the_index(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch, ready: bool
) -> None:
    client = make_client()
    seen: list[Settings] = []

    def index_ready(settings: Settings) -> bool:
        seen.append(settings)
        return ready

    monkeypatch.setattr("app.api.routes.health.index_ready", index_ready)

    assert client.get("/api/health").json()["vector_store_ready"] is ready
    assert seen == [client.app.dependency_overrides[get_settings]()]


def test_vector_store_not_ready_with_an_empty_chroma(make_client: MakeClient) -> None:
    """The real index_ready: no manifest and no Chroma in the isolated dirs, and none created."""
    client = make_client()
    settings = client.app.dependency_overrides[get_settings]()

    assert client.get("/api/health").json()["vector_store_ready"] is False
    assert not settings.chroma_dir.exists() and not settings.index_dir.exists()
