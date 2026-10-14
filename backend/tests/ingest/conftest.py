"""Offline fixtures for the ingest tests: trimmed real Fed pages and a raw cache built from them."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest

from app.core.config import Settings
from app.ingest.http import FedClient

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fed"


@pytest.fixture
def fed_page() -> Callable[[str], bytes]:
    """Load a fixture page by file name, e.g. fed_page("monetary20250917a.htm")."""
    return lambda name: (FIXTURES / name).read_bytes()


def _category(name: str) -> str:
    if name.startswith("fomccalendars") or name.startswith("fomchistorical"):
        return "calendar"
    return "statements" if name.startswith("monetary") else "minutes"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A data/ directory whose raw cache holds every fixture page, laid out as the pipeline does."""
    for page in FIXTURES.glob("*.htm"):
        target = tmp_path / "raw" / _category(page.name) / page.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(page, target)
    return tmp_path


@pytest.fixture
def settings(data_dir: Path) -> Settings:
    return Settings(_env_file=None, environment="test", data_dir=data_dir, database_url=None)


def _no_network(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"unexpected network request: {request.url}")


@pytest.fixture
def offline_factory() -> Callable[..., FedClient]:
    """A client factory for the CLI whose transport fails the test on any network request."""

    def make(raw_dir: Path, **kwargs: Any) -> FedClient:
        return FedClient(
            raw_dir, transport=httpx2.MockTransport(_no_network), sleep=lambda _: None, **kwargs
        )

    return make
