"""Fixtures for the evaluation tests: isolated settings and the fake question file."""

import json
from pathlib import Path

import pytest
from eval_fakes import QUESTIONS

from app.core.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        openai_api_key=None,
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        index_dir=tmp_path / "index",
    )


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "eval" / "rag_questions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS), encoding="utf-8")
    return path
