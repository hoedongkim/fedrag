"""Checks on the tracked smoke-check questions in data/eval/smoke.jsonl."""

import json

import pytest

from app.core.config import Settings
from app.db.load import DOCUMENTS_FILE
from app.rag.smoke import SMOKE_FILE, load_questions

SETTINGS = Settings(_env_file=None)
QUESTIONS = load_questions(SETTINGS.data_dir / SMOKE_FILE)


def test_smoke_set_shape() -> None:
    assert len(QUESTIONS) == 12
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)
    assert sum(not q.expected_doc_ids for q in QUESTIONS) == 1  # one unanswerable question
    assert any(q.as_of for q in QUESTIONS) and any(q.as_of is None for q in QUESTIONS)
    expected = {d for q in QUESTIONS for d in q.expected_doc_ids}
    assert {d.split("-")[0] for d in expected} == {"statement", "minutes"}
    years = {int(d.split("-")[1]) for d in expected}
    assert min(years) == 2015 and max(years) == 2026


def test_expected_documents_exist_and_were_public_by_as_of() -> None:
    path = SETTINGS.data_dir / "processed" / DOCUMENTS_FILE
    if not path.is_file():
        pytest.skip(f"{path} not found: run make ingest")
    released = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            released[row["doc_id"]] = row["release_date"]
    for q in QUESTIONS:
        for doc_id in q.expected_doc_ids:
            assert doc_id in released, (q.id, doc_id)
            assert q.as_of is None or released[doc_id] <= q.as_of.isoformat(), (q.id, doc_id)
