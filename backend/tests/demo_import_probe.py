"""Run by tests/test_demo.py in a fresh interpreter (pytest has torch, openai and Chroma loaded
long before): build the app as the public demo, without OPENAI_API_KEY and with the warm-up
setting on, call what the demo serves, and print JSON with the response of each call, the heavy
modules the process imported and its threads.

    python tests/demo_import_probe.py DATA_DIR INDEX_DIR CHROMA_DIR

The database is faked like tests/compare/test_api.py does, so the comparison runs end to end:
diff, cache lookup, the analyzer dependency.
"""

import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

# What the demo must never load: the reranker's stack, OpenAI's client, the vector store.
HEAVY = (
    "torch",
    "transformers",
    "sentence_transformers",
    "openai",
    "langchain_openai",
    "chromadb",
    "langchain_chroma",
)
BASE, TARGET = date(2024, 7, 31), date(2024, 9, 18)
PARAGRAPHS = {
    BASE: ("The Committee decided to maintain the target range at 5-1/4 to 5-1/2 percent.",),
    TARGET: ("The Committee decided to lower the target range to 4-3/4 to 5 percent.",),
}


def fake_database() -> Any:
    """A connection source whose statements come from PARAGRAPHS (app.db.compare patched)."""
    from app.compare.types import StatementText
    from app.db import compare as compare_db
    from app.db.compare import StatementRecord
    from app.schemas.common import SourceDocument

    def record(day: date) -> StatementRecord:
        doc_id = f"statement-{day}"
        source = SourceDocument(
            doc_id=doc_id,
            doc_type="statement",
            meeting_date=day,
            release_date=day,
            page_updated=None,
            title=f"FOMC Statement, {day}",
            url=f"https://www.federalreserve.gov/{doc_id}.htm",
        )
        return StatementRecord(source, StatementText(doc_id, day, day, PARAGRAPHS[day]))

    compare_db.statements_on = lambda conn, days: {d: record(d) for d in days if d in PARAGRAPHS}
    compare_db.stances_of = lambda conn, doc_ids, classifier: {}
    compare_db.read_comparison = lambda conn, key: None

    class FakeDatabase:
        @contextmanager
        def connection(self) -> Iterator[object]:
            yield object()

    return FakeDatabase()


def main(data_dir: Path, index_dir: Path, chroma_dir: Path) -> None:
    from fastapi.testclient import TestClient

    from app.api.database import get_db
    from app.core.config import Settings, get_settings
    from app.main import create_app

    settings = Settings(
        _env_file=None,
        environment="test",
        public_demo=True,
        warmup=True,  # the default: the demo must skip the warm-up by itself
        data_dir=data_dir,
        index_dir=index_dir,
        chroma_dir=chroma_dir,
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    database = fake_database()
    app.dependency_overrides[get_db] = lambda: database
    ask = {"question": "What did the FOMC decide in September 2024?"}
    pair = {"base_date": str(BASE), "target_date": str(TARGET)}
    with TestClient(app) as client:
        calls = {
            "health": client.get("/api/health"),
            "ask": client.post("/api/ask", json=ask),
            "ask_stream": client.post("/api/ask/stream", json=ask),
            "ask_invalid": client.post("/api/ask", json={"question": "Hi"}),
            "ask_examples": client.get("/api/ask/examples"),
            "evaluation": client.get("/api/evaluation"),
            "evaluation_question": client.get("/api/evaluation/questions/sf-07"),
            "compare_cached_only": client.post("/api/compare", json=pair),
            "compare_generate": client.post("/api/compare", json=pair | {"analysis": "generate"}),
        }
        summary = {
            name: {"status": response.status_code, "content_type": response.headers["content-type"]}
            for name, response in calls.items()
        }
        summary["health"]["demo_mode"] = calls["health"].json()["demo_mode"]
        summary["health"]["llm_configured"] = calls["health"].json()["llm_configured"]
        summary["ask"]["detail"] = calls["ask"].json()["detail"]
        summary["ask_examples"]["count"] = len(calls["ask_examples"].json())
        question = calls["evaluation_question"].json()
        summary["evaluation_question"]["passages_available"] = question["passages_available"]
        for name in ("compare_cached_only", "compare_generate"):
            summary[name]["analysis_status"] = calls[name].json()["analysis_status"]
        warm_up = app.state.warm_up
    print(
        json.dumps(
            {
                "calls": summary,
                "imported": [name for name in HEAVY if name in sys.modules],
                "warm_up": warm_up is not None,
                "threads": sorted(t.name for t in threading.enumerate()),
            }
        )
    )


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:4]))
