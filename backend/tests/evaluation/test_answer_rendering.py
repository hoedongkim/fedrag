"""The answer prompt's rendering outside its templates (app.rag.prompts.time_rule,
format_passages and prompt_inputs), pinned: cached answers are keyed by answer_prompt_label,
which hashes only the templates and generation settings, so a change here must bump
runner.ANSWER_PROMPT_REVISION or stale answers are reused."""

import hashlib
import json
from datetime import date

from app.evaluation import runner
from app.rag.prompts import format_passages, prompt_inputs, time_rule
from app.rag.types import Chunk, RetrievedChunk

BUMP = "answer rendering changed: bump ANSWER_PROMPT_REVISION in app/evaluation/runner.py"


def chunk(chunk_id: str, section: str | None, text: str) -> RetrievedChunk:
    doc_id, index = chunk_id.split("#")
    meeting = date.fromisoformat(doc_id.split("-", 1)[1])
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_type="statement" if section is None else "minutes",
            meeting_date=meeting,
            release_date=meeting,
            page_updated=None,
            title=f"FOMC document, {meeting.isoformat()}",
            url=f"https://www.federalreserve.gov/{doc_id}.htm",
            section=section,
            index=int(index),
            text=text,
            context="context",
        ),
        score=1.0,
    )


PASSAGES = [
    chunk("statement-2024-09-18#0000", None, "The Committee decided to lower the target range."),
    chunk("minutes-2024-09-18#0012", "Participants' Views", "Participants noted that inflation."),
]


# sha256 prefixes of the rendering above; recorded with ANSWER_PROMPT_REVISION 0.
PINNED = {
    "time_rule(None)": "4efad3f66a21df50",
    "time_rule(2020-01-01)": "8b20ae3c5088027f",
    "format_passages": "0da347bb61975877",
    "prompt_inputs(no as_of)": "5fa835462b547415",
    "prompt_inputs(as_of)": "7c96db65bd138bff",
}


def digest(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_the_rendering_outside_the_templates_is_pinned() -> None:
    rendered = {
        "time_rule(None)": digest(time_rule(None)),
        "time_rule(2020-01-01)": digest(time_rule(date(2020, 1, 1))),
        "format_passages": digest(format_passages(PASSAGES)),
        "prompt_inputs(no as_of)": digest(prompt_inputs("Q?", None, PASSAGES)),
        "prompt_inputs(as_of)": digest(prompt_inputs("Q?", date(2024, 10, 1), PASSAGES)),
    }

    assert rendered == PINNED, BUMP


def test_revision_zero_keeps_the_phase7_label() -> None:
    assert runner.ANSWER_PROMPT_REVISION == 0
    assert runner.answer_prompt_label() == "answer.7be4030b"
