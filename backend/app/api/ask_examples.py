"""Recorded example answers for routes: data/eval/ask_examples.json (tracked, read-only, re-read
whenever it changes on disk), written by python -m app.evaluation export-ask-examples."""

from pathlib import Path
from typing import Any

from fastapi import status
from pydantic import ValidationError

from app.api.file_cache import VersionedFileCache
from app.core.config import Settings
from app.schemas.ask import AskExample, AskExamplesFile
from app.schemas.common import ErrorResponse

ASK_EXAMPLES_FILE = Path("eval") / "ask_examples.json"  # under settings.data_dir
EXAMPLES_MISSING = "Example answers not exported yet: run make ask-examples."
EXAMPLES_INVALID = "Example answers file data/eval/ask_examples.json is invalid: {problems}"

EXAMPLES_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "The example answers have not been exported (make ask-examples), or their file does "
            "not fit the contract."
        ),
    }
}


class ExamplesUnavailableError(RuntimeError):
    """The examples file is missing or does not match the AskExamplesFile contract."""


_files = VersionedFileCache()


def _parse(path: Path) -> AskExamplesFile | str:
    """The validated file, or the error message to report (cached too, so a bad file is parsed
    once per version)."""
    try:
        return AskExamplesFile.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'file'}: {err['msg']}"
            for err in exc.errors(include_url=False)[:5]
        )
        return EXAMPLES_INVALID.format(problems=problems)
    except OSError as exc:
        return EXAMPLES_INVALID.format(problems=exc)


def load_ask_examples(settings: Settings) -> list[AskExample]:
    """The examples as they are on disk now; raises ExamplesUnavailableError when the file is
    missing or invalid."""
    parsed = _files.get(settings.data_dir / ASK_EXAMPLES_FILE, _parse)
    if parsed is None:
        raise ExamplesUnavailableError(EXAMPLES_MISSING)
    if isinstance(parsed, str):
        raise ExamplesUnavailableError(parsed)
    return parsed.examples
