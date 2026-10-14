"""The public demo (PUBLIC_DEMO=true), enforced where requests come in.

The demo serves everything that reads stored results: meetings, stance, comparisons with their
stored analyses, the evaluation and the recorded example answers. It serves nothing that would
call OpenAI, so it needs no OPENAI_API_KEY and costs nothing to run:

- POST /api/ask and /api/ask/stream answer 403 (ASK_DISABLED), through the ask_request
  dependency. It declares the request body, so validation runs first (an invalid body is still a
  422), and it runs before the chat-model and retriever dependencies, which on the demo are never
  built.
- POST /api/compare returns the diff and any stored analysis; for a pair without one the status
  is "disabled_on_demo", whatever the analysis mode (app.compare.service.analysis_for).
- Startup skips the retrieval warm-up (app.main), so the demo never loads the BM25 index, the
  vector store or the cross-encoder, and never imports torch or the openai package.
"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.schemas.ask import AskRequest
from app.schemas.common import ErrorResponse

ASK_DISABLED = (
    "Ask the Fed is disabled on the public demo to avoid API costs. Run FedRAG locally to ask "
    "your own questions."
)

ASK_DISABLED_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_403_FORBIDDEN: {
        "model": ErrorResponse,
        "description": (
            "Disabled on the public demo (PUBLIC_DEMO=true), which makes no OpenAI calls; "
            "GET /api/ask/examples serves recorded answers instead."
        ),
    }
}


def ask_request(
    request: AskRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> AskRequest:
    """The validated question, unless this is the public demo: 403 there.

    Declare it first among an ask route's dependencies: FastAPI resolves them in order, so on the
    demo the 403 is raised before the chat model or a retriever is built.
    """
    if settings.public_demo:
        raise HTTPException(status.HTTP_403_FORBIDDEN, ASK_DISABLED)
    return request


AskBody = Annotated[AskRequest, Depends(ask_request)]
