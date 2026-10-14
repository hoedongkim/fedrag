"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app import __version__
from app.api.database import DATABASE_ERROR_HANDLERS
from app.api.event_stream import EventStreamHeaders
from app.api.rag import start_warm_up
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.db.connection import Database
from app.rag.config import production_config

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "FedRAG analyses Federal Reserve (FOMC) communications: it answers questions with "
    "retrieval-augmented generation grounded in cited statements and minutes, classifies each "
    "meeting's policy stance as hawkish, neutral or dovish, compares two statements sentence by "
    "sentence, and benchmarks the retrieval strategies behind it. Endpoints backed by Postgres "
    "return 503 while DATABASE_URL is not set or the database cannot be reached, Ask the Fed "
    "returns 503 until OPENAI_API_KEY is set and the search index is built (make index), and the "
    "evaluation returns 503 until make rag-eval has written data/eval/rag_benchmark.json. On the "
    "public demo (PUBLIC_DEMO=true; GET /api/health reports demo_mode) nothing calls OpenAI: Ask "
    "the Fed returns 403 and GET /api/ask/examples serves recorded answers instead, and "
    'comparisons serve stored analyses only (analysis_status "disabled_on_demo" otherwise).'
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. `settings` sets middleware, the DB pool and warm-up (skipped on the public
    demo); routes use get_settings."""
    settings = settings or get_settings()
    production_config(settings)  # an unknown RETRIEVER_PRESET fails here, at startup
    database = Database(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.public_demo:  # never answers, so it loads no index, vector store or model
            logger.info(
                "Public demo: Ask the Fed and new comparison analyses are off; no OpenAI calls."
            )
        elif settings.warmup:  # in the background: the server takes requests meanwhile
            app.state.warm_up = start_warm_up(settings)
        yield
        database.close()

    app = FastAPI(
        title="FedRAG API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        exception_handlers=DATABASE_ERROR_HANDLERS,
    )
    app.state.database = database
    app.state.warm_up = None  # the warm-up thread, once lifespan starts one
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(EventStreamHeaders)  # keeps POST /api/ask/stream uncompressed by proxies
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app
