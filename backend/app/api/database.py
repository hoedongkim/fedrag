"""Postgres for routes: the get_db dependency, its documented 503 and the handlers that send it."""

import logging
from typing import Annotated, Any

import psycopg
from fastapi import Depends, Request, status
from fastapi.responses import JSONResponse

from app.db.connection import Database, DatabaseNotConfiguredError, DatabaseUrlError
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)

UNAVAILABLE = "Database unavailable: could not reach Postgres. Try again shortly."
NOT_LOADED = "Database has no fedrag tables yet: run make db-load."

DATABASE_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "Database not configured (DATABASE_URL is not set), not reachable, or not loaded "
            "yet (run make db-load)."
        ),
    }
}


def get_db(request: Request) -> Database:
    """The app's Postgres pool; override this dependency in tests.

    Returning the pool rather than a connection keeps request validation first: a bad parameter
    is a 422 even when no database is configured. Routes borrow a connection with
    `db.connection()`, and the handlers below turn a missing or unreachable database into a 503.
    """
    return request.app.state.database


Db = Annotated[Database, Depends(get_db)]


async def _not_configured(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status.HTTP_503_SERVICE_UNAVAILABLE)


async def _unavailable(request: Request, exc: Exception) -> JSONResponse:
    # Only the class name: psycopg_pool already logs connection errors, and a message could echo
    # parts of the connection string.
    logger.warning("Postgres unavailable (%s)", type(exc).__name__)
    return JSONResponse({"detail": UNAVAILABLE}, status.HTTP_503_SERVICE_UNAVAILABLE)


async def _not_loaded(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": NOT_LOADED}, status.HTTP_503_SERVICE_UNAVAILABLE)


DATABASE_ERROR_HANDLERS = {
    DatabaseNotConfiguredError: _not_configured,
    DatabaseUrlError: _not_configured,
    psycopg.OperationalError: _unavailable,
    psycopg.errors.UndefinedTable: _not_loaded,
}
