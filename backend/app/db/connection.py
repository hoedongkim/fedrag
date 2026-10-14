"""Postgres connections: one-off connections for scripts and a lazily opened pool for the API."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool
from pydantic import SecretStr

from app.core.config import Settings, get_settings

SCHEMA_FILE = Path(__file__).with_name("schema.sql")
NOT_CONFIGURED = "Database not configured: set DATABASE_URL in backend/.env."
MALFORMED_URL = (
    "DATABASE_URL is malformed: use postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require "
    "and percent-encode special characters (such as @ : / % # or a space) in the password."
)

# Neon suspends idle compute and wakes it on the next connection, which can take a few seconds, so
# connect_timeout leaves room for that. prepare_threshold=None turns off server-side prepared
# statements, which Neon's pooled endpoint (PgBouncer in transaction mode) cannot be relied on to
# keep between transactions.
CONNECT_KWARGS: dict[str, Any] = {"connect_timeout": 10, "prepare_threshold": None}


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when something needs Postgres but DATABASE_URL is not set."""

    def __init__(self) -> None:
        super().__init__(NOT_CONFIGURED)


class DatabaseUrlError(ValueError):
    """DATABASE_URL does not parse. The message never repeats it: libpq quotes the bad part."""

    def __init__(self) -> None:
        super().__init__(MALFORMED_URL)


def _conninfo(url: SecretStr) -> str:
    """The connection string, checked before libpq or the pool can echo a malformed one."""
    raw = url.get_secret_value()
    try:
        params = conninfo_to_dict(raw)
    except psycopg.ProgrammingError:
        params = None
    # An unencoded '@' in the password leaves the rest of it in the host name.
    # Raised outside the except block so no chained exception carries libpq's message.
    if params is None or "@" in str(params.get("host") or ""):
        raise DatabaseUrlError
    return raw


def connect(settings: Settings | None = None) -> psycopg.Connection[DictRow]:
    """Open a connection to DATABASE_URL that returns dict rows; use it as a context manager."""
    url = (settings or get_settings()).database_url
    if url is None:
        raise DatabaseNotConfiguredError
    return psycopg.connect(_conninfo(url), row_factory=dict_row, **CONNECT_KWARGS)


def apply_schema(conn: psycopg.Connection[Any]) -> None:
    """Create the fedrag schema, tables and indexes that are missing (schema.sql is idempotent)."""
    conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))


class Database:
    """The API's connection pool, opened on first use so startup never waits on Postgres."""

    def __init__(self, url: SecretStr | None, *, timeout: float = 15.0) -> None:
        self._url = url
        self._timeout = timeout
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection[Any]]:
        """Borrow an autocommit connection; raises DatabaseNotConfiguredError without a URL and
        DatabaseUrlError for a malformed one.

        A Postgres that cannot be reached within `timeout` seconds raises psycopg_pool.PoolTimeout,
        a psycopg.OperationalError.
        """
        with self._get_pool().connection() as conn:
            yield conn

    def close(self) -> None:
        """Close the pool if it was ever opened; safe to call more than once."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    def _get_pool(self) -> ConnectionPool:
        if self._url is None:
            raise DatabaseNotConfiguredError
        conninfo = _conninfo(self._url)
        with self._lock:
            if self._pool is None:
                # min_size=0 with max_idle under Neon's 5-minute autosuspend lets idle connections
                # go so the compute can scale to zero; check_connection on every checkout replaces
                # a connection that Neon dropped meanwhile instead of failing the request.
                self._pool = ConnectionPool(
                    conninfo,
                    kwargs={**CONNECT_KWARGS, "autocommit": True},
                    min_size=0,
                    max_size=4,
                    open=True,
                    check=ConnectionPool.check_connection,
                    timeout=self._timeout,
                    max_idle=240,
                    name="fedrag-api",
                )
            return self._pool
