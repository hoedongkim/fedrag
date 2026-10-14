"""Polite HTTP client for federalreserve.gov that caches raw pages under data/raw."""

import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Self
from urllib.parse import urljoin, urlsplit

import httpx2

BASE_URL = "https://www.federalreserve.gov"
USER_AGENT = "FedRAG/0.1 (research project; +https://github.com/hoedongkim/fedrag)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 60.0


def absolute_url(href: str) -> str:
    """Resolve a link from a Fed page (usually site-relative) to an absolute https URL."""
    return urljoin(BASE_URL + "/", href.strip())


class CacheMissError(LookupError):
    """Raised by an offline client when a page has not been downloaded yet."""


class FedClient:
    """GET pages at most once per `min_interval` seconds, retry 429/5xx, cache the raw bytes.

    Pages are cached as `raw_dir/<category>/<original file name>` and served from there on later
    runs. `refresh` re-downloads every page once per client (`get(..., refresh=True)` does that
    for one page); `offline` never touches the network. `sleep` and `clock` are injectable for
    tests.
    """

    def __init__(
        self,
        raw_dir: Path,
        *,
        refresh: bool = False,
        offline: bool = False,
        min_interval: float = 1.0,
        max_retries: int = 4,
        backoff: float = 2.0,
        timeout: float = 30.0,
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.raw_dir = raw_dir
        self.refresh = refresh
        self.offline = offline
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._refreshed: set[Path] = set()
        self._http = httpx2.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def cache_path(self, url: str, category: str) -> Path:
        """Where the page at `url` is cached: raw_dir/category/<file name from the URL>."""
        name = Path(urlsplit(absolute_url(url)).path).name
        if not name:
            raise ValueError(f"URL has no file name to cache under: {url}")
        return self.raw_dir / category / name

    def is_cached(self, url: str, category: str) -> bool:
        return self.cache_path(url, category).is_file()

    def get(self, url: str, category: str, *, refresh: bool = False) -> bytes:
        """Return the page body, from the cache when possible.

        `refresh` re-downloads this one page, once per client. An offline client always serves
        the cached copy.
        """
        path = self.cache_path(url, category)
        stale = (self.refresh or refresh) and path not in self._refreshed
        if path.is_file() and (self.offline or not stale):
            return path.read_bytes()
        if self.offline:
            raise CacheMissError(f"{url} is not cached at {path}")
        content = self._download(absolute_url(url))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(content)
        partial.replace(path)
        self._refreshed.add(path)
        return content

    def _download(self, url: str) -> bytes:
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                response = self._http.get(url)
            except httpx2.TransportError as exc:
                error: Exception = exc
                delay = self.backoff * 2**attempt
            else:
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response.content
                error = httpx2.HTTPStatusError(
                    f"{response.status_code} from {url}",
                    request=response.request,
                    response=response,
                )
                delay = _retry_after(response) or self.backoff * 2**attempt
            if attempt == self.max_retries:
                raise error
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait_turn(self) -> None:
        """Sleep until `min_interval` has passed since the previous request started."""
        if self._last_request is not None:
            wait = self._last_request + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()


def _retry_after(response: httpx2.Response) -> float | None:
    """Seconds from a numeric Retry-After header, capped; None when absent or a date."""
    value = response.headers.get("Retry-After", "").strip()
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)
