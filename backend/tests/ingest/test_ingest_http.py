"""Tests for the polite cached HTTP client, with a fake transport and a fake clock."""

from pathlib import Path

import httpx2
import pytest

from app.ingest.http import USER_AGENT, CacheMissError, FedClient, absolute_url

URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250917.htm"


class FakeTime:
    """A clock that only moves when the client sleeps, so rate limiting is observable."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(
    tmp_path: Path, responses: list[httpx2.Response | Exception], **kwargs: object
) -> tuple[FedClient, list[httpx2.Request], FakeTime]:
    requests: list[httpx2.Request] = []
    time = FakeTime()

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    client = FedClient(
        tmp_path,
        transport=httpx2.MockTransport(handler),
        sleep=time.sleep,
        clock=time.clock,
        **kwargs,  # type: ignore[arg-type]
    )
    return client, requests, time


def test_downloads_once_then_serves_from_cache(tmp_path: Path) -> None:
    client, requests, _ = make_client(tmp_path, [httpx2.Response(200, content=b"<html>v1</html>")])

    assert client.get(URL, "minutes") == b"<html>v1</html>"
    assert client.get(URL, "minutes") == b"<html>v1</html>"

    assert len(requests) == 1
    assert (tmp_path / "minutes" / "fomcminutes20250917.htm").read_bytes() == b"<html>v1</html>"
    assert not list(tmp_path.rglob("*.part"))


def test_sends_descriptive_user_agent_without_email(tmp_path: Path) -> None:
    client, requests, _ = make_client(tmp_path, [httpx2.Response(200, content=b"ok")])

    client.get(URL, "minutes")

    user_agent = requests[0].headers["User-Agent"]
    assert user_agent == USER_AGENT
    assert "FedRAG" in user_agent and "github.com" in user_agent
    assert "@" not in user_agent


def test_waits_min_interval_between_requests(tmp_path: Path) -> None:
    pages = [httpx2.Response(200, content=b"a"), httpx2.Response(200, content=b"b")]
    client, _, time = make_client(tmp_path, pages, min_interval=1.0)

    client.get(URL, "minutes")
    client.get(URL.replace("20250917", "20251029"), "minutes")

    assert time.sleeps == [pytest.approx(1.0)]


def test_cache_hits_are_not_rate_limited(tmp_path: Path) -> None:
    client, _, time = make_client(tmp_path, [httpx2.Response(200, content=b"a")])

    for _ in range(3):
        client.get(URL, "minutes")

    assert time.sleeps == []


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retries_transient_errors_with_backoff(tmp_path: Path, status: int) -> None:
    responses: list[httpx2.Response | Exception] = [
        httpx2.Response(status),
        httpx2.Response(status),
        httpx2.Response(200, content=b"finally"),
    ]
    client, requests, time = make_client(tmp_path, responses, backoff=2.0, min_interval=0.0)

    assert client.get(URL, "minutes") == b"finally"
    assert len(requests) == 3
    assert time.sleeps == [2.0, 4.0]


def test_honours_numeric_retry_after(tmp_path: Path) -> None:
    responses: list[httpx2.Response | Exception] = [
        httpx2.Response(429, headers={"Retry-After": "7"}),
        httpx2.Response(200, content=b"ok"),
    ]
    client, _, time = make_client(tmp_path, responses, min_interval=0.0)

    client.get(URL, "minutes")

    assert time.sleeps == [7.0]


def test_retries_transport_errors(tmp_path: Path) -> None:
    responses: list[httpx2.Response | Exception] = [
        httpx2.ConnectError("connection reset"),
        httpx2.Response(200, content=b"ok"),
    ]
    client, requests, _ = make_client(tmp_path, responses, min_interval=0.0)

    assert client.get(URL, "minutes") == b"ok"
    assert len(requests) == 2


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    responses: list[httpx2.Response | Exception] = [httpx2.Response(503) for _ in range(3)]
    client, requests, _ = make_client(tmp_path, responses, max_retries=2, min_interval=0.0)

    with pytest.raises(httpx2.HTTPStatusError):
        client.get(URL, "minutes")
    assert len(requests) == 3
    assert not client.is_cached(URL, "minutes")


def test_does_not_retry_not_found(tmp_path: Path) -> None:
    client, requests, time = make_client(tmp_path, [httpx2.Response(404)])

    with pytest.raises(httpx2.HTTPStatusError):
        client.get(URL, "minutes")
    assert len(requests) == 1
    assert time.sleeps == []


def test_offline_client_never_downloads(tmp_path: Path) -> None:
    client, requests, _ = make_client(tmp_path, [], offline=True)

    with pytest.raises(CacheMissError):
        client.get(URL, "minutes")
    assert requests == []


def test_refresh_downloads_each_page_once_per_client(tmp_path: Path) -> None:
    cached = tmp_path / "minutes" / "fomcminutes20250917.htm"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"stale")
    client, requests, _ = make_client(
        tmp_path, [httpx2.Response(200, content=b"fresh")], refresh=True
    )

    assert client.get(URL, "minutes") == b"fresh"
    assert client.get(URL, "minutes") == b"fresh"
    assert len(requests) == 1
    assert cached.read_bytes() == b"fresh"


def test_refresh_one_page_downloads_it_once_per_client(tmp_path: Path) -> None:
    cached = tmp_path / "minutes" / "fomcminutes20250917.htm"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"stale")
    client, requests, _ = make_client(tmp_path, [httpx2.Response(200, content=b"fresh")])

    assert client.get(URL, "minutes", refresh=True) == b"fresh"
    assert client.get(URL, "minutes", refresh=True) == b"fresh"
    assert len(requests) == 1
    assert cached.read_bytes() == b"fresh"


def test_offline_client_serves_the_cache_even_when_asked_to_refresh(tmp_path: Path) -> None:
    cached = tmp_path / "minutes" / "fomcminutes20250917.htm"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"stale")
    client, requests, _ = make_client(tmp_path, [], offline=True)

    assert client.get(URL, "minutes", refresh=True) == b"stale"
    assert requests == []


def test_cache_path_and_absolute_url(tmp_path: Path) -> None:
    client = FedClient(tmp_path)
    relative = "/newsevents/pressreleases/monetary20250917a.htm"

    assert absolute_url(relative) == f"https://www.federalreserve.gov{relative}"
    assert client.cache_path(relative, "statements") == (
        tmp_path / "statements" / "monetary20250917a.htm"
    )
    with pytest.raises(ValueError, match="no file name"):
        client.cache_path("https://www.federalreserve.gov/", "calendar")
    client.close()
