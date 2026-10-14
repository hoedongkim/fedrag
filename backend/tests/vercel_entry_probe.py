"""Run by tests/test_vercel.py and scripts/check_vercel.sh in a fresh interpreter: load the Vercel
entrypoint the way Vercel's Python runtime does (api/index.py by file path, as module api.index),
drive its ASGI app through startup, every call the frontend makes on the demo, and shutdown,
without an HTTP client, and print one JSON line: the app's class, where its code came from, each
call's status and size, the heavy modules and the third-party distributions the process imported.

    python tests/vercel_entry_probe.py BUNDLE_DIR [--only requirements.txt]

BUNDLE_DIR is laid out like the repository: api/index.py, backend/app, data/eval and data/index.
Callers copy those in, so backend/.env is never read. --only hides every installed distribution
the file does not list, so the full development environment behaves like Vercel's install of it.
With DATABASE_URL set (the demo's read-only role) the database routes must answer 200, without it
503. Any other status, a heavy import, a response over Vercel's 4.5 MB limit or an imported module
that no installed distribution owns exits 1, after printing the JSON.
"""

import argparse
import asyncio
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import resource
import sys
import sysconfig
import time
from collections.abc import Awaitable, Callable
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
MAX_BODY_BYTES = 4_500_000  # Vercel Functions' response body limit (4.5 MB)
STORED_PAIR = {"base_date": "2024-07-31", "target_date": "2024-09-18"}
UNSTORED_PAIR = {"base_date": "2019-01-30", "target_date": "2024-09-18", "analysis": "generate"}
ASK = {"question": "What did the FOMC decide in September 2024?"}

# name: method, path, JSON body, status without a database, status with one. These are the calls
# the frontend makes on the demo (the four classifiers share one route), plus the API docs.
CALLS: dict[str, tuple[str, str, dict[str, Any] | None, int, int]] = {
    "health": ("GET", "/api/health", None, 200, 200),
    "meetings": ("GET", "/api/meetings", None, 503, 200),
    "meetings_statements": ("GET", "/api/meetings?doc_type=statement", None, 503, 200),
    "stance": ("GET", "/api/stance", None, 503, 200),
    "stance_zero_shot": ("GET", "/api/stance?classifier=llm_zero_shot", None, 503, 200),
    "stance_meeting": ("GET", "/api/stance/2024-09-18", None, 503, 200),
    "stance_benchmark": ("GET", "/api/stance/benchmark", None, 200, 200),
    "evaluation": ("GET", "/api/evaluation", None, 200, 200),
    "evaluation_question": ("GET", "/api/evaluation/questions/sf-07", None, 200, 200),
    "ask_examples": ("GET", "/api/ask/examples", None, 200, 200),
    "ask": ("POST", "/api/ask", ASK, 403, 403),
    "ask_stream": ("POST", "/api/ask/stream", ASK, 403, 403),
    "ask_invalid": ("POST", "/api/ask", {"question": "Hi"}, 422, 422),
    "compare_stored": ("POST", "/api/compare", STORED_PAIR, 503, 200),
    "compare_unstored": ("POST", "/api/compare", UNSTORED_PAIR, 503, 200),
    "openapi": ("GET", "/openapi.json", None, 200, 200),
    "docs": ("GET", "/docs", None, 200, 200),
}

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
Asgi = Callable[[dict[str, Any], Receive, Send], Awaitable[None]]


async def request(
    app: Asgi, method: str, target: str, body: dict[str, Any] | None
) -> tuple[int, dict[str, str], bytes]:
    """One HTTP request through the ASGI interface: status, headers and body."""
    path, _, query = target.partition("?")
    payload = json.dumps(body).encode() if body is not None else b""
    headers = [(b"host", b"fedrag-api.vercel.app"), (b"accept-encoding", b"identity")]
    if body is not None:
        headers.append((b"content-type", b"application/json"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("fedrag-api.vercel.app", 443),
    }
    done = asyncio.Event()
    received = False
    status, response_headers, chunks = 0, [], []

    async def receive() -> dict[str, Any]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await done.wait()  # the client stays connected until the response is complete
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status, response_headers
        if message["type"] == "http.response.start":
            status, response_headers = message["status"], message.get("headers", [])
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                done.set()

    await app(scope, receive, send)
    return status, {k.decode(): v.decode() for k, v in response_headers}, b"".join(chunks)


async def drive(app: Asgi, with_database: bool) -> tuple[dict[str, Any], list[str]]:
    """Startup, every call in CALLS, shutdown; returns the calls' summary and what went wrong."""
    to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    lifespan = asyncio.create_task(
        app({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}, to_app.get, from_app.put)
    )
    await to_app.put({"type": "lifespan.startup"})
    problems = []
    if (started := await from_app.get())["type"] != "lifespan.startup.complete":
        problems.append(f"startup failed: {started}")

    calls: dict[str, Any] = {}
    for name, (method, path, body, without_db, with_db) in CALLS.items():
        began = time.perf_counter()
        status, headers, content = await request(app, method, path, body)
        row: dict[str, Any] = {
            "status": status,
            "bytes": len(content),
            "seconds": round(time.perf_counter() - began, 3),
            "content_type": headers.get("content-type", ""),
        }
        expected = with_db if with_database else without_db
        if status != expected:
            problems.append(f"{method} {path}: {status}, expected {expected}")
        if len(content) > MAX_BODY_BYTES:
            problems.append(f"{method} {path}: {len(content):,} bytes, over Vercel's 4.5 MB")
        if row["content_type"].startswith("application/json"):
            data = json.loads(content)
            if isinstance(data, list):
                row["items"] = len(data)
            elif isinstance(data, dict):
                for key in ("demo_mode", "llm_configured", "database_configured", "environment"):
                    if key in data:
                        row[key] = data[key]
                for key in ("passages_available", "analysis_status", "detail"):
                    if isinstance(data.get(key), str | bool):
                        row[key] = data[key]
        calls[name] = row

    if calls["health"].get("demo_mode") is not True:
        problems.append("GET /api/health: demo_mode is not true")
    if with_database:  # what the demo serves from Neon: stored analyses only
        for name, state in (("compare_stored", "ready"), ("compare_unstored", "disabled_on_demo")):
            if calls[name].get("analysis_status") != state:
                problems.append(f"{name}: analysis_status is not {state!r}")
    await to_app.put({"type": "lifespan.shutdown"})
    if (stopped := await from_app.get())["type"] != "lifespan.shutdown.complete":
        problems.append(f"shutdown failed: {stopped}")
    await lifespan
    return calls, problems


class Distributions:
    """Which installed distribution a module file belongs to, from the distributions' RECORDs."""

    def __init__(self) -> None:
        from importlib.metadata import distribution, packages_distributions

        self._distribution = distribution
        self._by_top_level = packages_distributions()
        self._files: dict[str, set[Path]] = {}
        paths = sysconfig.get_paths()
        self._stdlib = [Path(paths[k]).resolve() for k in ("stdlib", "platstdlib")]
        self._site = [Path(paths[k]).resolve() for k in ("purelib", "platlib")]

    def owners(self, module_name: str, path: Path) -> list[str]:
        """The distributions whose RECORD lists `path`: one, or none for code of no distribution."""
        top = module_name.partition(".")[0]
        return [d for d in sorted(set(self._by_top_level.get(top, ()))) if path in self._of(d)]

    def is_stdlib(self, module_name: str, path: Path) -> bool:
        if module_name.partition(".")[0] in sys.stdlib_module_names:
            return True
        in_site = any(path.is_relative_to(site) for site in self._site)
        return not in_site and any(path.is_relative_to(lib) for lib in self._stdlib)

    def _of(self, name: str) -> set[Path]:
        if name not in self._files:
            dist = self._distribution(name)
            self._files[name] = {Path(str(dist.locate_file(f))).resolve() for f in dist.files or ()}
        return self._files[name]


class ListedOnly:
    """A meta path finder that hides every installed distribution `requirements.txt` does not list,
    so a full development environment imports what Vercel's install of that file would: a missing
    hard dependency fails the import, an optional one (imported only when installed) is skipped."""

    def __init__(self, listed: set[str], distributions: Distributions) -> None:
        self.listed, self.distributions, self.hidden = listed, distributions, set()

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name.partition(".")[0] in sys.stdlib_module_names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not spec.has_location or not spec.origin:
            return None
        owners = self.distributions.owners(name, Path(spec.origin).resolve())
        if owners and not {canonical(o) for o in owners} & self.listed:
            self.hidden.update(owners)
            raise ModuleNotFoundError(f"{name!r} is not in requirements.txt", name=name)
        return None  # listed, or no distribution's: the usual finders load it


def canonical(name: str) -> str:
    """A distribution's normalized name (PEP 503): PyYAML -> pyyaml, typing_extensions ->
    typing-extensions."""
    return re.sub(r"[-_.]+", "-", name).lower()


def listed_in(requirements: Path) -> set[str]:
    names = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if line:
            names.add(canonical(re.split(r"[\[=<>!~;\s]", line, maxsplit=1)[0]))
    return names


def imported(
    baseline: set[str], skip: list[Path], distributions: Distributions
) -> tuple[list[str], list[str]]:
    """The distributions whose modules were imported after `baseline`, and the third-party modules
    no distribution owns. The standard library and the files under `skip` are left out."""
    found, unowned = set(), set()
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name in baseline or not path or getattr(module, "__name__", name) != name:
            continue  # loaded before, built in, a namespace package, or an alias of another
        path = Path(path).resolve()
        if distributions.is_stdlib(name, path) or any(path.is_relative_to(p) for p in skip):
            continue
        owners = distributions.owners(name, path)
        if owners:
            found.update(owners)
        else:
            unowned.add(name)
    return sorted(found), sorted(unowned)


def main(bundle: Path, requirements: Path | None) -> int:
    bundle = bundle.resolve()
    baseline = set(sys.modules)
    with_database = bool(os.environ.get("DATABASE_URL", "").strip())
    # psycopg_pool logs connection errors with the server's message; the app reports the class.
    logging.getLogger("psycopg").setLevel(logging.ERROR)
    distributions = Distributions()
    finder = None
    if requirements is not None:
        finder = ListedOnly(listed_in(requirements), distributions)
        sys.meta_path.insert(0, finder)

    began = time.perf_counter()
    spec = importlib.util.spec_from_file_location("api.index", bundle / "api" / "index.py")
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    sys.modules["api.index"] = entry
    spec.loader.exec_module(entry)
    import_seconds = time.perf_counter() - began

    from fastapi import FastAPI

    app = entry.app
    calls, problems = asyncio.run(drive(app, with_database))
    probe = Path(__file__).resolve()
    found, unowned = imported(baseline, [bundle, probe], distributions)
    heavy = [name for name in HEAVY if name in sys.modules]
    app_package = Path(sys.modules["app"].__file__).resolve()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10

    if not isinstance(app, FastAPI):
        problems.append(f"api/index.py's app is a {type(app).__module__}.{type(app).__name__}")
    if not app_package.is_relative_to(bundle / "backend" / "app"):
        problems.append(f"the app package was imported from {app_package}, not the bundle")
    if heavy:
        problems.append(f"heavy modules imported: {', '.join(heavy)}")
    if unowned:
        problems.append(f"modules no installed distribution owns: {', '.join(unowned)}")
    print(
        json.dumps(
            {
                "app_class": f"{type(app).__module__}.{type(app).__name__}",
                "app_package": os.path.relpath(app_package, bundle),
                "with_database": with_database,
                "calls": calls,
                "heavy": heavy,
                "distributions": found,
                "hidden": sorted(finder.hidden) if finder else None,
                "unowned_modules": unowned,
                "environment": {k: os.environ.get(k) for k in ("PUBLIC_DEMO", "ENVIRONMENT")},
                "import_seconds": round(import_seconds, 2),
                "peak_rss_mb": round(peak_mb),
                "problems": problems,
            }
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.partition("\n\n")[0])
    parser.add_argument("bundle", type=Path, help="a folder laid out like the repository")
    parser.add_argument(
        "--only",
        type=Path,
        metavar="REQUIREMENTS",
        help="hide every installed distribution this requirements file does not list",
    )
    args = parser.parse_args()
    sys.exit(main(args.bundle, args.only))
