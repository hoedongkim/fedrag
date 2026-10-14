"""The public demo API as a Vercel function: api/index.py, requirements.txt, .python-version,
vercel.json and .vercelignore at the repository root (docs/deploy.md).

The entrypoint runs in a fresh interpreter (tests/vercel_entry_probe.py), from a copy of the files
the function ships, laid out as in the repository: it must be the FastAPI app, always the public
demo, answer every call the frontend makes and import no heavy module. requirements.txt cannot
drift: each pin is the version constraints.txt pins, and it lists exactly what the demo imports
plus what those packages need on Vercel's Linux, checked with every other installed distribution
hidden from the entrypoint, and no other Python manifest that git would push can take its place.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from app.api.demo import ASK_DISABLED
from app.core.config import REPO_ROOT, Settings
from app.db.connection import NOT_CONFIGURED

BACKEND = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name("vercel_entry_probe.py")
ENTRY = "api/index.py"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
CONSTRAINTS = BACKEND / "constraints.txt"
PYTHON_VERSION = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
# What the function reads, which vercel.json's excludeFiles must leave in the bundle.
SHIPPED = ["api/index.py", "backend/app/main.py", "data/eval/rag_benchmark.json"]
SHIPPED += ["data/index/chunks.jsonl", "data/index/manifest.json"]
# Files Vercel's Python builder would read instead of requirements.txt and the root .python-version:
# it takes the manifest and .python-version nearest the entrypoint, walking up from api/, and in one
# folder ranks pyproject.toml (with uv.lock or pylock.toml), Pipfile.lock, Pipfile and the frozen
# requirements files above requirements.txt. `vercel build` writes the first two at the root.
SHADOWING = [
    *(
        f"api/{name}"
        for name in (
            "pyproject.toml",
            "Pipfile.lock",
            "Pipfile",
            "requirements.frozen.txt",
            "requirements-frozen.txt",
            "requirements.txt",
            "requirements.in",
            "requirements/prod.txt",
            ".python-version",
        )
    ),
    "pyproject.toml",
    "uv.lock",
    "pylock.toml",
    "Pipfile.lock",
    "Pipfile",
    "requirements.frozen.txt",
    "requirements-frozen.txt",
]
# The machine Vercel installs requirements.txt for, as environment markers see it.
VERCEL_MARKERS = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "python_version": PYTHON_VERSION,
    "python_full_version": f"{PYTHON_VERSION}.0",
}
DATABASE_CALLS = [
    "meetings",
    "meetings_statements",
    "stance",
    "stance_zero_shot",
    "stance_meeting",
    "compare_stored",
    "compare_unstored",
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The files the function reads, copied: never backend/.env, data/chroma or a model."""
    root = tmp_path_factory.mktemp("vercel-bundle")
    shutil.copytree(REPO_ROOT / "api", root / "api", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        BACKEND / "app", root / "backend" / "app", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(REPO_ROOT / "data" / "eval", root / "data" / "eval")
    (root / "data" / "index").mkdir(parents=True)
    for name in ("chunks.jsonl", "manifest.json"):
        source = REPO_ROOT / "data" / "index" / name
        assert source.is_file(), f"data/index/{name} is missing: it is tracked for the Vercel demo"
        shutil.copy2(source, root / "data" / "index" / name)
    return root


def run_probe(bundle: Path, *args: str) -> dict[str, Any]:
    settings_env = {name.upper() for name in Settings.model_fields}
    env = {k: v for k, v in os.environ.items() if k not in settings_env}  # no key, no database
    env["PUBLIC_DEMO"] = "false"  # what a mistyped project variable would say
    run = subprocess.run(
        [sys.executable, str(PROBE), str(bundle), *args],
        cwd=bundle,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probe(bundle: Path) -> dict[str, Any]:
    """The entrypoint in the full development environment."""
    return run_probe(bundle)


@pytest.fixture(scope="module")
def listed_only(bundle: Path) -> dict[str, Any]:
    """The entrypoint with every distribution requirements.txt does not list hidden."""
    return run_probe(bundle, "--only", str(REQUIREMENTS))


def pins(path: Path) -> dict[str, Requirement]:
    lines = path.read_text(encoding="utf-8").splitlines()
    requirements = [Requirement(line) for line in lines if line.strip() and line[0] != "#"]
    by_name = {canonicalize_name(r.name): r for r in requirements}
    assert len(by_name) == len(requirements), f"a package is listed twice in {path.name}"
    return by_name


def git(*args: str, cwd: Path = REPO_ROOT, stdin: str | None = None) -> list[str]:
    """git's output lines; the test is skipped without git or outside a git checkout."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    run = subprocess.run(
        ["git", *args], cwd=cwd, input=stdin, capture_output=True, text=True, check=False
    )
    if "not a git repository" in run.stderr:
        pytest.skip("not a git checkout")
    assert run.returncode in (0, 1), run.stderr  # check-ignore exits 1 when nothing matches
    return run.stdout.splitlines()


def pushed(*paths: str) -> list[str]:
    """The files under `paths` that the next push would hold: tracked, or new and not ignored."""
    return git("ls-files", "--cached", "--others", "--exclude-standard", "--", *paths)


def needs(roots: dict[str, set[str]]) -> set[str]:
    """`roots` (distribution name: extras) and everything their installed metadata requires on
    Vercel's machine, recursively."""
    seen: set[tuple[str, frozenset[str]]] = set()
    todo = [(name, frozenset(extras)) for name, extras in roots.items()]
    while todo:
        name, extras = todo.pop()
        if (canonicalize_name(name), extras) in seen:
            continue
        seen.add((canonicalize_name(name), extras))
        for spec in distribution(name).requires or []:
            requirement = Requirement(spec)
            marker = requirement.marker
            if marker is None or any(
                marker.evaluate(VERCEL_MARKERS | {"extra": extra}) for extra in {"", *extras}
            ):
                todo.append((requirement.name, frozenset(requirement.extras)))
    return {name for name, _ in seen}


# --- The entrypoint ---------------------------------------------------------------------------


@pytest.mark.parametrize("result", ["probe", "listed_only"])
def test_the_entrypoint_is_the_backends_fastapi_app(
    result: str, request: pytest.FixtureRequest
) -> None:
    run = request.getfixturevalue(result)

    assert run["app_class"] == "fastapi.applications.FastAPI"
    assert run["app_package"] == "backend/app/__init__.py"  # api/index.py's path, not the venv's
    assert run["problems"] == []


def test_it_is_always_the_public_demo(probe: dict[str, Any]) -> None:
    calls = probe["calls"]

    assert probe["environment"] == {"PUBLIC_DEMO": "true", "ENVIRONMENT": "production"}
    assert calls["health"] | {"seconds": None, "bytes": None} == {
        "status": 200,
        "bytes": None,
        "seconds": None,
        "content_type": "application/json",
        "demo_mode": True,
        "llm_configured": False,
        "database_configured": False,
        "environment": "production",
    }
    for name in ("ask", "ask_stream"):
        assert (calls[name]["status"], calls[name]["detail"]) == (403, ASK_DISABLED)
    assert calls["ask_invalid"]["status"] == 422


@pytest.mark.parametrize("result", ["probe", "listed_only"])
def test_every_demo_call_answers_without_a_database(
    result: str, request: pytest.FixtureRequest
) -> None:
    calls = request.getfixturevalue(result)["calls"]

    for name in ("stance_benchmark", "evaluation", "openapi", "docs"):
        assert calls[name]["status"] == 200, name
    assert calls["evaluation_question"] | {"seconds": None, "bytes": None} == {
        "status": 200,
        "bytes": None,
        "seconds": None,
        "content_type": "application/json",
        "passages_available": True,  # data/index/chunks.jsonl ships with the function
    }
    assert (calls["ask_examples"]["status"], calls["ask_examples"]["items"]) == (200, 8)
    for name in DATABASE_CALLS:  # the documented 503, not a crash
        assert (calls[name]["status"], calls[name]["detail"]) == (503, NOT_CONFIGURED), name
    assert max(call["bytes"] for call in calls.values()) < 4_500_000  # Vercel's response limit


def test_no_heavy_module_is_imported(probe: dict[str, Any]) -> None:
    assert probe["heavy"] == []
    assert not {"torch", "openai", "chromadb", "transformers"} & set(probe["distributions"])


# --- requirements.txt -------------------------------------------------------------------------


def test_requirements_pin_the_versions_constraints_pins() -> None:
    constraints = pins(CONSTRAINTS)

    for name, requirement in pins(REQUIREMENTS).items():
        assert requirement.marker is None and requirement.url is None, requirement
        assert [s.operator for s in requirement.specifier] == ["=="], requirement
        assert name in constraints, f"{name} is not in backend/constraints.txt"
        assert requirement.specifier == constraints[name].specifier, (
            f"requirements.txt pins {requirement}, backend/constraints.txt {constraints[name]}"
        )


def test_requirements_list_what_the_demo_imports_and_nothing_else(
    listed_only: dict[str, Any],
) -> None:
    listed = pins(REQUIREMENTS)
    imported = {canonicalize_name(name) for name in listed_only["distributions"]}
    needed = needs({name: set(listed[name].extras) for name in imported & set(listed)})

    assert sorted(imported - set(listed)) == [], "imported, but not in requirements.txt"
    assert sorted(needed - set(listed)) == [], "needed on Vercel, but not in requirements.txt"
    assert sorted(set(listed) - needed) == [], "in requirements.txt, but never needed"


def test_requirements_txt_is_the_manifest_vercel_installs() -> None:
    assert sorted(pushed("requirements.txt", ".python-version")) == [
        ".python-version",
        "requirements.txt",
    ]
    assert pushed(*SHADOWING) == []  # e.g. the pyproject.toml `vercel build` leaves is ignored


# --- vercel.json, .python-version and .vercelignore -------------------------------------------


def test_vercel_json_configures_the_entrypoint() -> None:
    config = json.loads((REPO_ROOT / "vercel.json").read_text(encoding="utf-8"))
    excluded = config["functions"][ENTRY]["excludeFiles"]
    match = re.fullmatch(r"\{([^{}]+)\}/\*\*", excluded)

    assert config["framework"] == "fastapi"
    assert (REPO_ROOT / ENTRY).is_file()
    assert len(config["regions"]) == 1  # the Hobby plan runs functions in one region
    assert len(excluded) <= 256 and match, excluded
    folders = match.group(1).split(",")
    for path in SHIPPED:
        assert not any(path.startswith(f"{folder}/") for folder in folders), path
    assert PYTHON_VERSION in {"3.12", "3.13", "3.14"}  # the versions Vercel's runtime offers


def test_vercelignore_keeps_local_files_out_of_a_cli_upload_and_needed_ones_in(
    tmp_path: Path,
) -> None:
    """`vercel deploy` from a checkout uploads whatever .vercelignore does not list, git-ignored or
    not. Matched with git's own rules, which .vercelignore follows, and no global excludes file."""
    needed = pushed(
        "api", "backend/app", "data", "requirements.txt", ".python-version", "vercel.json"
    )
    local_only = ["backend/.env", ".env", "frontend/.env.local", "backend/.venv/bin/python"]
    local_only += ["data/chroma/chroma.sqlite3", "data/models/finbert/config.json"]
    git("init", "--quiet", cwd=tmp_path)
    shutil.copy2(REPO_ROOT / ".vercelignore", tmp_path / ".gitignore")
    paths = "".join(f"{path}\n" for path in [*needed, *local_only])
    no_global = ("-c", "core.excludesFile=/dev/null")
    ignored = git(*no_global, "check-ignore", "--no-index", "--stdin", cwd=tmp_path, stdin=paths)

    assert set(SHIPPED) <= set(needed)
    assert sorted(ignored) == sorted(local_only)
