"""Tests for the published OpenAPI contract and its exporter."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MakeClient = Callable[..., TestClient]

EXPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_openapi.py"

OPERATIONS = {
    ("/api/health", "get"): "getHealth",
    ("/api/meetings", "get"): "listMeetings",
    ("/api/ask", "post"): "ask",
    ("/api/ask/stream", "post"): "askStream",
    ("/api/ask/examples", "get"): "listAskExamples",
    ("/api/stance", "get"): "getStanceTimeline",
    ("/api/stance/benchmark", "get"): "getStanceBenchmark",
    ("/api/stance/{meeting_date}", "get"): "getStanceForMeeting",
    ("/api/compare", "post"): "compareStatements",
    ("/api/evaluation", "get"): "getEvaluationReport",
    ("/api/evaluation/questions/{question_id}", "get"): "getEvaluationQuestion",
}


@pytest.fixture
def paths(make_client: MakeClient) -> dict:
    return make_client().app.openapi()["paths"]


@pytest.mark.parametrize(("path", "method"), OPERATIONS)
def test_operation_present(paths: dict, path: str, method: str) -> None:
    assert paths[path][method]["operationId"] == OPERATIONS[path, method]


@pytest.mark.parametrize(("path", "method"), OPERATIONS)
def test_no_operation_documents_501(paths: dict, path: str, method: str) -> None:
    assert "501" not in paths[path][method]["responses"]


def test_meetings_documents_503_and_as_of(paths: dict) -> None:
    operation = paths["/api/meetings"]["get"]

    assert operation["responses"]["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert {p["name"] for p in operation["parameters"]} == {"doc_type", "as_of"}


def test_stance_documents_its_errors(paths: dict) -> None:
    error = {"$ref": "#/components/schemas/ErrorResponse"}
    timeline = paths["/api/stance"]["get"]
    meeting = paths["/api/stance/{meeting_date}"]["get"]
    benchmark = paths["/api/stance/benchmark"]["get"]

    assert {p["name"] for p in timeline["parameters"]} == {"classifier", "start", "end"}
    assert timeline["responses"]["503"]["content"]["application/json"]["schema"] == error
    for code in ("404", "503"):
        assert meeting["responses"][code]["content"]["application/json"]["schema"] == error
    assert benchmark["responses"]["503"]["content"]["application/json"]["schema"] == error
    assert "parameters" not in benchmark


def test_ask_documents_502_and_503(paths: dict) -> None:
    responses = paths["/api/ask"]["post"]["responses"]

    for code in ("502", "503"):
        assert responses[code]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }


def test_ask_stream_documents_its_events(make_client: MakeClient) -> None:
    schema = make_client().app.openapi()
    responses = schema["paths"]["/api/ask/stream"]["post"]["responses"]
    stream = responses["200"]["content"]["text/event-stream"]
    payloads = ["StreamStatus", "StreamDelta", "StreamCitation", "AskResponse", "StreamError"]

    # The data payloads are components, so openapi-typescript generates their types.
    assert set(payloads) <= set(schema["components"]["schemas"])
    assert stream["schema"]["anyOf"] == [{"$ref": f"#/components/schemas/{p}"} for p in payloads]
    events = {
        variant["properties"]["event"]["const"]: variant["properties"]["data"]["contentSchema"]
        for variant in stream["itemSchema"]["oneOf"]
    }
    assert events == {
        "status": {"$ref": "#/components/schemas/StreamStatus"},
        "delta": {"$ref": "#/components/schemas/StreamDelta"},
        "citation": {"$ref": "#/components/schemas/StreamCitation"},
        "done": {"$ref": "#/components/schemas/AskResponse"},
        "error": {"$ref": "#/components/schemas/StreamError"},
    }
    # Errors before the stream starts are JSON, as for /api/ask; a 502 can only be an event.
    assert responses["503"]["content"] == {
        "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
    }
    assert "422" in responses and "502" not in responses
    assert responses["403"]["content"] == responses["503"]["content"]


@pytest.mark.parametrize("path", ["/api/ask", "/api/ask/stream"])
def test_ask_documents_the_demo_403(paths: dict, path: str) -> None:
    response = paths[path]["post"]["responses"]["403"]

    assert "public demo" in response["description"]
    assert response["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }


def test_ask_examples_documents_its_list_and_503(paths: dict) -> None:
    responses = paths["/api/ask/examples"]["get"]["responses"]

    assert responses["200"]["content"]["application/json"]["schema"] == {
        "items": {"$ref": "#/components/schemas/AskExample"},
        "title": "Response Listaskexamples",
        "type": "array",
    }
    assert responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }


def test_export_is_deterministic(tmp_path: Path) -> None:
    outputs = []
    for seed in ("0", "1"):
        out = tmp_path / f"openapi-{seed}.json"
        env = {**os.environ, "PYTHONHASHSEED": seed}
        subprocess.run([sys.executable, EXPORT_SCRIPT, out], check=True, env=env)
        outputs.append(out.read_bytes())

    assert outputs[0] == outputs[1]
    assert outputs[0].endswith(b"}\n")
    assert set(json.loads(outputs[0])["paths"]) == {path for path, _ in OPERATIONS}
