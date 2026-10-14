"""Tests that every endpoint validates input fully, and the app-wide routing and CORS."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

INVALID_REQUESTS = {
    "ask-question-too-short": ("POST", "/api/ask", {"question": "Hi"}),
    "ask-question-blank": ("POST", "/api/ask", {"question": "     "}),
    "ask-top-k-zero": ("POST", "/api/ask", {"question": "Rates?", "top_k": 0}),
    "ask-top-k-too-big": ("POST", "/api/ask", {"question": "Rates?", "top_k": 21}),
    "ask-unknown-retrieval": ("POST", "/api/ask", {"question": "Rates?", "retrieval": "tfidf"}),
    "ask-unknown-doc-type": ("POST", "/api/ask", {"question": "Rates?", "doc_types": ["memo"]}),
    "ask-unknown-field": ("POST", "/api/ask", {"question": "Rates?", "asof": "2020-03-01"}),
    "ask-empty-doc-types": ("POST", "/api/ask", {"question": "Rates?", "doc_types": []}),
    "compare-same-date": (
        "POST",
        "/api/compare",
        {"base_date": "2024-09-18", "target_date": "2024-09-18"},
    ),
    "compare-base-after-target": (
        "POST",
        "/api/compare",
        {"base_date": "2024-09-18", "target_date": "2024-07-31"},
    ),
    "compare-unknown-analysis-mode": (
        "POST",
        "/api/compare",
        {"base_date": "2024-07-31", "target_date": "2024-09-18", "analysis": "always"},
    ),
    "compare-doc-type-dropped": (
        "POST",
        "/api/compare",
        {"base_date": "2024-07-31", "target_date": "2024-09-18", "doc_type": "minutes"},
    ),
    "stance-start-after-end": ("GET", "/api/stance?start=2024-12-31&end=2024-01-01", None),
    "stance-unknown-classifier": ("GET", "/api/stance?classifier=vibes", None),
    "stance-unknown-param": ("GET", "/api/stance?start_date=2024-01-01", None),
    "stance-meeting-not-a-date": ("GET", "/api/stance/not-a-date", None),
    "stance-meeting-unknown-classifier": ("GET", "/api/stance/2024-09-18?classifier=vibes", None),
    "meetings-unknown-doc-type": ("GET", "/api/meetings?doc_type=memo", None),
    "meetings-as-of-not-a-date": ("GET", "/api/meetings?as_of=2024-02-30", None),
    "meetings-unknown-param": ("GET", "/api/meetings?asof=2024-10-01", None),
    "evaluation-question-bad-id": ("GET", "/api/evaluation/questions/q%201", None),
}


@pytest.mark.parametrize(
    ("method", "url", "body"), INVALID_REQUESTS.values(), ids=INVALID_REQUESTS.keys()
)
def test_invalid_request_is_422(
    client: TestClient, method: str, url: str, body: dict[str, Any] | None
) -> None:
    assert client.request(method, url, json=body).status_code == 422


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.is_redirect
    assert response.headers["location"] == "/docs"


def test_cors_allows_configured_origin(client: TestClient) -> None:
    response = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
