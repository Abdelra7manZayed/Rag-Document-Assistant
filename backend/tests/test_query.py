"""
Tests for the /query endpoint.
Run with: pytest tests/ -v
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── helpers ──────────────────────────────────────────────────────────────────

FAKE_CHUNKS = [
    {
        "document": "Python generators use the yield keyword to produce values one at a time.",
        "source": "python_advanced.txt",
        "chunk_id": "chunk_3",
    }
]


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    """GET /health should return 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_health_endpoint():
    """GET /api/v1/health should also return 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


@patch("app.api.routes.query.retrieve_chunks", return_value=FAKE_CHUNKS)
@patch(
    "app.api.routes.query.generate_answer",
    return_value="A generator uses yield. Sources used: chunk_3",
)
def test_query_happy_path(mock_gen, mock_retrieve):
    """POST /query with a valid question returns 200 with answer and sources."""
    response = client.post(
        "/api/v1/query",
        json={"question": "What is a Python generator?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "python_advanced.txt"


def test_query_missing_question_returns_422():
    """POST /query without a question body should return 422 Unprocessable Entity."""
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422


def test_query_too_short_returns_422():
    """Questions shorter than 3 chars should fail validation."""
    response = client.post("/api/v1/query", json={"question": "hi"})
    assert response.status_code == 422


@patch("app.api.routes.query.retrieve_chunks", side_effect=RuntimeError("store not loaded"))
def test_query_retrieval_failure_returns_503(mock_retrieve):
    """If the vector store fails, the endpoint returns 503."""
    response = client.post(
        "/api/v1/query",
        json={"question": "What is a generator?"},
    )
    assert response.status_code == 503
