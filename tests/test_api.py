"""Tests for the FastAPI chat endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


# ── Health ────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_response_structure(client):
    resp = client.get("/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"


# ── Input validation ──────────────────────────────────────────


def test_chat_empty_message(client):
    resp = client.post("/chat", json={"message": "", "history": []})
    assert resp.status_code == 400


def test_chat_whitespace_only_message(client):
    resp = client.post("/chat", json={"message": "   ", "history": []})
    assert resp.status_code == 400


def test_chat_message_too_long(client):
    resp = client.post("/chat", json={"message": "x" * 2001, "history": []})
    assert resp.status_code == 400
    assert "too long" in resp.json()["detail"].lower()


def test_chat_message_at_max_length(client):
    async def mock_chat(query, history):
        yield "ok"

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post("/chat", json={"message": "x" * 2000, "history": []})
        assert resp.status_code == 200


def test_chat_missing_message_field(client):
    resp = client.post("/chat", json={"history": []})
    assert resp.status_code == 422


def test_chat_invalid_json(client):
    resp = client.post(
        "/chat", data="not-json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


# ── Streaming response ────────────────────────────────────────


def test_chat_returns_stream(client):
    async def mock_chat(query, history):
        yield "Hello "
        yield "from Warwick!"

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post("/chat", json={"message": "Hello", "history": []})
        assert resp.status_code == 200
        assert "Hello" in resp.text


def test_chat_streams_all_tokens(client):
    async def mock_chat(query, history):
        for word in ["Warwick", " Prep", " School"]:
            yield word

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post(
            "/chat", json={"message": "Tell me about the school", "history": []}
        )
        assert resp.status_code == 200
        assert "Warwick Prep School" in resp.text


def test_chat_content_type_is_text_plain(client):
    async def mock_chat(query, history):
        yield "response"

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post("/chat", json={"message": "Hi", "history": []})
        assert "text/plain" in resp.headers["content-type"]


def test_chat_passes_history_to_pipeline(client):
    received_history = []

    async def mock_chat(query, history):
        received_history.extend(history)
        yield "ok"

    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    with patch("src.api.main.chat", side_effect=mock_chat):
        client.post("/chat", json={"message": "follow-up", "history": history})

    assert len(received_history) == 2
    assert received_history[0]["role"] == "user"


def test_chat_empty_history_is_valid(client):
    async def mock_chat(query, history):
        yield "response"

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post("/chat", json={"message": "Hi"})
        assert resp.status_code == 200


# ── Error handling ────────────────────────────────────────────


def test_chat_error_does_not_expose_traceback(client):
    async def mock_chat(query, history):
        raise RuntimeError("Internal connection failure with secret_key=abc123")
        yield  # make it a generator

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post("/chat", json={"message": "Hi", "history": []})
        assert resp.status_code == 200
        assert "secret_key" not in resp.text
        assert "Traceback" not in resp.text
        assert "RuntimeError" not in resp.text
        assert "error occurred" in resp.text.lower()


# ── Feedback endpoint ─────────────────────────────────────────


def test_feedback_thumbs_up_accepted(client):
    resp = client.post(
        "/feedback", json={"message": "Hello", "response": "Hi there", "rating": 1}
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_feedback_thumbs_down_accepted(client):
    resp = client.post(
        "/feedback", json={"message": "Hello", "response": "Hi there", "rating": -1}
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_feedback_invalid_rating_rejected(client):
    resp = client.post(
        "/feedback", json={"message": "Hello", "response": "Hi", "rating": 0}
    )
    assert resp.status_code == 400


def test_chat_debug_endpoint_removed(client):
    resp = client.post("/chat-debug", json={"message": "test", "history": []})
    assert resp.status_code == 404 or resp.status_code == 405
