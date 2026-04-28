"""Tests for the FastAPI chat endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_empty_message(client):
    resp = client.post("/chat", json={"message": "", "history": []})
    assert resp.status_code == 400


def test_chat_returns_stream(client):
    async def mock_chat(query, history):
        yield "Hello "
        yield "from Warwick!"

    with patch("src.api.main.chat", side_effect=mock_chat):
        resp = client.post("/chat", json={"message": "Hello", "history": []})
        assert resp.status_code == 200
        assert "Hello" in resp.text
