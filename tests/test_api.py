"""Tests for the FastAPI endpoints — health, auth, chat, settings."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import os

# Ensure test env vars are set before importing main
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-for-testing-only")


@pytest.fixture(autouse=True)
def mock_graph_and_db():
    """Mock agent creation and auth DB to avoid needing real services."""
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=MagicMock(next=[]))
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "final_response": "Test response",
            "intent": "unknown",
            "messages": [],
        }
    )
    mock_checkpointer = MagicMock()
    mock_checkpointer._storage = {}

    async def fake_create_agent():
        return mock_graph, mock_checkpointer

    with (
        patch("src.main.create_agent", side_effect=fake_create_agent),
        patch("src.main.init_auth_db", new_callable=AsyncMock),
        patch("src.main.close_auth_db", new_callable=AsyncMock),
    ):
        # Force re-import to pick up the patched create_agent
        import importlib
        import src.main

        importlib.reload(src.main)
        # Directly set the app state so tests work without lifespan
        src.main._agent = mock_graph
        src.main._checkpointer = mock_checkpointer
        yield


@pytest.fixture
def client():
    from src.main import app

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestFrontendServing:
    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Travel Planner" in resp.text


class TestAuthEndpoints:
    @patch("src.auth.service.AuthService.register", new_callable=AsyncMock)
    def test_register_success(self, mock_register, client):
        from src.auth.service import TokenResponse, UserProfile

        mock_register.return_value = TokenResponse(
            access_token="test.jwt.token",
            user=UserProfile(id="u1", email="test@example.com"),
        )
        resp = client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "password": "password123",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["access_token"] == "test.jwt.token"
        assert data["user"]["email"] == "test@example.com"

    @patch("src.auth.service.AuthService.register", new_callable=AsyncMock)
    def test_register_duplicate_email(self, mock_register, client):
        mock_register.side_effect = ValueError("Email already registered")
        resp = client.post(
            "/auth/register",
            json={
                "email": "existing@example.com",
                "password": "password123",
            },
        )
        assert resp.status_code == 400
        assert "Email already registered" in resp.json()["detail"]

    @patch("src.auth.service.AuthService.login", new_callable=AsyncMock)
    def test_login_success(self, mock_login, client):
        from src.auth.service import TokenResponse, UserProfile

        mock_login.return_value = TokenResponse(
            access_token="login.jwt.token",
            user=UserProfile(id="u1", email="test@example.com"),
        )
        resp = client.post(
            "/auth/login",
            json={
                "email": "test@example.com",
                "password": "password123",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["access_token"] == "login.jwt.token"

    @patch("src.auth.service.AuthService.login", new_callable=AsyncMock)
    def test_login_wrong_password(self, mock_login, client):
        mock_login.side_effect = ValueError("Invalid email or password")
        resp = client.post(
            "/auth/login",
            json={
                "email": "test@example.com",
                "password": "wrong",
            },
        )
        assert resp.status_code == 401


class TestChatEndpoint:
    def test_chat_without_auth(self, client):
        """Chat works without authentication (ephemeral session)."""
        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert data["response"] == "Test response"

    def test_chat_with_thread_id(self, client):
        """Chat continues with the same thread_id."""
        resp = client.post(
            "/chat",
            json={
                "message": "Follow up",
                "thread_id": "my-thread-123",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["thread_id"] == "my-thread-123"

    def test_chat_invalid_json(self, client):
        resp = client.post(
            "/chat", content="not json", headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422  # Validation error


class TestDeleteThread:
    def test_delete_clears_memory(self, client):
        """Delete clears MemorySaver storage."""
        # First send a message to create a thread
        resp = client.post("/chat", json={"message": "Hello"})
        thread_id = resp.json()["thread_id"]

        # Then delete it
        resp = client.delete(f"/chat/{thread_id}")
        assert resp.status_code == 200
        assert resp.json()["cleared"] is True
