"""Tests for the auth service — JWT, password hashing, token operations."""

import pytest
from src.auth.service import (
    _hash_password,
    _verify_password,
    create_access_token,
    decode_access_token,
)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = _hash_password("mysecretpassword")
        assert hashed != "mysecretpassword"
        assert _verify_password("mysecretpassword", hashed)

    def test_wrong_password_fails(self):
        hashed = _hash_password("correct_password")
        assert not _verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        h1 = _hash_password("password123")
        h2 = _hash_password("password123")
        assert h1 != h2  # bcrypt uses random salt


class TestJWTTokens:
    def test_create_and_decode_token(self):
        token = create_access_token("user-123", "test@example.com")
        claims = decode_access_token(token)
        assert claims is not None
        assert claims["sub"] == "user-123"
        assert claims["email"] == "test@example.com"
        assert "exp" in claims

    def test_invalid_token_returns_none(self):
        assert decode_access_token("invalid.token.here") is None

    def test_empty_token_returns_none(self):
        assert decode_access_token("") is None

    def test_tampered_token_returns_none(self):
        token = create_access_token("user-123", "test@example.com")
        tampered = token[:-5] + "xxxxx"
        assert decode_access_token(tampered) is None
