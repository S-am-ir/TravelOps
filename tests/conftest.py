"""Shared test fixtures and configuration."""

import pytest
import os

# Set test environment
os.environ.setdefault("GROQ_TOKEN", "test_key_not_used")
os.environ.setdefault("RAPIDAPI_KEY", "")
os.environ.setdefault("WEATHERAPI_KEY", "")
os.environ.setdefault("FAL_API_KEY", "")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-for-testing-only")
