"""Tests for the MCP comms server."""

import pytest


class TestCommsServer:
    def test_server_imports(self):
        from src.mcp.servers.comms import mcp

        assert mcp.name == "comms"
