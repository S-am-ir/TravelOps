"""
MCP Tool Loading — two modes:

1. DIRECT MODE (MCP_DIRECT=true, used on Render):
   Import tool functions directly from server modules and wrap as LangChain tools.
   Single process, zero network, zero ports, 100% reliable.

2. HTTP MODE  (MCP_DIRECT unset/false, used with docker-compose):
   Connect to MCP servers over HTTP using langchain-mcp-adapters.
   Each server runs in its own container.
"""

import os
import asyncio
import functools
import inspect
from langchain_core.tools import StructuredTool, BaseTool
from typing import List, Dict, Optional
from src.config.settings import settings


_tools: List[BaseTool] = []
_tool_server_map: Dict[str, str] = {}

# Single-process mode: import tools directly instead of via MCP HTTP.
# Set MCP_DIRECT=true on Render for reliable single-container deployment.
# Leave unset/false for docker-compose multi-container local dev.
MCP_DIRECT = os.environ.get("MCP_DIRECT", "false").lower() == "true"


async def get_mcp_tools(servers: Optional[List[str]] = None) -> List[BaseTool]:
    """Load and cache MCP tools. Pass `servers` to filter by server name(s)."""
    global _tools, _tool_server_map

    if not _tools:
        if MCP_DIRECT:
            await _load_direct_tools()
        else:
            await _load_tools_with_retry()

    if servers:
        return [t for t in _tools if _tool_server_map.get(t.name) in servers]

    return _tools


# ── Direct Tool Loading (single-process, no MCP HTTP) ────────────────


def _make_serializing_wrapper(fn):
    """Wrap an async tool function so it returns JSON strings for LangChain.

    MCP tool functions return Pydantic models (FlightResults, SearchResults, etc.).
    LangChain tools must return strings. This wrapper serializes the result.
    """
    @functools.wraps(fn)
    async def wrapper(**kwargs):
        result = await fn(**kwargs)
        # Pydantic v2
        if hasattr(result, "model_dump_json"):
            return result.model_dump_json()
        # Pydantic v1 fallback
        if hasattr(result, "json"):
            return result.json()
        return str(result)

    # Preserve the original function's signature so StructuredTool can
    # infer the correct args_schema from the type annotations.
    wrapper.__signature__ = inspect.signature(fn)
    annotations = dict(getattr(fn, "__annotations__", {}))
    annotations.pop("return", None)  # Remove return type (we return str now)
    wrapper.__annotations__ = annotations
    return wrapper


async def _load_direct_tools():
    """Import tool functions directly from MCP server modules.

    Used for single-container deployments (Render) where running 3 separate
    MCP HTTP server processes wastes memory and causes reliability issues
    (port binding races, startup timing, connection pool failures).

    The @mcp.tool() decorator registers the function AND returns it unchanged.
    So decorated functions can be called directly as normal async functions.
    """
    global _tools, _tool_server_map

    print("[MCP Direct] Loading tools directly from server modules (no HTTP)...")

    # Map server keys to their module paths
    server_modules = {
        "travel": "src.mcp.servers.travel",
        "search": "src.mcp.servers.search",
        "comms": "src.mcp.servers.comms",
    }

    all_tools = []
    tool_map = {}

    for server_key, module_path in server_modules.items():
        try:
            module = __import__(module_path, fromlist=[""])

            # Find all public async functions (these are the @mcp.tool() ones).
            # Helper functions are prefixed with _ and will be skipped.
            count = 0
            for name, func in inspect.getmembers(module, inspect.iscoroutinefunction):
                if name.startswith("_"):
                    continue

                wrapped = _make_serializing_wrapper(func)

                tool = StructuredTool.from_function(
                    coroutine=wrapped,
                    name=name,
                    description=func.__doc__ or f"Tool from {server_key} server",
                )
                all_tools.append(tool)
                tool_map[name] = server_key
                count += 1

            print(f"[MCP Direct] '{server_key}' → {count} tools loaded")

        except Exception as e:
            import traceback
            print(
                f"[MCP Direct] FAILED to load '{server_key}': "
                f"{traceback.format_exc()}"
            )

    _tools = all_tools
    _tool_server_map = tool_map
    print(f"[MCP Direct] Total: {len(_tools)} tools: {[t.name for t in _tools]}")
    print(f"[MCP Direct] Server map: {_tool_server_map}")


# ── MCP HTTP Tool Loading (multi-container, docker-compose) ──────────


def _build_mcp_servers() -> Dict:
    """Build MCP server config from settings."""
    return {
        "travel": {
            "url": f"http://{settings.mcp_travel_resolved_host}:{settings.mcp_travel_port}/mcp",
            "transport": "streamable_http",
        },
        "search": {
            "url": f"http://{settings.mcp_search_resolved_host}:{settings.mcp_search_port}/mcp",
            "transport": "streamable_http",
        },
        "comms": {
            "url": f"http://{settings.mcp_comms_resolved_host}:{settings.mcp_comms_port}/mcp",
            "transport": "streamable_http",
        },
    }


async def _load_single_server(server_key: str, server_config: dict) -> tuple:
    """Load tools from one MCP HTTP server. Returns (tools, success)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    try:
        single_client = MultiServerMCPClient({server_key: server_config})
        server_tools = await single_client.get_tools()
        print(f"[MCP HTTP] OK: '{server_key}' → {len(server_tools)} tools")
        return server_tools, True
    except BaseException as e:
        root_cause = str(e)
        if hasattr(e, "exceptions") and e.exceptions:
            root_cause = str(e.exceptions[0])
        print(f"[MCP HTTP] '{server_key}' not ready: {root_cause}")
        return [], False


async def _load_tools_with_retry(max_attempts: int = 8, base_delay: float = 3.0):
    """Load tools via MCP HTTP with exponential backoff retry."""
    global _tools, _tool_server_map

    mcp_servers = _build_mcp_servers()
    loaded_tools: Dict[str, List[BaseTool]] = {}
    pending_servers = dict(mcp_servers)

    for attempt in range(1, max_attempts + 1):
        if not pending_servers:
            break

        print(
            f"[MCP HTTP] Load attempt {attempt}/{max_attempts} "
            f"for: {list(pending_servers.keys())}"
        )

        still_pending = {}
        for server_key, server_config in pending_servers.items():
            tools, success = await _load_single_server(server_key, server_config)
            if success:
                loaded_tools[server_key] = tools
            else:
                still_pending[server_key] = server_config

        pending_servers = still_pending

        if pending_servers and attempt < max_attempts:
            delay = min(base_delay * (1.5 ** (attempt - 1)), 10.0)
            print(
                f"[MCP HTTP] {len(pending_servers)} server(s) not ready, "
                f"retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    all_tools = []
    tool_map = {}
    for server_key, tools in loaded_tools.items():
        for t in tools:
            tool_map[t.name] = server_key
        all_tools.extend(tools)

    if pending_servers:
        print(
            f"[MCP HTTP] WARNING: gave up on servers after {max_attempts} "
            f"attempts: {list(pending_servers.keys())}"
        )

    _tools = all_tools
    _tool_server_map = tool_map
    print(f"[MCP HTTP] Final: {len(_tools)} tools loaded: {[t.name for t in _tools]}")


# ── Reset ────────────────────────────────────────────────────────────


async def reset_mcp_client():
    """Clear tool cache and reload from MCP servers."""
    global _tools, _tool_server_map
    _tools = []
    _tool_server_map = {}
    if MCP_DIRECT:
        await _load_direct_tools()
    else:
        await _load_tools_with_retry()
    return _tools
