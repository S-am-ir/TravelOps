import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from typing import List, Dict, Optional
from src.config.settings import settings


def _build_mcp_servers() -> Dict:
    """Build MCP server config from settings. Supports per-service host overrides."""
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


_tools: List[BaseTool] = []
_tool_server_map: Dict[str, str] = {}


async def get_mcp_tools(servers: Optional[List[str]] = None) -> List[BaseTool]:
    """Load and cache MCP tools. If no tools loaded yet, try to load with retry.
    On subsequent calls, if still empty, try again (in case servers were slow to start)."""
    global _tools, _tool_server_map

    if not _tools:
        await _load_tools_with_retry()

    if servers:
        return [t for t in _tools if _tool_server_map.get(t.name) in servers]

    return _tools


async def _load_single_server(server_key: str, server_config: dict) -> tuple[List[BaseTool], bool]:
    """
    Attempt to load tools from a single MCP server.
    Returns (tools, success). Does NOT raise — failures are handled by caller.
    """
    try:
        single_client = MultiServerMCPClient({server_key: server_config})
        server_tools = await single_client.get_tools()
        print(f"[MCP] OK: '{server_key}' → {len(server_tools)} tools")
        return server_tools, True
    except BaseException as e:
        # Unwrap ExceptionGroup to get the real error message
        root_cause = str(e)
        if hasattr(e, "exceptions") and e.exceptions:
            root_cause = str(e.exceptions[0])
        print(f"[MCP] '{server_key}' not ready: {root_cause}")
        return [], False


async def _load_tools_with_retry(max_attempts: int = 8, base_delay: float = 3.0):
    """
    Try to load tools from all MCP servers with exponential backoff retry.

    The MCP servers (travel, search, comms) are started by supervisord at the
    same time as the API. On the first request, they may not have bound to their
    ports yet. This function retries up to max_attempts times before giving up.

    Strategy: Try all 3 servers. If any fail, wait and retry ALL failed ones.
    Servers that succeed early are kept and not re-fetched.
    """
    global _tools, _tool_server_map

    mcp_servers = _build_mcp_servers()
    loaded_tools: Dict[str, List[BaseTool]] = {}
    pending_servers = dict(mcp_servers)  # servers we still need to load

    for attempt in range(1, max_attempts + 1):
        if not pending_servers:
            break

        print(f"[MCP] Load attempt {attempt}/{max_attempts} for: {list(pending_servers.keys())}")

        still_pending = {}
        for server_key, server_config in pending_servers.items():
            tools, success = await _load_single_server(server_key, server_config)
            if success:
                loaded_tools[server_key] = tools
            else:
                still_pending[server_key] = server_config

        pending_servers = still_pending

        if pending_servers and attempt < max_attempts:
            # Exponential backoff but capped at 10s
            delay = min(base_delay * (1.5 ** (attempt - 1)), 10.0)
            print(f"[MCP] {len(pending_servers)} server(s) not ready, retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    # Build the final tool list and server map
    all_tools = []
    tool_server_map = {}
    for server_key, tools in loaded_tools.items():
        for t in tools:
            tool_server_map[t.name] = server_key
        all_tools.extend(tools)

    if pending_servers:
        print(f"[MCP] WARNING: gave up on servers after {max_attempts} attempts: {list(pending_servers.keys())}")
        print("[MCP] Agent will still work but without those server's tools.")

    _tools = all_tools
    _tool_server_map = tool_server_map
    print(f"[MCP] Final: {len(_tools)} tools loaded: {[t.name for t in _tools]}")
    print(f"[MCP] Server map: {_tool_server_map}")


async def reset_mcp_client():
    """Clear tool cache and reload from MCP servers."""
    global _tools, _tool_server_map
    _tools = []
    _tool_server_map = {}
    await _load_tools_with_retry()
    return _tools
