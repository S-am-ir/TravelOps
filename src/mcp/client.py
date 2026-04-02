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
    """Load and cache MCP tools. Pass `servers` to filter by server name(s)."""
    global _tools, _tool_server_map

    if not _tools:
        await _load_tools()

    if servers:
        return [t for t in _tools if _tool_server_map.get(t.name) in servers]

    return _tools


async def _load_tools():
    """Fetch all tools from all MCP servers and build the server map."""
    global _tools, _tool_server_map

    mcp_servers = _build_mcp_servers()
    all_tools = []
    _tool_server_map = {}

    for server_key, server_config in mcp_servers.items():
        try:
            # Load tools from each server individually to prevent TaskGroup exceptions
            # from taking down the entire MultiServer client
            single_client = MultiServerMCPClient({server_key: server_config})
            server_tools = await single_client.get_tools()
            all_tools.extend(server_tools)
            for t in server_tools:
                _tool_server_map[t.name] = server_key
        except BaseException as e:
            import traceback
            err_details = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print(f"[MCP] Critical Warning: loaded failed for '{server_key}':\n{err_details}")

    _tools = all_tools
    print(f"[MCP] Loaded {len(_tools)} tools: {[t.name for t in _tools]}")
    print(f"[MCP] Server map: {_tool_server_map}")



async def reset_mcp_client():
    """Clear tool cache and reload from MCP servers."""
    global _tools, _tool_server_map
    _tools = []
    _tool_server_map = {}
    await _load_tools()
    return _tools
