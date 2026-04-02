"""
Native Tool Registry (Replaced MCP)

All tools are now native LangChain tools. Zero network overhead, zero ports,
100% reliable execution in the same process.
"""

from typing import List, Optional
from langchain_core.tools import BaseTool

# Import the native LangChain tools directly
from src.mcp.servers.travel import get_weather, search_flights, search_hotels
from src.mcp.servers.search import web_search, web_search_multi
from src.mcp.servers.comms import send_telegram_message

_all_tools = [
    get_weather,
    search_flights,
    search_hotels,
    web_search,
    web_search_multi,
    send_telegram_message,
]


async def get_mcp_tools(servers: Optional[List[str]] = None) -> List[BaseTool]:
    """
    Returns the list of all available LangChain tools.
    (Kept name 'get_mcp_tools' for backward compatibility with existing imports)
    """
    if servers:
        # If specific server tools were requested, we just return the full list anyway
        # because the new architecture doesn't use the server boundary logic,
        # but you could map them here if needed.
        return _all_tools
    
    return _all_tools

async def reset_mcp_client():
    """No-op for backward compatibility."""
    return _all_tools
