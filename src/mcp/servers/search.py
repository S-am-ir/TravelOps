"""
Search MCP Server — real-time web search using Tavily.

Tavily provides clean, LLM-ready search results with AI-generated answers,
citations, and structured snippets. Much better than DuckDuckGo for agent consumption.

Split from the booking server (flights/hotels) so the agent can independently:
  - Search for buses, micro, transport options
  - Find places to visit, activities, local tips
  - Get real-time info (road conditions, events, prices)
  - Fallback when booking APIs return no results
  - Currency conversion, budget options, cheap accommodation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional, List
from src.config.settings import settings

mcp = FastMCP(
    "search",
    host=settings.mcp_host,
    port=settings.mcp_search_port,
    json_response=True,
)


# ── Models ────────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    title: str
    url: str
    content: str


class SearchResults(BaseModel):
    query: str
    answer: str  # AI-generated summary from Tavily
    results: List[SearchResult]
    total: int
    error: Optional[str] = None


# ── Tavily client helper ─────────────────────────────────────────────────


def _get_tavily():
    from tavily import TavilyClient

    api_key = (
        settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
    )
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


# ── Web Search ────────────────────────────────────────────────────────────


@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> SearchResults:
    """Search the web for real-time information using Tavily.

    Use this for:
    - Places to visit, things to do, local tips in any location
    - Bus, micro, local transport options, schedules, and prices
    - Road conditions, travel advisories, current events
    - Restaurant recommendations, nightlife, culture
    - Currency conversion rates
    - Budget accommodation, camping, guesthouses
    - Any general info not covered by flight/hotel booking tools

    Args:
        query: Search query. Be specific for better results.
               e.g. "best restaurants in Pokhara lakeside"
               e.g. "Kathmandu to Pokhara bus schedule price 2026"
               e.g. "USD to NPR exchange rate today"
               e.g. "cheap camping near Pokhara Nepal"
        max_results: Number of results to return (1-10, default 5).
        search_depth: "basic" (1 credit) or "advanced" (2 credits, more detail).

    Returns:
        SearchResults with AI-generated answer, plus title/url/content for each result.
    """
    client = _get_tavily()
    if not client:
        return SearchResults(
            query=query,
            answer="",
            results=[],
            total=0,
            error="TAVILY_API_KEY not configured",
        )

    try:
        max_results = max(1, min(max_results, 10))
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=False,
        )

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
            )
            for r in response.get("results", [])
        ]

        return SearchResults(
            query=query,
            answer=response.get("answer", ""),
            results=results,
            total=len(results),
        )

    except Exception as e:
        return SearchResults(
            query=query,
            answer="",
            results=[],
            total=0,
            error=str(e),
        )


# ── Multi-query Search ────────────────────────────────────────────────────


@mcp.tool()
async def web_search_multi(
    queries: List[str],
    max_results_per_query: int = 3,
) -> SearchResults:
    """Search multiple queries at once for comprehensive research.

    Use when the user's question needs information from multiple angles.
    Example: user asks "plan a trip to Pokhara" — search for:
      - "best time to visit Pokhara"
      - "Pokhara budget hotels 2026"
      - "things to do in Pokhara 3 days"
      - "Kathmandu to Pokhara bus options"

    Args:
        queries: List of search queries (2-5 recommended).
        max_results_per_query: Results per query (1-5, default 3).

    Returns:
        Combined SearchResults from all queries, deduplicated by URL.
    """
    client = _get_tavily()
    if not client:
        return SearchResults(
            query=str(queries),
            answer="",
            results=[],
            total=0,
            error="TAVILY_API_KEY not configured",
        )

    try:
        queries = queries[:5]
        max_results_per_query = max(1, min(max_results_per_query, 5))

        all_results = []
        seen_urls = set()
        answers = []

        for q in queries:
            try:
                response = client.search(
                    query=q,
                    max_results=max_results_per_query,
                    search_depth="basic",
                    include_answer=True,
                    include_raw_content=False,
                )
                if response.get("answer"):
                    answers.append(response["answer"])
                for r in response.get("results", []):
                    url = r.get("url", "")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(
                            SearchResult(
                                title=r.get("title", ""),
                                url=url,
                                content=r.get("content", ""),
                            )
                        )
            except Exception as e:
                print(f"[Search] query failed: {q}: {e}")

        combined_query = " | ".join(queries)
        combined_answer = " | ".join(answers) if answers else ""

        return SearchResults(
            query=combined_query,
            answer=combined_answer,
            results=all_results,
            total=len(all_results),
        )

    except Exception as e:
        return SearchResults(
            query=str(queries),
            answer="",
            results=[],
            total=0,
            error=str(e),
        )


if __name__ == "__main__":
    print(f"[MCP Search] running on {settings.mcp_host}:{settings.mcp_search_port}")
    mcp.run(transport="streamable-http")
