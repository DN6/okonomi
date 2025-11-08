import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

from ..config import config


os.environ["MCP_TIMEOUT"] = str(config.tools.mcp.timeout)

mcp = FastMCP("research_tools")


def get_tavily_client() -> TavilyClient:
    """Get or create Tavily API client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")
    return TavilyClient(api_key=api_key)


@mcp.tool()
def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> str:
    """Search the web using Tavily API for research and context gathering.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
        search_depth: Search depth - 'basic' or 'advanced' (default: 'basic').
        include_domains: Optional list of domains to include in search.
        exclude_domains: Optional list of domains to exclude from search.

    Returns:
        JSON string with search results containing title, url, content, and score.
    """
    client = get_tavily_client()

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
        )

        results = []
        text_blocks = []

        for idx, result in enumerate(response.get("results", []), 1):
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                }
            )

            # Format as text block
            text_blocks.append(
                f"[{idx}] {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Score: {result.get('score', 0.0):.2f}\n"
                f"Content: {result.get('content', 'No content available')}\n"
            )

        formatted_text = f"Search Results for: {query}\n{'=' * 80}\n\n" + "\n".join(text_blocks)

        return formatted_text

    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


@mcp.tool()
def get_search_context(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> str:
    """Get a concise context string from web search results, ideal for augmenting prompts.

    Args:
        query: Search query string.
        max_results: Maximum number of results to include (default: 5).
        search_depth: Search depth - 'basic' or 'advanced' (default: 'basic').

    Returns:
        JSON string with a formatted context summary from search results.
    """
    client = get_tavily_client()

    try:
        response = client.get_search_context(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )

        return json.dumps({"query": query, "context": response})

    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


@mcp.tool()
def qna_search(query: str, search_depth: str = "advanced") -> str:
    """Get a direct answer to a question using Tavily's QNA search.

    Args:
        query: Question to get answered.
        search_depth: Search depth - 'basic' or 'advanced' (default: 'advanced').

    Returns:
        JSON string with the answer to the question.
    """
    client = get_tavily_client()

    try:
        response = client.qna_search(
            query=query,
            search_depth=search_depth,
        )

        return json.dumps({"query": query, "answer": response})

    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


if __name__ == "__main__":
    mcp.run()
