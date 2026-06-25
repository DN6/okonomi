"""Web tools — search and page fetching backed by Tavily."""

import os
from typing import TypedDict

from tavily import TavilyClient

from okonomi.tools._metadata import ToolMetadataMixin


class WebResultItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float


class WebSearchResult(TypedDict, total=False):
    """Return type for web.web_search.

    ``query`` echoes the input, ``results`` is a list of
    ``WebResultItem`` dicts. ``error`` only set on failure.
    """

    query: str
    results: list[WebResultItem]
    error: str


class WebFetchResult(TypedDict, total=False):
    """Return type for web.web_fetch."""

    url: str
    content: str
    error: str


class WebTools(ToolMetadataMixin):
    __namespace__ = "web"
    __description__ = "Web search and page fetching via Tavily."

    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = TavilyClient(api_key=api_key) if api_key else None

    def web_search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "ultra-fast",
        include_images: bool = True,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> WebSearchResult:
        """Search the web for a query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 5).
            search_depth: ``'ultra-fast'`` for quick scans or ``'advanced'``
                for deeper, scored results.
            include_images: Include image results when available.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.

        Returns:
            ``{"query": str, "results": [{"title", "url", "content", "score"}, ...]}``
            on success, ``{"error": str, "query": str}`` on failure.
        """
        if not self.tavily_client:
            return {"error": "TAVILY_API_KEY not configured", "query": query}

        try:
            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_images=include_images,
                include_image_descriptions=True,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in response.get("results", [])
            ]
            return {"query": query, "results": results}
        except Exception as e:
            return {"error": str(e), "query": query}

    def web_fetch(
        self,
        url: str,
        extract_depth: str = "basic",
    ) -> WebFetchResult:
        """Fetch a URL and return its main content as clean text.

        Uses Tavily's extract endpoint, which strips boilerplate (nav,
        ads, footers) and returns the article body in markdown-ish form.
        Use this when you have a specific page to read — pair with
        ``web_search`` for discovery.

        Args:
            url: The page URL to fetch.
            extract_depth: ``'basic'`` (faster) or ``'advanced'`` (deeper
                extraction for content-heavy pages).

        Returns:
            ``{"url": str, "content": str}`` on success,
            ``{"error": str, "url": str}`` on failure.
        """
        if not self.tavily_client:
            return {"error": "TAVILY_API_KEY not configured", "url": url}

        try:
            response = self.tavily_client.extract(urls=[url], extract_depth=extract_depth)
            results = response.get("results", [])
            if not results:
                failed = response.get("failed_results", [])
                err = failed[0].get("error", "no content extracted") if failed else "no content extracted"
                return {"error": err, "url": url}
            first = results[0]
            return {
                "url": first.get("url", url),
                "content": first.get("raw_content", ""),
            }
        except Exception as e:
            return {"error": str(e), "url": url}
