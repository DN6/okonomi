"""Web tools — search via DuckDuckGo and page fetching via httpx."""

import re

import httpx
import lxml.html
from ddgs import DDGS
from typing_extensions import TypedDict


_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}
_STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form")


class WebResultItem(TypedDict, total=False):
    title: str
    url: str
    content: str


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


class WebTools:

    def web_search(self, query: str, max_results: int = 5) -> WebSearchResult:
        """Search the web for a query via DuckDuckGo.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 5).

        Returns:
            ``{"query": str, "results": [{"title", "url", "content"}, ...]}``
            on success, ``{"error": str, "query": str}`` on failure.
        """
        try:
            hits = DDGS().text(query, max_results=max_results)
            results = [
                {
                    "title": h.get("title", ""),
                    "url": h.get("href", ""),
                    "content": h.get("body", ""),
                }
                for h in hits
            ]
            return {"query": query, "results": results}
        except Exception as e:
            return {"error": str(e), "query": query}

    def web_fetch(self, url: str) -> WebFetchResult:
        """Fetch a URL and return its main text content.

        Downloads the page and strips boilerplate (scripts, styles, nav,
        header/footer) to return the readable body text. Use this when you
        have a specific page to read — pair with ``web_search`` for discovery.

        Args:
            url: The page URL to fetch.

        Returns:
            ``{"url": str, "content": str}`` on success,
            ``{"error": str, "url": str}`` on failure.
        """
        try:
            response = httpx.get(url, headers=_FETCH_HEADERS, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
        except Exception as e:
            return {"error": str(e), "url": url}

        try:
            tree = lxml.html.fromstring(response.text)
            for element in tree.iter(*_STRIP_TAGS):
                element.drop_tree()
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", tree.text_content()).strip()
            return {"url": str(response.url), "content": text}
        except Exception as e:
            return {"error": str(e), "url": url}
