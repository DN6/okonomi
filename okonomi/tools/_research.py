import os
from typing import Optional

from tavily import TavilyClient

from okonomi.tools._metadata import ToolMetadataMixin


class ResearchTools(ToolMetadataMixin):
    __namespace__ = "research"
    __description__ = "Tools for web research and context gathering using Tavily API"

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        self.client = TavilyClient(api_key=api_key)

    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> dict:
        """Search the web using Tavily API for research and context gathering.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 5).
            search_depth: Search depth - 'basic' or 'advanced' (default: 'basic').
            include_domains: Optional list of domains to include in search.
            exclude_domains: Optional list of domains to exclude from search.

        Returns:
            Dict with search results containing title, url, content, and score for each result.
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            results = []
            for result in response.get("results", []):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                    }
                )

            return {"query": query, "results": results}

        except Exception as e:
            return {"error": str(e), "query": query}

    async def get_search_context(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> dict:
        """Get a concise context string from web search results, ideal for augmenting prompts.

        Args:
            query: Search query string.
            max_results: Maximum number of results to include (default: 5).
            search_depth: Search depth - 'basic' or 'advanced' (default: 'basic').

        Returns:
            Dict with a formatted context summary from search results.
        """
        try:
            response = self.client.get_search_context(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )

            return {"query": query, "context": response}

        except Exception as e:
            return {"error": str(e), "query": query}

    async def qna_search(self, query: str, search_depth: str = "advanced") -> dict:
        """Get a direct answer to a question using Tavily's QNA search.

        Args:
            query: Question to get answered.
            search_depth: Search depth - 'basic' or 'advanced' (default: 'advanced').

        Returns:
            Dict with the answer to the question.
        """
        try:
            response = self.client.qna_search(
                query=query,
                search_depth=search_depth,
            )

            return {"query": query, "answer": response}

        except Exception as e:
            return {"error": str(e), "query": query}
