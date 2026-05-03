import os
from typing import Optional

from tavily import TavilyClient

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.tools._returns import WebSearchResult
from okonomi.utils import create_chat_client


class ResearchTools(ToolMetadataMixin):
    __namespace__ = "research"
    __description__ = "Tools for web research and context gathering"

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self._client = create_chat_client()

        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            self.tavily_client = TavilyClient(api_key=api_key)
        else:
            self.tavily_client = None

    def _chat_completion(
        self, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 1024
    ) -> str:
        """Create a chat completion using the configured provider."""
        model_id = self.config.agent.model_id

        response = self._client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def web_search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "ultra-fast",
        include_images: bool = True,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> WebSearchResult:
        """Search the web using Tavily API for research and context gathering.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default: 5).
            search_depth: Search depth - 'ultra-fast' or advanced' (default: 'ultra-fast'). Use ultra-fast for initial search and advanced to get relevant details
            include_domains: Optional list of domains to include in search.
            exclude_domains: Optional list of domains to exclude from search.

        Returns:
            Dict with search results containing title, url, content, and score for each result.
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
