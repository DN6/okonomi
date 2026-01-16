import os
from typing import Optional

from huggingface_hub import InferenceClient
from tavily import TavilyClient

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.utils import get_loras_catalog, update_loras_catalog


class ResearchTools(ToolMetadataMixin):
    __namespace__ = "research"
    __description__ = "Tools for web research, context gathering, and LoRA discovery"

    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.hf_client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            self.tavily_client = TavilyClient(api_key=api_key)
        else:
            self.tavily_client = None

    def search_web(
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
        if not self.tavily_client:
            return {"error": "TAVILY_API_KEY not configured", "query": query}

        try:
            response = self.tavily_client.search(
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

    def get_search_context(
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
        if not self.tavily_client:
            return {"error": "TAVILY_API_KEY not configured", "query": query}

        try:
            response = self.tavily_client.get_search_context(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )

            return {"query": query, "context": response}

        except Exception as e:
            return {"error": str(e), "query": query}

    def qna_search(self, query: str, search_depth: str = "advanced") -> dict:
        """Get a direct answer to a question using Tavily's QNA search.

        Args:
            query: Question to get answered.
            search_depth: Search depth - 'basic' or 'advanced' (default: 'advanced').

        Returns:
            Dict with the answer to the question.
        """
        if not self.tavily_client:
            return {"error": "TAVILY_API_KEY not configured", "query": query}

        try:
            response = self.tavily_client.qna_search(
                query=query,
                search_depth=search_depth,
            )

            return {"query": query, "answer": response}

        except Exception as e:
            return {"error": str(e), "query": query}

    def find_loras(self, query: str, max_results: int = 3) -> dict:
        """Find relevant LoRA models for a given style, effect, or use case.

        Searches through the available FLUX LoRA catalog and uses an LLM to find
        the most relevant LoRAs based on your description. Returns LoRA URLs that
        can be passed directly to image generation tools.

        Args:
            query: Description of the style, effect, or aesthetic you're looking for.
                   Examples: "vintage film photography", "anime style", "neon cyberpunk",
                   "oil painting texture", "pixel art retro game"
            max_results: Maximum number of LoRAs to return (default: 3).

        Returns:
            Dict with list of matching LoRAs, each containing:
            - name: LoRA model name
            - url: Full URL to the .safetensors file (use this in image generation)
            - trigger: Trigger words to include in your prompt
            - description: What the LoRA does
            - scale: Recommended scale value (0.8-1.2)

        Usage:
            1. Call find_loras with a description of the style you want
            2. Use the returned URLs and triggers in your image generation prompt
            3. Pass loras as: [(url, scale), ...] to the image generation tool
        """
        try:
            # Get the LoRA catalog
            catalog = get_loras_catalog()

            system_prompt = """You are a LoRA recommendation expert for FLUX image generation.

Given a user's query and a catalog of available LoRAs, find the most relevant LoRAs.

For each recommended LoRA, extract and return:
1. name: The model name (from ### heading)
2. url: The full URL (from **URL** field)
3. trigger: The trigger words (from **Description**, look for "Trigger Word:" or similar)
4. description: A brief summary of what the LoRA does
5. scale: Recommended scale (default 0.9 if not specified)

Return your response as a JSON array of objects. Only include LoRAs that are relevant to the query.
If no LoRAs match, return an empty array.

Example response format:
```json
[
  {
    "name": "flux-cinestill",
    "url": "https://huggingface.co/adirik/flux-cinestill/resolve/main/lora.safetensors",
    "trigger": "CNSTLL",
    "description": "Night and dusk photography with cinestill 800t film aesthetic and halation effect",
    "scale": 0.9
  }
]
```"""

            user_message = f"""Find the {max_results} most relevant LoRAs for this query: "{query}"

LORA CATALOG:
{catalog}

Return only the JSON array, no other text."""

            completion = self.hf_client.chat.completions.create(
                model=self.config.inference_providers.huggingface.prompt_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            response_text = completion.choices[0].message.content.strip()

            # Parse JSON from response
            import json

            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            loras = json.loads(response_text)

            return {"query": query, "loras": loras[:max_results]}

        except Exception as e:
            return {"error": str(e), "query": query}

    def refresh_loras_catalog(self) -> dict:
        """Refresh the LoRA catalog by fetching the latest from Hugging Face.

        Call this to update the local LoRA catalog with any new LoRAs that have
        been added to the collection.

        Returns:
            Dict with status and path to the updated catalog file.
        """
        try:
            path = update_loras_catalog()
            return {"status": "success", "path": str(path)}
        except Exception as e:
            return {"error": str(e)}
