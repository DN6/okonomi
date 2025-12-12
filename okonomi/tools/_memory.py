from textwrap import dedent

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.utils import get_chroma_client


class MemoryTools(ToolMetadataMixin):
    __namespace__ = "memory"
    __description__ = "Tools for retrieving semantic memory from past image generation sessions"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing memory and collection settings
        """
        self.config = config

    async def retrieve_from_memory(self, query: str, min_score: float = 0.0, max_score: float = 10.0) -> dict:
        """Retrieve semantically similar documents from ChromaDB memory.

        Args:
            query: Search query for finding similar image generation contexts.
            min_score: Minimum score threshold for results (default: 0.0).
            max_score: Maximum score threshold for results (default: 10.0).

        Returns:
            Dict with list of results containing score, evaluation, and image_url for each match.
        """
        try:
            client = get_chroma_client()
            collection = client.get_or_create_collection(name=self.config.memory.chroma.image_eval_collection_id)
            max_results = self.config.memory.chroma.max_image_eval_results

            result = collection.query(
                query_texts=[query],
                include=["documents", "metadatas"],
                where={"$and": [{"score": {"$gte": min_score}}, {"score": {"$lte": max_score}}]},
            )

            documents = result["documents"][0]
            metadatas = result["metadatas"][0]

            outputs = []
            for _, (document, metadata) in enumerate(zip(documents[:max_results], metadatas[:max_results])):
                # Get the overall_score from metadata, default to 'N/A' if not present
                score = metadata.get("score", "N/A")
                outputs.append(
                    {
                        "score": score,
                        "evaluation": dedent(document.strip()),
                        "image_url": metadata.get("image_url", "").replace("/blob/", "/resolve/"),
                    }
                )

            return {"results": outputs}

        except Exception as e:
            return {"error": str(e)}
