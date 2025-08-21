import json
import os
import re
from textwrap import dedent

from huggingface_hub import HfApi
from mcp.server.fastmcp import FastMCP

# Import shared functions from utils and config using relative import
from ..config import config
from ..utils import get_chroma_client


os.environ["MCP_TIMEOUT"] = str(config["tools"]["mcp"]["timeout"])

mcp = FastMCP("memory_tools")
hf_api = HfApi()


@mcp.tool()
def retrieve_from_memory(query: str) -> str:
    """Retrieve semantically similar documents from ChromaDB memory.

    Args:
        query: Search query for finding similar image generation contexts.

    Returns:
        JSON list with score, evaluation, and image_url for each result.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=config["memory"]["chroma"]["image_eval_collection_id"])
    max_results = config["memory"]["chroma"]["max_image_eval_results"]

    try:
        result = collection.query(
            query_texts=[query],
            include=["documents", "metadatas"],
        )
    except Exception as e:
        raise e

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    outputs = []
    for i, (document, metadata) in enumerate(zip(documents[:max_results], metadatas[:max_results])):
        # Get the overall_score from metadata, default to 'N/A' if not present
        score = metadata.get("score", "N/A")
        outputs.append(
            {
                "score": score,
                "evaluation": dedent(document.strip()),
                "image_url": metadata.get("image_url", "").replace("/blob/", "/resolve/"),
            }
        )

    return json.dumps(outputs)


@mcp.tool()
def find_candidate_loras(query: str) -> str:
    """Search for LoRA models and extract trigger words.

    Args:
        query: Search terms (style, technique, or use-case based).

    Returns:
        JSON list with lora_id, trigger_word, and description for each match.

    Note: Include trigger words in prompts when using found LoRAs.
    """
    max_results = config["memory"]["chroma"]["max_lora_results"]
    client = get_chroma_client()

    collection = client.get_or_create_collection(name=config["memory"]["chroma"]["lora_collection_id"])

    try:
        result = collection.query(
            query_texts=[query],
            include=["documents", "metadatas"],
        )
    except Exception as e:
        raise e

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    # Process each result
    output = []
    for idx, (document, metadata) in enumerate(zip(documents[:max_results], metadatas[:max_results]), 1):
        lora = {}

        # Extract LoRA ID
        lora_id = metadata.get("lora_id", "")
        lora["lora_id"] = lora_id

        # Extract trigger word if present in the document
        trigger_match = re.search(r"Trigger [Ww]ord:.*?`([^`]+)`", document)
        trigger_word = trigger_match.group(1) if trigger_match else None
        lora["trigger_word"] = trigger_word

        # Format description (truncate if too long)
        description = document.strip()
        lora["description"] = description

        output.append(lora)

    return json.dumps(output)


if __name__ == "__main__":
    mcp.run()
