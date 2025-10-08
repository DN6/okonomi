import json
import os
import uuid


import fal_client
from mcp.server.fastmcp import FastMCP

from ..config import config
from ..utils import image_to_uri


def validate_concept_id(concept_id: str) -> str:
    """Validate that concept_id is a valid UUID, generate new one if not."""
    try:
        # Try to parse as UUID
        uuid.UUID(concept_id)
        return concept_id
    except (ValueError, AttributeError):
        # If not a valid UUID, generate a new one
        return str(uuid.uuid4())


os.environ["MCP_TIMEOUT"] = str(config.tools.mcp.timeout)

# Initialize FastMCP server
mcp = FastMCP("image_tools")


@mcp.tool()
async def image_to_video(
    prompt: str,
    image: str,
    concept_id: str,
    image_size: str = "portrait_16_9",
    strength: float = 0.5,
    seed: int = 42,
) -> str:
    """Refine existing image while preserving composition, color, and style. This does not
    handle instruction based prompting. You must use a descriptive prompt similar to text to image.

    Args:
        prompt: Text description of desired modifications.
        image: URI to source image (URL or data URI from local file).
        concept_id: Unique identifier generated from the create_prompt tool.
                   This ID links the image generation to its original prompt concept.
        image_size: Output dimensions.
        strength: Controls how much the input image is changed. Range is between 0.5-0.75
        seed: Random seed.

    Returns:
        JSON with refined image URL and concept_id.
    """
    # Validate concept_id
    concept_id = validate_concept_id(concept_id)

    kwargs = {
        "prompt": prompt,
        "seed": seed,
        "guidance_scale": 3.5,
        "strength": strength,
        "num_inference_steps": 28,
        "image_size": image_size,
        "num_images": 1,
        "image_url": image_to_uri(image),
    }

    handler = await fal_client.submit_async(
        config.inference_providers.fal.image_to_video_model_id,
        arguments=kwargs,
    )
    request_id = handler.request_id
    output = await fal_client.result_async(config.inference_providers.fal.image_to_image_model_id, request_id)
    result = json.dumps({"image": output["images"][0]["url"], "concept_id": concept_id})

    return result


if __name__ == "__main__":
    mcp.run()
