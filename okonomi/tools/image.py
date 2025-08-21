import json
import os
import uuid
from typing import List, Tuple

import fal_client
from mcp.server.fastmcp import FastMCP

# Import shared config using relative import
from ..config import config


def validate_concept_id(concept_id: str) -> str:
    """Validate that concept_id is a valid UUID, generate new one if not."""
    try:
        # Try to parse as UUID
        uuid.UUID(concept_id)
        return concept_id
    except (ValueError, AttributeError):
        # If not a valid UUID, generate a new one
        return str(uuid.uuid4())


os.environ["MCP_TIMEOUT"] = str(config["tools"]["mcp"]["timeout"])

# Initialize FastMCP server
mcp = FastMCP("image_tools")


@mcp.tool()
async def text_to_image(
    prompt: str,
    concept_id: str,
    lora_id: List[Tuple[str, float]] = [],
    image_size: str = "portrait_16_9",
    seed: int = 42,
) -> str:
    """Generate images using FLUX Dev text-to-image model.

    Args:
        prompt: Text description with subject, style, composition, lighting details.
        concept_id: Unique identifier generated from the create_prompt tool.
                   This ID links the image generation to its original prompt concept.
        lora_id: List of (url, scale) tuples for LoRA models. Scale: 0.6-1.0.
            When combining multiple LoRAs ensure the sum of scales stays under 1.5
        image_size: Dimensions (square_hd, square, portrait/landscape_4_3/16_9).
        seed: Random seed for reproducibility.

    Returns:
        JSON with image URLs and concept_id.
    """
    # Validate concept_id
    concept_id = validate_concept_id(concept_id)

    kwargs = {
        "prompt": prompt,
        "seed": seed,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "image_size": image_size,
        "num_images": 1,
    }

    if lora_id:
        model_id = config["inference_providers"]["fal"]["text_to_image_lora_model_id"]
        kwargs.update({"loras": [{"path": id, "scale": scale} for id, scale in lora_id]})
    else:
        model_id = config["inference_providers"]["fal"]["text_to_image_model_id"]

    handler = await fal_client.submit_async(model_id, arguments=kwargs)
    request_id = handler.request_id
    output = await fal_client.result_async(model_id, request_id)
    result = json.dumps({"images": [o["url"] for o in output["images"]], "concept_id": concept_id})

    return result


@mcp.tool()
async def image_to_image(
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
        image: URI to source image.
        concept_id: Unique identifier generated from the create_prompt tool.
                   This ID links the image generation to its original prompt concept.
        image_size: Output dimensions.
        strength: Controls how much the input image is changed. Range is between 0.5-0.75
        seed: Random seed.

    Returns:
        JSON with refined image URLs and concept_id.
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
        "image_url": image,
    }

    handler = await fal_client.submit_async(
        config["inference_providers"]["fal"]["image_to_image_model_id"],
        arguments=kwargs,
    )
    request_id = handler.request_id
    output = await fal_client.result_async(config["inference_providers"]["fal"]["image_to_image_model_id"], request_id)
    result = json.dumps({"images": [o["url"] for o in output["images"]], "concept_id": concept_id})

    return result


@mcp.tool()
async def edit_image(
    prompt: str,
    image: str,
    concept_id: str,
    image_size: str = "portrait_16_9",
    seed: int = 42,
) -> str:
    """Tool to edit the contents of an image. Supports object changes, style transfer, and text edits.

    Key techniques:
    - Object mods: "Change car to red", "Replace hat with cap"
    - Style transfer: "Convert to watercolor", "Bauhaus art style"
    - Character consistency: "This person... now in beach setting"
    - Text edits: "Replace 'old' with 'new'"
    - Composition control: "Change background, keep person's position"

    Args:
        prompt: Edit instruction specifying changes and what to preserve.
        image: URI to source image.
        concept_id: Unique identifier generated from the create_prompt tool.
        image_size: Output dimensions.
        seed: Random seed.

    Returns:
        JSON with edited image URLs and concept_id.
    """
    # Validate concept_id
    concept_id = validate_concept_id(concept_id)

    kwargs = {
        "prompt": prompt,
        "seed": seed,
        "guidance_scale": 2.0,
        "num_inference_steps": 28,
        "image_size": image_size,
        "num_images": 1,
        "image_url": image,
    }

    handler = await fal_client.submit_async(
        config["inference_providers"]["fal"]["edit_image_model_id"], arguments=kwargs
    )
    request_id = handler.request_id
    output = await fal_client.result_async(config["inference_providers"]["fal"]["edit_image_model_id"], request_id)
    result = json.dumps({"images": [o["url"] for o in output["images"]], "concept_id": concept_id})

    return result


@mcp.tool()
async def reference_image(
    prompt: str,
    image: str,
    concept_id: str,
    image_size: str = "portrait_16_9",
    seed: int = 42,
) -> str:
    """Generate images using reference for style transfer or character consistency.

    Two main modes:
    1. Style Transfer: Reference provides artistic style for new content
       - Start with "Using this style," then describe new scene
       - Adopts colors, technique, composition from reference

    2. Character Consistency: Maintain identity across transformations
       - Use "This person..." to establish, then specify changes
       - Preserves facial features while changing scene/pose

    Args:
        prompt: New content description or character transformation.
        image: URI to style/character reference.
        concept_id: Unique identifier generated from the create_prompt tool.
                   This ID links the image generation to its original prompt concept.
        image_size: Output dimensions.
        seed: Random seed.

    Returns:
        JSON with generated image URLs and concept_id.
    """
    # Validate concept_id
    concept_id = validate_concept_id(concept_id)

    kwargs = {
        "prompt": prompt,
        "seed": seed,
        "guidance_scale": 2.5,
        "num_inference_steps": 28,
        "image_size": image_size,
        "num_images": 1,
        "image_url": image,
    }

    handler = await fal_client.submit_async(
        config["inference_providers"]["fal"]["edit_image_model_id"], arguments=kwargs
    )
    request_id = handler.request_id
    output = await fal_client.result_async(config["inference_providers"]["fal"]["edit_image_model_id"], request_id)
    result = json.dumps({"images": [o["url"] for o in output["images"]], "concept_id": concept_id})

    return result


if __name__ == "__main__":
    mcp.run()
