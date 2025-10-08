import json
import logging
import os
import uuid

from huggingface_hub import InferenceClient
from mcp.server.fastmcp import FastMCP

# Import shared config and utils using relative import
from ..config import config
from ..utils import image_to_uri

os.environ["MCP_TIMEOUT"] = str(config.tools.mcp.timeout)

mcp = FastMCP("prompt_tools")
create_prompt_client = InferenceClient(api_key=os.environ["HF_TOKEN"])


@mcp.tool()
def create_prompt(
    context: str,
) -> str:
    """Generate an optimized text-to-image prompt.

    This tool transforms context information into highly effective image generation prompts
    based on learned information from the session.

    Args:
        context: Context information to guide prompt generation. This can include:
                - The user's request or creative brief describing what image they want
                - instructions to execute specific creative strategies or approaches to apply
                - Specific feedback about what to improve or change
                - User preferences discovered through conversation
                - Successful patterns from previous prompts
                - Style insights from analyzed images
                - Technical requirements or constraints
                - What worked or didn't work in previous attempts
                - Any previous prompts that need refinement

    Returns:
        A JSON string containing:
        - candidate_prompt: The optimized text-to-image prompt
        - concept_id: A unique identifier for this prompt concept

    Usage:
        1. Provide comprehensive context including the user's request and any relevant information
        2. The tool will generate an optimized prompt based on all provided information
    """
    system_prompt = """You are an expert prompt engineer for text-to-image models.
    Your task is to transform context information into highly effective image generation prompts.

    PROMPT GUIDELINES:

    1. ESSENTIAL ELEMENTS TO INCLUDE:
    - Subject: Detailed description of main subjects
    - Style: Artistic style, reference artists, or visual approach
    - Composition: Frame, angle, perspective, layout
    - Lighting: Type, direction, quality, mood
    - Color: Palette, dominant colors, color harmony
    - Mood: Emotional tone and atmosphere
    - Technical (if photographic): Camera specs, lens type, DOF, ISO, etc.

    2. WRITING TECHNIQUES:
    - Be specific: "Close-up portrait of middle-aged woman with curly red hair, green eyes, freckles, blue silk blouse"
    - Reference artists: "Van Gogh's Starry Night style with futuristic cityscape"
    - Add tech specs: "24mm wide-angle, f/1.8, shallow DOF, high ISO grain"
    - Blend concepts: "Last Supper composition with robots at metal table"
    - Layer scenes: Background → Middle ground → Foreground → Atmosphere
    - Be succinct. Do not create extremely long and hard to follow prompts with superfluous information

    3. AVOID:
    - Overloading with too many conflicting elements
    - Vague descriptions that lack visual clarity
    - Ignoring lighting and style specifications
    - Contradictory instructions

    4. CONTEXT ANALYSIS:
    - Extract the user's request from the provided context
    - Identify any previous prompts mentioned and analyze their strengths/weaknesses
    - Incorporate all strategies, patterns, preferences, and feedback mentioned
    - Apply any technical requirements or constraints specified

    Apply any strategies or guidance thoughtfully to enhance the user's request while maintaining their core intent.
    Generate a single, cohesive prompt that maximizes visual impact and technical precision.
    """

    user_message = f"""Create an optimized image generation prompt based on the following context:

    CONTEXT:
    {context}

    Analyze all the information provided - including the user's request, any previous prompts, feedback, preferences, and strategies.
    Transform this into a detailed, technically precise prompt that will generate stunning images.
    Output ONLY the prompt text itself, no explanations or additional formatting."""

    try:
        completion = create_prompt_client.chat.completions.create(
            model=config.inference_providers.huggingface.prompt_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=config.inference_providers.huggingface.prompt_model_temperature,
            max_tokens=config.inference_providers.huggingface.prompt_max_tokens,
        )
        generated_prompt = completion.choices[0].message.content
        return json.dumps({"prompt": str(generated_prompt), "concept_id": str(uuid.uuid4())})

    except Exception as e:
        return f"Error generating prompt: {str(e)}"


@mcp.tool()
def describe_image(image_url: str) -> str:
    """Generate a detailed description of an image from its URI.

    This tool analyzes an image and provides a comprehensive description that can be used
    for image-to-image generation, style transfer, or as reference for creating similar images.

    Args:
        image_url: URI to the image (URL or data URI from local file).
                  Example: "https://example.com/image.jpg" or local file path

    Returns:
        A detailed description of the image including subject, style, composition,
        colors, lighting, mood, and technical details

    Usage:
        Use this tool when you need to:
        - Understand what's in a reference image
        - Extract style and composition details from an image
        - Create prompts based on existing images
        - Analyze images for image-to-image workflows
    """

    system_prompt = """You are an expert at analyzing images and creating detailed descriptions for image generation.

    Analyze the provided image and create a comprehensive description that includes:

    1. SUBJECT & CONTENT:
       - Main subjects and their characteristics
       - Background elements and setting
       - Any text, logos, or symbols present

    2. COMPOSITION & FRAMING:
       - Camera angle and perspective
       - Framing and cropping
       - Depth of field and focus areas
       - Rule of thirds or other compositional techniques

    3. VISUAL STYLE:
       - Art style (photorealistic, illustration, painting, etc.)
       - Artistic influences or movements
       - Level of detail and texture

    4. COLOR & LIGHTING:
       - Color palette and dominant colors
       - Lighting direction and quality
       - Shadows and highlights
       - Time of day if applicable

    5. MOOD & ATMOSPHERE:
       - Emotional tone
       - Energy level (calm, dynamic, etc.)
       - Overall atmosphere

    6. TECHNICAL DETAILS:
       - Apparent camera settings (if photographic)
       - Post-processing effects
       - Image quality and resolution characteristics

    Provide a flowing, natural description that could be used as a prompt for recreating or creating variations of this image.
    Focus on concrete, observable details rather than interpretations.
    """

    try:
        # Use the image prompt model from config
        model_id = config.inference_providers.huggingface.image_prompt_model_id
        # Create a new client for image description (vision model)
        vision_client = InferenceClient(model=model_id, api_key=os.environ["HF_TOKEN"])

        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_uri(image_url)}},
                {
                    "type": "text",
                    "text": "Describe this image in comprehensive detail following the guidelines provided.",
                },
            ],
        }

        completion = vision_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system_prompt}, user_message],
            temperature=0.3,  # Lower temperature for more consistent descriptions
            max_tokens=config.inference_providers.huggingface.prompt_max_tokens,
        )

        description = completion.choices[0].message.content
        return str(description)

    except Exception as e:
        return f"Error describing image: {str(e)}"


if __name__ == "__main__":
    mcp.run()
