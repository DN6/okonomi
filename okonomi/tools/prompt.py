import json
import os
import uuid
from typing import Optional

from huggingface_hub import InferenceClient
from mcp.server.fastmcp import FastMCP

# Import shared config using relative import
from ..config import config


os.environ["MCP_TIMEOUT"] = str(config["tools"]["mcp"]["timeout"])

mcp = FastMCP("prompt_tools")
create_prompt_client = InferenceClient(api_key=os.environ["HF_TOKEN"])


@mcp.tool()
def create_prompt(
    user_input: str,
    creative_strategy: str,
    candidate_prompt: Optional[str] = "",
    feedback: Optional[str] = "",
) -> str:
    """Generate an optimized text-to-image prompt using a creative strategy.

    This tool transforms user input into highly effective image generation prompts
    by applying creative strategies obtained from the fetch_creative_strategies tool.

    Args:
        user_input: The original user request or creative brief describing what image they want.
                   Example: "a sunset over mountains"

        creative_strategy: The detailed insturctions on how to apply a selected creative strategy.instructions.
                          DO NOT use summarized or modified strategies

        candidate_prompt: (Optional) A previous prompt that needs refinement. If provided,
                         the tool will improve upon this existing prompt rather than
                         creating a new one from scratch. Pass empty string "" if not refining.

        feedback: (Optional) Specific feedback about what to improve or change in the prompt.
                 Can be based on information obtained from past approaches or from current session
                 This helps guide refinements when a candidate_prompt is provided.
                 Example: "Add more detail about the lighting and atmosphere"
                 Pass empty string "" if no feedback.

    Returns:
        A JSON string containing:
        - candidate_prompt: The optimized text-to-image prompt
        - concept_id: A unique identifier for this prompt concept

    Usage:
        1. First call fetch_creative_strategies to obtain strategy instructions
        2. Pass the complete strategy to this tool's creative_strategy parameter
        3. The tool will apply that strategy to transform the user_input into an optimized prompt
    """
    system_prompt = f"""You are a prompt engineer who transforms conceptual ideas into visual prompts.

    YOUR TASK: Apply this creative strategy to generate a visual interpretation:
    {creative_strategy}

    PROCESS:
    1. First, apply the creative strategy to the user's input conceptually
    2. Then, translate your conceptual insights into visual elements
    3. Create a prompt that embodies the strategy's outcome visually

    PROMPT REQUIREMENTS:
    - Must visually represent the creative strategy's approach
    - Include: subject, style, composition, lighting, mood
    - Keep concise: 2-4 sentences
    - Make the strategy's influence obvious in the visual

    {
        f'''IMPROVE THIS PROMPT:
    {candidate_prompt}
    Ensure it better reflects the creative strategy's approach.
    '''
        if candidate_prompt
        else ""
    }

    {
        f'''ADDRESS FEEDBACK:
    {feedback}
    '''
        if feedback
        else ""
    }

    Output ONLY the final prompt."""

    user_message = f"""Apply the creative strategy to this input:
    {user_input}

    {f"Current prompt: {candidate_prompt}" if candidate_prompt else ""}
    {f"Feedback: {feedback}" if feedback else ""}

    Generate a prompt that visually embodies the creative strategy's approach."""

    system_prompt = f"""You are an expert prompt engineer for text-to-image models.
    Your task is to transform user input into highly effective image generation prompts using the specified creative strategy.

    CREATIVE STRATEGY TO APPLY:
    Apply the instructions outlined here to the user input when generating the prompt
    {creative_strategy}

    PROMPT GUIDELINES:

    1. ESSENTIAL ELEMENTS TO INCLUDE:
    - Subject: Detailed description of main subjects
    - Style: Artistic style, reference artists, or visual approach
    - Composition: Frame, angle, perspective, layout
    - Lighting: Type, direction, quality, mood
    - Color: Palette, dominant colors, color harmony
    - Mood: Emotional tone and atmosphere
    - Technical: Camera specs, lens type, DOF, ISO, etc.

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

    {
        '''4. WORKING WITH EXISTING PROMPT:
    You have a previously generated prompt to work from. Analyze its strengths and weaknesses.
    Consider what elements to keep, enhance, or replace while applying the creative strategy more effectively.
    The goal is to create an improved version that better captures the user's intent.
    '''
        if candidate_prompt
        else ""
    }
    {
        f'''5. FEEDBACK INCORPORATION:
    Previous feedback indicates: {feedback}
    Adjust the prompt to address these concerns while maintaining creative vision.
    '''
        if feedback
        else ""
    }
    Apply the creative strategy thoughtfully to enhance the user's request while maintaining their core intent.
    Generate a single, cohesive prompt that maximizes visual impact and technical precision.
    """

    user_message = f"""Create an optimized prompt based on:

    USER INPUT:
    {user_input}

    {
        f'''PREVIOUSLY GENERATED PROMPT:
    {candidate_prompt}

    Analyze this existing prompt and create an improved version that better applies the creative strategy and addresses any shortcomings.
    '''
        if candidate_prompt
        else ""
    }{
        f'''FEEDBACK TO INCORPORATE:
    {feedback}
    '''
        if feedback
        else ""
    }Transform this into a detailed, technically precise prompt that will generate stunning images using the creative strategy provided.
    Output ONLY the prompt text itself, no explanations or additional formatting."""

    try:
        completion = create_prompt_client.chat.completions.create(
            model=config["inference_providers"]["huggingface"]["prompt_model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=config["inference_providers"]["huggingface"]["prompt_model_temperature"],
            max_tokens=config["inference_providers"]["huggingface"]["prompt_max_tokens"],
        )
        generated_prompt = completion.choices[0].message.content
        return json.dumps({"candidate_prompt": str(generated_prompt), "concept_id": str(uuid.uuid4())})

    except Exception as e:
        return f"Error generating prompt: {str(e)}"


@mcp.tool()
def describe_image(image_url: str) -> str:
    """Generate a detailed description of an image from its URL.

    This tool analyzes an image and provides a comprehensive description that can be used
    for image-to-image generation, style transfer, or as reference for creating similar images.

    Args:
        image_url: The URL of the image to describe. Must be a publicly accessible image URL.
                  Example: "https://example.com/image.jpg"

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
        model_id = config["inference_providers"]["huggingface"]["image_prompt_model_id"]
        # Create a new client for image description (vision model)
        vision_client = InferenceClient(model=model_id, api_key=os.environ["HF_TOKEN"])

        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
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
            max_tokens=config["inference_providers"]["huggingface"]["prompt_max_tokens"],
        )

        description = completion.choices[0].message.content
        return str(description)

    except Exception as e:
        return f"Error describing image: {str(e)}"


if __name__ == "__main__":
    mcp.run()
