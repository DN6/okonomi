import json
import uuid
import os

from huggingface_hub import InferenceClient

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.utils import image_to_uri


class PromptTools(ToolMetadataMixin):
    __namespace__ = "prompt"
    __description__ = "Tools for generating and optimizing image generation prompts and describing images"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing model IDs and settings
        """
        self.config = config
        self.client = InferenceClient(api_key=os.environ["HF_TOKEN"])

    async def create_prompt(self, context: str) -> dict:
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
        system_prompt = """You are an expert prompt engineer for FLUX.2 text-to-image models.
        Your task is to transform context information into highly effective image generation prompts.

        FLUX.2 PROMPT STRUCTURE (word order matters - earlier elements are prioritized):
        Subject → Action → Style → Context

        - Subject: Main focus (person, object, character) with detailed attributes
        - Action: What the subject is doing, pose, or state
        - Style: Artistic approach, medium, aesthetic, or photographic technique
        - Context: Setting, lighting, time, mood, atmosphere

        PROMPT LENGTH GUIDELINES:
        - Short (10-30 words): Quick concepts and style exploration
        - Medium (30-80 words): Ideal for most projects
        - Long (80+ words): Complex scenes requiring detailed specifications

        PHOTOREALISTIC PROMPTS:
        Reference specific cameras and film for distinctive looks:
        - Modern Digital: "shot on Sony A7IV, clean sharp, high dynamic range"
        - 2000s Digicam: "early digital camera, slight noise, flash photography"
        - 80s Vintage: "film grain, warm color cast, soft focus"
        - Analog Film: "shot on Kodak Portra 400, natural grain"
        Specify camera models, lenses, and film stocks rather than generic terms.

        TEXT & TYPOGRAPHY:
        - Use quotation marks: "The text 'OPEN' appears in red neon letters"
        - Specify placement relative to other elements
        - Describe style: "elegant serif typography," "bold industrial lettering"
        - Use hex codes for branded text: "logo text 'ACME' in color #FF5733"

        HEX COLOR CODES:
        Associate hex codes with specific objects:
        - "The vase has color #02eb3c"
        - "The background is hex #1a1a2e"

        IMPORTANT RULES:
        - NO negative prompts - describe what you WANT, not what to avoid
        - Be specific and concrete with visual details
        - Layer scenes: Background → Middle ground → Foreground → Atmosphere
        - Keep prompts focused - avoid overloading with conflicting elements

        CONTEXT ANALYSIS:
        - Extract the user's request from the provided context
        - Identify any previous prompts and analyze their strengths/weaknesses
        - Incorporate strategies, patterns, preferences, and feedback mentioned
        - Apply any technical requirements or constraints specified

        Generate a single, cohesive prompt that maximizes visual impact and follows FLUX.2 best practices.
        """

        user_message = f"""Create an optimized image generation prompt based on the following context:

        CONTEXT:
        {context}

        Analyze all the information provided - including the user's request, any previous prompts, feedback, preferences, and strategies.
        Transform this into a detailed, technically precise prompt that will generate stunning images.
        Output ONLY the prompt text itself, no explanations or additional formatting."""

        try:
            completion = self.client.chat.completions.create(
                model=self.config.inference_providers.huggingface.prompt_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config.inference_providers.huggingface.prompt_model_temperature,
                max_tokens=self.config.inference_providers.huggingface.prompt_max_tokens,
            )
            generated_prompt = completion.choices[0].message.content
            return {"prompt": str(generated_prompt), "concept_id": str(uuid.uuid4())}

        except Exception as e:
            return {"error": str(e)}

    async def create_structured_prompt(self, context: str) -> dict:
        """Generate a JSON structured prompt for complex scenes with precise control.

        Use this instead of create_prompt when you need:
        - Complex scenes with multiple subjects and relationships
        - Precise color control with hex codes for specific elements
        - Production workflows requiring consistent structure
        - Automation and programmatic generation
        - Independent iteration on specific elements

        For simple, single-subject scenes or quick creative exploration, use create_prompt instead.

        Args:
            context: Context information to guide prompt generation. This can include:
                    - The user's request describing the complex scene they want
                    - Multiple subjects and their relationships
                    - Specific color requirements (brand colors, hex codes)
                    - Camera and composition requirements
                    - Style and mood specifications

        Returns:
            Dict containing:
            - structured_prompt: JSON object with scene components
            - concept_id: A unique identifier for this prompt concept

        JSON Schema:
            {
                "scene": "overall scene description",
                "subjects": [
                    {
                        "description": "detailed subject description",
                        "position": "where in frame (left, center, right, foreground, background)",
                        "action": "what they're doing",
                        "colors": ["#hex1", "#hex2"]
                    }
                ],
                "style": "artistic style or photographic approach",
                "color_palette": ["#hex1", "#hex2", "#hex3"],
                "lighting": "lighting description",
                "mood": "emotional tone",
                "background": "background details",
                "composition": "framing and layout",
                "camera": {
                    "angle": "camera angle (eye-level, low, high, bird's eye)",
                    "lens": "lens type (wide-angle, telephoto, 50mm, etc.)",
                    "depth_of_field": "shallow, deep, or specific f-stop"
                },
                "text": {
                    "content": "text to display",
                    "style": "typography style",
                    "color": "#hex",
                    "position": "where in frame"
                }
            }
        """
        system_prompt = """You are an expert prompt engineer for FLUX.2 text-to-image models.
        Your task is to create JSON structured prompts for complex image generation scenes.

        JSON STRUCTURED PROMPTING provides precise control over:
        - Multiple subjects with individual positioning and colors
        - Exact color matching with hex codes
        - Camera settings and composition
        - Typography and text placement

        OUTPUT FORMAT - Generate a valid JSON object with this schema:
        {
            "scene": "Brief overall scene description",
            "subjects": [
                {
                    "description": "Detailed subject description with physical attributes",
                    "position": "left/center/right/foreground/background",
                    "action": "What the subject is doing or their pose",
                    "colors": ["#hex1", "#hex2"] // Optional: specific colors for this subject
                }
            ],
            "style": "Artistic style, medium, or photographic approach",
            "color_palette": ["#hex1", "#hex2", "#hex3"], // Overall color scheme
            "lighting": "Lighting type, direction, quality, time of day",
            "mood": "Emotional tone and atmosphere",
            "background": "Background environment details",
            "composition": "Framing, layout, rule of thirds, etc.",
            "camera": {
                "angle": "eye-level/low/high/bird's eye/worm's eye",
                "lens": "wide-angle/telephoto/50mm/85mm/etc.",
                "depth_of_field": "shallow/deep/f-stop value",
                "film_stock": "Optional: Kodak Portra 400, etc."
            },
            "text": { // Optional: only include if text is needed
                "content": "The exact text to display",
                "style": "Typography style (serif, sans-serif, neon, etc.)",
                "color": "#hex",
                "position": "where the text appears"
            }
        }

        GUIDELINES:
        - Include only relevant fields - omit optional fields if not needed
        - Use hex codes for precise color control
        - Be specific with subject descriptions
        - Position subjects clearly relative to each other
        - For photorealism, specify camera/lens/film in the camera object

        IMPORTANT: Output ONLY valid JSON, no explanations or markdown formatting."""

        user_message = f"""Create a JSON structured prompt based on the following context:

        CONTEXT:
        {context}

        Analyze the request and create a structured JSON prompt that provides precise control over all elements.
        Output ONLY the JSON object, no explanations."""

        try:
            completion = self.client.chat.completions.create(
                model=self.config.inference_providers.huggingface.prompt_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config.inference_providers.huggingface.prompt_model_temperature,
                max_tokens=self.config.inference_providers.huggingface.prompt_max_tokens,
            )
            generated_content = completion.choices[0].message.content.strip()

            # Try to parse as JSON to validate
            try:
                # Remove markdown code blocks if present
                if generated_content.startswith("```"):
                    lines = generated_content.split("\n")
                    generated_content = "\n".join(lines[1:-1])
                structured_prompt = json.loads(generated_content)
            except json.JSONDecodeError:
                # Return raw content if JSON parsing fails
                structured_prompt = generated_content

            return {"structured_prompt": structured_prompt, "concept_id": str(uuid.uuid4())}

        except Exception as e:
            return {"error": str(e)}

    async def describe_image(self, image_url: str) -> dict:
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

            completion = self.client.chat.completions.create(
                model=self.config.inference_providers.huggingface.image_prompt_model_id,
                messages=[{"role": "system", "content": system_prompt}, user_message],
                temperature=0.3,  # Lower temperature for more consistent descriptions
                max_tokens=self.config.inference_providers.huggingface.prompt_max_tokens,
            )

            description = completion.choices[0].message.content
            return {"description": str(description)}

        except Exception as e:
            return {"error": str(e)}
