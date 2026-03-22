import json

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.utils import create_chat_client, image_to_uri


class PromptTools(ToolMetadataMixin):
    __namespace__ = "prompt"
    __description__ = "Tools for generating and optimizing image generation prompts and describing images"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing model IDs and settings
        """
        self.config = config
        self._client = create_chat_client()

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

    def create_prompt(self, context: str) -> dict:
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
            - prompt: The optimized text-to-image prompt

        Usage:
            1. Provide comprehensive context including the user's request and any relevant information
            2. The tool will generate an optimized prompt based on all provided information
        """
        system_prompt = """You are an expert prompt engineer for text-to-image models.
        Your task is to transform context information into highly effective image generation prompts.

        PROMPT STRUCTURE (word order matters - earlier elements are prioritized):
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
            generated_prompt = self._chat_completion(
                system_prompt,
                user_message,
                temperature=0.7,
                max_tokens=1024,
            )
            return {"prompt": str(generated_prompt)}

        except Exception as e:
            return {"error": str(e)}

    def describe_image(self, image_url: str) -> dict:
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

            completion = self._client.chat.completions.create(
                model=self.config.agent.vision_model_id,
                messages=[{"role": "system", "content": system_prompt}, user_message],
                temperature=0.3,
                max_tokens=1024,
            )

            description = completion.choices[0].message.content
            return {"description": str(description)}

        except Exception as e:
            return {"error": str(e)}
