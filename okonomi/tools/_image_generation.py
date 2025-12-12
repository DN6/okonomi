"""Tools for AI-powered image generation, editing, and transformation."""

import fal_client
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from okonomi.tools._metadata import ToolMetadataMixin


class ImageGenerationTools(ToolMetadataMixin):
    __namespace__ = "image_generation"
    __description__ = "Tools for AI-powered image generation, editing, and transformation using FLUX models"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing model IDs and settings
        """
        self.fal_client = fal_client
        self.config = config

    def validate_concept_id(self, concept_id: str) -> str:
        """Validate and return the concept_id.

        Args:
            concept_id: The concept identifier to validate

        Returns:
            The validated concept_id
        """
        return concept_id

    def image_to_uri(self, image: str) -> str:
        """Convert image path/URL to URI format.

        Args:
            image: Image path or URL

        Returns:
            Formatted URI string
        """
        return image

    async def text_to_image(
        self,
        prompt: str,
        concept_id: str,
        lora_id: list[tuple[str, float]] = [],
        image_size: str = "portrait_16_9",
        seed: int = 42,
    ) -> dict:
        """Generate images using FLUX text-to-image model with optional LoRA support.

        Args:
            prompt: Text description with subject, style, composition, lighting details.
                   Include LoRA trigger words in the prompt for best results.
            concept_id: Unique identifier generated from the create_prompt tool.
                       This ID links the image generation to its original prompt concept.
            lora_id: List of (url, scale) tuples for LoRA models. Scale: 0.6-1.0.
                    When combining multiple LoRAs ensure the sum of scales stays under 1.5.
                    Use research.find_loras() to discover available LoRAs.
                    Example: [("https://huggingface.co/.../lora.safetensors", 0.9)]
            image_size: Dimensions (square_hd, square, portrait/landscape_4_3/16_9).
            seed: Random seed for reproducibility.

        Returns:
            Dict with image URL and concept_id.
        """
        try:
            concept_id = self.validate_concept_id(concept_id)

            kwargs = {
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": 3.5,
                "num_inference_steps": 28,
                "image_size": image_size,
                "num_images": 1,
            }

            # Add LoRAs if provided
            if lora_id:
                kwargs["loras"] = [{"path": url, "scale": scale} for url, scale in lora_id]

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Generating image...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.text_to_image_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            return {"image": result["images"][0]["url"], "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}

    async def image_to_image(
        self,
        prompt: str,
        image: str,
        concept_id: str,
        image_size: str = "portrait_16_9",
        strength: float = 0.5,
        seed: int = 42,
    ) -> dict:
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
            Dict with refined image URL and concept_id.
        """
        try:
            concept_id = self.validate_concept_id(concept_id)

            kwargs = {
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": 3.5,
                "strength": strength,
                "num_inference_steps": 28,
                "image_size": image_size,
                "num_images": 1,
                "image_url": self.image_to_uri(image),
            }

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Refining image...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.image_to_image_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            return {"image": result["images"][0]["url"], "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}

    async def edit_image(
        self,
        prompt: str,
        image: str,
        concept_id: str,
        image_size: str = "portrait_16_9",
        seed: int = 42,
    ) -> dict:
        """Tool to edit the contents of an image. Supports object changes, style transfer, and text edits.

        Key techniques:
        - Object mods: "Change car to red", "Replace hat with cap"
        - Style transfer: "Convert to watercolor", "Bauhaus art style"
        - Character consistency: "This person... now in beach setting"
        - Text edits: "Replace 'old' with 'new'"
        - Composition control: "Change background, keep person's position"

        Args:
            prompt: Edit instruction specifying changes and what to preserve.
            image: URI to source image (URL or data URI from local file).
            concept_id: Unique identifier generated from the create_prompt tool.
            image_size: Output dimensions.
            seed: Random seed.

        Returns:
            Dict with edited image URL and concept_id.
        """
        try:
            concept_id = self.validate_concept_id(concept_id)

            kwargs = {
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": 2.0,
                "num_inference_steps": 28,
                "image_size": image_size,
                "num_images": 1,
                "image_urls": [self.image_to_uri(image)],
            }

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Editing image...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.edit_image_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            return {"image": result["images"][0]["url"], "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}

    async def reference_image(
        self,
        prompt: str,
        images: str | list[str],
        concept_id: str,
        image_size: str = "portrait_16_9",
        seed: int = 42,
    ) -> dict:
        """Generate images using one or more reference images for style transfer, character consistency, or compositing.

        FLUX.2 Multi-Reference Image Editing allows combining multiple input images to influence
        the final output. Supports up to 8 reference images at 1MP output resolution.

        KEY USE CASES:
        1. Style Transfer: Reference provides artistic style for new content
           - Prompt: "Using the style from reference 1, create [new scene description]"
           - Adopts colors, technique, composition from reference

        2. Character Consistency: Maintain identity across variations
           - Prompt: "The person from reference 1, now [new pose/scene/clothing]"
           - Preserves facial features while changing context

        3. Fashion/Product Composites: Combine multiple elements
           - Prompt: "Model wearing the dress from reference 1, shoes from reference 2,
             in front of the background from reference 3"
           - Each reference contributes specific elements

        4. Interior Design: Place furniture and decor in rooms
           - Prompt: "Living room with the sofa from reference 1, lamp from reference 2"

        PROMPTING BEST PRACTICES:
        - Explicitly describe what element comes from which reference image
        - Specify how references should be positioned or integrated
        - Define the overall style and context desired
        - Don't assume the model will automatically combine references intuitively

        REFERENCE LIMITS (based on output resolution):
        - At 1MP output: up to 8 reference images
        - At 2MP output: up to 7 reference images
        - Total limit: 9MP for input + output combined

        Args:
            prompt: Detailed description specifying how each reference image contributes.
                   Be explicit about which elements come from which reference.
                   Example: "The woman from reference 1 wearing the red dress from reference 2,
                   standing in the garden setting from reference 3, shot on Kodak Portra 400"
            images: Single image URI or list of image URIs (URLs or data URIs from local files).
                   When using multiple references, order matters - reference them as
                   "reference 1", "reference 2", etc. in your prompt.
            concept_id: Unique identifier generated from the create_prompt tool.
            image_size: Output dimensions (square_hd, square, portrait/landscape_4_3/16_9).
            seed: Random seed for reproducibility.

        Returns:
            Dict with generated image URL and concept_id.
        """
        try:
            concept_id = self.validate_concept_id(concept_id)

            # Handle single image or list of images
            if isinstance(images, str):
                image_urls = [self.image_to_uri(images)]
            else:
                image_urls = [self.image_to_uri(img) for img in images]

            kwargs = {
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": 2.5,
                "num_inference_steps": 28,
                "image_size": image_size,
                "num_images": 1,
                "image_urls": image_urls,
            }

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Generating with reference...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.edit_image_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            return {"image": result["images"][0]["url"], "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}

    async def generate_multiple_angles(
        self,
        image: str,
        concept_id: str,
        rotate_right_left: float = 0,
        move_forward: float = 0,
        wide_angle_lens: bool = False,
        lora_scale: float = 1.25,
    ) -> dict:
        """Generate multiple viewing angles of an image using AI with camera control.

        This tool takes a single image and generates multiple variations showing the subject
        from different angles and perspectives based on camera movement parameters. Useful for
        creating product shots, character turnarounds, or exploring different viewpoints.

        Args:
            image: URI to source image (URL or data URI from local file).
            concept_id: Unique identifier generated from the create_prompt tool.
                       This ID links the image generation to its original prompt concept.
            rotate_right_left: Horizontal rotation. Positive values rotate right, negative rotate left.
                              Example: 50 = rotate right, -50 = rotate left.
            move_forward: Forward camera movement amount (e.g., 3.6 for forward movement).
            wide_angle_lens: If True, uses wide-angle lens for the shot.
            lora_scale: LoRA scale for style control (default: 1.25).

        Returns:
            Dict with list of image URLs showing different angles and concept_id, or error message if generation fails.
            Example: {"images": ["https://...", "https://...", ...], "concept_id": "..."}
        """
        try:
            concept_id = self.validate_concept_id(concept_id)

            kwargs = {
                "image_urls": [self.image_to_uri(image)],
                "rotate_right_left": rotate_right_left,
                "move_forward": move_forward,
                "wide_angle_lens": wide_angle_lens,
                "lora_scale": lora_scale,
            }

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Generating multiple angles...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.multi_angle_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            images = [img["url"] for img in result["images"]]
            return {"images": images, "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}
