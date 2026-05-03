"""Tools for AI-powered image generation, editing, and transformation."""

import fal_client

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.tools._returns import ImageGenResult
from okonomi.utils import image_to_uri


class ImageGenerationTools(ToolMetadataMixin):
    __namespace__ = "image_generation"
    __description__ = "Tools for prompt based AI-powered image generation, editing, and transformation"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing model IDs and settings
        """
        self.fal_client = fal_client
        self.config = config

    def text_to_image(
        self,
        prompt: str,
        image_size: str = "portrait_16_9",
        seed: int = 42,
    ) -> ImageGenResult:
        """Generate images using FLUX text-to-image model.

        Args:
            prompt: Text description with subject, style, composition, lighting details.
            image_size: Dimensions (square_hd, square, portrait/landscape_4_3/16_9).
            seed: Random seed for reproducibility.

        Returns:
            Dict with "images" (list of URL strings), or "error" (str) on failure.
        """
        try:
            kwargs = {
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": 3.5,
                "num_inference_steps": 40,
                "image_size": image_size,
                "num_images": 1,
            }

            model_id = self.config.fal.text_to_image_model_id
            result = self.fal_client.subscribe(
                model_id,
                arguments=kwargs,
                with_logs=True,
            )

            return {
                "images": [img["url"] for img in result["images"]],
            }

        except Exception as e:
            return {"error": str(e)}

    def edit_image(
        self,
        prompt: str,
        image: str,
    ) -> ImageGenResult:
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
            image_size: Output dimensions.
            seed: Random seed.

        Returns:
            Dict with "images" (list of URL strings), or "error" (str) on failure.
        """
        try:
            kwargs = {"prompt": prompt, "image_urls": [image_to_uri(image)]}
            model_id = self.config.fal.edit_image_model_id
            result = self.fal_client.subscribe(
                model_id,
                arguments=kwargs,
            )

            return {
                "images": [img["url"] for img in result["images"]],
            }

        except Exception as e:
            return {"error": str(e)}

    def reference_image(
        self,
        prompt: str,
        images: str | list[str],
        image_size: str = "portrait_16_9",
        seed: int = 42,
    ) -> ImageGenResult:
        """Generate images using one or more reference images for style transfer, character consistency, or compositing.

        Multi-Reference Image Editing allows combining multiple input images to influence
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
            image_size: Output dimensions (square_hd, square, portrait/landscape_4_3/16_9).
            seed: Random seed for reproducibility.

        Returns:
            Dict with "images" (list of URL strings), or "error" (str) on failure.
        """
        try:
            # Handle single image or list of images
            if isinstance(images, str):
                image_urls = [image_to_uri(images)]
            else:
                image_urls = [image_to_uri(img) for img in images]

            kwargs = {
                "prompt": prompt,
                "seed": seed,
                "image_size": image_size,
                "num_images": 1,
                "image_urls": image_urls,
            }

            model_id = self.config.fal.edit_image_model_id
            result = self.fal_client.subscribe(
                model_id,
                arguments=kwargs,
                with_logs=True,
            )

            return {
                "images": [img["url"] for img in result["images"]],
            }

        except Exception as e:
            return {"error": str(e)}
