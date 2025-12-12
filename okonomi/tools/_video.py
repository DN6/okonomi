"""Video generation tools for Okonomi.

Provides tools for converting images to videos using AI models.
"""

import uuid
import fal_client
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.utils import image_to_uri


class VideoTools(ToolMetadataMixin):
    __namespace__ = "video"
    __description__ = "Tools for generating videos from images using AI models"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing model IDs and settings
        """
        self.config = config
        self.fal_client = fal_client

    def _validate_concept_id(self, concept_id: str) -> str:
        """Validate that concept_id is a valid UUID, generate new one if not."""
        try:
            # Try to parse as UUID
            uuid.UUID(concept_id)
            return concept_id
        except (ValueError, AttributeError):
            # If not a valid UUID, generate a new one
            return str(uuid.uuid4())

    async def image_to_video(
        self,
        prompt: str,
        image: str,
        concept_id: str,
        image_size: str = "portrait_16_9",
        strength: float = 0.5,
        seed: int = 42,
    ) -> dict:
        """Convert an image to a video with motion based on the prompt.

        This tool generates a video from a static image, animating it based on the provided
        prompt description. This does not handle instruction-based prompting - you must use
        a descriptive prompt similar to text-to-image generation.

        Args:
            prompt: Text description of desired video motion and content.
            image: URI to source image (URL or data URI from local file).
            concept_id: Unique identifier generated from the create_prompt tool.
                       This ID links the video generation to its original prompt concept.
            image_size: Output dimensions (e.g., "portrait_16_9", "landscape_16_9", "square").
            strength: Controls how much the input image is changed. Range is between 0.5-0.75.
                     Lower values stay closer to the original image.
            seed: Random seed for reproducibility.

        Returns:
            Dict with video URL and concept_id, or error message if generation fails.
            Example: {"video": "https://...", "concept_id": "..."}
        """
        try:
            # Validate concept_id
            concept_id = self._validate_concept_id(concept_id)

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

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Generating video...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.image_to_video_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            return {"video": result["video"]["url"], "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}

    async def first_last_frame_to_video(
        self,
        first_frame: str,
        last_frame: str,
        prompt: str,
        concept_id: str,
    ) -> dict:
        """Generate a video transitioning between two images using AI.

        This tool creates a video that smoothly transitions from a first frame to a last frame,
        guided by a text prompt that describes the desired motion and content. The model generates
        intermediate frames to create a coherent video sequence.

        Args:
            first_frame: URI to the first frame image (URL or data URI from local file).
            last_frame: URI to the last frame image (URL or data URI from local file).
            prompt: Text description of the desired video content and motion between frames.
                   Example: "A woman looks into the camera, breathes in, then exclaims energetically"
            concept_id: Unique identifier generated from the create_prompt tool.
                       This ID links the video generation to its original prompt concept.

        Returns:
            Dict with video URL and concept_id, or error message if generation fails.
            Example: {"video": "https://...", "concept_id": "..."}
        """
        try:
            # Validate concept_id
            concept_id = self._validate_concept_id(concept_id)

            kwargs = {
                "first_frame_url": image_to_uri(first_frame),
                "last_frame_url": image_to_uri(last_frame),
                "prompt": prompt,
            }

            console = Console()
            with Live(
                Spinner("dots", text="[#CD5C5C]Generating video from frames...[/#CD5C5C]", style="#CD5C5C"),
                console=console,
                refresh_per_second=10,
            ):
                result = self.fal_client.subscribe(
                    self.config.inference_providers.fal.first_last_frame_to_video_model_id,
                    arguments=kwargs,
                    with_logs=True,
                )

            return {"video": result["video"]["url"], "concept_id": concept_id}

        except Exception as e:
            return {"error": str(e), "concept_id": concept_id}
