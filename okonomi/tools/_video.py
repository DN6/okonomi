"""Video generation tools for Okonomi.

Provides tools for converting images to videos using AI models.
"""

import fal_client

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.tools._returns import VideoResult
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

    def image_to_video(
        self,
        prompt: str,
        image: str,
    ) -> VideoResult:
        """Convert an image to a video with motion based on the prompt.

        This tool generates a video from a static image, animating it based on the provided
        prompt description. This does not handle instruction-based prompting - you must use
        a descriptive prompt similar to text-to-image generation.

        IMPORTANT: The parameter is `image`, not `image_url`.

        Args:
            prompt: Text description of desired video motion and content.
                Use descriptive language, not instructions. Example: "A serene lake with
                gentle ripples, soft wind blowing through the trees" rather than
                "make the water move".
            image: URI to source image (URL or data URI from local file).
                Pass the image URL directly from a prior image generation result.

        Returns:
            Dict with video URL, or error message if generation fails.
            Example: {"video": "https://..."}

        Usage:
            result = video.image_to_video(
                prompt="A serene lake with gentle ripples, camera slowly panning right",
                image="https://fal.media/files/example.jpg",
            )
            print(result["video"])
        """
        try:
            model_id = self.config.fal.image_to_video_model_id
            kwargs = self._remap_image_to_video_kwargs(model_id, image_to_uri(image), prompt)
            result = self.fal_client.subscribe(
                model_id,
                arguments=kwargs,
            )

            return {"video": result["video"]["url"], "model_id": model_id}

        except Exception as e:
            return {"error": str(e)}

    def first_last_frame_to_video(
        self,
        first_frame: str,
        last_frame: str,
        prompt: str,
    ) -> VideoResult:
        """Generate a video transitioning between two images using AI.

        This tool creates a video that smoothly transitions from a first frame to a last frame,
        guided by a text prompt that describes the desired motion and content. The model generates
        intermediate frames to create a coherent video sequence.

        Args:
            first_frame: URI to the first frame image (URL or data URI from local file).
                Pass the image URL directly from a prior image generation result.
            last_frame: URI to the last frame image (URL or data URI from local file).
                Pass the image URL directly from a prior image generation result.
            prompt: Text description of the desired video content and motion between frames.
                Use descriptive language. Example: "A woman looks into the camera,
                breathes in, then exclaims energetically"

        Returns:
            Dict with video URL, or error message if generation fails.
            Example: {"video": "https://..."}

        Usage:
            result = video.first_last_frame_to_video(
                first_frame="https://fal.media/files/start.jpg",
                last_frame="https://fal.media/files/end.jpg",
                prompt="A flower blooming in timelapse, petals unfurling gracefully",
            )
            print(result["video"])
        """
        try:
            model_id = self.config.fal.first_last_frame_to_video_model_id
            kwargs = self._remap_frame_kwargs(model_id, image_to_uri(first_frame), image_to_uri(last_frame), prompt)
            result = self.fal_client.subscribe(
                model_id,
                arguments=kwargs,
                with_logs=True,
            )

            return {
                "video": result["video"]["url"],
                "model_id": model_id,
            }

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _remap_image_to_video_kwargs(model_id: str, image_uri: str, prompt: str) -> dict:
        if "kling" in model_id:
            return {
                "prompt": prompt,
                "start_image_url": image_uri,
                "generate_audio": False,
            }
        return {
            "prompt": prompt,
            "image_url": image_uri,
            "generate_audio": False,
        }

    @staticmethod
    def _remap_frame_kwargs(model_id: str, first_frame_uri: str, last_frame_uri: str, prompt: str) -> dict:
        if "veo3" in model_id:
            return {
                "first_frame_url": first_frame_uri,
                "end_frame_url": last_frame_uri,
                "prompt": prompt,
                "generate_audio": False,
            }
        return {
            "start_image_url": first_frame_uri,
            "end_image_url": last_frame_uri,
            "prompt": prompt,
            "generate_audio": False,
        }
