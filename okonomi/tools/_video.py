"""Video generation tools for Okonomi.

Provides tools for converting images to videos using AI models.
"""

import subprocess
import tempfile
from pathlib import Path

import fal_client
import httpx

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

    def image_to_video(
        self,
        prompt: str,
        image: str,
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
            image_size: Output dimensions (e.g., "portrait_16_9", "landscape_16_9", "square").
            strength: Controls how much the input image is changed. Range is between 0.5-0.75.
                     Lower values stay closer to the original image.
            seed: Random seed for reproducibility.

        Returns:
            Dict with video URL, or error message if generation fails.
            Example: {"video": "https://..."}
        """
        try:
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

            model_id = self.config.inference_providers.fal.image_to_video_model_id
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

    def first_last_frame_to_video(
        self,
        first_frame: str,
        last_frame: str,
        prompt: str,
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

        Returns:
            Dict with video URL, or error message if generation fails.
            Example: {"video": "https://..."}
        """
        try:
            kwargs = {
                "start_image_url": image_to_uri(first_frame),
                "end_image_url": image_to_uri(last_frame),
                "prompt": prompt,
            }

            model_id = self.config.inference_providers.fal.first_last_frame_to_video_model_id
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

    def stitch_videos(
        self,
        video_urls: list[str],
        output_path: str,
    ) -> dict:
        """Stitch multiple videos together into a single video file.

        Downloads videos from URLs and concatenates them in order using ffmpeg.
        The output is saved to the specified path.

        Args:
            video_urls: List of video URLs to stitch together, in order.
            output_path: Path where the stitched video will be saved.
                        Use get_session_dir() to save to the current session folder.

        Returns:
            Dict with output path, or error message if stitching fails.
            Example: {"video_path": "/path/to/stitched_video.mp4"}
        """
        try:
            # Create temp directory for downloaded videos
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                video_files = []

                # Download all videos
                with httpx.Client(timeout=120.0) as client:
                    for i, url in enumerate(video_urls):
                        response = client.get(url, follow_redirects=True)
                        response.raise_for_status()

                        video_file = temp_path / f"video_{i:03d}.mp4"
                        with open(video_file, "wb") as f:
                            f.write(response.content)
                        video_files.append(video_file)

                # Create concat file for ffmpeg
                concat_file = temp_path / "concat.txt"
                with open(concat_file, "w") as f:
                    for video_file in video_files:
                        f.write(f"file '{video_file}'\n")

                # Ensure output directory exists
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)

                # Run ffmpeg to concatenate videos
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", str(concat_file),
                        "-c", "copy",
                        "-y",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode != 0:
                    return {"error": f"ffmpeg failed: {result.stderr}"}

                return {"video_path": str(output)}

        except Exception as e:
            return {"error": str(e)}
