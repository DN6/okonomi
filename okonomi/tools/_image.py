"""Tools for loading and processing images and videos."""

import tempfile
from io import BytesIO
from pathlib import Path
from typing import List

import httpx
from PIL import Image

from okonomi.tools._metadata import ToolMetadataMixin


REQUEST_TIMEOUT = 120


class ImageTools(ToolMetadataMixin):
    __namespace__ = "image"
    __description__ = "Tools for loading images and videos from URLs"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing settings
        """
        self.config = config

    async def load_image_from_url(self, url: str) -> dict:
        """Load an image from a URL and return it as a PIL Image.

        Args:
            url: URL of the image to load.

        Returns:
            Dict with PIL Image object or error message.
            Example: {"image": <PIL.Image>, "width": 1920, "height": 1080, "format": "JPEG"}
        """
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(url)
                response.raise_for_status()

            image = Image.open(BytesIO(response.content))

            return {
                "image": image,
                "width": image.width,
                "height": image.height,
                "format": image.format,
            }

        except Exception as e:
            return {"error": str(e)}

    async def load_images_from_urls(self, urls: List[str]) -> dict:
        """Load multiple images from URLs and return them as PIL Images.

        Args:
            urls: List of URLs of images to load.

        Returns:
            Dict with list of PIL Image objects or error message.
            Example: {"images": [<PIL.Image>, ...], "count": 3}
        """
        try:
            images = []
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                for url in urls:
                    response = client.get(url)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    images.append(image)

            return {"images": images, "count": len(images)}

        except Exception as e:
            return {"error": str(e)}

    async def load_video_frames_from_url(
        self,
        url: str,
        max_frames: int = 16,
        sample_rate: int = 1,
    ) -> dict:
        """Load a video from URL and extract frames as PIL Images.

        Downloads the video and extracts frames at the specified sample rate.
        Useful for video analysis or processing individual frames.

        Args:
            url: URL of the video to load.
            max_frames: Maximum number of frames to extract (default: 16).
            sample_rate: Extract every Nth frame (default: 1 = every frame).

        Returns:
            Dict with list of PIL Image frames or error message.
            Example: {"frames": [<PIL.Image>, ...], "count": 16, "fps": 30.0, "duration": 5.0}
        """
        try:
            import imageio.v3 as iio

            # Download video to temp file
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.get(url)
                response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Get video metadata
            meta = iio.immeta(tmp_path, plugin="pyav")
            fps = meta.get("fps", 30.0)
            duration = meta.get("duration", 0.0)

            # Extract frames using imageio
            frames = []
            frame_idx = 0

            for frame in iio.imiter(tmp_path, plugin="pyav"):
                if len(frames) >= max_frames:
                    break

                if frame_idx % sample_rate == 0:
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)

                frame_idx += 1

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            return {
                "frames": frames,
                "count": len(frames),
                "fps": fps,
                "duration": duration,
                "total_frames": frame_idx,
            }

        except ImportError:
            return {"error": "imageio is required for video processing. Install with: pip install imageio[pyav]"}
        except Exception as e:
            return {"error": str(e)}
