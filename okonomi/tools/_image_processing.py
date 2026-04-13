"""Tools for loading and processing images and videos."""

import tempfile
from io import BytesIO
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import fal_client
import httpx
from PIL import Image

from okonomi.tools._metadata import ToolMetadataMixin
from okonomi.utils import image_to_uri


REQUEST_TIMEOUT = 120


def _is_url(uri: str) -> bool:
    """Check if a string is a URL."""
    parsed = urlparse(uri)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


class EditingTools(ToolMetadataMixin):
    __namespace__ = "image_processing"
    __description__ = "Tools for loading, segmenting, and decomposing images and videos"

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing settings
        """
        self.config = config

    def load_images(self, uris: str | List[str]) -> dict:
        """Load images from URIs (URLs or file paths) and return them as PIL Images.

        Args:
            uris: Single URI or list of URIs (URLs or local file paths) to load.

        Returns:
            Dict with list of PIL Image objects or error message.
            Example: {"images": [<PIL.Image>, ...], "count": 3}
        """
        try:
            if isinstance(uris, str):
                uris = [uris]

            images = []
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                for uri in uris:
                    if _is_url(uri):
                        response = client.get(uri)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))

                    else:
                        image = Image.open(uri)
                    images.append(image)

            return {"images": images, "count": len(images)}

        except Exception as e:
            return {"error": str(e)}

    def get_image_layers(
        self,
        image: str,
    ) -> dict:
        """Decompose an image into semantic layers using AI.

        Splits an image into separate layers (e.g., foreground subjects, background,
        objects) that can be individually edited, recomposed, or used as references.

        Args:
            image: URI to source image (URL or data URI from local file).

        Returns:
            Dict with "layers" (list of layer image URLs), or "error" (str) on failure.
        """
        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-layered",
                arguments={"image_url": image_to_uri(image)},
                with_logs=True,
            )

            layers = []
            if "layers" in result:
                layers = [layer["url"] for layer in result["layers"]]
            elif "images" in result:
                layers = [img["url"] for img in result["images"]]

            return {"layers": layers}

        except Exception as e:
            return {"error": str(e)}

    def segment_image(
        self,
        image: str,
        prompt: str = "",
    ) -> dict:
        """Segment objects in an image using SAM 3.1.

        Detects and segments objects matching the text prompt, returning
        a preview image with the segmented mask applied.

        Args:
            image: URI to source image (URL or data URI from local file).
            prompt: Text description of the object(s) to segment. Example: "car", "person", "dog".

        Returns:
            Dict with "images" (list containing the segmented mask preview URL), or "error" (str) on failure.
        """
        try:
            kwargs = {
                "image_url": image_to_uri(image),
                "prompt": prompt,
            }

            result = fal_client.subscribe(
                "fal-ai/sam-3-1/image",
                arguments=kwargs,
            )

            return {"images": [result["image"]["url"]]}

        except Exception as e:
            return {"error": str(e)}

    def load_video_frames(
        self,
        uri: str,
        max_frames: int = 16,
        sample_rate: int = 1,
    ) -> dict:
        """Load a video from a URI (URL or file path) and extract frames as PIL Images.

        Downloads the video (if URL) and extracts frames at the specified sample rate.
        Useful for video analysis or processing individual frames.

        Args:
            uri: URL or local file path of the video to load.
            max_frames: Maximum number of frames to extract (default: 16).
            sample_rate: Extract every Nth frame (default: 1 = every frame).

        Returns:
            Dict with list of PIL Image frames or error message.
            Example: {"frames": [<PIL.Image>, ...], "count": 16, "fps": 30.0, "duration": 5.0}
        """
        try:
            import imageio.v3 as iio

            if _is_url(uri):
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    response = client.get(uri)
                    response.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name
                cleanup_temp = True
            else:
                video_path = uri
                cleanup_temp = False

            meta = iio.immeta(video_path, plugin="pyav")
            fps = meta.get("fps", 30.0)
            duration = meta.get("duration", 0.0)

            frames = []
            frame_idx = 0

            for frame in iio.imiter(video_path, plugin="pyav"):
                if len(frames) >= max_frames:
                    break

                if frame_idx % sample_rate == 0:
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)

                frame_idx += 1

            if cleanup_temp:
                Path(video_path).unlink(missing_ok=True)

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
