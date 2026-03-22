import base64
import os
import uuid
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


REQUEST_TIMEOUT = 600


def create_chat_client():
    from openai import OpenAI

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))


def create_session_id() -> str:
    return str(uuid.uuid4())


def image_to_uri(image: str, max_size_mb: float = 3.5) -> str:
    """Convert an image to a URI.

    Args:
        image: Either a URL string or local file path string
        max_size_mb: Maximum size in MB for data URIs (default 3.5MB to stay under 4MB limit)

    Returns:
        str: URL if input is a URL, data URI if input is a local file
    """
    # Parse the string to check if it's a valid URL
    parsed = urlparse(image)

    # Check if it has a valid scheme (http/https) and network location
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return image

    # Otherwise treat it as a local file path
    # First check file size
    file_size_mb = os.path.getsize(image) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        # Resize image to reduce file size
        with Image.open(image) as img:
            # Calculate scaling factor to get under size limit
            scale_factor = (max_size_mb / file_size_mb) ** 0.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)

            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to RGB if necessary
            if img_resized.mode != "RGB":
                img_resized = img_resized.convert("RGB")

            # Save to buffer with compression
            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
    else:
        # File is small enough, read directly
        with open(image, "rb") as f:
            img_str = base64.b64encode(f.read()).decode("utf-8")
            # Detect image format from file extension
            ext = Path(image).suffix.lower()
            mime_type = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
            return f"data:image/{mime_type};base64,{img_str}"
