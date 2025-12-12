import base64
import os
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import chromadb
import httpx
from huggingface_hub import HfApi
from PIL import Image

from .config import config


REQUEST_TIMEOUT = 600
hf_api = HfApi()

# Default LoRA collection on Hugging Face
LORA_COLLECTION_ID = "dn6/flux-loras-66ec0a8b322dcfe976d4e152"
LORAS_FILE_PATH = Path.home() / ".okonomi" / "loras.md"


def _parse_lora_note(note: str) -> dict:
    """Parse a LoRA note to extract trigger words and description.

    Args:
        note: Raw note text from the collection item

    Returns:
        Dict with 'trigger' and 'description' keys
    """
    if not note:
        return {"trigger": None, "description": None}

    note = note.strip()
    trigger = None
    description = None

    # Common patterns for trigger words
    import re

    # Look for trigger word patterns
    trigger_patterns = [
        r"Trigger [Ww]ord[s]?:?\s*[Ii]nclude\s+(?:the\s+)?(?:word[s]?)?\s*[`\"']([^`\"']+)[`\"']",
        r"Trigger [Ww]ord[s]?:?\s*[`\"']([^`\"']+)[`\"']",
        r"[Yy]ou should use\s+[`\"']([^`\"']+)[`\"']",
        r"[Ii]nclude\s+(?:the\s+)?(?:word[s]?)?\s*[`\"']([^`\"']+)[`\"']\s+(?:in the prompt\s+)?to trigger",
    ]

    for pattern in trigger_patterns:
        match = re.search(pattern, note)
        if match:
            trigger = match.group(1).strip()
            break

    # Extract description - remove trigger word instructions and get the main description
    lines = note.split("\n")
    desc_lines = []

    for line in lines:
        line = line.strip()
        # Skip lines that are primarily about trigger words
        if re.match(r"^Trigger [Ww]ord", line, re.IGNORECASE):
            continue
        if re.match(r"^[Yy]ou should use", line):
            continue
        if line.startswith("Include") and "trigger" in line.lower():
            continue
        if line:
            desc_lines.append(line)

    if desc_lines:
        description = " ".join(desc_lines)
        # Clean up description
        description = re.sub(r"\s+", " ", description).strip()

    return {"trigger": trigger, "description": description}


def update_loras_catalog(collection_id: str = LORA_COLLECTION_ID) -> Path:
    """Fetch LoRAs from a Hugging Face collection and generate a markdown catalog.

    Args:
        collection_id: Hugging Face collection ID (default: dn6/flux-loras)

    Returns:
        Path to the generated loras.md file
    """
    collection = hf_api.get_collection(collection_id)

    # Build markdown content
    lines = [
        "# Available FLUX LoRA Models",
        "",
        f"Collection: [{collection.title}](https://huggingface.co/collections/{collection_id})",
        f"Total LoRAs: {len(collection.items)}",
        "",
        "## How to Use LoRAs",
        "",
        "Pass LoRAs as a list of tuples `[(lora_url, scale), ...]` to image generation:",
        "- **lora_url**: The full URL to the .safetensors file",
        "- **scale**: Weight of the LoRA effect (typically 0.8-1.2, sum should stay under 1.5)",
        "",
        "---",
        "",
    ]

    for item in collection.items:
        # Get safetensors file URL
        try:
            files = hf_api.list_repo_files(item.item_id)
            safetensors = [f for f in files if f.endswith(".safetensors")]
            if not safetensors:
                continue
            lora_url = f"https://huggingface.co/{item.item_id}/resolve/main/{safetensors[0]}"
        except Exception:
            continue

        # Extract model name from ID
        model_name = item.item_id.split("/")[-1]

        # Parse the note
        parsed = _parse_lora_note(item.note)

        lines.append(f"### {model_name}")
        lines.append(f"- **ID**: `{item.item_id}`")
        lines.append(f"- **URL**: `{lora_url}`")

        if parsed["trigger"]:
            lines.append(f"- **Trigger**: `{parsed['trigger']}`")

        if parsed["description"]:
            lines.append(f"- **Description**: {parsed['description']}")

        lines.append("")

    # Write to file
    LORAS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LORAS_FILE_PATH.write_text("\n".join(lines))

    return LORAS_FILE_PATH


def get_loras_catalog() -> str:
    """Get the LoRA catalog content, generating it if it doesn't exist.

    Returns:
        Content of the loras.md file
    """
    if not LORAS_FILE_PATH.exists():
        update_loras_catalog()

    return LORAS_FILE_PATH.read_text()


def get_chroma_client() -> chromadb.PersistentClient:
    """Get ChromaDB client in local persistent mode.

    Returns:
        ChromaDB PersistentClient instance storing data locally.
    """
    # Use local persistent client - stores at ~/.okonomi/chroma_db by default
    storage_path = config.memory.chroma.storage_path
    local_db_path = Path(storage_path).expanduser()
    local_db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(local_db_path))


@dataclass
class MemoryData:
    concept_id: str = ""
    image_id: str = ""
    image_url: str = ""
    creative_strategy: str = ""
    user_input: str = ""
    image: str = ""
    prompt: str = ""
    visual_critique_score: float = 0.0
    prompt_concept_score: float = 0.0
    lora_id: Optional[List[Tuple[str, float]]] = None
    recommendations: str = ""
    tool_name: str = ""

    def __post_init__(self):
        if self.lora_id is None:
            self.lora_id = []

    @classmethod
    def from_dict(cls, dict):
        field_names = set(cls.__dataclass_fields__.keys())
        kwargs = {key: value for key, value in dict.items() if key in field_names}

        return cls(**kwargs)

    @property
    def score(self) -> float:
        scores = [
            self.prompt_concept_score,
            self.visual_critique_score,
        ]
        return sum(scores) // len(scores)


def download_from_url(url: str, dir: str) -> str:
    """Download an image from URL and save it to a directory.

    Args:
        url: URL of the image to download
        dir: Directory to save the image to

    Returns:
        str: Path to the saved image file

    Raises:
        httpx.HTTPError: If download fails
        IOError: If image processing fails
    """
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB") if image.mode != "RGB" else image

            image_id = Path(url).stem
            filepath = f"{dir}/{image_id}.png"
            image.save(filepath)

            return filepath
    except httpx.HTTPError as e:
        raise httpx.HTTPError(f"Failed to download image from {url}: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to process image from {url}: {str(e)}")


def upload_image_to_dataset(filepath: str, session_id: str) -> str:
    """Upload an image to Hugging Face dataset repository.

    Args:
        filepath: Path to the image file to upload
        session_id: Session ID for organizing uploads

    Returns:
        str: URL of the uploaded image

    Raises:
        Exception: If upload fails
    """
    try:
        image_id = Path(filepath).stem
        url = hf_api.upload_file(
            path_or_fileobj=filepath,
            repo_id=config.memory.huggingface.image_repo_id,
            repo_type="dataset",
            path_in_repo=f"{session_id}/{image_id}.png",
        )
        return url
    except Exception as e:
        raise Exception(f"Failed to upload image to Hugging Face: {str(e)}")


def prepare_document_text(memory_data: MemoryData) -> str:
    lora_id_str = "\n".join(f"{item[0]} (scale: {item[1]})" for item in memory_data.lora_id)
    creative_strategy_str = memory_data.creative_strategy.strip().split("\n")[0].strip(":")

    return dedent(
        f"""
User Input:
{memory_data.user_input}
Creative Strategy:
{creative_strategy_str}
Candidate Prompt:
{memory_data.prompt}
Recommendations:
{memory_data.recommendations}
LoRA IDs:
{lora_id_str}
""".strip()
    )


def prepare_metadata(memory_data: MemoryData, hf_image_url: str) -> dict:
    return {
        "image_url": hf_image_url,
        "input_image": memory_data.image,
        "score": memory_data.score,
        "prompt_concept_score": memory_data.prompt_concept_score,
        "visual_critique_score": memory_data.visual_critique_score,
        "concept_id": memory_data.concept_id,
        "tool_name": memory_data.tool_name,
    }


def create_session_id() -> str:
    return str(uuid.uuid4())


def write_to_permanent_memory(data: List[dict]) -> dict:
    """
    Store image generation memory data in ChromaDB vector database.

    If image_repo_id is configured: downloads images, uploads to Hugging Face dataset,
    and stores metadata with HF URLs for permanent storage.

    If image_repo_id is not configured: stores metadata with original image URLs
    (no download/upload, but still enables memory/learning functionality).

    """

    # Check if image repo is configured
    image_repo_id = config.get("memory", {}).get("huggingface", {}).get("image_repo_id", "")

    session_id = create_session_id()

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=config.memory.chroma.image_eval_collection_id)

    try:
        ids = []
        documents = []
        metadata = []

        if image_repo_id:
            # Full workflow: download, upload to HF, store with HF URLs
            with tempfile.TemporaryDirectory() as dir:
                filepaths = [download_from_url(d["image_url"], dir) for d in data]

                for idx, fp in enumerate(filepaths):
                    memory_data = MemoryData.from_dict(data[idx])
                    dataset_url = upload_image_to_dataset(fp, session_id)

                    ids.append(f"{session_id}_{Path(fp).stem}")
                    metadata.append(prepare_metadata(memory_data, dataset_url))
                    documents.append(prepare_document_text(memory_data))
        else:
            for idx, item in enumerate(data):
                memory_data = MemoryData.from_dict(item)

                ids.append(f"{session_id}_{idx}")
                # Use original image URL directly
                metadata.append(prepare_metadata(memory_data, item["image_url"]))
                documents.append(prepare_document_text(memory_data))

        collection.upsert(ids=ids, documents=documents, metadatas=metadata)

        return {"status": "success"}

    except Exception as e:
        return {"status": "failed to save to memory", "error": str(e)}


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
