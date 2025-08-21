import os
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple, Union

import chromadb
import httpx
from huggingface_hub import HfApi
from PIL import Image

from .config import config


REQUEST_TIMEOUT = 600
hf_api = HfApi()


def get_chroma_client() -> Union[chromadb.CloudClient, chromadb.PersistentClient]:
    """Get ChromaDB client - cloud if API key available, otherwise local persistent.

    Returns:
        ChromaDB client instance (CloudClient or PersistentClient).
    """
    if os.getenv("CHROMA_API_KEY"):
        return chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=config["memory"]["chroma"]["tenant"],
            database=config["memory"]["chroma"]["database"],
        )
    else:
        # Use local persistent client
        local_db_path = Path.home() / ".okonomi" / "chroma_db"
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
            repo_id=config["memory"]["huggingface"]["image_repo_id"],
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
    collection = client.get_or_create_collection(name=config["memory"]["chroma"]["image_eval_collection_id"])

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


def update_available_loras() -> None:
    """
    Retrieves all available LoRA (Low-Rank Adaptation) models from a Hugging Face collection
    and updates the ChromaDB collection with them.

    This function fetches LoRA models from a predefined Hugging Face collection, extracts
    their IDs and descriptions from their model cards, and stores them in ChromaDB for
    similarity search. When parsing the description pay close attention to the specific
    trigger word needed in the prompt to apply the LoRA.

    If lora_collection_id is not configured (empty string), this function will be a no-op.

    Notes:
        - Requires access to the Hugging Face API and the predefined LORA_COLLECTION_ID
        - The description field often contains important trigger words needed to activate the LoRA
        - The LoRA URLs point directly to .safetensors files which can be used with compatible
          image generation tools
        - Pay attention to the description field as it may contain usage instructions and examples
    """
    # Check if LoRA collection is configured
    lora_collection_id = config.get("memory", {}).get("huggingface", {}).get("lora_collection_id", "")
    if not lora_collection_id:
        print("Note: Skipping LoRA collection update - lora_collection_id not configured in config.json")
        return

    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=config["memory"]["chroma"]["lora_collection_id"])
        lora_collection = hf_api.get_collection(lora_collection_id)

        ids = []
        documents = []
        metadata = []
        for idx, lora in enumerate(lora_collection.items):
            lora_filename = [file for file in hf_api.list_repo_files(lora.item_id) if file.endswith(".safetensors")][0]
            ids.append(str(idx))
            metadata.append({"lora_id": f"https://huggingface.co/{lora.item_id}/resolve/main/{lora_filename}"})
            note = lora.note
            note = note or ""
            documents.append(note)

        collection.upsert(ids=ids, documents=documents, metadatas=metadata)

    except Exception as e:
        print(f"Warning: Failed to update LoRA collection: {e}")
