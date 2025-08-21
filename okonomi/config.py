"""Configuration management for Okonomi."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


# Default configuration fallback
DEFAULT_CONFIG = {
    "agent": {
        "model_type": "inference_client",
        "model_id": "zai-org/GLM-4.5",
        "model_provider": "fireworks-ai",
        "temperature": 0.2,
        "top_p": 1.0,
        "planning_interval": 2,
        "max_steps": 50,
        "max_tokens": 5000,
        "verbosity_level": 1,
        "autoplan": True,
    },
    "inference_providers": {
        "huggingface": {
            "prompt_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt_eval_model_id": "openai/gpt-oss-20b",
            "prompt_model_temperature": 1.0,
            "prompt_eval_model_temperature": 0.2,
            "prompt_max_tokens": 1000,
            "prompt_eval_max_tokens": 1000,
            "image_eval_model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
            "image_prompt_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "fal": {
            "text_to_image_model_id": "fal-ai/flux-krea-lora",
            "text_to_image_lora_model_id": "fal-ai/flux-lora",
            "image_to_image_model_id": "fal-ai/flux-krea-lora/image-to-image",
            "edit_image_model_id": "fal-ai/flux-pro/kontext",
        },
    },
    "tools": {"mcp": {"timeout": 600}, "callbacks": {"autoeval": True}},
    "memory": {
        "chroma": {
            "tenant": "",
            "database": "",
            "image_eval_collection_id": "image-evaluations",
            "lora_collection_id": "lora-collection",
            "max_lora_results": 3,
            "max_image_eval_results": 3,
        },
        "huggingface": {"lora_collection_id": "", "image_repo_id": ""},
    },
    "environment_variables": {"required": ["HF_TOKEN", "FAL_KEY"], "optional": ["CHROMA_API_KEY"]},
}


class Config:
    """Centralized configuration management."""

    _instance = None
    _config = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Don't auto-load config on initialization - make it lazy
        pass

    def _ensure_loaded(self):
        """Ensure config is loaded before access."""
        if not self._initialized:
            self._load_config()
            self._initialized = True

    def _load_config(self) -> None:
        """Load configuration from file and validate environment variables."""
        # Load environment variables from .env file
        load_dotenv()

        # Get config file path
        config_filename = os.getenv("OKONOMI_CONFIG", "config.json")
        config_path = self._resolve_config_path(config_filename)

        # Load config file or use default
        try:
            with open(config_path, "r") as f:
                self._config = json.load(f)
        except FileNotFoundError:
            self._config = DEFAULT_CONFIG.copy()

        # Validate required environment variables
        self._validate_env_vars()

    def _resolve_config_path(self, config_filename: str) -> Path:
        """Resolve config file path based on filename."""
        if os.path.isabs(config_filename):
            # Absolute path - use as-is
            return Path(config_filename)
        else:
            # All relative paths (including simple filenames) - relative to current working directory
            # Try current working directory first
            cwd_path = Path.cwd() / config_filename
            if cwd_path.exists():
                return cwd_path

            # Fallback to package directory (installed location)
            package_path = Path(__file__).parent.parent / config_filename
            if package_path.exists():
                return package_path

            # Last resort: check if default config.json exists in package
            default_config = Path(__file__).parent.parent / "config.json"
            if default_config.exists():
                return default_config

            # If nothing exists, return cwd path (will be handled as FileNotFoundError in _load_config)
            return cwd_path

    def _validate_env_vars(self) -> None:
        """Validate required environment variables."""
        required_vars = self._config.get("environment_variables", {}).get("required", [])

        for env_variable in required_vars:
            # Make CHROMA_API_KEY optional - use local DB if not set
            if env_variable == "CHROMA_API_KEY":
                if not os.getenv(env_variable):
                    print(f"Note: {env_variable} not set, using local ChromaDB storage")
                continue

            if os.getenv(env_variable) is None:
                raise EnvironmentError(f"Required environment variable: {env_variable} is not set")

    def __getitem__(self, key):
        """Allow dictionary-style access to config."""
        self._ensure_loaded()
        return self._config[key]

    def __setitem__(self, key, value):
        """Allow dictionary-style setting (for runtime modifications)."""
        self._ensure_loaded()
        self._config[key] = value

    def __contains__(self, key):
        """Allow 'key in config' checks."""
        self._ensure_loaded()
        return key in self._config

    def get(self, key, default=None):
        """Get configuration value with optional default."""
        self._ensure_loaded()
        return self._config.get(key, default)

    def keys(self):
        """Return configuration keys."""
        self._ensure_loaded()
        return self._config.keys()

    def values(self):
        """Return configuration values."""
        self._ensure_loaded()
        return self._config.values()

    def items(self):
        """Return configuration items."""
        self._ensure_loaded()
        return self._config.items()

    def set_config_path(self, config_path: str) -> None:
        """Set a new config path and reload configuration.

        Args:
            config_path: Path to new config file
        """
        os.environ["OKONOMI_CONFIG"] = config_path
        self._config = None
        self._initialized = False
        self._ensure_loaded()

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = None
        self._initialized = False
        self._ensure_loaded()


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
