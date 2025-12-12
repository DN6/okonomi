"""Configuration management for Okonomi."""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

from dotenv import load_dotenv


@dataclass
class AgentConfig:
    model_type: str = "inference_client"
    model_id: str = "moonshotai/Kimi-K2-Instruct-0905"
    model_provider: str = "together"
    temperature: float = 0.2
    top_p: float = 1.0
    planning_interval: int = 2
    max_steps: int = 50
    max_tokens: int = 1000
    verbosity_level: int = 1
    autoplan: bool = True


@dataclass
class HuggingfaceConfig:
    prompt_model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    prompt_eval_model_id: str = "openai/gpt-oss-20b"
    prompt_model_temperature: float = 0.6
    prompt_eval_model_temperature: float = 0.2
    prompt_max_tokens: int = 512
    prompt_eval_max_tokens: int = 512
    image_eval_model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    image_prompt_model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"


@dataclass
class FalConfig:
    text_to_image_model_id: str = "fal-ai/flux-lora"
    image_to_image_model_id: str = "fal-ai/flux-2-pro/edit"
    edit_image_model_id: str = "fal-ai/flux-2-pro/edit"
    multi_angle_model_id: str = "fal-ai/qwen-image-edit-plus-lora-gallery/multiple-angles"
    image_to_video_model_id: str = "fal-ai/wan/v2.2-a14b/image-to-video"
    first_last_frame_to_video_model_id: str = "fal-ai/veo3.1/fast/first-last-frame-to-video"


@dataclass
class InferenceProvidersConfig:
    huggingface: HuggingfaceConfig = field(default_factory=HuggingfaceConfig)
    fal: FalConfig = field(default_factory=FalConfig)


@dataclass
class McpConfig:
    timeout: int = 600


@dataclass
class CallbacksConfig:
    autoeval: bool = True


@dataclass
class ToolsConfig:
    mcp: McpConfig = field(default_factory=McpConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)


@dataclass
class ChromaConfig:
    storage_path: str = "~/.okonomi/chroma_db"
    image_eval_collection_id: str = "image-evaluations"
    max_image_eval_results: int = 3


@dataclass
class HuggingfaceMemoryConfig:
    image_repo_id: str = ""


@dataclass
class SessionConfig:
    storage_dir: str = ""  # Empty string means use default ~/.okonomi/sessions
    auto_save: bool = True


@dataclass
class MemoryConfig:
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    huggingface: HuggingfaceMemoryConfig = field(default_factory=HuggingfaceMemoryConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)


@dataclass
class EnvironmentVariablesConfig:
    required: List[str] = field(default_factory=lambda: ["HF_TOKEN", "FAL_KEY"])


@dataclass
class DefaultConfig:
    """Main configuration class for Okonomi."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    inference_providers: InferenceProvidersConfig = field(default_factory=InferenceProvidersConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    environment_variables: EnvironmentVariablesConfig = field(default_factory=EnvironmentVariablesConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)


class OkonomiConfig:
    """Configuration management for Okonomi.

    Uses Python dataclasses for configuration with environment variable validation.
    Provides both dictionary-style and attribute-style access.
    """

    def __init__(self):
        """Initialize configuration with default values."""
        # Load environment variables
        load_dotenv()

        # Create the default configuration directly from dataclasses
        self._config = DefaultConfig()

        # Keep a dict representation for backward compatibility with dict access
        self._data = asdict(self._config)

        # Validate environment variables
        self._validate_env_vars()

    def _validate_env_vars(self) -> None:
        """Validate required environment variables.

        Raises:
            EnvironmentError: If a required environment variable is not set
        """
        # Check required variables
        missing_vars = []
        for var in self._config.environment_variables.required:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise EnvironmentError(f"Required environment variables not set: {', '.join(missing_vars)}")

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value for runtime modifications."""
        self._data[key] = value
        # Note: This doesn't update the dataclass structure,
        # only the dict representation

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._data.get(key, default)

    def keys(self):
        """Return configuration keys."""
        return self._data.keys()

    def values(self):
        """Return configuration values."""
        return self._data.values()

    def items(self):
        """Return configuration items."""
        return self._data.items()

    # Attribute-style access properties
    @property
    def agent(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config.agent

    @property
    def inference_providers(self) -> InferenceProvidersConfig:
        """Get inference providers configuration."""
        return self._config.inference_providers

    @property
    def tools(self) -> ToolsConfig:
        """Get tools configuration."""
        return self._config.tools

    @property
    def memory(self) -> MemoryConfig:
        """Get memory configuration."""
        return self._config.memory

    @property
    def sessions(self) -> SessionConfig:
        """Get session configuration."""
        return self._config.memory.sessions

    @property
    def environment_variables(self) -> EnvironmentVariablesConfig:
        """Get environment variables configuration."""
        return self._config.environment_variables


# Global configuration instance
config = OkonomiConfig()
