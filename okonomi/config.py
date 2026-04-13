"""Configuration management for Okonomi."""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv


@dataclass
class AgentConfig:
    model_id: str = "anthropic/claude-sonnet-4-5"
    vision_model_id: str = "anthropic/claude-sonnet-4-5"
    temperature: float = 0.2
    top_p: float = 1.0
    max_steps: int = 50
    max_tokens: int = 4096


@dataclass
class FalConfig:
    text_to_image_model_id: str = "fal-ai/flux-2-pro"
    edit_image_model_id: str = "fal-ai/flux-2-pro/edit"
    image_to_video_model_id: str = "fal-ai/veo3.1/fast/image-to-video"
    first_last_frame_to_video_model_id: str = "fal-ai/veo3.1/fast/first-last-frame-to-video"


@dataclass
class McpConfig:
    timeout: int = 600


@dataclass
class ToolsConfig:
    mcp: McpConfig = field(default_factory=McpConfig)


@dataclass
class SessionConfig:
    storage_dir: str = ""  # Empty string means use default ~/.okonomi/sessions
    auto_save: bool = True


@dataclass
class ContextConfig:
    sessions: SessionConfig = field(default_factory=SessionConfig)
    context_length: int = 200_000
    compression_threshold: float = 0.85
    prune_threshold: float = 0.65


@dataclass
class EnvironmentVariablesConfig:
    required: List[str] = field(default_factory=lambda: ["OPENROUTER_API_KEY", "HF_TOKEN", "FAL_KEY"])


@dataclass
class TimeoutConfig:
    """Timeout settings for various operations."""

    api_request: float = 30.0
    image_generation: float = 120.0
    video_generation: float = 300.0
    code_execution: float = 60.0


@dataclass
class DefaultConfig:
    """Main configuration class for Okonomi."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    fal: FalConfig = field(default_factory=FalConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    environment_variables: EnvironmentVariablesConfig = field(default_factory=EnvironmentVariablesConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)

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
    def fal(self) -> FalConfig:
        """Get Fal configuration."""
        return self._config.fal

    @property
    def tools(self) -> ToolsConfig:
        """Get tools configuration."""
        return self._config.tools

    @property
    def context(self) -> ContextConfig:
        """Get context configuration."""
        return self._config.context

    @property
    def sessions(self) -> SessionConfig:
        """Get session configuration."""
        return self._config.context.sessions

    @property
    def environment_variables(self) -> EnvironmentVariablesConfig:
        """Get environment variables configuration."""
        return self._config.environment_variables

    @property
    def timeouts(self) -> TimeoutConfig:
        """Get timeout configuration."""
        return self._config.timeouts


# Global configuration instance
config = OkonomiConfig()
