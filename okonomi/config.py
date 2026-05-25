"""Configuration for Okonomi.

One dataclass per concern (agent, timeouts, context, sessions), one
top-level ``OkonomiConfig`` that aggregates them. Construct once; read
attributes directly. No dict-style access, no shadow dicts, no nested
wrappers — if you want a field, it's right there on the relevant group.
"""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv


@dataclass
class AgentConfig:
    model_id: str = "anthropic/claude-sonnet-4-5"


@dataclass
class TimeoutConfig:
    chat_completion: float = 60.0
    agent_turn: float = 600.0


@dataclass
class ContextConfig:
    context_length: int = 200_000
    compression_threshold: float = 0.85
    prune_threshold: float = 0.65


@dataclass
class SessionConfig:
    storage_dir: str = ""  # empty → default ~/.okonomi/sessions


@dataclass
class OkonomiConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    required_env_vars: List[str] = field(default_factory=lambda: ["OPENROUTER_API_KEY", "HF_TOKEN", "FAL_KEY"])

    def __post_init__(self) -> None:
        load_dotenv()
        missing = [v for v in self.required_env_vars if not os.getenv(v)]
        if missing:
            raise EnvironmentError(f"Required environment variables not set: {', '.join(missing)}")


# Global instance for modules that need defaults at import time
# (session.py, cli.py). Per-agent overrides still construct their own.
config = OkonomiConfig()
