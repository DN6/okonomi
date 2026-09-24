"""Startup configuration — filesystem layout, environment variables, and
model-provider resolution, resolved once at import.

The model provider is auto-detected from which API key is set: exactly
one key → that provider; none → error; more than one → error asking the
user to disambiguate via ``OKONOMI_PROVIDER``.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# --- Filesystem layout, rooted at $OKONOMI_HOME (default ~/.okonomi) ---
OKONOMI_HOME = Path(os.getenv("OKONOMI_HOME", str(Path.home() / ".okonomi")))
OKONOMI_SESSIONS_DIR = OKONOMI_HOME / "sessions"
OKONOMI_MIND_DIR = OKONOMI_HOME / "mind"
OKONOMI_SKILLS_DIR = OKONOMI_HOME / "skills"
OKONOMI_FAVOURITES_DIR = OKONOMI_HOME / "favourites"
OKONOMI_LOVED_DIR = OKONOMI_HOME / "loved"


# --- Model providers ---
# Provider string -> the env var that activates it.
PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
}
PROVIDERS = tuple(PROVIDER_ENV_KEYS)

# Per-provider default model id. Override any of these with OKONOMI_MODEL.
PROVIDER_DEFAULT_MODELS = {
    "openrouter": "anthropic/claude-sonnet-5",
    "anthropic": "claude-sonnet-5",
    "openai": "openai/gpt-5.6-terra",
    "huggingface": "moonshotai/Kimi-K3",
}


def _providers_with_keys() -> list[str]:
    return [p for p, env in PROVIDER_ENV_KEYS.items() if os.getenv(env)]


def select_provider() -> str:
    """Return the active model provider.

    ``OKONOMI_PROVIDER`` wins when set. Otherwise the provider is
    auto-detected: exactly one key present → that provider; none →
    error; more than one → error listing the valid provider strings.
    """
    forced = (os.getenv("OKONOMI_PROVIDER") or "").strip().lower()
    if forced:
        if forced not in PROVIDER_ENV_KEYS:
            raise EnvironmentError(
                f"OKONOMI_PROVIDER='{forced}' is not a known provider. " f"Set it to one of: {', '.join(PROVIDERS)}."
            )
        key = PROVIDER_ENV_KEYS[forced]
        if not os.getenv(key):
            raise EnvironmentError(f"OKONOMI_PROVIDER='{forced}' but {key} is not set.")
        return forced

    available = _providers_with_keys()
    if not available:
        keys = ", ".join(PROVIDER_ENV_KEYS.values())
        raise EnvironmentError(f"No model-provider key set. Set one of: {keys}.")
    if len(available) > 1:
        raise EnvironmentError(
            f"Multiple provider keys set ({', '.join(available)}). "
            f"Set OKONOMI_PROVIDER to pick one of: {', '.join(PROVIDERS)}."
        )
    return available[0]


def resolve_model_id(provider: str) -> str:
    """Model id for ``provider``: ``OKONOMI_MODEL`` overrides everything;
    otherwise the provider's default from ``PROVIDER_DEFAULT_MODELS``."""
    override = (os.getenv("OKONOMI_MODEL") or "").strip()
    if override:
        return override
    default = PROVIDER_DEFAULT_MODELS.get(provider)
    if not default:
        raise EnvironmentError(f"No default model for provider '{provider}'. Set OKONOMI_MODEL to a model id.")
    return default


def load_env_or_raise() -> None:
    """Load ``.env`` and resolve the provider.

    Call once at app startup. Raises ``EnvironmentError`` if no key is
    set, if multiple keys are set without ``OKONOMI_PROVIDER``, or if
    ``OKONOMI_PROVIDER`` is invalid — so the failure happens before
    agent construction.
    """
    load_dotenv()
    select_provider()
