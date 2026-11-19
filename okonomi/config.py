"""Startup configuration — filesystem layout, environment variables, the
settings file and model-provider resolution.

The model provider is auto-detected from which API key is set: exactly
one key → that provider; none → error; more than one → error asking the
user to disambiguate via ``OKONOMI_PROVIDER``.

Options live in ``$OKONOMI_HOME/settings.json``; an ``OKONOMI_*``
environment variable overrides the matching key:

    {"provider": "openrouter",
     "model": "anthropic/claude-sonnet-5",
     "subagentModels": ["a", "b"],
     "approvals": {"mode": "manual", "allow": [...], "deny": [...]},
     "mcpServers": {...}}
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv


# --- Filesystem layout, rooted at $OKONOMI_HOME (default ~/.okonomi) ---
OKONOMI_HOME = Path(os.getenv("OKONOMI_HOME", str(Path.home() / ".okonomi")))
OKONOMI_SESSIONS_DIR = OKONOMI_HOME / "sessions"
OKONOMI_MIND_DIR = OKONOMI_HOME / "mind"
OKONOMI_SKILLS_DIR = OKONOMI_HOME / "skills"
OKONOMI_FAVORITES_DIR = OKONOMI_HOME / "favorites"
OKONOMI_SETTINGS_FILE = OKONOMI_HOME / "settings.json"

_SETTINGS_TYPES = {
    "provider": str,
    "model": str,
    "subagentModels": list,
    "approvals": dict,
    "mcpServers": dict,
}


def load_settings() -> dict:
    """The parsed settings file, ``{}`` when it does not exist. Re-read on
    every call so edits apply without a restart."""
    if not OKONOMI_SETTINGS_FILE.exists():
        return {}
    data = json.loads(OKONOMI_SETTINGS_FILE.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{OKONOMI_SETTINGS_FILE} must be a JSON object, got: {data!r}")
    for key, expected in _SETTINGS_TYPES.items():
        if key in data and not isinstance(data[key], expected):
            raise ValueError(
                f"'{key}' in {OKONOMI_SETTINGS_FILE} must be a JSON {expected.__name__}, got: {data[key]!r}"
            )
    return data


def save_settings(data: dict) -> None:
    OKONOMI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OKONOMI_SETTINGS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def approval_mode() -> str:
    """``manual`` (prompt before dangerous commands, the default) or ``off``,
    from ``OKONOMI_APPROVALS`` or ``approvals.mode`` in the settings file."""
    mode = os.getenv("OKONOMI_APPROVALS") or load_settings().get("approvals", {}).get("mode") or "manual"
    mode = str(mode).strip().lower()
    if mode not in ("manual", "off"):
        raise EnvironmentError(
            f"Approval mode must be 'manual' or 'off', got {mode!r}. "
            f"Fix OKONOMI_APPROVALS or approvals.mode in {OKONOMI_SETTINGS_FILE}."
        )
    return mode


# --- Model providers ---
# Provider string -> the env var that activates it.
PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
}
PROVIDERS = tuple(PROVIDER_ENV_KEYS)

# Per-provider default model id. Override any of these with OKONOMI_MODEL
# or "model" in the settings file.
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

    ``OKONOMI_PROVIDER`` wins when set, then ``provider`` in the settings
    file. Otherwise the provider is auto-detected: exactly one key present
    → that provider; none → error; more than one → error listing the valid
    provider strings.
    """
    forced = (os.getenv("OKONOMI_PROVIDER") or load_settings().get("provider") or "").strip().lower()
    if forced:
        if forced not in PROVIDER_ENV_KEYS:
            raise EnvironmentError(
                f"Provider '{forced}' is not a known provider. Set OKONOMI_PROVIDER or "
                f"\"provider\" in {OKONOMI_SETTINGS_FILE} to one of: {', '.join(PROVIDERS)}."
            )
        key = PROVIDER_ENV_KEYS[forced]
        if not os.getenv(key):
            raise EnvironmentError(f"Provider is '{forced}' but {key} is not set.")
        return forced

    available = _providers_with_keys()
    if not available:
        keys = ", ".join(PROVIDER_ENV_KEYS.values())
        raise EnvironmentError(f"No model-provider key set. Set one of: {keys}.")
    if len(available) > 1:
        raise EnvironmentError(
            f"Multiple provider keys set ({', '.join(available)}). "
            f"Set OKONOMI_PROVIDER or \"provider\" in {OKONOMI_SETTINGS_FILE} to one of: {', '.join(PROVIDERS)}."
        )
    return available[0]


def resolve_model_id(provider: str) -> str:
    """Model id for ``provider``: ``OKONOMI_MODEL`` overrides everything,
    then ``model`` in the settings file; otherwise the provider's default
    from ``PROVIDER_DEFAULT_MODELS``."""
    override = (os.getenv("OKONOMI_MODEL") or load_settings().get("model") or "").strip()
    if override:
        return override
    default = PROVIDER_DEFAULT_MODELS.get(provider)
    if not default:
        raise EnvironmentError(f"No default model for provider '{provider}'. Set OKONOMI_MODEL to a model id.")
    return default


def subagent_models() -> list[str]:
    """Model ids a parent agent may pick for a sub-agent, from the
    comma-separated ``OKONOMI_SUBAGENT_MODELS`` or the ``subagentModels``
    list in the settings file. Empty when unset: sub-agents then always
    run on the parent's model."""
    raw = os.getenv("OKONOMI_SUBAGENT_MODELS")
    if raw:
        return [model.strip() for model in raw.split(",") if model.strip()]
    return load_settings().get("subagentModels", [])


def load_env_or_raise() -> None:
    """Load ``.env`` and resolve the provider.

    Call once at app startup. Raises ``EnvironmentError`` if no key is
    set, if multiple keys are set without a chosen provider, or if the
    chosen provider is invalid, and ``ValueError`` if the settings file
    is malformed — so the failure happens before agent construction.
    """
    load_dotenv()
    select_provider()
    env_file = os.getenv("OKONOMI_ENV_FILE")
    if env_file and not Path(env_file).expanduser().is_file():
        raise EnvironmentError(f"OKONOMI_ENV_FILE points to {env_file!r}, which is not a file. Create it or unset OKONOMI_ENV_FILE.")
    approval_mode()
