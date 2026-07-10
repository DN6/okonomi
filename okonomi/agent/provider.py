"""Provider-layer glue for OpenRouter — the only module that should know
about provider-specific quirks (caching, response-schema patches)."""

import logging
from typing import Any

from pydantic_ai.models.openrouter import OpenRouterModel


# Patch: OpenRouter returns service_tier="standard" which pydantic-ai doesn't accept.
# Widen the field to accept any string until upstream fixes it.
try:
    from pydantic_ai.models.openrouter import _OpenRouterChatCompletionChunk

    _OpenRouterChatCompletionChunk.model_fields["service_tier"].annotation = str | None  # type: ignore
    _OpenRouterChatCompletionChunk.model_rebuild()
except Exception:
    pass


logger = logging.getLogger(__name__)


class CachedOpenRouterModel(OpenRouterModel):
    """OpenRouterModel with Anthropic prompt caching.

    Uses OpenRouter's top-level cache_control to automatically cache
    the conversation prefix for Claude models, reducing input token
    costs by ~75% on multi-turn conversations.
    """

    def __init__(self, model_name: str, enable_caching: bool = True, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self._enable_caching = enable_caching and "claude" in model_name.lower()
        if self._enable_caching:
            logger.info("Prompt caching enabled for %s", model_name)

    async def _completions_create(self, *args: Any, **kwargs: Any) -> Any:
        if self._enable_caching:
            model_settings = args[2] if len(args) > 2 else kwargs.get("model_settings", {})

            extra_body = dict(model_settings.get("extra_body", {}) or {})
            extra_body["cache_control"] = {"type": "ephemeral"}
            if isinstance(model_settings, dict):
                model_settings = {**model_settings, "extra_body": extra_body}
            else:
                model_settings["extra_body"] = extra_body
            if len(args) > 2:
                args = (*args[:2], model_settings, *args[3:])
            else:
                kwargs["model_settings"] = model_settings

        return await super()._completions_create(*args, **kwargs)
