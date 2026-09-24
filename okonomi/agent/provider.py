"""Provider glue — builds the pydantic-ai model for the active provider.

Each provider uses its stock pydantic-ai model class. Provider-specific
SDKs (anthropic, huggingface_hub, openai) are imported lazily inside the
matching branch so a user only needs the SDK for the provider they use.
"""

import logging
import os

from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings


# Patch: OpenRouter returns service_tier="standard" which pydantic-ai doesn't accept.
# Widen the field to accept any string until upstream fixes it.
try:
    from pydantic_ai.models.openrouter import _OpenRouterChatCompletionChunk

    _OpenRouterChatCompletionChunk.model_fields["service_tier"].annotation = str | None  # type: ignore
    _OpenRouterChatCompletionChunk.model_rebuild()
except Exception:
    pass


logger = logging.getLogger(__name__)


def build_model(provider: str, model_id: str, timeout: float) -> Model:
    """Construct the pydantic-ai model for ``provider``.

    ``openrouter`` keeps ``openrouter_usage`` so per-request USD cost is
    returned in each response; the other providers use stock settings.
    """
    if provider == "openrouter":
        # openrouter_usage={"include": True} makes OpenRouter return
        # usage.cost (actual USD spent) per response — pulled out in
        # _accumulate_usage and persisted as actual_cost_usd.
        return OpenRouterModel(
            model_id,
            settings=OpenRouterModelSettings(
                timeout=timeout,
                openrouter_usage={"include": True},
            ),
        )

    from pydantic_ai.settings import ModelSettings

    settings = ModelSettings(timeout=timeout)

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            model_id,
            provider=AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
            settings=settings,
        )

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
            settings=settings,
        )

    if provider == "huggingface":
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            model_id,
            # Not HuggingFaceModel: huggingface_hub types a replayed tool call's
            # `function` as a function *definition*, so `arguments` is dropped on
            # serialization and strict providers 400. The router's OpenAI-compatible
            # endpoint auto-selects the inference provider, same as provider_name="auto".
            provider=OpenAIProvider(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"]),
            # The router streams cumulative usage on every chunk; without this
            # pydantic-ai sums the chunks and token counts inflate ~50-100x.
            settings=OpenAIChatModelSettings(timeout=timeout, openai_continuous_usage_stats=True),
        )

    raise ValueError(f"Unknown provider '{provider}'")
