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
        from pydantic_ai.models.huggingface import HuggingFaceModel
        from pydantic_ai.providers.huggingface import HuggingFaceProvider

        return HuggingFaceModel(
            model_id,
            # provider_name="auto" routes to whichever inference provider
            # serves the model (HF needs base_url/provider_name/hf_client).
            provider=HuggingFaceProvider(api_key=os.environ["HF_TOKEN"], provider_name="auto"),
            settings=settings,
        )

    raise ValueError(f"Unknown provider '{provider}'")
