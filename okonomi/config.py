"""Environment / startup helpers.

The old ``OkonomiConfig`` aggregated nested dataclasses (AgentConfig,
TimeoutConfig, ContextConfig, SessionConfig) that almost no one read.
Defaults now live on the components themselves (``OkonomiAgent``,
``ContextManager``, ``Session``); this module is just the env
loader + required-var check.
"""

import os

from dotenv import load_dotenv


REQUIRED_ENV_VARS = ("OPENROUTER_API_KEY",)


def load_env_or_raise() -> None:
    """Load ``.env`` and assert required keys are present.

    Call once at app startup (CLI, tests that need real APIs).
    Raises ``EnvironmentError`` if anything's missing so the
    failure happens before agent construction, not on the first call.
    """
    load_dotenv()
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Required environment variables not set: {', '.join(missing)}")
