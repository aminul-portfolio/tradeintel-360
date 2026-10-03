from __future__ import annotations

import os

from .contracts import ProviderConfig, ProviderFailure
from .errors import ProviderFailureCode

PROVIDER_NAME = "anthropic"
MODEL_ID = "claude-sonnet-5-5"
TIMEOUT_SECONDS = 30
MAX_OUTPUT_TOKENS = 8192
ENABLED_ENV = "TRADEINTEL_LLM_ENABLED"
API_KEY_ENV = "ANTHROPIC_API_KEY"
ENABLED_VALUE = "1"


def load_provider_config() -> ProviderConfig:
    enabled_raw = os.environ.get(ENABLED_ENV)
    api_key_raw = os.environ.get(API_KEY_ENV)
    api_key = api_key_raw if isinstance(api_key_raw, str) else ""
    return ProviderConfig(
        provider_name=PROVIDER_NAME,
        model_id=MODEL_ID,
        timeout_seconds=TIMEOUT_SECONDS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        enabled=enabled_raw == ENABLED_VALUE,
        api_key=api_key,
    )


def resolve_provider_config() -> ProviderConfig | ProviderFailure:
    config = load_provider_config()
    if not config.enabled or not config.api_key.strip():
        return ProviderFailure(
            category=ProviderFailureCode.PROVIDER_NOT_CONFIGURED,
            provider_name=PROVIDER_NAME,
        )
    return config
