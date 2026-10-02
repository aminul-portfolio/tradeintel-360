from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from typing import Any

from .config import resolve_provider_config
from .contracts import ProviderConfig, ProviderFailure, ProviderRequest, ProviderResponse
from .errors import ProviderFailureCode, StopCategory

MAX_RETRIES = 0
STREAM = False
THINKING = {"type": "between_tools"}
OUTPUT_CONFIG = {"effort": "low"}


def _load_anthropic_sdk() -> Any:
    return importlib.import_module("anthropic")


def _failure(category: ProviderFailureCode, provider_name: str) -> ProviderFailure:
    return ProviderFailure(category=category, provider_name=provider_name)


def _map_sdk_exception(exc: Exception, sdk: Any, provider_name: str) -> ProviderFailure:
    if isinstance(exc, getattr(sdk, "AuthenticationError", ())):
        return _failure(ProviderFailureCode.PROVIDER_AUTH_FAILED, provider_name)
    if isinstance(exc, getattr(sdk, "PermissionDeniedError", ())):
        return _failure(ProviderFailureCode.PROVIDER_AUTH_FAILED, provider_name)
    if isinstance(exc, getattr(sdk, "RateLimitError", ())):
        return _failure(ProviderFailureCode.PROVIDER_RATE_LIMITED, provider_name)
    if isinstance(exc, getattr(sdk, "APITimeoutError", ())):
        return _failure(ProviderFailureCode.PROVIDER_TIMEOUT, provider_name)
    if isinstance(exc, getattr(sdk, "APIConnectionError", ())):
        return _failure(ProviderFailureCode.PROVIDER_ERROR, provider_name)
    if isinstance(exc, getattr(sdk, "APIStatusError", ())):
        return _failure(ProviderFailureCode.PROVIDER_ERROR, provider_name)
    if isinstance(exc, getattr(sdk, "APIError", ())):
        return _failure(ProviderFailureCode.PROVIDER_ERROR, provider_name)
    return _failure(ProviderFailureCode.PROVIDER_ERROR, provider_name)


def _block_value(block: object, name: str) -> object:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _extract_raw_text(response: object) -> str | None:
    content = getattr(response, "content", None)
    if content is None:
        return None
    if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
        return None
    if len(content) != 1:
        return None
    block = content[0]
    if _block_value(block, "type") != "text":
        return None
    text = _block_value(block, "text")
    if not isinstance(text, str):
        return None
    return text


def _as_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _stop_category(stop_reason: object) -> StopCategory:
    if stop_reason == "end_turn":
        return StopCategory.COMPLETE
    if stop_reason == "max_tokens":
        return StopCategory.MAX_TOKENS
    return StopCategory.OTHER


def _reported_model(response: object) -> str | None:
    model = getattr(response, "model", None)
    return model if isinstance(model, str) else None


def _usage_tokens(response: object) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return _as_optional_int(getattr(usage, "input_tokens", None)), _as_optional_int(
        getattr(usage, "output_tokens", None)
    )


class AnthropicProvider:
    def __init__(self, client: object, config: ProviderConfig, sdk: Any | None = None) -> None:
        self._client = client
        self._config = config
        self._sdk = sdk

    def generate(self, request: ProviderRequest) -> ProviderResponse | ProviderFailure:
        provider_name = self._config.provider_name
        started = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self._config.model_id,
                max_tokens=self._config.max_output_tokens,
                messages=[{"role": "user", "content": request.prompt_text}],
                stream=STREAM,
                thinking=dict(THINKING),
                output_config=dict(OUTPUT_CONFIG),
            )
        except Exception as exc:
            sdk = self._sdk if self._sdk is not None else _load_anthropic_sdk()
            return _map_sdk_exception(exc, sdk, provider_name)
        duration_ms = int((time.monotonic() - started) * 1000)
        raw_text = _extract_raw_text(response)
        if raw_text is None:
            return _failure(ProviderFailureCode.PROVIDER_RESPONSE_INVALID, provider_name)
        input_tokens, output_tokens = _usage_tokens(response)
        return ProviderResponse(
            raw_text=raw_text,
            provider_name=provider_name,
            model_id_requested=self._config.model_id,
            model_id_reported=_reported_model(response),
            stop_category=_stop_category(getattr(response, "stop_reason", None)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )


def build_anthropic_provider() -> AnthropicProvider | ProviderFailure:
    resolved = resolve_provider_config()
    if isinstance(resolved, ProviderFailure):
        return resolved
    sdk = _load_anthropic_sdk()
    try:
        client = sdk.Anthropic(
            api_key=resolved.api_key,
            timeout=resolved.timeout_seconds,
            max_retries=MAX_RETRIES,
        )
    except Exception as exc:
        return _map_sdk_exception(exc, sdk, resolved.provider_name)
    return AnthropicProvider(client, resolved, sdk)
