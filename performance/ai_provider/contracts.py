from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .errors import ProviderFailureCode, StopCategory


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider_name: str
    model_id: str
    timeout_seconds: int
    max_output_tokens: int
    enabled: bool
    api_key: str = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    prompt_text: str = field(repr=False)
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    raw_text: str = field(repr=False)
    provider_name: str
    model_id_requested: str
    model_id_reported: str | None
    stop_category: StopCategory
    input_tokens: int | None
    output_tokens: int | None
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    category: ProviderFailureCode
    provider_name: str


class Provider(Protocol):
    def generate(self, request: ProviderRequest) -> ProviderResponse | ProviderFailure: ...
