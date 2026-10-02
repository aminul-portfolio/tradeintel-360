from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..ai_grounding import (
    GroundingError,
    GroundingValidationResult,
    RejectionCode,
    ValidationStatus,
    build_evidence_packet,
    prompt_hash,
    render_evidence_summary_prompt,
    validate_grounded_response,
)
from ..ai_provider import (
    Provider,
    ProviderFailure,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
    StopCategory,
)
from ..rag.retrieval import RetrievalResponse

UNKNOWN_PROVIDER_NAME = "unknown"


class GroundedGenerationOutcome(str, Enum):
    NOT_ATTEMPTED_PACKET_ERROR = "NOT_ATTEMPTED_PACKET_ERROR"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"
    PASSED_DETERMINISTIC_CHECKS = "PASSED_DETERMINISTIC_CHECKS"


FROZEN_GROUNDED_GENERATION_OUTCOMES = (
    "NOT_ATTEMPTED_PACKET_ERROR",
    "PROVIDER_FAILED",
    "VALIDATION_REJECTED",
    "PASSED_DETERMINISTIC_CHECKS",
)


@dataclass(frozen=True, slots=True)
class ProviderResponseMeta:
    provider_name: str
    model_id_requested: str
    model_id_reported: str | None
    stop_category: StopCategory
    input_tokens: int | None
    output_tokens: int | None
    duration_ms: int


@dataclass(frozen=True, slots=True)
class GroundedGenerationResult:
    outcome: GroundedGenerationOutcome
    prompt_sha256: str | None
    packet_error_code: RejectionCode | None
    provider_failure: ProviderFailure | None
    provider_response_meta: ProviderResponseMeta | None
    validation_result: GroundingValidationResult | None
    raw_candidate: str | None = field(default=None, repr=False, compare=False)
    review_required: bool = field(init=False, default=True)


def _packet_error_result(code: RejectionCode) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        outcome=GroundedGenerationOutcome.NOT_ATTEMPTED_PACKET_ERROR,
        prompt_sha256=None,
        packet_error_code=code,
        provider_failure=None,
        provider_response_meta=None,
        validation_result=None,
        raw_candidate=None,
    )


def _provider_failed_result(
    *,
    prompt_sha256: str,
    provider_failure: ProviderFailure,
) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        outcome=GroundedGenerationOutcome.PROVIDER_FAILED,
        prompt_sha256=prompt_sha256,
        packet_error_code=None,
        provider_failure=provider_failure,
        provider_response_meta=None,
        validation_result=None,
        raw_candidate=None,
    )


def _unknown_provider_error(prompt_sha256: str) -> GroundedGenerationResult:
    return _provider_failed_result(
        prompt_sha256=prompt_sha256,
        provider_failure=ProviderFailure(
            category=ProviderFailureCode.PROVIDER_ERROR,
            provider_name=UNKNOWN_PROVIDER_NAME,
        ),
    )


def _meta_from_response(response: ProviderResponse) -> ProviderResponseMeta:
    return ProviderResponseMeta(
        provider_name=response.provider_name,
        model_id_requested=response.model_id_requested,
        model_id_reported=response.model_id_reported,
        stop_category=response.stop_category,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        duration_ms=response.duration_ms,
    )


def generate_grounded_response(
    retrieval: RetrievalResponse,
    question: str,
    provider: Provider,
) -> GroundedGenerationResult:
    try:
        request, context = build_evidence_packet(retrieval, question)
    except GroundingError as exc:
        return _packet_error_result(exc.code)
    try:
        rendered_prompt = render_evidence_summary_prompt(request)
    except GroundingError as exc:
        return _packet_error_result(exc.code)
    prompt_sha256 = prompt_hash(rendered_prompt)
    provider_request = ProviderRequest(prompt_text=rendered_prompt, prompt_sha256=prompt_sha256)
    try:
        generated = provider.generate(provider_request)
    except Exception:
        return _unknown_provider_error(prompt_sha256)
    if isinstance(generated, ProviderFailure):
        return _provider_failed_result(prompt_sha256=prompt_sha256, provider_failure=generated)
    if not isinstance(generated, ProviderResponse):
        return _unknown_provider_error(prompt_sha256)
    if not isinstance(generated.raw_text, str):
        return _provider_failed_result(
            prompt_sha256=prompt_sha256,
            provider_failure=ProviderFailure(
                category=ProviderFailureCode.PROVIDER_RESPONSE_INVALID,
                provider_name=generated.provider_name,
            ),
        )
    meta = _meta_from_response(generated)
    validation_result = validate_grounded_response(request, context, generated.raw_text)
    if validation_result.status == ValidationStatus.PASSED_DETERMINISTIC_CHECKS:
        outcome = GroundedGenerationOutcome.PASSED_DETERMINISTIC_CHECKS
    else:
        outcome = GroundedGenerationOutcome.VALIDATION_REJECTED
    return GroundedGenerationResult(
        outcome=outcome,
        prompt_sha256=prompt_sha256,
        packet_error_code=None,
        provider_failure=None,
        provider_response_meta=meta,
        validation_result=validation_result,
        raw_candidate=generated.raw_text,
    )
