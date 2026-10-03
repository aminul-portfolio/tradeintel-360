from __future__ import annotations

import os
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from performance.ai_grounding import build_evidence_packet, prompt_hash, render_evidence_summary_prompt
from performance.ai_orchestration.canary_fixture import CANARY_QUESTION, load_canary_retrieval
from performance.ai_orchestration.grounded import (
    GroundedGenerationOutcome,
    GroundedGenerationResult,
    generate_grounded_response,
)
from performance.ai_provider import MODEL_ID, PROVIDER_NAME, ProviderFailure, ProviderFailureCode
from performance.ai_provider.config import ENABLED_ENV, ENABLED_VALUE

SPRINT9_PRESET_QUESTION_ID = "historical-evidence-restatement"
SPRINT9_FIXTURE_ID = "sprint-9-synthetic-historical-evidence-001"
SPRINT9_PRESET_QUESTION = CANARY_QUESTION
QUESTION_ALLOWLIST = {SPRINT9_PRESET_QUESTION_ID: SPRINT9_PRESET_QUESTION}
APPROVAL_TTL_SECONDS = 600
SESSION_APPROVAL_KEY = "sprint9_provider_inspector_approval"
CACHE_NONCE_PREFIX = "sprint9.provider_inspector.nonce:"
TEMPLATE_NAME = "performance/provider_grounding_inspector.html"

STATE_PROVIDER_NOT_ENABLED = "PROVIDER_NOT_ENABLED"
STATE_PREVIEW_READY = "PREVIEW_READY"
STATE_CONSENT_REQUIRED = "CONSENT_REQUIRED"
STATE_APPROVAL_INVALID = "APPROVAL_INVALID"
STATE_APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
STATE_QUESTION_REFUSED = "QUESTION_REFUSED"
STATE_PROMPT_BINDING_MISMATCH = "PROMPT_BINDING_MISMATCH"
STATE_PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"
STATE_REQUEST_COMPLETED = "REQUEST_COMPLETED"
STATE_WORKFLOW_EXCEPTION = "WORKFLOW_EXCEPTION"

OUTCOME_PACKET_ERROR = GroundedGenerationOutcome.NOT_ATTEMPTED_PACKET_ERROR.value
OUTCOME_PROVIDER_FAILED = GroundedGenerationOutcome.PROVIDER_FAILED.value
OUTCOME_VALIDATION_REJECTED = GroundedGenerationOutcome.VALIDATION_REJECTED.value
OUTCOME_PASSED = GroundedGenerationOutcome.PASSED_DETERMINISTIC_CHECKS.value


@dataclass(frozen=True, slots=True)
class ProviderInspectorResultDisplay:
    outcome: str
    review_required: bool
    validation_status: str
    rejection_codes: str
    packet_error_code: str
    provider_failure_category: str
    provider_name: str
    model_id_requested: str
    model_id_reported: str
    stop_category: str
    input_tokens: str
    output_tokens: str
    duration_ms: str
    prompt_sha256_short: str
    raw_output: str
    transport_response_received: bool
    show_raw_output: bool
    show_validator: bool
    show_provider_metadata: bool


def detect_test_runner(argv: list[str] | None = None) -> bool:
    values = list(sys.argv if argv is None else argv)
    if not values:
        return False
    script = Path(values[0]).name.casefold()
    if "pytest" in script:
        return True
    for item in values:
        token = str(item)
        name = Path(token).name.casefold()
        if token == "test" or token == "pytest" or name == "pytest" or name.startswith("pytest"):
            return True
    return False


def provider_enabled() -> bool:
    return os.environ.get(ENABLED_ENV) == ENABLED_VALUE


def resolve_preset_question(question_id: object) -> str | None:
    if not isinstance(question_id, str):
        return None
    return QUESTION_ALLOWLIST.get(question_id)


def cache_nonce_key(nonce: str) -> str:
    return CACHE_NONCE_PREFIX + nonce


def consume_approval_nonce(nonce: str) -> bool:
    return cache.add(cache_nonce_key(nonce), "consumed", APPROVAL_TTL_SECONDS)


def get_provider():
    if detect_test_runner():
        return ProviderFailure(
            category=ProviderFailureCode.PROVIDER_NOT_CONFIGURED,
            provider_name=PROVIDER_NAME,
        )
    from performance.ai_provider.adapter_anthropic import build_anthropic_provider

    return build_anthropic_provider()


def build_server_owned_prompt(question_id: str) -> tuple[str, str] | None:
    question = resolve_preset_question(question_id)
    if question is None:
        return None
    retrieval = load_canary_retrieval()
    request, _context = build_evidence_packet(retrieval, question)
    rendered = render_evidence_summary_prompt(request)
    return rendered, prompt_hash(rendered)


def _short_hash(value: object) -> str:
    if isinstance(value, str) and value:
        return value[:12]
    return ""


def _optional_int_text(value: object) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else ""


def build_result_display_from_failure(
    failure: ProviderFailure,
    prompt_sha256: str,
) -> ProviderInspectorResultDisplay:
    return ProviderInspectorResultDisplay(
        outcome=OUTCOME_PROVIDER_FAILED,
        review_required=True,
        validation_status="",
        rejection_codes="",
        packet_error_code="",
        provider_failure_category=failure.category.value,
        provider_name="",
        model_id_requested="",
        model_id_reported="",
        stop_category="",
        input_tokens="",
        output_tokens="",
        duration_ms="",
        prompt_sha256_short=_short_hash(prompt_sha256),
        raw_output="",
        transport_response_received=False,
        show_raw_output=False,
        show_validator=False,
        show_provider_metadata=False,
    )


def build_result_display(result: GroundedGenerationResult) -> ProviderInspectorResultDisplay:
    outcome = result.outcome.value
    validation = result.validation_result
    failure = result.provider_failure
    meta = result.provider_response_meta
    raw = result.raw_candidate if isinstance(result.raw_candidate, str) else ""
    transport = meta is not None
    show_raw = outcome in {OUTCOME_PASSED, OUTCOME_VALIDATION_REJECTED} and bool(raw)
    show_validator = outcome in {OUTCOME_PASSED, OUTCOME_VALIDATION_REJECTED} and validation is not None
    show_meta = transport and outcome in {OUTCOME_PASSED, OUTCOME_VALIDATION_REJECTED}
    return ProviderInspectorResultDisplay(
        outcome=outcome,
        review_required=True,
        validation_status="" if validation is None else validation.status.value,
        rejection_codes=""
        if validation is None
        else ",".join(code.value for code in validation.rejection_codes),
        packet_error_code="" if result.packet_error_code is None else result.packet_error_code.value,
        provider_failure_category="" if failure is None else failure.category.value,
        provider_name="" if meta is None or not show_meta else meta.provider_name,
        model_id_requested="" if meta is None or not show_meta else meta.model_id_requested,
        model_id_reported=""
        if meta is None or not show_meta or meta.model_id_reported is None
        else meta.model_id_reported,
        stop_category="" if meta is None or not show_meta else meta.stop_category.value,
        input_tokens="" if meta is None or not show_meta else _optional_int_text(meta.input_tokens),
        output_tokens="" if meta is None or not show_meta else _optional_int_text(meta.output_tokens),
        duration_ms="" if meta is None or not show_meta else str(meta.duration_ms),
        prompt_sha256_short=_short_hash(result.prompt_sha256),
        raw_output=raw if show_raw else "",
        transport_response_received=transport,
        show_raw_output=show_raw,
        show_validator=show_validator,
        show_provider_metadata=show_meta,
    )


def _parse_issued_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timezone.is_naive(parsed):
        return timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def approval_expired(issued_at: object, *, now: datetime | None = None) -> bool:
    parsed = _parse_issued_at(issued_at)
    if parsed is None:
        return True
    current = now if now is not None else timezone.now()
    return (current - parsed).total_seconds() >= APPROVAL_TTL_SECONDS


def _base_context() -> dict[str, object]:
    return {
        "provider_name": PROVIDER_NAME,
        "model_id": MODEL_ID,
        "question_id": SPRINT9_PRESET_QUESTION_ID,
        "fixture_id": SPRINT9_FIXTURE_ID,
        "provider_enabled": False,
        "inspector_state": STATE_PROVIDER_NOT_ENABLED,
        "preview_prompt": "",
        "prompt_sha256": "",
        "prompt_sha256_short": "",
        "nonce": "",
        "outcome": "",
        "review_required": True,
        "result_display": None,
    }


def _store_approval(session, *, nonce: str, prompt_sha256: str) -> None:
    session[SESSION_APPROVAL_KEY] = {
        "nonce": nonce,
        "prompt_sha256": prompt_sha256,
        "fixture_id": SPRINT9_FIXTURE_ID,
        "question_id": SPRINT9_PRESET_QUESTION_ID,
        "issued_at": timezone.now().isoformat(),
    }
    session.modified = True


def _render_preview(request: HttpRequest, context: dict[str, object]) -> HttpResponse:
    if not provider_enabled():
        context["inspector_state"] = STATE_PROVIDER_NOT_ENABLED
        context["provider_enabled"] = False
        return render(request, TEMPLATE_NAME, context)
    built = build_server_owned_prompt(SPRINT9_PRESET_QUESTION_ID)
    if built is None:
        context["inspector_state"] = STATE_QUESTION_REFUSED
        return render(request, TEMPLATE_NAME, context)
    prompt, prompt_sha256 = built
    nonce = secrets.token_urlsafe(32)
    _store_approval(request.session, nonce=nonce, prompt_sha256=prompt_sha256)
    context.update(
        {
            "provider_enabled": True,
            "inspector_state": STATE_PREVIEW_READY,
            "preview_prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "prompt_sha256_short": prompt_sha256[:12],
            "nonce": nonce,
        }
    )
    return render(request, TEMPLATE_NAME, context)


def _refuse(request: HttpRequest, context: dict[str, object], state: str) -> HttpResponse:
    context["inspector_state"] = state
    context["provider_enabled"] = provider_enabled()
    return render(request, TEMPLATE_NAME, context)


def _render_result(
    request: HttpRequest,
    context: dict[str, object],
    display: ProviderInspectorResultDisplay,
) -> HttpResponse:
    context.update(
        {
            "provider_enabled": True,
            "inspector_state": STATE_REQUEST_COMPLETED,
            "outcome": display.outcome,
            "review_required": True,
            "result_display": display,
            "prompt_sha256_short": display.prompt_sha256_short,
        }
    )
    return render(request, TEMPLATE_NAME, context)


def _handle_post(request: HttpRequest, context: dict[str, object]) -> HttpResponse:
    if not provider_enabled():
        return _refuse(request, context, STATE_PROVIDER_NOT_ENABLED)
    if request.POST.get("confirm_send") != "1":
        return _refuse(request, context, STATE_CONSENT_REQUIRED)
    approval = request.session.pop(SESSION_APPROVAL_KEY, None)
    request.session.modified = True
    request.session.save()
    posted_nonce = request.POST.get("nonce")
    posted_question_id = request.POST.get("question_id")
    if not isinstance(approval, dict):
        return _refuse(request, context, STATE_APPROVAL_INVALID)
    if approval.get("nonce") != posted_nonce or not isinstance(posted_nonce, str) or not posted_nonce:
        return _refuse(request, context, STATE_APPROVAL_INVALID)
    if approval.get("fixture_id") != SPRINT9_FIXTURE_ID:
        return _refuse(request, context, STATE_APPROVAL_INVALID)
    if approval.get("question_id") != SPRINT9_PRESET_QUESTION_ID:
        return _refuse(request, context, STATE_APPROVAL_INVALID)
    if posted_question_id != SPRINT9_PRESET_QUESTION_ID:
        return _refuse(request, context, STATE_QUESTION_REFUSED)
    if approval_expired(approval.get("issued_at")):
        return _refuse(request, context, STATE_APPROVAL_EXPIRED)
    if not consume_approval_nonce(posted_nonce):
        return _refuse(request, context, STATE_APPROVAL_INVALID)
    rebuilt = build_server_owned_prompt(SPRINT9_PRESET_QUESTION_ID)
    if rebuilt is None:
        return _refuse(request, context, STATE_QUESTION_REFUSED)
    _prompt, prompt_sha256 = rebuilt
    if prompt_sha256 != approval.get("prompt_sha256"):
        return _refuse(request, context, STATE_PROMPT_BINDING_MISMATCH)
    try:
        provider = get_provider()
        if isinstance(provider, ProviderFailure):
            return _render_result(
                request,
                context,
                build_result_display_from_failure(provider, prompt_sha256),
            )
        result = generate_grounded_response(
            load_canary_retrieval(),
            SPRINT9_PRESET_QUESTION,
            provider,
        )
        return _render_result(request, context, build_result_display(result))
    except Exception:
        return _refuse(request, context, STATE_WORKFLOW_EXCEPTION)


@login_required
@require_http_methods(["GET", "POST"])
def provider_grounding_inspector(request: HttpRequest) -> HttpResponse:
    if not request.user.is_staff:
        return HttpResponseForbidden("Staff access is required.")
    context = _base_context()
    if request.method == "POST":
        return _handle_post(request, context)
    return _render_preview(request, context)
