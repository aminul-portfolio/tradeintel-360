from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from performance.ai_orchestration.canary_fixture import CANARY_CASE_ID, CANARY_QUESTION, load_canary_retrieval
from performance.ai_orchestration.grounded import GroundedGenerationResult, generate_grounded_response
from performance.ai_provider import MODEL_ID, PROVIDER_NAME
from performance.ai_provider.adapter_anthropic import build_anthropic_provider
from performance.ai_provider.contracts import ProviderFailure, ProviderRequest, ProviderResponse


class RecordingProvider:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped
        self.call_count = 0
        self.prompt_char_count: int | None = None

    def generate(self, request: ProviderRequest) -> ProviderResponse | ProviderFailure:
        if self.call_count >= 1:
            raise RuntimeError("canary recorder refuses a second provider call")
        self.call_count += 1
        self.prompt_char_count = len(request.prompt_text)
        return self._wrapped.generate(request)


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


def _raw_output_sha256(raw_candidate: object) -> str:
    if not isinstance(raw_candidate, str):
        return ""
    return hashlib.sha256(raw_candidate.encode("utf-8")).hexdigest()


def format_canary_proof(
    result: GroundedGenerationResult,
    *,
    case_id: str,
    provider_call_attempted: bool,
    transport_response_received: bool,
    prompt_char_count: int | None = None,
) -> str:
    meta = result.provider_response_meta
    failure = result.provider_failure
    validation = result.validation_result
    rejection_codes = ""
    validation_status = ""
    if validation is not None:
        validation_status = validation.status.value
        rejection_codes = ",".join(code.value for code in validation.rejection_codes)
    reported = ""
    if meta is not None and meta.model_id_reported:
        reported = meta.model_id_reported
    raw_candidate = result.raw_candidate
    raw_sha = _raw_output_sha256(raw_candidate)
    raw_count = str(len(raw_candidate)) if isinstance(raw_candidate, str) else ""
    prompt_count = "" if prompt_char_count is None else str(prompt_char_count)
    lines = (
        "CANARY_CASE_ID=" + case_id,
        "PROVIDER_NAME=" + (meta.provider_name if meta is not None else PROVIDER_NAME),
        "MODEL_ID_REQUESTED=" + (meta.model_id_requested if meta is not None else MODEL_ID),
        "MODEL_ID_REPORTED=" + reported,
        "PROVIDER_CALL_ATTEMPTED=" + ("YES" if provider_call_attempted else "NO"),
        "TRANSPORT_RESPONSE_RECEIVED=" + ("YES" if transport_response_received else "NO"),
        "OUTCOME=" + result.outcome.value,
        "PROVIDER_FAILURE_CATEGORY=" + (failure.category.value if failure is not None else ""),
        "STOP_CATEGORY=" + (meta.stop_category.value if meta is not None else ""),
        "INPUT_TOKENS=" + ("" if meta is None or meta.input_tokens is None else str(meta.input_tokens)),
        "OUTPUT_TOKENS=" + ("" if meta is None or meta.output_tokens is None else str(meta.output_tokens)),
        "DURATION_MS=" + ("" if meta is None else str(meta.duration_ms)),
        "PROMPT_SHA256=" + (result.prompt_sha256 or ""),
        "VALIDATION_STATUS=" + validation_status,
        "REJECTION_CODES=" + rejection_codes,
        "REVIEW_REQUIRED=" + str(result.review_required),
        "PROMPT_CHAR_COUNT=" + prompt_count,
        "RAW_OUTPUT_SHA256=" + raw_sha,
        "RAW_OUTPUT_CHAR_COUNT=" + raw_count,
    )
    return "\n".join(lines) + "\n"


def emit_operator_raw_candidate(stream, raw_candidate: object) -> None:
    if not isinstance(raw_candidate, str):
        return
    stream.write("RAW_OUTPUT_HUMAN_REVIEW_BEGIN\n", ending="")
    stream.write(raw_candidate, ending="")
    stream.write("\nRAW_OUTPUT_HUMAN_REVIEW_END\n", ending="")


class Command(BaseCommand):
    help = "Run the locked Sprint 8 controlled provider canary. Requires --confirm-live-canary."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--confirm-live-canary",
            action="store_true",
            dest="confirm_live_canary",
            help="Required confirmation. Refuses without this flag.",
        )

    def handle(self, *args, **options) -> None:
        if not options.get("confirm_live_canary"):
            raise CommandError("Refusing: --confirm-live-canary is required.")
        if detect_test_runner():
            raise CommandError("Refusing: test runner detected.")
        provider = build_anthropic_provider()
        if isinstance(provider, ProviderFailure):
            raise CommandError("Refusing: provider is not configured.")
        recorder = RecordingProvider(provider)
        result = generate_grounded_response(load_canary_retrieval(), CANARY_QUESTION, recorder)
        transport_received = result.provider_response_meta is not None
        self.stdout.write(
            format_canary_proof(
                result,
                case_id=CANARY_CASE_ID,
                provider_call_attempted=recorder.call_count > 0,
                transport_response_received=transport_received,
                prompt_char_count=recorder.prompt_char_count,
            )
        )
        emit_operator_raw_candidate(self.stderr, result.raw_candidate)
