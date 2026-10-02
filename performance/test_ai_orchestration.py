from __future__ import annotations

import ast
import dataclasses
import json
import logging
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from .ai_grounding import (
    GroundingValidationResult,
    RejectionCode,
    ValidationStatus,
    build_evidence_packet,
    prompt_hash,
    render_evidence_summary_prompt,
    validate_grounded_response,
)
from .ai_grounding.validation import LEAKAGE_TOKENS
from .ai_orchestration import (
    FROZEN_GROUNDED_GENERATION_OUTCOMES,
    GroundedGenerationOutcome,
    GroundedGenerationResult,
    ProviderResponseMeta,
    generate_grounded_response,
)
from .ai_provider import (
    MODEL_ID,
    PROVIDER_NAME,
    ProviderFailure,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
    StopCategory,
)
from .rag.retrieval import (
    RETRIEVAL_METHOD_TFIDF_COSINE,
    RETRIEVAL_STATE_EMPTY_QUERY,
    RETRIEVAL_STATE_OK,
    RetrievalResponse,
    RetrievalResult,
)
from .rag.schema import DOCUMENT_TYPE_KPI, kpi_document_id

PACKAGE_ROOT = Path(__file__).resolve().parent
ORCH_ROOT = PACKAGE_ROOT / "ai_orchestration"
PROVIDER_ROOT = PACKAGE_ROOT / "ai_provider"
EVALUATION_ROOT = PACKAGE_ROOT / "ai_evaluation"
QUESTION = "Restate the supplied historical evidence."
FINGERPRINT = "c" * 64
EXCEPTION_SENTINEL = "SECRET-ORCH-EXC-DO-NOT-LEAK"
BLOCKED_SDK = frozenset({"anthropic", "openai"})
BLOCKED_NETWORK = frozenset(
    {
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "urllib.request",
        "socket",
        "subprocess",
        "celery",
    }
)
BLOCKED_PERSISTENCE = frozenset({"django.db", "django.db.models"})


def _iter_python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return tuple(names)


def _kpi_result() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        document_id=kpi_document_id("Total Trades"),
        document_type=DOCUMENT_TYPE_KPI,
        source_record_key="Total Trades",
        content="Total Trades: 6",
        journal_fingerprint=FINGERPRINT,
        evidence_status="AUTHORITATIVE",
        retrieval_method=RETRIEVAL_METHOD_TFIDF_COSINE,
        score=0.4,
        provenance={},
    )


def _retrieval(
    *,
    state: str = RETRIEVAL_STATE_OK,
    results: tuple[RetrievalResult, ...] | None = None,
) -> RetrievalResponse:
    packed = _kpi_result() if results is None and state == RETRIEVAL_STATE_OK else None
    items = results if results is not None else (() if packed is None else (packed,))
    return RetrievalResponse(
        state=state,
        query=QUESTION,
        results=items,
        top_k=5,
        corpus_size=1,
        candidate_count=len(items),
    )


def _valid_candidate() -> str:
    return json.dumps(
        {
            "schema_version": "grounded-ai-response-v1",
            "answer": "Total trades is 6 [E1].",
            "claims": [{"evidence": "E1", "field": "total_trades", "value": "6"}],
            "citations": ["E1"],
            "limitation_codes": [],
        },
        ensure_ascii=True,
    )


def _provider_response(raw_text: object = None) -> ProviderResponse:
    text = _valid_candidate() if raw_text is None else raw_text
    return ProviderResponse(
        raw_text=text,
        provider_name=PROVIDER_NAME,
        model_id_requested=MODEL_ID,
        model_id_reported=None,
        stop_category=StopCategory.COMPLETE,
        input_tokens=4,
        output_tokens=8,
        duration_ms=11,
    )


class _RecordingProvider:
    def __init__(self, payload: object | None = None, *, error: BaseException | None = None) -> None:
        self.calls: list[ProviderRequest] = []
        self.payload = payload if payload is not None else _provider_response()
        self.error = error

    def generate(self, request: ProviderRequest) -> object:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.payload


def _meta_fields() -> set[str]:
    return {item.name for item in dataclasses.fields(ProviderResponseMeta)}


class AiOrchestrationTests(SimpleTestCase):
    def test_outcome_enum_contains_exactly_four_values(self):
        self.assertEqual(len(FROZEN_GROUNDED_GENERATION_OUTCOMES), 4)
        self.assertEqual(
            FROZEN_GROUNDED_GENERATION_OUTCOMES,
            tuple(item.value for item in GroundedGenerationOutcome),
        )

    def test_provider_response_meta_is_immutable(self):
        meta = ProviderResponseMeta(
            provider_name=PROVIDER_NAME,
            model_id_requested=MODEL_ID,
            model_id_reported=None,
            stop_category=StopCategory.COMPLETE,
            input_tokens=1,
            output_tokens=1,
            duration_ms=1,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            meta.duration_ms = 9

    def test_grounded_generation_result_is_immutable(self):
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.outcome = GroundedGenerationOutcome.PROVIDER_FAILED

    def test_review_required_is_always_true_and_not_caller_settable(self):
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        self.assertIs(result.review_required, True)
        with self.assertRaises(TypeError):
            GroundedGenerationResult(
                outcome=GroundedGenerationOutcome.PASSED_DETERMINISTIC_CHECKS,
                prompt_sha256="a" * 64,
                packet_error_code=None,
                provider_failure=None,
                provider_response_meta=None,
                validation_result=None,
                raw_candidate=None,
                review_required=False,
            )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.review_required = False

    def test_raw_candidate_excluded_from_result_repr(self):
        candidate = _valid_candidate()
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider(_provider_response(candidate)))
        text = repr(result)
        self.assertNotIn(candidate, text)
        self.assertNotIn("raw_candidate", text)

    def test_provider_response_meta_has_no_raw_text_field(self):
        self.assertNotIn("raw_text", _meta_fields())
        self.assertNotIn("prompt_text", _meta_fields())
        self.assertNotIn("api_key", _meta_fields())

    def test_valid_packet_and_candidate_pass_deterministic_checks(self):
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        self.assertEqual(result.outcome, GroundedGenerationOutcome.PASSED_DETERMINISTIC_CHECKS)
        self.assertIsNotNone(result.validation_result)
        self.assertEqual(result.validation_result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        self.assertEqual(result.validation_result.rejection_codes, ())
        self.assertIs(result.review_required, True)

    def test_validation_result_is_retained_exactly(self):
        captured: dict[str, GroundingValidationResult] = {}
        real = validate_grounded_response

        def wrapped(request, context, candidate):
            value = real(request, context, candidate)
            captured["result"] = value
            return value

        with patch("performance.ai_orchestration.grounded.validate_grounded_response", wrapped):
            result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        self.assertIs(result.validation_result, captured["result"])

    def test_same_request_and_context_identity_reach_validator(self):
        built: dict[str, object] = {}
        seen: dict[str, object] = {}
        real_build = build_evidence_packet
        real_validate = validate_grounded_response

        def wrap_build(retrieval, question):
            request, context = real_build(retrieval, question)
            built["request"] = request
            built["context"] = context
            return request, context

        def wrap_validate(request, context, candidate):
            seen["request"] = request
            seen["context"] = context
            seen["candidate"] = candidate
            return real_validate(request, context, candidate)

        with (
            patch("performance.ai_orchestration.grounded.build_evidence_packet", wrap_build),
            patch("performance.ai_orchestration.grounded.validate_grounded_response", wrap_validate),
        ):
            generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        self.assertIs(seen["request"], built["request"])
        self.assertIs(seen["context"], built["context"])

    def test_raw_candidate_reaches_validator_unchanged(self):
        candidate = _valid_candidate()
        seen: dict[str, str] = {}
        real_validate = validate_grounded_response

        def wrap_validate(request, context, raw):
            seen["candidate"] = raw
            return real_validate(request, context, raw)

        with patch("performance.ai_orchestration.grounded.validate_grounded_response", wrap_validate):
            result = generate_grounded_response(
                _retrieval(),
                QUESTION,
                _RecordingProvider(_provider_response(candidate)),
            )
        self.assertEqual(seen["candidate"], candidate)
        self.assertIs(result.raw_candidate, candidate)
        self.assertEqual(result.raw_candidate, candidate)

    def test_prompt_and_hash_match_production_renderer(self):
        provider = _RecordingProvider()
        retrieval = _retrieval()
        request, _context = build_evidence_packet(retrieval, QUESTION)
        expected_prompt = render_evidence_summary_prompt(request)
        result = generate_grounded_response(retrieval, QUESTION, provider)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0].prompt_text, expected_prompt)
        self.assertEqual(provider.calls[0].prompt_sha256, prompt_hash(expected_prompt))
        self.assertEqual(result.prompt_sha256, prompt_hash(expected_prompt))

    def test_packet_error_does_not_call_provider_and_preserves_code(self):
        provider = _RecordingProvider()
        empty = _retrieval(results=())
        result = generate_grounded_response(empty, QUESTION, provider)
        self.assertEqual(result.outcome, GroundedGenerationOutcome.NOT_ATTEMPTED_PACKET_ERROR)
        self.assertEqual(result.packet_error_code, RejectionCode.REQUEST_EMPTY_EVIDENCE)
        self.assertEqual(provider.calls, [])
        self.assertIsNone(result.validation_result)
        self.assertIsNone(result.raw_candidate)
        self.assertIs(result.review_required, True)

    def test_non_ok_retrieval_is_packet_error_without_provider(self):
        provider = _RecordingProvider()
        result = generate_grounded_response(
            _retrieval(state=RETRIEVAL_STATE_EMPTY_QUERY, results=()),
            QUESTION,
            provider,
        )
        self.assertEqual(result.outcome, GroundedGenerationOutcome.NOT_ATTEMPTED_PACKET_ERROR)
        self.assertEqual(result.packet_error_code, RejectionCode.REQUEST_STATE_NOT_OK)
        self.assertEqual(provider.calls, [])

    def test_each_provider_failure_code_maps_to_provider_failed(self):
        for code in ProviderFailureCode:
            failure = ProviderFailure(category=code, provider_name=PROVIDER_NAME)
            provider = _RecordingProvider(failure)
            with patch("performance.ai_orchestration.grounded.validate_grounded_response") as mocked:
                result = generate_grounded_response(_retrieval(), QUESTION, provider)
            self.assertEqual(result.outcome, GroundedGenerationOutcome.PROVIDER_FAILED, code)
            self.assertIs(result.provider_failure, failure)
            self.assertIsNone(result.validation_result)
            self.assertIsNone(result.raw_candidate)
            mocked.assert_not_called()

    def test_provider_exception_maps_to_provider_error_without_leakage(self):
        provider = _RecordingProvider(error=RuntimeError(EXCEPTION_SENTINEL))
        logger = logging.getLogger("performance.ai_orchestration")
        with (
            patch("performance.ai_orchestration.grounded.validate_grounded_response") as mocked,
            self.assertNoLogs(logger, level="DEBUG"),
        ):
            result = generate_grounded_response(_retrieval(), QUESTION, provider)
        self.assertEqual(result.outcome, GroundedGenerationOutcome.PROVIDER_FAILED)
        self.assertEqual(result.provider_failure.category, ProviderFailureCode.PROVIDER_ERROR)
        self.assertEqual(result.provider_failure.provider_name, "unknown")
        self.assertIsNone(result.validation_result)
        mocked.assert_not_called()
        self.assertNotIn(EXCEPTION_SENTINEL, repr(result))
        self.assertNotIn(EXCEPTION_SENTINEL, str(result))
        self.assertNotIn(EXCEPTION_SENTINEL, repr(result.provider_failure))

    def test_invalid_provider_return_maps_to_provider_error(self):
        provider = _RecordingProvider(object())
        with patch("performance.ai_orchestration.grounded.validate_grounded_response") as mocked:
            result = generate_grounded_response(_retrieval(), QUESTION, provider)
        self.assertEqual(result.outcome, GroundedGenerationOutcome.PROVIDER_FAILED)
        self.assertEqual(result.provider_failure.category, ProviderFailureCode.PROVIDER_ERROR)
        self.assertEqual(result.provider_failure.provider_name, "unknown")
        mocked.assert_not_called()

    def test_non_string_raw_text_is_response_invalid_and_skips_validator(self):
        provider = _RecordingProvider(_provider_response(raw_text=123))
        with patch("performance.ai_orchestration.grounded.validate_grounded_response") as mocked:
            result = generate_grounded_response(_retrieval(), QUESTION, provider)
        self.assertEqual(result.outcome, GroundedGenerationOutcome.PROVIDER_FAILED)
        self.assertEqual(result.provider_failure.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)
        self.assertEqual(result.provider_failure.provider_name, PROVIDER_NAME)
        mocked.assert_not_called()
        self.assertIsNone(result.raw_candidate)

    def test_fenced_json_is_unrepaired_and_schema_invalid(self):
        fenced = "```json\n" + _valid_candidate() + "\n```"
        seen: dict[str, str] = {}
        real_validate = validate_grounded_response

        def wrap_validate(request, context, raw):
            seen["candidate"] = raw
            return real_validate(request, context, raw)

        with patch("performance.ai_orchestration.grounded.validate_grounded_response", wrap_validate):
            result = generate_grounded_response(
                _retrieval(),
                QUESTION,
                _RecordingProvider(_provider_response(fenced)),
            )
        self.assertEqual(seen["candidate"], fenced)
        self.assertEqual(result.raw_candidate, fenced)
        self.assertEqual(result.outcome, GroundedGenerationOutcome.VALIDATION_REJECTED)
        self.assertIn(RejectionCode.SCHEMA_INVALID_JSON, result.validation_result.rejection_codes)
        self.assertNotEqual(result.outcome, GroundedGenerationOutcome.PROVIDER_FAILED)

    def test_valid_grounded_json_still_requires_review(self):
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        self.assertEqual(result.outcome, GroundedGenerationOutcome.PASSED_DETERMINISTIC_CHECKS)
        self.assertIs(result.review_required, True)
        self.assertIs(result.validation_result.review_required, True)

    def test_validator_rejection_is_not_provider_failed(self):
        invalid = json.dumps(
            {
                "schema_version": "grounded-ai-response-v1",
                "answer": "",
                "claims": [{"evidence": "E1", "field": "total_trades", "value": "6"}],
                "citations": ["E1"],
                "limitation_codes": [],
            },
            ensure_ascii=True,
        )
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider(_provider_response(invalid)))
        self.assertEqual(result.outcome, GroundedGenerationOutcome.VALIDATION_REJECTED)
        self.assertNotEqual(result.outcome, GroundedGenerationOutcome.PROVIDER_FAILED)
        self.assertIsNone(result.provider_failure)

    def test_prompt_excludes_server_owned_identity_and_leakage_tokens(self):
        retrieval = _retrieval()
        request, context = build_evidence_packet(retrieval, QUESTION)
        provider = _RecordingProvider()
        generate_grounded_response(retrieval, QUESTION, provider)
        prompt = provider.calls[0].prompt_text
        self.assertNotIn(context.request_sha256, prompt)
        self.assertNotIn(context.journal_fingerprint, prompt)
        for _alias, document_id in context.alias_to_document_id:
            self.assertNotIn(document_id, prompt)
        for token in LEAKAGE_TOKENS:
            self.assertNotIn(token, prompt)

    def test_ai_orchestration_does_not_import_evaluation_or_persistence(self):
        for path in _iter_python_files(ORCH_ROOT):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_evaluation", blob)
            names = _imported_modules(path)
            for name in names:
                self.assertNotIn("ai_evaluation", name)
                self.assertNotIn(name, BLOCKED_PERSISTENCE)
                self.assertNotEqual(name, "django.db")
                root = name.split(".", 1)[0]
                self.assertNotEqual(root, "views")
                self.assertNotEqual(name, "performance.views")
                self.assertNotEqual(name, "performance.models")

    def test_ai_orchestration_has_no_network_or_sdk_import(self):
        for path in _iter_python_files(ORCH_ROOT):
            for name in _imported_modules(path):
                root = name.split(".", 1)[0]
                self.assertNotIn(root, BLOCKED_SDK)
                self.assertNotIn(name, BLOCKED_NETWORK)
                self.assertNotIn(root, BLOCKED_NETWORK)

    def test_ai_evaluation_does_not_import_provider_or_orchestration(self):
        for path in _iter_python_files(EVALUATION_ROOT):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_provider", blob)
            self.assertNotIn("ai_orchestration", blob)
            for name in _imported_modules(path):
                self.assertNotIn("ai_provider", name)
                self.assertNotIn("ai_orchestration", name)

    def test_no_production_fake_provider_exists(self):
        for path in _iter_python_files(ORCH_ROOT) + _iter_python_files(PROVIDER_ROOT):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("class Fake", blob)
            self.assertNotIn("class _Fake", blob)
            self.assertNotIn("fake_provider", blob)
            self.assertNotIn("FakeProvider", blob)

    def test_provider_is_called_exactly_once_on_success(self):
        provider = _RecordingProvider()
        generate_grounded_response(_retrieval(), QUESTION, provider)
        self.assertEqual(len(provider.calls), 1)

    def test_response_meta_excludes_raw_text_and_keeps_requested_model(self):
        result = generate_grounded_response(_retrieval(), QUESTION, _RecordingProvider())
        self.assertIsNotNone(result.provider_response_meta)
        self.assertEqual(result.provider_response_meta.provider_name, PROVIDER_NAME)
        self.assertEqual(result.provider_response_meta.model_id_requested, MODEL_ID)
        self.assertEqual(result.provider_response_meta.duration_ms, 11)
        self.assertFalse(hasattr(result.provider_response_meta, "raw_text"))
