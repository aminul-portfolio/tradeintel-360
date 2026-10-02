from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import io
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.core.management import call_command, load_command_class
from django.core.management.base import CommandError
from django.test import SimpleTestCase

from .ai_grounding import (
    RejectionCode,
    ValidationStatus,
    build_evidence_packet,
)
from .ai_grounding.validation import GroundingValidationResult
from .ai_orchestration.canary_fixture import CANARY_CASE_ID, CANARY_QUESTION, load_canary_retrieval
from .ai_orchestration.grounded import GroundedGenerationOutcome, GroundedGenerationResult, ProviderResponseMeta
from .ai_provider import (
    MODEL_ID,
    PROVIDER_NAME,
    ProviderFailure,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
)
from .ai_provider.adapter_anthropic import (
    MAX_RETRIES,
    OUTPUT_CONFIG,
    STREAM,
    THINKING,
    AnthropicProvider,
    build_anthropic_provider,
)
from .ai_provider.config import API_KEY_ENV, ENABLED_ENV
from .ai_provider.errors import StopCategory
from .management.commands.run_provider_canary import (
    RecordingProvider,
    detect_test_runner,
    format_canary_proof,
)
from .rag.retrieval import RETRIEVAL_STATE_EMPTY_QUERY, RetrievalResponse

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ADAPTER_PATH = PACKAGE_ROOT / "ai_provider" / "adapter_anthropic.py"
SENTINEL_KEY = "TEST-SENTINEL-KEY-DO-NOT-LEAK"
PROMPT_SENTINEL = "CANARY-PROMPT-SENTINEL-DO-NOT-PRINT"
CANDIDATE_SENTINEL = "CANARY-CANDIDATE-SENTINEL-DO-NOT-PRINT"
EXCEPTION_SENTINEL = "CANARY-EXC-SENTINEL-DO-NOT-LEAK"
REQUEST_ID_SENTINEL = "req_sentinel_do_not_leak"
HEADER_SENTINEL = "x-sentinel-header"
BODY_SENTINEL = "http-body-sentinel-do-not-leak"
PROVIDER_ENV = frozenset({ENABLED_ENV, API_KEY_ENV})
FORBIDDEN_COMMAND_DESTS = frozenset(
    {
        "journal",
        "filename",
        "user_id",
        "account",
        "ticket",
        "symbol",
        "question",
        "prompt",
        "model",
        "provider",
        "retries",
        "timeout",
        "max_tokens",
        "max_token",
        "file",
        "path",
    }
)


def _iter_python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))


def _imported_modules(path: Path, *, module_level_only: bool = False) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if module_level_only else ast.walk(tree)
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return tuple(names)


def _isolate_provider_env(**updates: str):
    mapping = {key: value for key, value in os.environ.items() if key not in PROVIDER_ENV}
    mapping.update(updates)
    return patch.dict(os.environ, mapping, clear=True)


def _text_block(text: object) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _sample_request() -> ProviderRequest:
    return ProviderRequest(prompt_text=PROMPT_SENTINEL, prompt_sha256="a" * 64)


def _sdk_response(
    *,
    content: object,
    stop_reason: object = "end_turn",
    model: object = MODEL_ID,
    input_tokens: object = 3,
    output_tokens: object = 7,
    request_id: object = REQUEST_ID_SENTINEL,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        id=request_id,
        request_id=request_id,
        headers={HEADER_SENTINEL: "1"},
        body=BODY_SENTINEL,
    )


class _FakeSDK:
    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] | None = None
        self.create_calls: list[dict[str, object]] = []
        self.response: object = _sdk_response(content=[_text_block("exact-raw-text")])
        self.error: Exception | None = None
        self.AuthenticationError = type("AuthenticationError", (Exception,), {})
        self.PermissionDeniedError = type("PermissionDeniedError", (Exception,), {})
        self.RateLimitError = type("RateLimitError", (Exception,), {})
        self.APITimeoutError = type("APITimeoutError", (Exception,), {})
        self.APIConnectionError = type("APIConnectionError", (Exception,), {})
        self.APIError = type("APIError", (Exception,), {})
        self.APIStatusError = type("APIStatusError", (Exception,), {})
        self.Anthropic = self._client_class()

    def _client_class(self):
        sdk = self

        class Anthropic:
            def __init__(self, **kwargs):
                sdk.init_kwargs = kwargs
                self.messages = self

            def create(self, **kwargs):
                sdk.create_calls.append(kwargs)
                if sdk.error is not None:
                    raise sdk.error
                return sdk.response

        return Anthropic


class _CountingProvider:
    def __init__(self) -> None:
        self.calls: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResponse | ProviderFailure:
        self.calls.append(request)
        return ProviderResponse(
            raw_text="unused-raw",
            provider_name=PROVIDER_NAME,
            model_id_requested=MODEL_ID,
            model_id_reported=MODEL_ID,
            stop_category=StopCategory.COMPLETE,
            input_tokens=1,
            output_tokens=1,
            duration_ms=1,
        )


def _configured_provider(sdk: _FakeSDK) -> AnthropicProvider:
    with (
        _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
        patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk", return_value=sdk),
    ):
        built = build_anthropic_provider()
    assert isinstance(built, AnthropicProvider)
    return built


def _sample_meta() -> ProviderResponseMeta:
    return ProviderResponseMeta(
        provider_name=PROVIDER_NAME,
        model_id_requested=MODEL_ID,
        model_id_reported=MODEL_ID,
        stop_category=StopCategory.COMPLETE,
        input_tokens=3,
        output_tokens=7,
        duration_ms=12,
    )


def _sample_result(**overrides) -> GroundedGenerationResult:
    body = {
        "outcome": GroundedGenerationOutcome.PASSED_DETERMINISTIC_CHECKS,
        "prompt_sha256": "b" * 64,
        "packet_error_code": None,
        "provider_failure": None,
        "provider_response_meta": _sample_meta(),
        "validation_result": GroundingValidationResult(
            status=ValidationStatus.PASSED_DETERMINISTIC_CHECKS,
            rejection_codes=(),
            review_required=True,
        ),
        "raw_candidate": CANDIDATE_SENTINEL,
    }
    body.update(overrides)
    return GroundedGenerationResult(**body)


class ProviderCanaryOfflineTests(SimpleTestCase):
    def test_requirements_pin_is_exact(self):
        text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("anthropic==1.11.0", text)
        self.assertNotIn("anthropic[", text)
        self.assertEqual(importlib.metadata.version("anthropic"), "1.11.0")

    def test_importing_adapter_does_not_import_sdk(self):
        names = _imported_modules(ADAPTER_PATH)
        for name in names:
            self.assertFalse(name == "anthropic" or name.startswith("anthropic."))
        self.assertNotIn("anthropic", sys.modules)
        importlib.import_module("performance.ai_provider.adapter_anthropic")
        self.assertNotIn("anthropic", sys.modules)

    def test_disabled_config_returns_not_configured_before_sdk_load(self):
        loader = patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk")
        with _isolate_provider_env(), loader as mocked:
            result = build_anthropic_provider()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)
        mocked.assert_not_called()

    def test_missing_key_returns_not_configured_before_sdk_load(self):
        loader = patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk")
        with _isolate_provider_env(**{ENABLED_ENV: "1"}), loader as mocked:
            result = build_anthropic_provider()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)
        mocked.assert_not_called()

    def test_configured_factory_loads_sdk_lazily(self):
        sdk = _FakeSDK()
        with (
            _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
            patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk", return_value=sdk) as mocked,
        ):
            result = build_anthropic_provider()
        mocked.assert_called()
        self.assertIsInstance(result, AnthropicProvider)

    def test_constructor_receives_api_key_timeout_and_zero_retries(self):
        sdk = _FakeSDK()
        _configured_provider(sdk)
        self.assertEqual(sdk.init_kwargs["api_key"], SENTINEL_KEY)
        self.assertEqual(sdk.init_kwargs["timeout"], 30)
        self.assertEqual(sdk.init_kwargs["max_retries"], 0)
        self.assertEqual(MAX_RETRIES, 0)

    def test_generate_calls_messages_create_exactly_once_with_locked_request(self):
        sdk = _FakeSDK()
        provider = _configured_provider(sdk)
        result = provider.generate(_sample_request())
        self.assertIsInstance(result, ProviderResponse)
        self.assertEqual(len(sdk.create_calls), 1)
        kwargs = sdk.create_calls[0]
        self.assertEqual(kwargs["model"], "claude-sonnet-5-5")
        self.assertEqual(kwargs["max_tokens"], 8192)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": PROMPT_SENTINEL}])
        self.assertIs(kwargs["stream"], STREAM)
        self.assertFalse(kwargs["stream"])
        self.assertEqual(kwargs["thinking"], {"type": "between_tools"})
        self.assertEqual(kwargs["output_config"], {"effort": "low"})
        self.assertEqual(kwargs["thinking"], THINKING)
        self.assertEqual(kwargs["output_config"], OUTPUT_CONFIG)
        self.assertNotIn("system", kwargs)
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("response_format", kwargs)
        self.assertNotIn("metadata", kwargs)
        self.assertEqual(set(kwargs), {"model", "max_tokens", "messages", "stream", "thinking", "output_config"})

    def test_single_text_block_returns_identical_raw_text(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block("byte-identical-raw")])
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertIsInstance(result, ProviderResponse)
        self.assertEqual(result.raw_text, "byte-identical-raw")
        self.assertIs(result.raw_text, sdk.response.content[0].text)

    def test_zero_blocks_are_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[])
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_two_text_blocks_are_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block("a"), _text_block("b")])
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_one_non_text_block_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[SimpleNamespace(type="thinking", text="x")])
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_text_plus_non_text_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(
            content=[_text_block("a"), SimpleNamespace(type="thinking", thinking="x")]
        )
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_non_str_text_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block(123)])
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_missing_content_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = SimpleNamespace(stop_reason="end_turn", model=MODEL_ID, usage=None)
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_end_turn_maps_to_complete(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block("ok")], stop_reason="end_turn")
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.stop_category, StopCategory.COMPLETE)

    def test_max_tokens_stop_reason_is_not_transport_failure(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block("partial")], stop_reason="max_tokens")
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertIsInstance(result, ProviderResponse)
        self.assertEqual(result.stop_category, StopCategory.MAX_TOKENS)

    def test_unknown_stop_reason_maps_to_other(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block("ok")], stop_reason="refusal")
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.stop_category, StopCategory.OTHER)

    def test_usage_and_reported_model_map_without_raw_response_leakage(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(
            content=[_text_block("ok")],
            model="claude-sonnet-5-5",
            input_tokens=11,
            output_tokens=22,
        )
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.input_tokens, 11)
        self.assertEqual(result.output_tokens, 22)
        self.assertEqual(result.model_id_reported, "claude-sonnet-5-5")
        self.assertEqual(result.provider_name, "anthropic")
        leaked = repr(result) + str(result)
        self.assertNotIn(REQUEST_ID_SENTINEL, leaked)
        self.assertNotIn(HEADER_SENTINEL, leaked)
        self.assertNotIn(BODY_SENTINEL, leaked)
        self.assertNotIn(SENTINEL_KEY, leaked)

    def test_non_string_reported_model_is_none(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block("ok")], model=None)
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertIsNone(result.model_id_reported)

    def _assert_mapped_exception(self, attr: str, expected: ProviderFailureCode) -> None:
        sdk = _FakeSDK()
        sdk.error = getattr(sdk, attr)(EXCEPTION_SENTINEL)
        logger = logging.getLogger("performance.ai_provider")
        with self.assertNoLogs(logger, level="DEBUG"):
            result = _configured_provider(sdk).generate(_sample_request())
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, expected)
        leaked = repr(result) + str(result)
        self.assertNotIn(EXCEPTION_SENTINEL, leaked)
        self.assertNotIn(REQUEST_ID_SENTINEL, leaked)
        self.assertNotIn(HEADER_SENTINEL, leaked)
        self.assertNotIn(BODY_SENTINEL, leaked)
        self.assertNotIn(attr, leaked)
        self.assertNotIn("Exception", leaked)

    def test_authentication_error_maps_to_auth_failed(self):
        self._assert_mapped_exception("AuthenticationError", ProviderFailureCode.PROVIDER_AUTH_FAILED)

    def test_permission_denied_error_maps_to_auth_failed(self):
        self._assert_mapped_exception("PermissionDeniedError", ProviderFailureCode.PROVIDER_AUTH_FAILED)

    def test_rate_limit_error_maps_to_rate_limited(self):
        self._assert_mapped_exception("RateLimitError", ProviderFailureCode.PROVIDER_RATE_LIMITED)

    def test_timeout_error_maps_to_timeout(self):
        self._assert_mapped_exception("APITimeoutError", ProviderFailureCode.PROVIDER_TIMEOUT)

    def test_connection_error_maps_to_provider_error(self):
        self._assert_mapped_exception("APIConnectionError", ProviderFailureCode.PROVIDER_ERROR)

    def test_generic_api_error_maps_to_provider_error(self):
        self._assert_mapped_exception("APIError", ProviderFailureCode.PROVIDER_ERROR)

    def test_unexpected_exception_maps_to_provider_error(self):
        sdk = _FakeSDK()
        sdk.error = RuntimeError(EXCEPTION_SENTINEL)
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_ERROR)
        self.assertNotIn(EXCEPTION_SENTINEL, repr(result))

    def test_canary_fixture_builds_through_real_packet_builder(self):
        retrieval = load_canary_retrieval()
        request, context = build_evidence_packet(retrieval, CANARY_QUESTION)
        self.assertEqual(request.question, CANARY_QUESTION)
        self.assertEqual(len(request.evidence), 1)
        self.assertTrue(context.request_sha256)
        content = retrieval.results[0].content
        self.assertEqual(content, "Total Trades: 6")
        self.assertNotIn("history.csv", content)
        self.assertNotIn("user_id", content)
        self.assertNotIn("owner_id", content)
        self.assertNotIn(".xlsx", content)
        self.assertEqual(CANARY_CASE_ID, "sprint-8-controlled-live-canary-001")

    def test_command_refuses_without_confirm_flag(self):
        loader = patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk")
        factory = patch("performance.management.commands.run_provider_canary.build_anthropic_provider")
        with loader as mocked_loader, factory as mocked_factory:
            with self.assertRaises(CommandError):
                call_command("run_provider_canary")
        mocked_loader.assert_not_called()
        mocked_factory.assert_not_called()

    def test_missing_configuration_refuses_before_provider_creation(self):
        loader = patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk")
        with (
            _isolate_provider_env(),
            patch("performance.management.commands.run_provider_canary.detect_test_runner", return_value=False),
            loader as mocked,
        ):
            with self.assertRaises(CommandError) as raised:
                call_command("run_provider_canary", "--confirm-live-canary")
        self.assertIn("not configured", str(raised.exception))
        mocked.assert_not_called()

    def test_test_runner_refuses_before_provider_creation(self):
        factory = patch("performance.management.commands.run_provider_canary.build_anthropic_provider")
        with factory as mocked:
            with self.assertRaises(CommandError) as raised:
                call_command("run_provider_canary", "--confirm-live-canary")
        self.assertIn("test runner", str(raised.exception))
        mocked.assert_not_called()
        self.assertTrue(detect_test_runner())

    def test_command_exposes_no_custom_journal_or_model_options(self):
        command = load_command_class("performance", "run_provider_canary")
        parser = command.create_parser("manage.py", "run_provider_canary")
        dests = {action.dest for action in parser._actions}
        self.assertIn("confirm_live_canary", dests)
        self.assertTrue(FORBIDDEN_COMMAND_DESTS.isdisjoint(dests))

    def test_proof_formatter_excludes_secret_sentinels_and_is_deterministic(self):
        result = _sample_result()
        first = format_canary_proof(
            result,
            case_id=CANARY_CASE_ID,
            provider_call_attempted=True,
            transport_response_received=True,
        )
        second = format_canary_proof(
            result,
            case_id=CANARY_CASE_ID,
            provider_call_attempted=True,
            transport_response_received=True,
        )
        self.assertEqual(first, second)
        self.assertIn("REVIEW_REQUIRED=True", first)
        self.assertNotIn(SENTINEL_KEY, first)
        self.assertNotIn(PROMPT_SENTINEL, first)
        self.assertNotIn(CANDIDATE_SENTINEL, first)
        self.assertNotIn(EXCEPTION_SENTINEL, first)
        self.assertNotIn(REQUEST_ID_SENTINEL, first)

    def test_validation_rejected_proof_is_not_rewritten_as_provider_failure(self):
        result = _sample_result(
            outcome=GroundedGenerationOutcome.VALIDATION_REJECTED,
            validation_result=GroundingValidationResult(
                status=ValidationStatus.REJECTED,
                rejection_codes=(RejectionCode.SCHEMA_INVALID_JSON,),
                review_required=True,
            ),
        )
        text = format_canary_proof(
            result,
            case_id=CANARY_CASE_ID,
            provider_call_attempted=True,
            transport_response_received=True,
        )
        self.assertIn("OUTCOME=VALIDATION_REJECTED", text)
        self.assertIn("VALIDATION_STATUS=REJECTED", text)
        self.assertIn("REJECTION_CODES=SCHEMA_INVALID_JSON", text)
        self.assertTrue(text.split("PROVIDER_FAILURE_CATEGORY=")[1].startswith("\n"))

    def test_passed_deterministic_checks_still_requires_review(self):
        result = _sample_result()
        self.assertIs(result.review_required, True)
        text = format_canary_proof(
            result,
            case_id=CANARY_CASE_ID,
            provider_call_attempted=True,
            transport_response_received=True,
        )
        self.assertIn("OUTCOME=PASSED_DETERMINISTIC_CHECKS", text)
        self.assertIn("REVIEW_REQUIRED=True", text)

    def test_no_retry_loop_in_adapter_or_command(self):
        adapter = ADAPTER_PATH.read_text(encoding="utf-8")
        command = (PACKAGE_ROOT / "management" / "commands" / "run_provider_canary.py").read_text(encoding="utf-8")
        self.assertIn("max_retries=MAX_RETRIES", adapter)
        self.assertNotIn("for attempt", adapter)
        self.assertNotIn("while True", adapter)
        self.assertNotIn("for attempt", command)
        self.assertEqual(adapter.count("messages.create"), 1)

    def test_no_file_or_database_persistence(self):
        paths = (
            ADAPTER_PATH,
            PACKAGE_ROOT / "ai_orchestration" / "canary_fixture.py",
            PACKAGE_ROOT / "management" / "commands" / "run_provider_canary.py",
        )
        for path in paths:
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn(".save(", blob)
            self.assertNotIn("open(", blob)
            self.assertNotIn("Path.write", blob)

    def test_command_imports_no_views_or_models(self):
        path = PACKAGE_ROOT / "management" / "commands" / "run_provider_canary.py"
        names = _imported_modules(path)
        for name in names:
            self.assertNotIn("views", name)
            self.assertNotIn("models", name)
            self.assertNotIn("django.db", name)

    def test_phase_modules_do_not_import_ai_evaluation(self):
        roots = (
            PACKAGE_ROOT / "ai_provider" / "adapter_anthropic.py",
            PACKAGE_ROOT / "ai_orchestration" / "canary_fixture.py",
            PACKAGE_ROOT / "management",
        )
        files: list[Path] = []
        for root in roots:
            files.extend(_iter_python_files(root) if root.is_dir() else [root])
        for path in files:
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_evaluation", blob)
            for name in _imported_modules(path):
                self.assertNotIn("ai_evaluation", name)

    def test_patched_command_writes_proof_without_live_sdk(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block('{"not":"valid"}')])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
            patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk", return_value=sdk),
            patch("performance.management.commands.run_provider_canary.detect_test_runner", return_value=False),
        ):
            call_command("run_provider_canary", "--confirm-live-canary", stdout=stdout, stderr=stderr)
        text = stdout.getvalue()
        self.assertIn("CANARY_CASE_ID=" + CANARY_CASE_ID, text)
        self.assertIn("REVIEW_REQUIRED=True", text)
        self.assertNotIn(SENTINEL_KEY, text)
        self.assertEqual(len(sdk.create_calls), 1)

    def test_tool_block_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[SimpleNamespace(type="tool_use", name="x")])
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_malformed_content_object_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=object())
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_detect_test_runner_recognises_pytest_script(self):
        self.assertTrue(detect_test_runner([r"C:\tools\pytest.exe"]))
        self.assertTrue(detect_test_runner(["python", "-m", "pytest"]))
        self.assertFalse(detect_test_runner(["manage.py", "run_provider_canary", "--confirm-live-canary"]))

    def test_thinking_then_text_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(
            content=[SimpleNamespace(type="thinking", thinking="x"), _text_block("a")]
        )
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_string_content_is_invalid(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content="not-a-sequence-of-blocks")
        result = _configured_provider(sdk).generate(_sample_request())
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_api_status_error_maps_to_provider_error(self):
        self._assert_mapped_exception("APIStatusError", ProviderFailureCode.PROVIDER_ERROR)

    def test_blank_key_refuses_before_sdk_load(self):
        loader = patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk")
        with _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: "   "}), loader as mocked:
            result = build_anthropic_provider()
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)
        mocked.assert_not_called()

    def test_synthetic_fixture_has_no_real_journal_user_or_file_input(self):
        blob = (PACKAGE_ROOT / "ai_orchestration" / "canary_fixture.py").read_text(encoding="utf-8")
        retrieval = load_canary_retrieval()
        combined = blob + retrieval.query + retrieval.results[0].content + retrieval.results[0].source_record_key
        for token in ("history.csv", "ticket", "account", "user@", "owner_id", "C:\\", "/home/", ".xlsx"):
            self.assertNotIn(token, combined)

    def test_proof_excludes_fingerprint_and_document_ids(self):
        result = _sample_result()
        text = format_canary_proof(
            result,
            case_id=CANARY_CASE_ID,
            provider_call_attempted=True,
            transport_response_received=True,
        )
        self.assertNotIn("c" * 64, text)
        self.assertNotIn("document_id", text)
        self.assertNotIn("alias_to_document_id", text)
        self.assertNotIn("ServerGroundingContext", text)

    def test_adapter_does_not_catch_base_exception(self):
        adapter = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn("except Exception as exc:", adapter)
        self.assertNotIn("except BaseException", adapter)

    def test_packet_error_proof_does_not_claim_provider_call(self):
        inner = _CountingProvider()
        empty = RetrievalResponse(
            state=RETRIEVAL_STATE_EMPTY_QUERY,
            query="",
            results=(),
            top_k=5,
            corpus_size=1,
            candidate_count=0,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
            patch("performance.management.commands.run_provider_canary.detect_test_runner", return_value=False),
            patch(
                "performance.management.commands.run_provider_canary.build_anthropic_provider",
                return_value=inner,
            ),
            patch(
                "performance.management.commands.run_provider_canary.load_canary_retrieval",
                return_value=empty,
            ),
            patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk") as loader,
        ):
            call_command("run_provider_canary", "--confirm-live-canary", stdout=stdout, stderr=stderr)
        text = stdout.getvalue()
        self.assertIn("PROVIDER_CALL_ATTEMPTED=NO", text)
        self.assertIn("OUTCOME=NOT_ATTEMPTED_PACKET_ERROR", text)
        self.assertEqual(len(inner.calls), 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertRegex(text, r"PROMPT_CHAR_COUNT=\n")
        self.assertRegex(text, r"RAW_OUTPUT_SHA256=\n")
        self.assertRegex(text, r"RAW_OUTPUT_CHAR_COUNT=\n")
        loader.assert_not_called()

    def test_normal_provider_execution_records_exactly_one_call(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block('{"not":"valid"}')])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
            patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk", return_value=sdk),
            patch("performance.management.commands.run_provider_canary.detect_test_runner", return_value=False),
        ):
            call_command("run_provider_canary", "--confirm-live-canary", stdout=stdout, stderr=stderr)
        text = stdout.getvalue()
        self.assertIn("PROVIDER_CALL_ATTEMPTED=YES", text)
        self.assertEqual(len(sdk.create_calls), 1)

    def test_proof_contains_prompt_and_raw_output_metadata_without_secrets(self):
        result = _sample_result()
        expected_sha = hashlib.sha256(CANDIDATE_SENTINEL.encode("utf-8")).hexdigest()
        text = format_canary_proof(
            result,
            case_id=CANARY_CASE_ID,
            provider_call_attempted=True,
            transport_response_received=True,
            prompt_char_count=len(PROMPT_SENTINEL),
        )
        self.assertIn("PROMPT_CHAR_COUNT=" + str(len(PROMPT_SENTINEL)), text)
        self.assertIn("RAW_OUTPUT_SHA256=" + expected_sha, text)
        self.assertIn("RAW_OUTPUT_CHAR_COUNT=" + str(len(CANDIDATE_SENTINEL)), text)
        self.assertNotIn(PROMPT_SENTINEL, text)
        self.assertNotIn(CANDIDATE_SENTINEL, text)
        self.assertNotIn(SENTINEL_KEY, text)

    def test_raw_candidate_is_on_stderr_and_absent_from_stdout_proof(self):
        sdk = _FakeSDK()
        sdk.response = _sdk_response(content=[_text_block(CANDIDATE_SENTINEL)])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
            patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk", return_value=sdk),
            patch("performance.management.commands.run_provider_canary.detect_test_runner", return_value=False),
        ):
            call_command("run_provider_canary", "--confirm-live-canary", stdout=stdout, stderr=stderr)
        proof = stdout.getvalue()
        review = stderr.getvalue()
        self.assertNotIn(CANDIDATE_SENTINEL, proof)
        self.assertIn("RAW_OUTPUT_HUMAN_REVIEW_BEGIN\n", review)
        begin = review.index("RAW_OUTPUT_HUMAN_REVIEW_BEGIN\n") + len("RAW_OUTPUT_HUMAN_REVIEW_BEGIN\n")
        end = review.index("\nRAW_OUTPUT_HUMAN_REVIEW_END\n")
        self.assertEqual(review[begin:end], CANDIDATE_SENTINEL)
        self.assertIn("PROVIDER_CALL_ATTEMPTED=YES", proof)

    def test_unknown_custom_argument_is_rejected_before_provider_use(self):
        loader = patch("performance.ai_provider.adapter_anthropic._load_anthropic_sdk")
        factory = patch("performance.management.commands.run_provider_canary.build_anthropic_provider")
        with (
            _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}),
            patch("performance.management.commands.run_provider_canary.detect_test_runner", return_value=False),
            loader as mocked_loader,
            factory as mocked_factory,
        ):
            with self.assertRaises(CommandError):
                call_command(
                    "run_provider_canary",
                    "--confirm-live-canary",
                    "--journal",
                    "synthetic.csv",
                )
        mocked_loader.assert_not_called()
        mocked_factory.assert_not_called()

    def test_recording_wrapper_refuses_a_second_provider_call(self):
        inner = _CountingProvider()
        recorder = RecordingProvider(inner)
        first = recorder.generate(_sample_request())
        self.assertIsInstance(first, ProviderResponse)
        self.assertEqual(recorder.call_count, 1)
        self.assertEqual(recorder.prompt_char_count, len(PROMPT_SENTINEL))
        with self.assertRaises(RuntimeError):
            recorder.generate(_sample_request())
        self.assertEqual(len(inner.calls), 1)
