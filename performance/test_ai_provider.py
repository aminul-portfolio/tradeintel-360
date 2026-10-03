from __future__ import annotations

import ast
import dataclasses
import io
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from .ai_provider import (
    FROZEN_PROVIDER_FAILURE_CODES,
    FROZEN_STOP_CATEGORIES,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    PROVIDER_NAME,
    TIMEOUT_SECONDS,
    Provider,
    ProviderConfig,
    ProviderFailure,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
    StopCategory,
    load_provider_config,
    resolve_provider_config,
)
from .ai_provider.config import API_KEY_ENV, ENABLED_ENV

PACKAGE_ROOT = Path(__file__).resolve().parent
PROVIDER_ROOT = PACKAGE_ROOT / "ai_provider"
EVALUATION_ROOT = PACKAGE_ROOT / "ai_evaluation"
SENTINEL_KEY = "TEST-SENTINEL-KEY-DO-NOT-LEAK"
PROVIDER_ENV = frozenset({ENABLED_ENV, API_KEY_ENV})
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
BLOCKED_INTERNAL = frozenset(
    {
        "performance.rag",
        "performance.ai_grounding",
        "performance.ai_orchestration",
        "performance.ai_evaluation",
    }
)


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


def _isolate_provider_env(**updates: str) -> patch:
    mapping = {key: value for key, value in os.environ.items() if key not in PROVIDER_ENV}
    mapping.update(updates)
    return patch.dict(os.environ, mapping, clear=True)


class _FakeProvider:
    def generate(self, request: ProviderRequest) -> ProviderResponse | ProviderFailure:
        return ProviderResponse(
            raw_text="untrusted-raw-candidate",
            provider_name=PROVIDER_NAME,
            model_id_requested=MODEL_ID,
            model_id_reported=None,
            stop_category=StopCategory.COMPLETE,
            input_tokens=1,
            output_tokens=1,
            duration_ms=0,
        )


def _sample_config(*, enabled: bool = True, api_key: str = SENTINEL_KEY) -> ProviderConfig:
    return ProviderConfig(
        provider_name=PROVIDER_NAME,
        model_id=MODEL_ID,
        timeout_seconds=TIMEOUT_SECONDS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        enabled=enabled,
        api_key=api_key,
    )


def _sample_request() -> ProviderRequest:
    return ProviderRequest(prompt_text="Restate the supplied historical evidence.", prompt_sha256="a" * 64)


def _sample_response() -> ProviderResponse:
    return ProviderResponse(
        raw_text="untrusted-raw-candidate",
        provider_name=PROVIDER_NAME,
        model_id_requested=MODEL_ID,
        model_id_reported=None,
        stop_category=StopCategory.COMPLETE,
        input_tokens=None,
        output_tokens=None,
        duration_ms=12,
    )


def _sample_failure() -> ProviderFailure:
    return ProviderFailure(
        category=ProviderFailureCode.PROVIDER_NOT_CONFIGURED,
        provider_name=PROVIDER_NAME,
    )


class AiProviderBoundaryTests(SimpleTestCase):
    def test_provider_config_is_immutable(self):
        config = _sample_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.enabled = False

    def test_provider_request_is_immutable(self):
        request = _sample_request()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.prompt_sha256 = "b" * 64

    def test_provider_response_is_immutable(self):
        response = _sample_response()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            response.duration_ms = 99

    def test_provider_failure_is_immutable(self):
        failure = _sample_failure()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            failure.provider_name = "other"

    def test_frozen_failure_taxonomy_contains_exactly_six_codes(self):
        self.assertEqual(len(FROZEN_PROVIDER_FAILURE_CODES), 6)
        self.assertEqual(FROZEN_PROVIDER_FAILURE_CODES, tuple(code.value for code in ProviderFailureCode))
        self.assertEqual(
            set(FROZEN_PROVIDER_FAILURE_CODES),
            {
                "PROVIDER_NOT_CONFIGURED",
                "PROVIDER_AUTH_FAILED",
                "PROVIDER_RATE_LIMITED",
                "PROVIDER_TIMEOUT",
                "PROVIDER_RESPONSE_INVALID",
                "PROVIDER_ERROR",
            },
        )

    def test_stop_category_contains_exactly_three_values(self):
        self.assertEqual(FROZEN_STOP_CATEGORIES, ("COMPLETE", "MAX_TOKENS", "OTHER"))
        self.assertEqual(tuple(item.value for item in StopCategory), FROZEN_STOP_CATEGORIES)

    def test_provider_disabled_by_default(self):
        with _isolate_provider_env():
            config = load_provider_config()
        self.assertFalse(config.enabled)
        self.assertEqual(config.api_key, "")

    def test_literal_one_enables_configuration(self):
        with _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}):
            config = load_provider_config()
        self.assertTrue(config.enabled)

    def test_true_does_not_enable_configuration(self):
        with _isolate_provider_env(**{ENABLED_ENV: "true", API_KEY_ENV: SENTINEL_KEY}):
            config = load_provider_config()
        self.assertFalse(config.enabled)

    def test_yes_does_not_enable_configuration(self):
        with _isolate_provider_env(**{ENABLED_ENV: "yes", API_KEY_ENV: SENTINEL_KEY}):
            config = load_provider_config()
        self.assertFalse(config.enabled)

    def test_zero_does_not_enable_configuration(self):
        with _isolate_provider_env(**{ENABLED_ENV: "0", API_KEY_ENV: SENTINEL_KEY}):
            config = load_provider_config()
        self.assertFalse(config.enabled)

    def test_missing_api_key_while_enabled_is_not_configured(self):
        with _isolate_provider_env(**{ENABLED_ENV: "1"}):
            result = resolve_provider_config()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)

    def test_blank_api_key_while_enabled_is_not_configured(self):
        with _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: "   "}):
            result = resolve_provider_config()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)

    def test_disabled_provider_is_not_configured_even_with_key(self):
        with _isolate_provider_env(**{ENABLED_ENV: "0", API_KEY_ENV: SENTINEL_KEY}):
            result = resolve_provider_config()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)

    def test_enabled_nonblank_key_resolves_provider_config(self):
        with _isolate_provider_env(**{ENABLED_ENV: "1", API_KEY_ENV: SENTINEL_KEY}):
            result = resolve_provider_config()
        self.assertIsInstance(result, ProviderConfig)
        self.assertTrue(result.enabled)
        self.assertEqual(result.api_key, SENTINEL_KEY)

    def test_factory_reads_enabled_flag_at_call_time(self):
        with _isolate_provider_env():
            self.assertFalse(load_provider_config().enabled)
            os.environ[ENABLED_ENV] = "1"
            os.environ[API_KEY_ENV] = SENTINEL_KEY
            self.assertTrue(load_provider_config().enabled)

    def test_factory_reads_api_key_at_call_time(self):
        with _isolate_provider_env(**{ENABLED_ENV: "1"}):
            first = load_provider_config()
            self.assertEqual(first.api_key, "")
            os.environ[API_KEY_ENV] = SENTINEL_KEY
            second = load_provider_config()
            self.assertEqual(second.api_key, SENTINEL_KEY)

    def test_provider_name_is_fixed_to_anthropic(self):
        with _isolate_provider_env():
            config = load_provider_config()
        self.assertEqual(PROVIDER_NAME, "anthropic")
        self.assertEqual(config.provider_name, "anthropic")

    def test_model_id_is_fixed(self):
        with _isolate_provider_env():
            config = load_provider_config()
        self.assertEqual(MODEL_ID, "claude-sonnet-5-5")
        self.assertEqual(config.model_id, "claude-sonnet-5-5")

    def test_timeout_is_fixed_to_thirty_seconds(self):
        with _isolate_provider_env():
            config = load_provider_config()
        self.assertEqual(TIMEOUT_SECONDS, 30)
        self.assertEqual(config.timeout_seconds, 30)

    def test_max_output_tokens_is_fixed(self):
        with _isolate_provider_env():
            config = load_provider_config()
        self.assertEqual(MAX_OUTPUT_TOKENS, 8192)
        self.assertEqual(config.max_output_tokens, 8192)

    def test_api_key_absent_from_provider_config_repr(self):
        text = repr(_sample_config())
        self.assertNotIn(SENTINEL_KEY, text)
        self.assertNotIn("api_key", text)

    def test_api_key_absent_from_provider_config_str(self):
        text = str(_sample_config())
        self.assertNotIn(SENTINEL_KEY, text)
        self.assertNotIn("api_key", text)

    def test_api_key_excluded_from_provider_config_equality(self):
        left = _sample_config(api_key=SENTINEL_KEY)
        right = _sample_config(api_key="OTHER-NON-SECRET")
        self.assertEqual(left, right)

    def test_raw_text_absent_from_provider_response_repr(self):
        response = _sample_response()
        text = repr(response)
        self.assertNotIn(response.raw_text, text)
        self.assertNotIn("raw_text", text)

    def test_prompt_text_absent_from_provider_request_repr(self):
        request = _sample_request()
        text = repr(request)
        self.assertNotIn(request.prompt_text, text)
        self.assertNotIn("prompt_text", text)

    def test_provider_failure_has_no_message_field(self):
        names = {item.name for item in dataclasses.fields(ProviderFailure)}
        self.assertEqual(names, {"category", "provider_name"})
        self.assertNotIn("message", names)
        self.assertNotIn("exception", names)
        failure = _sample_failure()
        self.assertFalse(hasattr(failure, "message"))

    def test_test_only_fake_satisfies_provider_protocol(self):
        provider: Provider = _FakeProvider()
        result = provider.generate(_sample_request())
        self.assertIsInstance(result, ProviderResponse)
        self.assertEqual(result.provider_name, PROVIDER_NAME)
        self.assertEqual(result.stop_category, StopCategory.COMPLETE)

    def test_ai_provider_has_no_provider_sdk_import(self):
        for path in _iter_python_files(PROVIDER_ROOT):
            for name in _imported_modules(path):
                root = name.split(".", 1)[0]
                self.assertNotIn(root, BLOCKED_SDK, path.name)

    def test_ai_provider_has_no_network_library_import(self):
        for path in _iter_python_files(PROVIDER_ROOT):
            for name in _imported_modules(path):
                self.assertNotIn(name, BLOCKED_NETWORK, path.name)
                self.assertNotIn(name.split(".", 1)[0], BLOCKED_NETWORK, path.name)

    def test_ai_provider_does_not_import_forbidden_internal_packages(self):
        for path in _iter_python_files(PROVIDER_ROOT):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_grounding", blob)
            self.assertNotIn("ai_evaluation", blob)
            self.assertNotIn("ai_orchestration", blob)
            self.assertNotIn("performance.rag", blob)
            for name in _imported_modules(path):
                self.assertNotIn(name, BLOCKED_INTERNAL)

    def test_ai_evaluation_does_not_import_provider_or_orchestration(self):
        for path in _iter_python_files(EVALUATION_ROOT):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_provider", blob)
            self.assertNotIn("ai_orchestration", blob)
            for name in _imported_modules(path):
                self.assertNotIn("ai_provider", name)
                self.assertNotIn("ai_orchestration", name)

    def test_sentinel_absent_from_captured_stdout(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            print(repr(_sample_config()))
            print(str(_sample_config()))
            print(repr(_sample_request()))
            print(repr(_sample_response()))
            print(repr(_sample_failure()))
        self.assertNotIn(SENTINEL_KEY, stdout.getvalue())

    def test_sentinel_absent_from_captured_stderr(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            print(repr(_sample_config()), file=sys.stderr)
            print(str(_sample_config()), file=sys.stderr)
        self.assertNotIn(SENTINEL_KEY, stderr.getvalue())

    def test_sentinel_absent_from_captured_logs(self):
        logger = logging.getLogger("performance.ai_provider")
        with self.assertLogs(logger, level="INFO") as captured:
            logger.info("%r", _sample_config())
            logger.info("%s", _sample_config())
            logger.info("%r", _sample_request())
            logger.info("%r", _sample_response())
            logger.info("%r", _sample_failure())
        joined = "\n".join(captured.output)
        self.assertNotIn(SENTINEL_KEY, joined)

    def test_absent_enabled_flag_does_not_enable(self):
        with _isolate_provider_env(**{API_KEY_ENV: SENTINEL_KEY}):
            config = load_provider_config()
            result = resolve_provider_config()
        self.assertFalse(config.enabled)
        self.assertIsInstance(result, ProviderFailure)

    def test_false_and_blank_enabled_flags_do_not_enable(self):
        for value in ("", "false", "TRUE", "Yes", "on"):
            with _isolate_provider_env(**{ENABLED_ENV: value, API_KEY_ENV: SENTINEL_KEY}):
                self.assertFalse(load_provider_config().enabled, value)

    def test_provider_request_has_only_prompt_fields(self):
        names = {item.name for item in dataclasses.fields(ProviderRequest)}
        self.assertEqual(names, {"prompt_text", "prompt_sha256"})

    def test_provider_config_has_no_temperature_or_retry_fields(self):
        names = {item.name for item in dataclasses.fields(ProviderConfig)}
        self.assertEqual(
            names,
            {
                "provider_name",
                "model_id",
                "timeout_seconds",
                "max_output_tokens",
                "enabled",
                "api_key",
            },
        )

    def test_failure_repr_does_not_include_sentinel_or_http_body(self):
        text = repr(_sample_failure())
        self.assertNotIn(SENTINEL_KEY, text)
        names = {item.name for item in dataclasses.fields(ProviderFailure)}
        self.assertNotIn("headers", names)
        self.assertNotIn("request_id", names)

    def test_resolve_does_not_import_sdk_when_not_configured(self):
        with _isolate_provider_env():
            result = resolve_provider_config()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category.value, "PROVIDER_NOT_CONFIGURED")
        self.assertNotIn("anthropic", sys.modules)
