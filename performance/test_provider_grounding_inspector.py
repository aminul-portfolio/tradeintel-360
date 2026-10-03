from __future__ import annotations

import ast
import json
import logging
import os
import socket
import threading
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from .ai_grounding import RejectionCode
from .ai_orchestration.canary_fixture import CANARY_QUESTION
from .ai_orchestration.grounded import GroundedGenerationOutcome, GroundedGenerationResult
from .ai_provider import (
    MODEL_ID,
    PROVIDER_NAME,
    ProviderFailure,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
)
from .ai_provider.config import API_KEY_ENV, ENABLED_ENV
from .ai_provider.errors import StopCategory
from .provider_inspector import (
    APPROVAL_TTL_SECONDS,
    SESSION_APPROVAL_KEY,
    SPRINT9_FIXTURE_ID,
    SPRINT9_PRESET_QUESTION_ID,
    build_server_owned_prompt,
    consume_approval_nonce,
    detect_test_runner,
    get_provider,
    provider_enabled,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
MODULE_PATH = PACKAGE_ROOT / "provider_inspector.py"
PROMPT_SENTINEL = "CLIENT-SUPPLIED-PROMPT-MUST-BE-IGNORED"
CANDIDATE_SENTINEL = "CLIENT-SUPPLIED-CANDIDATE-MUST-BE-IGNORED"
JOURNAL_SYMBOL = "NZDUSD"
SECRET_SENTINEL = "SPRINT9-SECRET-MUST-NEVER-APPEAR"
REQUEST_ID_SENTINEL = "req_sprint9_must_not_leak"
SCRIPT_PAYLOAD = '<script>alert("x")</script>'
AFFIRMATIVE_CLAIMS = (
    "output is verified",
    "verified output",
    "AI is accurate",
    "approved output",
    "reliable AI analysis",
    "fact-checked",
    "output is hallucination-free",
    "AI insight",
    "trading signal",
    "trading recommendation",
    "trading prediction",
)


def _valid_pass_candidate() -> str:
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


class RecordingProvider:
    def __init__(
        self,
        raw_text: str | None = None,
        failure: ProviderFailure | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[ProviderRequest] = []
        self.raw_text = '{"not":"valid"}' if raw_text is None else raw_text
        self.failure = failure
        self.error = error

    def generate(self, request: ProviderRequest) -> ProviderResponse | ProviderFailure:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if self.failure is not None:
            return self.failure
        return ProviderResponse(
            raw_text=self.raw_text,
            provider_name=PROVIDER_NAME,
            model_id_requested=MODEL_ID,
            model_id_reported=MODEL_ID,
            stop_category=StopCategory.COMPLETE,
            input_tokens=11,
            output_tokens=22,
            duration_ms=33,
        )


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return tuple(names)


class ProviderGroundingInspectorTests(TestCase):
    def setUp(self):
        cache.clear()
        self._socket_connect = patch(
            "socket.socket.connect",
            side_effect=RuntimeError("SPRINT9_NETWORK_BLOCKED"),
        )
        self._socket_create = patch(
            "socket.create_connection",
            side_effect=RuntimeError("SPRINT9_NETWORK_BLOCKED"),
        )
        self._socket_connect.start()
        self._socket_create.start()
        self.addCleanup(self._socket_connect.stop)
        self.addCleanup(self._socket_create.stop)
        env = {key: value for key, value in os.environ.items() if key != API_KEY_ENV}
        env[ENABLED_ENV] = "1"
        self._env = patch.dict(os.environ, env, clear=True)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.staff = User.objects.create_user(
            username="sprint9-staff",
            password="test-password-123",
            is_staff=True,
        )
        self.user = User.objects.create_user(
            username="sprint9-user",
            password="test-password-123",
            is_staff=False,
        )
        self.url = reverse("performance:provider_grounding_inspector")
        self.recorder = RecordingProvider()
        self.client.force_login(self.staff)

    def _preview(self):
        return self.client.get(self.url)

    def _confirm(self, **updates):
        preview = self._preview()
        payload = {
            "confirm_send": "1",
            "nonce": preview.context["nonce"],
            "question_id": SPRINT9_PRESET_QUESTION_ID,
        }
        payload.update(updates)
        return preview, self.client.post(self.url, payload)

    def test_anonymous_get_redirects_to_login(self):
        anon = Client()
        response = anon.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])

    def test_non_staff_get_is_forbidden(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(self.url).status_code, 403)

    def test_non_staff_post_is_forbidden(self):
        self.client.force_login(self.user)
        response = self.client.post(self.url, {"confirm_send": "1", "question_id": SPRINT9_PRESET_QUESTION_ID})
        self.assertEqual(response.status_code, 403)

    def test_staff_get_is_allowed(self):
        response = self._preview()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Send once to provider")
        self.assertContains(response, "Only synthetic, version-controlled evidence is used.")

    def test_csrf_protected_post_rejects_missing_token(self):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.staff)
        preview = csrf_client.get(self.url)
        response = csrf_client.post(
            self.url,
            {
                "confirm_send": "1",
                "nonce": preview.context["nonce"],
                "question_id": SPRINT9_PRESET_QUESTION_ID,
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_get_never_calls_get_provider_or_generate(self):
        with (
            patch("performance.provider_inspector.get_provider") as mocked_get,
            patch("performance.provider_inspector.generate_grounded_response") as mocked_generate,
        ):
            first = self.client.get(self.url)
            second = self.client.get(self.url)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        mocked_get.assert_not_called()
        mocked_generate.assert_not_called()
        self.assertEqual(len(self.recorder.calls), 0)

    def test_disabled_get_mints_no_approval_nonce(self):
        with patch.dict(os.environ, {ENABLED_ENV: "0"}):
            self.assertFalse(provider_enabled())
            response = self.client.get(self.url)
        self.assertContains(response, "External provider not enabled")
        self.assertEqual(response.context["nonce"], "")
        self.assertNotIn(SESSION_APPROVAL_KEY, self.client.session)

    def test_active_journal_is_not_required_or_sent(self):
        session = self.client.session
        session["cleaned_data"] = JOURNAL_SYMBOL + "-journal-payload"
        session.save()
        response = self._preview()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(JOURNAL_SYMBOL, response.context["preview_prompt"])
        self.assertNotIn("cleaned_data", response.context["preview_prompt"])
        self.assertIn(CANARY_QUESTION, response.context["preview_prompt"])
        self.assertContains(response, "No data from your trading journal is sent.")

    def test_unknown_question_id_is_refused_with_zero_calls(self):
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            _preview, response = self._confirm(question_id="unknown-question")
        self.assertEqual(response.context["inspector_state"], "QUESTION_REFUSED")
        self.assertEqual(len(self.recorder.calls), 0)

    def test_free_text_question_is_not_accepted(self):
        expected = build_server_owned_prompt(SPRINT9_PRESET_QUESTION_ID)
        assert expected is not None
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            _preview, response = self._confirm(question="Invent a new journal question.")
        self.assertEqual(response.context["inspector_state"], "REQUEST_COMPLETED")
        self.assertEqual(self.recorder.calls[0].prompt_text, expected[0])
        self.assertNotIn("Invent a new journal question.", self.recorder.calls[0].prompt_text)

    def test_nonce_and_session_approval_binding(self):
        first = self._preview()
        second = self._preview()
        self.assertTrue(first.context["nonce"])
        self.assertTrue(second.context["nonce"])
        self.assertNotEqual(first.context["nonce"], second.context["nonce"])
        approval = self.client.session[SESSION_APPROVAL_KEY]
        self.assertEqual(set(approval), {"nonce", "prompt_sha256", "fixture_id", "question_id", "issued_at"})
        self.assertEqual(approval["nonce"], second.context["nonce"])
        self.assertEqual(approval["prompt_sha256"], second.context["prompt_sha256"])
        self.assertEqual(approval["fixture_id"], SPRINT9_FIXTURE_ID)
        self.assertEqual(approval["question_id"], SPRINT9_PRESET_QUESTION_ID)
        self.assertTrue(approval["issued_at"])
        session_blob = repr(dict(self.client.session.items()))
        self.assertNotIn(second.context["preview_prompt"], session_blob)
        self.assertNotIn('{"not":"valid"}', session_blob)

    def test_missing_consent_makes_zero_provider_calls(self):
        preview = self._preview()
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            response = self.client.post(
                self.url,
                {"nonce": preview.context["nonce"], "question_id": SPRINT9_PRESET_QUESTION_ID},
            )
        self.assertEqual(response.context["inspector_state"], "CONSENT_REQUIRED")
        self.assertEqual(len(self.recorder.calls), 0)
        self.assertIn(SESSION_APPROVAL_KEY, self.client.session)

    def test_valid_confirmed_post_invokes_generate_once(self):
        expected = build_server_owned_prompt(SPRINT9_PRESET_QUESTION_ID)
        assert expected is not None
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            preview, response = self._confirm()
        self.assertEqual(response.context["inspector_state"], "REQUEST_COMPLETED")
        self.assertEqual(len(self.recorder.calls), 1)
        self.assertEqual(self.recorder.calls[0].prompt_text, expected[0])
        self.assertEqual(self.recorder.calls[0].prompt_sha256, expected[1])
        self.assertIs(response.context["review_required"], True)
        self.assertNotIn(SESSION_APPROVAL_KEY, self.client.session)
        self.assertNotIn(preview.context["preview_prompt"], repr(dict(self.client.session.items())))

    def test_replay_and_second_post_make_zero_additional_calls(self):
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            preview, first = self._confirm()
            replay = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
            second = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(first.context["inspector_state"], "REQUEST_COMPLETED")
        self.assertEqual(replay.context["inspector_state"], "APPROVAL_INVALID")
        self.assertEqual(second.context["inspector_state"], "APPROVAL_INVALID")
        self.assertEqual(len(self.recorder.calls), 1)

    def test_approval_is_consumed_before_provider_invocation(self):
        preview = self._preview()
        nonce = preview.context["nonce"]
        seen: dict[str, object] = {}

        class OrderProvider(RecordingProvider):
            def generate(inner, request: ProviderRequest) -> ProviderResponse:
                seen["session"] = SESSION_APPROVAL_KEY in self.client.session
                seen["nonce_consumed"] = cache.get("sprint9.provider_inspector.nonce:" + nonce)
                return super().generate(request)

        order_provider = OrderProvider()
        with patch("performance.provider_inspector.get_provider", return_value=order_provider):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": nonce,
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(response.context["inspector_state"], "REQUEST_COMPLETED")
        self.assertFalse(seen["session"])
        self.assertIsNotNone(seen["nonce_consumed"])
        self.assertEqual(len(order_provider.calls), 1)

    def test_expired_approval_makes_zero_calls(self):
        preview = self._preview()
        session = self.client.session
        approval = dict(session[SESSION_APPROVAL_KEY])
        approval["issued_at"] = (timezone.now() - timedelta(seconds=APPROVAL_TTL_SECONDS + 1)).isoformat()
        session[SESSION_APPROVAL_KEY] = approval
        session.save()
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(response.context["inspector_state"], "APPROVAL_EXPIRED")
        self.assertEqual(len(self.recorder.calls), 0)

    def test_wrong_nonce_makes_zero_calls(self):
        self._preview()
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": "wrong-nonce",
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(response.context["inspector_state"], "APPROVAL_INVALID")
        self.assertEqual(len(self.recorder.calls), 0)

    def test_tampered_prompt_hash_makes_zero_calls(self):
        preview = self._preview()
        session = self.client.session
        approval = dict(session[SESSION_APPROVAL_KEY])
        approval["prompt_sha256"] = "0" * 64
        session[SESSION_APPROVAL_KEY] = approval
        session.save()
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(response.context["inspector_state"], "PROMPT_BINDING_MISMATCH")
        self.assertEqual(len(self.recorder.calls), 0)

    def test_malicious_client_fields_cannot_change_provider_prompt(self):
        expected = build_server_owned_prompt(SPRINT9_PRESET_QUESTION_ID)
        assert expected is not None
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            _preview, response = self._confirm(
                prompt=PROMPT_SENTINEL,
                evidence="forged-evidence",
                provider="openai",
                model="other-model",
                prompt_sha256="f" * 64,
                raw_text=CANDIDATE_SENTINEL,
            )
        self.assertEqual(response.context["inspector_state"], "REQUEST_COMPLETED")
        request = self.recorder.calls[0]
        self.assertEqual(request.prompt_text, expected[0])
        self.assertEqual(request.prompt_sha256, expected[1])
        self.assertNotIn(PROMPT_SENTINEL, request.prompt_text)
        self.assertNotIn(CANDIDATE_SENTINEL, request.prompt_text)

    def test_concurrent_same_nonce_results_in_at_most_one_generate(self):
        nonce = "shared-sprint9-nonce"
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def attempt() -> None:
            try:
                barrier.wait(timeout=2)
                if consume_approval_nonce(nonce):
                    self.recorder.generate(
                        ProviderRequest(prompt_text="server-owned", prompt_sha256="a" * 64)
                    )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(self.recorder.calls), 1)

    def test_socket_and_create_connection_are_blocked(self):
        with self.assertRaises(RuntimeError):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        blocked = socket.socket()
        try:
            with self.assertRaises(RuntimeError):
                blocked.connect(("127.0.0.1", 9))
        finally:
            blocked.close()

    def test_get_provider_refuses_live_factory_under_test_runner(self):
        self.assertTrue(detect_test_runner())
        with patch("performance.ai_provider.adapter_anthropic.build_anthropic_provider") as factory:
            result = get_provider()
        self.assertIsInstance(result, ProviderFailure)
        self.assertEqual(result.category, ProviderFailureCode.PROVIDER_NOT_CONFIGURED)
        factory.assert_not_called()

    def test_unpatched_post_does_not_construct_live_provider(self):
        with patch("performance.ai_provider.adapter_anthropic.build_anthropic_provider") as factory:
            _preview, response = self._confirm()
        display = response.context["result_display"]
        self.assertEqual(response.context["outcome"], "PROVIDER_FAILED")
        self.assertEqual(display.provider_failure_category, "PROVIDER_NOT_CONFIGURED")
        self.assertFalse(display.transport_response_received)
        factory.assert_not_called()
        self.assertEqual(len(self.recorder.calls), 0)

    def test_module_import_boundary(self):
        names = _imported_modules(MODULE_PATH)
        blob = MODULE_PATH.read_text(encoding="utf-8")
        for name in names:
            root = name.split(".", 1)[0]
            self.assertNotEqual(root, "anthropic")
            self.assertNotIn("ai_evaluation", name)
        self.assertNotIn("performance.views", names)
        self.assertNotIn("performance.models", names)
        self.assertNotIn("ai_evaluation", blob)
        self.assertIn("generate_grounded_response", blob)
        self.assertIn("get_provider", blob)

    def test_existing_grounding_inspector_route_unchanged(self):
        response = self.client.get(reverse("performance:grounding_inspector"))
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(reverse("performance:grounding_inspector"), self.url)

    def test_disabled_post_makes_zero_calls(self):
        preview = self._preview()
        with (
            patch.dict(os.environ, {ENABLED_ENV: "0"}),
            patch("performance.provider_inspector.get_provider", return_value=self.recorder),
        ):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(response.context["inspector_state"], "PROVIDER_NOT_ENABLED")
        self.assertEqual(len(self.recorder.calls), 0)

    def test_api_key_absent_from_preview_context(self):
        response = self._preview()
        self.assertIsNone(response.context.get("api_key"))
        self.assertNotIn("api_key", response.content.decode())
        self.assertNotContains(response, API_KEY_ENV)
        self.assertNotIn(API_KEY_ENV, os.environ)

    def test_stale_nonce_after_replaced_get_is_rejected(self):
        first = self._preview()
        self._preview()
        with patch("performance.provider_inspector.get_provider", return_value=self.recorder):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": first.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        self.assertEqual(response.context["inspector_state"], "APPROVAL_INVALID")
        self.assertEqual(len(self.recorder.calls), 0)


class ProviderGroundingInspectorResultTests(TestCase):
    def setUp(self):
        cache.clear()
        self._socket_connect = patch(
            "socket.socket.connect",
            side_effect=RuntimeError("SPRINT9_NETWORK_BLOCKED"),
        )
        self._socket_create = patch(
            "socket.create_connection",
            side_effect=RuntimeError("SPRINT9_NETWORK_BLOCKED"),
        )
        self._socket_connect.start()
        self._socket_create.start()
        self.addCleanup(self._socket_connect.stop)
        self.addCleanup(self._socket_create.stop)
        env = {key: value for key, value in os.environ.items() if key != API_KEY_ENV}
        env[ENABLED_ENV] = "1"
        env[API_KEY_ENV] = SECRET_SENTINEL
        self._env = patch.dict(os.environ, env, clear=True)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.staff = User.objects.create_user(
            username="sprint9-result-staff",
            password="test-password-123",
            is_staff=True,
        )
        self.url = reverse("performance:provider_grounding_inspector")
        self.client.force_login(self.staff)

    def _preview(self):
        return self.client.get(self.url)

    def _confirm(self, provider, **updates):
        preview = self._preview()
        payload = {
            "confirm_send": "1",
            "nonce": preview.context["nonce"],
            "question_id": SPRINT9_PRESET_QUESTION_ID,
        }
        payload.update(updates)
        with patch("performance.provider_inspector.get_provider", return_value=provider):
            return preview, self.client.post(self.url, payload)

    def _html(self, response) -> str:
        return response.content.decode()

    def _assert_secret_absent(self, response, logs: str = "") -> None:
        html = self._html(response)
        display = response.context.get("result_display")
        context_blob = repr(display) + repr(response.context.get("outcome"))
        self.assertNotIn(SECRET_SENTINEL, html)
        self.assertNotIn(SECRET_SENTINEL, context_blob)
        self.assertNotIn(SECRET_SENTINEL, logs)
        self.assertNotIn(SECRET_SENTINEL, repr(dict(self.client.session.items())))

    def _assert_no_result_persistence(self, response, raw_output: str = "") -> None:
        session_blob = repr(dict(self.client.session.items()))
        self.assertNotIn(SESSION_APPROVAL_KEY, self.client.session)
        if raw_output:
            self.assertNotIn(raw_output, session_blob)
        self.assertNotIn(REQUEST_ID_SENTINEL, session_blob)
        self.assertNotIn(SECRET_SENTINEL, session_blob)

    def _assert_claim_safety(self, response) -> None:
        lowered = self._html(response).casefold()
        for claim in AFFIRMATIVE_CLAIMS:
            self.assertNotIn(claim.casefold(), lowered)
        self.assertNotIn("cost estimate", lowered)
        self.assertNotIn("price-per-token", lowered)
        self.assertContains(response, "Human review required.")

    def _assert_no_server_owned_leakage(self, response) -> None:
        html = self._html(response)
        self.assertNotIn("request_sha256", html)
        self.assertNotIn("journal_fingerprint", html)
        self.assertNotIn("alias_to_document_id", html)
        self.assertNotIn("KPI:kpi:", html)

    def test_not_attempted_packet_error_renders_without_provider_metadata(self):
        provider = RecordingProvider()
        packet_result = GroundedGenerationResult(
            outcome=GroundedGenerationOutcome.NOT_ATTEMPTED_PACKET_ERROR,
            prompt_sha256=None,
            packet_error_code=RejectionCode.REQUEST_STATE_NOT_OK,
            provider_failure=None,
            provider_response_meta=None,
            validation_result=None,
            raw_candidate=None,
        )
        preview = self._preview()
        with (
            patch("performance.provider_inspector.get_provider", return_value=provider),
            patch(
                "performance.provider_inspector.generate_grounded_response",
                return_value=packet_result,
            ),
        ):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        display = response.context["result_display"]
        self.assertEqual(display.outcome, "NOT_ATTEMPTED_PACKET_ERROR")
        self.assertContains(response, "NOT_ATTEMPTED_PACKET_ERROR")
        self.assertContains(response, "Packet error code")
        self.assertContains(response, display.packet_error_code)
        self.assertContains(response, "Provider call not attempted")
        self.assertContains(response, "Validator not run")
        self.assertFalse(display.transport_response_received)
        self.assertFalse(display.show_raw_output)
        self.assertFalse(display.show_provider_metadata)
        self.assertEqual(len(provider.calls), 0)
        self.assertNotContains(response, "Untrusted model output - not verified.")
        self._assert_claim_safety(response)
        self._assert_no_server_owned_leakage(response)
        self._assert_secret_absent(response)
        self._assert_no_result_persistence(response)

    def test_validation_rejected_renders_status_codes_and_escaped_output(self):
        provider = RecordingProvider(raw_text="{not-valid-json")
        _preview, response = self._confirm(provider)
        display = response.context["result_display"]
        self.assertEqual(display.outcome, "VALIDATION_REJECTED")
        self.assertContains(response, "VALIDATION_REJECTED")
        self.assertContains(
            response,
            "The provider responded; the deterministic validator rejected the response.",
        )
        self.assertContains(response, "Validation status REJECTED")
        self.assertContains(response, display.rejection_codes)
        self.assertTrue(display.rejection_codes)
        self.assertContains(response, "Untrusted model output - not verified.")
        self.assertContains(response, "{not-valid-json")
        self.assertNotContains(response, "integration failed")
        self.assertNotContains(response, "integration failure")
        self.assertTrue(display.transport_response_received)
        self.assertTrue(display.show_raw_output)
        self.assertTrue(display.show_validator)
        self.assertContains(response, "Provider-reported input tokens")
        self.assertContains(response, "Provider-reported output tokens")
        self.assertContains(response, MODEL_ID)
        self.assertContains(response, "COMPLETE")
        self._assert_claim_safety(response)
        self._assert_secret_absent(response)
        self._assert_no_result_persistence(response, raw_output='{"not":"valid"}')

    def test_passed_deterministic_checks_uses_claim_safe_wording(self):
        raw = _valid_pass_candidate()
        provider = RecordingProvider(raw_text=raw)
        _preview, response = self._confirm(provider)
        display = response.context["result_display"]
        self.assertEqual(display.outcome, "PASSED_DETERMINISTIC_CHECKS")
        self.assertContains(response, "PASSED_DETERMINISTIC_CHECKS")
        self.assertContains(response, "Passed deterministic grounding checks - not verified as correct.")
        self.assertContains(response, "This does not mean the output is accurate, factual or safe.")
        self.assertContains(response, "Untrusted model output - not verified.")
        self.assertContains(response, "Total trades is 6 [E1].")
        self.assertIs(response.context["review_required"], True)
        self.assertTrue(display.transport_response_received)
        self.assertContains(response, "Provider-reported input tokens")
        self.assertContains(response, "Duration ms")
        self._assert_claim_safety(response)
        self._assert_no_server_owned_leakage(response)
        self._assert_secret_absent(response)
        self._assert_no_result_persistence(response, raw_output=raw)

    def _assert_generate_failure(self, category: ProviderFailureCode) -> None:
        provider = RecordingProvider(
            failure=ProviderFailure(category=category, provider_name=PROVIDER_NAME)
        )
        _preview, response = self._confirm(provider)
        display = response.context["result_display"]
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(display.outcome, "PROVIDER_FAILED")
        self.assertEqual(display.provider_failure_category, category.value)
        self.assertContains(response, "PROVIDER_FAILED")
        self.assertContains(response, f"The provider request did not complete ({category.value}).")
        self.assertContains(response, "Integration was not proven for this attempt.")
        self.assertFalse(display.show_raw_output)
        self.assertFalse(display.show_validator)
        self.assertNotContains(response, "Untrusted model output - not verified.")
        self.assertNotContains(response, "Validation status")
        self.assertNotIn("Traceback", self._html(response))
        self._assert_claim_safety(response)
        self._assert_secret_absent(response)
        self._assert_no_result_persistence(response)

    def test_provider_not_configured_failure_display(self):
        failure = ProviderFailure(
            category=ProviderFailureCode.PROVIDER_NOT_CONFIGURED,
            provider_name=PROVIDER_NAME,
        )
        preview = self._preview()
        with patch("performance.provider_inspector.get_provider", return_value=failure):
            response = self.client.post(
                self.url,
                {
                    "confirm_send": "1",
                    "nonce": preview.context["nonce"],
                    "question_id": SPRINT9_PRESET_QUESTION_ID,
                },
            )
        display = response.context["result_display"]
        self.assertEqual(display.outcome, "PROVIDER_FAILED")
        self.assertEqual(display.provider_failure_category, "PROVIDER_NOT_CONFIGURED")
        self.assertFalse(display.transport_response_received)
        self.assertContains(response, "The provider request did not complete (PROVIDER_NOT_CONFIGURED).")
        self.assertEqual(len(RecordingProvider().calls), 0)
        self._assert_secret_absent(response)

    def test_provider_auth_failed_failure_display(self):
        self._assert_generate_failure(ProviderFailureCode.PROVIDER_AUTH_FAILED)

    def test_provider_rate_limited_failure_display(self):
        self._assert_generate_failure(ProviderFailureCode.PROVIDER_RATE_LIMITED)

    def test_provider_timeout_failure_display(self):
        self._assert_generate_failure(ProviderFailureCode.PROVIDER_TIMEOUT)

    def test_provider_response_invalid_failure_display(self):
        self._assert_generate_failure(ProviderFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_provider_error_failure_display(self):
        self._assert_generate_failure(ProviderFailureCode.PROVIDER_ERROR)

    def test_script_payload_is_escaped_and_not_safe(self):
        provider = RecordingProvider(raw_text=SCRIPT_PAYLOAD)
        _preview, response = self._confirm(provider)
        html = self._html(response)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)
        template = (PACKAGE_ROOT / "templates" / "performance" / "provider_grounding_inspector.html").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("|safe", template)
        self.assertContains(response, "Untrusted model output - not verified.")

    def test_unexpected_exception_is_contained(self):
        message = f"{SECRET_SENTINEL} {REQUEST_ID_SENTINEL} HTTP/1.1 500 body=secret-body"
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            preview = self._preview()
            with (
                patch("performance.provider_inspector.get_provider", return_value=RecordingProvider()),
                patch(
                    "performance.provider_inspector.generate_grounded_response",
                    side_effect=RuntimeError(message),
                ),
            ):
                response = self.client.post(
                    self.url,
                    {
                        "confirm_send": "1",
                        "nonce": preview.context["nonce"],
                        "question_id": SPRINT9_PRESET_QUESTION_ID,
                    },
                )
        finally:
            root.removeHandler(handler)
        html = self._html(response)
        self.assertEqual(response.context["inspector_state"], "WORKFLOW_EXCEPTION")
        self.assertContains(
            response,
            "The provider workflow could not complete. No diagnostic exception details are displayed.",
        )
        self.assertContains(response, "Human review required.")
        self.assertNotIn(message, html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn(REQUEST_ID_SENTINEL, html)
        self.assertNotIn("secret-body", html)
        self._assert_secret_absent(response, logs=stream.getvalue())
        self._assert_no_result_persistence(response)

    def test_secret_sentinel_absent_from_get_and_consent_refusal(self):
        get_response = self._preview()
        self.assertNotIn(SECRET_SENTINEL, self._html(get_response))
        refusal = self.client.post(
            self.url,
            {"nonce": get_response.context["nonce"], "question_id": SPRINT9_PRESET_QUESTION_ID},
        )
        self.assertEqual(refusal.context["inspector_state"], "CONSENT_REQUIRED")
        self.assertNotIn(SECRET_SENTINEL, self._html(refusal))

    def test_view_does_not_read_api_key_env_directly(self):
        blob = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("ANTHROPIC_API_KEY", blob)
        self.assertNotIn("API_KEY_ENV", blob)

    def test_import_boundary_unchanged(self):
        names = _imported_modules(MODULE_PATH)
        blob = MODULE_PATH.read_text(encoding="utf-8")
        for name in names:
            self.assertNotEqual(name.split(".", 1)[0], "anthropic")
            self.assertNotIn("ai_evaluation", name)
        self.assertNotIn("performance.views", names)
        self.assertNotIn("performance.models", names)
        self.assertNotIn("ai_evaluation", blob)
