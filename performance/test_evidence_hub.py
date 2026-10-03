from __future__ import annotations

import ast
import os
import socket
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

HUB_PATH = Path(__file__).resolve().with_name("evidence_hub.py")
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "performance" / "evidence_hub.html"

PROHIBITED_CLAIMS = (
    "hallucination-free",
    "verified correct",
    "fact-checked",
    "accurate AI",
    "safe AI",
    "guaranteed correctness",
    "production-grade",
    "production-ready",
    "natural-language analytics",
    "AI-powered trading analytics",
    "trading signals",
    "predictions",
    "autonomous trading",
    "trade execution",
)


class _NetworkGuard:
    def __enter__(self) -> _NetworkGuard:
        self._socket = socket.socket
        self._create_connection = socket.create_connection

        def blocked(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("external network communication was attempted")

        socket.socket = blocked  # type: ignore[method-assign]
        socket.create_connection = blocked  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: object) -> None:
        socket.socket = self._socket  # type: ignore[method-assign]
        socket.create_connection = self._create_connection  # type: ignore[method-assign]


class EvidenceHubTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(
            username="evidence-user",
            password="test-password-123",
        )
        self.staff = User.objects.create_user(
            username="evidence-staff",
            password="test-password-123",
            is_staff=True,
        )
        self.url = reverse("performance:engineering_evidence")
        self.analytical_url = reverse("performance:analytical_query_inspector")
        self.retrieval_url = reverse("performance:retrieval_inspector")
        self.grounding_url = reverse("performance:grounding_inspector")
        self.provider_url = reverse("performance:provider_grounding_inspector")

    def test_anonymous_get_redirects_to_login(self) -> None:
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_authenticated_user_receives_200(self) -> None:
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Engineering Evidence")
        self.assertContains(response, "What each engineering surface demonstrates and what it does not.")

    def test_anonymous_pages_do_not_expose_nav_link(self) -> None:
        home = self.client.get(reverse("core:home"))
        login = self.client.get(reverse("login"))
        self.assertEqual(home.status_code, 200)
        self.assertEqual(login.status_code, 200)
        self.assertNotContains(home, self.url)
        self.assertNotContains(home, "Engineering Evidence")
        self.assertNotContains(login, self.url)
        self.assertNotContains(login, "Engineering Evidence")

    def test_authenticated_nav_exposes_evidence_link(self) -> None:
        self.client.force_login(self.user)
        home = self.client.get(reverse("core:home"))
        self.assertContains(home, self.url)
        self.assertContains(home, "Engineering Evidence")

    def test_deterministic_inspector_links_reverse_and_render(self) -> None:
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertContains(response, self.analytical_url)
        self.assertContains(response, self.retrieval_url)
        self.assertContains(response, self.grounding_url)
        self.assertTrue(self.analytical_url)
        self.assertTrue(self.retrieval_url)
        self.assertTrue(self.grounding_url)

    def test_staff_sees_provider_inspector_link(self) -> None:
        self.client.force_login(self.staff)
        response = self.client.get(self.url)
        self.assertContains(response, self.provider_url)
        self.assertContains(
            response,
            "Open inspector (staff only; opening the page does not send a provider request)",
        )

    def test_non_staff_sees_staff_only_without_provider_link(self) -> None:
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertNotContains(response, self.provider_url)
        self.assertContains(response, "Staff only")
        self.assertNotContains(response, "Try it")
        self.assertNotContains(response, "Run AI")
        self.assertNotContains(response, "Generate")
        self.assertNotContains(response, "Ask AI")

    def test_evidence_get_makes_zero_provider_generation_calls(self) -> None:
        self.client.force_login(self.staff)
        with patch("performance.provider_inspector.generate_grounded_response") as mocked:
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        mocked.assert_not_called()

    def test_staff_provider_inspector_get_does_not_generate(self) -> None:
        self.client.force_login(self.staff)
        sentinel = "sentinel-test-key"
        with patch.dict(
            os.environ,
            {
                "TRADEINTEL_LLM_ENABLED": "1",
                "ANTHROPIC_API_KEY": sentinel,
            },
        ):
            self.assertEqual(os.environ.get("TRADEINTEL_LLM_ENABLED"), "1")
            self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), sentinel)
            with patch("performance.provider_inspector.generate_grounded_response") as mocked:
                response = self.client.get(self.provider_url)
            mocked.assert_not_called()
            self.assertEqual(mocked.call_count, 0)
        self.assertNotEqual(os.environ.get("ANTHROPIC_API_KEY"), sentinel)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["provider_enabled"])

    def test_evidence_hub_ast_has_no_provider_or_rag_imports(self) -> None:
        tree = ast.parse(HUB_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported.append(module)
                imported.extend(alias.name for alias in node.names)
        banned = (
            "performance.ai_provider",
            "performance.ai_orchestration",
            "performance.ai_grounding",
            "performance.ai_evaluation",
            "performance.rag",
            "ai_provider",
            "ai_orchestration",
            "ai_grounding",
            "ai_evaluation",
            "rag",
        )
        for name in imported:
            self.assertNotIn(name, banned, name)
            self.assertNotIn("ai_", name)
            self.assertNotIn("rag", name.lower())

    def test_empty_session_unchanged(self) -> None:
        self.client.force_login(self.user)
        before = dict(self.client.session.items())
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(dict(self.client.session.items()), before)

    def test_cleaned_data_session_unchanged(self) -> None:
        self.client.force_login(self.user)
        session = self.client.session
        session["cleaned_data"] = '{"columns":[],"data":[],"index":[]}'
        session["last_uploaded_file"] = "history.csv"
        session.save()
        before = dict(self.client.session.items())
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(dict(self.client.session.items()), before)
        self.assertEqual(self.client.session.get("cleaned_data"), before["cleaned_data"])
        self.assertEqual(self.client.session.get("last_uploaded_file"), "history.csv")

    def test_deterministic_surfaces_do_not_open_network(self) -> None:
        self.client.force_login(self.user)
        targets = (self.url, self.analytical_url, self.retrieval_url, self.grounding_url)
        for target in targets:
            with _NetworkGuard():
                response = self.client.get(target)
            self.assertEqual(response.status_code, 200, target)

    def test_existing_pages_still_render_after_nav_change(self) -> None:
        home = self.client.get(reverse("core:home"))
        self.assertEqual(home.status_code, 200)
        self.client.force_login(self.user)
        upload = self.client.get(reverse("performance:upload_file"))
        dashboard = self.client.get(reverse("performance:dashboard"))
        kpi = self.client.get(reverse("performance:kpi_report"))
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(dashboard.status_code, 200)
        self.assertEqual(kpi.status_code, 200)
        self.assertContains(dashboard, "Engineering Evidence")
        self.assertContains(kpi, "Engineering Evidence")

    def test_required_claim_boundary_wording(self) -> None:
        self.client.force_login(self.user)
        response = self.client.get(self.url)
        self.assertContains(response, "Deterministic means reproducible computation and deterministic checks.")
        self.assertContains(response, "It does not establish that source journal data or model output is factually")
        self.assertContains(response, "Provider output remains subject to human review.")
        self.assertContains(response, "SCHEMA_INVALID_JSON")
        self.assertContains(response, "does not mean the output is factually verified")
        self.assertContains(response, "Commission and Swap are excluded")
        self.assertContains(response, "no timezone conversion")
        body = response.content.decode().casefold()
        for claim in PROHIBITED_CLAIMS:
            self.assertNotIn(claim.casefold(), body, claim)

    def test_template_has_no_unsafe_markup_or_prefetch(self) -> None:
        source = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("|safe", source)
        self.assertNotIn("rel=\"prefetch\"", source)
        self.assertNotIn("prefetch", source)
        self.assertNotIn("prerender", source)
        self.assertNotIn("<script", source)
        self.assertNotIn("javascript:", source.lower())
