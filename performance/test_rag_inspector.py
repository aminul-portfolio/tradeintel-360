from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .rag.retrieval import (
    RETRIEVAL_METHOD_EXACT_TICKET,
    RETRIEVAL_METHOD_TFIDF_COSINE,
    RETRIEVAL_STATE_CORPUS_INVALID,
    RETRIEVAL_STATE_EMPTY_QUERY,
    RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
    RETRIEVAL_STATE_OK,
)
from .rag.schema import (
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    CorpusError,
)
from .views import INSPECTOR_STATE_NO_ACTIVE_JOURNAL


class RagInspectorTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="inspector-user",
            password="test-password-123",
        )
        self.client.force_login(self.user)
        self.url = reverse("performance:retrieval_inspector")
        self.dashboard_url = reverse("performance:dashboard")

    def _journal(self):
        return pd.DataFrame(
            {
                "Ticket": [1, 2],
                "Open Time": [
                    "25 Jun 2026 10:30:15",
                    "25 Jun 2026 10:31:20",
                ],
                "Close Time": [
                    "25 Jun 2026 10:32:15",
                    "25 Jun 2026 10:33:20",
                ],
                "Symbol": ["NZDUSD", "EURUSD"],
                "Type": ["buy", "buy"],
                "Entry": [0.62, 1.20],
                "Exit": [0.63, 1.10],
                "Profit": [10.0, 5.0],
                "Commission": [0.2, 0.2],
                "Swap": [0.0, 0.0],
                "Pips": [10.0, 10.0],
            }
        )

    def _store_journal(self, frame=None):
        journal = frame if frame is not None else self._journal()
        session = self.client.session
        session["cleaned_data"] = journal.to_json(orient="split", date_format="iso")
        session.save()

    def _run(self, **params):
        payload = {"run": "1"}
        payload.update(params)
        return self.client.get(self.url, payload)

    def test_anonymous_access_redirects_to_login(self):
        self.client.logout()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_authenticated_user_can_open_inspector(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Retrieval Inspector")
        self.assertContains(response, "Private diagnostic view of deterministic evidence retrieval.")

    def test_no_active_journal_renders_distinct_state(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["inspector_state"],
            INSPECTOR_STATE_NO_ACTIVE_JOURNAL,
        )
        self.assertIsNone(response.context["retrieval_response"])
        self.assertContains(response, "NO_ACTIVE_JOURNAL")
        self.assertContains(
            response,
            "Load or upload an active journal before retrieval can run.",
        )

    def test_initial_get_does_not_execute_retrieval(self):
        self._store_journal()
        response = self.client.get(self.url, {"q": "buy"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["inspector_state"])
        self.assertIsNone(response.context["retrieval_response"])
        self.assertFalse(response.context["run_requested"])
        self.assertNotContains(response, "EMPTY_QUERY")
        self.assertContains(response, "Loaded (2 rows)")

    def test_exact_ticket_uses_real_corpus_and_retrieval_stack(self):
        self._store_journal()
        response = self._run(ticket="1")
        payload = response.context["retrieval_response"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["inspector_state"], RETRIEVAL_STATE_OK)
        self.assertEqual(payload.state, RETRIEVAL_STATE_OK)
        self.assertEqual(len(payload.results), 1)
        self.assertEqual(payload.results[0].retrieval_method, RETRIEVAL_METHOD_EXACT_TICKET)
        self.assertIsNone(payload.results[0].score)
        self.assertEqual(payload.results[0].source_record_key, "1")
        self.assertGreater(payload.corpus_size, 0)
        self.assertEqual(payload.candidate_count, 1)
        self.assertContains(response, "Not applicable - exact ticket match")
        self.assertContains(response, "corpus_size")
        self.assertContains(response, "candidate_count")

    def test_lexical_query_uses_tfidf_and_safe_score_wording(self):
        self._store_journal()
        response = self._run(q="nzdusd")
        payload = response.context["retrieval_response"]
        self.assertEqual(payload.state, RETRIEVAL_STATE_OK)
        self.assertEqual(payload.results[0].retrieval_method, RETRIEVAL_METHOD_TFIDF_COSINE)
        self.assertIsNotNone(payload.results[0].score)
        self.assertGreater(payload.results[0].score, 0)
        self.assertContains(response, "Lexical similarity score")
        self.assertContains(
            response,
            "Lexical similarity is a deterministic retrieval ranking value.",
        )
        self.assertContains(response, "It is not a confidence or probability score.")

    def test_symbol_filter_uses_structured_provenance(self):
        self._store_journal()
        unfiltered = self._run(q="buy")
        self.assertGreaterEqual(len(unfiltered.context["retrieval_response"].results), 2)
        response = self._run(q="buy", symbol="nzdusd")
        payload = response.context["retrieval_response"]
        self.assertEqual(payload.state, RETRIEVAL_STATE_OK)
        self.assertEqual(len(payload.results), 1)
        self.assertEqual(payload.results[0].source_record_key, "1")
        unmatched = self._run(q="nzdusd", symbol="EURUSD")
        self.assertEqual(
            unmatched.context["inspector_state"],
            RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
        )

    def test_document_type_filter_reaches_retrieval(self):
        self._store_journal()
        trades = self._run(q="nzdusd", document_type=DOCUMENT_TYPE_TRADE)
        self.assertEqual(trades.context["retrieval_response"].state, RETRIEVAL_STATE_OK)
        self.assertEqual(
            trades.context["retrieval_response"].results[0].document_type,
            DOCUMENT_TYPE_TRADE,
        )
        kpis = self._run(q="nzdusd", document_type=DOCUMENT_TYPE_KPI)
        self.assertEqual(
            kpis.context["inspector_state"],
            RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
        )

    def test_top_k_get_conversion_and_audit_counts(self):
        self._store_journal()
        response = self._run(q="ticket", top_k="1")
        payload = response.context["retrieval_response"]
        self.assertEqual(payload.top_k, 1)
        self.assertLessEqual(len(payload.results), 1)
        self.assertLessEqual(payload.candidate_count, payload.corpus_size)
        self.assertLessEqual(len(payload.results), payload.candidate_count)
        self.assertContains(response, "top_k")
        self.assertContains(response, str(payload.corpus_size))
        self.assertContains(response, str(payload.candidate_count))

    def test_empty_query_state_is_distinct(self):
        self._store_journal()
        response = self._run()
        self.assertEqual(response.context["inspector_state"], RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(len(response.context["retrieval_response"].results), 0)
        self.assertContains(response, "EMPTY_QUERY")
        self.assertContains(
            response,
            "No usable lexical query or explicit ticket was provided.",
        )
        self.assertNotContains(response, "NO_RELEVANT_EVIDENCE")
        whitespace = self._run(q="   \n")
        self.assertEqual(whitespace.context["inspector_state"], RETRIEVAL_STATE_EMPTY_QUERY)

    def test_no_relevant_evidence_state_is_distinct(self):
        self._store_journal()
        response = self._run(q="zzzznotpresent")
        self.assertEqual(
            response.context["inspector_state"],
            RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
        )
        self.assertContains(response, "NO_RELEVANT_EVIDENCE")
        self.assertContains(
            response,
            "No evidence matched the deterministic retrieval request.",
        )
        self.assertNotContains(response, "EMPTY_QUERY")

    def test_corpus_invalid_renders_safely(self):
        self._store_journal()
        with patch(
            "performance.rag.retrieval.validate_evidence_corpus",
            side_effect=CorpusError("MIXED_JOURNAL_CORPUS"),
        ):
            response = self._run(q="nzdusd")
        payload = response.context["retrieval_response"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["inspector_state"], RETRIEVAL_STATE_CORPUS_INVALID)
        self.assertEqual(payload.corpus_size, 0)
        self.assertEqual(payload.candidate_count, 0)
        self.assertEqual(payload.results, ())
        self.assertContains(response, "CORPUS_INVALID")
        self.assertNotContains(response, "MIXED_JOURNAL_CORPUS")
        html = response.content.decode()
        self.assertNotIn("Traceback", html)

    def test_invalid_top_k_is_handled_without_http_500(self):
        self._store_journal()
        response = self._run(q="nzdusd", top_k="not-a-number")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["inspector_error_code"], "INVALID_TOP_K")
        self.assertIsNone(response.context["retrieval_response"])
        self.assertContains(response, "INVALID_TOP_K")
        invalid_type = self._run(q="nzdusd", document_type="NOT_A_TYPE")
        self.assertEqual(invalid_type.status_code, 200)
        self.assertEqual(invalid_type.context["inspector_error_code"], "INVALID_DOCUMENT_TYPE")

    def test_rendered_html_excludes_identity_and_local_paths(self):
        self._store_journal()
        response = self._run(ticket="1")
        html = response.content.decode()
        lowered = html.lower()
        self.assertNotIn("owner_id", lowered)
        self.assertNotIn("user_id", lowered)
        self.assertNotIn("g:\\final_polish", lowered)
        self.assertNotIn("g:\\workflow_tools", lowered)
        self.assertNotIn("workflow_tools", lowered)
        self.assertNotIn("true mfe", lowered)
        self.assertNotIn("optimal exit", lowered)
        self.assertNotIn("missed profit", lowered)

    def test_dashboard_contains_retrieval_inspector_link(self):
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Retrieval Inspector")
        self.assertContains(response, self.url)
        self.assertNotContains(response, "AI Assistant")
        self.assertNotContains(response, "Recommendation Engine")

    def test_post_is_rejected(self):
        response = self.client.post(self.url, {"run": "1", "q": "nzdusd"})
        self.assertEqual(response.status_code, 405)
