from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .ai_grounding.codes import ValidationStatus
from .ai_grounding.prompt import prompt_hash, prompt_template_hash, render_evidence_summary_prompt
from .rag.retrieval import RETRIEVAL_STATE_EMPTY_QUERY, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE
from .test_ai_grounding_prompt import LOCKED_PROMPT_HASH, LOCKED_PROMPT_TEMPLATE_HASH, _golden_request
from .views import INSPECTOR_PASS_WORDING, INSPECTOR_STATE_NO_ACTIVE_JOURNAL


class GroundingInspectorTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="grounding-inspector-user",
            password="test-password-123",
        )
        self.client.force_login(self.user)
        self.url = reverse("performance:grounding_inspector")

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
        payload = {"run": "1", "q": "profit"}
        payload.update(params)
        return self.client.get(self.url, payload)

    def _candidate_json(self, grounded):
        chosen_item = None
        chosen_field = None
        for item in grounded.evidence:
            for field_key in item.claimable_field_keys:
                field = next(entry for entry in item.fields if entry.field_key == field_key)
                chosen_item = item
                chosen_field = field
                if field_key == "profit":
                    break
            else:
                continue
            break
        if chosen_item is None or chosen_field is None:
            raise AssertionError("grounding request has no claimable field")
        payload = {
            "schema_version": "grounded-ai-response-v1",
            "answer": f"{chosen_field.label} is {chosen_field.canonical_value} [{chosen_item.alias}].",
            "claims": [
                {
                    "evidence": chosen_item.alias,
                    "field": chosen_field.field_key,
                    "value": chosen_field.canonical_value,
                }
            ],
            "citations": [chosen_item.alias],
            "limitation_codes": [code.value for code in grounded.required_limitation_codes],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _post(self, candidate_json, **params):
        payload = {
            "q": "profit",
            "ticket": "",
            "symbol": "",
            "document_type": "",
            "top_k": "",
            "candidate_json": candidate_json,
        }
        payload.update(params)
        return self.client.post(self.url, payload)

    def test_anonymous_access_redirects_to_login(self):
        self.client.logout()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)
        posted = self.client.post(self.url, {"q": "profit", "candidate_json": "{}"})
        self.assertEqual(posted.status_code, 302)
        self.assertIn("/login/", posted.url)

    def test_authentication_required_for_authenticated_user(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Grounding Inspector")

    def test_active_authenticated_journal_required(self):
        response = self.client.get(self.url)
        self.assertEqual(response.context["inspector_state"], INSPECTOR_STATE_NO_ACTIVE_JOURNAL)
        self.assertIsNone(response.context["grounding_request"])
        self.assertContains(response, "NO_ACTIVE_JOURNAL")

    def test_retrieval_non_ok_does_not_build_grounding_request(self):
        self._store_journal()
        response = self._run(q="")
        self.assertEqual(response.context["inspector_state"], RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertIsNone(response.context["grounding_request"])
        self.assertIsNone(response.context["grounding_prompt"])
        self.assertIsNone(response.context["validation_result"])

    def test_empty_evidence_does_not_build_grounding_request(self):
        self._store_journal()
        response = self._run(q="zzzznotpresent")
        self.assertEqual(
            response.context["inspector_state"],
            RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
        )
        self.assertIsNone(response.context["grounding_request"])
        self.assertIsNone(response.context["validation_result"])

    def test_get_does_not_call_any_provider(self):
        self._store_journal()
        response = self._run()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["provider_connected"])
        source = Path(__file__).resolve().parent.joinpath("views.py").read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertNotIn("openai", lowered)
        self.assertNotIn("anthropic", lowered)
        self.assertNotIn("api_key", lowered)

    def test_post_does_not_call_any_provider(self):
        self._store_journal()
        built = self._run()
        response = self._post(self._candidate_json(built.context["grounding_request"]))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["provider_connected"])
        self.assertContains(response, "Provider: Not connected")

    def test_page_states_provider_not_connected(self):
        response = self.client.get(self.url)
        self.assertContains(response, "Provider: Not connected")

    def test_page_states_no_llm_is_called(self):
        response = self.client.get(self.url)
        self.assertContains(response, "No LLM is called in this version.")

    def test_page_states_manual_candidate_wording(self):
        self._store_journal()
        response = self._run()
        self.assertContains(response, "Manually supplied test candidate - not AI-generated")

    def test_request_generated_server_side(self):
        self._store_journal()
        response = self._run()
        grounded = response.context["grounding_request"]
        self.assertIsNotNone(grounded)
        self.assertEqual(grounded.question, "profit")
        self.assertTrue(grounded.evidence)
        self.assertIsNone(response.context.get("grounding_context"))
        self.assertIsNotNone(response.context["request_id_short"])
        self.assertEqual(len(response.context["request_id_short"]), 12)

    def test_candidate_json_manually_accepted(self):
        self._store_journal()
        built = self._run()
        candidate = self._candidate_json(built.context["grounding_request"])
        response = self._post(candidate)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["form_candidate_json"], candidate)

    def test_post_rebuilds_request_server_side(self):
        self._store_journal()
        built = self._run()
        first_prompt = built.context["grounding_prompt"]
        response = self._post(self._candidate_json(built.context["grounding_request"]))
        self.assertIsNotNone(response.context["grounding_request"])
        self.assertEqual(response.context["grounding_prompt"], first_prompt)
        self.assertIsNotNone(response.context["validation_result"])

    def test_post_does_not_trust_request_sha256(self):
        self._store_journal()
        built = self._run()
        candidate = self._candidate_json(built.context["grounding_request"])
        response = self._post(candidate, request_sha256="a" * 64)
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_post_does_not_trust_journal_fingerprint(self):
        self._store_journal()
        built = self._run()
        candidate = self._candidate_json(built.context["grounding_request"])
        response = self._post(candidate, journal_fingerprint="c" * 64)
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_post_does_not_trust_alias_mapping_or_internal_ids(self):
        self._store_journal()
        built = self._run()
        candidate = self._candidate_json(built.context["grounding_request"])
        response = self._post(
            candidate,
            alias_to_document_id="E1=TRADE_EVIDENCE:ticket:99",
            document_id="TRADE_EVIDENCE:ticket:99",
            source_record_key="99",
            owner_id="someone-else",
            user_id="someone-else",
            corpus_size="1",
            candidate_count="1",
            returned_count="1",
        )
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_candidate_with_review_required_rejected(self):
        self._store_journal()
        built = self._run()
        payload = json.loads(self._candidate_json(built.context["grounding_request"]))
        payload["review_required"] = False
        response = self._post(json.dumps(payload))
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        values = tuple(code.value for code in result.rejection_codes)
        self.assertIn("SCHEMA_UNKNOWN_FIELD", values)
        self.assertContains(response, "REJECTED")
        self.assertContains(response, "SCHEMA_UNKNOWN_FIELD")

    def test_valid_candidate_produces_passed_deterministic_checks(self):
        self._store_journal()
        built = self._run()
        response = self._post(self._candidate_json(built.context["grounding_request"]))
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        self.assertEqual(result.status.value, "PASSED_DETERMINISTIC_CHECKS")
        self.assertContains(response, "PASSED_DETERMINISTIC_CHECKS")
        self.assertContains(response, INSPECTOR_PASS_WORDING)
        self.assertContains(
            response,
            "Passed deterministic grounding checks - not verified as correct",
        )
        self.assertNotContains(response, "Verified")
        self.assertNotContains(response, "Fact checked")
        self.assertNotContains(response, "AI-generated response")

    def test_review_required_true_shown(self):
        self._store_journal()
        built = self._run()
        response = self._post(self._candidate_json(built.context["grounding_request"]))
        self.assertIs(response.context["validation_result"].review_required, True)
        self.assertContains(response, "review_required")
        self.assertContains(response, "True")
        self.assertContains(response, "Human review is required.")

    def test_invalid_candidate_produces_rejected_codes_in_order(self):
        self._store_journal()
        built = self._run()
        payload = json.loads(self._candidate_json(built.context["grounding_request"]))
        payload["answer"] = f"{payload['answer']} you should buy."
        payload["review_required"] = True
        response = self._post(json.dumps(payload))
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        values = tuple(code.value for code in result.rejection_codes)
        self.assertEqual(values, tuple(sorted(values)))
        self.assertGreaterEqual(len(values), 2)
        html = response.content.decode()
        self.assertLess(html.index(values[0]), html.index(values[-1]))
        self.assertContains(response, "REJECTED")

    def test_required_limitation_codes_are_displayed(self):
        self._store_journal()
        response = self._run()
        grounded = response.context["grounding_request"]
        self.assertIsNotNone(grounded)
        if grounded.required_limitation_codes:
            for code in grounded.required_limitation_codes:
                self.assertContains(response, code.value)
        else:
            self.assertContains(response, "Required limitation codes")

    def test_provider_remains_disconnected(self):
        self._store_journal()
        built = self._run()
        response = self._post(self._candidate_json(built.context["grounding_request"]))
        self.assertFalse(response.context["provider_connected"])
        self.assertContains(response, "Provider: Not connected")
        self.assertContains(response, "No LLM is called in this version.")

    def test_prompt_version_displayed_and_rendering_is_deterministic(self):
        self._store_journal()
        first = self._run()
        second = self._run()
        self.assertContains(first, "evidence-summary-v1")
        self.assertEqual(first.context["grounding_prompt"], second.context["grounding_prompt"])
        self.assertEqual(
            first.context["grounding_prompt"],
            render_evidence_summary_prompt(first.context["grounding_request"]),
        )

    def test_server_only_metadata_absent_from_model_facing_display(self):
        self._store_journal()
        response = self._run()
        html = response.content.decode().lower()
        self.assertNotIn("owner_id", html)
        self.assertNotIn("user_id", html)
        self.assertNotIn("journal_fingerprint", html)
        self.assertNotIn("content_sha256", html)
        self.assertNotIn("alias_to_document_id", html)
        self.assertNotIn("g:\\final_polish", html)
        self.assertNotIn("workflow_tools", html)
        self.assertNotIn("api_key", html)
        prompt = response.context["grounding_prompt"]
        self.assertNotIn("owner_id", prompt)
        self.assertNotIn("journal_fingerprint", prompt)
        self.assertNotIn("TRADE_EVIDENCE:ticket:", prompt)

    def test_no_persistence_of_validation_result(self):
        self._store_journal()
        built = self._run()
        posted = self._post(self._candidate_json(built.context["grounding_request"]))
        self.assertIsNotNone(posted.context["validation_result"])
        later = self.client.get(self.url)
        self.assertIsNone(later.context["validation_result"])
        self.assertIsNone(later.context["grounding_request"])

    def test_prompt_hashes_remain_locked(self):
        self.assertEqual(prompt_template_hash(), LOCKED_PROMPT_TEMPLATE_HASH)
        self.assertEqual(prompt_hash(render_evidence_summary_prompt(_golden_request())), LOCKED_PROMPT_HASH)

    def test_no_provider_imports_in_inspector_view(self):
        blocked = {"anthropic", "openai", "requests", "httpx", "aiohttp", "urllib.request"}
        tree = ast.parse(Path(__file__).resolve().parent.joinpath("views.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            for name in names:
                self.assertNotIn(name, blocked)

    def test_initial_get_does_not_build_grounding_request(self):
        self._store_journal()
        response = self.client.get(self.url, {"q": "profit"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["run_requested"])
        self.assertIsNone(response.context["grounding_request"])
        self.assertIsNone(response.context["validation_result"])

    def test_get_does_not_validate_candidate(self):
        self._store_journal()
        built = self._run()
        response = self.client.get(
            self.url,
            {
                "run": "1",
                "q": "profit",
                "candidate_json": self._candidate_json(built.context["grounding_request"]),
            },
        )
        self.assertIsNotNone(response.context["grounding_request"])
        self.assertIsNone(response.context["validation_result"])

    def test_empty_candidate_is_rejected_after_server_rebuild(self):
        self._store_journal()
        response = self._post("")
        self.assertIsNotNone(response.context["grounding_request"])
        result = response.context["validation_result"]
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertEqual(
            tuple(code.value for code in result.rejection_codes),
            ("SCHEMA_INVALID_JSON",),
        )
        self.assertContains(response, "REJECTED")
        self.assertContains(response, "SCHEMA_INVALID_JSON")

    def test_hidden_form_omits_server_owned_identity(self):
        self._store_journal()
        response = self._run()
        html = response.content.decode()
        self.assertNotIn('name="request_sha256"', html)
        self.assertNotIn('name="journal_fingerprint"', html)
        self.assertNotIn('name="alias_to_document_id"', html)
        self.assertNotIn('name="document_id"', html)
        self.assertNotIn('name="owner_id"', html)
        self.assertNotIn('name="user_id"', html)
        self.assertIn('name="q"', html)
        self.assertIn('name="candidate_json"', html)

    def test_sprint5_retrieval_inspector_post_still_rejected(self):
        retrieval_url = reverse("performance:retrieval_inspector")
        response = self.client.post(retrieval_url, {"run": "1", "q": "profit"})
        self.assertEqual(response.status_code, 405)

    def test_no_model_persistence_on_validation(self):
        from .models import TradingFile

        before = TradingFile.objects.count()
        self._store_journal()
        built = self._run()
        posted = self._post(self._candidate_json(built.context["grounding_request"]))
        self.assertEqual(posted.context["validation_result"].status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        self.assertEqual(TradingFile.objects.count(), before)
        session_keys = set(self.client.session.keys())
        self.assertNotIn("request_sha256", session_keys)
        self.assertNotIn("candidate_json", session_keys)
        self.assertNotIn("grounding_prompt", session_keys)
        self.assertNotIn("validation_result", session_keys)
