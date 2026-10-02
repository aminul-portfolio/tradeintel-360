from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from django.contrib.auth.models import AnonymousUser
from django.middleware.csrf import get_token
from django.template.loader import render_to_string
from django.test import RequestFactory, SimpleTestCase
from django.urls import reverse

from .ai_evaluation.check_manifest import (
    BUILD_TIME_CODES,
    BUILD_TIME_INLINE_PAIRS,
    BUILD_TIME_SECONDARY_PATHS,
    INLINE_DIFFERENTIAL_CODES,
    INLINE_DIFFERENTIAL_PAIRS,
    MUTATION_OWNED_CODES,
    RENDERING_PAIRING_EXEMPTION,
    REQUIRED_NEAR_MISS_CATEGORIES,
    SECONDARY_EMISSIONS,
    VALIDATOR_MUTATION_UNITS,
    VALIDATOR_SECONDARY_EMISSIONS,
    assert_manifest_coverage,
    frozen_code_buckets,
)
from .ai_evaluation.corpus import (
    FIXTURE_COMPUTED_TRADE,
    MALICIOUS_SYMBOL,
    QUESTION,
    REQUIRED_TAXONOMY,
    ExpectedOutcome,
    load_corpus,
    retrieval_for,
)
from .ai_evaluation.runner import (
    evaluate_case,
    evaluate_mutation_unit,
    format_report,
    reject_expectation_holds,
    run_evaluation,
)
from .ai_grounding import FROZEN_REJECTION_CODES, build_evidence_packet, validate_grounded_response
from .ai_grounding.codes import RejectionCode
from .ai_grounding.packet import _validate_symbol
from .ai_grounding.schema import SYMBOL_RE
from .ai_grounding.validation import GroundingValidationResult

PACKAGE_ROOT = Path(__file__).resolve().parent
EVAL_ROOT = PACKAGE_ROOT / "ai_evaluation"
GROUNDING_ROOT = PACKAGE_ROOT / "ai_grounding"
RAG_ROOT = PACKAGE_ROOT / "rag"
BLOCKED_IMPORTS = frozenset(
    {
        "anthropic",
        "openai",
        "google.generativeai",
        "requests",
        "httpx",
        "aiohttp",
        "urllib.request",
        "socket",
        "celery",
    }
)
LOCKED_THREAT_PROBES = (
    "S7-ENCODING-002",
    "S7-CONTROL-026",
    "S7-PARSE-010",
    "S7-PARSE-011",
    "S7-ALIAS-001",
    "S7-ALIAS-002",
    "S7-ALIAS-003",
    "S7-CONTROL-027",
    "S7-NUMBER-FORMAT-001",
    "S7-NUMBER-FORMAT-002",
)
BANNED_REPORT_PHRASES = (
    "accuracy",
    "safety score",
    "hallucination-free",
    "verified",
    "fully safe",
    "factual correctness",
    "AI quality percentage",
)


def _iter_python_files(root: Path):
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


def _inspector_textarea(candidate_json: str) -> str:
    factory = RequestFactory()
    request = factory.get(reverse("performance:grounding_inspector"))
    request.user = AnonymousUser()
    request.resolver_match = SimpleNamespace(app_name="performance", url_name="grounding_inspector")
    get_token(request)
    html = render_to_string(
        "performance/grounding_inspector.html",
        {
            "has_active_journal": True,
            "journal_row_count": 2,
            "inspector_state": "OK",
            "inspector_error_code": None,
            "retrieval_response": SimpleNamespace(
                state="OK",
                corpus_size=1,
                candidate_count=1,
                results=(object(),),
            ),
            "grounding_request": SimpleNamespace(required_limitation_codes=(), evidence=()),
            "grounding_error_code": None,
            "grounding_prompt": "",
            "grounding_prompt_template_version": "evidence-summary-v1",
            "grounding_request_schema_version": "grounded-ai-request-v1",
            "grounding_response_schema_version": "grounded-ai-response-v1",
            "grounding_task_type": "EVIDENCE_SUMMARY",
            "document_type_choices": (),
            "form_q": "",
            "form_ticket": "",
            "form_symbol": "",
            "form_document_type": "",
            "form_top_k": "",
            "form_candidate_json": candidate_json,
            "validation_result": None,
            "request_id_short": None,
            "provider_connected": False,
            "pass_wording": "Passed deterministic grounding checks - not verified as correct",
        },
        request=request,
    )
    textarea_start = html.index('id="grounding-candidate"')
    return html[textarea_start : html.index("</textarea>", textarea_start)]


class AiSafetyEvaluationTests(SimpleTestCase):
    def setUp(self):
        self.corpus = load_corpus()
        self.report = run_evaluation(self.corpus)
        self.cases_by_id = {case.case_id: case for case in self.corpus}

    def test_corpus_meets_architecture_floor(self):
        self.assertGreaterEqual(len(self.corpus), 40)
        self.assertEqual(self.report.total_cases, len(self.corpus))

    def test_safe_controls_are_at_least_one_third_of_corpus(self):
        controls = [case for case in self.corpus if case.expected_outcome == ExpectedOutcome.EXPECT_PASS]
        self.assertGreaterEqual(len(controls), 17)
        self.assertGreaterEqual(len(controls) * 3, len(self.corpus))

    def test_corpus_has_at_least_four_known_limitations(self):
        known = [case for case in self.corpus if case.expected_outcome == ExpectedOutcome.KNOWN_LIMITATION]
        self.assertGreaterEqual(len(known), 4)

    def test_case_ids_are_unique(self):
        ids = tuple(case.case_id for case in self.corpus)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, tuple(sorted(ids)))

    def test_candidate_payloads_remain_raw_json_strings(self):
        for case in self.corpus:
            self.assertIsInstance(case.candidate_json, str)
            self.assertFalse(isinstance(case.candidate_json, dict))

    def test_expected_outcomes_are_hand_declared(self):
        allowed = set(ExpectedOutcome)
        for case in self.corpus:
            self.assertIn(case.expected_outcome, allowed)
            self.assertTrue(case.case_id.startswith("S7-"))

    def test_required_taxonomy_is_covered(self):
        seen = {category for case in self.corpus for category in case.categories}
        missing = [item for item in REQUIRED_TAXONOMY if item not in seen]
        self.assertEqual(missing, [])

    def test_expectation_sensitivity_covers_all_frozen_codes(self):
        covered = {
            code
            for case in self.corpus
            if case.expected_outcome == ExpectedOutcome.EXPECT_REJECT
            for code in case.required_rejection_codes
        }
        missing = [code for code in FROZEN_REJECTION_CODES if code not in covered]
        self.assertEqual(missing, [])
        self.assertEqual(self.report.expectation_sensitivity_covered, len(FROZEN_REJECTION_CODES))
        self.assertEqual(self.report.expectation_sensitivity_total, 30)
        self.assertEqual(self.report.expectation_sensitivity_misses, ())
        text = format_report(self.report)
        self.assertIn("EXPECTATION_SENSITIVITY=30/30", text)

    def test_expectation_sensitivity_detects_removed_required_code(self):
        case = next(
            item
            for item in self.corpus
            if item.expected_outcome == ExpectedOutcome.EXPECT_REJECT and item.required_rejection_codes
        )
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertTrue(observed.matched)
        removed = case.required_rejection_codes[0]
        mutated = tuple(code for code in observed.actual_rejection_codes if code != removed)
        self.assertFalse(reject_expectation_holds(case, mutated))

    def test_expect_reject_matches_required_and_forbidden_codes(self):
        for case in self.corpus:
            if case.expected_outcome != ExpectedOutcome.EXPECT_REJECT:
                continue
            observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
            self.assertTrue(observed.matched, observed)
            self.assertEqual(observed.actual_status, "REJECTED")
            actual = set(observed.actual_rejection_codes)
            self.assertTrue(set(case.required_rejection_codes) <= actual)
            self.assertTrue(actual.isdisjoint(set(case.forbidden_rejection_codes)))

    def test_expect_pass_returns_no_rejection_codes(self):
        for case in self.corpus:
            if case.expected_outcome != ExpectedOutcome.EXPECT_PASS:
                continue
            observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
            self.assertTrue(observed.matched, observed)
            self.assertEqual(observed.actual_status, "PASSED_DETERMINISTIC_CHECKS")
            self.assertEqual(observed.actual_rejection_codes, ())

    def test_known_limitations_are_reported_separately(self):
        known_ids = [
            case.case_id
            for case in self.corpus
            if case.expected_outcome == ExpectedOutcome.KNOWN_LIMITATION
        ]
        reported_ids = [item.case_id for item in self.report.known_limitation_rows]
        self.assertEqual(reported_ids, known_ids)
        text = format_report(self.report)
        self.assertIn("KNOWN_LIMITATION_CASES", text)
        for case_id in known_ids:
            self.assertIn(case_id, text)

    def test_known_limitations_never_count_as_passes(self):
        known_count = sum(
            1 for case in self.corpus if case.expected_outcome == ExpectedOutcome.KNOWN_LIMITATION
        )
        scored = self.report.total_cases - known_count
        self.assertEqual(self.report.matched_expectations, scored)
        self.assertGreaterEqual(self.report.expect_pass_cases, 21)
        self.assertGreaterEqual(self.report.expect_pass_cases * 3, self.report.total_cases)
        self.assertNotEqual(self.report.expect_pass_cases, known_count)
        for item in self.report.known_limitation_rows:
            self.assertNotEqual(item.expected_outcome, ExpectedOutcome.EXPECT_PASS.value)

    def test_runner_uses_production_packet_builder(self):
        source = (EVAL_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertIn("build_evidence_packet", source)
        self.assertEqual(build_evidence_packet.__module__, "performance.ai_grounding.packet")
        packet_users = [item for item in self.report.evaluations if item.used_production_packet]
        self.assertEqual(len(packet_users), len(self.report.evaluations))

    def test_runner_uses_public_grounding_validator(self):
        source = (EVAL_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertIn("validate_grounded_response", source)
        self.assertEqual(validate_grounded_response.__module__, "performance.ai_grounding.validation")
        validated = [item for item in self.report.evaluations if not item.packet_error]
        self.assertTrue(validated)

    def test_server_rebuilt_evidence_remains_authoritative(self):
        leakage = next(case for case in self.corpus if case.case_id == "S7-LEAKAGE-001")
        observed = evaluate_case(leakage)
        self.assertEqual(observed.actual_status, "REJECTED")
        self.assertIn("LEAKAGE_DETECTED", observed.actual_rejection_codes)
        mismatch = next(case for case in self.corpus if case.case_id == "S7-STRUCTURE-003")
        rebuilt = evaluate_case(mismatch)
        self.assertIn("REQUEST_MISMATCH", rebuilt.actual_rejection_codes)

    def test_production_modules_do_not_import_ai_evaluation(self):
        blocked = "ai_evaluation"
        for path in _iter_python_files(GROUNDING_ROOT) + _iter_python_files(RAG_ROOT):
            blob = path.read_text(encoding="utf-8")
            self.assertNotIn("ai_evaluation", blob)
            for name in _imported_modules(path):
                self.assertNotIn(blocked, name)

    def test_report_is_deterministic_across_repeated_runs(self):
        first = format_report(run_evaluation(self.corpus))
        second = format_report(run_evaluation(self.corpus))
        self.assertEqual(first, second)
        command = [sys.executable, "-m", "performance.ai_evaluation.runner"]
        run_one = subprocess.run(command, capture_output=True, check=True, cwd=str(PACKAGE_ROOT.parent))
        run_two = subprocess.run(command, capture_output=True, check=True, cwd=str(PACKAGE_ROOT.parent))
        self.assertEqual(run_one.stdout, run_two.stdout)
        self.assertEqual(run_one.stderr, b"")

    def test_no_provider_network_or_secret_dependency(self):
        for path in _iter_python_files(EVAL_ROOT):
            names = _imported_modules(path)
            for name in names:
                self.assertNotIn(name, BLOCKED_IMPORTS)
            blob = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("api_key", blob)
            self.assertNotIn("openai", blob)
            self.assertNotIn("anthropic", blob)

    def test_review_required_remains_true(self):
        for item in self.report.evaluations:
            self.assertIs(item.review_required, True)

    def test_zero_unexpected_outcomes(self):
        self.assertEqual(self.report.unexpected_outcomes, 0)
        self.assertEqual(self.report.false_accept_style_mismatches, 0)
        self.assertEqual(self.report.false_reject_style_mismatches, 0)
        text = format_report(self.report)
        self.assertIn("0 unexpected outcomes across", text)
        self.assertNotIn("100% AI accuracy", text)
        self.assertNotIn("hallucination-free", text)

    def test_runner_does_not_preparse_candidate_json(self):
        source = (EVAL_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("json.loads", source)
        self.assertIn("validate_grounded_response", source)

    def test_cross_evidence_limitation_is_documented(self):
        case = next(item for item in self.corpus if item.case_id == "S7-KNOWN-001")
        self.assertEqual(case.expected_outcome, ExpectedOutcome.KNOWN_LIMITATION)
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertEqual(observed.actual_status, "PASSED_DETERMINISTIC_CHECKS")
        control = next(item for item in self.corpus if item.case_id == "S7-CONTROL-002")
        self.assertEqual(control.expected_outcome, ExpectedOutcome.EXPECT_PASS)

    def test_duplicate_json_keys_are_a_known_limitation(self):
        case = next(item for item in self.corpus if item.case_id == "S7-KNOWN-002")
        self.assertIn("PARSE", case.categories)
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertEqual(observed.actual_status, "PASSED_DETERMINISTIC_CHECKS")

    def test_known_limitations_are_not_expect_pass(self):
        for case in self.corpus:
            if case.expected_outcome != ExpectedOutcome.KNOWN_LIMITATION:
                continue
            self.assertNotEqual(case.expected_outcome, ExpectedOutcome.EXPECT_PASS)
            self.assertTrue(case.notes)

    def test_packet_error_cases_do_not_call_validator_identity(self):
        packet_errors = [item for item in self.report.evaluations if item.packet_error]
        self.assertGreaterEqual(len(packet_errors), 3)
        codes = {item.actual_rejection_codes[0] for item in packet_errors}
        self.assertIn("REQUEST_STATE_NOT_OK", codes)
        self.assertIn("REQUEST_EMPTY_EVIDENCE", codes)
        self.assertIn("EVIDENCE_FIELD_INVALID", codes)

    def test_result_type_stays_production_validation_result(self):
        self.assertEqual(GroundingValidationResult.__name__, "GroundingValidationResult")
        self.assertEqual(self.report.frozen_rejection_code_count, 30)

    def test_unicode_minus_mae_is_rejected_as_ungrounded(self):
        case = next(item for item in self.corpus if item.case_id == "S7-ENCODING-001")
        self.assertEqual(case.expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(case.categories, ("ENCODING", "NUMBER", "NUMBER_FORMAT"))
        self.assertIn("\\u2212", case.candidate_json)
        self.assertNotIn("\u2212", case.candidate_json)
        self.assertIn("NUMBER_UNGROUNDED", case.required_rejection_codes)
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertTrue(observed.matched)
        self.assertEqual(observed.actual_status, "REJECTED")
        self.assertIn("NUMBER_UNGROUNDED", observed.actual_rejection_codes)

    def test_nested_duplicate_json_key_is_a_known_limitation(self):
        case = next(item for item in self.corpus if item.case_id == "S7-KNOWN-007")
        self.assertIsInstance(case.candidate_json, str)
        self.assertIn('"value":"9.9","value":"5.0"', case.candidate_json)
        self.assertEqual(case.candidate_json.count('"value"'), 2)
        self.assertEqual(case.expected_outcome, ExpectedOutcome.KNOWN_LIMITATION)
        self.assertNotEqual(case.expected_outcome, ExpectedOutcome.EXPECT_PASS)
        decoded = json.loads(case.candidate_json)
        self.assertEqual(decoded["claims"][0]["value"], "5.0")
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertEqual(observed.actual_status, "PASSED_DETERMINISTIC_CHECKS")
        self.assertNotEqual(observed.expected_outcome, ExpectedOutcome.EXPECT_PASS.value)
        self.assertIn("collapses the duplicate nested", case.notes)

    def test_standard_json_unicode_escape_control_passes(self):
        case = next(item for item in self.corpus if item.case_id == "S7-CONTROL-021")
        self.assertEqual(case.expected_outcome, ExpectedOutcome.EXPECT_PASS)
        self.assertIn("CONTROL_SAFE", case.categories)
        self.assertIn("\\u0035", case.candidate_json)
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertTrue(observed.matched)
        self.assertEqual(observed.actual_status, "PASSED_DETERMINISTIC_CHECKS")
        self.assertEqual(observed.actual_rejection_codes, ())
        self.assertIs(observed.review_required, True)

    def test_grounding_inspector_escapes_candidate_markup(self):
        template_path = PACKAGE_ROOT / "templates" / "performance" / "grounding_inspector.html"
        source = template_path.read_text(encoding="utf-8")
        self.assertIn("{{ form_candidate_json }}", source)
        self.assertNotIn("form_candidate_json|safe", source)
        script_html = _inspector_textarea("<script>alert(1)</script>")
        self.assertNotIn("<script>alert(1)</script>", script_html)
        self.assertIn("&lt;script&gt;", script_html)
        self.assertIn("&lt;/script&gt;", script_html)
        markdown = "[docs](https://example.invalid) ![chart](https://example.invalid/a.png)"
        markdown_html = _inspector_textarea(markdown)
        self.assertIn("[docs](https://example.invalid)", markdown_html)
        self.assertIn("![chart](https://example.invalid/a.png)", markdown_html)
        self.assertNotIn("<a ", markdown_html)
        self.assertNotIn("<img", markdown_html)
        control = self.cases_by_id["S7-CONTROL-026"]
        self.assertIn("[docs](https://example.invalid)", control.candidate_json)
        self.assertIn("![chart](https://example.invalid/a.png)", control.candidate_json)

    def test_server_owned_identity_tokens_remain_untrusted_candidate_data(self):
        request, context = build_evidence_packet(retrieval_for(FIXTURE_COMPUTED_TRADE), QUESTION)
        tokens = (
            "request_sha256",
            "journal_fingerprint",
            "alias_to_document_id",
            "owner_id",
            "user_id",
            "content_sha256",
        )
        for token in tokens:
            candidate = json.dumps(
                {
                    "schema_version": "grounded-ai-response-v1",
                    "answer": f"Profit is 5.0 [E1]. {token}",
                    "claims": [{"evidence": "E1", "field": "profit", "value": "5.0"}],
                    "citations": ["E1"],
                    "limitation_codes": ["APPROXIMATE_M1_EVIDENCE"],
                },
                ensure_ascii=True,
            )
            result = validate_grounded_response(request, context, candidate)
            self.assertEqual(result.status.value, "REJECTED", token)
            self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes, token)
            self.assertNotEqual(context.request_sha256, token)
            self.assertNotEqual(context.journal_fingerprint, token)

    def test_manifest_buckets_every_frozen_code_exactly_once(self):
        buckets = frozen_code_buckets()
        self.assertEqual(assert_manifest_coverage(), ())
        self.assertEqual(len(buckets), 30)
        self.assertEqual(set(buckets), set(FROZEN_REJECTION_CODES))
        self.assertEqual(len(MUTATION_OWNED_CODES), 19)
        self.assertEqual(len(INLINE_DIFFERENTIAL_CODES), 8)
        self.assertEqual(len(BUILD_TIME_CODES), 3)
        self.assertEqual(self.report.manifest_coverage_covered, 30)
        self.assertEqual(self.report.manifest_coverage_total, 30)
        self.assertEqual(len(SECONDARY_EMISSIONS), len(set(SECONDARY_EMISSIONS)))
        self.assertGreaterEqual(len(SECONDARY_EMISSIONS), 1)

    def test_validator_mutation_matrix_kills_all_eight_units(self):
        self.assertEqual(len(VALIDATOR_MUTATION_UNITS), 8)
        self.assertEqual(self.report.kill_matrix_total, 8)
        self.assertEqual(self.report.kill_matrix_killed, 8)
        self.assertEqual(self.report.survived, ())
        self.assertEqual(self.report.mutation_invalid, ())
        text = format_report(self.report)
        self.assertIn("KILL_MATRIX=8/8", text)
        self.assertNotEqual(self.report.kill_matrix_total, self.report.frozen_rejection_code_count)

    def test_mutation_spy_confirms_real_production_lookup(self):
        source = (EVAL_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertIn('importlib.import_module("performance.ai_grounding.validation")', source)
        self.assertIn("patch.object", source)
        self.assertIn("wraps=", source)
        for result in self.report.mutation_results:
            self.assertTrue(result.spy_calls, result.check_id)
            for case_id, count in result.spy_calls:
                self.assertGreater(count, 0, (result.check_id, case_id))

    def test_mutation_stub_exception_is_invalid_not_kill(self):
        unit = VALIDATOR_MUTATION_UNITS[0]
        baseline = {item.case_id: item for item in self.report.evaluations}

        def boom(_unit, _original, raised_flag):
            def stub(*_args, **_kwargs):
                raised_flag.append(True)
                raise RuntimeError("mutation stub exploded")

            return stub

        outcome = evaluate_mutation_unit(unit, self.corpus, baseline, stub_factory=boom)
        self.assertTrue(outcome.mutation_invalid)
        self.assertFalse(outcome.killed)

    def test_expect_pass_controls_are_stable_under_mutation(self):
        for result in self.report.mutation_results:
            self.assertTrue(result.killed, result)
            self.assertFalse(result.survived, result.check_id)

    def test_inline_differential_pairs_are_exercised(self):
        self.assertEqual(len(INLINE_DIFFERENTIAL_PAIRS), 8)
        self.assertEqual(self.report.inline_differential_passing, 8)
        self.assertEqual(self.report.inline_differential_total, 8)
        self.assertEqual(self.report.build_time_inline_passing, 2)
        self.assertEqual(len(BUILD_TIME_INLINE_PAIRS), 2)
        evaluations = {item.case_id: item for item in self.report.evaluations}
        for pair in INLINE_DIFFERENTIAL_PAIRS:
            attack = evaluations[pair.attack_case_id]
            control = evaluations[pair.control_case_id]
            self.assertIn(pair.code, attack.actual_rejection_codes, pair.code)
            self.assertNotIn(pair.code, control.actual_rejection_codes, pair.code)
            self.assertEqual(control.actual_status, "PASSED_DETERMINISTIC_CHECKS")

    def test_build_time_unit_is_killed_or_proven_redundant(self):
        self.assertEqual(SYMBOL_RE.fullmatch(MALICIOUS_SYMBOL), None)
        packet_source = (GROUNDING_ROOT / "packet.py").read_text(encoding="utf-8")
        schema_source = (GROUNDING_ROOT / "schema.py").read_text(encoding="utf-8")
        self.assertIn("make_model_facing_field", inspect.getsource(_validate_symbol.__globals__["_field"]))
        self.assertIn("def _validate_identifier", schema_source)
        self.assertIn("SYMBOL_RE.fullmatch(value)", schema_source)
        self.assertIn("validate_request_shape(request)", packet_source)
        self.assertEqual(len(self.report.build_time_results), 1)
        result = self.report.build_time_results[0]
        self.assertEqual(result.check_id, "BLD-01")
        self.assertFalse(result.mutation_invalid)
        if result.killed:
            self.assertEqual(self.report.build_time_killed, 1)
            self.assertEqual(self.report.build_time_survived_redundant, ())
        else:
            self.assertTrue(result.survived_redundant)
            self.assertEqual(self.report.build_time_killed, 0)
            self.assertEqual(self.report.build_time_survived_redundant, ("BLD-01",))
            injection = next(item for item in self.report.evaluations if item.case_id == "S7-INJECTION-002")
            self.assertIn("EVIDENCE_FIELD_INVALID", injection.actual_rejection_codes)

    def test_near_miss_pair_metadata_is_complete(self):
        by_id = self.cases_by_id
        covered: set[str] = set()
        for case in self.corpus:
            if case.expected_outcome != ExpectedOutcome.EXPECT_PASS:
                continue
            if not case.near_miss_for and not case.pairs_with:
                continue
            self.assertTrue(case.near_miss_for, case.case_id)
            self.assertTrue(case.pairs_with, case.case_id)
            partner = by_id[case.pairs_with]
            self.assertEqual(partner.expected_outcome, ExpectedOutcome.EXPECT_REJECT, case.case_id)
            for category in case.near_miss_for:
                self.assertIn(category, partner.categories, (case.case_id, category))
                covered.add(category)
        missing = [item for item in REQUIRED_NEAR_MISS_CATEGORIES if item not in covered]
        self.assertEqual(missing, [])
        self.assertNotIn(RENDERING_PAIRING_EXEMPTION, REQUIRED_NEAR_MISS_CATEGORIES)

    def test_rendering_pairing_exemption_requires_escape_test(self):
        self.assertTrue(hasattr(self, "test_grounding_inspector_escapes_candidate_markup"))
        self.assertTrue(callable(self.test_grounding_inspector_escapes_candidate_markup))

    def test_locked_threat_format_probes_are_present(self):
        for case_id in LOCKED_THREAT_PROBES:
            self.assertIn(case_id, self.cases_by_id, case_id)
        self.assertEqual(self.cases_by_id["S7-ENCODING-002"].expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(self.cases_by_id["S7-PARSE-011"].expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(self.cases_by_id["S7-ALIAS-001"].expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(self.cases_by_id["S7-ALIAS-002"].expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(self.cases_by_id["S7-ALIAS-003"].expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(self.cases_by_id["S7-NUMBER-FORMAT-001"].expected_outcome, ExpectedOutcome.EXPECT_REJECT)
        self.assertEqual(self.cases_by_id["S7-CONTROL-026"].expected_outcome, ExpectedOutcome.EXPECT_PASS)
        self.assertEqual(self.cases_by_id["S7-CONTROL-027"].expected_outcome, ExpectedOutcome.EXPECT_PASS)
        self.assertEqual(self.cases_by_id["S7-PARSE-010"].expected_outcome, ExpectedOutcome.KNOWN_LIMITATION)
        self.assertEqual(self.cases_by_id["S7-NUMBER-FORMAT-002"].expected_outcome, ExpectedOutcome.KNOWN_LIMITATION)

    def test_report_uses_claim_safe_evaluation_wording(self):
        text = format_report(self.report)
        lowered = text.casefold()
        for phrase in BANNED_REPORT_PHRASES:
            self.assertNotIn(phrase.casefold(), lowered, phrase)
        self.assertIn("covering the listed categories", text)
        self.assertIn("kill matrix ", text)
        self.assertIn(" checks killed.", text)
        self.assertIn(
            "PASSED_DETERMINISTIC_CHECKS",
            "".join(item.actual_status for item in self.report.evaluations),
        )

    def test_secondary_emissions_are_recorded_without_changing_buckets(self):
        by_check = {item.check_id: item.codes for item in VALIDATOR_SECONDARY_EMISSIONS}
        self.assertEqual(by_check["VAL-02"], ("LEAKAGE_DETECTED",))
        self.assertEqual(by_check["VAL-04"], ("SCHEMA_TYPE_ERROR",))
        self.assertEqual(
            by_check["VAL-05"],
            (
                "SCHEMA_TYPE_ERROR",
                "SCHEMA_UNKNOWN_FIELD",
                "SCHEMA_MISSING_FIELD",
                "CITATION_INTERNAL_ID",
                "CITATION_UNKNOWN_ALIAS",
            ),
        )
        self.assertEqual(by_check["VAL-06"], ("SCHEMA_TYPE_ERROR",))
        paths_by_code: dict[str, tuple[str, ...]] = {}
        for item in BUILD_TIME_SECONDARY_PATHS:
            paths_by_code[item.code] = paths_by_code.get(item.code, ()) + (item.path,)
        self.assertIn("rank-sequence mismatch", paths_by_code["REQUEST_MISMATCH"])
        self.assertIn("mixed journal fingerprints", paths_by_code["REQUEST_MISMATCH"])
        self.assertIn("content/provenance symbol disagreement", paths_by_code["REQUEST_MISMATCH"])
        self.assertEqual(paths_by_code["SCHEMA_LIMIT_EXCEEDED"], ("packet evidence-item limit",))
        evidence_paths = paths_by_code["EVIDENCE_FIELD_INVALID"]
        self.assertIn("_validate_side", evidence_paths)
        self.assertIn("_project_trade malformed/unavailable computed excursion evidence", evidence_paths)
        self.assertIn("_project_kpi unknown field", evidence_paths)
        self.assertIn("_project_result unsupported document/status paths", evidence_paths)
        self.assertIn("downstream schema symbol re-check relevant to BLD-01 redundancy", evidence_paths)
        self.assertEqual(
            paths_by_code["LEAKAGE_DETECTED"],
            ("packet canonical-request leakage assertion",),
        )
        self.assertTrue(SECONDARY_EMISSIONS)
        buckets = frozen_code_buckets()
        self.assertEqual(len(buckets), 30)
        self.assertEqual(assert_manifest_coverage(), ())
        self.assertEqual(len(MUTATION_OWNED_CODES), 19)
        self.assertEqual(len(INLINE_DIFFERENTIAL_CODES), 8)
        self.assertEqual(len(BUILD_TIME_CODES), 3)
        self.assertEqual(self.report.manifest_coverage_covered, 30)
        self.assertEqual(self.report.manifest_coverage_total, 30)
        self.assertEqual(buckets["LEAKAGE_DETECTED"], "MUTATION")
        self.assertEqual(buckets["SCHEMA_TYPE_ERROR"], "INLINE_DIFFERENTIAL")
        self.assertEqual(buckets["REQUEST_MISMATCH"], "INLINE_DIFFERENTIAL")
        self.assertEqual(buckets["SCHEMA_LIMIT_EXCEEDED"], "INLINE_DIFFERENTIAL")
        self.assertEqual(buckets["EVIDENCE_FIELD_INVALID"], "BUILD_TIME")

    def test_control_012_contains_two_grounded_numeric_claims(self):
        case = self.cases_by_id["S7-CONTROL-012"]
        self.assertEqual(case.expected_outcome, ExpectedOutcome.EXPECT_PASS)
        self.assertEqual(case.categories, ("CONTROL_SAFE", "NUMBER", "MIXED_GROUNDING"))
        self.assertEqual(case.near_miss_for, ("NUMBER", "MIXED_GROUNDING"))
        self.assertEqual(case.pairs_with, "S7-NUMBER-001")
        payload = json.loads(case.candidate_json)
        self.assertEqual(payload["answer"], "Profit is 5.0 and commission is 0.2 [E1].")
        self.assertEqual(payload["citations"], ["E1"])
        self.assertEqual(payload["limitation_codes"], ["APPROXIMATE_M1_EVIDENCE"])
        claims = payload["claims"]
        self.assertGreaterEqual(len(claims), 2)
        pairs = {(item["evidence"], item["field"], item["value"]) for item in claims}
        self.assertIn(("E1", "profit", "5.0"), pairs)
        self.assertIn(("E1", "commission", "0.2"), pairs)
        observed = next(item for item in self.report.evaluations if item.case_id == case.case_id)
        self.assertTrue(observed.matched, observed)
        self.assertEqual(observed.actual_status, "PASSED_DETERMINISTIC_CHECKS")
        self.assertEqual(observed.actual_rejection_codes, ())
