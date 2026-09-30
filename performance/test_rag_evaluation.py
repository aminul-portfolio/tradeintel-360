from __future__ import annotations

from django.test import SimpleTestCase

from .rag.evaluation import (
    PRECISION_K,
    EvaluationCase,
    EvaluationError,
    build_synthetic_evaluation_cases,
    evaluate_case,
    evaluate_cases,
    evaluate_synthetic_baseline,
    first_relevant_rank,
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
)
from .rag.retrieval import (
    RETRIEVAL_METHOD_EXACT_TICKET,
    RETRIEVAL_METHOD_TFIDF_COSINE,
    RETRIEVAL_STATE_CORPUS_INVALID,
    RETRIEVAL_STATE_EMPTY_QUERY,
    RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
    RETRIEVAL_STATE_OK,
)
from .rag.schema import trade_document_id


class RagEvaluationTests(SimpleTestCase):
    def _summary(self):
        return evaluate_synthetic_baseline()

    def _by_id(self, summary, case_id):
        for item in summary.results:
            if item.case_id == case_id:
                return item
        self.fail(f"missing case {case_id}")

    def test_synthetic_case_count_and_unique_ids(self):
        cases = build_synthetic_evaluation_cases()
        ids = [item.case_id for item in cases]
        self.assertGreaterEqual(len(cases), 40)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(case_id.strip() for case_id in ids))

    def test_contract_pass_rate_is_complete(self):
        summary = self._summary()
        failed = [item.case_id for item in summary.results if not item.contract_passed]
        self.assertEqual(failed, [])
        self.assertEqual(summary.contract_cases_passed, summary.total_cases)
        self.assertEqual(summary.contract_pass_rate, 1.0)
        self.assertGreaterEqual(summary.total_cases, 40)

    def test_hit_at_1_hit_at_3_and_mrr_formulas(self):
        self.assertEqual(hit_at_k(1, 1), 1.0)
        self.assertEqual(hit_at_k(2, 1), 0.0)
        self.assertEqual(hit_at_k(3, 3), 1.0)
        self.assertEqual(hit_at_k(4, 3), 0.0)
        self.assertEqual(hit_at_k(None, 3), 0.0)
        self.assertEqual(mean_reciprocal_rank(1), 1.0)
        self.assertEqual(mean_reciprocal_rank(2), 0.5)
        self.assertEqual(mean_reciprocal_rank(None), 0.0)
        self.assertEqual(first_relevant_rank(("a", "b"), ("b",)), 2)
        self.assertIsNone(first_relevant_rank(("a",), ("z",)))
        kiwi = self._by_id(self._summary(), "LEX_KIWI")
        self.assertTrue(kiwi.eligible_single)
        self.assertEqual(kiwi.hit_at_1, 1.0)
        self.assertEqual(kiwi.hit_at_3, 1.0)
        self.assertEqual(kiwi.mrr, 1.0)

    def test_precision_at_k_and_multi_eligibility(self):
        self.assertEqual(PRECISION_K, 5)
        self.assertEqual(precision_at_k(("a", "b"), ("a", "c"), 5), 0.2)
        self.assertEqual(precision_at_k(("a", "b", "c"), ("a", "b"), 5), 0.4)
        summary = self._summary()
        multi = [item for item in summary.results if item.eligible_multi]
        single = [item for item in summary.results if item.eligible_single]
        self.assertGreaterEqual(len(multi), 6)
        self.assertGreaterEqual(len(single), 10)
        for item in multi:
            self.assertFalse(item.single_relevant)
            self.assertIsNotNone(item.precision_at_k)
            self.assertIsNone(item.hit_at_1)
        for item in single:
            self.assertTrue(item.single_relevant)
            self.assertIsNone(item.precision_at_k)
        commission = self._by_id(summary, "MULTI_COMMISSION")
        self.assertEqual(commission.precision_at_k, 2 / PRECISION_K)

    def test_exact_ticket_score_is_none(self):
        result = self._by_id(self._summary(), "EXACT_T1")
        self.assertEqual(result.actual_state, RETRIEVAL_STATE_OK)
        self.assertEqual(result.retrieval_methods, (RETRIEVAL_METHOD_EXACT_TICKET,))
        self.assertEqual(result.lexical_scores, (None,))
        self.assertEqual(result.retrieved_document_ids, (trade_document_id("1"),))

    def test_tfidf_scores_are_positive_finite_lexical_scores(self):
        result = self._by_id(self._summary(), "LEX_KIWI")
        self.assertEqual(result.retrieval_methods, (RETRIEVAL_METHOD_TFIDF_COSINE,))
        self.assertEqual(len(result.lexical_scores), 1)
        score = result.lexical_scores[0]
        self.assertIsNotNone(score)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)

    def test_corpus_invalid_audit_counts_are_zero(self):
        summary = self._summary()
        for case_id in ("INVALID_MIXED", "INVALID_WRONG_SCOPE"):
            result = self._by_id(summary, case_id)
            self.assertEqual(result.actual_state, RETRIEVAL_STATE_CORPUS_INVALID)
            self.assertEqual(result.corpus_size, 0)
            self.assertEqual(result.candidate_count, 0)
            self.assertEqual(result.result_count, 0)

    def test_no_relevant_and_empty_query_states(self):
        summary = self._summary()
        zero = self._by_id(summary, "ZERO_ZZZ")
        unknown_ticket = self._by_id(summary, "UNK_T99")
        empty = self._by_id(summary, "EMPTY")
        whitespace = self._by_id(summary, "WHITESPACE")
        self.assertEqual(zero.actual_state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(unknown_ticket.actual_state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(empty.actual_state, RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(whitespace.actual_state, RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(zero.result_count, 0)
        self.assertEqual(empty.result_count, 0)
        self.assertGreater(empty.corpus_size, 0)
        self.assertGreater(zero.candidate_count, 0)

    def test_candidate_result_and_fingerprint_invariants(self):
        summary = self._summary()
        fingerprint = build_synthetic_evaluation_cases()[0].scope.journal_fingerprint
        for item in summary.results:
            self.assertGreaterEqual(item.corpus_size, 0)
            self.assertGreaterEqual(item.candidate_count, 0)
            self.assertLessEqual(item.result_count, item.candidate_count)
            self.assertLessEqual(item.result_count, item.top_k)
            self.assertEqual(item.ranks, tuple(range(1, item.result_count + 1)))
            if item.actual_state == RETRIEVAL_STATE_OK:
                self.assertTrue(item.retrieved_document_ids)
            else:
                self.assertEqual(item.retrieved_document_ids, ())
        kiwi = self._by_id(summary, "LEX_KIWI")
        self.assertTrue(kiwi.retrieved_document_ids[0].startswith("TRADE_EVIDENCE:"))
        cases = {item.case_id: item for item in build_synthetic_evaluation_cases()}
        self.assertEqual(cases["LEX_KIWI"].scope.journal_fingerprint, fingerprint)

    def test_repeat_evaluation_is_identical(self):
        first = evaluate_synthetic_baseline().as_plain()
        second = evaluate_synthetic_baseline().as_plain()
        self.assertEqual(first, second)

    def test_serialised_output_excludes_identity_and_confidence_terms(self):
        payload = str(self._summary().as_plain()).lower()
        self.assertNotIn("owner_id", payload)
        self.assertNotIn("user_id", payload)
        self.assertNotIn("confidence", payload)
        self.assertNotIn("probability", payload)
        self.assertNotIn("certainty", payload)
        banned = (
            "true mfe",
            "true mae",
            "optimal exit",
            "missed profit",
            "recommendation",
        )
        for phrase in banned:
            self.assertNotIn(phrase, payload)

    def test_malformed_evaluation_case_is_rejected(self):
        valid = build_synthetic_evaluation_cases()[0]
        broken = EvaluationCase(
            case_id="",
            scope=valid.scope,
            documents=valid.documents,
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_OK,
            relevant_document_ids=valid.relevant_document_ids,
            single_relevant=True,
        )
        with self.assertRaises(EvaluationError) as raised:
            evaluate_case(broken)
        self.assertEqual(raised.exception.reason, "INVALID_CASE_ID")
        with self.assertRaises(EvaluationError) as duplicates:
            evaluate_cases((valid, valid))
        self.assertEqual(duplicates.exception.reason, "DUPLICATE_CASE_ID")
        with self.assertRaises(EvaluationError):
            evaluate_cases(())
        with self.assertRaises(EvaluationError) as wrong_flag:
            evaluate_case(
                EvaluationCase(
                    case_id="BAD_SINGLE",
                    scope=valid.scope,
                    documents=valid.documents,
                    query="kiwifruit",
                    expected_state=RETRIEVAL_STATE_OK,
                    relevant_document_ids=(),
                    single_relevant=True,
                )
            )
        self.assertEqual(wrong_flag.exception.reason, "INVALID_SINGLE_RELEVANT")
        with self.assertRaises(EvaluationError) as unknown:
            evaluate_case(
                EvaluationCase(
                    case_id="BAD_ID",
                    scope=valid.scope,
                    documents=valid.documents,
                    query="kiwifruit",
                    expected_state=RETRIEVAL_STATE_OK,
                    relevant_document_ids=("TRADE_EVIDENCE:ticket:missing",),
                    single_relevant=True,
                )
            )
        self.assertEqual(unknown.exception.reason, "UNKNOWN_RELEVANT_ID")
        with self.assertRaises(EvaluationError) as scope:
            evaluate_case(
                EvaluationCase(
                    case_id="BAD_SCOPE",
                    scope="nope",  # type: ignore[arg-type]
                    documents=valid.documents,
                    query="kiwifruit",
                    expected_state=RETRIEVAL_STATE_OK,
                    relevant_document_ids=valid.relevant_document_ids,
                    single_relevant=True,
                )
            )
        self.assertEqual(scope.exception.reason, "INVALID_SCOPE")

    def test_metric_values_remain_within_unit_interval(self):
        summary = self._summary()
        for value in (
            summary.contract_pass_rate,
            summary.hit_at_1,
            summary.hit_at_3,
            summary.mrr,
            summary.precision_at_k,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        for item in summary.results:
            for value in (item.hit_at_1, item.hit_at_3, item.mrr, item.precision_at_k):
                if value is None:
                    continue
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_numeric_tie_and_top_k_behaviour(self):
        summary = self._summary()
        tied = self._by_id(summary, "MULTI_TIELEX")
        self.assertEqual(
            tied.retrieved_document_ids[:2],
            (trade_document_id("9"), trade_document_id("10")),
        )
        capped = self._by_id(summary, "TOPK_3_BUY")
        self.assertEqual(capped.top_k, 3)
        self.assertLessEqual(capped.result_count, 3)
        self.assertEqual(capped.result_count, len(capped.retrieved_document_ids))
