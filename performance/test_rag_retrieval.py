from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from .rag.corpus import build_evidence_corpus
from .rag.retrieval import (
    DEFAULT_TOP_K,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    RETRIEVAL_METHOD_EXACT_TICKET,
    RETRIEVAL_METHOD_TFIDF_COSINE,
    RETRIEVAL_STATE_CORPUS_INVALID,
    RETRIEVAL_STATE_EMPTY_QUERY,
    RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
    RETRIEVAL_STATE_OK,
    RetrievalError,
    retrieve_evidence,
    tokenize,
)
from .rag.schema import (
    DATASET_DOCUMENT_ID,
    DATASET_SOURCE_RECORD_KEY,
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    EVIDENCE_STATUS_NO_M1,
    kpi_document_id,
    make_evidence_document,
    trade_document_id,
)
from .rag.scope import RetrievalScope, resolve_retrieval_scope
from .utils import compute_kpis


class RagRetrievalTests(SimpleTestCase):
    def _user(self, pk=42):
        return SimpleNamespace(is_authenticated=True, pk=pk)

    def _journal(self):
        return pd.DataFrame(
            [
                {
                    "Ticket": 2,
                    "Open Time": "25 Jun 2026 10:30:15",
                    "Close Time": "25 Jun 2026 10:31:20",
                    "Symbol": "EURUSD",
                    "Type": "sell",
                    "Entry": 1.20,
                    "Exit": 1.10,
                    "Profit": 5.0,
                    "Commission": 0.2,
                    "Swap": 0.0,
                    "Pips": 10.0,
                },
                {
                    "Ticket": 1,
                    "Open Time": "25 Jun 2026 10:30:15",
                    "Close Time": "25 Jun 2026 10:31:20",
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 105.0,
                    "Profit": 10.0,
                    "Commission": 1.0,
                    "Swap": 0.1,
                    "Pips": 5.0,
                },
            ]
        )

    def _build_live(self, journal=None, pk=42):
        frame = journal if journal is not None else self._journal()
        scope = resolve_retrieval_scope(self._user(pk), frame)
        documents = build_evidence_corpus(
            scope,
            frame,
            kpi_mapping=compute_kpis(frame),
        )
        return scope, documents

    def _synthetic(
        self,
        *,
        fingerprint="a" * 64,
        trades=None,
        kpis=None,
        dataset_content="Journal row count: 2",
    ):
        if trades is None:
            trades = (
                ("1", "Ticket: 1\nSymbol: US100.cash\nType: buy\nShared: alpha", "US100.cash"),
                ("2", "Ticket: 2\nSymbol: EURUSD\nType: sell\nShared: alpha", "EURUSD"),
            )
        documents = []
        for spec in trades:
            ticket, content, *rest = spec
            symbol = rest[0] if rest else None
            documents.append(
                make_evidence_document(
                    document_id=trade_document_id(ticket),
                    document_type=DOCUMENT_TYPE_TRADE,
                    journal_fingerprint=fingerprint,
                    source_record_key=ticket,
                    content=content,
                    evidence_status=EVIDENCE_STATUS_NO_M1,
                    provenance={"source_record_key": ticket, "symbol": symbol},
                )
            )
        if kpis:
            for key, content in kpis:
                documents.append(
                    make_evidence_document(
                        document_id=kpi_document_id(key),
                        document_type=DOCUMENT_TYPE_KPI,
                        journal_fingerprint=fingerprint,
                        source_record_key=key,
                        content=content,
                        evidence_status="AUTHORITATIVE",
                        provenance={"source_record_key": key},
                    )
                )
        documents.append(
            make_evidence_document(
                document_id=DATASET_DOCUMENT_ID,
                document_type=DOCUMENT_TYPE_DATASET,
                journal_fingerprint=fingerprint,
                source_record_key=DATASET_SOURCE_RECORD_KEY,
                content=dataset_content,
                evidence_status="BOUND",
                provenance={"source_record_key": DATASET_SOURCE_RECORD_KEY},
            )
        )
        scope = RetrievalScope(owner_id=42, journal_fingerprint=fingerprint)
        return scope, tuple(documents)

    def test_tokenize_case_punctuation_and_signed_numbers(self):
        self.assertEqual(tokenize("EURUSD"), tokenize("eurusd"))
        self.assertEqual(tokenize("Hello, World!"), ("hello", "world"))
        self.assertEqual(tokenize("123456"), ("123456",))
        self.assertEqual(tokenize("-54.7"), ("-54.7",))
        self.assertEqual(tokenize("70.85"), ("70.85",))
        self.assertEqual(tokenize("MFE -54.7 pts"), ("mfe", "-54.7", "pts"))

    def test_exact_ticket_retrieval_score_none_and_rank(self):
        scope, documents = self._build_live()
        response = retrieve_evidence(documents, scope, ticket="1")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertEqual(len(response.results), 1)
        result = response.results[0]
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.source_record_key, "1")
        self.assertEqual(result.document_type, DOCUMENT_TYPE_TRADE)
        self.assertEqual(result.retrieval_method, RETRIEVAL_METHOD_EXACT_TICKET)
        self.assertIsNone(result.score)
        self.assertEqual(result.document_id, trade_document_id("1"))

    def test_free_text_ticket_does_not_use_exact_path(self):
        scope, documents = self._build_live()
        response = retrieve_evidence(documents, scope, query="show ticket 123")
        self.assertNotEqual(
            {item.retrieval_method for item in response.results},
            {RETRIEVAL_METHOD_EXACT_TICKET},
        )
        if response.state == RETRIEVAL_STATE_OK:
            for item in response.results:
                self.assertEqual(item.retrieval_method, RETRIEVAL_METHOD_TFIDF_COSINE)
                self.assertIsNotNone(item.score)

    def test_document_type_filter_returns_only_requested_type(self):
        scope, documents = self._synthetic(
            kpis=(("win_rate", "win_rate: 0.5 alpha"),),
            dataset_content="Journal row count: 2 alpha",
        )
        response = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            document_type=DOCUMENT_TYPE_KPI,
        )
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertTrue(response.results)
        self.assertEqual(
            {item.document_type for item in response.results},
            {DOCUMENT_TYPE_KPI},
        )

    def test_unknown_document_type_is_rejected(self):
        scope, documents = self._synthetic()
        with self.assertRaises(RetrievalError) as raised:
            retrieve_evidence(
                documents,
                scope,
                query="alpha",
                document_type="MARKET_EVIDENCE",
            )
        self.assertEqual(raised.exception.reason, "INVALID_DOCUMENT_TYPE")

    def test_symbol_filter_is_case_insensitive_and_trade_only(self):
        scope, documents = self._synthetic(
            kpis=(("win_rate", "win_rate: EURUSD"),),
            dataset_content="Journal row count: 2 EURUSD",
        )
        response = retrieve_evidence(documents, scope, query="alpha", symbol="eurusd")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].source_record_key, "2")
        self.assertEqual(response.results[0].document_type, DOCUMENT_TYPE_TRADE)

    def test_filters_apply_before_ranking(self):
        scope, documents = self._synthetic(
            trades=(
                ("1", "Ticket: 1\nSymbol: US100.cash\nUniqueBuy: zebra", "US100.cash"),
                ("2", "Ticket: 2\nSymbol: EURUSD\nUniqueSell: alpha", "EURUSD"),
            )
        )
        response = retrieve_evidence(
            documents,
            scope,
            query="zebra",
            symbol="EURUSD",
        )
        self.assertEqual(response.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(response.results, ())

    def test_empty_and_whitespace_query_without_ticket(self):
        scope, documents = self._synthetic()
        empty = retrieve_evidence(documents, scope, query="")
        whitespace = retrieve_evidence(documents, scope, query="   \n\t  ")
        self.assertEqual(empty.state, RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(whitespace.state, RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(empty.results, ())
        self.assertEqual(whitespace.results, ())
        self.assertEqual(empty.query, "")
        self.assertEqual(whitespace.query, "")

    def test_unknown_ticket_and_symbol_have_no_relevant_evidence(self):
        scope, documents = self._synthetic()
        unknown_ticket = retrieve_evidence(documents, scope, ticket="99")
        unknown_symbol = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            symbol="GBPUSD",
        )
        self.assertEqual(unknown_ticket.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(unknown_symbol.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(unknown_ticket.results, ())
        self.assertEqual(unknown_symbol.results, ())

    def test_zero_similarity_query_excludes_zero_score_documents(self):
        scope, documents = self._synthetic()
        response = retrieve_evidence(documents, scope, query="zzzznotpresent")
        self.assertEqual(response.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(response.results, ())

    def test_default_top_k_and_sequential_ranks(self):
        trades = tuple(
            (str(index), f"Ticket: {index}\nSymbol: EURUSD\nShared: alpha")
            for index in range(1, 9)
        )
        scope, documents = self._synthetic(trades=trades)
        response = retrieve_evidence(documents, scope, query="alpha")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertEqual(response.top_k, DEFAULT_TOP_K)
        self.assertEqual(len(response.results), DEFAULT_TOP_K)
        self.assertEqual([item.rank for item in response.results], [1, 2, 3, 4, 5])
        self.assertEqual(response.corpus_size, 9)
        self.assertEqual(response.candidate_count, 9)
        self.assertLessEqual(len(response.results), response.candidate_count)

    def test_max_top_k_is_clamped(self):
        trades = tuple(
            (str(index), f"Ticket: {index}\nSymbol: EURUSD\nShared: alpha")
            for index in range(1, 13)
        )
        scope, documents = self._synthetic(trades=trades)
        response = retrieve_evidence(documents, scope, query="alpha", top_k=100)
        self.assertEqual(response.top_k, MAX_TOP_K)
        self.assertEqual(len(response.results), MAX_TOP_K)
        self.assertEqual(
            [item.rank for item in response.results],
            list(range(1, MAX_TOP_K + 1)),
        )

    def test_non_positive_top_k_is_rejected(self):
        scope, documents = self._synthetic()
        for value in (0, -1, True, 2.5, "5"):
            with self.assertRaises(RetrievalError) as raised:
                retrieve_evidence(documents, scope, query="alpha", top_k=value)
            self.assertEqual(raised.exception.reason, "INVALID_TOP_K")

    def test_type_priority_and_numeric_ticket_tie_break(self):
        fingerprint = "b" * 64
        shared = "shared lexical token"
        scope, documents = self._synthetic(
            fingerprint=fingerprint,
            trades=(
                ("10", shared),
                ("9", shared),
            ),
            kpis=(("win_rate", shared),),
            dataset_content=shared,
        )
        response = retrieve_evidence(documents, scope, query="lexical", top_k=10)
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        ordered = [
            (item.document_type, item.source_record_key) for item in response.results
        ]
        self.assertEqual(
            ordered,
            [
                (DOCUMENT_TYPE_TRADE, "9"),
                (DOCUMENT_TYPE_TRADE, "10"),
                (DOCUMENT_TYPE_KPI, "win_rate"),
                (DOCUMENT_TYPE_DATASET, DATASET_SOURCE_RECORD_KEY),
            ],
        )
        scores = [item.score for item in response.results]
        self.assertEqual(len(set(scores)), 1)

    def test_repeat_run_determinism(self):
        scope, documents = self._synthetic(
            kpis=(("win_rate", "alpha kpi"),),
            dataset_content="alpha dataset",
        )
        first = retrieve_evidence(documents, scope, query="alpha", top_k=10)
        second = retrieve_evidence(documents, scope, query="alpha", top_k=10)
        self.assertEqual(first.as_plain(), second.as_plain())
        self.assertEqual(first.corpus_size, second.corpus_size)
        self.assertEqual(first.candidate_count, second.candidate_count)
        self.assertIn("corpus_size", first.as_plain())
        self.assertIn("candidate_count", first.as_plain())

    def test_scope_fingerprint_is_asserted_after_retrieval(self):
        scope, documents = self._synthetic()
        poisoned = replace(documents[0], journal_fingerprint="0" * 64)
        mixed = (poisoned, *documents[1:])
        with patch(
            "performance.rag.retrieval.validate_evidence_corpus",
            return_value=mixed,
        ):
            with self.assertRaises(RetrievalError) as raised:
                retrieve_evidence(mixed, scope, query="alpha")
        self.assertEqual(raised.exception.reason, "SCOPE_VIOLATION")

    def test_invalid_corpus_returns_corpus_invalid_without_partial_results(self):
        scope, documents = self._synthetic()
        mixed = (
            replace(documents[0], journal_fingerprint="0" * 64),
            *documents[1:],
        )
        wrong_scope = RetrievalScope(owner_id=1, journal_fingerprint="0" * 64)
        mixed_response = retrieve_evidence(mixed, scope, query="alpha")
        wrong_response = retrieve_evidence(documents, wrong_scope, query="alpha")
        self.assertEqual(mixed_response.state, RETRIEVAL_STATE_CORPUS_INVALID)
        self.assertEqual(wrong_response.state, RETRIEVAL_STATE_CORPUS_INVALID)
        self.assertEqual(mixed_response.results, ())
        self.assertEqual(wrong_response.results, ())

    def test_owner_id_is_absent_from_retrieval_results(self):
        scope, documents = self._build_live(pk=987654)
        response = retrieve_evidence(documents, scope, ticket="1")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        payload = str(response.as_plain())
        self.assertNotIn("987654", payload)
        self.assertNotIn("owner_id", payload)
        self.assertNotIn("user_id", payload)
        for item in response.results:
            self.assertNotIn("owner_id", item.provenance)
            self.assertNotIn("user_id", item.provenance)

    def test_tfidf_scores_are_finite_and_positive(self):
        scope, documents = self._synthetic()
        response = retrieve_evidence(documents, scope, query="alpha")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        for item in response.results:
            self.assertEqual(item.retrieval_method, RETRIEVAL_METHOD_TFIDF_COSINE)
            self.assertIsNotNone(item.score)
            self.assertTrue(math.isfinite(item.score))
            self.assertGreater(item.score, 0)
            self.assertLessEqual(item.score, 1)

    def test_evidence_document_and_provenance_are_not_mutated(self):
        scope, documents = self._synthetic()
        original = documents[0]
        original_content = original.content
        original_status = original.evidence_status
        nested = original.provenance
        response = retrieve_evidence(documents, scope, query="alpha")
        self.assertIs(documents[0], original)
        self.assertEqual(original.content, original_content)
        self.assertEqual(original.evidence_status, original_status)
        self.assertIs(original.provenance, nested)
        result = response.results[0]
        with self.assertRaises(TypeError):
            result.provenance["injected"] = 1
        with self.assertRaises(TypeError):
            original.provenance["injected"] = 1

    def test_ticket_may_be_retrieved_with_empty_query(self):
        scope, documents = self._synthetic()
        response = retrieve_evidence(documents, scope, query="", ticket="2")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertEqual(response.results[0].source_record_key, "2")
        self.assertIsNone(response.results[0].score)

    def test_oversized_query_is_rejected(self):
        scope, documents = self._synthetic()
        with self.assertRaises(RetrievalError) as raised:
            retrieve_evidence(
                documents,
                scope,
                query="a" * (MAX_QUERY_LENGTH + 1),
            )
        self.assertEqual(raised.exception.reason, "INVALID_QUERY")

    def test_tfidf_prefers_higher_term_frequency(self):
        scope, documents = self._synthetic(
            trades=(
                ("1", "alpha alpha alpha uniqueone"),
                ("2", "alpha uniquetwo"),
            )
        )
        response = retrieve_evidence(documents, scope, query="alpha")
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertGreaterEqual(len(response.results), 2)
        self.assertEqual(response.results[0].source_record_key, "1")
        self.assertGreater(response.results[0].score, response.results[1].score)

    def test_symbol_filter_uses_provenance_not_content(self):
        scope, documents = self._synthetic(
            trades=(
                (
                    "1",
                    "Ticket: 1\nSymbol: EURUSD\nShared: alpha",
                    "US100.cash",
                ),
                (
                    "2",
                    "Ticket: 2\nSymbol: EURUSD\nShared: alpha",
                    None,
                ),
            )
        )
        matched = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            symbol="us100.cash",
        )
        self.assertEqual(matched.state, RETRIEVAL_STATE_OK)
        self.assertEqual(len(matched.results), 1)
        self.assertEqual(matched.results[0].source_record_key, "1")
        self.assertEqual(matched.results[0].provenance["symbol"], "US100.cash")
        self.assertIn("Symbol: EURUSD", matched.results[0].content)
        conflicting = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            symbol="EURUSD",
        )
        self.assertEqual(conflicting.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(conflicting.results, ())
        omitted_scope, omitted_docs = self._synthetic(
            trades=(
                ("1", "Ticket: 1\nType: buy\nShared: alpha", "US100.cash"),
                ("2", "Ticket: 2\nSymbol: EURUSD\nShared: alpha", None),
            )
        )
        omitted_match = retrieve_evidence(
            omitted_docs,
            omitted_scope,
            query="alpha",
            symbol="us100.cash",
        )
        self.assertEqual(omitted_match.state, RETRIEVAL_STATE_OK)
        self.assertEqual(omitted_match.results[0].source_record_key, "1")
        self.assertNotIn("Symbol:", omitted_match.results[0].content)

    def test_tfidf_cosine_matches_locked_formula(self):
        scope, documents = self._synthetic(
            trades=(
                ("1", "alpha alpha beta"),
                ("2", "alpha gamma"),
            )
        )
        response = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            document_type=DOCUMENT_TYPE_TRADE,
        )
        self.assertEqual(response.state, RETRIEVAL_STATE_OK)
        self.assertEqual(len(response.results), 2)
        candidate_count = 2
        document_frequency = {"alpha": 2, "beta": 1, "gamma": 1}
        idf = {
            term: math.log((1 + candidate_count) / (1 + df)) + 1
            for term, df in document_frequency.items()
        }

        def l2(values):
            norm = math.sqrt(sum(value * value for value in values))
            return [value / norm for value in values]

        query_vector = l2([1 * idf["alpha"], 0.0, 0.0])
        doc_one = l2([2 * idf["alpha"], 1 * idf["beta"], 0.0])
        doc_two = l2([1 * idf["alpha"], 0.0, 1 * idf["gamma"]])
        raw_one = sum(left * right for left, right in zip(query_vector, doc_one, strict=True))
        raw_two = sum(left * right for left, right in zip(query_vector, doc_two, strict=True))
        expected_one = round(raw_one / 1e-9) * 1e-9
        expected_two = round(raw_two / 1e-9) * 1e-9
        self.assertGreater(expected_one, expected_two)
        self.assertEqual(
            [item.source_record_key for item in response.results],
            ["1", "2"],
        )
        self.assertEqual(response.results[0].score, expected_one)
        self.assertEqual(response.results[1].score, expected_two)

    def test_tfidf_audit_counts_and_as_plain(self):
        scope, documents = self._synthetic(
            kpis=(("win_rate", "win_rate: 0.5 alpha"),),
            dataset_content="Journal row count: 2 alpha",
        )
        unfiltered = retrieve_evidence(documents, scope, query="alpha", top_k=10)
        self.assertEqual(unfiltered.state, RETRIEVAL_STATE_OK)
        self.assertEqual(unfiltered.corpus_size, len(documents))
        self.assertEqual(unfiltered.candidate_count, len(documents))
        self.assertLessEqual(len(unfiltered.results), unfiltered.candidate_count)
        payload = unfiltered.as_plain()
        self.assertEqual(payload["corpus_size"], unfiltered.corpus_size)
        self.assertEqual(payload["candidate_count"], unfiltered.candidate_count)
        typed = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            document_type=DOCUMENT_TYPE_KPI,
        )
        self.assertEqual(typed.corpus_size, len(documents))
        self.assertEqual(typed.candidate_count, 1)
        self.assertLess(typed.candidate_count, typed.corpus_size)
        self.assertLessEqual(len(typed.results), typed.candidate_count)
        symbol = retrieve_evidence(documents, scope, query="alpha", symbol="eurusd")
        self.assertEqual(symbol.corpus_size, len(documents))
        self.assertEqual(symbol.candidate_count, 1)
        self.assertEqual(len(symbol.results), 1)

    def test_ticket_and_unknown_symbol_audit_counts(self):
        scope, documents = self._synthetic()
        known = retrieve_evidence(documents, scope, ticket="1")
        self.assertEqual(known.state, RETRIEVAL_STATE_OK)
        self.assertEqual(known.corpus_size, len(documents))
        self.assertEqual(known.candidate_count, 1)
        self.assertEqual(len(known.results), 1)
        unknown_ticket = retrieve_evidence(documents, scope, ticket="99")
        self.assertEqual(unknown_ticket.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(unknown_ticket.corpus_size, len(documents))
        self.assertEqual(unknown_ticket.candidate_count, 0)
        self.assertEqual(unknown_ticket.results, ())
        unknown_symbol = retrieve_evidence(
            documents,
            scope,
            query="alpha",
            symbol="GBPUSD",
        )
        self.assertEqual(unknown_symbol.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(unknown_symbol.corpus_size, len(documents))
        self.assertEqual(unknown_symbol.candidate_count, 0)
        self.assertEqual(unknown_symbol.results, ())

    def test_zero_similarity_keeps_filtered_candidate_count(self):
        scope, documents = self._synthetic()
        response = retrieve_evidence(documents, scope, query="zzzznotpresent")
        self.assertEqual(response.state, RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE)
        self.assertEqual(response.corpus_size, len(documents))
        self.assertEqual(response.candidate_count, len(documents))
        self.assertGreater(response.candidate_count, 0)
        self.assertEqual(response.results, ())
        self.assertLessEqual(len(response.results), response.candidate_count)

    def test_empty_query_and_invalid_corpus_audit_counts(self):
        scope, documents = self._synthetic()
        empty = retrieve_evidence(documents, scope, query="")
        whitespace = retrieve_evidence(documents, scope, query="   \n\t  ")
        typed = retrieve_evidence(
            documents,
            scope,
            query="",
            document_type=DOCUMENT_TYPE_TRADE,
        )
        self.assertEqual(empty.state, RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(whitespace.state, RETRIEVAL_STATE_EMPTY_QUERY)
        self.assertEqual(empty.corpus_size, len(documents))
        self.assertEqual(empty.candidate_count, len(documents))
        self.assertEqual(whitespace.corpus_size, len(documents))
        self.assertEqual(whitespace.candidate_count, len(documents))
        self.assertEqual(typed.corpus_size, len(documents))
        self.assertEqual(typed.candidate_count, 2)
        self.assertEqual(empty.results, ())
        mixed = (
            replace(documents[0], journal_fingerprint="0" * 64),
            *documents[1:],
        )
        invalid = retrieve_evidence(mixed, scope, query="alpha")
        self.assertEqual(invalid.state, RETRIEVAL_STATE_CORPUS_INVALID)
        self.assertEqual(invalid.corpus_size, 0)
        self.assertEqual(invalid.candidate_count, 0)
        self.assertEqual(invalid.results, ())
        self.assertEqual(invalid.as_plain()["corpus_size"], 0)
        self.assertEqual(invalid.as_plain()["candidate_count"], 0)
