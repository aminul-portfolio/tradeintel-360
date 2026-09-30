from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pandas as pd
from django.contrib.auth.models import AnonymousUser
from django.test import SimpleTestCase

from .excursion import canonical_ticket, compute_journal_fingerprint
from .market_data import EXCURSION_CONTRACT_VERSION
from .rag.corpus import build_evidence_corpus, source_basename, validate_evidence_corpus
from .rag.schema import (
    CORPUS_SCHEMA_VERSION,
    CROSS_SYMBOL_WARNING,
    DATASET_DOCUMENT_ID,
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    EVIDENCE_STATUS_COMPUTED,
    EVIDENCE_STATUS_NO_M1,
    EVIDENCE_STATUS_NO_MARKET_DATA,
    RENDER_TEMPLATE_VERSION,
    CorpusError,
    content_sha256,
    make_evidence_document,
    trade_document_id,
)
from .rag.scope import RetrievalScope, RetrievalScopeError, resolve_retrieval_scope
from .trade_review import (
    BAR_EVIDENCE_LABELS,
    BAR_EVIDENCE_UNBOUND_LABEL,
    attach_excursion_evidence,
    enrich_trade_review,
    format_display_value,
    format_evidence_value,
    format_movement_value,
)
from .utils import compute_kpis

BANNED_CLAIMS = (
    "true mfe",
    "true mae",
    "exact high",
    "exact low",
    "optimal exit",
    "missed profit",
    "left on the table",
    "should have exited",
    "should have held",
    "verified ftmo",
    "recommendation",
)


class RagCorpusTests(SimpleTestCase):
    def _user(self, pk=42):
        return SimpleNamespace(is_authenticated=True, pk=pk)

    def _journal(self, rows=None):
        if rows is None:
            rows = [
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
                    "Notes": "secret note",
                    "Strategy": "breakout",
                    "Comment": "do not index",
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
                    "Notes": "keep out",
                    "Tag": "alpha",
                    "Reason": "news",
                },
            ]
        return pd.DataFrame(rows)

    def _scope(self, journal=None, pk=42):
        frame = journal if journal is not None else self._journal()
        return resolve_retrieval_scope(self._user(pk), frame)

    def _evidence_item(self, ticket, **overrides):
        item = {
            "ticket": str(ticket),
            "status": "COMPUTED",
            "reason_code": "OK",
            "interval_high": 16172.8,
            "interval_low": 16135.2,
            "mfe": 10.0,
            "mae": 4.0,
            "high_from_boundary_bar": False,
            "low_from_boundary_bar": False,
            "entry_outside_first_bar_range": False,
            "exit_outside_last_bar_range": False,
            "realised_outside_interval": False,
        }
        item.update(overrides)
        return item

    def _bound_state(self, journal, evidence, fingerprint=None):
        return {
            "contract_version": EXCURSION_CONTRACT_VERSION,
            "journal_fingerprint": fingerprint or compute_journal_fingerprint(journal),
            "market_file_sha256": "abc123",
            "time_basis": {"kind": "UTC", "resolution": "USER_DECLARED"},
            "market_provenance": {
                "source_basename": "bars.csv",
                "market_file_sha256": "abc123",
            },
            "run_status": "OK",
            "status_counts": {},
            "reason_counts": {},
            "evidence": evidence,
        }

    def _build(self, journal=None, **kwargs):
        frame = journal if journal is not None else self._journal()
        scope = kwargs.pop("scope", None) or self._scope(frame)
        if "kpi_mapping" not in kwargs:
            kwargs["kpi_mapping"] = compute_kpis(frame)
        return scope, build_evidence_corpus(scope, frame, **kwargs)

    def _by_type(self, documents, document_type):
        return [item for item in documents if item.document_type == document_type]

    def _complete_evidence(self):
        return {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
        }

    def _reject_bound_state(self, **updates):
        journal = self._journal()
        state = self._bound_state(journal, self._complete_evidence())
        state.update(updates)
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=state)
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")
        return raised

    def _reject_bound_evidence(self, evidence):
        journal = self._journal()
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=self._bound_state(journal, evidence))
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")
        return raised

    def test_authenticated_scope_uses_journal_fingerprint(self):
        journal = self._journal()
        scope = resolve_retrieval_scope(self._user(7), journal)
        self.assertEqual(scope.owner_id, 7)
        self.assertEqual(scope.journal_fingerprint, compute_journal_fingerprint(journal))

    def test_unauthenticated_scope_is_rejected(self):
        with self.assertRaises(RetrievalScopeError) as raised:
            resolve_retrieval_scope(AnonymousUser(), self._journal())
        self.assertEqual(raised.exception.reason, "UNAUTHENTICATED_SCOPE")

    def test_missing_user_identity_is_rejected(self):
        user = SimpleNamespace(is_authenticated=True, pk=None)
        with self.assertRaises(RetrievalScopeError) as raised:
            resolve_retrieval_scope(user, self._journal())
        self.assertEqual(raised.exception.reason, "UNAUTHENTICATED_SCOPE")

    def test_empty_journal_and_invalid_fingerprint_fail_closed(self):
        with self.assertRaises(RetrievalScopeError) as empty:
            resolve_retrieval_scope(self._user(), pd.DataFrame())
        self.assertEqual(empty.exception.reason, "INVALID_JOURNAL_FINGERPRINT")
        duplicates = self._journal(
            [
                {"Ticket": 1, "Symbol": "US100.cash", "Type": "buy", "Entry": 1, "Exit": 2},
                {"Ticket": 1, "Symbol": "EURUSD", "Type": "sell", "Entry": 2, "Exit": 1},
            ]
        )
        with self.assertRaises(RetrievalScopeError) as invalid:
            resolve_retrieval_scope(self._user(), duplicates)
        self.assertEqual(invalid.exception.reason, "INVALID_JOURNAL_FINGERPRINT")
        self.assertIsNone(compute_journal_fingerprint(duplicates))

    def test_three_document_types_and_readable_ids(self):
        scope, documents = self._build()
        types = {item.document_type for item in documents}
        self.assertEqual(
            types,
            {DOCUMENT_TYPE_TRADE, DOCUMENT_TYPE_KPI, DOCUMENT_TYPE_DATASET},
        )
        trades = self._by_type(documents, DOCUMENT_TYPE_TRADE)
        self.assertEqual(
            [item.document_id for item in trades],
            ["TRADE_EVIDENCE:ticket:1", "TRADE_EVIDENCE:ticket:2"],
        )
        self.assertEqual(trades[0].source_record_key, "1")
        kpis = self._by_type(documents, DOCUMENT_TYPE_KPI)
        self.assertTrue(all(item.document_id.startswith("KPI_EVIDENCE:kpi:") for item in kpis))
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].document_id, DATASET_DOCUMENT_ID)
        self.assertEqual(scope.journal_fingerprint, documents[0].journal_fingerprint)

    def test_content_sha256_is_utf8_of_content_only(self):
        _scope, documents = self._build()
        trade = self._by_type(documents, DOCUMENT_TYPE_TRADE)[0]
        self.assertEqual(trade.content_sha256, content_sha256(trade.content))
        self.assertEqual(
            trade.content_sha256,
            hashlib.sha256(trade.content.encode("utf-8")).hexdigest(),
        )
        mutated = replace(trade, journal_fingerprint="deadbeef")
        self.assertEqual(mutated.content_sha256, trade.content_sha256)

    def test_repeated_build_is_deterministic(self):
        journal = self._journal()
        kpis = {"Win Rate (%)": "50.00", "Total Trades": 2}
        first = build_evidence_corpus(
            self._scope(journal),
            journal,
            kpi_mapping=kpis,
            source_filename=r"C:\Users\someone\data\journal.csv",
        )
        second = build_evidence_corpus(
            self._scope(journal),
            journal,
            kpi_mapping=dict(reversed(list(kpis.items()))),
            source_filename="/home/someone/data/journal.csv",
        )
        self.assertEqual(
            [(item.document_id, item.content_sha256) for item in first],
            [(item.document_id, item.content_sha256) for item in second],
        )
        self.assertEqual([item.content for item in first], [item.content for item in second])

    def test_kpi_documents_are_sorted_by_key(self):
        journal = self._journal()
        mapping = {"z_loss": "1.00", "a_win": "2.00", "m_mid": "3.00"}
        _scope, documents = self._build(journal, kpi_mapping=mapping)
        kpi_ids = [item.document_id for item in self._by_type(documents, DOCUMENT_TYPE_KPI)]
        self.assertEqual(
            kpi_ids,
            [
                "KPI_EVIDENCE:kpi:a_win",
                "KPI_EVIDENCE:kpi:m_mid",
                "KPI_EVIDENCE:kpi:z_loss",
            ],
        )
        self.assertEqual(documents[-1].document_type, DOCUMENT_TYPE_DATASET)

    def test_source_filename_is_basename_only(self):
        journal = self._journal()
        for path in (
            r"C:\Users\someone\data\journal.csv",
            "/home/someone/data/journal.csv",
        ):
            self.assertEqual(source_basename(path), "journal.csv")
            _scope, documents = self._build(journal, source_filename=path)
            joined = "\n".join(item.content for item in documents)
            provenance = " ".join(str(dict(item.provenance)) for item in documents)
            self.assertIn("journal.csv", joined)
            self.assertNotIn("Users", joined)
            self.assertNotIn("someone", joined)
            self.assertNotIn("/home/", provenance)
            self.assertNotIn(r"C:\Users", provenance)

    def test_free_text_fields_are_excluded(self):
        _scope, documents = self._build()
        joined = "\n".join(item.content for item in documents).lower()
        for phrase in (
            "secret note",
            "keep out",
            "breakout",
            "do not index",
            "alpha",
            "news",
        ):
            self.assertNotIn(phrase, joined)
        self.assertNotIn("notes:", joined)
        self.assertNotIn("strategy:", joined)
        self.assertNotIn("comment:", joined)

    def test_unbound_evidence_uses_no_m1_status(self):
        _scope, documents = self._build()
        trade = self._by_type(documents, DOCUMENT_TYPE_TRADE)[0]
        self.assertEqual(trade.evidence_status, EVIDENCE_STATUS_NO_M1)
        self.assertIn(f"Bar Evidence: {BAR_EVIDENCE_UNBOUND_LABEL}", trade.content)
        self.assertIn("Approx. MFE: not available (NO_M1_EVIDENCE)", trade.content)
        self.assertIn("Approx. MAE: not available (NO_M1_EVIDENCE)", trade.content)
        self.assertNotIn("Approx. MFE: 0", trade.content)
        self.assertNotIn("Bar Evidence: Computed", trade.content)
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        self.assertIn(BAR_EVIDENCE_UNBOUND_LABEL, dataset.content)

    def test_bound_computed_matches_trade_review_formatter(self):
        journal = self._journal()
        high = 16172.8
        low = 16135.2
        mfe = 5.399999999999782
        mae = 4.0
        evidence = {
            "1": self._evidence_item(
                1,
                interval_high=high,
                interval_low=low,
                mfe=mfe,
                mae=mae,
            ),
            "2": self._evidence_item(
                2,
                status="NO_MARKET_DATA",
                reason_code="OUTSIDE_FILE_RANGE",
                interval_high=None,
                interval_low=None,
                mfe=None,
                mae=None,
            ),
        }
        _scope, documents = self._build(
            journal,
            excursion_state=self._bound_state(journal, evidence),
        )
        trades = {
            item.source_record_key: item
            for item in self._by_type(documents, DOCUMENT_TYPE_TRADE)
        }
        computed = trades["1"]
        missing = trades["2"]
        self.assertEqual(computed.evidence_status, EVIDENCE_STATUS_COMPUTED)
        self.assertIn(
            f"Approx. Window High: {format_evidence_value(high)}",
            computed.content,
        )
        self.assertIn(
            f"Approx. Window Low: {format_evidence_value(low)}",
            computed.content,
        )
        self.assertIn(
            f"Approx. MFE: {format_evidence_value(mfe)} price pts",
            computed.content,
        )
        self.assertIn(
            f"Approx. MAE: {format_evidence_value(mae)} price pts",
            computed.content,
        )
        self.assertIn("Bar Evidence: Computed", computed.content)
        self.assertEqual(missing.evidence_status, EVIDENCE_STATUS_NO_MARKET_DATA)
        self.assertIn(
            f"Bar Evidence: {BAR_EVIDENCE_LABELS['NO_MARKET_DATA']}",
            missing.content,
        )
        self.assertIn("not available (NO_MARKET_DATA)", missing.content)
        self.assertNotIn(BAR_EVIDENCE_UNBOUND_LABEL, missing.content)
        expected_move = format_movement_value(5.0, "Pips")
        self.assertIn(f"Realised movement: {expected_move}", computed.content)

    def test_owner_identity_stays_out_of_corpus(self):
        journal = self._journal()
        _scope, documents = self._build(journal, scope=self._scope(journal, pk=987654))
        joined = "\n".join(item.content for item in documents)
        self.assertNotIn("987654", joined)
        for item in documents:
            self.assertNotIn("owner_id", item.provenance)
            self.assertNotIn("user_id", item.provenance)
            self.assertNotIn("987654", item.document_id)

    def test_duplicate_document_id_is_rejected(self):
        scope, documents = self._build()
        trade = self._by_type(documents, DOCUMENT_TYPE_TRADE)[0]
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus((*documents, trade), scope)
        self.assertEqual(raised.exception.reason, "DUPLICATE_DOCUMENT_ID")

    def test_wrong_and_mixed_fingerprints_are_rejected(self):
        scope, documents = self._build()
        mixed = replace(documents[0], journal_fingerprint="0" * 64)
        with self.assertRaises(CorpusError) as mixed_raised:
            validate_evidence_corpus((mixed, *documents[1:]), scope)
        self.assertEqual(mixed_raised.exception.reason, "MIXED_JOURNAL_CORPUS")
        mixed_scope = RetrievalScope(owner_id=1, journal_fingerprint="0" * 64)
        with self.assertRaises(CorpusError) as mismatched:
            validate_evidence_corpus(documents, mixed_scope)
        self.assertEqual(mismatched.exception.reason, "FINGERPRINT_MISMATCH")

    def test_unsupported_version_type_and_hash_are_rejected(self):
        scope, documents = self._build()
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        remainder = [item for item in documents if item is not dataset]
        with self.assertRaises(CorpusError) as schema:
            validate_evidence_corpus(
                (replace(dataset, corpus_schema_version="nope"), *remainder),
                scope,
            )
        self.assertEqual(schema.exception.reason, "UNSUPPORTED_SCHEMA_VERSION")
        with self.assertRaises(CorpusError) as render:
            validate_evidence_corpus(
                (replace(dataset, render_template_version="nope"), *remainder),
                scope,
            )
        self.assertEqual(render.exception.reason, "UNSUPPORTED_RENDER_VERSION")
        with self.assertRaises(CorpusError) as kind:
            validate_evidence_corpus(
                (replace(dataset, document_type="MARKET_EVIDENCE"), *remainder),
                scope,
            )
        self.assertEqual(kind.exception.reason, "UNSUPPORTED_DOCUMENT_TYPE")
        with self.assertRaises(CorpusError) as hashed:
            validate_evidence_corpus(
                (replace(dataset, content_sha256="deadbeef"), *remainder),
                scope,
            )
        self.assertEqual(hashed.exception.reason, "INVALID_CONTENT_HASH")

    def test_malformed_provenance_and_context_count_are_rejected(self):
        scope, documents = self._build()
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        remainder = [item for item in documents if item is not dataset]
        with self.assertRaises(CorpusError) as provenance:
            validate_evidence_corpus(
                (
                    replace(
                        dataset,
                        provenance=MappingProxyType({"bad": float("nan")}),
                    ),
                    *remainder,
                ),
                scope,
            )
        self.assertEqual(provenance.exception.reason, "INVALID_PROVENANCE")
        extra = replace(
            dataset,
            document_id="DATASET_CONTEXT:other",
            source_record_key="other",
            content="extra",
            content_sha256=content_sha256("extra"),
        )
        with self.assertRaises(CorpusError) as context:
            validate_evidence_corpus((*documents, extra), scope)
        self.assertEqual(context.exception.reason, "INVALID_CONTEXT_COUNT")

    def test_banned_claim_phrases_and_warning_are_preserved(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2, status="NO_MARKET_DATA", mfe=None, mae=None),
        }
        _scope, documents = self._build(
            journal,
            excursion_state=self._bound_state(journal, evidence),
        )
        joined = "\n".join(item.content for item in documents).lower()
        for phrase in BANNED_CLAIMS:
            self.assertNotIn(phrase, joined)
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        self.assertIn(CROSS_SYMBOL_WARNING, dataset.content)
        self.assertEqual(CORPUS_SCHEMA_VERSION, "tradeintel.evidence.v1")
        self.assertEqual(RENDER_TEMPLATE_VERSION, "tradeintel.evidence.render.v1")
        self.assertIn("UTC", dataset.content)
        self.assertIn("bars.csv", dataset.content)
        self.assertNotIn("abc123", dataset.content)

    def test_wrong_bound_state_fingerprint_fails_closed(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
        }
        state = self._bound_state(journal, evidence, fingerprint="0" * 64)
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=state)
        self.assertEqual(raised.exception.reason, "FINGERPRINT_MISMATCH")

    def test_missing_bound_state_fingerprint_fails_closed(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
        }
        state = self._bound_state(journal, evidence)
        del state["journal_fingerprint"]
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=state)
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")

    def test_raw_evidence_map_is_rejected(self):
        journal = self._journal()
        raw = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
        }
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=raw)
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")

    def test_bound_evidence_missing_journal_ticket_is_rejected(self):
        journal = self._journal()
        evidence = {"1": self._evidence_item(1)}
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=self._bound_state(journal, evidence))
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")

    def test_bound_evidence_unexpected_ticket_is_rejected(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
            "99": self._evidence_item(99),
        }
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=self._bound_state(journal, evidence))
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")

    def test_malformed_computed_evidence_is_rejected(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1, mfe=None, mae=None),
            "2": self._evidence_item(2),
        }
        with self.assertRaises(CorpusError) as raised:
            self._build(journal, excursion_state=self._bound_state(journal, evidence))
        self.assertEqual(raised.exception.reason, "INVALID_BOUND_EVIDENCE")

    def test_mixed_document_fingerprints_raise_mixed_journal_corpus(self):
        scope, documents = self._build()
        fingerprint_a = scope.journal_fingerprint
        fingerprint_b = "b" * 64
        self.assertNotEqual(fingerprint_a, fingerprint_b)
        mixed = (
            replace(documents[0], journal_fingerprint=fingerprint_a),
            replace(documents[1], journal_fingerprint=fingerprint_b),
            *documents[2:],
        )
        fingerprints = {item.journal_fingerprint for item in mixed}
        self.assertEqual(fingerprints, {fingerprint_a, fingerprint_b})
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus(mixed, scope)
        self.assertEqual(raised.exception.reason, "MIXED_JOURNAL_CORPUS")

    def test_single_wrong_corpus_fingerprint_raises_fingerprint_mismatch(self):
        scope, documents = self._build()
        wrong_fp = "0" * 64
        self.assertNotEqual(scope.journal_fingerprint, wrong_fp)
        wrong_docs = tuple(
            replace(item, journal_fingerprint=wrong_fp) for item in documents
        )
        self.assertEqual({item.journal_fingerprint for item in wrong_docs}, {wrong_fp})
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus(wrong_docs, scope)
        self.assertEqual(raised.exception.reason, "FINGERPRINT_MISMATCH")

    def test_non_mapping_provenance_raises_invalid_provenance(self):
        scope, documents = self._build()
        broken = replace(documents[0], provenance=["not-a-mapping"])
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus((broken, *documents[1:]), scope)
        self.assertEqual(raised.exception.reason, "INVALID_PROVENANCE")

    def test_empty_evidence_status_is_rejected(self):
        scope, documents = self._build()
        broken = replace(documents[0], evidence_status="")
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus((broken, *documents[1:]), scope)
        self.assertEqual(raised.exception.reason, "MISSING_REQUIRED_FIELD")
        with self.assertRaises(CorpusError) as constructed:
            make_evidence_document(
                document_id=trade_document_id("1"),
                document_type=DOCUMENT_TYPE_TRADE,
                journal_fingerprint=scope.journal_fingerprint,
                source_record_key="1",
                content="Ticket: 1",
                evidence_status="",
            )
        self.assertEqual(constructed.exception.reason, "MISSING_REQUIRED_FIELD")

    def test_invalid_document_identity_is_rejected(self):
        scope, documents = self._build()
        trade = self._by_type(documents, DOCUMENT_TYPE_TRADE)[0]
        kpi = self._by_type(documents, DOCUMENT_TYPE_KPI)[0]
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        remainder = [
            item for item in documents if item is not trade and item is not kpi
        ]
        with self.assertRaises(CorpusError) as trade_id:
            validate_evidence_corpus(
                (replace(trade, document_id="TRADE_EVIDENCE:ticket:nope"), *remainder),
                scope,
            )
        self.assertEqual(trade_id.exception.reason, "INVALID_DOCUMENT_ID")
        with self.assertRaises(CorpusError) as kpi_id:
            validate_evidence_corpus(
                (
                    replace(kpi, document_id="KPI_EVIDENCE:kpi:nope"),
                    *[item for item in documents if item is not kpi],
                ),
                scope,
            )
        self.assertEqual(kpi_id.exception.reason, "INVALID_DOCUMENT_ID")
        with self.assertRaises(CorpusError) as dataset_id:
            validate_evidence_corpus(
                (
                    replace(dataset, source_record_key="other"),
                    *[item for item in documents if item is not dataset],
                ),
                scope,
            )
        self.assertEqual(dataset_id.exception.reason, "INVALID_DOCUMENT_ID")

    def test_nested_provenance_is_immutable_and_detached(self):
        inner = {"kind": "UTC", "tags": ["keep"]}
        payload = {"time_basis": inner, "source": "journal.csv"}
        document = make_evidence_document(
            document_id=trade_document_id("1"),
            document_type=DOCUMENT_TYPE_TRADE,
            journal_fingerprint="a" * 64,
            source_record_key="1",
            content="Ticket: 1",
            evidence_status=EVIDENCE_STATUS_NO_M1,
            provenance=payload,
        )
        inner["kind"] = "IANA"
        inner["tags"].append("mutated")
        payload["extra"] = "caller-owned"
        self.assertEqual(document.provenance["time_basis"]["kind"], "UTC")
        self.assertEqual(document.provenance["time_basis"]["tags"], ("keep",))
        self.assertNotIn("extra", document.provenance)
        with self.assertRaises(TypeError):
            document.provenance["injected"] = 1
        with self.assertRaises(TypeError):
            document.provenance["time_basis"]["kind"] = "IANA"

    def test_fixed_offset_time_basis_is_preserved(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
        }
        state = self._bound_state(journal, evidence)
        state["time_basis"] = {
            "kind": "FIXED_OFFSET",
            "resolution": "USER_DECLARED",
            "offset_minutes": 120,
        }
        _scope, documents = self._build(journal, excursion_state=state)
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        self.assertIn(
            "Declared time basis: FIXED_OFFSET (+120 minutes)",
            dataset.content,
        )
        self.assertEqual(dataset.provenance["time_basis"]["kind"], "FIXED_OFFSET")
        self.assertEqual(dataset.provenance["time_basis"]["offset_minutes"], 120)

    def test_iana_time_basis_zone_is_preserved(self):
        journal = self._journal()
        evidence = {
            "1": self._evidence_item(1),
            "2": self._evidence_item(2),
        }
        state = self._bound_state(journal, evidence)
        state["time_basis"] = {
            "kind": "IANA",
            "resolution": "USER_DECLARED",
            "zone": "Europe/London",
        }
        _scope, documents = self._build(journal, excursion_state=state)
        dataset = self._by_type(documents, DOCUMENT_TYPE_DATASET)[0]
        self.assertIn("Declared time basis: IANA (Europe/London)", dataset.content)
        self.assertEqual(dataset.provenance["time_basis"]["kind"], "IANA")
        self.assertEqual(dataset.provenance["time_basis"]["zone"], "Europe/London")

    def test_full_trade_content_parity_with_trade_review_helpers(self):
        journal = self._journal()
        high = 16172.8
        low = 16135.2
        mfe = 5.399999999999782
        mae = 4.0
        evidence = {
            "1": self._evidence_item(
                1,
                interval_high=high,
                interval_low=low,
                mfe=mfe,
                mae=mae,
            ),
            "2": self._evidence_item(
                2,
                status="NO_MARKET_DATA",
                reason_code="OUTSIDE_FILE_RANGE",
                interval_high=None,
                interval_low=None,
                mfe=None,
                mae=None,
            ),
        }
        bound = self._bound_state(journal, evidence)
        enriched = enrich_trade_review(
            attach_excursion_evidence(journal.copy(), evidence)
        )
        row = enriched.loc[enriched["Ticket"].map(canonical_ticket) == "1"].iloc[0]
        _scope, documents = self._build(journal, excursion_state=bound)
        trade = {
            item.source_record_key: item
            for item in self._by_type(documents, DOCUMENT_TYPE_TRADE)
        }["1"]
        expected_lines = (
            "Ticket: 1",
            f"Symbol: {format_display_value(row['Symbol'])}",
            f"Type: {format_display_value(row['Type'])}",
            f"Open Time: {format_display_value(row['Open Time'])}",
            f"Close Time: {format_display_value(row['Close Time'])}",
            f"Entry: {format_display_value(row['Entry'])}",
            f"Exit: {format_display_value(row['Exit'])}",
            f"Profit: {format_display_value(row['Profit'])}",
            f"Commission: {format_display_value(row['Commission'])}",
            f"Swap: {format_display_value(row['Swap'])}",
            "Realised movement: "
            + format_movement_value(
                row.get("_movement_value"),
                row.get("_movement_label") or "",
            ),
            f"Bar Evidence: {BAR_EVIDENCE_LABELS['COMPUTED']}",
            f"Approx. Window High: {format_evidence_value(high)}",
            f"Approx. Window Low: {format_evidence_value(low)}",
            f"Approx. MFE: {format_evidence_value(mfe)} price pts",
            f"Approx. MAE: {format_evidence_value(mae)} price pts",
        )
        for line in expected_lines:
            self.assertIn(line, trade.content)

    def test_unknown_time_basis_kind_is_rejected(self):
        self._reject_bound_state(
            time_basis={"kind": "LOCAL", "resolution": "USER_DECLARED"},
        )

    def test_wrong_time_basis_resolution_is_rejected(self):
        self._reject_bound_state(
            time_basis={"kind": "UTC", "resolution": "INFERRED"},
        )

    def test_fixed_offset_below_minimum_is_rejected(self):
        self._reject_bound_state(
            time_basis={
                "kind": "FIXED_OFFSET",
                "resolution": "USER_DECLARED",
                "offset_minutes": -735,
            },
        )

    def test_fixed_offset_above_maximum_is_rejected(self):
        self._reject_bound_state(
            time_basis={
                "kind": "FIXED_OFFSET",
                "resolution": "USER_DECLARED",
                "offset_minutes": 855,
            },
        )

    def test_fixed_offset_not_divisible_by_15_is_rejected(self):
        self._reject_bound_state(
            time_basis={
                "kind": "FIXED_OFFSET",
                "resolution": "USER_DECLARED",
                "offset_minutes": 121,
            },
        )

    def test_fixed_offset_bool_is_rejected(self):
        self._reject_bound_state(
            time_basis={
                "kind": "FIXED_OFFSET",
                "resolution": "USER_DECLARED",
                "offset_minutes": True,
            },
        )

    def test_invalid_iana_zone_is_rejected(self):
        self._reject_bound_state(
            time_basis={
                "kind": "IANA",
                "resolution": "USER_DECLARED",
                "zone": "Not/ARealZone",
            },
        )

    def test_conflicting_utc_offset_or_zone_is_rejected(self):
        self._reject_bound_state(
            time_basis={
                "kind": "UTC",
                "resolution": "USER_DECLARED",
                "offset_minutes": 0,
            },
        )
        self._reject_bound_state(
            time_basis={
                "kind": "UTC",
                "resolution": "USER_DECLARED",
                "zone": "UTC",
            },
        )

    def test_unknown_bound_evidence_status_is_rejected(self):
        self._reject_bound_evidence(
            {
                "1": self._evidence_item(1, status="LOOKS_GOOD"),
                "2": self._evidence_item(2),
            }
        )

    def test_bound_no_m1_evidence_status_is_rejected(self):
        self._reject_bound_evidence(
            {
                "1": self._evidence_item(1, status=EVIDENCE_STATUS_NO_M1),
                "2": self._evidence_item(2),
            }
        )

    def test_each_authoritative_bound_status_is_accepted(self):
        journal = self._journal()
        self.assertEqual(
            set(BAR_EVIDENCE_LABELS),
            {
                "COMPUTED",
                "NO_MARKET_DATA",
                "INVALID_TRADE_DATA",
                "TIMEZONE_AMBIGUOUS",
                "TIME_BASIS_INCONSISTENT",
                "INCOMPLETE_COVERAGE",
                "INVARIANT_VIOLATION",
            },
        )
        for status in BAR_EVIDENCE_LABELS:
            if status == "COMPUTED":
                evidence = self._complete_evidence()
            else:
                evidence = {
                    "1": self._evidence_item(
                        1,
                        status=status,
                        interval_high=None,
                        interval_low=None,
                        mfe=None,
                        mae=None,
                    ),
                    "2": self._evidence_item(
                        2,
                        status=status,
                        interval_high=None,
                        interval_low=None,
                        mfe=None,
                        mae=None,
                    ),
                }
            _scope, documents = self._build(
                journal,
                excursion_state=self._bound_state(journal, evidence),
            )
            trades = self._by_type(documents, DOCUMENT_TYPE_TRADE)
            self.assertEqual({item.evidence_status for item in trades}, {status})
            for trade in trades:
                self.assertIn(
                    f"Bar Evidence: {BAR_EVIDENCE_LABELS[status]}",
                    trade.content,
                )
                self.assertNotEqual(trade.evidence_status, EVIDENCE_STATUS_NO_M1)

    def test_duplicate_canonical_evidence_keys_are_rejected(self):
        item = self._evidence_item(1)
        self._reject_bound_evidence(
            {
                1: item,
                "1": self._evidence_item(1),
                "2": self._evidence_item(2),
            }
        )

    def test_invalid_internal_item_ticket_is_rejected(self):
        invalid = self._evidence_item(1)
        invalid["ticket"] = ""
        self._reject_bound_evidence(
            {
                "1": invalid,
                "2": self._evidence_item(2),
            }
        )

    def test_mismatched_internal_item_ticket_is_rejected(self):
        mismatched = self._evidence_item(1)
        mismatched["ticket"] = "99"
        self._reject_bound_evidence(
            {
                "1": mismatched,
                "2": self._evidence_item(2),
            }
        )

    def test_nested_owner_id_provenance_is_rejected(self):
        scope, documents = self._build()
        nested = MappingProxyType(
            {
                "journal_fingerprint": scope.journal_fingerprint,
                "context": MappingProxyType({"owner_id": 42}),
            }
        )
        broken = replace(documents[0], provenance=nested)
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus((broken, *documents[1:]), scope)
        self.assertEqual(raised.exception.reason, "INVALID_PROVENANCE")

    def test_nested_user_id_provenance_is_rejected(self):
        scope, documents = self._build()
        nested = MappingProxyType(
            {
                "journal_fingerprint": scope.journal_fingerprint,
                "context": MappingProxyType({"user_id": 42}),
            }
        )
        broken = replace(documents[0], provenance=nested)
        with self.assertRaises(CorpusError) as raised:
            validate_evidence_corpus((broken, *documents[1:]), scope)
        self.assertEqual(raised.exception.reason, "INVALID_PROVENANCE")
