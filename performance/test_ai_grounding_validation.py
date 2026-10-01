from __future__ import annotations

import ast
import json
from pathlib import Path

from django.test import SimpleTestCase

from .ai_grounding.codes import (
    FROZEN_LIMITATION_CODES,
    FROZEN_REJECTION_CODES,
    FieldType,
    LimitationCode,
    RejectionCode,
    ValidationStatus,
)
from .ai_grounding.prompt import prompt_hash, prompt_template_hash, render_evidence_summary_prompt
from .ai_grounding.schema import (
    DEFAULT_CONSTRAINTS,
    GROUNDING_REQUEST_SCHEMA_VERSION,
    GROUNDING_RESPONSE_SCHEMA_VERSION,
    MAX_ANSWER_CHARS,
    MAX_CITATIONS,
    MAX_CLAIMS,
    MAX_LIMITATION_CODES,
    PROMPT_TEMPLATE_VERSION,
    TASK_TYPE,
    GroundedAIRequest,
    GroundedEvidenceItem,
    GroundingConstraints,
    ServerGroundingContext,
    make_model_facing_field,
    request_sha256,
)
from .ai_grounding.validation import ANSWER_NUMBER_RE, validate_grounded_response
from .test_ai_grounding_prompt import (
    LOCKED_PROMPT_HASH,
    LOCKED_PROMPT_TEMPLATE_HASH,
    _golden_request,
)


def _ticket(value="1"):
    return make_model_facing_field("ticket", "Ticket", value, FieldType.IDENTIFIER)


def _symbol(value="EURUSD"):
    return make_model_facing_field("symbol", "Symbol", value, FieldType.IDENTIFIER)


def _profit(value="5.0"):
    return make_model_facing_field("profit", "Profit", value, FieldType.DECIMAL)


def _open_time(value="2026-06-25 10:30:15"):
    return make_model_facing_field("open_time", "Open Time", value, FieldType.TIMESTAMP)


def _excursion():
    return (
        make_model_facing_field("approx_window_high", "Approx. Window High", "1.25", FieldType.DECIMAL),
        make_model_facing_field("approx_window_low", "Approx. Window Low", "1.05", FieldType.DECIMAL),
        make_model_facing_field("approx_mfe", "Approx. MFE", "0.05", FieldType.DECIMAL, "price pts"),
        make_model_facing_field("approx_mae", "Approx. MAE", "-54.7", FieldType.DECIMAL, "price pts"),
    )


def _trade_item(alias="E1", ticket="1", symbol="EURUSD", extra=()):
    fields = (_ticket(ticket), _symbol(symbol), _profit()) + extra + _excursion()
    claimable = tuple(
        field.field_key for field in fields if field.value_type in {FieldType.DECIMAL, FieldType.INTEGER}
    )
    return GroundedEvidenceItem(
        alias=alias,
        document_type="TRADE_EVIDENCE",
        evidence_status="COMPUTED",
        fields=fields,
        claimable_field_keys=claimable,
    )


def _no_market_item(alias="E1"):
    return GroundedEvidenceItem(
        alias=alias,
        document_type="TRADE_EVIDENCE",
        evidence_status="NO_MARKET_DATA",
        fields=(_ticket(), _profit()),
        claimable_field_keys=("profit",),
    )


def _integer_item(alias="E1", value="99"):
    total = make_model_facing_field("total_trades", "Total Trades", value, FieldType.INTEGER)
    return GroundedEvidenceItem(
        alias=alias,
        document_type="KPI_EVIDENCE",
        evidence_status="AUTHORITATIVE",
        fields=(total,),
        claimable_field_keys=("total_trades",),
    )


def _detached_context():
    return ServerGroundingContext(
        request_sha256="b" * 64,
        alias_to_document_id=(("E1", "TRADE_EVIDENCE:ticket:1"),),
        journal_fingerprint="c" * 64,
        corpus_schema_version="tradeintel.evidence.v1",
        render_template_version="tradeintel.evidence.render.v1",
        retrieval_method="TFIDF_COSINE",
        returned_count=1,
        candidate_count=1,
        corpus_size=9,
    )


def _unicode_escape(text):
    return "".join(f"\\u{ord(character):04x}" for character in text)


def _kpi_item(alias="E1"):
    rate = make_model_facing_field("win_rate_pct", "Win Rate (%)", "75", FieldType.DECIMAL, "%")
    return GroundedEvidenceItem(
        alias=alias,
        document_type="KPI_EVIDENCE",
        evidence_status="AUTHORITATIVE",
        fields=(rate,),
        claimable_field_keys=("win_rate_pct",),
    )


def _request(evidence=None, limitation_codes=None, question="What is ticket 1?"):
    items = evidence if evidence is not None else (_trade_item(),)
    codes = (
        (LimitationCode.APPROXIMATE_M1_EVIDENCE,)
        if limitation_codes is None
        else limitation_codes
    )
    return GroundedAIRequest(
        schema_version=GROUNDING_REQUEST_SCHEMA_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        response_schema_version=GROUNDING_RESPONSE_SCHEMA_VERSION,
        task_type=TASK_TYPE,
        question=question,
        evidence=items,
        required_limitation_codes=codes,
        constraints=DEFAULT_CONSTRAINTS,
    )


def _context(request, document_id="TRADE_EVIDENCE:ticket:1"):
    aliases = []
    for index, item in enumerate(request.evidence, start=1):
        bound_id = document_id if index == 1 else f"{document_id}:{index}"
        aliases.append((item.alias, bound_id))
    return ServerGroundingContext(
        request_sha256=request_sha256(request),
        alias_to_document_id=tuple(aliases),
        journal_fingerprint="c" * 64,
        corpus_schema_version="tradeintel.evidence.v1",
        render_template_version="tradeintel.evidence.render.v1",
        retrieval_method="TFIDF_COSINE",
        returned_count=len(request.evidence),
        candidate_count=len(request.evidence),
        corpus_size=9,
    )


def _payload(
    answer="Profit is 5.0 [E1].",
    claims=None,
    citations=None,
    limitation_codes=None,
    **extra,
):
    body = {
        "schema_version": GROUNDING_RESPONSE_SCHEMA_VERSION,
        "answer": answer,
        "claims": claims if claims is not None else [{"evidence": "E1", "field": "profit", "value": "5.0"}],
        "citations": citations if citations is not None else ["E1"],
        "limitation_codes": (
            limitation_codes
            if limitation_codes is not None
            else ["APPROXIMATE_M1_EVIDENCE"]
        ),
    }
    body.update(extra)
    return body


def _dumps(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate(request=None, context=None, payload=None, candidate_json=None):
    req = request if request is not None else _request()
    ctx = context if context is not None else _context(req)
    raw = candidate_json if candidate_json is not None else _dumps(payload if payload is not None else _payload())
    return validate_grounded_response(req, ctx, raw)


class AiGroundingValidationTests(SimpleTestCase):
    def test_valid_minimal_response(self):
        result = _validate()
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        self.assertEqual(result.status.value, "PASSED_DETERMINISTIC_CHECKS")
        self.assertEqual(result.rejection_codes, ())
        self.assertTrue(result.review_required)

    def test_valid_numeric_claim(self):
        result = _validate(payload=_payload(answer="Profit equals 5.0 [E1]."))
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_valid_percentage_numeric_binding(self):
        request = _request(evidence=(_kpi_item(),), limitation_codes=())
        payload = _payload(
            answer="Win rate is 75% [E1].",
            claims=[{"evidence": "E1", "field": "win_rate_pct", "value": "75"}],
            limitation_codes=[],
        )
        result = _validate(request=request, context=_context(request, "KPI_EVIDENCE:kpi:win_rate_pct"), payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_multiple_valid_claims(self):
        payload = _payload(
            answer="Profit is 5.0 and approx MAE is -54.7 [E1].",
            claims=[
                {"evidence": "E1", "field": "profit", "value": "5.0"},
                {"evidence": "E1", "field": "approx_mae", "value": "-54.7"},
            ],
        )
        result = _validate(payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_multiple_citations(self):
        request = _request(
            evidence=(
                _trade_item(alias="E1", ticket="1"),
                _trade_item(alias="E2", ticket="2"),
            )
        )
        payload = _payload(
            answer="Ticket 1 profit is 5.0 [E1]. Ticket 2 profit is 5.0 [E2].",
            claims=[
                {"evidence": "E1", "field": "profit", "value": "5.0"},
                {"evidence": "E2", "field": "profit", "value": "5.0"},
            ],
            citations=["E1", "E2"],
        )
        result = _validate(request=request, context=_context(request), payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_invalid_json(self):
        result = _validate(candidate_json="{not-json")
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertEqual(result.rejection_codes, (RejectionCode.SCHEMA_INVALID_JSON,))

    def test_non_object_json(self):
        result = _validate(candidate_json="[]")
        self.assertIn(RejectionCode.SCHEMA_TYPE_ERROR, result.rejection_codes)

    def test_missing_top_level_field(self):
        payload = _payload()
        del payload["citations"]
        result = _validate(payload=payload)
        self.assertIn(RejectionCode.SCHEMA_MISSING_FIELD, result.rejection_codes)

    def test_unknown_top_level_field(self):
        result = _validate(payload=_payload(notes="extra"))
        self.assertIn(RejectionCode.SCHEMA_UNKNOWN_FIELD, result.rejection_codes)

    def test_review_required_rejected_as_unknown_field(self):
        result = _validate(payload=_payload(review_required=False))
        self.assertIn(RejectionCode.SCHEMA_UNKNOWN_FIELD, result.rejection_codes)
        self.assertTrue(result.review_required)

    def test_wrong_schema_version(self):
        payload = _payload()
        payload["schema_version"] = "grounded-ai-response-v0"
        result = _validate(payload=payload)
        self.assertIn(RejectionCode.SCHEMA_VERSION_MISMATCH, result.rejection_codes)

    def test_wrong_field_types(self):
        payload = _payload()
        payload["answer"] = 12
        payload["claims"] = {"evidence": "E1"}
        result = _validate(payload=payload)
        self.assertIn(RejectionCode.SCHEMA_TYPE_ERROR, result.rejection_codes)

    def test_answer_empty(self):
        result = _validate(payload=_payload(answer="   "))
        self.assertIn(RejectionCode.ANSWER_EMPTY, result.rejection_codes)

    def test_answer_exceeds_max_chars(self):
        answer = "Profit is 5.0 [E1]. " + ("x" * (MAX_ANSWER_CHARS + 1))
        result = _validate(payload=_payload(answer=answer))
        self.assertIn(RejectionCode.SCHEMA_LIMIT_EXCEEDED, result.rejection_codes)
        self.assertEqual(MAX_ANSWER_CHARS, 2000)

    def test_claims_exceed_max(self):
        claim = {"evidence": "E1", "field": "profit", "value": "5.0"}
        result = _validate(payload=_payload(claims=[claim] * (MAX_CLAIMS + 1)))
        self.assertIn(RejectionCode.SCHEMA_LIMIT_EXCEEDED, result.rejection_codes)
        self.assertEqual(MAX_CLAIMS, 160)

    def test_citations_exceed_max(self):
        result = _validate(payload=_payload(citations=["E1"] * (MAX_CITATIONS + 1)))
        self.assertIn(RejectionCode.SCHEMA_LIMIT_EXCEEDED, result.rejection_codes)
        self.assertEqual(MAX_CITATIONS, 10)

    def test_limitations_exceed_max(self):
        result = _validate(
            payload=_payload(limitation_codes=["APPROXIMATE_M1_EVIDENCE"] * (MAX_LIMITATION_CODES + 1))
        )
        self.assertIn(RejectionCode.SCHEMA_LIMIT_EXCEEDED, result.rejection_codes)
        self.assertEqual(MAX_LIMITATION_CODES, 4)

    def test_unknown_citation_alias(self):
        result = _validate(
            payload=_payload(
                answer="Profit is 5.0 [E1] [E9].",
                citations=["E1", "E9"],
            )
        )
        self.assertIn(RejectionCode.CITATION_UNKNOWN_ALIAS, result.rejection_codes)

    def test_duplicate_citation(self):
        result = _validate(
            payload=_payload(
                answer="Profit is 5.0 [E1].",
                citations=["E1", "E1"],
            )
        )
        self.assertIn(RejectionCode.CITATION_DUPLICATE, result.rejection_codes)

    def test_missing_citations(self):
        result = _validate(payload=_payload(answer="Profit is 5.0.", citations=[]))
        self.assertIn(RejectionCode.CITATION_MISSING, result.rejection_codes)

    def test_inline_citation_mismatch(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 E1."))
        self.assertIn(RejectionCode.CITATION_INLINE_MISMATCH, result.rejection_codes)

    def test_internal_document_id_citation(self):
        result = _validate(
            payload=_payload(
                answer="Profit is 5.0 [E1].",
                citations=["TRADE_EVIDENCE:ticket:1"],
            )
        )
        self.assertIn(RejectionCode.CITATION_INTERNAL_ID, result.rejection_codes)
        self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes)

    def test_claim_alias_not_cited(self):
        request = _request(
            evidence=(
                _trade_item(alias="E1", ticket="1"),
                _trade_item(alias="E2", ticket="2"),
            )
        )
        payload = _payload(
            answer="Profit is 5.0 [E1].",
            claims=[
                {"evidence": "E1", "field": "profit", "value": "5.0"},
                {"evidence": "E2", "field": "profit", "value": "5.0"},
            ],
            citations=["E1"],
        )
        result = _validate(request=request, context=_context(request), payload=payload)
        self.assertIn(RejectionCode.CLAIM_ALIAS_NOT_CITED, result.rejection_codes)

    def test_claim_field_not_claimable(self):
        result = _validate(
            payload=_payload(claims=[{"evidence": "E1", "field": "ticket", "value": "1"}])
        )
        self.assertIn(RejectionCode.CLAIM_FIELD_NOT_CLAIMABLE, result.rejection_codes)

    def test_claim_field_unavailable(self):
        request = _request(
            evidence=(_no_market_item(),),
            limitation_codes=(LimitationCode.EVIDENCE_UNAVAILABLE,),
        )
        payload = _payload(
            answer="Profit is 5.0 [E1].",
            claims=[{"evidence": "E1", "field": "approx_mfe", "value": "0.05"}],
            limitation_codes=["EVIDENCE_UNAVAILABLE"],
        )
        result = _validate(request=request, context=_context(request), payload=payload)
        self.assertIn(RejectionCode.CLAIM_FIELD_UNAVAILABLE, result.rejection_codes)

    def test_claim_value_mismatch(self):
        result = _validate(
            payload=_payload(claims=[{"evidence": "E1", "field": "profit", "value": "9.9"}])
        )
        self.assertIn(RejectionCode.CLAIM_VALUE_MISMATCH, result.rejection_codes)

    def test_ungrounded_integer(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 and there were 99 trades [E1]."))
        self.assertIn(RejectionCode.NUMBER_UNGROUNDED, result.rejection_codes)

    def test_ungrounded_decimal(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 and also 9.9 [E1]."))
        self.assertIn(RejectionCode.NUMBER_UNGROUNDED, result.rejection_codes)

    def test_grounded_negative_decimal(self):
        payload = _payload(
            answer="Approx MAE is -54.7 [E1].",
            claims=[{"evidence": "E1", "field": "approx_mae", "value": "-54.7"}],
        )
        result = _validate(payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_ticket_number_not_misclassified_as_numeric_claim(self):
        request = _request(evidence=(_trade_item(ticket="12345"),), question="What is ticket 12345?")
        payload = _payload(answer="Ticket 12345 profit is 5.0 [E1].")
        result = _validate(request=request, context=_context(request), payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_timestamp_numbers_not_misclassified(self):
        request = _request(evidence=(_trade_item(extra=(_open_time(),)),))
        payload = _payload(answer="Opened at 2026-06-25 10:30:15 with profit 5.0 [E1].")
        result = _validate(request=request, context=_context(request), payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_explicit_unsupported_ticket(self):
        result = _validate(payload=_payload(answer="ticket 99999 profit is 5.0 [E1]."))
        self.assertIn(RejectionCode.IDENTIFIER_UNGROUNDED, result.rejection_codes)

    def test_explicit_unsupported_symbol(self):
        result = _validate(payload=_payload(answer="symbol GBPUSD profit is 5.0 [E1]."))
        self.assertIn(RejectionCode.IDENTIFIER_UNGROUNDED, result.rejection_codes)

    def test_unknown_limitation_code(self):
        result = _validate(
            payload=_payload(limitation_codes=["APPROXIMATE_M1_EVIDENCE", "NOT_A_LIMITATION"])
        )
        self.assertIn(RejectionCode.LIMITATION_UNKNOWN_CODE, result.rejection_codes)

    def test_missing_required_limitation(self):
        result = _validate(payload=_payload(limitation_codes=[]))
        self.assertIn(RejectionCode.LIMITATION_REQUIRED_MISSING, result.rejection_codes)

    def test_extra_known_limitation(self):
        result = _validate(
            payload=_payload(limitation_codes=["APPROXIMATE_M1_EVIDENCE", "EVIDENCE_UNAVAILABLE"])
        )
        self.assertIn(RejectionCode.STATUS_CONTRADICTION, result.rejection_codes)

    def test_recommendation_phrase(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 [E1]. you should buy."))
        self.assertEqual(result.rejection_codes, (RejectionCode.PROHIBITED_RECOMMENDATION,))

    def test_prediction_phrase(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 [E1]. It will rise."))
        self.assertEqual(result.rejection_codes, (RejectionCode.PROHIBITED_PREDICTION,))

    def test_precision_phrase(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 [E1]. This is the exact MFE."))
        self.assertEqual(result.rejection_codes, (RejectionCode.PROHIBITED_PRECISION_CLAIM,))

    def test_counterfactual_phrase(self):
        result = _validate(payload=_payload(answer="Profit is 5.0 [E1]. This would have made more."))
        self.assertEqual(result.rejection_codes, (RejectionCode.PROHIBITED_COUNTERFACTUAL,))

    def test_request_sha_mismatch(self):
        request = _request()
        context = _context(request)
        context = ServerGroundingContext(
            request_sha256="a" * 64,
            alias_to_document_id=context.alias_to_document_id,
            journal_fingerprint=context.journal_fingerprint,
            corpus_schema_version=context.corpus_schema_version,
            render_template_version=context.render_template_version,
            retrieval_method=context.retrieval_method,
            returned_count=context.returned_count,
            candidate_count=context.candidate_count,
            corpus_size=context.corpus_size,
        )
        result = _validate(request=request, context=context)
        self.assertEqual(result.rejection_codes, (RejectionCode.REQUEST_MISMATCH,))

    def test_server_metadata_leakage(self):
        request = _request()
        context = _context(request)
        result = _validate(
            request=request,
            context=context,
            payload=_payload(answer=f"Profit is 5.0 [E1]. {context.request_sha256}"),
        )
        self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes)

    def test_internal_document_id_leakage(self):
        result = _validate(
            payload=_payload(answer="Profit is 5.0 [E1]. TRADE_EVIDENCE:ticket:1")
        )
        self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes)
        self.assertNotIn(RejectionCode.CITATION_INTERNAL_ID, result.rejection_codes)

    def test_multiple_simultaneous_failures_are_sorted(self):
        request = _request()
        context = ServerGroundingContext(
            request_sha256="a" * 64,
            alias_to_document_id=_context(request).alias_to_document_id,
            journal_fingerprint="c" * 64,
            corpus_schema_version="tradeintel.evidence.v1",
            render_template_version="tradeintel.evidence.render.v1",
            retrieval_method="TFIDF_COSINE",
            returned_count=1,
            candidate_count=1,
            corpus_size=9,
        )
        result = _validate(
            request=request,
            context=context,
            payload=_payload(answer="Profit is 5.0 [E1]. you should sell.", review_required=True),
        )
        self.assertEqual(
            result.rejection_codes,
            (
                RejectionCode.PROHIBITED_RECOMMENDATION,
                RejectionCode.REQUEST_MISMATCH,
                RejectionCode.SCHEMA_UNKNOWN_FIELD,
            ),
        )
        values = tuple(code.value for code in result.rejection_codes)
        self.assertEqual(values, tuple(sorted(values)))

    def test_duplicate_rejection_codes_are_eliminated(self):
        payload = _payload(
            claims=[
                {"evidence": "E1", "field": "profit", "value": "9.9"},
                {"evidence": "E1", "field": "approx_mae", "value": "0.0"},
            ]
        )
        result = _validate(payload=payload)
        self.assertEqual(result.rejection_codes.count(RejectionCode.CLAIM_VALUE_MISMATCH), 1)
        self.assertEqual(len(result.rejection_codes), len(set(result.rejection_codes)))

    def test_repeat_run_determinism(self):
        first = _validate()
        second = _validate()
        self.assertEqual(first, second)
        rejected = _validate(payload=_payload(answer="Profit is 5.0 [E1]. you should buy."))
        self.assertEqual(rejected, _validate(payload=_payload(answer="Profit is 5.0 [E1]. you should buy.")))

    def test_review_required_true_on_pass(self):
        result = _validate()
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        self.assertIs(result.review_required, True)

    def test_review_required_true_on_rejected(self):
        result = _validate(candidate_json="{")
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertIs(result.review_required, True)

    def test_passed_deterministic_checks_wording(self):
        self.assertEqual(
            ValidationStatus.PASSED_DETERMINISTIC_CHECKS.value,
            "PASSED_DETERMINISTIC_CHECKS",
        )
        self.assertNotIn("VERIFIED", ValidationStatus.PASSED_DETERMINISTIC_CHECKS.value)
        self.assertNotIn("FACT", ValidationStatus.PASSED_DETERMINISTIC_CHECKS.value)

    def test_frozen_rejection_codes_unchanged(self):
        self.assertEqual(len(FROZEN_REJECTION_CODES), 30)
        self.assertEqual(tuple(code.value for code in RejectionCode), FROZEN_REJECTION_CODES)

    def test_frozen_limitation_codes_unchanged(self):
        self.assertEqual(len(FROZEN_LIMITATION_CODES), 4)
        self.assertEqual(tuple(code.value for code in LimitationCode), FROZEN_LIMITATION_CODES)

    def test_prompt_template_hash_unchanged(self):
        self.assertEqual(prompt_template_hash(), LOCKED_PROMPT_TEMPLATE_HASH)

    def test_golden_prompt_hash_unchanged(self):
        rendered = render_evidence_summary_prompt(_golden_request())
        self.assertEqual(prompt_hash(rendered), LOCKED_PROMPT_HASH)

    def test_number_extractor_is_pinned(self):
        self.assertEqual(
            ANSWER_NUMBER_RE.pattern,
            r"(?<![A-Za-z0-9._-])([+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)(%)?(?![A-Za-z0-9_-]|\.[A-Za-z0-9._-])",
        )
        self.assertEqual(
            ANSWER_NUMBER_RE.findall("Profit 5.0 and 75% and -54.7"),
            [("5.0", ""), ("75", "%"), ("-54.7", "")],
        )
        self.assertEqual(ANSWER_NUMBER_RE.findall("99."), [("99", "")])
        self.assertEqual(ANSWER_NUMBER_RE.findall("9.9."), [("9.9", "")])
        self.assertEqual(ANSWER_NUMBER_RE.findall("-54.7."), [("-54.7", "")])
        self.assertEqual(ANSWER_NUMBER_RE.findall("+5.0."), [("+5.0", "")])
        self.assertEqual(ANSWER_NUMBER_RE.findall("75%."), [("75", "%")])
        self.assertEqual(ANSWER_NUMBER_RE.findall("US100.cash"), [])

    def test_grounded_integer_followed_by_sentence_period(self):
        request = _request(evidence=(_integer_item(),), limitation_codes=())
        payload = _payload(
            answer="There were 99. [E1]",
            claims=[{"evidence": "E1", "field": "total_trades", "value": "99"}],
            limitation_codes=[],
        )
        result = _validate(
            request=request,
            context=_context(request, "KPI_EVIDENCE:kpi:total_trades"),
            payload=payload,
        )
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_grounded_decimal_followed_by_sentence_period(self):
        result = _validate(payload=_payload(answer="Profit is 5.0. [E1]"))
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_grounded_negative_decimal_followed_by_sentence_period(self):
        payload = _payload(
            answer="Approx MAE is -54.7. [E1]",
            claims=[{"evidence": "E1", "field": "approx_mae", "value": "-54.7"}],
        )
        result = _validate(payload=payload)
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_grounded_percentage_followed_by_sentence_period(self):
        request = _request(evidence=(_kpi_item(),), limitation_codes=())
        payload = _payload(
            answer="Win rate is 75%. [E1]",
            claims=[{"evidence": "E1", "field": "win_rate_pct", "value": "75"}],
            limitation_codes=[],
        )
        result = _validate(
            request=request,
            context=_context(request, "KPI_EVIDENCE:kpi:win_rate_pct"),
            payload=payload,
        )
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_ungrounded_integer_followed_by_sentence_period(self):
        result = _validate(payload=_payload(answer="Unsupported value is 99. [E1]"))
        self.assertIn(RejectionCode.NUMBER_UNGROUNDED, result.rejection_codes)

    def test_ungrounded_decimal_followed_by_sentence_period(self):
        result = _validate(payload=_payload(answer="Unsupported value is 9.9. [E1]"))
        self.assertIn(RejectionCode.NUMBER_UNGROUNDED, result.rejection_codes)

    def test_ungrounded_negative_decimal_followed_by_sentence_period(self):
        result = _validate(payload=_payload(answer="Unsupported value is -12.5. [E1]"))
        self.assertIn(RejectionCode.NUMBER_UNGROUNDED, result.rejection_codes)

    def test_supplied_identifiers_with_terminal_period(self):
        result = _validate(payload=_payload(answer="Ticket 1. Profit is 5.0 [E1]."))
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        request = _request(evidence=(_trade_item(ticket="12345"),), question="What is ticket 12345?")
        result = _validate(
            request=request,
            context=_context(request),
            payload=_payload(answer="Ticket 12345. Profit is 5.0 [E1]."),
        )
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        result = _validate(payload=_payload(answer="Symbol EURUSD. Profit is 5.0 [E1]."))
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)
        request = _request(evidence=(_trade_item(symbol="US100.cash"),))
        result = _validate(
            request=request,
            context=_context(request),
            payload=_payload(answer="Symbol US100.cash. Profit is 5.0 [E1]."),
        )
        self.assertEqual(result.status, ValidationStatus.PASSED_DETERMINISTIC_CHECKS)

    def test_unsupported_identifier_followed_by_period_still_rejects(self):
        result = _validate(payload=_payload(answer="ticket 99999. Profit is 5.0 [E1]."))
        self.assertIn(RejectionCode.IDENTIFIER_UNGROUNDED, result.rejection_codes)
        result = _validate(payload=_payload(answer="symbol GBPUSD. Profit is 5.0 [E1]."))
        self.assertIn(RejectionCode.IDENTIFIER_UNGROUNDED, result.rejection_codes)

    def test_invalid_constraints_request_fails_closed(self):
        request = GroundedAIRequest(
            schema_version=GROUNDING_REQUEST_SCHEMA_VERSION,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            response_schema_version=GROUNDING_RESPONSE_SCHEMA_VERSION,
            task_type=TASK_TYPE,
            question="What is ticket 1?",
            evidence=(_trade_item(),),
            required_limitation_codes=(LimitationCode.APPROXIMATE_M1_EVIDENCE,),
            constraints=GroundingConstraints(
                cite_only_supplied_aliases=True,
                no_recommendations=False,
                no_predictions=True,
                no_counterfactuals=True,
                no_causal_explanations=True,
                max_answer_chars=MAX_ANSWER_CHARS,
                max_claims=MAX_CLAIMS,
                max_citations=MAX_CITATIONS,
                max_limitation_codes=MAX_LIMITATION_CODES,
            ),
        )
        result = validate_grounded_response(request, _detached_context(), _dumps(_payload()))
        self.assertEqual(result.status, ValidationStatus.REJECTED)
        self.assertIs(result.review_required, True)
        self.assertIn(RejectionCode.REQUEST_MISMATCH, result.rejection_codes)

    def test_malformed_evidence_request_fails_closed_without_exception(self):
        request = GroundedAIRequest(
            schema_version=GROUNDING_REQUEST_SCHEMA_VERSION,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            response_schema_version=GROUNDING_RESPONSE_SCHEMA_VERSION,
            task_type=TASK_TYPE,
            question="What is ticket 1?",
            evidence=(object(),),
            required_limitation_codes=(LimitationCode.APPROXIMATE_M1_EVIDENCE,),
            constraints=DEFAULT_CONSTRAINTS,
        )
        first = validate_grounded_response(request, _detached_context(), _dumps(_payload()))
        second = validate_grounded_response(request, _detached_context(), _dumps(_payload()))
        self.assertEqual(first, second)
        self.assertEqual(first.status, ValidationStatus.REJECTED)
        self.assertIs(first.review_required, True)
        listed = GroundedAIRequest(
            schema_version=GROUNDING_REQUEST_SCHEMA_VERSION,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            response_schema_version=GROUNDING_RESPONSE_SCHEMA_VERSION,
            task_type=TASK_TYPE,
            question="What is ticket 1?",
            evidence=[],
            required_limitation_codes=(LimitationCode.APPROXIMATE_M1_EVIDENCE,),
            constraints=DEFAULT_CONSTRAINTS,
        )
        listed_result = validate_grounded_response(listed, _detached_context(), _dumps(_payload()))
        self.assertEqual(listed_result.status, ValidationStatus.REJECTED)
        self.assertIs(listed_result.review_required, True)
        self.assertIn(RejectionCode.SCHEMA_TYPE_ERROR, listed_result.rejection_codes)

    def test_unicode_escaped_request_sha256_key_is_leakage(self):
        key = _unicode_escape("request_sha256")
        raw = (
            '{"schema_version":"grounded-ai-response-v1",'
            '"answer":"Profit is 5.0 [E1].",'
            '"claims":[{"evidence":"E1","field":"profit","value":"5.0"}],'
            '"citations":["E1"],'
            '"limitation_codes":["APPROXIMATE_M1_EVIDENCE"],'
            f'"{key}":"x"}}'
        )
        self.assertNotIn("request_sha256", raw)
        result = _validate(candidate_json=raw)
        self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes)

    def test_unicode_escaped_request_sha_value_is_leakage(self):
        request = _request()
        context = _context(request)
        escaped_sha = _unicode_escape(context.request_sha256)
        raw = (
            '{"schema_version":"grounded-ai-response-v1",'
            f'"answer":"Profit is 5.0 [E1]. {escaped_sha}",'
            '"claims":[{"evidence":"E1","field":"profit","value":"5.0"}],'
            '"citations":["E1"],'
            '"limitation_codes":["APPROXIMATE_M1_EVIDENCE"]}'
        )
        self.assertNotIn(context.request_sha256, raw)
        result = _validate(request=request, context=context, candidate_json=raw)
        self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes)

    def test_unicode_escaped_internal_document_id_is_leakage(self):
        document_id = "TRADE_EVIDENCE:ticket:1"
        escaped = _unicode_escape(document_id)
        raw = (
            '{"schema_version":"grounded-ai-response-v1",'
            f'"answer":"Profit is 5.0 [E1]. {escaped}",'
            '"claims":[{"evidence":"E1","field":"profit","value":"5.0"}],'
            '"citations":["E1"],'
            '"limitation_codes":["APPROXIMATE_M1_EVIDENCE"]}'
        )
        self.assertNotIn(document_id, raw)
        result = _validate(candidate_json=raw)
        self.assertIn(RejectionCode.LEAKAGE_DETECTED, result.rejection_codes)
        self.assertNotIn(RejectionCode.CITATION_INTERNAL_ID, result.rejection_codes)

    def test_parenthesis_and_bare_alias_are_not_inline_citations(self):
        for answer in ("Profit is 5.0 (E1).", "Profit is 5.0 {E1}.", "Profit is 5.0 TRADE_EVIDENCE:ticket:1."):
            result = _validate(payload=_payload(answer=answer))
            self.assertIn(RejectionCode.CITATION_INLINE_MISMATCH, result.rejection_codes)

    def test_no_provider_or_network_imports(self):
        blocked = {
            "anthropic",
            "openai",
            "google.generativeai",
            "requests",
            "httpx",
            "aiohttp",
            "urllib.request",
            "socket",
        }
        path = Path(__file__).resolve().parent / "ai_grounding" / "validation.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            for name in names:
                self.assertNotIn(name, blocked)
                self.assertFalse(any(name.startswith(item + ".") for item in blocked))
        lowered = source.lower()
        self.assertNotIn("api_key", lowered)
        self.assertNotIn("anthropic", lowered)
        self.assertNotIn("openai", lowered)
