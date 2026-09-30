from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .retrieval import (
    RETRIEVAL_METHOD_EXACT_TICKET,
    RETRIEVAL_METHOD_TFIDF_COSINE,
    RETRIEVAL_STATE_CORPUS_INVALID,
    RETRIEVAL_STATE_EMPTY_QUERY,
    RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
    RETRIEVAL_STATE_OK,
    RetrievalError,
    retrieve_evidence,
)
from .schema import (
    DATASET_DOCUMENT_ID,
    DATASET_SOURCE_RECORD_KEY,
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    EVIDENCE_STATUS_NO_M1,
    EvidenceDocument,
    kpi_document_id,
    make_evidence_document,
    trade_document_id,
)
from .scope import RetrievalScope

PRECISION_K = 5
EVAL_FINGERPRINT = "c" * 64
ALT_FINGERPRINT = "d" * 64

RETRIEVAL_STATES = frozenset(
    {
        RETRIEVAL_STATE_OK,
        RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
        RETRIEVAL_STATE_EMPTY_QUERY,
        RETRIEVAL_STATE_CORPUS_INVALID,
    }
)


class EvaluationError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    scope: RetrievalScope
    documents: tuple[EvidenceDocument, ...]
    query: str
    expected_state: str
    relevant_document_ids: tuple[str, ...]
    single_relevant: bool
    ticket: str | None = None
    symbol: str | None = None
    document_type: str | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    case_id: str
    expected_state: str
    actual_state: str
    contract_passed: bool
    failure_codes: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    retrieval_methods: tuple[str, ...]
    lexical_scores: tuple[float | None, ...]
    corpus_size: int
    candidate_count: int
    result_count: int
    top_k: int
    single_relevant: bool
    eligible_single: bool
    eligible_multi: bool
    hit_at_1: float | None
    hit_at_3: float | None
    mrr: float | None
    precision_at_k: float | None

    def as_plain(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "contract_passed": self.contract_passed,
            "failure_codes": list(self.failure_codes),
            "retrieved_document_ids": list(self.retrieved_document_ids),
            "ranks": list(self.ranks),
            "retrieval_methods": list(self.retrieval_methods),
            "lexical_scores": list(self.lexical_scores),
            "corpus_size": self.corpus_size,
            "candidate_count": self.candidate_count,
            "result_count": self.result_count,
            "top_k": self.top_k,
            "single_relevant": self.single_relevant,
            "eligible_single": self.eligible_single,
            "eligible_multi": self.eligible_multi,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "mrr": self.mrr,
            "precision_at_k": self.precision_at_k,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total_cases: int
    contract_cases_passed: int
    contract_pass_rate: float
    single_relevant_cases: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    multi_relevant_cases: int
    precision_at_k: float
    precision_k: int
    results: tuple[EvaluationCaseResult, ...]

    def as_plain(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "contract_cases_passed": self.contract_cases_passed,
            "contract_pass_rate": self.contract_pass_rate,
            "single_relevant_cases": self.single_relevant_cases,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "mrr": self.mrr,
            "multi_relevant_cases": self.multi_relevant_cases,
            "precision_at_k": self.precision_at_k,
            "precision_k": self.precision_k,
            "results": [item.as_plain() for item in self.results],
        }


def first_relevant_rank(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
) -> int | None:
    relevant = set(relevant_document_ids)
    for index, document_id in enumerate(retrieved_document_ids, start=1):
        if document_id in relevant:
            return index
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0


def mean_reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / rank


def precision_at_k(
    retrieved_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
    k: int,
) -> float:
    if k < 1:
        raise EvaluationError("INVALID_PRECISION_K")
    relevant = set(relevant_document_ids)
    considered = retrieved_document_ids[: min(k, len(retrieved_document_ids))]
    hits = sum(1 for document_id in considered if document_id in relevant)
    return hits / k


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _validate_case(case: EvaluationCase) -> None:
    if not isinstance(case.case_id, str) or not case.case_id.strip():
        raise EvaluationError("INVALID_CASE_ID")
    if not isinstance(case.scope, RetrievalScope):
        raise EvaluationError("INVALID_SCOPE")
    if not isinstance(case.documents, tuple) or not case.documents:
        raise EvaluationError("INVALID_DOCUMENTS")
    if case.expected_state not in RETRIEVAL_STATES:
        raise EvaluationError("INVALID_EXPECTED_STATE")
    if not isinstance(case.relevant_document_ids, tuple):
        raise EvaluationError("INVALID_RELEVANT_IDS")
    if not isinstance(case.single_relevant, bool):
        raise EvaluationError("INVALID_RELEVANCE_FLAG")
    if case.single_relevant and len(case.relevant_document_ids) != 1:
        raise EvaluationError("INVALID_SINGLE_RELEVANT")
    if (
        case.expected_state == RETRIEVAL_STATE_OK
        and not case.single_relevant
        and len(case.relevant_document_ids) < 2
    ):
        raise EvaluationError("INVALID_MULTI_RELEVANT")
    document_ids = {item.document_id for item in case.documents}
    for document_id in case.relevant_document_ids:
        if document_id not in document_ids:
            raise EvaluationError("UNKNOWN_RELEVANT_ID")


def _contract_failures(case: EvaluationCase, response: Any) -> list[str]:
    failures: list[str] = []
    if response.state != case.expected_state:
        failures.append("STATE_MISMATCH")
    if response.corpus_size < 0:
        failures.append("INVALID_CORPUS_SIZE")
    if response.candidate_count < 0:
        failures.append("INVALID_CANDIDATE_COUNT")
    if len(response.results) > response.candidate_count:
        failures.append("RESULT_COUNT_EXCEEDS_CANDIDATES")
    if len(response.results) > response.top_k:
        failures.append("RESULT_COUNT_EXCEEDS_TOP_K")
    expected_ranks = tuple(range(1, len(response.results) + 1))
    actual_ranks = tuple(item.rank for item in response.results)
    if actual_ranks != expected_ranks:
        failures.append("RANK_SEQUENCE_INVALID")
    for item in response.results:
        if item.journal_fingerprint != case.scope.journal_fingerprint:
            failures.append("FINGERPRINT_MISMATCH")
            break
    if case.ticket is None:
        if any(
            item.retrieval_method == RETRIEVAL_METHOD_EXACT_TICKET
            for item in response.results
        ):
            failures.append("EXACT_TICKET_WITHOUT_TICKET")
    elif response.state == RETRIEVAL_STATE_OK:
        for item in response.results:
            if item.retrieval_method != RETRIEVAL_METHOD_EXACT_TICKET:
                failures.append("EXACT_TICKET_METHOD")
                break
            if item.score is not None:
                failures.append("EXACT_TICKET_SCORE_PRESENT")
                break
    if case.ticket is None and response.state == RETRIEVAL_STATE_OK:
        for item in response.results:
            if item.retrieval_method != RETRIEVAL_METHOD_TFIDF_COSINE:
                failures.append("TFIDF_METHOD")
                break
            score = item.score
            if score is None or not math.isfinite(score) or score <= 0 or score > 1:
                failures.append("TFIDF_SCORE_INVALID")
                break
    if response.state != RETRIEVAL_STATE_OK and response.results:
        failures.append("NON_OK_HAS_RESULTS")
    if response.state == RETRIEVAL_STATE_CORPUS_INVALID and (
        response.corpus_size != 0 or response.candidate_count != 0
    ):
        failures.append("CORPUS_INVALID_COUNTS")
    return failures


def evaluate_case(case: EvaluationCase) -> EvaluationCaseResult:
    _validate_case(case)
    try:
        response = retrieve_evidence(
            case.documents,
            case.scope,
            query=case.query,
            ticket=case.ticket,
            symbol=case.symbol,
            document_type=case.document_type,
            top_k=case.top_k,
        )
    except RetrievalError as exc:
        raise EvaluationError("RETRIEVAL_ERROR", exc.reason) from exc
    failures = tuple(_contract_failures(case, response))
    contract_passed = not failures
    retrieved_ids = tuple(item.document_id for item in response.results)
    rank = first_relevant_rank(retrieved_ids, case.relevant_document_ids)
    eligible_single = (
        contract_passed
        and case.expected_state == RETRIEVAL_STATE_OK
        and case.single_relevant
    )
    eligible_multi = (
        contract_passed
        and case.expected_state == RETRIEVAL_STATE_OK
        and not case.single_relevant
    )
    return EvaluationCaseResult(
        case_id=case.case_id,
        expected_state=case.expected_state,
        actual_state=response.state,
        contract_passed=contract_passed,
        failure_codes=failures,
        retrieved_document_ids=retrieved_ids,
        ranks=tuple(item.rank for item in response.results),
        retrieval_methods=tuple(item.retrieval_method for item in response.results),
        lexical_scores=tuple(item.score for item in response.results),
        corpus_size=response.corpus_size,
        candidate_count=response.candidate_count,
        result_count=len(response.results),
        top_k=response.top_k,
        single_relevant=case.single_relevant,
        eligible_single=eligible_single,
        eligible_multi=eligible_multi,
        hit_at_1=hit_at_k(rank, 1) if eligible_single else None,
        hit_at_3=hit_at_k(rank, 3) if eligible_single else None,
        mrr=mean_reciprocal_rank(rank) if eligible_single else None,
        precision_at_k=(
            precision_at_k(retrieved_ids, case.relevant_document_ids, PRECISION_K)
            if eligible_multi
            else None
        ),
    )


def evaluate_cases(cases: Sequence[EvaluationCase]) -> EvaluationSummary:
    if not cases:
        raise EvaluationError("EMPTY_CASE_SET")
    seen: set[str] = set()
    results: list[EvaluationCaseResult] = []
    for case in cases:
        if case.case_id in seen:
            raise EvaluationError("DUPLICATE_CASE_ID")
        seen.add(case.case_id)
        results.append(evaluate_case(case))
    passed = sum(1 for item in results if item.contract_passed)
    single_hits_1 = tuple(item.hit_at_1 or 0.0 for item in results if item.eligible_single)
    single_hits_3 = tuple(item.hit_at_3 or 0.0 for item in results if item.eligible_single)
    single_mrr = tuple(item.mrr or 0.0 for item in results if item.eligible_single)
    multi_precision = tuple(
        item.precision_at_k or 0.0 for item in results if item.eligible_multi
    )
    return EvaluationSummary(
        total_cases=len(results),
        contract_cases_passed=passed,
        contract_pass_rate=passed / len(results),
        single_relevant_cases=len(single_hits_1),
        hit_at_1=_mean(single_hits_1),
        hit_at_3=_mean(single_hits_3),
        mrr=_mean(single_mrr),
        multi_relevant_cases=len(multi_precision),
        precision_at_k=_mean(multi_precision),
        precision_k=PRECISION_K,
        results=tuple(results),
    )


def _trade(
    ticket: str,
    content: str,
    symbol: str | None,
    fingerprint: str = EVAL_FINGERPRINT,
) -> EvidenceDocument:
    return make_evidence_document(
        document_id=trade_document_id(ticket),
        document_type=DOCUMENT_TYPE_TRADE,
        journal_fingerprint=fingerprint,
        source_record_key=ticket,
        content=content,
        evidence_status=EVIDENCE_STATUS_NO_M1,
        provenance={"source_record_key": ticket, "symbol": symbol},
    )


def _kpi(
    key: str,
    content: str,
    fingerprint: str = EVAL_FINGERPRINT,
) -> EvidenceDocument:
    return make_evidence_document(
        document_id=kpi_document_id(key),
        document_type=DOCUMENT_TYPE_KPI,
        journal_fingerprint=fingerprint,
        source_record_key=key,
        content=content,
        evidence_status="AUTHORITATIVE",
        provenance={"source_record_key": key},
    )


def _dataset(
    content: str,
    fingerprint: str = EVAL_FINGERPRINT,
) -> EvidenceDocument:
    return make_evidence_document(
        document_id=DATASET_DOCUMENT_ID,
        document_type=DOCUMENT_TYPE_DATASET,
        journal_fingerprint=fingerprint,
        source_record_key=DATASET_SOURCE_RECORD_KEY,
        content=content,
        evidence_status="BOUND",
        provenance={"source_record_key": DATASET_SOURCE_RECORD_KEY},
    )


def _scope(fingerprint: str = EVAL_FINGERPRINT) -> RetrievalScope:
    return RetrievalScope(owner_id=1, journal_fingerprint=fingerprint)


def _base_documents(fingerprint: str = EVAL_FINGERPRINT) -> tuple[EvidenceDocument, ...]:
    return (
        _trade(
            "1",
            "Ticket: 1\nType: buy\nkiwifruit orchard commission -54.7 sharedlex",
            "NZDUSD",
            fingerprint,
        ),
        _trade(
            "2",
            "Ticket: 2\nType: sell\nbaguette bakery commission 1.20 sharedlex",
            "EURUSD",
            fingerprint,
        ),
        _trade(
            "10",
            "Ticket: 10\nType: buy\nnasdaq cash mfe 70.85 sharedlex tielex",
            "US100.cash",
            fingerprint,
        ),
        _trade(
            "4",
            "Ticket: 4\nType: buy\nparis spread 0.2",
            "EURUSD",
            fingerprint,
        ),
        _trade(
            "5",
            "Ticket: 5\nType: sell\nsydney session close",
            "AUDUSD",
            fingerprint,
        ),
        _trade(
            "9",
            "Ticket: 9\nType: sell\nsterling pound swap 0.10 tielex",
            "GBPUSD",
            fingerprint,
        ),
        _kpi("win_rate", "win_rate: 42.5 lexicalkpi", fingerprint),
        _kpi("total_trades", "total_trades: 6 countkpi", fingerprint),
        _dataset("Journal row count: 6 datasetcontext", fingerprint),
    )


def _case(
    case_id: str,
    *,
    query: str,
    expected_state: str,
    relevant: tuple[str, ...],
    single_relevant: bool,
    documents: tuple[EvidenceDocument, ...] | None = None,
    scope: RetrievalScope | None = None,
    ticket: str | None = None,
    symbol: str | None = None,
    document_type: str | None = None,
    top_k: int | None = None,
) -> EvaluationCase:
    docs = documents if documents is not None else _base_documents()
    return EvaluationCase(
        case_id=case_id,
        scope=scope if scope is not None else _scope(),
        documents=docs,
        query=query,
        expected_state=expected_state,
        relevant_document_ids=relevant,
        single_relevant=single_relevant,
        ticket=ticket,
        symbol=symbol,
        document_type=document_type,
        top_k=top_k,
    )


def build_synthetic_evaluation_cases() -> tuple[EvaluationCase, ...]:
    docs = _base_documents()
    t1 = trade_document_id("1")
    t2 = trade_document_id("2")
    t4 = trade_document_id("4")
    t5 = trade_document_id("5")
    t9 = trade_document_id("9")
    t10 = trade_document_id("10")
    kpi_win = kpi_document_id("win_rate")
    dataset_id = DATASET_DOCUMENT_ID
    buys = (t1, t10, t4)
    sells = (t2, t5, t9)
    mixed = (replace(docs[0], journal_fingerprint=ALT_FINGERPRINT), *docs[1:])
    return (
        _case(
            "EXACT_T1",
            query="",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
            ticket="1",
        ),
        _case(
            "EXACT_T2",
            query="",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t2,),
            single_relevant=True,
            ticket="2",
        ),
        _case(
            "EXACT_T4",
            query="",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t4,),
            single_relevant=True,
            ticket="4",
        ),
        _case(
            "EXACT_T5",
            query="",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t5,),
            single_relevant=True,
            ticket="5",
        ),
        _case(
            "EXACT_T9",
            query="",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t9,),
            single_relevant=True,
            ticket="9",
        ),
        _case(
            "EXACT_T10",
            query="",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t10,),
            single_relevant=True,
            ticket="10",
        ),
        _case(
            "UNK_T99",
            query="",
            expected_state=RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            relevant=(),
            single_relevant=False,
            ticket="99",
        ),
        _case(
            "UNK_T0",
            query="",
            expected_state=RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            relevant=(),
            single_relevant=False,
            ticket="0",
        ),
        _case(
            "UNK_T88",
            query="ignored",
            expected_state=RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            relevant=(),
            single_relevant=False,
            ticket="88",
        ),
        _case(
            "LEX_KIWI",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
        ),
        _case(
            "LEX_BAGUETTE",
            query="baguette",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t2,),
            single_relevant=True,
        ),
        _case(
            "LEX_NASDAQ",
            query="nasdaq",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t10,),
            single_relevant=True,
        ),
        _case(
            "LEX_STERLING",
            query="sterling",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t9,),
            single_relevant=True,
        ),
        _case(
            "LEX_PARIS",
            query="paris",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t4,),
            single_relevant=True,
        ),
        _case(
            "LEX_SYDNEY",
            query="sydney",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t5,),
            single_relevant=True,
        ),
        _case(
            "LEX_SIGNED",
            query="-54.7",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
        ),
        _case(
            "LEX_MFE_NUM",
            query="70.85",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t10,),
            single_relevant=True,
        ),
        _case(
            "LEX_KPI_WIN",
            query="lexicalkpi",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(kpi_win,),
            single_relevant=True,
        ),
        _case(
            "LEX_DATASET",
            query="datasetcontext",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(dataset_id,),
            single_relevant=True,
        ),
        _case(
            "LEX_KIWI_REPEAT",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
        ),
        _case(
            "LEX_SPREAD",
            query="spread",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t4,),
            single_relevant=True,
        ),
        _case(
            "LEX_ORCHARD",
            query="orchard",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
        ),
        _case(
            "MULTI_COMMISSION",
            query="commission",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1, t2),
            single_relevant=False,
        ),
        _case(
            "MULTI_BUY",
            query="buy",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=buys,
            single_relevant=False,
        ),
        _case(
            "MULTI_SELL",
            query="sell",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=sells,
            single_relevant=False,
        ),
        _case(
            "MULTI_SHAREDLEX",
            query="sharedlex",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1, t2, t10),
            single_relevant=False,
        ),
        _case(
            "MULTI_TIELEX",
            query="tielex",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t9, t10),
            single_relevant=False,
        ),
        _case(
            "MULTI_TICKET_WORD",
            query="ticket",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1, t2, t10, t4, t5, t9),
            single_relevant=False,
        ),
        _case(
            "SYM_NZD_KIWI",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
            symbol="nzdusd",
        ),
        _case(
            "SYM_US100",
            query="nasdaq",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t10,),
            single_relevant=True,
            symbol="US100.cash",
        ),
        _case(
            "SYM_GBP",
            query="sterling",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t9,),
            single_relevant=True,
            symbol="gbpusd",
        ),
        _case(
            "SYM_UNKNOWN",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            relevant=(),
            single_relevant=False,
            symbol="XYZUSD",
        ),
        _case(
            "DTYPE_KPI",
            query="lexicalkpi",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(kpi_win,),
            single_relevant=True,
            document_type=DOCUMENT_TYPE_KPI,
        ),
        _case(
            "DTYPE_TRADE",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
            document_type=DOCUMENT_TYPE_TRADE,
        ),
        _case(
            "DTYPE_DATASET",
            query="datasetcontext",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(dataset_id,),
            single_relevant=True,
            document_type=DOCUMENT_TYPE_DATASET,
        ),
        _case(
            "ZERO_ZZZ",
            query="zzzznotpresent",
            expected_state=RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            relevant=(),
            single_relevant=False,
        ),
        _case(
            "ZERO_QWERTY",
            query="qwertyunknown",
            expected_state=RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            relevant=(),
            single_relevant=False,
        ),
        _case(
            "EMPTY",
            query="",
            expected_state=RETRIEVAL_STATE_EMPTY_QUERY,
            relevant=(),
            single_relevant=False,
        ),
        _case(
            "WHITESPACE",
            query="   \n\t  ",
            expected_state=RETRIEVAL_STATE_EMPTY_QUERY,
            relevant=(),
            single_relevant=False,
        ),
        _case(
            "INVALID_MIXED",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_CORPUS_INVALID,
            relevant=(),
            single_relevant=False,
            documents=mixed,
        ),
        _case(
            "INVALID_WRONG_SCOPE",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_CORPUS_INVALID,
            relevant=(),
            single_relevant=False,
            scope=_scope(ALT_FINGERPRINT),
        ),
        _case(
            "TOPK_3_BUY",
            query="buy",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=buys,
            single_relevant=False,
            top_k=3,
        ),
        _case(
            "TOPK_1_KIWI",
            query="kiwifruit",
            expected_state=RETRIEVAL_STATE_OK,
            relevant=(t1,),
            single_relevant=True,
            top_k=1,
        ),
    )


def evaluate_synthetic_baseline() -> EvaluationSummary:
    return evaluate_cases(build_synthetic_evaluation_cases())


__all__ = [
    "PRECISION_K",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationError",
    "EvaluationSummary",
    "build_synthetic_evaluation_cases",
    "evaluate_case",
    "evaluate_cases",
    "evaluate_synthetic_baseline",
    "first_relevant_rank",
    "hit_at_k",
    "mean_reciprocal_rank",
    "precision_at_k",
]
