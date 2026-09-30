from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from ..excursion import canonical_ticket
from .corpus import validate_evidence_corpus
from .schema import (
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    SUPPORTED_DOCUMENT_TYPES,
    CorpusError,
    EvidenceDocument,
    freeze_provenance,
    provenance_as_plain,
)
from .scope import RetrievalScope

RETRIEVAL_STATE_OK = "OK"
RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
RETRIEVAL_STATE_EMPTY_QUERY = "EMPTY_QUERY"
RETRIEVAL_STATE_CORPUS_INVALID = "CORPUS_INVALID"

RETRIEVAL_METHOD_EXACT_TICKET = "EXACT_TICKET"
RETRIEVAL_METHOD_TFIDF_COSINE = "TFIDF_COSINE"

DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MAX_QUERY_LENGTH = 500
SCORE_QUANTUM = 1e-9

DOCUMENT_TYPE_PRIORITY = {
    DOCUMENT_TYPE_TRADE: 0,
    DOCUMENT_TYPE_KPI: 1,
    DOCUMENT_TYPE_DATASET: 2,
}

TOKEN_PATTERN = re.compile(r"-?\d+(?:\.\d+)?|[a-z0-9]+")


class RetrievalError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    rank: int
    document_id: str
    document_type: str
    source_record_key: str
    content: str
    journal_fingerprint: str
    evidence_status: str
    retrieval_method: str
    score: float | None
    provenance: Mapping[str, Any]

    def as_plain(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "source_record_key": self.source_record_key,
            "content": self.content,
            "journal_fingerprint": self.journal_fingerprint,
            "evidence_status": self.evidence_status,
            "retrieval_method": self.retrieval_method,
            "score": self.score,
            "provenance": provenance_as_plain(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    state: str
    query: str
    results: tuple[RetrievalResult, ...]
    top_k: int
    corpus_size: int
    candidate_count: int

    def as_plain(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "query": self.query,
            "top_k": self.top_k,
            "corpus_size": self.corpus_size,
            "candidate_count": self.candidate_count,
            "results": [item.as_plain() for item in self.results],
        }


def tokenize(text: str) -> tuple[str, ...]:
    normalised = unicodedata.normalize("NFKC", text).lower()
    return tuple(TOKEN_PATTERN.findall(normalised))


def _normalise_top_k(top_k: Any) -> int:
    if top_k is None:
        return DEFAULT_TOP_K
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RetrievalError("INVALID_TOP_K")
    if top_k < 1:
        raise RetrievalError("INVALID_TOP_K")
    if top_k > MAX_TOP_K:
        return MAX_TOP_K
    return top_k


def _normalise_query(query: Any) -> str:
    if query is None:
        return ""
    if not isinstance(query, str):
        raise RetrievalError("INVALID_QUERY")
    normalised = query.strip()
    if len(normalised) > MAX_QUERY_LENGTH:
        raise RetrievalError("INVALID_QUERY")
    return normalised


def _explicit_ticket(ticket: Any) -> str | None:
    if ticket is None:
        return None
    if isinstance(ticket, bool):
        raise RetrievalError("INVALID_TICKET")
    if isinstance(ticket, str) and not ticket.strip():
        return None
    canonical = canonical_ticket(ticket)
    if not canonical:
        raise RetrievalError("INVALID_TICKET")
    return canonical


def _explicit_symbol(symbol: Any) -> str | None:
    if symbol is None:
        return None
    if not isinstance(symbol, str):
        raise RetrievalError("INVALID_SYMBOL")
    normalised = unicodedata.normalize("NFKC", symbol).strip()
    if not normalised:
        return None
    return normalised


def _explicit_document_type(document_type: Any) -> str | None:
    if document_type is None:
        return None
    if not isinstance(document_type, str) or not document_type.strip():
        raise RetrievalError("INVALID_DOCUMENT_TYPE")
    if document_type not in SUPPORTED_DOCUMENT_TYPES:
        raise RetrievalError("INVALID_DOCUMENT_TYPE")
    return document_type


def _trade_symbol(document: EvidenceDocument) -> str | None:
    if document.document_type != DOCUMENT_TYPE_TRADE:
        return None
    provenance = document.provenance
    if not isinstance(provenance, Mapping):
        return None
    symbol = provenance.get("symbol")
    if not isinstance(symbol, str):
        return None
    stripped = symbol.strip()
    return stripped or None


def _symbol_matches(document: EvidenceDocument, requested: str) -> bool:
    actual = _trade_symbol(document)
    if actual is None:
        return False
    left = unicodedata.normalize("NFKC", actual).casefold()
    right = unicodedata.normalize("NFKC", requested).casefold()
    return left == right


def _quantise_score(score: float) -> float:
    return round(score / SCORE_QUANTUM) * SCORE_QUANTUM


def _source_sort_key(source_record_key: str) -> tuple[int, int | str]:
    canonical = canonical_ticket(source_record_key) or source_record_key
    try:
        number = Decimal(canonical)
    except (InvalidOperation, ValueError):
        return (1, canonical)
    if not number.is_finite():
        return (1, canonical)
    integral = number.to_integral_value()
    if number == integral:
        return (0, int(integral))
    return (1, canonical)


def _rank_sort_key(
    score: float | None,
    document: EvidenceDocument,
) -> tuple[float, int, tuple[int, int | str], str]:
    quantised = 0.0 if score is None else -_quantise_score(score)
    priority = DOCUMENT_TYPE_PRIORITY.get(document.document_type, 99)
    return (
        quantised,
        priority,
        _source_sort_key(document.source_record_key),
        document.document_id,
    )


def _empty_response(
    state: str,
    query: str,
    top_k: int,
    *,
    corpus_size: int,
    candidate_count: int,
) -> RetrievalResponse:
    return RetrievalResponse(
        state=state,
        query=query,
        results=(),
        top_k=top_k,
        corpus_size=corpus_size,
        candidate_count=candidate_count,
    )


def _token_counts(tokens: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _l2_normalise(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _tfidf_scores(
    query_tokens: Sequence[str],
    documents: Sequence[EvidenceDocument],
) -> list[tuple[float, EvidenceDocument]]:
    tokenised = [tokenize(document.content) for document in documents]
    document_counts = [_token_counts(tokens) for tokens in tokenised]
    vocabulary = tuple(sorted({token for tokens in tokenised for token in tokens}))
    if not vocabulary:
        return []
    n_documents = len(documents)
    idf: dict[str, float] = {}
    for term in vocabulary:
        df = sum(1 for counts in document_counts if counts.get(term, 0) > 0)
        idf[term] = math.log((1 + n_documents) / (1 + df)) + 1
    query_counts = _token_counts(query_tokens)
    query_vector = _l2_normalise(
        [query_counts.get(term, 0) * idf[term] for term in vocabulary]
    )
    if all(value == 0 for value in query_vector):
        return []
    scored: list[tuple[float, EvidenceDocument]] = []
    for document, counts in zip(documents, document_counts, strict=True):
        document_vector = _l2_normalise(
            [counts.get(term, 0) * idf[term] for term in vocabulary]
        )
        score = sum(
            left * right for left, right in zip(query_vector, document_vector, strict=True)
        )
        scored.append((_quantise_score(score), document))
    return scored


def _make_result(
    rank: int,
    document: EvidenceDocument,
    *,
    method: str,
    score: float | None,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        document_id=document.document_id,
        document_type=document.document_type,
        source_record_key=document.source_record_key,
        content=document.content,
        journal_fingerprint=document.journal_fingerprint,
        evidence_status=document.evidence_status,
        retrieval_method=method,
        score=score,
        provenance=freeze_provenance(provenance_as_plain(document.provenance)),
    )


def _assert_scope(
    documents: Sequence[EvidenceDocument],
    scope: RetrievalScope,
) -> None:
    for document in documents:
        if document.journal_fingerprint != scope.journal_fingerprint:
            raise RetrievalError("SCOPE_VIOLATION")


def _finalise(
    ranked: Sequence[tuple[float | None, EvidenceDocument]],
    *,
    method: str,
    query: str,
    top_k: int,
    scope: RetrievalScope,
    corpus_size: int,
    candidate_count: int,
) -> RetrievalResponse:
    selected = list(ranked[:top_k])
    _assert_scope([document for _, document in selected], scope)
    results = tuple(
        _make_result(index, document, method=method, score=score)
        for index, (score, document) in enumerate(selected, start=1)
    )
    return RetrievalResponse(
        state=RETRIEVAL_STATE_OK,
        query=query,
        results=results,
        top_k=top_k,
        corpus_size=corpus_size,
        candidate_count=candidate_count,
    )


def retrieve_evidence(
    documents: Sequence[EvidenceDocument],
    scope: RetrievalScope,
    *,
    query: str | None = None,
    ticket: Any = None,
    symbol: str | None = None,
    document_type: str | None = None,
    top_k: int | None = None,
) -> RetrievalResponse:
    resolved_top_k = _normalise_top_k(top_k)
    normalised_query = _normalise_query(query)
    try:
        validated = validate_evidence_corpus(documents, scope)
    except CorpusError:
        return _empty_response(
            RETRIEVAL_STATE_CORPUS_INVALID,
            normalised_query,
            resolved_top_k,
            corpus_size=0,
            candidate_count=0,
        )
    corpus_size = len(validated)
    requested_type = _explicit_document_type(document_type)
    requested_symbol = _explicit_symbol(symbol)
    requested_ticket = _explicit_ticket(ticket)
    candidates = list(validated)
    if requested_type is not None:
        candidates = [
            item for item in candidates if item.document_type == requested_type
        ]
    if requested_symbol is not None:
        candidates = [
            item for item in candidates if _symbol_matches(item, requested_symbol)
        ]
    if requested_ticket is not None:
        matched = []
        for item in candidates:
            if item.document_type != DOCUMENT_TYPE_TRADE:
                continue
            if canonical_ticket(item.source_record_key) != requested_ticket:
                continue
            matched.append(item)
        if not matched:
            return _empty_response(
                RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
                normalised_query,
                resolved_top_k,
                corpus_size=corpus_size,
                candidate_count=0,
            )
        ranked = sorted(
            ((None, item) for item in matched),
            key=lambda pair: _rank_sort_key(pair[0], pair[1]),
        )
        return _finalise(
            ranked,
            method=RETRIEVAL_METHOD_EXACT_TICKET,
            query=normalised_query,
            top_k=resolved_top_k,
            scope=scope,
            corpus_size=corpus_size,
            candidate_count=len(matched),
        )
    if not normalised_query:
        return _empty_response(
            RETRIEVAL_STATE_EMPTY_QUERY,
            normalised_query,
            resolved_top_k,
            corpus_size=corpus_size,
            candidate_count=len(candidates),
        )
    if not candidates:
        return _empty_response(
            RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            normalised_query,
            resolved_top_k,
            corpus_size=corpus_size,
            candidate_count=0,
        )
    candidate_count = len(candidates)
    scored = _tfidf_scores(tokenize(normalised_query), candidates)
    positive = [
        (score, document) for score, document in scored if score is not None and score > 0
    ]
    if not positive:
        return _empty_response(
            RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE,
            normalised_query,
            resolved_top_k,
            corpus_size=corpus_size,
            candidate_count=candidate_count,
        )
    ranked = sorted(positive, key=lambda pair: _rank_sort_key(pair[0], pair[1]))
    return _finalise(
        ranked,
        method=RETRIEVAL_METHOD_TFIDF_COSINE,
        query=normalised_query,
        top_k=resolved_top_k,
        scope=scope,
        corpus_size=corpus_size,
        candidate_count=candidate_count,
    )


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_QUERY_LENGTH",
    "MAX_TOP_K",
    "RETRIEVAL_METHOD_EXACT_TICKET",
    "RETRIEVAL_METHOD_TFIDF_COSINE",
    "RETRIEVAL_STATE_CORPUS_INVALID",
    "RETRIEVAL_STATE_EMPTY_QUERY",
    "RETRIEVAL_STATE_NO_RELEVANT_EVIDENCE",
    "RETRIEVAL_STATE_OK",
    "RetrievalError",
    "RetrievalResponse",
    "RetrievalResult",
    "retrieve_evidence",
    "tokenize",
]
