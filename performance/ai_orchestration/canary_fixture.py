from __future__ import annotations

from ..rag.retrieval import (
    RETRIEVAL_METHOD_TFIDF_COSINE,
    RETRIEVAL_STATE_OK,
    RetrievalResponse,
    RetrievalResult,
)
from ..rag.schema import DOCUMENT_TYPE_KPI, kpi_document_id

CANARY_CASE_ID = "sprint-8-controlled-live-canary-001"
CANARY_QUESTION = "Restate the supplied historical evidence."
CANARY_FINGERPRINT = "c" * 64


def load_canary_retrieval() -> RetrievalResponse:
    result = RetrievalResult(
        rank=1,
        document_id=kpi_document_id("Total Trades"),
        document_type=DOCUMENT_TYPE_KPI,
        source_record_key="Total Trades",
        content="Total Trades: 6",
        journal_fingerprint=CANARY_FINGERPRINT,
        evidence_status="AUTHORITATIVE",
        retrieval_method=RETRIEVAL_METHOD_TFIDF_COSINE,
        score=0.4,
        provenance={},
    )
    return RetrievalResponse(
        state=RETRIEVAL_STATE_OK,
        query=CANARY_QUESTION,
        results=(result,),
        top_k=5,
        corpus_size=1,
        candidate_count=1,
    )
