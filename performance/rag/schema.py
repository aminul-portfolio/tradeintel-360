from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

CORPUS_SCHEMA_VERSION = "tradeintel.evidence.v1"
RENDER_TEMPLATE_VERSION = "tradeintel.evidence.render.v1"

DOCUMENT_TYPE_TRADE = "TRADE_EVIDENCE"
DOCUMENT_TYPE_KPI = "KPI_EVIDENCE"
DOCUMENT_TYPE_DATASET = "DATASET_CONTEXT"

SUPPORTED_DOCUMENT_TYPES = frozenset(
    {
        DOCUMENT_TYPE_TRADE,
        DOCUMENT_TYPE_KPI,
        DOCUMENT_TYPE_DATASET,
    }
)

EVIDENCE_STATUS_NO_M1 = "NO_M1_EVIDENCE"
EVIDENCE_STATUS_NO_MARKET_DATA = "NO_MARKET_DATA"
EVIDENCE_STATUS_COMPUTED = "COMPUTED"

CROSS_SYMBOL_WARNING = (
    "MFE/MAE use each trade's instrument price points. "
    "Values across different symbols are not directly comparable."
)

DATASET_SOURCE_RECORD_KEY = "journal"
DATASET_DOCUMENT_ID = f"{DOCUMENT_TYPE_DATASET}:{DATASET_SOURCE_RECORD_KEY}"


class CorpusError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    corpus_schema_version: str
    render_template_version: str
    document_id: str
    document_type: str
    journal_fingerprint: str
    source_record_key: str
    content: str
    content_sha256: str
    evidence_status: str
    provenance: Mapping[str, Any]


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _require_nonempty_str(value: Any, reason: str = "MISSING_REQUIRED_FIELD") -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(reason)
    return value


def _freeze_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CorpusError("INVALID_PROVENANCE")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CorpusError("INVALID_PROVENANCE")
            frozen[key] = _freeze_json_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise CorpusError("INVALID_PROVENANCE")


def freeze_provenance(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is None:
        return MappingProxyType({})
    if not isinstance(payload, Mapping):
        raise CorpusError("INVALID_PROVENANCE")
    frozen = _freeze_json_value(payload)
    if not isinstance(frozen, Mapping):
        raise CorpusError("INVALID_PROVENANCE")
    return frozen


def provenance_as_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CorpusError("INVALID_PROVENANCE")
        return value
    if isinstance(value, Mapping):
        plain: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CorpusError("INVALID_PROVENANCE")
            plain[key] = provenance_as_plain(item)
        return plain
    if isinstance(value, tuple):
        return [provenance_as_plain(item) for item in value]
    if isinstance(value, list):
        return [provenance_as_plain(item) for item in value]
    raise CorpusError("INVALID_PROVENANCE")


def trade_document_id(ticket: str) -> str:
    return f"{DOCUMENT_TYPE_TRADE}:ticket:{ticket}"


def kpi_document_id(kpi_key: str) -> str:
    return f"{DOCUMENT_TYPE_KPI}:kpi:{kpi_key}"


def assert_document_identity(
    document_type: str,
    document_id: str,
    source_record_key: str,
) -> None:
    if document_type == DOCUMENT_TYPE_TRADE:
        if document_id != trade_document_id(source_record_key):
            raise CorpusError("INVALID_DOCUMENT_ID")
        return
    if document_type == DOCUMENT_TYPE_KPI:
        if document_id != kpi_document_id(source_record_key):
            raise CorpusError("INVALID_DOCUMENT_ID")
        return
    if document_type == DOCUMENT_TYPE_DATASET:
        if (
            document_id != DATASET_DOCUMENT_ID
            or source_record_key != DATASET_SOURCE_RECORD_KEY
        ):
            raise CorpusError("INVALID_DOCUMENT_ID")


def make_evidence_document(
    *,
    document_id: str,
    document_type: str,
    journal_fingerprint: str,
    source_record_key: str,
    content: str,
    evidence_status: str,
    provenance: Mapping[str, Any] | None = None,
    corpus_schema_version: str = CORPUS_SCHEMA_VERSION,
    render_template_version: str = RENDER_TEMPLATE_VERSION,
) -> EvidenceDocument:
    _require_nonempty_str(document_id)
    _require_nonempty_str(document_type)
    _require_nonempty_str(journal_fingerprint)
    _require_nonempty_str(source_record_key)
    _require_nonempty_str(content)
    _require_nonempty_str(evidence_status)
    _require_nonempty_str(corpus_schema_version)
    _require_nonempty_str(render_template_version)
    assert_document_identity(document_type, document_id, source_record_key)
    return EvidenceDocument(
        corpus_schema_version=corpus_schema_version,
        render_template_version=render_template_version,
        document_id=document_id,
        document_type=document_type,
        journal_fingerprint=journal_fingerprint,
        source_record_key=source_record_key,
        content=content,
        content_sha256=content_sha256(content),
        evidence_status=evidence_status,
        provenance=freeze_provenance(provenance),
    )
