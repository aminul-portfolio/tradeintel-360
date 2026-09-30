from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..excursion import canonical_ticket, compute_journal_fingerprint
from ..market_data import EXCURSION_CONTRACT_VERSION
from ..time_basis import (
    TIME_BASIS_FIXED_OFFSET,
    TIME_BASIS_IANA,
    TIME_BASIS_RESOLUTION,
    TimeBasisValidationError,
    create_time_basis,
)
from ..trade_review import (
    BAR_EVIDENCE_LABELS,
    BAR_EVIDENCE_UNBOUND_LABEL,
    ENTRY_COLUMNS,
    EXIT_COLUMNS,
    SIDE_COLUMNS,
    attach_excursion_evidence,
    enrich_trade_review,
    format_bar_evidence_status,
    format_calculated_move_note,
    format_display_value,
    format_evidence_value,
    format_movement_value,
)
from .schema import (
    CORPUS_SCHEMA_VERSION,
    CROSS_SYMBOL_WARNING,
    DATASET_DOCUMENT_ID,
    DATASET_SOURCE_RECORD_KEY,
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    EVIDENCE_STATUS_COMPUTED,
    EVIDENCE_STATUS_NO_M1,
    RENDER_TEMPLATE_VERSION,
    SUPPORTED_DOCUMENT_TYPES,
    CorpusError,
    EvidenceDocument,
    assert_document_identity,
    content_sha256,
    kpi_document_id,
    make_evidence_document,
    provenance_as_plain,
    trade_document_id,
)
from .scope import RetrievalScope

OPEN_TIME_COLUMNS = ("Open Time", "Open", "Date")
CLOSE_TIME_COLUMNS = ("Close Time", "Close")
COMMISSION_COLUMNS = ("Commission", "Commissions")
SWAP_COLUMNS = ("Swap",)
PROFIT_COLUMNS = ("Profit",)
SYMBOL_COLUMNS = ("Symbol",)
TICKET_COLUMNS = ("Ticket",)

REQUIRED_BOUND_KEYS = (
    "contract_version",
    "journal_fingerprint",
    "market_file_sha256",
    "time_basis",
    "market_provenance",
    "evidence",
)
COMPUTED_EVIDENCE_FIELDS = ("interval_high", "interval_low", "mfe", "mae")
BOUND_EVIDENCE_STATUSES = frozenset(BAR_EVIDENCE_LABELS)
OWNER_IDENTITY_KEYS = frozenset({"owner_id", "user_id"})
ENVELOPE_TEXT_FIELDS = (
    "document_id",
    "document_type",
    "journal_fingerprint",
    "source_record_key",
    "content",
    "evidence_status",
)


def _first_present(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def source_basename(source_filename: str | None) -> str | None:
    if not source_filename:
        return None
    normalised = str(source_filename).replace("\\", "/")
    basename = normalised.rsplit("/", 1)[-1].strip()
    return basename or None


def _cell(row: Any, columns, candidates) -> Any:
    name = _first_present(columns, candidates)
    if name is None:
        return None
    if hasattr(row, "get"):
        return row.get(name)
    return row[name]


def _text(value: Any) -> str:
    display = format_display_value(value)
    if display is None or display == "":
        return ""
    return str(display)


def _structured_symbol(row: Any, columns) -> str | None:
    raw = _cell(row, columns, SYMBOL_COLUMNS)
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(raw, bool):
        return None
    text = str(raw).strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise CorpusError("INVALID_PROVENANCE", "Provenance is not JSON-safe")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise CorpusError("INVALID_PROVENANCE", "Provenance is not JSON-safe")


def _provenance(**fields: Any) -> dict[str, Any]:
    payload = {}
    for key, value in fields.items():
        if value is None:
            payload[key] = None
            continue
        payload[key] = _json_safe(value)
    return payload


def _require_bound_text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    return value


def _require_bound_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    return value


def _validate_bound_time_basis(time_basis: Mapping[str, Any]) -> None:
    if time_basis.get("resolution") != TIME_BASIS_RESOLUTION:
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    kind = time_basis.get("kind")
    if not isinstance(kind, str) or not kind:
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    try:
        create_time_basis(
            kind,
            offset_minutes=time_basis.get("offset_minutes"),
            zone=time_basis.get("zone"),
        )
    except TimeBasisValidationError as exc:
        raise CorpusError("INVALID_BOUND_EVIDENCE") from exc


def _declared_time_basis_line(time_basis: Mapping[str, Any]) -> str:
    kind = time_basis.get("kind")
    if kind == TIME_BASIS_FIXED_OFFSET:
        offset = time_basis.get("offset_minutes")
        sign = "+" if offset >= 0 else ""
        return f"Declared time basis: FIXED_OFFSET ({sign}{offset} minutes)"
    if kind == TIME_BASIS_IANA:
        return f"Declared time basis: IANA ({time_basis.get('zone')})"
    return f"Declared time basis: {kind}"


def _safe_market_provenance(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    if "source_basename" in payload:
        payload["source_basename"] = source_basename(payload.get("source_basename"))
    return payload


def _resolve_binding(
    excursion_state: Mapping[str, Any] | None,
    fingerprint: str,
) -> tuple[bool, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if excursion_state is None:
        return False, None, None
    if not isinstance(excursion_state, Mapping):
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    for key in REQUIRED_BOUND_KEYS:
        if key not in excursion_state:
            raise CorpusError("INVALID_BOUND_EVIDENCE")
    if excursion_state.get("contract_version") != EXCURSION_CONTRACT_VERSION:
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    bound_fingerprint = _require_bound_text(excursion_state.get("journal_fingerprint"))
    if bound_fingerprint != fingerprint:
        raise CorpusError("FINGERPRINT_MISMATCH")
    _require_bound_text(excursion_state.get("market_file_sha256"))
    time_basis = _require_bound_mapping(excursion_state.get("time_basis"))
    _validate_bound_time_basis(time_basis)
    _require_bound_mapping(excursion_state.get("market_provenance"))
    evidence = _require_bound_mapping(excursion_state.get("evidence"))
    return True, excursion_state, evidence


def _canonical_journal_tickets(journal: pd.DataFrame) -> set[str]:
    tickets: set[str] = set()
    columns = journal.columns
    for _, row in journal.iterrows():
        ticket = canonical_ticket(_cell(row, columns, TICKET_COLUMNS))
        if not ticket:
            raise CorpusError(
                "INVALID_JOURNAL_FINGERPRINT",
                "Trade row has no canonical ticket",
            )
        tickets.add(ticket)
    return tickets


def _canonical_evidence_tickets(evidence: Mapping[str, Any]) -> set[str]:
    tickets: set[str] = set()
    for key, item in evidence.items():
        ticket = canonical_ticket(key)
        if not ticket or not isinstance(item, Mapping):
            raise CorpusError("INVALID_BOUND_EVIDENCE")
        if ticket in tickets:
            raise CorpusError("INVALID_BOUND_EVIDENCE")
        if "ticket" in item:
            normalised = canonical_ticket(item.get("ticket"))
            if not normalised or normalised != ticket:
                raise CorpusError("INVALID_BOUND_EVIDENCE")
        tickets.add(ticket)
    return tickets


def _assert_bound_ticket_parity(
    journal: pd.DataFrame,
    evidence: Mapping[str, Any],
) -> None:
    if _canonical_journal_tickets(journal) != _canonical_evidence_tickets(evidence):
        raise CorpusError("INVALID_BOUND_EVIDENCE")


def _bound_item_status(item: Mapping[str, Any]) -> str:
    status = item.get("status")
    if not isinstance(status, str) or not status:
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    if status == EVIDENCE_STATUS_NO_M1 or status not in BOUND_EVIDENCE_STATUSES:
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    return status


def _assert_computed_evidence(evidence: Mapping[str, Any]) -> None:
    for item in evidence.values():
        if not isinstance(item, Mapping):
            raise CorpusError("INVALID_BOUND_EVIDENCE")
        status = _bound_item_status(item)
        if status != EVIDENCE_STATUS_COMPUTED:
            continue
        for field in COMPUTED_EVIDENCE_FIELDS:
            if format_evidence_value(item.get(field)) == "":
                raise CorpusError("INVALID_BOUND_EVIDENCE")


def _status_label(status: str) -> str:
    if status == EVIDENCE_STATUS_NO_M1:
        return BAR_EVIDENCE_UNBOUND_LABEL
    return format_bar_evidence_status(status) or status


def _unavailable_line(label: str, status: str) -> str:
    return f"{label}: not available ({status})"


def _trade_status(
    *,
    bound: bool,
    evidence_item: Mapping[str, Any] | None,
) -> str:
    if not bound:
        return EVIDENCE_STATUS_NO_M1
    if not isinstance(evidence_item, Mapping):
        raise CorpusError("INVALID_BOUND_EVIDENCE")
    return _bound_item_status(evidence_item)


def _append_field(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _trade_content(
    row: Any,
    columns,
    *,
    ticket: str,
    status: str,
) -> str:
    lines = [f"Ticket: {ticket}"]
    _append_field(lines, "Symbol", _text(_cell(row, columns, SYMBOL_COLUMNS)))
    _append_field(lines, "Type", _text(_cell(row, columns, SIDE_COLUMNS)))
    _append_field(lines, "Open Time", _text(_cell(row, columns, OPEN_TIME_COLUMNS)))
    _append_field(lines, "Close Time", _text(_cell(row, columns, CLOSE_TIME_COLUMNS)))
    _append_field(lines, "Entry", _text(_cell(row, columns, ENTRY_COLUMNS)))
    _append_field(lines, "Exit", _text(_cell(row, columns, EXIT_COLUMNS)))
    _append_field(lines, "Profit", _text(_cell(row, columns, PROFIT_COLUMNS)))
    _append_field(lines, "Commission", _text(_cell(row, columns, COMMISSION_COLUMNS)))
    _append_field(lines, "Swap", _text(_cell(row, columns, SWAP_COLUMNS)))
    movement = format_movement_value(
        row.get("_movement_value"),
        row.get("_movement_label") or "",
    )
    if movement:
        lines.append(f"Realised movement: {movement}")
        if bool(row.get("_movement_disagreement")):
            note = format_calculated_move_note(row.get("_movement_calculated"))
            if note:
                lines.append(note)
    lines.append(f"Bar Evidence: {_status_label(status)}")
    if status == EVIDENCE_STATUS_COMPUTED:
        lines.append(
            "Approx. Window High: "
            + format_evidence_value(row.get("_excursion_interval_high"))
        )
        lines.append(
            "Approx. Window Low: "
            + format_evidence_value(row.get("_excursion_interval_low"))
        )
        lines.append(
            "Approx. MFE: "
            + format_evidence_value(row.get("_excursion_mfe"))
            + " price pts"
        )
        lines.append(
            "Approx. MAE: "
            + format_evidence_value(row.get("_excursion_mae"))
            + " price pts"
        )
    else:
        lines.append(_unavailable_line("Approx. Window High", status))
        lines.append(_unavailable_line("Approx. Window Low", status))
        lines.append(_unavailable_line("Approx. MFE", status))
        lines.append(_unavailable_line("Approx. MAE", status))
    return "\n".join(lines)


def _prepare_frame(
    journal: pd.DataFrame,
    evidence_map: Mapping[str, Any] | None,
) -> pd.DataFrame:
    frame = journal.copy()
    if evidence_map is not None:
        frame = attach_excursion_evidence(frame, dict(evidence_map))
    return enrich_trade_review(frame)


def _trade_documents(
    scope: RetrievalScope,
    frame: pd.DataFrame,
    *,
    bound: bool,
    evidence_map: Mapping[str, Any] | None,
    source_basename: str | None,
    bound_payload: Mapping[str, Any] | None,
) -> list[EvidenceDocument]:
    columns = frame.columns
    documents: list[tuple[str, EvidenceDocument]] = []
    for _, row in frame.iterrows():
        ticket = canonical_ticket(_cell(row, columns, TICKET_COLUMNS))
        if not ticket:
            raise CorpusError(
                "INVALID_JOURNAL_FINGERPRINT",
                "Trade row has no canonical ticket",
            )
        item = None
        if evidence_map is not None:
            candidate = evidence_map.get(ticket)
            item = candidate if isinstance(candidate, Mapping) else None
        status = _trade_status(bound=bound, evidence_item=item)
        content = _trade_content(row, columns, ticket=ticket, status=status)
        provenance = _provenance(
            journal_fingerprint=scope.journal_fingerprint,
            document_type=DOCUMENT_TYPE_TRADE,
            source_record_key=ticket,
            source_basename=source_basename,
            market_data_sha256=(
                bound_payload.get("market_file_sha256") if bound_payload else None
            ),
            excursion_contract_version=(
                bound_payload.get("contract_version") if bound_payload else None
            ),
            time_basis=bound_payload.get("time_basis") if bound_payload else None,
            bar_evidence_status=status,
            symbol=_structured_symbol(row, columns),
        )
        documents.append(
            (
                ticket,
                make_evidence_document(
                    document_id=trade_document_id(ticket),
                    document_type=DOCUMENT_TYPE_TRADE,
                    journal_fingerprint=scope.journal_fingerprint,
                    source_record_key=ticket,
                    content=content,
                    evidence_status=status,
                    provenance=provenance,
                ),
            )
        )
    documents.sort(key=lambda item: item[0])
    return [document for _, document in documents]


def _kpi_documents(
    scope: RetrievalScope,
    kpi_mapping: Mapping[str, Any] | None,
    source_basename: str | None,
) -> list[EvidenceDocument]:
    mapping = dict(kpi_mapping) if isinstance(kpi_mapping, Mapping) else {}
    documents = []
    for key in sorted(mapping, key=str):
        record_key = str(key)
        documents.append(
            make_evidence_document(
                document_id=kpi_document_id(record_key),
                document_type=DOCUMENT_TYPE_KPI,
                journal_fingerprint=scope.journal_fingerprint,
                source_record_key=record_key,
                content=f"{record_key}: {mapping[key]}",
                evidence_status="AUTHORITATIVE",
                provenance=_provenance(
                    journal_fingerprint=scope.journal_fingerprint,
                    document_type=DOCUMENT_TYPE_KPI,
                    source_record_key=record_key,
                    source_basename=source_basename,
                    market_data_sha256=None,
                    excursion_contract_version=None,
                    time_basis=None,
                    bar_evidence_status=None,
                ),
            )
        )
    return documents


def _dataset_document(
    scope: RetrievalScope,
    journal: pd.DataFrame,
    *,
    bound: bool,
    source_basename: str | None,
    bound_payload: Mapping[str, Any] | None,
) -> EvidenceDocument:
    availability = BAR_EVIDENCE_UNBOUND_LABEL if not bound else "Bound"
    status = EVIDENCE_STATUS_NO_M1 if not bound else "BOUND"
    lines = [f"Journal row count: {len(journal)}"]
    if source_basename:
        lines.append(f"Source filename: {source_basename}")
    lines.append(f"Journal evidence availability: {availability}")
    time_basis = bound_payload.get("time_basis") if bound_payload else None
    market_sha = bound_payload.get("market_file_sha256") if bound_payload else None
    contract_version = bound_payload.get("contract_version") if bound_payload else None
    market_provenance = None
    if bound_payload is not None:
        raw_market = bound_payload.get("market_provenance")
        if isinstance(raw_market, Mapping):
            market_provenance = _safe_market_provenance(raw_market)
            market_basename = market_provenance.get("source_basename")
            if market_basename:
                lines.append(f"Market-data source: {market_basename}")
    if isinstance(time_basis, Mapping):
        lines.append(_declared_time_basis_line(time_basis))
    lines.append(CROSS_SYMBOL_WARNING)
    return make_evidence_document(
        document_id=DATASET_DOCUMENT_ID,
        document_type=DOCUMENT_TYPE_DATASET,
        journal_fingerprint=scope.journal_fingerprint,
        source_record_key=DATASET_SOURCE_RECORD_KEY,
        content="\n".join(lines),
        evidence_status=status,
        provenance=_provenance(
            journal_fingerprint=scope.journal_fingerprint,
            document_type=DOCUMENT_TYPE_DATASET,
            source_record_key=DATASET_SOURCE_RECORD_KEY,
            source_basename=source_basename,
            market_data_sha256=market_sha,
            excursion_contract_version=contract_version,
            time_basis=time_basis,
            bar_evidence_status=status,
            market_provenance=market_provenance,
        ),
    )


def _require_envelope_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError("MISSING_REQUIRED_FIELD")
    return value


def _reject_owner_identity_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in OWNER_IDENTITY_KEYS:
                raise CorpusError("INVALID_PROVENANCE")
            _reject_owner_identity_keys(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_owner_identity_keys(item)


def validate_evidence_corpus(
    documents: Sequence[EvidenceDocument],
    scope: RetrievalScope,
) -> tuple[EvidenceDocument, ...]:
    if not documents:
        raise CorpusError("MISSING_REQUIRED_FIELD", "Corpus has no documents")
    fingerprints: set[str] = set()
    document_ids: set[str] = set()
    dataset_count = 0
    for document in documents:
        if not isinstance(document, EvidenceDocument):
            raise CorpusError("MISSING_REQUIRED_FIELD", "Corpus item is not a document")
        for field in ENVELOPE_TEXT_FIELDS:
            _require_envelope_text(getattr(document, field))
        if document.corpus_schema_version != CORPUS_SCHEMA_VERSION:
            raise CorpusError("UNSUPPORTED_SCHEMA_VERSION")
        if document.render_template_version != RENDER_TEMPLATE_VERSION:
            raise CorpusError("UNSUPPORTED_RENDER_VERSION")
        if document.document_type not in SUPPORTED_DOCUMENT_TYPES:
            raise CorpusError("UNSUPPORTED_DOCUMENT_TYPE")
        if document.content_sha256 != content_sha256(document.content):
            raise CorpusError("INVALID_CONTENT_HASH")
        if not isinstance(document.provenance, Mapping):
            raise CorpusError("INVALID_PROVENANCE")
        try:
            plain = provenance_as_plain(document.provenance)
            json.dumps(
                plain,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except CorpusError:
            raise
        except (TypeError, ValueError) as exc:
            raise CorpusError("INVALID_PROVENANCE") from exc
        _reject_owner_identity_keys(document.provenance)
        fingerprints.add(document.journal_fingerprint)
        if document.document_id in document_ids:
            raise CorpusError("DUPLICATE_DOCUMENT_ID")
        document_ids.add(document.document_id)
        if document.document_type == DOCUMENT_TYPE_DATASET:
            dataset_count += 1
    if len(fingerprints) != 1:
        raise CorpusError("MIXED_JOURNAL_CORPUS")
    only_fingerprint = next(iter(fingerprints))
    if only_fingerprint != scope.journal_fingerprint:
        raise CorpusError("FINGERPRINT_MISMATCH")
    if dataset_count != 1:
        raise CorpusError("INVALID_CONTEXT_COUNT")
    for document in documents:
        assert_document_identity(
            document.document_type,
            document.document_id,
            document.source_record_key,
        )
    return tuple(documents)


def build_evidence_corpus(
    scope: RetrievalScope,
    journal: pd.DataFrame,
    *,
    kpi_mapping: Mapping[str, Any] | None = None,
    source_filename: str | None = None,
    excursion_state: Mapping[str, Any] | None = None,
) -> tuple[EvidenceDocument, ...]:
    if journal is None or journal.empty:
        raise CorpusError("INVALID_JOURNAL_FINGERPRINT", "Journal is empty")
    current_fingerprint = compute_journal_fingerprint(journal)
    if not current_fingerprint or current_fingerprint != scope.journal_fingerprint:
        raise CorpusError(
            "FINGERPRINT_MISMATCH",
            "Scope fingerprint does not match the journal",
        )
    basename = source_basename(source_filename)
    bound, bound_payload, evidence_map = _resolve_binding(
        excursion_state,
        scope.journal_fingerprint,
    )
    if bound:
        _assert_bound_ticket_parity(journal, evidence_map)
        _assert_computed_evidence(evidence_map)
    frame = _prepare_frame(journal, evidence_map if bound else None)
    trades = _trade_documents(
        scope,
        frame,
        bound=bound,
        evidence_map=evidence_map if bound else None,
        source_basename=basename,
        bound_payload=bound_payload,
    )
    kpis = _kpi_documents(scope, kpi_mapping, basename)
    dataset = _dataset_document(
        scope,
        journal,
        bound=bound,
        source_basename=basename,
        bound_payload=bound_payload,
    )
    return validate_evidence_corpus((*trades, *kpis, dataset), scope)
