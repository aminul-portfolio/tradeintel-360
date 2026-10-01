from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..rag.retrieval import (
    RETRIEVAL_STATE_OK,
    RetrievalResponse,
    RetrievalResult,
)
from ..rag.schema import (
    CORPUS_SCHEMA_VERSION,
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    EVIDENCE_STATUS_COMPUTED,
    EVIDENCE_STATUS_NO_MARKET_DATA,
    RENDER_TEMPLATE_VERSION,
    SUPPORTED_DOCUMENT_TYPES,
)
from .codes import FieldType, GroundingError, LimitationCode, RejectionCode
from .schema import (
    ACCEPTED_EVIDENCE_STATUSES,
    ACCEPTED_SIDES,
    DEFAULT_CONSTRAINTS,
    GROUNDING_REQUEST_SCHEMA_VERSION,
    GROUNDING_RESPONSE_SCHEMA_VERSION,
    MAX_EVIDENCE_ITEMS,
    PROMPT_TEMPLATE_VERSION,
    SYMBOL_RE,
    TASK_TYPE,
    GroundedAIRequest,
    GroundedEvidenceItem,
    ModelFacingField,
    ServerGroundingContext,
    canonical_limitation_codes,
    canonical_request_json,
    make_model_facing_field,
    normalise_question,
    request_sha256,
    validate_request_shape,
)

SUPPORTED_EVIDENCE_STATUSES = ACCEPTED_EVIDENCE_STATUSES
UNAVAILABLE_PREFIX = "not available"
EXCLUDED_CONTENT_LABELS = frozenset(
    {
        "tag",
        "tags",
        "notes",
        "comment",
        "comments",
        "strategy",
        "reason",
        "source filename",
        "market-data source",
    }
)
KPI_FIELD_MAP = {
    "Total Trades": ("total_trades", "Total Trades", FieldType.INTEGER, ""),
    "Winning Trades": ("winning_trades", "Winning Trades", FieldType.INTEGER, ""),
    "Losing Trades": ("losing_trades", "Losing Trades", FieldType.INTEGER, ""),
    "Break-even Trades": ("breakeven_trades", "Break-even Trades", FieldType.INTEGER, ""),
    "Win Rate (%)": ("win_rate_pct", "Win Rate (%)", FieldType.DECIMAL, "%"),
    "Total Profit": ("total_profit", "Total Profit", FieldType.DECIMAL, ""),
    "Average Profit": ("average_profit", "Average Profit", FieldType.DECIMAL, ""),
    "Gross Profit": ("gross_profit", "Gross Profit", FieldType.DECIMAL, ""),
    "Gross Loss": ("gross_loss", "Gross Loss", FieldType.DECIMAL, ""),
    "Average Win": ("average_win", "Average Win", FieldType.DECIMAL, ""),
    "Average Loss": ("average_loss", "Average Loss", FieldType.DECIMAL, ""),
    "Profit Factor": ("profit_factor", "Profit Factor", FieldType.TEXT, ""),
    "Expectancy": ("expectancy", "Expectancy", FieldType.DECIMAL, ""),
    "Best Trade": ("best_trade", "Best Trade", FieldType.DECIMAL, ""),
    "Worst Trade": ("worst_trade", "Worst Trade", FieldType.DECIMAL, ""),
    "Max Drawdown": ("max_drawdown", "Max Drawdown", FieldType.DECIMAL, ""),
    "Sharpe": ("sharpe", "Sharpe", FieldType.DECIMAL, ""),
    "Volatility": ("volatility", "Volatility", FieldType.DECIMAL, ""),
}
CLAIMABLE_TYPES = frozenset({FieldType.DECIMAL, FieldType.INTEGER})


def _content_pairs(content: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for line in content.splitlines():
        if ": " not in line:
            continue
        label, value = line.split(": ", 1)
        stripped_label = label.strip()
        if stripped_label.casefold() in EXCLUDED_CONTENT_LABELS:
            continue
        pairs.append((stripped_label, value.strip()))
    return tuple(pairs)


def _content_map(content: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label, value in _content_pairs(content):
        mapping[label] = value
    return mapping


def _validate_symbol(raw: str) -> str:
    if not SYMBOL_RE.fullmatch(raw):
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
    return raw


def _validate_side(raw: str) -> str:
    if raw.casefold() not in ACCEPTED_SIDES:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
    return raw.casefold()


def _field(
    field_key: str,
    label: str,
    canonical_value: str,
    value_type: FieldType,
    unit: str = "",
) -> ModelFacingField:
    return make_model_facing_field(field_key, label, canonical_value, value_type, unit)


def _unavailable(value: str) -> bool:
    return value.casefold().startswith(UNAVAILABLE_PREFIX)


def _strip_price_pts(value: str) -> str:
    suffix = " price pts"
    if value.endswith(suffix):
        return value[: -len(suffix)].strip()
    return value


def _provenance_symbol_raw(result: RetrievalResult) -> str | None:
    provenance = result.provenance
    if not isinstance(provenance, Mapping):
        return None
    raw = provenance.get("symbol")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _resolve_trade_symbol(result: RetrievalResult, mapping: dict[str, str]) -> str | None:
    content_raw = mapping.get("Symbol")
    content_symbol = _validate_symbol(content_raw) if content_raw else None
    provenance_raw = _provenance_symbol_raw(result)
    provenance_symbol = _validate_symbol(provenance_raw) if provenance_raw else None
    if content_symbol is not None and provenance_symbol is not None:
        if content_symbol != provenance_symbol:
            # Both Sprint 5 content Symbol and provenance.symbol are present.
            # Disagreement is a request identity mismatch, not a malformed token.
            raise GroundingError(
                RejectionCode.REQUEST_MISMATCH,
                "Content Symbol and provenance symbol disagree",
            )
        return content_symbol
    return content_symbol if content_symbol is not None else provenance_symbol


def _project_trade(result: RetrievalResult) -> tuple[tuple[ModelFacingField, ...], tuple[str, ...]]:
    mapping = _content_map(result.content)
    fields: list[ModelFacingField] = []
    if "Ticket" in mapping:
        fields.append(_field("ticket", "Ticket", mapping["Ticket"], FieldType.IDENTIFIER))
    symbol = _resolve_trade_symbol(result, mapping)
    if symbol is not None:
        fields.append(_field("symbol", "Symbol", symbol, FieldType.IDENTIFIER))
    if "Type" in mapping:
        fields.append(_field("side", "Type", _validate_side(mapping["Type"]), FieldType.ENUM))
    if "Open Time" in mapping:
        fields.append(_field("open_time", "Open Time", mapping["Open Time"], FieldType.TIMESTAMP))
    if "Close Time" in mapping:
        fields.append(_field("close_time", "Close Time", mapping["Close Time"], FieldType.TIMESTAMP))
    if "Entry" in mapping:
        fields.append(_field("entry", "Entry", mapping["Entry"], FieldType.DECIMAL))
    if "Exit" in mapping:
        fields.append(_field("exit", "Exit", mapping["Exit"], FieldType.DECIMAL))
    if "Profit" in mapping:
        fields.append(_field("profit", "Profit", mapping["Profit"], FieldType.DECIMAL))
    if "Commission" in mapping:
        fields.append(_field("commission", "Commission", mapping["Commission"], FieldType.DECIMAL))
    if "Swap" in mapping:
        fields.append(_field("swap", "Swap", mapping["Swap"], FieldType.DECIMAL))
    fields.append(
        _field(
            "bar_evidence_status",
            "Bar Evidence Status",
            result.evidence_status,
            FieldType.ENUM,
        )
    )
    if result.evidence_status == EVIDENCE_STATUS_COMPUTED:
        for label, key, unit in (
            ("Approx. Window High", "approx_window_high", ""),
            ("Approx. Window Low", "approx_window_low", ""),
            ("Approx. MFE", "approx_mfe", "price pts"),
            ("Approx. MAE", "approx_mae", "price pts"),
        ):
            if label not in mapping:
                raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
            raw = mapping[label]
            if _unavailable(raw):
                raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
            canonical = _strip_price_pts(raw)
            fields.append(_field(key, label, canonical, FieldType.DECIMAL, unit))
    claimable = tuple(
        item.field_key for item in fields if item.value_type in CLAIMABLE_TYPES
    )
    return tuple(fields), claimable


def _project_kpi(result: RetrievalResult) -> tuple[tuple[ModelFacingField, ...], tuple[str, ...]]:
    mapping = _content_map(result.content)
    fields: list[ModelFacingField] = []
    for label, canonical in mapping.items():
        spec = KPI_FIELD_MAP.get(label)
        if spec is None:
            raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
        field_key, field_label, value_type, unit = spec
        fields.append(_field(field_key, field_label, canonical, value_type, unit))
    claimable = tuple(
        item.field_key for item in fields if item.value_type in CLAIMABLE_TYPES
    )
    return tuple(fields), claimable


def _project_dataset(result: RetrievalResult) -> tuple[tuple[ModelFacingField, ...], tuple[str, ...]]:
    mapping = _content_map(result.content)
    fields: list[ModelFacingField] = []
    if "Journal row count" in mapping:
        fields.append(
            _field(
                "journal_row_count",
                "Journal row count",
                mapping["Journal row count"],
                FieldType.INTEGER,
            )
        )
    if "Declared time basis" in mapping:
        fields.append(
            _field(
                "declared_time_basis",
                "Declared time basis",
                mapping["Declared time basis"],
                FieldType.TEXT,
            )
        )
    claimable = tuple(
        item.field_key for item in fields if item.value_type in CLAIMABLE_TYPES
    )
    return tuple(fields), claimable


def _project_result(result: RetrievalResult) -> GroundedEvidenceItem:
    if result.document_type not in SUPPORTED_DOCUMENT_TYPES:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
    if result.evidence_status not in SUPPORTED_EVIDENCE_STATUSES:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
    if result.document_type == DOCUMENT_TYPE_TRADE:
        fields, claimable = _project_trade(result)
    elif result.document_type == DOCUMENT_TYPE_KPI:
        fields, claimable = _project_kpi(result)
    elif result.document_type == DOCUMENT_TYPE_DATASET:
        fields, claimable = _project_dataset(result)
    else:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID)
    return GroundedEvidenceItem(
        alias="",
        document_type=result.document_type,
        evidence_status=result.evidence_status,
        fields=fields,
        claimable_field_keys=claimable,
    )


def _required_limitation_codes(
    items: Sequence[GroundedEvidenceItem],
    *,
    candidate_count: int,
    returned_count: int,
) -> tuple[LimitationCode, ...]:
    codes: list[LimitationCode] = []
    has_approx = False
    has_unavailable = False
    symbols: set[str] = set()
    for item in items:
        if item.evidence_status == EVIDENCE_STATUS_NO_MARKET_DATA:
            has_unavailable = True
        if item.document_type == DOCUMENT_TYPE_TRADE:
            keys = {field.field_key for field in item.fields}
            if "approx_mfe" in keys or "approx_mae" in keys:
                has_approx = True
                for field in item.fields:
                    if field.field_key == "symbol":
                        symbols.add(field.canonical_value)
    if has_approx:
        codes.append(LimitationCode.APPROXIMATE_M1_EVIDENCE)
    if has_unavailable:
        codes.append(LimitationCode.EVIDENCE_UNAVAILABLE)
    if len(symbols) > 1:
        codes.append(LimitationCode.CROSS_SYMBOL_NOT_COMPARABLE)
    if candidate_count > returned_count:
        codes.append(LimitationCode.PARTIAL_EVIDENCE)
    return tuple(codes)


def build_evidence_packet(
    retrieval: RetrievalResponse,
    question: str,
) -> tuple[GroundedAIRequest, ServerGroundingContext]:
    if retrieval.state != RETRIEVAL_STATE_OK:
        raise GroundingError(RejectionCode.REQUEST_STATE_NOT_OK)
    if not retrieval.results:
        raise GroundingError(RejectionCode.REQUEST_EMPTY_EVIDENCE)
    if len(retrieval.results) > MAX_EVIDENCE_ITEMS:
        raise GroundingError(RejectionCode.SCHEMA_LIMIT_EXCEEDED)
    expected_ranks = tuple(range(1, len(retrieval.results) + 1))
    actual_ranks = tuple(item.rank for item in retrieval.results)
    if actual_ranks != expected_ranks:
        raise GroundingError(RejectionCode.REQUEST_MISMATCH)
    normalised_question = normalise_question(question)
    projected: list[GroundedEvidenceItem] = []
    alias_map: list[tuple[str, str]] = []
    methods: set[str] = set()
    fingerprints: set[str] = set()
    for index, result in enumerate(retrieval.results, start=1):
        alias = f"E{index}"
        item = _project_result(result)
        projected.append(
            GroundedEvidenceItem(
                alias=alias,
                document_type=item.document_type,
                evidence_status=item.evidence_status,
                fields=item.fields,
                claimable_field_keys=item.claimable_field_keys,
            )
        )
        alias_map.append((alias, result.document_id))
        methods.add(result.retrieval_method)
        fingerprints.add(result.journal_fingerprint)
    if len(fingerprints) != 1:
        raise GroundingError(RejectionCode.REQUEST_MISMATCH)
    limitations = canonical_limitation_codes(
        _required_limitation_codes(
            projected,
            candidate_count=retrieval.candidate_count,
            returned_count=len(retrieval.results),
        )
    )
    request = GroundedAIRequest(
        schema_version=GROUNDING_REQUEST_SCHEMA_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        response_schema_version=GROUNDING_RESPONSE_SCHEMA_VERSION,
        task_type=TASK_TYPE,
        question=normalised_question,
        evidence=tuple(projected),
        required_limitation_codes=limitations,
        constraints=DEFAULT_CONSTRAINTS,
    )
    validate_request_shape(request)
    digest = request_sha256(request)
    if "request_sha256" in canonical_request_json(request):
        raise GroundingError(RejectionCode.LEAKAGE_DETECTED)
    context = ServerGroundingContext(
        request_sha256=digest,
        alias_to_document_id=tuple(alias_map),
        journal_fingerprint=next(iter(fingerprints)),
        corpus_schema_version=CORPUS_SCHEMA_VERSION,
        render_template_version=RENDER_TEMPLATE_VERSION,
        retrieval_method=next(iter(methods)) if len(methods) == 1 else "MIXED",
        returned_count=len(retrieval.results),
        candidate_count=retrieval.candidate_count,
        corpus_size=retrieval.corpus_size,
    )
    return request, context


__all__ = [
    "SUPPORTED_EVIDENCE_STATUSES",
    "build_evidence_packet",
]
