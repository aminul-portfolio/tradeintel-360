from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Any

from ..rag.schema import (
    DOCUMENT_TYPE_DATASET,
    DOCUMENT_TYPE_KPI,
    DOCUMENT_TYPE_TRADE,
    EVIDENCE_STATUS_COMPUTED,
    EVIDENCE_STATUS_NO_M1,
    EVIDENCE_STATUS_NO_MARKET_DATA,
)
from .codes import (
    FieldType,
    GroundingError,
    LimitationCode,
    RejectionCode,
)

GROUNDING_REQUEST_SCHEMA_VERSION = "grounded-ai-request-v1"
GROUNDING_RESPONSE_SCHEMA_VERSION = "grounded-ai-response-v1"
PROMPT_TEMPLATE_VERSION = "evidence-summary-v1"
TASK_TYPE = "EVIDENCE_SUMMARY"

MAX_QUESTION_LENGTH = 500
MAX_EVIDENCE_ITEMS = 10
MIN_EVIDENCE_ITEMS = 1
MAX_ANSWER_CHARS = 2000
MAX_CLAIMS = 160
MAX_CITATIONS = 10
MAX_LIMITATION_CODES = 4
MAX_TEXT_LENGTH = 80
SYMBOL_PATTERN = r"^[A-Za-z0-9._-]{1,32}$"
TICKET_PATTERN = r"^[A-Za-z0-9._-]{1,32}$"
INTEGER_PATTERN = r"^-?(?:0|[1-9][0-9]{0,15})$"
DECIMAL_PATTERN = r"^-?(?:0|[1-9][0-9]{0,15})(?:\.[0-9]{1,18})?$"
ISO_TIMESTAMP_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$"
FTMO_TIMESTAMP_PATTERN = (
    r"^[0-9]{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}$"
)
FIXED_OFFSET_BASIS_PATTERN = r"^FIXED_OFFSET \([+-][0-9]{1,4} minutes\)$"
IANA_BASIS_PATTERN = r"^IANA \([A-Za-z][A-Za-z0-9_/+-]{0,63}\)$"
ACCEPTED_SIDES = frozenset({"buy", "sell"})
# Sprint 5 corpus pairs: trade bar statuses + unbound NO_M1; KPI AUTHORITATIVE; dataset BOUND or unbound NO_M1.
TRADE_EVIDENCE_STATUSES = frozenset(
    {
        EVIDENCE_STATUS_COMPUTED,
        EVIDENCE_STATUS_NO_MARKET_DATA,
        EVIDENCE_STATUS_NO_M1,
        "INVALID_TRADE_DATA",
        "TIMEZONE_AMBIGUOUS",
        "TIME_BASIS_INCONSISTENT",
        "INCOMPLETE_COVERAGE",
        "INVARIANT_VIOLATION",
    }
)
KPI_EVIDENCE_STATUSES = frozenset({"AUTHORITATIVE"})
DATASET_EVIDENCE_STATUSES = frozenset({EVIDENCE_STATUS_NO_M1, "BOUND"})
DOCUMENT_ALLOWED_STATUSES = MappingProxyType(
    {
        DOCUMENT_TYPE_TRADE: TRADE_EVIDENCE_STATUSES,
        DOCUMENT_TYPE_KPI: KPI_EVIDENCE_STATUSES,
        DOCUMENT_TYPE_DATASET: DATASET_EVIDENCE_STATUSES,
    }
)
ACCEPTED_EVIDENCE_STATUSES = frozenset(
    status for allowed in DOCUMENT_ALLOWED_STATUSES.values() for status in allowed
)
REQUIRED_COMPUTED_EXCURSION_KEYS = (
    "approx_window_high",
    "approx_window_low",
    "approx_mfe",
    "approx_mae",
)
PROFIT_FACTOR_SPECIAL = frozenset({"N/A", "\u221e"})
STRUCTURAL_TOKENS = ("<<<", ">>>", "{", "}", "[", "]")
FTMO_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
SYMBOL_RE = re.compile(SYMBOL_PATTERN)
TICKET_RE = re.compile(TICKET_PATTERN)
INTEGER_RE = re.compile(INTEGER_PATTERN)
DECIMAL_RE = re.compile(DECIMAL_PATTERN)
ISO_TIMESTAMP_RE = re.compile(ISO_TIMESTAMP_PATTERN)
FTMO_TIMESTAMP_RE = re.compile(FTMO_TIMESTAMP_PATTERN)
FIXED_OFFSET_BASIS_RE = re.compile(FIXED_OFFSET_BASIS_PATTERN)
IANA_BASIS_RE = re.compile(IANA_BASIS_PATTERN)
CLAIMABLE_TYPES = frozenset({FieldType.DECIMAL, FieldType.INTEGER})
EXCURSION_FIELD_KEYS = frozenset(
    {
        "approx_window_high",
        "approx_window_low",
        "approx_mfe",
        "approx_mae",
    }
)


@dataclass(frozen=True, slots=True)
class AuthorisedFieldSpec:
    label: str
    value_type: FieldType
    unit: str
    document_types: frozenset[str]


def _spec(label: str, value_type: FieldType, unit: str, *document_types: str) -> AuthorisedFieldSpec:
    return AuthorisedFieldSpec(label, value_type, unit, frozenset(document_types))


AUTHORISED_FIELDS = {
    "ticket": _spec("Ticket", FieldType.IDENTIFIER, "", DOCUMENT_TYPE_TRADE),
    "symbol": _spec("Symbol", FieldType.IDENTIFIER, "", DOCUMENT_TYPE_TRADE),
    "side": _spec("Type", FieldType.ENUM, "", DOCUMENT_TYPE_TRADE),
    "open_time": _spec("Open Time", FieldType.TIMESTAMP, "", DOCUMENT_TYPE_TRADE),
    "close_time": _spec("Close Time", FieldType.TIMESTAMP, "", DOCUMENT_TYPE_TRADE),
    "entry": _spec("Entry", FieldType.DECIMAL, "", DOCUMENT_TYPE_TRADE),
    "exit": _spec("Exit", FieldType.DECIMAL, "", DOCUMENT_TYPE_TRADE),
    "profit": _spec("Profit", FieldType.DECIMAL, "", DOCUMENT_TYPE_TRADE),
    "commission": _spec("Commission", FieldType.DECIMAL, "", DOCUMENT_TYPE_TRADE),
    "swap": _spec("Swap", FieldType.DECIMAL, "", DOCUMENT_TYPE_TRADE),
    "bar_evidence_status": _spec(
        "Bar Evidence Status",
        FieldType.ENUM,
        "",
        DOCUMENT_TYPE_TRADE,
    ),
    "approx_window_high": _spec(
        "Approx. Window High",
        FieldType.DECIMAL,
        "",
        DOCUMENT_TYPE_TRADE,
    ),
    "approx_window_low": _spec(
        "Approx. Window Low",
        FieldType.DECIMAL,
        "",
        DOCUMENT_TYPE_TRADE,
    ),
    "approx_mfe": _spec("Approx. MFE", FieldType.DECIMAL, "price pts", DOCUMENT_TYPE_TRADE),
    "approx_mae": _spec("Approx. MAE", FieldType.DECIMAL, "price pts", DOCUMENT_TYPE_TRADE),
    "total_trades": _spec("Total Trades", FieldType.INTEGER, "", DOCUMENT_TYPE_KPI),
    "winning_trades": _spec("Winning Trades", FieldType.INTEGER, "", DOCUMENT_TYPE_KPI),
    "losing_trades": _spec("Losing Trades", FieldType.INTEGER, "", DOCUMENT_TYPE_KPI),
    "breakeven_trades": _spec("Break-even Trades", FieldType.INTEGER, "", DOCUMENT_TYPE_KPI),
    "win_rate_pct": _spec("Win Rate (%)", FieldType.DECIMAL, "%", DOCUMENT_TYPE_KPI),
    "total_profit": _spec("Total Profit", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "average_profit": _spec("Average Profit", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "gross_profit": _spec("Gross Profit", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "gross_loss": _spec("Gross Loss", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "average_win": _spec("Average Win", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "average_loss": _spec("Average Loss", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "profit_factor": _spec("Profit Factor", FieldType.TEXT, "", DOCUMENT_TYPE_KPI),
    "expectancy": _spec("Expectancy", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "best_trade": _spec("Best Trade", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "worst_trade": _spec("Worst Trade", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "max_drawdown": _spec("Max Drawdown", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "sharpe": _spec("Sharpe", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "volatility": _spec("Volatility", FieldType.DECIMAL, "", DOCUMENT_TYPE_KPI),
    "journal_row_count": _spec(
        "Journal row count",
        FieldType.INTEGER,
        "",
        DOCUMENT_TYPE_DATASET,
    ),
    "declared_time_basis": _spec(
        "Declared time basis",
        FieldType.TEXT,
        "",
        DOCUMENT_TYPE_DATASET,
    ),
}


@dataclass(frozen=True, slots=True)
class ModelFacingField:
    field_key: str
    label: str
    canonical_value: str
    value_type: FieldType
    unit: str


@dataclass(frozen=True, slots=True)
class GroundedEvidenceItem:
    alias: str
    document_type: str
    evidence_status: str
    fields: tuple[ModelFacingField, ...]
    claimable_field_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundingConstraints:
    cite_only_supplied_aliases: bool
    no_recommendations: bool
    no_predictions: bool
    no_counterfactuals: bool
    no_causal_explanations: bool
    max_answer_chars: int
    max_claims: int
    max_citations: int
    max_limitation_codes: int


DEFAULT_CONSTRAINTS = GroundingConstraints(
    cite_only_supplied_aliases=True,
    no_recommendations=True,
    no_predictions=True,
    no_counterfactuals=True,
    no_causal_explanations=True,
    max_answer_chars=MAX_ANSWER_CHARS,
    max_claims=MAX_CLAIMS,
    max_citations=MAX_CITATIONS,
    max_limitation_codes=MAX_LIMITATION_CODES,
)


@dataclass(frozen=True, slots=True)
class GroundedAIRequest:
    schema_version: str
    prompt_template_version: str
    response_schema_version: str
    task_type: str
    question: str
    evidence: tuple[GroundedEvidenceItem, ...]
    required_limitation_codes: tuple[LimitationCode, ...]
    constraints: GroundingConstraints


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    evidence: str
    field: str
    value: str


@dataclass(frozen=True, slots=True)
class GroundedAIResponse:
    schema_version: str
    answer: str
    claims: tuple[GroundedClaim, ...]
    citations: tuple[str, ...]
    limitation_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ServerGroundingContext:
    request_sha256: str
    alias_to_document_id: tuple[tuple[str, str], ...]
    journal_fingerprint: str
    corpus_schema_version: str
    render_template_version: str
    retrieval_method: str
    returned_count: int
    candidate_count: int
    corpus_size: int


def _strip_controls(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def _reject_field(message: str = "") -> None:
    raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, message)


def _require_plain_token(value: str) -> None:
    if not value:
        _reject_field("Empty canonical value")
    if value.strip() != value:
        _reject_field("Canonical value is not trimmed")
    if any(unicodedata.category(ch)[0] == "C" for ch in value):
        _reject_field("Canonical value contains control characters")
    if any(token in value for token in STRUCTURAL_TOKENS):
        _reject_field("Canonical value contains structural tokens")


def _validate_integer(value: str) -> None:
    if not INTEGER_RE.fullmatch(value):
        _reject_field("INTEGER value is not canonical integer syntax")


def _validate_decimal(value: str) -> None:
    if not DECIMAL_RE.fullmatch(value):
        _reject_field("DECIMAL value is not canonical decimal syntax")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        _reject_field("DECIMAL value is not a finite Decimal")
    if not parsed.is_finite():
        _reject_field("DECIMAL value is not finite")


def _validate_timestamp(value: str) -> None:
    if ISO_TIMESTAMP_RE.fullmatch(value):
        try:
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            _reject_field("TIMESTAMP is not a valid ISO calendar value")
        return
    if not FTMO_TIMESTAMP_RE.fullmatch(value):
        _reject_field("TIMESTAMP is not an authorised Sprint 5 representation")
    day_text, month_text, year_text, clock = value.split(" ")
    hour_text, minute_text, second_text = clock.split(":")
    try:
        datetime(
            int(year_text),
            FTMO_MONTHS[month_text],
            int(day_text),
            int(hour_text),
            int(minute_text),
            int(second_text),
        )
    except (KeyError, ValueError):
        _reject_field("TIMESTAMP is not a valid FTMO calendar value")


def _validate_identifier(field_key: str, value: str) -> None:
    if field_key == "ticket":
        if not TICKET_RE.fullmatch(value):
            _reject_field("ticket identifier is not a safe canonical token")
        return
    if field_key == "symbol":
        if not SYMBOL_RE.fullmatch(value):
            _reject_field("symbol identifier is not a safe canonical token")
        return
    _reject_field("Unknown IDENTIFIER field")


def _validate_enum(field_key: str, value: str) -> None:
    if field_key == "side":
        if value not in ACCEPTED_SIDES:
            _reject_field("side is not an accepted trade side")
        return
    if field_key == "bar_evidence_status":
        if value not in ACCEPTED_EVIDENCE_STATUSES:
            _reject_field("bar_evidence_status is not an accepted status")
        return
    _reject_field("Unknown ENUM field")


def _validate_text(field_key: str, value: str) -> None:
    if len(value) > MAX_TEXT_LENGTH:
        _reject_field("TEXT value exceeds the authorised length")
    if field_key == "profit_factor":
        if value in PROFIT_FACTOR_SPECIAL:
            return
        _validate_decimal(value)
        return
    if field_key == "declared_time_basis":
        if value == "UTC":
            return
        if FIXED_OFFSET_BASIS_RE.fullmatch(value):
            return
        if IANA_BASIS_RE.fullmatch(value):
            return
        _reject_field("declared_time_basis is not an authorised time-basis representation")
    _reject_field("Unknown TEXT field")


def make_model_facing_field(
    field_key: str,
    label: str,
    canonical_value: str,
    value_type: FieldType,
    unit: str = "",
) -> ModelFacingField:
    spec = AUTHORISED_FIELDS.get(field_key)
    if spec is None:
        _reject_field("Unauthorised field_key")
    if label != spec.label or value_type != spec.value_type or unit != spec.unit:
        _reject_field("Field label, type, or unit is not authorised")
    if not isinstance(canonical_value, str):
        _reject_field("Canonical value must be text")
    _require_plain_token(canonical_value)
    if value_type == FieldType.INTEGER:
        _validate_integer(canonical_value)
    elif value_type == FieldType.DECIMAL:
        _validate_decimal(canonical_value)
    elif value_type == FieldType.IDENTIFIER:
        _validate_identifier(field_key, canonical_value)
    elif value_type == FieldType.ENUM:
        _validate_enum(field_key, canonical_value)
    elif value_type == FieldType.TIMESTAMP:
        _validate_timestamp(canonical_value)
    elif value_type == FieldType.TEXT:
        _validate_text(field_key, canonical_value)
    else:
        _reject_field("Unknown field type")
    return ModelFacingField(
        field_key=field_key,
        label=label,
        canonical_value=canonical_value,
        value_type=value_type,
        unit=unit,
    )


def normalise_question(raw: Any) -> str:
    if not isinstance(raw, str):
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Question must be text")
    normalised = unicodedata.normalize("NFKC", raw)
    normalised = _strip_controls(normalised).strip()
    if not normalised:
        raise GroundingError(RejectionCode.SCHEMA_MISSING_FIELD, "Question is empty")
    if len(normalised) > MAX_QUESTION_LENGTH:
        raise GroundingError(RejectionCode.SCHEMA_LIMIT_EXCEEDED, "Question exceeds 500 characters")
    return normalised


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Float values are not canonical")
    raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Unsupported canonical JSON type")


def canonical_request_json(request: GroundedAIRequest) -> str:
    return json.dumps(
        _jsonable(request),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def request_sha256(request: GroundedAIRequest) -> str:
    payload = canonical_request_json(request)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_limitation_codes(codes: Sequence[LimitationCode]) -> tuple[LimitationCode, ...]:
    return tuple(sorted(codes, key=lambda item: item.value))


def _validate_question(question: Any) -> None:
    if not isinstance(question, str):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Question must be text")
    canonical = normalise_question(question)
    if canonical != question:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Question is not canonical")


def _validate_constraints(constraints: Any) -> None:
    if not isinstance(constraints, GroundingConstraints):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Constraints are not typed")
    if constraints != DEFAULT_CONSTRAINTS:
        raise GroundingError(RejectionCode.REQUEST_MISMATCH, "Constraints are not the locked Sprint 6 defaults")
    if (
        constraints.max_answer_chars <= 0
        or constraints.max_claims <= 0
        or constraints.max_citations <= 0
        or constraints.max_limitation_codes <= 0
    ):
        raise GroundingError(RejectionCode.SCHEMA_LIMIT_EXCEEDED, "Output bounds must be positive")


def _validate_limitation_codes(codes: Any) -> None:
    if not isinstance(codes, tuple):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Limitation codes must be a tuple")
    if any(not isinstance(code, LimitationCode) for code in codes):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Limitation codes must be LimitationCode values")
    if len(set(codes)) != len(codes):
        raise GroundingError(RejectionCode.REQUEST_MISMATCH, "Limitation codes are not unique")
    if codes != canonical_limitation_codes(codes):
        raise GroundingError(RejectionCode.REQUEST_MISMATCH, "Limitation codes are not canonically sorted")


def _validate_model_facing_field(field: Any, document_type: str) -> None:
    if not isinstance(field, ModelFacingField):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Evidence field is not typed")
    spec = AUTHORISED_FIELDS.get(field.field_key)
    if spec is None or document_type not in spec.document_types:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Field is not authorised for this document")
    rebuilt = make_model_facing_field(
        field.field_key,
        field.label,
        field.canonical_value,
        field.value_type,
        field.unit,
    )
    if rebuilt != field:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Field failed typed reconstruction")


def _validate_evidence_item(item: Any, alias: str) -> None:
    if not isinstance(item, GroundedEvidenceItem):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Evidence item is not typed")
    if item.alias != alias:
        raise GroundingError(RejectionCode.REQUEST_MISMATCH, "Evidence aliases are not E1..En")
    if item.document_type not in DOCUMENT_ALLOWED_STATUSES:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Unsupported document_type")
    if item.evidence_status not in DOCUMENT_ALLOWED_STATUSES[item.document_type]:
        raise GroundingError(
            RejectionCode.EVIDENCE_FIELD_INVALID,
            "evidence_status is not permitted for this document_type",
        )
    if not item.fields:
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Evidence item has no fields")
    seen: set[str] = set()
    by_key: dict[str, ModelFacingField] = {}
    for field in item.fields:
        _validate_model_facing_field(field, item.document_type)
        if field.field_key in seen:
            raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Duplicate field_key")
        seen.add(field.field_key)
        by_key[field.field_key] = field
    if not isinstance(item.claimable_field_keys, tuple):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "claimable_field_keys must be a tuple")
    if len(set(item.claimable_field_keys)) != len(item.claimable_field_keys):
        raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "claimable_field_keys are not unique")
    for key in item.claimable_field_keys:
        field = by_key.get(key)
        if field is None:
            raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "claimable key is absent from fields")
        if field.value_type not in CLAIMABLE_TYPES:
            raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "claimable key is not DECIMAL or INTEGER")
        if item.evidence_status == EVIDENCE_STATUS_NO_MARKET_DATA and key in EXCURSION_FIELD_KEYS:
            raise GroundingError(RejectionCode.EVIDENCE_FIELD_INVALID, "Unavailable excursion field is claimable")
    if item.document_type == DOCUMENT_TYPE_TRADE:
        _validate_trade_excursion_state(item, by_key)
        _validate_bar_evidence_status_parity(item, by_key)


def _validate_trade_excursion_state(
    item: GroundedEvidenceItem,
    by_key: Mapping[str, ModelFacingField],
) -> None:
    if item.evidence_status == EVIDENCE_STATUS_COMPUTED:
        if any(key not in by_key for key in REQUIRED_COMPUTED_EXCURSION_KEYS):
            raise GroundingError(
                RejectionCode.EVIDENCE_FIELD_INVALID,
                "COMPUTED trade evidence is missing a required excursion field",
            )
        return
    if EXCURSION_FIELD_KEYS.intersection(by_key):
        raise GroundingError(
            RejectionCode.EVIDENCE_FIELD_INVALID,
            "Non-COMPUTED trade evidence must not include excursion fields",
        )


def _validate_bar_evidence_status_parity(
    item: GroundedEvidenceItem,
    by_key: Mapping[str, ModelFacingField],
) -> None:
    field = by_key.get("bar_evidence_status")
    if field is None:
        return
    if field.canonical_value != item.evidence_status:
        raise GroundingError(
            RejectionCode.STATUS_CONTRADICTION,
            "bar_evidence_status does not match evidence_status",
        )


def _evidence_derived_limitation_conditions(
    items: Sequence[GroundedEvidenceItem],
) -> tuple[bool, bool, bool]:
    has_approx = False
    has_unavailable = False
    symbols: set[str] = set()
    for item in items:
        if item.document_type != DOCUMENT_TYPE_TRADE:
            continue
        if item.evidence_status == EVIDENCE_STATUS_NO_MARKET_DATA:
            has_unavailable = True
        if item.evidence_status != EVIDENCE_STATUS_COMPUTED:
            continue
        keys = {field.field_key for field in item.fields}
        if "approx_mfe" not in keys and "approx_mae" not in keys:
            continue
        has_approx = True
        for field in item.fields:
            if field.field_key == "symbol":
                symbols.add(field.canonical_value)
    return has_approx, has_unavailable, len(symbols) > 1


def _validate_evidence_derived_limitations(
    items: Sequence[GroundedEvidenceItem],
    codes: tuple[LimitationCode, ...],
) -> None:
    has_approx, has_unavailable, has_cross = _evidence_derived_limitation_conditions(items)
    present = set(codes)
    required: list[LimitationCode] = []
    forbidden: list[LimitationCode] = []
    if has_approx:
        required.append(LimitationCode.APPROXIMATE_M1_EVIDENCE)
    else:
        forbidden.append(LimitationCode.APPROXIMATE_M1_EVIDENCE)
    if has_unavailable:
        required.append(LimitationCode.EVIDENCE_UNAVAILABLE)
    else:
        forbidden.append(LimitationCode.EVIDENCE_UNAVAILABLE)
    if has_cross:
        required.append(LimitationCode.CROSS_SYMBOL_NOT_COMPARABLE)
    else:
        forbidden.append(LimitationCode.CROSS_SYMBOL_NOT_COMPARABLE)
    for code in required:
        if code not in present:
            raise GroundingError(RejectionCode.LIMITATION_REQUIRED_MISSING)
    for code in forbidden:
        if code in present:
            raise GroundingError(RejectionCode.STATUS_CONTRADICTION)


def validate_request_shape(request: GroundedAIRequest) -> None:
    if not isinstance(request, GroundedAIRequest):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Request is not a GroundedAIRequest")
    if request.schema_version != GROUNDING_REQUEST_SCHEMA_VERSION:
        raise GroundingError(RejectionCode.SCHEMA_VERSION_MISMATCH)
    if request.prompt_template_version != PROMPT_TEMPLATE_VERSION:
        raise GroundingError(RejectionCode.SCHEMA_VERSION_MISMATCH)
    if request.response_schema_version != GROUNDING_RESPONSE_SCHEMA_VERSION:
        raise GroundingError(RejectionCode.SCHEMA_VERSION_MISMATCH)
    if request.task_type != TASK_TYPE:
        raise GroundingError(RejectionCode.SCHEMA_VERSION_MISMATCH)
    _validate_constraints(request.constraints)
    _validate_question(request.question)
    if not isinstance(request.evidence, tuple):
        raise GroundingError(RejectionCode.SCHEMA_TYPE_ERROR, "Evidence must be a tuple")
    count = len(request.evidence)
    if count < MIN_EVIDENCE_ITEMS:
        raise GroundingError(RejectionCode.REQUEST_EMPTY_EVIDENCE)
    if count > MAX_EVIDENCE_ITEMS:
        raise GroundingError(RejectionCode.SCHEMA_LIMIT_EXCEEDED)
    expected = tuple(f"E{index}" for index in range(1, count + 1))
    actual = tuple(item.alias if isinstance(item, GroundedEvidenceItem) else "" for item in request.evidence)
    if actual != expected:
        raise GroundingError(RejectionCode.REQUEST_MISMATCH, "Evidence aliases are not E1..En")
    for alias, item in zip(expected, request.evidence, strict=True):
        _validate_evidence_item(item, alias)
    _validate_limitation_codes(request.required_limitation_codes)
    _validate_evidence_derived_limitations(request.evidence, request.required_limitation_codes)


def candidate_response_example() -> dict[str, Any]:
    return {
        "schema_version": GROUNDING_RESPONSE_SCHEMA_VERSION,
        "answer": "...",
        "claims": [
            {
                "evidence": "E1",
                "field": "approx_mae",
                "value": "-54.7",
            }
        ],
        "citations": ["E1"],
        "limitation_codes": ["APPROXIMATE_M1_EVIDENCE"],
    }


__all__ = [
    "ACCEPTED_EVIDENCE_STATUSES",
    "ACCEPTED_SIDES",
    "AUTHORISED_FIELDS",
    "CLAIMABLE_TYPES",
    "DATASET_EVIDENCE_STATUSES",
    "DEFAULT_CONSTRAINTS",
    "DOCUMENT_ALLOWED_STATUSES",
    "GROUNDING_REQUEST_SCHEMA_VERSION",
    "GROUNDING_RESPONSE_SCHEMA_VERSION",
    "KPI_EVIDENCE_STATUSES",
    "MAX_ANSWER_CHARS",
    "MAX_CITATIONS",
    "MAX_CLAIMS",
    "MAX_EVIDENCE_ITEMS",
    "MAX_LIMITATION_CODES",
    "MAX_QUESTION_LENGTH",
    "MIN_EVIDENCE_ITEMS",
    "PROMPT_TEMPLATE_VERSION",
    "SYMBOL_PATTERN",
    "SYMBOL_RE",
    "TASK_TYPE",
    "TRADE_EVIDENCE_STATUSES",
    "FieldType",
    "GroundedAIRequest",
    "GroundedAIResponse",
    "GroundedClaim",
    "GroundedEvidenceItem",
    "GroundingConstraints",
    "LimitationCode",
    "ModelFacingField",
    "RejectionCode",
    "ServerGroundingContext",
    "canonical_limitation_codes",
    "canonical_request_json",
    "candidate_response_example",
    "make_model_facing_field",
    "normalise_question",
    "request_sha256",
    "validate_request_shape",
]
