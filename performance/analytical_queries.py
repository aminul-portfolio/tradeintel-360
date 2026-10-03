from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from numbers import Integral, Real

import pandas as pd

ANALYTICAL_QUERY_RESULT_SCHEMA_VERSION = "analytical-query-result-v1"
ENGINE_CONTRACT_VERSION = "deterministic-analytical-query-v1"
GROUPING_BASIS_RECORDED_WALL_CLOCK = "JOURNAL_RECORDED_WALL_CLOCK"

DATE_COLUMN_PRECEDENCE = ("Open Time", "Open", "Date")
PROFIT_COLUMN = "Profit"
SYMBOL_COLUMN = "Symbol"
UNIT_TRADES = "trades"
UNIT_PROFIT = "account currency (as recorded)"


class AnalyticalQueryId(str, Enum):
    TRADE_COUNT = "TRADE_COUNT"
    TRADE_COUNT_BY_DAY = "TRADE_COUNT_BY_DAY"
    BUSIEST_TRADING_DAY = "BUSIEST_TRADING_DAY"
    PROFIT_BY_DAY = "PROFIT_BY_DAY"
    TRADE_COUNT_BY_SYMBOL = "TRADE_COUNT_BY_SYMBOL"
    MOST_TRADED_SYMBOL = "MOST_TRADED_SYMBOL"
    PROFIT_BY_SYMBOL = "PROFIT_BY_SYMBOL"
    MONTHLY_PROFIT = "MONTHLY_PROFIT"


class AnalyticalQueryStatus(str, Enum):
    OK = "OK"
    NO_DATA = "NO_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    REQUIRED_COLUMN_MISSING = "REQUIRED_COLUMN_MISSING"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"
    AMBIGUOUS_TIME_BASIS = "AMBIGUOUS_TIME_BASIS"


class AnalyticalResultKind(str, Enum):
    SCALAR = "SCALAR"
    ROWS = "ROWS"
    TIED_MAX = "TIED_MAX"


class AnalyticalWarningCode(str, Enum):
    DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION = "DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION"
    SYMBOL_VARIANTS_DETECTED = "SYMBOL_VARIANTS_DETECTED"
    DUPLICATE_ROWS_DETECTED = "DUPLICATE_ROWS_DETECTED"
    PROFIT_EXCLUDES_COMMISSION_AND_SWAP = "PROFIT_EXCLUDES_COMMISSION_AND_SWAP"


class AnalyticalIssueCode(str, Enum):
    PROFIT_MISSING = "PROFIT_MISSING"
    PROFIT_NOT_NUMERIC = "PROFIT_NOT_NUMERIC"
    PROFIT_NON_FINITE = "PROFIT_NON_FINITE"
    DATE_UNPARSEABLE = "DATE_UNPARSEABLE"
    SYMBOL_MISSING = "SYMBOL_MISSING"


FROZEN_QUERY_IDS = tuple(item.value for item in AnalyticalQueryId)
FROZEN_STATUSES = tuple(item.value for item in AnalyticalQueryStatus)
FROZEN_RESULT_KINDS = tuple(item.value for item in AnalyticalResultKind)
FROZEN_WARNING_CODES = tuple(item.value for item in AnalyticalWarningCode)
FROZEN_ISSUE_CODES = tuple(item.value for item in AnalyticalIssueCode)

_QUERY_IDS = frozenset(FROZEN_QUERY_IDS)
_DATE_QUERIES = frozenset(
    {
        AnalyticalQueryId.TRADE_COUNT_BY_DAY.value,
        AnalyticalQueryId.BUSIEST_TRADING_DAY.value,
        AnalyticalQueryId.PROFIT_BY_DAY.value,
        AnalyticalQueryId.MONTHLY_PROFIT.value,
    }
)
_PROFIT_QUERIES = frozenset(
    {
        AnalyticalQueryId.PROFIT_BY_DAY.value,
        AnalyticalQueryId.PROFIT_BY_SYMBOL.value,
        AnalyticalQueryId.MONTHLY_PROFIT.value,
    }
)
_SYMBOL_QUERIES = frozenset(
    {
        AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL.value,
        AnalyticalQueryId.MOST_TRADED_SYMBOL.value,
        AnalyticalQueryId.PROFIT_BY_SYMBOL.value,
    }
)


def _freeze_filters(filters: object) -> tuple[tuple[str, str], ...]:
    if filters is None:
        return ()
    if isinstance(filters, Mapping):
        raw_items = list(filters.items())
    elif isinstance(filters, (tuple, list)):
        raw_items = []
        for entry in filters:
            if not isinstance(entry, (tuple, list)):
                raise ValueError("each filter entry must be a (key, value) pair")
            if len(entry) != 2:
                raise ValueError("each filter entry must contain exactly two items")
            raw_items.append((entry[0], entry[1]))
    else:
        raise ValueError("filters must be a mapping or a sequence of string pairs")
    seen: set[str] = set()
    frozen: list[tuple[str, str]] = []
    for key, value in raw_items:
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("filter keys and values must be strings")
        if key in seen:
            raise ValueError("duplicate filter keys are not allowed")
        seen.add(key)
        frozen.append((key, value))
    return tuple(sorted(frozen, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class AnalyticalInputContext:
    source_row_count: int
    filtered_row_count: int
    filters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_row_count, int) or isinstance(self.source_row_count, bool):
            raise ValueError("source_row_count must be a non-negative int")
        if not isinstance(self.filtered_row_count, int) or isinstance(self.filtered_row_count, bool):
            raise ValueError("filtered_row_count must be a non-negative int")
        if self.source_row_count < 0 or self.filtered_row_count < 0:
            raise ValueError("row counts must be non-negative")
        if self.filtered_row_count > self.source_row_count:
            raise ValueError("filtered_row_count cannot exceed source_row_count")
        object.__setattr__(self, "filters", _freeze_filters(self.filters))


@dataclass(frozen=True, slots=True)
class AnalyticalIssue:
    reason_code: AnalyticalIssueCode
    column: str
    row_count: int


@dataclass(frozen=True, slots=True)
class AnalyticalProvenance:
    engine_contract_version: str
    columns_used: tuple[str, ...]
    input_sha256: str | None


@dataclass(frozen=True, slots=True)
class AnalyticalQueryResult:
    schema_version: str
    query_id: str
    status: AnalyticalQueryStatus
    result_kind: AnalyticalResultKind | None
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    scalar: int | None
    tie: bool
    units: tuple[tuple[str, str], ...]
    source_row_count: int
    filtered_row_count: int
    filters: tuple[tuple[str, str], ...]
    grouping_basis: str | None
    date_column: str | None
    date_offset: str | None
    warnings: tuple[AnalyticalWarningCode, ...]
    warning_counts: tuple[tuple[str, int], ...]
    issues: tuple[AnalyticalIssue, ...]
    provenance: AnalyticalProvenance


def _empty_payload() -> dict[str, object]:
    return {
        "result_kind": None,
        "columns": (),
        "rows": (),
        "scalar": None,
        "tie": False,
        "units": (),
        "grouping_basis": None,
        "date_column": None,
        "date_offset": None,
        "warnings": (),
        "warning_counts": (),
        "issues": (),
    }


def _sort_issues(issues: tuple[AnalyticalIssue, ...]) -> tuple[AnalyticalIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda item: (item.reason_code.value, item.column, item.row_count),
        )
    )


def _sort_warnings(codes: set[AnalyticalWarningCode]) -> tuple[AnalyticalWarningCode, ...]:
    return tuple(sorted(codes, key=lambda item: item.value))


def _build_result(
    *,
    query_id: str,
    status: AnalyticalQueryStatus,
    context: AnalyticalInputContext,
    provenance: AnalyticalProvenance,
    issues: tuple[AnalyticalIssue, ...] = (),
    **payload: object,
) -> AnalyticalQueryResult:
    body = _empty_payload()
    body.update(payload)
    if status is not AnalyticalQueryStatus.OK:
        body = _empty_payload()
    return AnalyticalQueryResult(
        schema_version=ANALYTICAL_QUERY_RESULT_SCHEMA_VERSION,
        query_id=query_id,
        status=status,
        result_kind=body["result_kind"],  # type: ignore[arg-type]
        columns=body["columns"],  # type: ignore[arg-type]
        rows=body["rows"],  # type: ignore[arg-type]
        scalar=body["scalar"],  # type: ignore[arg-type]
        tie=bool(body["tie"]),
        units=body["units"],  # type: ignore[arg-type]
        source_row_count=context.source_row_count,
        filtered_row_count=context.filtered_row_count,
        filters=context.filters,
        grouping_basis=body["grouping_basis"],  # type: ignore[arg-type]
        date_column=body["date_column"],  # type: ignore[arg-type]
        date_offset=body["date_offset"],  # type: ignore[arg-type]
        warnings=body["warnings"],  # type: ignore[arg-type]
        warning_counts=body["warning_counts"],  # type: ignore[arg-type]
        issues=_sort_issues(issues) if status is not AnalyticalQueryStatus.OK else (),
        provenance=provenance,
    )


def _normalise_query_id(query_id: object) -> str | None:
    if isinstance(query_id, AnalyticalQueryId):
        return query_id.value
    if isinstance(query_id, str) and query_id in _QUERY_IDS:
        return query_id
    return None


def _is_missing_marker(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return type(value).__name__ == "NAType"


def _is_blank_string(value: object) -> bool:
    return isinstance(value, str) and value.strip() == ""


def _canonical_float_token(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return repr(value)


def _canonical_scalar(value: object) -> object:
    if _is_missing_marker(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return {"float": _canonical_float_token(value)}
    if isinstance(value, Real) and not isinstance(value, bool):
        return {"float": _canonical_float_token(float(value))}
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return {"timestamp": value.isoformat()}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if isinstance(value, Decimal):
        return {"decimal": format(value, "f")}
    raise TypeError("unsupported scalar type: " + type(value).__name__)


def _row_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[list[object]]:
    if not columns:
        return [[] for _ in range(len(frame))]
    rows: list[list[object]] = []
    for record in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        rows.append([_canonical_scalar(item) for item in record])
    return rows


def compute_input_sha256(frame: pd.DataFrame, columns_used: tuple[str, ...]) -> str:
    canonical_rows = _row_values(frame, columns_used)
    canonical_rows.sort(key=lambda row: json.dumps(row, ensure_ascii=True, separators=(",", ":")))
    payload = {
        "columns": list(columns_used),
        "row_count": int(len(frame)),
        "rows": canonical_rows,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duplicate_row_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame.duplicated(keep=False).sum())


def _common_warnings(
    frame: pd.DataFrame,
    *,
    date_ok: bool,
    profit_ok: bool,
    symbol_values: tuple[str, ...] = (),
) -> tuple[tuple[AnalyticalWarningCode, ...], tuple[tuple[str, int], ...]]:
    codes: set[AnalyticalWarningCode] = set()
    counts: list[tuple[str, int]] = []
    duplicate_count = _duplicate_row_count(frame)
    if duplicate_count:
        codes.add(AnalyticalWarningCode.DUPLICATE_ROWS_DETECTED)
        counts.append((AnalyticalWarningCode.DUPLICATE_ROWS_DETECTED.value, duplicate_count))
    if date_ok:
        codes.add(AnalyticalWarningCode.DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION)
    if profit_ok:
        codes.add(AnalyticalWarningCode.PROFIT_EXCLUDES_COMMISSION_AND_SWAP)
    if symbol_values:
        folded: dict[str, set[str]] = {}
        for symbol in symbol_values:
            folded.setdefault(symbol.strip().casefold(), set()).add(symbol)
        if any(len(group) >= 2 for group in folded.values()):
            codes.add(AnalyticalWarningCode.SYMBOL_VARIANTS_DETECTED)
    return _sort_warnings(codes), tuple(sorted(counts, key=lambda item: item[0]))


def _resolve_date_column(frame: pd.DataFrame) -> str | None:
    for name in DATE_COLUMN_PRECEDENCE:
        if name in frame.columns:
            return name
    return None


def _format_offset(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _wall_and_offset(value: datetime) -> tuple[date, timedelta | None] | None:
    wall = date(value.year, value.month, value.day)
    if value.tzinfo is None:
        return wall, None
    offset = value.utcoffset()
    if offset is None:
        return None
    return wall, offset


def _parse_recorded_date(value: object) -> tuple[date, timedelta | None] | None:
    if _is_missing_marker(value) or _is_blank_string(value):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime):
        return _wall_and_offset(value)
    if isinstance(value, date):
        return value, None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        wall = date(int(value.year), int(value.month), int(value.day))
        if value.tzinfo is None:
            return wall, None
        offset = value.utcoffset()
        if offset is None:
            return None
        return wall, offset
    if isinstance(value, str):
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
            return None
        if pd.isna(stamp):
            return None
        return _parse_recorded_date(stamp)
    return None


def _collect_dates(
    series: pd.Series,
) -> tuple[list[date], str | None, tuple[AnalyticalIssue, ...], bool]:
    parsed: list[date] = []
    offsets: list[timedelta | None] = []
    unparseable = 0
    for value in series.tolist():
        item = _parse_recorded_date(value)
        if item is None:
            unparseable += 1
            continue
        wall, offset = item
        parsed.append(wall)
        offsets.append(offset)
    if unparseable:
        issue = AnalyticalIssue(
            reason_code=AnalyticalIssueCode.DATE_UNPARSEABLE,
            column=str(series.name),
            row_count=unparseable,
        )
        return [], None, (issue,), False
    naive = [item is None for item in offsets]
    if any(naive) and not all(naive):
        return [], None, (), True
    if all(naive):
        return parsed, None, (), False
    first = offsets[0]
    if first is None or any(item != first for item in offsets):
        return [], None, (), True
    return parsed, _format_offset(first), (), False


def _classify_profit(value: object) -> tuple[float | None, AnalyticalIssueCode | None]:
    if _is_missing_marker(value) or _is_blank_string(value):
        return None, AnalyticalIssueCode.PROFIT_MISSING
    if isinstance(value, bool):
        return None, AnalyticalIssueCode.PROFIT_NOT_NUMERIC
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return None, AnalyticalIssueCode.PROFIT_NOT_NUMERIC
        if not math.isfinite(number):
            return None, AnalyticalIssueCode.PROFIT_NON_FINITE
        return number, None
    if isinstance(value, Decimal):
        number = float(value)
        if not math.isfinite(number):
            return None, AnalyticalIssueCode.PROFIT_NON_FINITE
        return number, None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return float(int(value)), None
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number) or not math.isfinite(number):
            return None, AnalyticalIssueCode.PROFIT_NON_FINITE
        return number, None
    return None, AnalyticalIssueCode.PROFIT_NOT_NUMERIC


def _collect_profits(series: pd.Series) -> tuple[list[float], tuple[AnalyticalIssue, ...]]:
    values: list[float] = []
    counts: Counter[AnalyticalIssueCode] = Counter()
    for value in series.tolist():
        number, code = _classify_profit(value)
        if code is not None:
            counts[code] += 1
            continue
        assert number is not None
        values.append(number)
    if counts:
        issues = tuple(
            AnalyticalIssue(reason_code=code, column=PROFIT_COLUMN, row_count=count)
            for code, count in counts.items()
        )
        return [], issues
    return values, ()


def _ordered_fsum(values: list[float]) -> float:
    ordered = sorted(values, key=lambda item: (math.copysign(1.0, item), abs(item), repr(item)))
    return math.fsum(ordered)


def _profit_display(value_raw: float) -> str:
    quantized = Decimal(repr(value_raw)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    return format(quantized, "f")


def _collect_symbols(series: pd.Series) -> tuple[list[str], tuple[AnalyticalIssue, ...]]:
    symbols: list[str] = []
    missing = 0
    for value in series.tolist():
        if _is_missing_marker(value) or _is_blank_string(value):
            missing += 1
            continue
        if not isinstance(value, str):
            missing += 1
            continue
        symbols.append(value)
    if missing:
        issue = AnalyticalIssue(
            reason_code=AnalyticalIssueCode.SYMBOL_MISSING,
            column=SYMBOL_COLUMN,
            row_count=missing,
        )
        return [], (issue,)
    return symbols, ()


def _count_units() -> tuple[tuple[str, str], ...]:
    return (("trade_count", UNIT_TRADES),)


def _profit_units() -> tuple[tuple[str, str], ...]:
    return (
        ("profit_raw", UNIT_PROFIT),
        ("profit_display", UNIT_PROFIT),
    )


def _count_by_key(keys: list[str]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


def _sum_by_key(keys: list[str], amounts: list[float]) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = {}
    for key, amount in zip(keys, amounts, strict=True):
        grouped.setdefault(key, []).append(amount)
    return [(key, _ordered_fsum(grouped[key])) for key in sorted(grouped)]


def _tied_max(pairs: list[tuple[str, int]]) -> tuple[tuple[tuple[object, ...], ...], bool]:
    if not pairs:
        return (), False
    maximum = max(item[1] for item in pairs)
    winners = tuple((key, count) for key, count in pairs if count == maximum)
    return winners, len(winners) >= 2


def _columns_used(query_id: str, frame: pd.DataFrame) -> tuple[str, ...]:
    used: list[str] = []
    if query_id in _DATE_QUERIES:
        resolved = _resolve_date_column(frame)
        if resolved is not None:
            used.append(resolved)
    if query_id in _SYMBOL_QUERIES and SYMBOL_COLUMN in frame.columns:
        used.append(SYMBOL_COLUMN)
    if query_id in _PROFIT_QUERIES and PROFIT_COLUMN in frame.columns:
        used.append(PROFIT_COLUMN)
    return tuple(used)


def _missing_required_columns(query_id: str, frame: pd.DataFrame) -> bool:
    if query_id == AnalyticalQueryId.TRADE_COUNT.value:
        return False
    if query_id in _DATE_QUERIES and _resolve_date_column(frame) is None:
        return True
    if query_id in _SYMBOL_QUERIES and SYMBOL_COLUMN not in frame.columns:
        return True
    if query_id in _PROFIT_QUERIES and PROFIT_COLUMN not in frame.columns:
        return True
    return False


def execute_analytical_query(
    dataframe: object,
    query_id: object,
    context: AnalyticalInputContext,
) -> AnalyticalQueryResult:
    normalised = _normalise_query_id(query_id)
    empty_provenance = AnalyticalProvenance(
        engine_contract_version=ENGINE_CONTRACT_VERSION,
        columns_used=(),
        input_sha256=None,
    )
    if normalised is None:
        return _build_result(
            query_id=str(query_id),
            status=AnalyticalQueryStatus.UNSUPPORTED_QUERY,
            context=context,
            provenance=empty_provenance,
        )
    if int(context.filtered_row_count) != len(dataframe):  # type: ignore[arg-type]
        raise ValueError("filtered_row_count does not match dataframe length")
    working = dataframe.copy(deep=True)
    columns_used = _columns_used(normalised, working)
    provenance = AnalyticalProvenance(
        engine_contract_version=ENGINE_CONTRACT_VERSION,
        columns_used=columns_used,
        input_sha256=compute_input_sha256(working, columns_used),
    )
    if len(working) == 0:
        return _build_result(
            query_id=normalised,
            status=AnalyticalQueryStatus.NO_DATA,
            context=context,
            provenance=provenance,
        )
    if _missing_required_columns(normalised, working):
        return _build_result(
            query_id=normalised,
            status=AnalyticalQueryStatus.REQUIRED_COLUMN_MISSING,
            context=context,
            provenance=provenance,
        )
    return _execute_known_query(working, normalised, context, provenance)


def _execute_known_query(
    frame: pd.DataFrame,
    query_id: str,
    context: AnalyticalInputContext,
    provenance: AnalyticalProvenance,
) -> AnalyticalQueryResult:
    if query_id == AnalyticalQueryId.TRADE_COUNT.value:
        warnings, warning_counts = _common_warnings(frame, date_ok=False, profit_ok=False)
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.SCALAR,
            columns=(),
            rows=(),
            scalar=int(len(frame)),
            tie=False,
            units=_count_units(),
            warnings=warnings,
            warning_counts=warning_counts,
        )

    dates: list[date] = []
    date_column: str | None = None
    date_offset: str | None = None
    if query_id in _DATE_QUERIES:
        date_column = _resolve_date_column(frame)
        assert date_column is not None
        dates, date_offset, date_issues, ambiguous = _collect_dates(frame[date_column])
        if date_issues:
            return _build_result(
                query_id=query_id,
                status=AnalyticalQueryStatus.INVALID_INPUT,
                context=context,
                provenance=provenance,
                issues=date_issues,
            )
        if ambiguous:
            return _build_result(
                query_id=query_id,
                status=AnalyticalQueryStatus.AMBIGUOUS_TIME_BASIS,
                context=context,
                provenance=provenance,
            )

    symbols: list[str] = []
    if query_id in _SYMBOL_QUERIES:
        symbols, symbol_issues = _collect_symbols(frame[SYMBOL_COLUMN])
        if symbol_issues:
            return _build_result(
                query_id=query_id,
                status=AnalyticalQueryStatus.INVALID_INPUT,
                context=context,
                provenance=provenance,
                issues=symbol_issues,
            )

    profits: list[float] = []
    if query_id in _PROFIT_QUERIES:
        profits, profit_issues = _collect_profits(frame[PROFIT_COLUMN])
        if profit_issues:
            return _build_result(
                query_id=query_id,
                status=AnalyticalQueryStatus.INVALID_INPUT,
                context=context,
                provenance=provenance,
                issues=profit_issues,
            )

    warnings, warning_counts = _common_warnings(
        frame,
        date_ok=query_id in _DATE_QUERIES,
        profit_ok=query_id in _PROFIT_QUERIES,
        symbol_values=tuple(symbols),
    )
    grouping = GROUPING_BASIS_RECORDED_WALL_CLOCK if query_id in _DATE_QUERIES else None

    if query_id == AnalyticalQueryId.TRADE_COUNT_BY_DAY.value:
        pairs = _count_by_key([item.isoformat() for item in dates])
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.ROWS,
            columns=("day", "trade_count"),
            rows=tuple((day, count) for day, count in pairs),
            units=_count_units(),
            grouping_basis=grouping,
            date_column=date_column,
            date_offset=date_offset,
            warnings=warnings,
            warning_counts=warning_counts,
        )

    if query_id == AnalyticalQueryId.BUSIEST_TRADING_DAY.value:
        pairs = _count_by_key([item.isoformat() for item in dates])
        winners, tied = _tied_max(pairs)
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.TIED_MAX,
            columns=("day", "trade_count"),
            rows=winners,
            tie=tied,
            units=_count_units(),
            grouping_basis=grouping,
            date_column=date_column,
            date_offset=date_offset,
            warnings=warnings,
            warning_counts=warning_counts,
        )

    if query_id == AnalyticalQueryId.PROFIT_BY_DAY.value:
        pairs = _sum_by_key([item.isoformat() for item in dates], profits)
        rows = tuple((day, raw, _profit_display(raw)) for day, raw in pairs)
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.ROWS,
            columns=("day", "profit_raw", "profit_display"),
            rows=rows,
            units=_profit_units(),
            grouping_basis=grouping,
            date_column=date_column,
            date_offset=date_offset,
            warnings=warnings,
            warning_counts=warning_counts,
        )

    if query_id == AnalyticalQueryId.MONTHLY_PROFIT.value:
        months = [f"{item.year:04d}-{item.month:02d}" for item in dates]
        pairs = _sum_by_key(months, profits)
        rows = tuple((month, raw, _profit_display(raw)) for month, raw in pairs)
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.ROWS,
            columns=("month", "profit_raw", "profit_display"),
            rows=rows,
            units=_profit_units(),
            grouping_basis=grouping,
            date_column=date_column,
            date_offset=date_offset,
            warnings=warnings,
            warning_counts=warning_counts,
        )

    if query_id == AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL.value:
        pairs = _count_by_key(symbols)
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.ROWS,
            columns=("symbol", "trade_count"),
            rows=tuple((symbol, count) for symbol, count in pairs),
            units=_count_units(),
            warnings=warnings,
            warning_counts=warning_counts,
        )

    if query_id == AnalyticalQueryId.MOST_TRADED_SYMBOL.value:
        pairs = _count_by_key(symbols)
        winners, tied = _tied_max(pairs)
        return _build_result(
            query_id=query_id,
            status=AnalyticalQueryStatus.OK,
            context=context,
            provenance=provenance,
            result_kind=AnalyticalResultKind.TIED_MAX,
            columns=("symbol", "trade_count"),
            rows=winners,
            tie=tied,
            units=_count_units(),
            warnings=warnings,
            warning_counts=warning_counts,
        )

    pairs = _sum_by_key(symbols, profits)
    rows = tuple((symbol, raw, _profit_display(raw)) for symbol, raw in pairs)
    return _build_result(
        query_id=query_id,
        status=AnalyticalQueryStatus.OK,
        context=context,
        provenance=provenance,
        result_kind=AnalyticalResultKind.ROWS,
        columns=("symbol", "profit_raw", "profit_display"),
        rows=rows,
        units=_profit_units(),
        warnings=warnings,
        warning_counts=warning_counts,
    )


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN/inf cannot appear in canonical result JSON")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    raise TypeError("unsupported result value: " + type(value).__name__)


def canonical_result_json(result: AnalyticalQueryResult) -> str:
    return json.dumps(
        _jsonable(result),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
