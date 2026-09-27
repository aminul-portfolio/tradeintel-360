from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

import pandas as pd

from .time_basis import NormalisationStatus, TimeBasis, normalise_journal_datetime

MAX_TRADE_WINDOW_MINUTES = 10_080
DURATION_TOLERANCE_SECONDS = 1
PRICE_COMPARISON_EPSILON = 1e-9

RUN_STATUS_OK = "OK"
RUN_STATUS_TIME_BASIS_NOT_DECLARED = "TIME_BASIS_NOT_DECLARED"
RUN_STATUS_JOIN_KEY_UNAVAILABLE = "JOIN_KEY_UNAVAILABLE"

STATUS_INVALID_TRADE_DATA = "INVALID_TRADE_DATA"
STATUS_TIMEZONE_AMBIGUOUS = "TIMEZONE_AMBIGUOUS"
STATUS_TIME_BASIS_INCONSISTENT = "TIME_BASIS_INCONSISTENT"
STATUS_NO_MARKET_DATA = "NO_MARKET_DATA"
STATUS_INCOMPLETE_COVERAGE = "INCOMPLETE_COVERAGE"
STATUS_INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
STATUS_COMPUTED = "COMPUTED"

REASON_WINDOW_EXCEEDS_LIMIT = "WINDOW_EXCEEDS_LIMIT"
REASON_OUTSIDE_FILE_RANGE = "OUTSIDE_FILE_RANGE"
REASON_BOUNDARY_BAR_MISSING = "BOUNDARY_BAR_MISSING"
REASON_INTERIOR_GAP = "INTERIOR_GAP"
REASON_AMBIGUOUS = "AMBIGUOUS"
REASON_NONEXISTENT = "NONEXISTENT"
REASON_INVARIANT_FAILED = "INVARIANT_FAILED"

TICKET_COLUMNS = ("Ticket",)
OPEN_COLUMNS = ("Open Time", "Open", "Date")
CLOSE_COLUMNS = ("Close Time", "Close")
SYMBOL_COLUMNS = ("Symbol",)
SIDE_COLUMNS = ("Type", "Side")
ENTRY_COLUMNS = ("Entry", "Price", "Open Price", "Entry Price")
EXIT_COLUMNS = ("Exit", "Price.1", "Close Price", "Exit Price")
DURATION_COLUMNS = ("Trade duration in seconds",)

_BUY_SIDES = frozenset({"buy", "long"})
_SELL_SIDES = frozenset({"sell", "short"})


class ExcursionError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True)
class JournalColumns:
    ticket_column: str | None
    open_column: str | None
    close_column: str | None
    symbol_column: str | None
    side_column: str | None
    entry_column: str | None
    exit_column: str | None
    duration_column: str | None

    def as_json(self) -> dict[str, str | None]:
        return {
            "ticket_column": self.ticket_column,
            "open_column": self.open_column,
            "close_column": self.close_column,
            "symbol_column": self.symbol_column,
            "side_column": self.side_column,
            "entry_column": self.entry_column,
            "exit_column": self.exit_column,
            "duration_column": self.duration_column,
        }


@dataclass(frozen=True)
class ExcursionRunResult:
    run_status: str
    reason_code: str | None
    journal_fingerprint: str | None
    evidence: MappingProxyType
    status_counts: dict[str, int]
    reason_counts: dict[str, int]

    def as_json(self) -> dict[str, Any]:
        return {
            "run_status": self.run_status,
            "reason_code": self.reason_code,
            "journal_fingerprint": self.journal_fingerprint,
            "evidence": dict(self.evidence),
            "status_counts": dict(self.status_counts),
            "reason_counts": dict(self.reason_counts),
        }


def _first_present(columns: Any, candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    return None


def resolve_journal_columns(journal: pd.DataFrame) -> JournalColumns:
    columns = journal.columns
    return JournalColumns(
        ticket_column=_first_present(columns, TICKET_COLUMNS),
        open_column=_first_present(columns, OPEN_COLUMNS),
        close_column=_first_present(columns, CLOSE_COLUMNS),
        symbol_column=_first_present(columns, SYMBOL_COLUMNS),
        side_column=_first_present(columns, SIDE_COLUMNS),
        entry_column=_first_present(columns, ENTRY_COLUMNS),
        exit_column=_first_present(columns, EXIT_COLUMNS),
        duration_column=_first_present(columns, DURATION_COLUMNS),
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def canonical_ticket(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return format(value, "f").rstrip("0").rstrip(".")
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return text
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return format(number.to_integral_value(), "f")
    return format(number, "f")


def _json_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ExcursionError("NON_FINITE_VALUE", "Value is not JSON-safe")
    return number


def _blank_flags() -> dict[str, bool]:
    return {
        "high_from_boundary_bar": False,
        "low_from_boundary_bar": False,
        "entry_outside_first_bar_range": False,
        "exit_outside_last_bar_range": False,
        "realised_outside_interval": False,
    }


def _evidence_payload(
    ticket: str,
    status: str,
    reason_code: str | None,
    *,
    interval_high: float | None = None,
    interval_low: float | None = None,
    mfe: float | None = None,
    mae: float | None = None,
    bars_used: int | None = None,
    bars_expected: int | None = None,
    flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    computed = status == STATUS_COMPUTED
    resolved_flags = flags if computed and flags is not None else _blank_flags()
    return {
        "ticket": ticket,
        "status": status,
        "reason_code": reason_code,
        "interval_high": interval_high if computed else None,
        "interval_low": interval_low if computed else None,
        "mfe": mfe if computed else None,
        "mae": mae if computed else None,
        "bars_used": bars_used,
        "bars_expected": bars_expected,
        "high_from_boundary_bar": resolved_flags["high_from_boundary_bar"],
        "low_from_boundary_bar": resolved_flags["low_from_boundary_bar"],
        "entry_outside_first_bar_range": resolved_flags["entry_outside_first_bar_range"],
        "exit_outside_last_bar_range": resolved_flags["exit_outside_last_bar_range"],
        "realised_outside_interval": resolved_flags["realised_outside_interval"],
    }


def _failed(
    ticket: str,
    status: str,
    reason_code: str | None,
    *,
    bars_used: int | None = None,
    bars_expected: int | None = None,
) -> dict[str, Any]:
    return _evidence_payload(
        ticket,
        status,
        reason_code,
        bars_used=bars_used,
        bars_expected=bars_expected,
    )


def _cell(row: Any, column: str | None) -> Any:
    if column is None:
        return None
    if column not in row.index:
        return None
    return row[column]


def _parse_price(value: Any) -> float | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _parse_duration(value: Any) -> tuple[str, float | None]:
    if _is_missing(value):
        return "skip", None
    text = str(value).strip()
    if text == "":
        return "skip", None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return "invalid", None
    if not math.isfinite(number) or number < 0:
        return "invalid", None
    return "ok", number


def _parse_naive_datetime(value: Any) -> tuple[str, datetime | None]:
    if _is_missing(value):
        return "invalid", None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return "aware", None
        return "ok", value.to_pydatetime().replace(tzinfo=None)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return "aware", None
        return "ok", value
    parsed = pd.to_datetime(value, errors="coerce")
    if _is_missing(parsed):
        return "invalid", None
    if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
        return "aware", None
    return "ok", parsed.to_pydatetime().replace(tzinfo=None)


def _canonical_side(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in _BUY_SIDES:
        return "buy"
    if text in _SELL_SIDES:
        return "sell"
    return None


def _canonical_symbol(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def _source_datetime_token(value: Any) -> Any:
    kind, parsed = _parse_naive_datetime(value)
    if kind == "ok" and parsed is not None:
        return parsed.strftime("%Y-%m-%dT%H:%M:%S.%f")
    raw = "" if _is_missing(value) else str(value)
    return {"kind": kind, "raw": raw}


def _source_number_token(value: Any) -> Any:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        return {"kind": "invalid", "raw": text}
    if not number.is_finite():
        return {"kind": "nonfinite", "raw": text}
    if number == 0:
        return "0"
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _floor_minute_utc(value: datetime) -> datetime:
    utc_value = value.astimezone(timezone.utc)
    return utc_value.replace(second=0, microsecond=0)


def _index_market(market: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if market is None or market.empty:
        return {}
    required = {"TimestampUTC", "Symbol", "Open", "High", "Low", "Close"}
    if not required.issubset(set(market.columns)):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for symbol, group in market.groupby("Symbol", sort=True):
        ordered = group.sort_values("TimestampUTC", kind="mergesort").reset_index(drop=True)
        stamps = pd.to_datetime(ordered["TimestampUTC"], utc=True)
        stamps = stamps.dt.tz_convert("UTC").dt.floor("min")
        frame = ordered.copy()
        frame.index = pd.DatetimeIndex(stamps, name="TimestampUTC")
        indexed[str(symbol)] = {
            "frame": frame,
            "minimum": frame.index.min(),
            "maximum": frame.index.max(),
        }
    return indexed


def _counts(evidence: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for item in evidence.values():
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        reason = item["reason_code"]
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return (
        dict(sorted(status_counts.items())),
        dict(sorted(reason_counts.items())),
    )


def _empty_result(
    run_status: str,
    reason_code: str | None,
    fingerprint: str | None,
) -> ExcursionRunResult:
    return ExcursionRunResult(
        run_status=run_status,
        reason_code=reason_code,
        journal_fingerprint=fingerprint,
        evidence=MappingProxyType({}),
        status_counts={},
        reason_counts={},
    )


def _join_tickets(
    journal: pd.DataFrame,
    columns: JournalColumns,
) -> tuple[str | None, list[str] | None]:
    if columns.ticket_column is None:
        return "MISSING_TICKET", None
    tickets: list[str] = []
    seen: set[str] = set()
    for value in journal[columns.ticket_column]:
        ticket = canonical_ticket(value)
        if ticket is None:
            if _is_missing(value) or str(value).strip() == "":
                return "BLANK_TICKET" if not _is_missing(value) else "NULL_TICKET", None
            return "INVALID_TICKET", None
        if ticket in seen:
            return "DUPLICATE_TICKET", None
        seen.add(ticket)
        tickets.append(ticket)
    return None, tickets


def compute_journal_fingerprint(journal: pd.DataFrame) -> str | None:
    columns = resolve_journal_columns(journal)
    join_reason, tickets = _join_tickets(journal, columns)
    if join_reason is not None or tickets is None:
        return None
    rows = []
    for ticket, (_, raw) in zip(tickets, journal.iterrows(), strict=True):
        row = {
            "ticket": ticket,
            "open": _source_datetime_token(_cell(raw, columns.open_column)),
            "close": _source_datetime_token(_cell(raw, columns.close_column)),
            "symbol": None
            if _is_missing(_cell(raw, columns.symbol_column))
            else str(_cell(raw, columns.symbol_column)).strip(),
            "side": None
            if _is_missing(_cell(raw, columns.side_column))
            else str(_cell(raw, columns.side_column)).strip(),
            "entry": _source_number_token(_cell(raw, columns.entry_column)),
            "exit": _source_number_token(_cell(raw, columns.exit_column)),
        }
        if columns.duration_column is not None:
            row["duration"] = _source_number_token(_cell(raw, columns.duration_column))
        rows.append(row)
    rows.sort(key=lambda item: item["ticket"])
    payload = {
        "columns": columns.as_json(),
        "rows": rows,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalise_leg(
    value: datetime,
    time_basis: TimeBasis,
) -> tuple[str | None, str | None, datetime | None]:
    result = normalise_journal_datetime(value, time_basis)
    if result.status == NormalisationStatus.VALID.value:
        return None, None, result.utc_datetime
    if result.status == NormalisationStatus.AMBIGUOUS.value:
        return STATUS_TIMEZONE_AMBIGUOUS, REASON_AMBIGUOUS, None
    if result.status == NormalisationStatus.NONEXISTENT.value:
        return STATUS_TIMEZONE_AMBIGUOUS, REASON_NONEXISTENT, None
    return STATUS_INVALID_TRADE_DATA, "INVALID_TIMESTAMP", None


def _outside_range(price: float, low: float, high: float) -> bool:
    return price < (low - PRICE_COMPARISON_EPSILON) or price > (
        high + PRICE_COMPARISON_EPSILON
    )


def _compute_trade(
    ticket: str,
    raw: Any,
    columns: JournalColumns,
    time_basis: TimeBasis,
    market_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if columns.open_column is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "MISSING_OPEN")
    if columns.close_column is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "MISSING_CLOSE")
    if columns.symbol_column is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "MISSING_SYMBOL")
    if columns.side_column is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "MISSING_SIDE")
    if columns.entry_column is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "MISSING_ENTRY")
    if columns.exit_column is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "MISSING_EXIT")

    open_kind, open_local = _parse_naive_datetime(_cell(raw, columns.open_column))
    if open_kind == "aware":
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "AWARE_TIMESTAMP")
    if open_kind != "ok" or open_local is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "INVALID_OPEN")
    close_kind, close_local = _parse_naive_datetime(_cell(raw, columns.close_column))
    if close_kind == "aware":
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "AWARE_TIMESTAMP")
    if close_kind != "ok" or close_local is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "INVALID_CLOSE")

    symbol = _canonical_symbol(_cell(raw, columns.symbol_column))
    if symbol is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "BLANK_SYMBOL")
    side = _canonical_side(_cell(raw, columns.side_column))
    if side is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "INVALID_SIDE")
    entry = _parse_price(_cell(raw, columns.entry_column))
    if entry is None:
        raw_entry = _cell(raw, columns.entry_column)
        reason = "INVALID_ENTRY"
        try:
            number = float(str(raw_entry).strip())
            if math.isfinite(number) and number <= 0:
                reason = "NON_POSITIVE_ENTRY"
        except (TypeError, ValueError):
            reason = "INVALID_ENTRY"
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, reason)
    exit_price = _parse_price(_cell(raw, columns.exit_column))
    if exit_price is None:
        raw_exit = _cell(raw, columns.exit_column)
        reason = "INVALID_EXIT"
        try:
            number = float(str(raw_exit).strip())
            if math.isfinite(number) and number <= 0:
                reason = "NON_POSITIVE_EXIT"
        except (TypeError, ValueError):
            reason = "INVALID_EXIT"
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, reason)

    duration_value = None
    if columns.duration_column is not None:
        duration_state, duration_value = _parse_duration(_cell(raw, columns.duration_column))
        if duration_state == "invalid":
            return _failed(ticket, STATUS_INVALID_TRADE_DATA, "INVALID_DURATION")

    open_status, open_reason, open_utc = _normalise_leg(open_local, time_basis)
    if open_status is not None:
        return _failed(ticket, open_status, open_reason)
    close_status, close_reason, close_utc = _normalise_leg(close_local, time_basis)
    if close_status is not None:
        return _failed(ticket, close_status, close_reason)
    if open_utc is None or close_utc is None:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "INVALID_TIMESTAMP")
    if close_utc < open_utc:
        return _failed(ticket, STATUS_INVALID_TRADE_DATA, "CLOSE_BEFORE_OPEN")

    if duration_value is not None:
        normalised = (close_utc - open_utc).total_seconds()
        if abs(normalised - duration_value) > DURATION_TOLERANCE_SECONDS:
            return _failed(ticket, STATUS_TIME_BASIS_INCONSISTENT, "DURATION_MISMATCH")

    symbol_bars = market_index.get(symbol)
    if symbol_bars is None:
        return _failed(ticket, STATUS_NO_MARKET_DATA, "NO_MARKET_DATA")

    start_minute = _floor_minute_utc(open_utc)
    end_minute = _floor_minute_utc(close_utc)
    elapsed_minutes = (end_minute - start_minute).total_seconds() / 60
    expected_count = int(elapsed_minutes) + 1
    if elapsed_minutes > MAX_TRADE_WINDOW_MINUTES:
        return _failed(
            ticket,
            STATUS_INCOMPLETE_COVERAGE,
            REASON_WINDOW_EXCEEDS_LIMIT,
            bars_expected=expected_count,
        )

    start_ts = pd.Timestamp(start_minute)
    end_ts = pd.Timestamp(end_minute)
    minimum = symbol_bars["minimum"]
    maximum = symbol_bars["maximum"]
    if start_ts < minimum or end_ts > maximum:
        return _failed(
            ticket,
            STATUS_INCOMPLETE_COVERAGE,
            REASON_OUTSIDE_FILE_RANGE,
            bars_expected=expected_count,
        )
    frame = symbol_bars["frame"]
    has_start = start_ts in frame.index
    has_end = end_ts in frame.index
    window = frame.loc[start_ts:end_ts]
    bars_used = int(len(window))
    if not has_start or not has_end:
        return _failed(
            ticket,
            STATUS_INCOMPLETE_COVERAGE,
            REASON_BOUNDARY_BAR_MISSING,
            bars_used=bars_used,
            bars_expected=expected_count,
        )
    if bars_used < expected_count:
        return _failed(
            ticket,
            STATUS_INCOMPLETE_COVERAGE,
            REASON_INTERIOR_GAP,
            bars_used=bars_used,
            bars_expected=expected_count,
        )

    highs = [float(value) for value in window["High"]]
    lows = [float(value) for value in window["Low"]]
    interval_high = max(highs)
    interval_low = min(lows)
    if side == "buy":
        mfe = max(0.0, interval_high - entry)
        mae = min(0.0, interval_low - entry)
        realised = exit_price - entry
    else:
        mfe = max(0.0, entry - interval_low)
        mae = min(0.0, entry - interval_high)
        realised = entry - exit_price

    if (
        interval_high + PRICE_COMPARISON_EPSILON < interval_low
        or mfe < -PRICE_COMPARISON_EPSILON
        or mae > PRICE_COMPARISON_EPSILON
    ):
        return _failed(
            ticket,
            STATUS_INVARIANT_VIOLATION,
            REASON_INVARIANT_FAILED,
            bars_used=bars_used,
            bars_expected=expected_count,
        )

    first = window.iloc[0]
    last = window.iloc[-1]
    first_high = float(first["High"])
    first_low = float(first["Low"])
    last_high = float(last["High"])
    last_low = float(last["Low"])
    flags = {
        "high_from_boundary_bar": (
            abs(interval_high - first_high) <= PRICE_COMPARISON_EPSILON
            or abs(interval_high - last_high) <= PRICE_COMPARISON_EPSILON
        ),
        "low_from_boundary_bar": (
            abs(interval_low - first_low) <= PRICE_COMPARISON_EPSILON
            or abs(interval_low - last_low) <= PRICE_COMPARISON_EPSILON
        ),
        "entry_outside_first_bar_range": _outside_range(entry, first_low, first_high),
        "exit_outside_last_bar_range": _outside_range(exit_price, last_low, last_high),
        "realised_outside_interval": (
            realised < mae - PRICE_COMPARISON_EPSILON
            or realised > mfe + PRICE_COMPARISON_EPSILON
        ),
    }
    return _evidence_payload(
        ticket,
        STATUS_COMPUTED,
        None,
        interval_high=_json_float(interval_high),
        interval_low=_json_float(interval_low),
        mfe=_json_float(mfe),
        mae=_json_float(mae),
        bars_used=bars_used,
        bars_expected=expected_count,
        flags=flags,
    )


def compute_excursion_evidence(
    journal: pd.DataFrame,
    market: pd.DataFrame | None,
    time_basis: TimeBasis | None,
) -> ExcursionRunResult:
    columns = resolve_journal_columns(journal)
    join_reason, tickets = _join_tickets(journal, columns)
    fingerprint = None if join_reason is not None else compute_journal_fingerprint(journal)
    if join_reason is not None:
        return _empty_result(
            RUN_STATUS_JOIN_KEY_UNAVAILABLE,
            join_reason,
            None,
        )
    if time_basis is None:
        return _empty_result(
            RUN_STATUS_TIME_BASIS_NOT_DECLARED,
            "TIME_BASIS_NOT_DECLARED",
            fingerprint,
        )
    if tickets is None:
        return _empty_result(RUN_STATUS_JOIN_KEY_UNAVAILABLE, "INVALID_TICKET", None)

    market_index = _index_market(market)
    evidence: dict[str, dict[str, Any]] = {}
    for ticket, (_, raw) in zip(tickets, journal.iterrows(), strict=True):
        evidence[ticket] = _compute_trade(
            ticket,
            raw,
            columns,
            time_basis,
            market_index,
        )
    status_counts, reason_counts = _counts(evidence)
    return ExcursionRunResult(
        run_status=RUN_STATUS_OK,
        reason_code=None,
        journal_fingerprint=fingerprint,
        evidence=MappingProxyType(evidence),
        status_counts=status_counts,
        reason_counts=reason_counts,
    )


def expected_window(open_utc: datetime, close_utc: datetime) -> tuple[datetime, datetime, int]:
    start_minute = _floor_minute_utc(open_utc)
    end_minute = _floor_minute_utc(close_utc)
    elapsed_minutes = int((end_minute - start_minute).total_seconds() / 60)
    return start_minute, end_minute, elapsed_minutes + 1
