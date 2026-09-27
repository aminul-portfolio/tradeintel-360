from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

MAX_MARKET_DATA_BYTES = 16 * 1024 * 1024
MAX_MARKET_DATA_ROWS = 200_000
MAX_SYMBOL_LENGTH = 64
MIN_MARKET_DATA_ROWS = 2
MARKET_DATA_RESOLUTION = "M1"
EVIDENCE_CLASS = "BROKER_BAR_SOURCE"
EVIDENCE_PRECISION = "M1_BAR_APPROXIMATE"
PRICE_BASIS = "UNSPECIFIED_CTRADER_HISTORICAL_BAR"
BAR_TIMESTAMP_SEMANTIC = "BAR_OPEN_TIME"
EXCURSION_CONTRACT_VERSION = "EXCURSION_CONTRACT_V1"

REQUIRED_COLUMNS = (
    "TimestampUTC",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
)

_TIMESTAMP_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?P<frac>\.\d+)?"
    r"(?P<offset>Z|\+00:00)$"
)
_NAIVE_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?$"
)
_OFFSET_TIMESTAMP_RE = re.compile(
    r"(?P<offset>[+-]\d{2}:\d{2}|Z)$"
)


class MarketDataValidationError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    coverage_start: str
    coverage_end: str

    def as_json(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "coverage_start": self.coverage_start,
            "coverage_end": self.coverage_end,
        }


@dataclass(frozen=True)
class MarketDataProvenance:
    source_basename: str
    market_file_sha256: str
    file_byte_size: int
    declared_source: str
    declared_export_method: str
    bar_timestamp_semantic: str
    resolution: str
    evidence_class: str
    precision: str
    price_basis: str
    contract_version: str
    source_row_count: int
    valid_row_count: int
    symbols: tuple[str, ...]
    coverage: tuple[SymbolCoverage, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "source_basename": self.source_basename,
            "market_file_sha256": self.market_file_sha256,
            "file_byte_size": self.file_byte_size,
            "declared_source": self.declared_source,
            "declared_export_method": self.declared_export_method,
            "bar_timestamp_semantic": self.bar_timestamp_semantic,
            "resolution": self.resolution,
            "evidence_class": self.evidence_class,
            "precision": self.precision,
            "price_basis": self.price_basis,
            "contract_version": self.contract_version,
            "source_row_count": self.source_row_count,
            "valid_row_count": self.valid_row_count,
            "symbols": list(self.symbols),
            "coverage": [item.as_json() for item in self.coverage],
        }


@dataclass(frozen=True)
class MarketDataParseResult:
    dataframe: pd.DataFrame
    provenance: MarketDataProvenance


def _safe_basename(filename: str) -> str:
    normalised = str(filename).replace("\\", "/")
    return normalised.rsplit("/", 1)[-1].strip()


def _raise(reason: str, message: str = "") -> None:
    raise MarketDataValidationError(reason, message)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp_utc(value: Any) -> datetime:
    if value is None:
        _raise("INVALID_TIMESTAMP", "TimestampUTC is missing")
    text = str(value).strip()
    match = _TIMESTAMP_RE.fullmatch(text)
    if match is None:
        offset_match = _OFFSET_TIMESTAMP_RE.search(text)
        if offset_match and offset_match.group("offset") not in {"Z", "+00:00"}:
            _raise("NON_UTC_TIMESTAMP", "TimestampUTC must be UTC")
        if _NAIVE_TIMESTAMP_RE.fullmatch(text):
            _raise("INVALID_TIMESTAMP", "TimestampUTC must include a UTC offset")
        _raise("INVALID_TIMESTAMP", "TimestampUTC is not a supported UTC value")

    fraction = match.group("frac")
    if fraction and any(digit != "0" for digit in fraction[1:]):
        _raise(
            "TIMESTAMP_NOT_MINUTE_ALIGNED",
            "TimestampUTC must be minute aligned",
        )
    iso_text = f"{match.group('head')}+00:00"
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        _raise("INVALID_TIMESTAMP", "TimestampUTC could not be parsed")
    if parsed.tzinfo is None:
        _raise("INVALID_TIMESTAMP", "TimestampUTC must be UTC-aware")
    utc_value = parsed.astimezone(timezone.utc)
    if utc_value.second != 0 or utc_value.microsecond != 0:
        _raise(
            "TIMESTAMP_NOT_MINUTE_ALIGNED",
            "TimestampUTC must be minute aligned",
        )
    return utc_value


def _parse_price(value: Any, field: str) -> float:
    if value is None:
        _raise("INVALID_OHLC", f"{field} is missing")
    text = str(value).strip()
    if text == "":
        _raise("INVALID_OHLC", f"{field} is missing")
    try:
        number = float(text)
    except ValueError:
        _raise("INVALID_OHLC", f"{field} is not numeric")
    if not math.isfinite(number):
        _raise("INVALID_OHLC", f"{field} must be finite")
    if number <= 0:
        _raise("NON_POSITIVE_OHLC", f"{field} must be greater than zero")
    return number


def _validate_ohlc(open_: float, high: float, low: float, close: float) -> None:
    if not (
        high >= open_
        and high >= close
        and high >= low
        and low <= open_
        and low <= close
        and low <= high
    ):
        _raise("IMPOSSIBLE_OHLC", "OHLC values violate High/Low invariants")


def _read_csv_rows(text: str) -> list[dict[str, str]]:
    stream = io.StringIO(text)
    try:
        reader = csv.reader(stream, strict=True)
        try:
            header_row = next(reader)
        except StopIteration:
            _raise("TOO_FEW_ROWS", "CSV contains no header")
        columns = [str(name).strip() for name in header_row]
        rows: list[dict[str, str]] = []
        for index, values in enumerate(reader):
            if index >= MAX_MARKET_DATA_ROWS + 1:
                break
            row = {
                column: (values[position] if position < len(values) else "")
                for position, column in enumerate(columns)
            }
            rows.append(row)
    except csv.Error as exc:
        _raise("CSV_PARSE_ERROR", str(exc))
    if len(rows) > MAX_MARKET_DATA_ROWS:
        _raise(
            "ROW_LIMIT_EXCEEDED",
            "CSV exceeds the maximum number of data rows",
        )
    return rows


def _validate_m1_resolution(rows: list[dict[str, Any]]) -> None:
    by_symbol: dict[str, list[datetime]] = {}
    for row in rows:
        by_symbol.setdefault(row["Symbol"], []).append(row["TimestampUTC"])
    for timestamps in by_symbol.values():
        ordered = sorted(timestamps)
        if len(ordered) < 2:
            _raise(
                "M1_RESOLUTION_UNCONFIRMED",
                "A symbol must include at least two M1 bars",
            )
        deltas = [
            (ordered[index] - ordered[index - 1]).total_seconds()
            for index in range(1, len(ordered))
        ]
        if min(deltas) != 60:
            _raise(
                "M1_RESOLUTION_UNCONFIRMED",
                "Adjacent bars must establish a 60-second M1 resolution",
            )


def parse_market_data_bytes(
    raw_bytes: bytes,
    filename: str,
    *,
    declared_source: str,
    declared_export_method: str,
) -> MarketDataParseResult:
    if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        _raise("INVALID_INPUT", "raw_bytes must be bytes-like")
    payload = bytes(raw_bytes)
    if len(payload) == 0:
        _raise("EMPTY_FILE", "Market data file is empty")
    if len(payload) > MAX_MARKET_DATA_BYTES:
        _raise("FILE_TOO_LARGE", "Market data file exceeds the byte limit")

    digest = hashlib.sha256(payload).hexdigest()
    basename = _safe_basename(filename)
    if not basename.lower().endswith(".csv"):
        _raise("NOT_CSV", "Market data filename must use a .csv extension")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        _raise("DECODE_ERROR", "Market data must be UTF-8 or UTF-8-BOM")

    raw_rows = _read_csv_rows(text)
    if len(raw_rows) < MIN_MARKET_DATA_ROWS:
        _raise("TOO_FEW_ROWS", "Market data must contain at least two data rows")

    available = set(raw_rows[0].keys())
    missing = [name for name in REQUIRED_COLUMNS if name not in available]
    if missing:
        _raise(
            "MISSING_REQUIRED_COLUMNS",
            "Market data is missing required columns",
        )

    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, datetime]] = set()
    for raw_row in raw_rows:
        symbol = str(raw_row.get("Symbol", "")).strip()
        if symbol == "":
            _raise("EMPTY_SYMBOL", "Symbol must be non-empty after trimming")
        if len(symbol) > MAX_SYMBOL_LENGTH:
            _raise("SYMBOL_TOO_LONG", "Symbol exceeds the maximum length")
        timestamp = _parse_timestamp_utc(raw_row.get("TimestampUTC"))
        open_ = _parse_price(raw_row.get("Open"), "Open")
        high = _parse_price(raw_row.get("High"), "High")
        low = _parse_price(raw_row.get("Low"), "Low")
        close = _parse_price(raw_row.get("Close"), "Close")
        _validate_ohlc(open_, high, low, close)
        key = (symbol, timestamp)
        if key in seen:
            _raise(
                "DUPLICATE_BAR",
                "Duplicate Symbol and TimestampUTC combination",
            )
        seen.add(key)
        validated.append(
            {
                "TimestampUTC": timestamp,
                "Symbol": symbol,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
            }
        )

    _validate_m1_resolution(validated)
    validated.sort(key=lambda row: (row["Symbol"], row["TimestampUTC"]))

    frame = pd.DataFrame(validated, columns=list(REQUIRED_COLUMNS))
    frame["TimestampUTC"] = pd.to_datetime(frame["TimestampUTC"], utc=True)
    symbols = tuple(sorted(frame["Symbol"].unique().tolist()))
    coverage = []
    for symbol in symbols:
        symbol_times = frame.loc[
            frame["Symbol"] == symbol,
            "TimestampUTC",
        ]
        coverage.append(
            SymbolCoverage(
                symbol=symbol,
                coverage_start=_format_utc(symbol_times.min().to_pydatetime()),
                coverage_end=_format_utc(symbol_times.max().to_pydatetime()),
            )
        )

    provenance = MarketDataProvenance(
        source_basename=basename,
        market_file_sha256=digest,
        file_byte_size=len(payload),
        declared_source=str(declared_source),
        declared_export_method=str(declared_export_method),
        bar_timestamp_semantic=BAR_TIMESTAMP_SEMANTIC,
        resolution=MARKET_DATA_RESOLUTION,
        evidence_class=EVIDENCE_CLASS,
        precision=EVIDENCE_PRECISION,
        price_basis=PRICE_BASIS,
        contract_version=EXCURSION_CONTRACT_VERSION,
        source_row_count=len(validated),
        valid_row_count=len(validated),
        symbols=symbols,
        coverage=tuple(coverage),
    )
    return MarketDataParseResult(dataframe=frame, provenance=provenance)
