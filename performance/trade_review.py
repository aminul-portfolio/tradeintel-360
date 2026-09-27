from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

import pandas as pd

from .excursion import canonical_ticket

PIPS_MOVEMENT_TOLERANCE = 0.1
TRADE_PAGE_SIZE = 10

MOVEMENT_SOURCE_BROKER = "broker_pips"
MOVEMENT_SOURCE_PRICE_MOVE = "price_move"
MOVEMENT_SOURCE_UNAVAILABLE = "unavailable"

MOVEMENT_STATUS_OK = "ok"
MOVEMENT_STATUS_DISAGREEMENT = "disagreement"
MOVEMENT_STATUS_FALLBACK = "fallback"
MOVEMENT_STATUS_UNAVAILABLE = "unavailable"

LABEL_PIPS = "Pips"
LABEL_PRICE_MOVE = "Price Move"

PIPS_COLUMNS = ("Pips",)
ENTRY_COLUMNS = ("Entry", "Price", "Open Price", "Entry Price")
EXIT_COLUMNS = ("Exit", "Price.1", "Close Price", "Exit Price")
SIDE_COLUMNS = ("Type", "Side")

SORT_WHITELIST = {
    "ticket": {
        "columns": ("Ticket",),
        "kind": "numeric",
    },
    "open_time": {
        "columns": ("Open Time", "Open", "Date"),
        "kind": "date",
    },
    "close_time": {
        "columns": ("Close Time", "Close"),
        "kind": "date",
    },
    "symbol": {
        "columns": ("Symbol",),
        "kind": "text",
    },
    "type": {
        "columns": ("Type", "Side"),
        "kind": "text",
    },
    "volume": {
        "columns": ("Volume", "Size"),
        "kind": "numeric",
    },
    "profit": {
        "columns": ("Profit",),
        "kind": "numeric",
    },
    "pips": {
        "columns": ("Pips",),
        "kind": "numeric",
    },
    "movement": {
        "columns": ("_movement_value",),
        "kind": "numeric",
    },
    "interval_high": {
        "columns": ("_excursion_interval_high",),
        "kind": "numeric",
    },
    "interval_low": {
        "columns": ("_excursion_interval_low",),
        "kind": "numeric",
    },
    "mfe": {
        "columns": ("_excursion_mfe",),
        "kind": "numeric",
    },
    "mae": {
        "columns": ("_excursion_mae",),
        "kind": "numeric",
    },
}

DISPLAY_COLUMNS = (
    ("ticket", "Ticket", ("Ticket",)),
    ("open_time", "Open Time", ("Open Time", "Open", "Date")),
    ("close_time", "Close Time", ("Close Time", "Close")),
    ("symbol", "Symbol", ("Symbol",)),
    ("type", "Type", ("Type", "Side")),
    ("volume", "Volume", ("Volume", "Size")),
    ("entry", "Entry", ("Entry", "Price")),
    ("exit", "Exit", ("Exit", "Price.1")),
    ("sl", "SL", ("SL",)),
    ("tp", "TP", ("TP",)),
    ("profit", "Profit", ("Profit",)),
    ("commission", "Commission", ("Commission", "Commissions")),
    ("swap", "Swap", ("Swap",)),
)

EXCURSION_INTERNAL_COLUMNS = (
    "_excursion_interval_high",
    "_excursion_interval_low",
    "_excursion_mfe",
    "_excursion_mae",
    "_excursion_status",
    "_excursion_reason",
    "_excursion_high_from_boundary_bar",
    "_excursion_low_from_boundary_bar",
    "_excursion_entry_outside_first_bar_range",
    "_excursion_exit_outside_last_bar_range",
    "_excursion_realised_outside_interval",
)

EVIDENCE_DISPLAY_COLUMNS = (
    (
        "interval_high",
        "Approx. Window High",
        ("_excursion_interval_high",),
        "evidence_numeric",
    ),
    (
        "interval_low",
        "Approx. Window Low",
        ("_excursion_interval_low",),
        "evidence_numeric",
    ),
    (
        "mfe",
        "Approx. MFE (price pts)",
        ("_excursion_mfe",),
        "evidence_numeric",
    ),
    (
        "mae",
        "Approx. MAE (price pts)",
        ("_excursion_mae",),
        "evidence_numeric",
    ),
    (
        "bar_evidence",
        "Bar Evidence",
        ("_excursion_status",),
        "evidence_status",
    ),
)

BAR_EVIDENCE_LABELS = {
    "COMPUTED": "Computed",
    "INVALID_TRADE_DATA": "Invalid trade data",
    "TIMEZONE_AMBIGUOUS": "Time unresolved",
    "TIME_BASIS_INCONSISTENT": "Time basis inconsistent",
    "NO_MARKET_DATA": "No market data",
    "INCOMPLETE_COVERAGE": "Incomplete coverage",
    "INVARIANT_VIOLATION": "Evidence unavailable",
}


def _first_present(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def is_valid_pips(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def classify_side(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"buy", "long"}:
        return "buy"
    if text in {"sell", "short"}:
        return "sell"
    return None


def calculated_movement(side: str | None, entry: Any, exit_price: Any):
    resolved_side = classify_side(side) if side not in {"buy", "sell"} else side
    if resolved_side not in {"buy", "sell"}:
        return None
    try:
        entry_number = float(entry)
        exit_number = float(exit_price)
    except (TypeError, ValueError):
        return None
    if not (
        math.isfinite(entry_number)
        and math.isfinite(exit_number)
    ):
        return None
    if resolved_side == "buy":
        return exit_number - entry_number
    return entry_number - exit_number


def resolve_realised_movement(
    pips: Any,
    side: Any,
    entry: Any,
    exit_price: Any,
) -> dict[str, Any]:
    resolved_side = classify_side(side)
    calculated = calculated_movement(
        resolved_side,
        entry,
        exit_price,
    )
    if is_valid_pips(pips):
        pips_value = float(pips)
        disagreement = (
            calculated is not None
            and abs(pips_value - calculated)
            > PIPS_MOVEMENT_TOLERANCE
        )
        return {
            "value": pips_value,
            "label": LABEL_PIPS,
            "source": MOVEMENT_SOURCE_BROKER,
            "status": (
                MOVEMENT_STATUS_DISAGREEMENT
                if disagreement
                else MOVEMENT_STATUS_OK
            ),
            "disagreement": disagreement,
        }
    if calculated is not None:
        return {
            "value": calculated,
            "label": LABEL_PRICE_MOVE,
            "source": MOVEMENT_SOURCE_PRICE_MOVE,
            "status": MOVEMENT_STATUS_FALLBACK,
            "disagreement": False,
        }
    return {
        "value": None,
        "label": "",
        "source": MOVEMENT_SOURCE_UNAVAILABLE,
        "status": MOVEMENT_STATUS_UNAVAILABLE,
        "disagreement": False,
    }


def enrich_trade_review(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    pips_col = _first_present(frame.columns, PIPS_COLUMNS)
    entry_col = _first_present(frame.columns, ENTRY_COLUMNS)
    exit_col = _first_present(frame.columns, EXIT_COLUMNS)
    side_col = _first_present(frame.columns, SIDE_COLUMNS)

    values = []
    labels = []
    sources = []
    statuses = []
    flags = []

    for _, row in frame.iterrows():
        result = resolve_realised_movement(
            row[pips_col] if pips_col else None,
            row[side_col] if side_col else None,
            row[entry_col] if entry_col else None,
            row[exit_col] if exit_col else None,
        )
        values.append(result["value"])
        labels.append(result["label"])
        sources.append(result["source"])
        statuses.append(result["status"])
        flags.append(result["disagreement"])

    frame["_movement_value"] = values
    frame["_movement_label"] = labels
    frame["_movement_source"] = sources
    frame["_movement_status"] = statuses
    frame["_movement_disagreement"] = flags
    return frame


def normalize_sort_key(value: Any) -> str | None:
    key = str(value or "").strip().lower()
    if key in SORT_WHITELIST:
        return key
    return None


def normalize_sort_direction(
    value: Any,
    default: str = "asc",
) -> str:
    direction = str(value or "").strip().lower()
    if direction in {"asc", "desc"}:
        return direction
    return default


def next_sort_direction(
    current_key: str,
    current_direction: str,
    clicked_key: str,
) -> str:
    if (
        current_key == clicked_key
        and current_direction == "asc"
    ):
        return "desc"
    return "asc"


def _sort_series(series: pd.Series, kind: str) -> pd.Series:
    if kind == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if kind == "date":
        return pd.to_datetime(series, errors="coerce")
    text = series.astype(str)
    missing = series.isna()
    blank = text.str.strip().isin({"", "nan", "none", "nat"})
    text = text.str.lower()
    text = text.mask(missing | blank)
    return text


def sort_trade_review(
    dataframe: pd.DataFrame,
    sort_key: Any,
    sort_direction: Any,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe.copy() if dataframe is not None else pd.DataFrame()

    key = normalize_sort_key(sort_key)
    direction = normalize_sort_direction(sort_direction)
    if key is None:
        return dataframe.copy()

    spec = SORT_WHITELIST[key]
    column = _first_present(dataframe.columns, spec["columns"])
    if column is None:
        return dataframe.copy()

    converted = _sort_series(dataframe[column], spec["kind"])
    sort_frame = pd.DataFrame(
        {
            "_na": converted.isna(),
            "_val": converted,
        },
        index=dataframe.index,
    )
    by = ["_na", "_val"]
    ascending = [True, direction == "asc"]

    if "Ticket" in dataframe.columns:
        sort_frame["_ticket"] = pd.to_numeric(
            dataframe["Ticket"],
            errors="coerce",
        )
        by.append("_ticket")
        ascending.append(True)

    ordered = sort_frame.sort_values(
        by=by,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    )
    return dataframe.loc[ordered.index].copy()


def attach_excursion_evidence(
    dataframe: pd.DataFrame,
    evidence: Any,
) -> pd.DataFrame:
    if dataframe is None:
        return pd.DataFrame()
    frame = dataframe.copy()
    for name in EXCURSION_INTERNAL_COLUMNS:
        frame[name] = None
    mapping = evidence if isinstance(evidence, dict) else {}
    if "Ticket" not in frame.columns:
        return frame
    values = {name: [] for name in EXCURSION_INTERNAL_COLUMNS}
    for raw_ticket in frame["Ticket"]:
        ticket = canonical_ticket(raw_ticket)
        item = mapping.get(ticket) if ticket else None
        if not isinstance(item, dict):
            item = {}
        values["_excursion_interval_high"].append(item.get("interval_high"))
        values["_excursion_interval_low"].append(item.get("interval_low"))
        values["_excursion_mfe"].append(item.get("mfe"))
        values["_excursion_mae"].append(item.get("mae"))
        values["_excursion_status"].append(item.get("status"))
        values["_excursion_reason"].append(item.get("reason_code"))
        values["_excursion_high_from_boundary_bar"].append(
            bool(item.get("high_from_boundary_bar"))
        )
        values["_excursion_low_from_boundary_bar"].append(
            bool(item.get("low_from_boundary_bar"))
        )
        values["_excursion_entry_outside_first_bar_range"].append(
            bool(item.get("entry_outside_first_bar_range"))
        )
        values["_excursion_exit_outside_last_bar_range"].append(
            bool(item.get("exit_outside_last_bar_range"))
        )
        values["_excursion_realised_outside_interval"].append(
            bool(item.get("realised_outside_interval"))
        )
    for name, column_values in values.items():
        frame[name] = column_values
    return frame


def build_trade_review_columns(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    columns = []
    available = set(dataframe.columns) if dataframe is not None else set()
    for key, label, candidates in DISPLAY_COLUMNS:
        source = _first_present(available, candidates)
        if source is None:
            continue
        columns.append(
            {
                "key": key,
                "label": label,
                "sort_key": key if key in SORT_WHITELIST else "",
                "source_column": source,
                "kind": "data",
            }
        )
    columns.append(
        {
            "key": "movement",
            "label": "Realised movement",
            "sort_key": "movement",
            "source_column": "_movement_value",
            "kind": "movement",
        }
    )
    if all(name in available for name in ("_excursion_status", "_excursion_mfe")):
        for key, label, candidates, kind in EVIDENCE_DISPLAY_COLUMNS:
            source = _first_present(available, candidates)
            if source is None:
                continue
            columns.append(
                {
                    "key": key,
                    "label": label,
                    "sort_key": key if key in SORT_WHITELIST else "",
                    "source_column": source,
                    "kind": kind,
                }
            )
    return columns


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_display_value(value: Any) -> Any:
    if _is_missing(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def format_volume_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    try:
        if not math.isfinite(float(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return ""
    if not decimal_value.is_finite():
        return ""
    if decimal_value == 0:
        return "0"
    quantized = decimal_value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    if quantized == 0:
        if decimal_value > 0:
            return "<0.01"
        return ">-0.01"
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _cell_alignment_class(column_key: str) -> str:
    if column_key in {
        "ticket",
        "volume",
        "profit",
        "entry",
        "exit",
        "sl",
        "tp",
        "commission",
        "swap",
        "movement",
        "interval_high",
        "interval_low",
        "mfe",
        "mae",
    }:
        return "trade-review-cell--numeric"
    if column_key in {"open_time", "close_time"}:
        return "trade-review-cell--date"
    return ""


def format_evidence_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number == 0:
        return "0"
    try:
        normalised = Decimal(format(number, ".15g"))
    except (InvalidOperation, ValueError):
        return ""
    if normalised == 0:
        try:
            normalised = Decimal(format(number, ".17g"))
        except (InvalidOperation, ValueError):
            return ""
        if normalised == 0:
            return "0"
    text = format(normalised, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-", "-0"}:
        return "0"
    return text


def format_bar_evidence_status(value: Any) -> str:
    if _is_missing(value):
        return ""
    return BAR_EVIDENCE_LABELS.get(str(value), str(value))


def bar_evidence_notes(raw: Any) -> tuple[str, str]:
    reason = raw.get("_excursion_reason") or ""
    if str(raw.get("_excursion_status") or "") != "COMPUTED":
        return str(reason) if reason and not _is_missing(reason) else "", ""
    notes = []
    if raw.get("_excursion_high_from_boundary_bar") or raw.get(
        "_excursion_low_from_boundary_bar"
    ):
        notes.append("Boundary extreme")
    if raw.get("_excursion_entry_outside_first_bar_range"):
        notes.append("Entry outside first M1 bar range")
    if raw.get("_excursion_exit_outside_last_bar_range"):
        notes.append("Exit outside last M1 bar range")
    if raw.get("_excursion_realised_outside_interval"):
        notes.append("Realised move outside M1 envelope")
    return str(reason) if reason and not _is_missing(reason) else "", "; ".join(notes)


def format_movement_value(value: Any, label: Any = "") -> str:
    if _is_missing(value):
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    if number.is_integer():
        magnitude = str(int(abs(number)))
    else:
        magnitude = f"{abs(number):.2f}".rstrip("0").rstrip(".")
    unit = "pips" if label == LABEL_PIPS else "price pts"
    if number == 0:
        return f"0 {unit}"
    if number > 0:
        return f"+{magnitude} {unit}"
    return f"-{magnitude} {unit}"


def movement_css_class(value: Any) -> str:
    if _is_missing(value):
        return ""
    number = float(value)
    if number > 0:
        return "trade-review-move--pos"
    if number < 0:
        return "trade-review-move--neg"
    return ""


def build_trade_review_rows(
    dataframe: pd.DataFrame,
    columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if dataframe is None or dataframe.empty:
        return []

    rows = []
    for _, raw in dataframe.iterrows():
        cells = []
        for column in columns:
            if column["kind"] == "movement":
                value = raw.get("_movement_value")
                label = raw.get("_movement_label") or ""
                disagreement = bool(raw.get("_movement_disagreement"))
                cells.append(
                    {
                        "value": format_movement_value(value, label),
                        "css_class": " ".join(
                            part
                            for part in (
                                _cell_alignment_class("movement"),
                                movement_css_class(value),
                            )
                            if part
                        ),
                        "note": label,
                        "flag": "Mismatch" if disagreement else "",
                        "title": (
                            "Broker Pips and calculated price move differ by more than 0.1."
                            if disagreement
                            else ""
                        ),
                    }
                )
                continue
            if column["kind"] == "evidence_numeric":
                value = raw.get(column["source_column"])
                cells.append(
                    {
                        "value": format_evidence_value(value),
                        "css_class": _cell_alignment_class(column["key"]),
                        "note": "",
                        "flag": "",
                        "title": "",
                    }
                )
                continue
            if column["kind"] == "evidence_status":
                note, flag = bar_evidence_notes(raw)
                cells.append(
                    {
                        "value": format_bar_evidence_status(
                            raw.get(column["source_column"])
                        ),
                        "css_class": "trade-review-status-cell",
                        "note": note,
                        "flag": flag,
                        "title": "",
                    }
                )
                continue
            source = column["source_column"]
            raw_value = (
                raw[source] if source in raw.index else ""
            )
            if column["key"] == "volume":
                display_value = format_volume_value(raw_value)
            else:
                display_value = format_display_value(raw_value)
            cells.append(
                {
                    "value": display_value,
                    "css_class": " ".join(
                        part
                        for part in (
                            _cell_alignment_class(column["key"]),
                            (
                                movement_css_class(raw_value)
                                if column["key"] == "profit"
                                else ""
                            ),
                        )
                        if part
                    ),
                    "note": "",
                    "flag": "",
                    "title": "",
                }
            )
        rows.append({"cells": cells})
    return rows
