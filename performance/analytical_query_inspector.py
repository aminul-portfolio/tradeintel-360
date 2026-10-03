from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from performance.analytical_queries import (
    AnalyticalInputContext,
    AnalyticalQueryResult,
    execute_analytical_query,
)
from performance.analytics import apply_analysis

TEMPLATE_NAME = "performance/analytical_query_inspector.html"

STATE_READY = "READY"
STATE_NO_ACTIVE_JOURNAL = "NO_ACTIVE_JOURNAL"
STATE_JOURNAL_INVALID = "JOURNAL_INVALID"
STATE_REQUEST_INVALID = "REQUEST_INVALID"
STATE_RESULT_AVAILABLE = "RESULT_AVAILABLE"
STATE_WORKFLOW_ERROR = "WORKFLOW_ERROR"

QUERY_CATALOGUE = (
    ("TRADE_COUNT", "Trade count"),
    ("TRADE_COUNT_BY_DAY", "Trade count by day"),
    ("BUSIEST_TRADING_DAY", "Busiest trading day"),
    ("PROFIT_BY_DAY", "Profit by day"),
    ("TRADE_COUNT_BY_SYMBOL", "Trade count by symbol"),
    ("MOST_TRADED_SYMBOL", "Most traded symbol"),
    ("PROFIT_BY_SYMBOL", "Profit by symbol"),
    ("MONTHLY_PROFIT", "Monthly profit"),
)

ROW_IDENTITY_PREFIX = "__ti360_analytical_row_pos__"

WARNING_COPY = {
    "DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION": (
        "Dates are grouped as recorded in the journal; no timezone conversion was applied."
    ),
    "PROFIT_EXCLUDES_COMMISSION_AND_SWAP": (
        "Profit uses the recorded Profit column only; Commission and Swap are not added."
    ),
    "SYMBOL_VARIANTS_DETECTED": (
        "Distinct symbol spellings/casing/spacing were kept as separate groups."
    ),
    "DUPLICATE_ROWS_DETECTED": "Duplicate journal rows were retained and counted.",
}

STATUS_COPY = {
    "NO_DATA": "No rows remained after the current symbol filter.",
    "INVALID_INPUT": (
        "The selected query could not be computed because required journal values were missing or unparseable."
    ),
    "REQUIRED_COLUMN_MISSING": "The selected query requires a journal column that is not present.",
    "UNSUPPORTED_QUERY": "The supplied query ID is not in the fixed catalogue.",
    "AMBIGUOUS_TIME_BASIS": (
        "Date values do not share one recorded time basis, so the date query was not computed."
    ),
}


def _source_basename(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return ""
    normalised = value.replace("\\", "/")
    return Path(normalised).name


def _row_identity_column(columns: object) -> str:
    existing = {str(name) for name in columns}
    if ROW_IDENTITY_PREFIX not in existing:
        return ROW_IDENTITY_PREFIX
    index = 2
    while True:
        candidate = f"{ROW_IDENTITY_PREFIX}{index}__"
        if candidate not in existing:
            return candidate
        index += 1


def _load_session_journal(cleaned_data: object) -> tuple[pd.DataFrame | None, str | None]:
    if cleaned_data is None:
        return None, STATE_NO_ACTIVE_JOURNAL
    if not isinstance(cleaned_data, str):
        return None, STATE_JOURNAL_INVALID
    if cleaned_data.strip() == "":
        return None, STATE_NO_ACTIVE_JOURNAL
    try:
        frame = pd.read_json(StringIO(cleaned_data), orient="split")
    except Exception:
        return None, STATE_JOURNAL_INVALID
    if frame is None:
        return None, STATE_JOURNAL_INVALID
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame, None


def _symbol_filter_payload(symbol: str) -> dict[str, str]:
    if symbol == "":
        return {}
    return {"symbol": symbol}


def _select_raw_rows(raw_df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, object]:
    working = raw_df.copy(deep=True)
    identity = _row_identity_column(working.columns)
    working[identity] = list(range(len(working)))
    filtered_working, analysis_context = apply_analysis(working, _symbol_filter_payload(symbol))
    if identity in filtered_working.columns and len(filtered_working) > 0:
        positions = [int(value) for value in filtered_working[identity].tolist()]
        selected = raw_df.iloc[positions].copy(deep=True)
    else:
        selected = raw_df.iloc[[]].copy(deep=True)
    return selected, analysis_context


def _warning_display(result: AnalyticalQueryResult) -> tuple[dict[str, object], ...]:
    counts = {code: count for code, count in result.warning_counts}
    rows: list[dict[str, object]] = []
    for warning in result.warnings:
        code = warning.value if hasattr(warning, "value") else str(warning)
        rows.append(
            {
                "code": code,
                "explanation": WARNING_COPY.get(code, ""),
                "count": counts.get(code),
            }
        )
    return tuple(rows)


def _hash_short(result: AnalyticalQueryResult | None) -> str:
    if result is None or result.provenance.input_sha256 is None:
        return ""
    return result.provenance.input_sha256[:12]


def _base_context(form_query_id: str, form_symbol: str) -> dict[str, object]:
    return {
        "inspector_state": STATE_READY,
        "inspector_message": "",
        "query_catalogue": QUERY_CATALOGUE,
        "form_query_id": form_query_id,
        "form_symbol": form_symbol,
        "has_active_journal": False,
        "source_filename": "",
        "source_row_count": None,
        "filtered_row_count": None,
        "selected_query_id": form_query_id,
        "active_symbol_filter": "",
        "analytical_result": None,
        "input_hash_short": "",
        "warning_display": (),
        "status_explanation": "",
    }


def _accepted_inputs(request: HttpRequest) -> tuple[str | None, str | None, bool]:
    query_ids = request.GET.getlist("query_id")
    symbols = request.GET.getlist("symbol")
    if len(query_ids) > 1 or len(symbols) > 1:
        return None, None, False
    query_id = query_ids[0] if query_ids else ""
    symbol = symbols[0] if symbols else ""
    return query_id, symbol, True


@login_required
@require_GET
def analytical_query_inspector(request: HttpRequest) -> HttpResponse:
    query_id_raw, symbol_raw, accepted = _accepted_inputs(request)
    form_query_id = query_id_raw or ""
    form_symbol = symbol_raw or ""
    context = _base_context(form_query_id, form_symbol)
    context["source_filename"] = _source_basename(request.session.get("last_uploaded_file"))

    if not accepted:
        context["inspector_state"] = STATE_REQUEST_INVALID
        context["inspector_message"] = (
            "The request contained a duplicated accepted parameter, so no analytical query was run."
        )
        return render(request, TEMPLATE_NAME, context)

    raw_df, journal_state = _load_session_journal(request.session.get("cleaned_data"))
    if journal_state == STATE_NO_ACTIVE_JOURNAL:
        context["inspector_state"] = STATE_NO_ACTIVE_JOURNAL
        context["inspector_message"] = (
            "Load or upload an active journal before running an analytical query."
        )
        return render(request, TEMPLATE_NAME, context)
    if journal_state == STATE_JOURNAL_INVALID:
        context["inspector_state"] = STATE_JOURNAL_INVALID
        context["inspector_message"] = (
            "The active journal could not be read, so no analytical query was run."
        )
        return render(request, TEMPLATE_NAME, context)

    assert raw_df is not None
    context["has_active_journal"] = True
    context["source_row_count"] = int(len(raw_df))

    query_id = form_query_id
    if query_id == "":
        context["inspector_state"] = STATE_READY
        return render(request, TEMPLATE_NAME, context)

    try:
        selected, analysis_context = _select_raw_rows(raw_df, form_symbol)
        input_context = AnalyticalInputContext(
            source_row_count=int(len(raw_df)),
            filtered_row_count=int(len(selected)),
            filters=analysis_context.active_filters,
        )
        result = execute_analytical_query(selected, query_id, input_context)
    except Exception:
        context["inspector_state"] = STATE_WORKFLOW_ERROR
        context["inspector_message"] = (
            "The analytical workflow could not complete this request."
        )
        return render(request, TEMPLATE_NAME, context)

    context["inspector_state"] = STATE_RESULT_AVAILABLE
    context["analytical_result"] = result
    context["filtered_row_count"] = result.filtered_row_count
    context["source_row_count"] = result.source_row_count
    context["selected_query_id"] = result.query_id
    active_symbol = ""
    for key, value in result.filters:
        if key == "symbol":
            active_symbol = value
            break
    context["active_symbol_filter"] = active_symbol
    context["input_hash_short"] = _hash_short(result)
    context["warning_display"] = _warning_display(result)
    status = result.status.value if hasattr(result.status, "value") else str(result.status)
    context["status_explanation"] = STATUS_COPY.get(status, "")
    return render(request, TEMPLATE_NAME, context)
