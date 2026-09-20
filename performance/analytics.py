from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

import pandas as pd

DATE_COLUMNS = (
    "Open Time",
    "Open",
    "Date",
)

SMART_TEXT_COLUMNS = (
    "Symbol",
    "Type",
    "Side",
    "Tag",
    "Tags",
    "Notes",
    "Comment",
    "Comments",
    "Strategy",
    "Reason",
)

SMART_NUMERIC_COLUMNS = (
    "Profit",
    "Commission",
    "Swap",
    "Balance",
    "Pips",
    "Size",
    "Volume",
)


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    source_filename: str | None
    source_row_count: int
    filtered_row_count: int
    active_filters: Mapping[str, str]
    is_zero_result: bool


def _clean_param(
    query_params: Mapping[str, Any] | None,
    key: str,
) -> str:
    if query_params is None:
        return ""

    value = query_params.get(key, "")
    if value is None:
        return ""

    return str(value).strip()


def _parse_iso_date(value: str) -> date | None:
    if not value:
        return None

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()
    except ValueError:
        return None


def _basename(source_filename: str | None) -> str | None:
    if not source_filename:
        return None

    normalised = str(source_filename).replace("\\", "/")
    basename = normalised.rsplit("/", 1)[-1].strip()

    return basename or None


def _first_date_column(columns) -> str | None:
    for column in DATE_COLUMNS:
        if column in columns:
            return column

    return None


def _smart_filter_df(
    dataframe: pd.DataFrame,
    query: str,
) -> pd.DataFrame:
    if dataframe.empty or not query:
        return dataframe.copy()

    query = str(query).strip()
    if not query:
        return dataframe.copy()

    mask = pd.Series(
        False,
        index=dataframe.index,
        dtype=bool,
    )

    text_columns = [
        column
        for column in SMART_TEXT_COLUMNS
        if column in dataframe.columns
    ]

    if not text_columns:
        text_columns = list(
            dataframe.select_dtypes(
                include=["object", "string"]
            ).columns
        )

    for column in text_columns:
        mask |= (
            dataframe[column]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False,
                regex=False,
            )
        )

    try:
        numeric_query = float(
            query.replace(",", "")
        )
    except ValueError:
        numeric_query = None

    if numeric_query is not None:
        numeric_columns = [
            column
            for column in SMART_NUMERIC_COLUMNS
            if column in dataframe.columns
        ]

        for column in numeric_columns:
            numeric_values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).round(8)

            mask |= numeric_values.eq(numeric_query)

    for date_column in DATE_COLUMNS:
        if date_column not in dataframe.columns:
            continue

        date_values = pd.to_datetime(
            dataframe[date_column],
            errors="coerce",
        )

        mask |= (
            date_values
            .dt.strftime("%Y-%m-%d")
            .fillna("")
            .str.contains(
                query,
                case=False,
                na=False,
                regex=False,
            )
        )

        mask |= (
            date_values
            .dt.strftime("%Y-%m")
            .fillna("")
            .str.contains(
                query,
                case=False,
                na=False,
                regex=False,
            )
        )

    return dataframe.loc[mask].copy()


def apply_analysis(
    dataframe: pd.DataFrame | None,
    query_params: Mapping[str, Any] | None,
    *,
    source_filename: str | None = None,
) -> tuple[pd.DataFrame, AnalysisContext]:
    if dataframe is None:
        working_df = pd.DataFrame()
    else:
        working_df = dataframe.copy(deep=True)

    source_row_count = len(working_df)
    active_filters: dict[str, str] = {}

    start_date_raw = _clean_param(
        query_params,
        "start_date",
    )
    end_date_raw = _clean_param(
        query_params,
        "end_date",
    )
    symbol = _clean_param(
        query_params,
        "symbol",
    )
    smart_query = _clean_param(
        query_params,
        "q",
    )

    start_date = _parse_iso_date(start_date_raw)
    end_date = _parse_iso_date(end_date_raw)

    date_column = _first_date_column(
        working_df.columns
    )

    date_values = None

    if date_column:
        working_df[date_column] = pd.to_datetime(
            working_df[date_column],
            errors="coerce",
            dayfirst=True,
        )
        date_values = working_df[date_column]

    if date_values is not None and (
            start_date is not None
            or end_date is not None
    ):
        date_mask = pd.Series(
            True,
            index=working_df.index,
            dtype=bool,
        )

        if start_date is not None:
            date_mask &= (
                    date_values.dt.date >= start_date
            )
            active_filters["start_date"] = (
                start_date.isoformat()
            )

        if end_date is not None:
            date_mask &= (
                    date_values.dt.date <= end_date
            )
            active_filters["end_date"] = (
                end_date.isoformat()
            )

        working_df = working_df.loc[
            date_mask
        ].copy()
    if symbol and "Symbol" in working_df.columns:
        symbol_mask = (
            working_df["Symbol"]
            .astype(str)
            .str.contains(
                symbol,
                case=False,
                na=False,
                regex=False,
            )
        )

        working_df = working_df.loc[
            symbol_mask
        ].copy()

        active_filters["symbol"] = symbol

    if smart_query:
        working_df = _smart_filter_df(
            working_df,
            smart_query,
        )

        active_filters["q"] = smart_query

    context = AnalysisContext(
        source_filename=_basename(
            source_filename
        ),
        source_row_count=source_row_count,
        filtered_row_count=len(working_df),
        active_filters=MappingProxyType(
            dict(active_filters)
        ),
        is_zero_result=working_df.empty,
    )

    return working_df, context
