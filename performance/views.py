from io import BytesIO, StringIO
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as opy
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import get_template
from django.utils import timezone
from django.views.decorators.http import require_POST
from xhtml2pdf import pisa

from .analytics import apply_analysis
from .excursion import (
    DURATION_TOLERANCE_SECONDS,
    PRICE_COMPARISON_EPSILON,
    RUN_STATUS_OK,
    STATUS_COMPUTED,
    STATUS_INCOMPLETE_COVERAGE,
    STATUS_INVARIANT_VIOLATION,
    compute_excursion_evidence,
    compute_journal_fingerprint,
)
from .excursion_state import (
    ExcursionStateError,
    clear_excursion_state,
    load_bound_excursion_state,
    store_excursion_state,
)
from .forms import FilterForm, MarketDataUploadForm, TradingFileForm
from .ingestion import IngestionError, clean_ftmo_csv
from .market_data import MarketDataValidationError, parse_market_data_bytes
from .models import TradingFile
from .trade_review import (
    TRADE_PAGE_SIZE,
    attach_excursion_evidence,
    build_trade_review_columns,
    build_trade_review_rows,
    enrich_trade_review,
    next_sort_direction,
    normalize_sort_direction,
    normalize_sort_key,
    sort_trade_review,
)
from .utils import compute_kpis

CLEANED_DATA_SESSION_KEY = "cleaned_data"
# ─────────────────────────────────────────────────────────────────────
# CHART CONFIG
# displayModeBar: False hides the Plotly toolbar — the single biggest
# signal that breaks the premium feel.
# ─────────────────────────────────────────────────────────────────────
CHART_CONFIG = {
    "displayModeBar": False,
    "responsive":     True,
    "staticPlot":     False,
}

# ─────────────────────────────────────────────────────────────────────
# DESIGN TOKENS — keep in sync with tradeintel_design_system.css
# ─────────────────────────────────────────────────────────────────────
T = {
    "bg_card":    "#121827",
    "bg_surface": "#0e1420",
    "bg_hover":   "#1a2236",
    "text_1":     "#f0f4ff",
    "text_2":     "#8896b3",
    "text_3":     "#4a5470",
    "accent":     "#00e5b4",
    "green":      "#22c55e",
    "red":        "#f43f5e",
    "amber":      "#f59e0b",
    "border":     "rgba(255,255,255,0.07)",
    "font":       "IBM Plex Mono, monospace",
}

PIE_COLORS = [
    "#00e5b4", "#2a7fff", "#f59e0b", "#f43f5e",
    "#a78bfa", "#34d399", "#fb923c", "#60a5fa",
    "#e879f9", "#facc15", "#4ade80", "#f87171",
]
PIE_OTHER_COLOR = "#4a5470"
PIE_THRESHOLD   = 2.5


# ─────────────────────────────────────────────────────────────────────
# DARK THEME HELPER
# Pass title="" (default) when the template's .chart-shell__title
# already labels the chart — avoids the duplicate title issue.
# ─────────────────────────────────────────────────────────────────────
def _apply_dark_theme(fig, title: str = "") -> None:
    fig.update_layout(
        title=dict(
            text=title.upper() if title else "",
            x=0.0,
            xanchor="left",
            pad=dict(l=4),
            font=dict(family=T["font"], size=9, color=T["text_3"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=T["font"], color=T["text_2"], size=10),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.07)",
            tickcolor="rgba(255,255,255,0.07)",
            tickfont=dict(size=9, color=T["text_3"]),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.07)",
            tickcolor="rgba(255,255,255,0.07)",
            tickfont=dict(size=9, color=T["text_3"]),
            zeroline=False,
        ),
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=T["border"],
            borderwidth=1,
            font=dict(size=9, color=T["text_2"]),
        ),
        hoverlabel=dict(
            bgcolor=T["bg_hover"],
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(family=T["font"], size=11, color=T["text_1"]),
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────
def _query_without(request, *keys):
    q = request.GET.copy()
    for key in keys:
        q.pop(key, None)
    return q.urlencode()


def _trade_sort_query(request, sort_key, sort_dir):
    query = request.GET.copy()
    query.pop("trade_page", None)
    query["trade_sort"] = sort_key
    query["trade_dir"] = sort_dir
    return query.urlencode()


def _first_present(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def _read_session_df(cleaned_data):
    if not cleaned_data:
        return None
    try:
        df = pd.read_json(StringIO(cleaned_data), orient="split")
    except Exception:
        return None
    if df is None:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _safe_compute_kpis(df):
    if df is None or df.empty or "Profit" not in df.columns:
        return {}
    working_df = df.copy()
    working_df["Profit"] = pd.to_numeric(working_df["Profit"], errors="coerce")
    if working_df["Profit"].notna().sum() == 0:
        return {}
    try:
        return compute_kpis(working_df)
    except Exception:
        return {}


def _bind_dashboard_excursion_state(request, full_df):
    if full_df is None or full_df.empty:
        clear_excursion_state(request.session)
        return None
    fingerprint = compute_journal_fingerprint(full_df)
    if not fingerprint:
        clear_excursion_state(request.session)
        return None
    return load_bound_excursion_state(
        request.session,
        current_journal_fingerprint=fingerprint,
    )


def _excursion_summary(state):
    if not state:
        return None
    counts = state.get("status_counts") or {}
    matched = (
        int(counts.get(STATUS_COMPUTED, 0))
        + int(counts.get(STATUS_INCOMPLETE_COVERAGE, 0))
        + int(counts.get(STATUS_INVARIANT_VIOLATION, 0))
    )
    return {
        "matched_trade_count": matched,
        "status_counts": counts,
        "reason_counts": state.get("reason_counts") or {},
        "journal_fingerprint": state.get("journal_fingerprint"),
        "time_basis": state.get("time_basis") or {},
        "market_provenance": state.get("market_provenance") or {},
    }


def _is_trade_like_df(df):
    if df is None or df.empty:
        return False
    markers = ["Open", "Open Time", "Date", "Symbol", "Type", "Side", "Profit"]
    return any(col in df.columns for col in markers)

# ─────────────────────────────────────────────────────────────────────
# PIE CHART BUILDER
# ─────────────────────────────────────────────────────────────────────
def _build_pie(title: str, labels: list, values: list) -> str:
    total = sum(values) or 1
    pcts  = [v / total * 100 for v in values]

    main_labels, main_values, main_pcts = [], [], []
    other_total = 0.0

    for lbl, val, p in zip(labels, values, pcts):
        if p >= PIE_THRESHOLD:
            main_labels.append(lbl)
            main_values.append(val)
            main_pcts.append(p)
        else:
            other_total += val

    if other_total > 0:
        main_labels.append("Other")
        main_values.append(other_total)
        main_pcts.append(other_total / total * 100)

    colors    = []
    color_idx = 0
    for lbl in main_labels:
        if lbl == "Other":
            colors.append(PIE_OTHER_COLOR)
        else:
            colors.append(PIE_COLORS[color_idx % len(PIE_COLORS)])
            color_idx += 1

    max_pct = max(main_pcts) if main_pcts else 0
    pull    = [0.04 if p == max_pct else 0 for p in main_pcts]

    fig = go.Figure(data=[go.Pie(
        labels=main_labels,
        values=main_values,
        hole=0.52,
        sort=True,
        direction="clockwise",
        textinfo="none",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "%{percent:.1%}<br>"
            "Count: %{value}<extra></extra>"
        ),
        marker=dict(
            colors=colors,
            line=dict(color=T["bg_card"], width=2),
        ),
        pull=pull,
    )])

    fig.update_layout(
        title=dict(
            text=title.upper(),
            x=0.0,
            xanchor="left",
            pad=dict(l=4),
            font=dict(family=T["font"], size=9, color=T["text_3"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=T["font"], color=T["text_2"], size=10),
        margin=dict(l=16, r=16, t=32, b=16),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=9, color=T["text_2"]),
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            itemwidth=80,
            entrywidth=120,
        ),
        hoverlabel=dict(
            bgcolor=T["bg_hover"],
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(family=T["font"], size=11, color=T["text_1"]),
        ),
        height=420,
        autosize=True,
    )

    return opy.plot(fig, auto_open=False, output_type="div", config=CHART_CONFIG)


# ─────────────────────────────────────────────────────────────────────
# MAIN VIEWS
# ─────────────────────────────────────────────────────────────────────
@login_required
def dashboard(request):
    cleaned_data = request.session.get(CLEANED_DATA_SESSION_KEY)
    df = _read_session_df(cleaned_data)

    df_html = None
    kpis = {}
    chart_equity = None
    chart_profit = None
    chart_hist = None
    chart_month = None
    trade_page = None
    chart_pie_sections = {}
    analysis_context = None
    analysis_query = ""
    review_df = None
    trade_sort = ""
    trade_dir = "asc"
    trade_review_columns = []
    excursion_state = _bind_dashboard_excursion_state(request, df)
    excursion_summary = _excursion_summary(excursion_state)

    filter_form = FilterForm(request.GET or None)
    trade_q = request.GET.get("q", "").strip()
    file_q = request.GET.get("file_q", "").strip()
    file_status = request.GET.get("file_status", "").strip()

    if df is not None and not df.empty:
        working_df, analysis_context = apply_analysis(
            df,
            request.GET,
            source_filename=request.session.get(
                "last_uploaded_file"
            ),
        )

        analysis_query = urlencode(
            dict(analysis_context.active_filters)
        )
        review_df = working_df.copy()
        if excursion_state:
            review_df = attach_excursion_evidence(
                review_df,
                excursion_state.get("evidence"),
            )

        kpis = _safe_compute_kpis(
            working_df
        )

        # ── Charts ────────────────────────────────────────────────────
        if "Profit" in working_df.columns:
            working_df["Profit"] = pd.to_numeric(
                working_df["Profit"],
                errors="coerce",
            )

            maybe_date = None
            for col in [
                "Open Time",
                "Open",
                "Date",
            ]:
                if (
                    col in working_df.columns
                    and pd.api.types.is_datetime64_any_dtype(
                        working_df[col]
                    )
                ):
                    maybe_date = col
                    break

            # Monthly P&L ─────────────────────────────────────────────
            if maybe_date:
                df_month = working_df.dropna(
                    subset=[maybe_date]
                ).copy()

                if not df_month.empty:
                    monthly = (
                        df_month
                        .groupby(
                            df_month[maybe_date]
                            .dt.to_period("M")
                            .astype(str)
                        )["Profit"]
                        .sum()
                        .reset_index(
                            name="Profit"
                        )
                    )

                    monthly.rename(
                        columns={
                            monthly.columns[0]: "Month"
                        },
                        inplace=True,
                    )

                    fig_month = go.Figure(
                        data=[
                            go.Bar(
                                x=monthly["Month"],
                                y=monthly["Profit"],
                                marker_color=[
                                    (
                                        T["green"]
                                        if value >= 0
                                        else T["red"]
                                    )
                                    for value
                                    in monthly["Profit"]
                                ],
                                marker_line_width=0,
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    "P&L: %{y:,.2f}"
                                    "<extra></extra>"
                                ),
                            )
                        ]
                    )

                    _apply_dark_theme(
                        fig_month
                    )

                    chart_month = opy.plot(
                        fig_month,
                        auto_open=False,
                        output_type="div",
                        config=CHART_CONFIG,
                    )

            # Equity curve ────────────────────────────────────────────
            working_df["Cumulative Profit"] = (
                working_df["Profit"]
                .fillna(0)
                .cumsum()
            )

            total_pnl = (
                working_df["Cumulative Profit"].iloc[-1]
                if len(working_df)
                else 0
            )

            fig_equity = go.Figure(
                data=[
                    go.Scatter(
                        x=list(
                            range(
                                len(working_df)
                            )
                        ),
                        y=working_df[
                            "Cumulative Profit"
                        ],
                        mode="lines",
                        line=dict(
                            color=(
                                T["green"]
                                if total_pnl >= 0
                                else T["red"]
                            ),
                            width=1.5,
                        ),
                        fill="tozeroy",
                        fillcolor=(
                            "rgba(34,197,94,0.07)"
                            if total_pnl >= 0
                            else "rgba(244,63,94,0.07)"
                        ),
                        hovertemplate=(
                            "Trade %{x}<br>"
                            "Cumulative: %{y:,.2f}"
                            "<extra></extra>"
                        ),
                    )
                ]
            )

            _apply_dark_theme(
                fig_equity
            )

            fig_equity.add_hline(
                y=0,
                line_dash="dot",
                line_color=(
                    "rgba(255,255,255,0.1)"
                ),
                line_width=1,
            )

            chart_equity = opy.plot(
                fig_equity,
                auto_open=False,
                output_type="div",
                config=CHART_CONFIG,
            )

            # Profit per trade ────────────────────────────────────────
            profit_vals = (
                working_df["Profit"]
                .fillna(0)
                .tolist()
            )

            fig_profit = go.Figure(
                data=[
                    go.Bar(
                        x=list(
                            range(
                                len(working_df)
                            )
                        ),
                        y=profit_vals,
                        marker_color=[
                            (
                                T["green"]
                                if value >= 0
                                else T["red"]
                            )
                            for value
                            in profit_vals
                        ],
                        marker_line_width=0,
                        hovertemplate=(
                            "Trade %{x}<br>"
                            "P&L: %{y:,.2f}"
                            "<extra></extra>"
                        ),
                    )
                ]
            )

            _apply_dark_theme(
                fig_profit
            )

            fig_profit.add_hline(
                y=0,
                line_dash="dot",
                line_color=(
                    "rgba(255,255,255,0.1)"
                ),
                line_width=1,
            )

            chart_profit = opy.plot(
                fig_profit,
                auto_open=False,
                output_type="div",
                config=CHART_CONFIG,
            )

            # Profit distribution histogram ───────────────────────────
            profit_clean = (
                working_df["Profit"]
                .dropna()
            )

            if not profit_clean.empty:
                p_min = profit_clean.min()
                p_max = profit_clean.max()
                n_bins = 40

                abs_max = max(
                    abs(p_min),
                    abs(p_max),
                )

                bins = np.linspace(
                    -abs_max,
                    abs_max,
                    n_bins + 1,
                )

                counts, edges = np.histogram(
                    profit_clean,
                    bins=bins,
                )

                midpoints = (
                    edges[:-1]
                    + edges[1:]
                ) / 2

                bar_width = float(
                    edges[1]
                    - edges[0]
                )

                bar_colors = [
                    (
                        T["green"]
                        if midpoint >= 0
                        else T["red"]
                    )
                    for midpoint
                    in midpoints
                ]

                fig_hist = go.Figure(
                    data=[
                        go.Bar(
                            x=midpoints.tolist(),
                            y=counts.tolist(),
                            width=(
                                bar_width
                                * 0.88
                            ),
                            marker_color=(
                                bar_colors
                            ),
                            marker_line_width=0,
                            hovertemplate=(
                                "Range: %{x:,.0f}<br>"
                                "Count: %{y}"
                                "<extra></extra>"
                            ),
                        )
                    ]
                )

                _apply_dark_theme(
                    fig_hist
                )

                fig_hist.add_vline(
                    x=0,
                    line_dash="dot",
                    line_color=(
                        "rgba(255,255,255,0.15)"
                    ),
                    line_width=1,
                )

                chart_hist = opy.plot(
                    fig_hist,
                    auto_open=False,
                    output_type="div",
                    config=CHART_CONFIG,
                )

            # Breakdown pie charts ────────────────────────────────────
            sections = [
                (
                    "Overall",
                    working_df,
                )
            ]

            if "Type" in working_df.columns:
                sections.extend(
                    [
                        (
                            "Buy",
                            working_df[
                                working_df[
                                    "Type"
                                ]
                                .astype(str)
                                .str.lower()
                                == "buy"
                            ],
                        ),
                        (
                            "Sell",
                            working_df[
                                working_df[
                                    "Type"
                                ]
                                .astype(str)
                                .str.lower()
                                == "sell"
                            ],
                        ),
                    ]
                )

            for label, section_df in sections:
                if (
                    section_df.empty
                    or "Profit"
                    not in section_df.columns
                    or "Symbol"
                    not in section_df.columns
                ):
                    continue

                section_df = (
                    section_df.copy()
                )

                section_df["Profit"] = (
                    pd.to_numeric(
                        section_df[
                            "Profit"
                        ],
                        errors="coerce",
                    )
                )

                win_loss = (
                    section_df
                    .groupby(
                        "Symbol"
                    )["Profit"]
                    .apply(
                        lambda values: pd.Series(
                            {
                                "Wins": (
                                    values > 0
                                ).sum(),
                                "Losses": (
                                    values <= 0
                                ).sum(),
                            }
                        )
                    )
                    .unstack()
                    .fillna(0)
                )

                wins = (
                    list(
                        win_loss["Wins"]
                    )
                    if "Wins"
                    in win_loss.columns
                    else []
                )

                losses = (
                    list(
                        win_loss["Losses"]
                    )
                    if "Losses"
                    in win_loss.columns
                    else []
                )

                chart_pie_sections[
                    f"{label.lower()}_count"
                ] = _build_pie(
                    title=(
                        f"{label} "
                        "Win/Loss Count by Pair"
                    ),
                    labels=(
                        [
                            f"{symbol} Wins"
                            for symbol
                            in win_loss.index
                        ]
                        + [
                            f"{symbol} Losses"
                            for symbol
                            in win_loss.index
                        ]
                    ),
                    values=(
                        wins
                        + losses
                    ),
                )

    if review_df is not None:
        review_df = enrich_trade_review(review_df)
        trade_sort = normalize_sort_key(
            request.GET.get("trade_sort")
        ) or ""
        trade_dir = normalize_sort_direction(
            request.GET.get("trade_dir")
        )
        if trade_sort:
            review_df = sort_trade_review(
                review_df,
                trade_sort,
                trade_dir,
            )
        trade_review_columns = build_trade_review_columns(
            review_df
        )
        for column in trade_review_columns:
            sort_key = column.get("sort_key")
            if not sort_key:
                column["sort_query"] = ""
                column["is_active"] = False
                continue
            column["sort_query"] = _trade_sort_query(
                request,
                sort_key,
                next_sort_direction(
                    trade_sort,
                    trade_dir,
                    sort_key,
                ),
            )
            column["is_active"] = sort_key == trade_sort
        trade_page = Paginator(
            build_trade_review_rows(
                review_df,
                trade_review_columns,
            ),
            TRADE_PAGE_SIZE,
        ).get_page(
            request.GET.get("trade_page")
        )

    trade_query = _query_without(
        request,
        "trade_page",
    )

    files_query = _query_without(
        request,
        "page",
    )

    files_qs = (
        TradingFile.objects
        .filter(
            user=request.user
        )
        .order_by(
            "-uploaded_at"
        )
    )

    if file_q:
        files_qs = files_qs.filter(
            file__icontains=file_q
        )

    if file_status:
        files_qs = files_qs.filter(
            status__iexact=file_status
        )

    files = Paginator(
        files_qs,
        5,
    ).get_page(
        request.GET.get("page")
    )

    return render(
        request,
        "performance/dashboard.html",
        {
            "df_html": df_html,
            "kpis": kpis,
            "chart_equity": chart_equity,
            "chart_profit": chart_profit,
            "chart_hist": chart_hist,
            "chart_month": chart_month,
            "chart_pie_sections": (
                chart_pie_sections
            ),
            "filter_form": filter_form,
            "files": files,
            "trade_page": trade_page,
            "trade_q": trade_q,
            "file_q": file_q,
            "file_status": file_status,
            "trade_query": trade_query,
            "files_query": files_query,
            "analysis_context": (
                analysis_context
            ),
            "analysis_query": (
                analysis_query
            ),
            "trade_sort": trade_sort,
            "trade_dir": trade_dir,
            "trade_review_columns": (
                trade_review_columns
            ),
            "market_data_form": MarketDataUploadForm(),
            "excursion_state": excursion_state,
            "excursion_summary": excursion_summary,
            "price_comparison_epsilon": PRICE_COMPARISON_EPSILON,
            "duration_tolerance_seconds": DURATION_TOLERANCE_SECONDS,
        },
    )


@login_required
@require_POST
def upload_market_data(request):
    clear_excursion_state(request.session)
    form = MarketDataUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(
            request,
            "Broker-bar upload was not accepted.",
        )
        return redirect("performance:dashboard")

    journal = _read_session_df(
        request.session.get(CLEANED_DATA_SESSION_KEY)
    )
    if journal is None or journal.empty:
        messages.error(
            request,
            "Load a cleaned journal before uploading market bars.",
        )
        return redirect("performance:dashboard")

    uploaded = form.cleaned_data["market_file"]
    raw_bytes = uploaded.read()
    try:
        parsed = parse_market_data_bytes(
            raw_bytes,
            uploaded.name,
            declared_source=form.cleaned_data["declared_source"],
            declared_export_method=form.cleaned_data["declared_export_method"],
        )
    except MarketDataValidationError as exc:
        messages.error(
            request,
            f"Market data was rejected ({exc.reason}).",
        )
        return redirect("performance:dashboard")

    result = compute_excursion_evidence(
        journal,
        parsed.dataframe,
        form.cleaned_data["time_basis"],
    )
    if result.run_status != RUN_STATUS_OK or not result.journal_fingerprint:
        messages.error(
            request,
            (
                "Excursion evidence could not be stored "
                f"({result.reason_code or result.run_status})."
            ),
        )
        return redirect("performance:dashboard")

    try:
        store_excursion_state(
            request.session,
            result,
            time_basis=form.cleaned_data["time_basis"],
            market_provenance=parsed.provenance,
        )
    except ExcursionStateError as exc:
        messages.error(
            request,
            f"Evidence could not be stored ({exc.reason}).",
        )
        return redirect("performance:dashboard")

    messages.success(
        request,
        "Broker-bar evidence was stored for the current journal.",
    )
    return redirect("performance:dashboard")


# ─────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────
@login_required
def upload_file(request):
    if request.method == "POST":
        form = TradingFileForm(request.POST, request.FILES)
        if form.is_valid():
            trading_file = form.save(commit=False)
            trading_file.user = request.user
            trading_file.status = "pending"
            trading_file.save()

            try:
                df = clean_ftmo_csv(trading_file.file.path)
                request.session["last_uploaded_file"] = trading_file.file.name
                request.session[CLEANED_DATA_SESSION_KEY] = df.to_json(
                    orient="split",
                    date_format="iso",
                )

                trading_file.status = "processed"
                trading_file.save(update_fields=["status"])

                messages.success(
                    request,
                    "File uploaded and processed successfully.",
                )
                return redirect("performance:dashboard")

            except IngestionError as exc:
                request.session.pop(CLEANED_DATA_SESSION_KEY, None)

                trading_file.status = "error"
                trading_file.save(update_fields=["status"])

                messages.error(request, str(exc))

            except Exception:
                request.session.pop(CLEANED_DATA_SESSION_KEY, None)

                trading_file.status = "error"
                trading_file.save(update_fields=["status"])

                messages.error(
                    request,
                    "An error occurred while processing your file.",
                )
    else:
        form = TradingFileForm()

    return render(
        request,
        "performance/upload_file.html",
        {"form": form},
    )


# ─────────────────────────────────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────────────────────────────────
@staff_member_required
def admin_all_files(request):
    file_q = request.GET.get("file_q", "").strip()
    status = request.GET.get("status", "").strip()

    all_files_qs = TradingFile.objects.select_related("user").order_by("-uploaded_at")
    if file_q:
        all_files_qs = all_files_qs.filter(file__icontains=file_q)
    if status:
        all_files_qs = all_files_qs.filter(status__iexact=status)

    all_files   = Paginator(all_files_qs, 15).get_page(request.GET.get("page"))
    files_query = _query_without(request, "page")

    return render(request, "performance/admin_files.html", {
        "all_files":   all_files,
        "file_q":      file_q,
        "status":      status,
        "files_query": files_query,
    })


@staff_member_required
def admin_delete_file(request, file_id: int):
    file_obj = get_object_or_404(TradingFile, id=file_id)
    file_obj.delete()
    return redirect("performance:admin_all_files")


@login_required
def load_file(request, file_id: int):
    trading_file = get_object_or_404(
        TradingFile,
        id=file_id,
        user=request.user,
    )

    try:
        df = clean_ftmo_csv(trading_file.file.path)

        request.session["last_uploaded_file"] = trading_file.file.name
        request.session[CLEANED_DATA_SESSION_KEY] = df.to_json(
            orient="split",
            date_format="iso",
        )

        messages.success(
            request,
            f"Loaded {trading_file.file.name} successfully.",
        )

    except IngestionError as exc:
        request.session.pop(CLEANED_DATA_SESSION_KEY, None)
        messages.error(request, str(exc))

    except Exception:
        request.session.pop(CLEANED_DATA_SESSION_KEY, None)
        messages.error(
            request,
            "An error occurred while loading the trading file.",
        )

    return redirect("performance:dashboard")


# ─────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────
@login_required
def download_cleaned_csv(request):
    df = _read_session_df(request.session.get(CLEANED_DATA_SESSION_KEY))
    if df is None or df.empty:
        return HttpResponse("No cleaned session data is available.", status=400)

    response = HttpResponse(df.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="cleaned_trading_data.csv"'
    return response


@login_required
def download_excel(request):
    df = _read_session_df(request.session.get(CLEANED_DATA_SESSION_KEY))
    if df is None or df.empty:
        return HttpResponse("No cleaned session data is available.", status=400)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Trades")
    output.seek(0)

    response = HttpResponse(
        output.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = 'attachment; filename="trades.xlsx"'
    return response


@login_required
def download_pdf(request):
    df = _read_session_df(request.session.get(CLEANED_DATA_SESSION_KEY))
    if df is None or df.empty:
        return HttpResponse("No data available.", status=400)

    is_trade_like = _is_trade_like_df(df)
    kpis          = _safe_compute_kpis(df)

    def choose_report_columns(pdf_df):
        preferred = [
            "Open Time", "Open", "Date", "Symbol", "Type", "Side",
            "Size", "Volume", "Profit", "Commission", "Swap", "Balance", "Pips",
        ]
        present = [c for c in preferred if c in pdf_df.columns]
        if present:
            return present[:8]
        excluded = ["shipping", "address", "line1", "line2", "city", "postcode",
                    "country", "stripe", "intent", "charge", "refund", "email"]
        compact = [c for c in pdf_df.columns if not any(w in c.lower() for w in excluded)]
        return compact[:8] if compact else list(pdf_df.columns[:8])

    pdf_df = df.copy()
    for col in pdf_df.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf_df[col]):
            pdf_df[col] = pdf_df[col].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        elif pd.api.types.is_numeric_dtype(pdf_df[col]):
            pdf_df[col] = pd.to_numeric(pdf_df[col], errors="coerce").round(2)
    pdf_df = pdf_df.fillna("")

    report_columns   = choose_report_columns(pdf_df)
    preview_df       = pdf_df[report_columns].head(25)
    trade_columns    = [str(col) for col in preview_df.columns.tolist()]
    trade_rows       = preview_df.astype(str).values.tolist()

    template = get_template("performance/pdf_report.html")
    html = template.render({
        "user":                 request.user,
        "kpis":                 kpis,
        "trade_columns":        trade_columns,
        "trade_rows":           trade_rows,
        "has_trades":           len(trade_rows) > 0,
        "total_trade_rows":     len(pdf_df),
        "displayed_trade_rows": len(trade_rows),
        "is_trade_like":        is_trade_like,
        "now":                  timezone.now(),
    })

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="trading_report.pdf"'
    try:
        pisa_status = pisa.CreatePDF(html, dest=response)
    except Exception as e:
        return HttpResponse(f"PDF generation exception: {e}", status=500)
    if pisa_status.err:
        return HttpResponse("Error generating PDF", status=500)
    return response


@login_required
def kpi_report(request):
    df = _read_session_df(
        request.session.get(CLEANED_DATA_SESSION_KEY)
    )

    analysis_context = None
    analysis_query = ""
    kpis = {}

    if df is not None and not df.empty:
        working_df, analysis_context = apply_analysis(
            df,
            request.GET,
            source_filename=request.session.get(
                "last_uploaded_file"
            ),
        )

        kpis = _safe_compute_kpis(working_df)

        analysis_query = urlencode(
            dict(analysis_context.active_filters)
        )

    kpi_rows = [
        {"metric": key, "value": value}
        for key, value in (kpis or {}).items()
    ]
    kpi_q = request.GET.get("kpi_q", "").strip()

    if kpi_q:
        kpi_rows = [
            row for row in kpi_rows
            if kpi_q.lower() in str(row["metric"]).lower()
            or kpi_q.lower() in str(row["value"]).lower()
        ]

    kpi_page = Paginator(kpi_rows, 10).get_page(
        request.GET.get("kpi_page")
    )

    kpi_query_params = {}
    if analysis_context is not None:
        kpi_query_params.update(
            dict(analysis_context.active_filters)
        )
    if kpi_q:
        kpi_query_params["kpi_q"] = kpi_q
    kpi_query = urlencode(kpi_query_params)

    return render(request, "performance/kpi_report.html", {
        "kpis": kpis,
        "kpi_page": kpi_page,
        "kpi_q": kpi_q,
        "kpi_query": kpi_query,
        "generated_at": timezone.now(),
        "analysis_context": analysis_context,
        "analysis_query": analysis_query,
    })


@login_required
def export_excel(request):
    fallback_columns = [
        "entry_time", "exit_time", "symbol", "side", "qty",
        "entry", "exit", "sl", "tp", "pnl", "rr", "tag", "notes",
    ]

    df = _read_session_df(request.session.get(CLEANED_DATA_SESSION_KEY))

    def get_available_columns(dataframe):
        if dataframe is None or dataframe.empty:
            return fallback_columns
        preferred = [
            "Open Time", "Open", "Date", "Close Time",
            "Symbol", "Type", "Side",
            "Size", "Volume", "Entry", "Exit",
            "Profit", "Commission", "Swap", "Balance",
            "Pips", "SL", "TP", "RR", "rr",
            "Tag", "Tags", "Notes", "Comment", "Comments",
        ]
        present = [c for c in preferred if c in dataframe.columns]
        return present if present else list(dataframe.columns)

    analysis_context = None
    analysis_query = ""
    analysis_df = None
    export_scope_df = None
    min_rr_applied = False
    applied_min_rr = None

    min_rr = (request.GET.get("min_rr") or "").strip()
    include_kpis = "include_kpis" in request.GET

    if df is not None and not df.empty:
        analysis_df, analysis_context = apply_analysis(
            df,
            request.GET,
            source_filename=request.session.get(
                "last_uploaded_file"
            ),
        )
        analysis_query = urlencode(
            dict(analysis_context.active_filters)
        )
        export_scope_df = analysis_df.copy()

        rr_col = _first_present(
            export_scope_df.columns,
            ["RR", "rr", "R:R", "risk_reward", "Risk Reward"],
        )
        if min_rr and rr_col:
            try:
                min_rr_value = float(min_rr)
                export_scope_df[rr_col] = pd.to_numeric(
                    export_scope_df[rr_col],
                    errors="coerce",
                )
                export_scope_df = export_scope_df[
                    export_scope_df[rr_col] >= min_rr_value
                ]
                min_rr_applied = True
                applied_min_rr = min_rr
            except ValueError:
                pass

    available_columns = get_available_columns(df)
    selected_cols = (
        request.GET.getlist("cols")
        if request.GET.getlist("cols")
        else available_columns
    )
    exported_row_count = (
        len(export_scope_df)
        if export_scope_df is not None
        else 0
    )

    if "download" not in request.GET:
        return render(request, "performance/excel_export.html", {
            "columns": available_columns,
            "selected_cols": selected_cols,
            "data_ready": bool(df is not None and not df.empty),
            "available_rows": len(df) if df is not None else 0,
            "analysis_context": analysis_context,
            "analysis_query": analysis_query,
            "exported_row_count": exported_row_count,
            "min_rr_applied": min_rr_applied,
            "applied_min_rr": applied_min_rr,
        })

    if df is None or df.empty:
        return HttpResponse(
            "No cleaned session data is available. Upload a trade history file first.",
            status=400,
        )

    selected_cols = [
        c for c in request.GET.getlist("cols")
        if c in export_scope_df.columns
    ]
    if not selected_cols:
        selected_cols = [
            c for c in available_columns
            if c in export_scope_df.columns
        ]
    if not selected_cols:
        selected_cols = list(export_scope_df.columns)

    export_df = export_scope_df[selected_cols].copy()
    for col in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[col]):
            export_df[col] = export_df[col].dt.strftime("%Y-%m-%d %H:%M")
    export_df = export_df.fillna("")

    source_file = (
        analysis_context.source_filename
        if analysis_context.source_filename
        else "Session dataset"
    )
    analysis_filters = analysis_query if analysis_query else "None"
    min_rr_meta = applied_min_rr if min_rr_applied else "Not applied"

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Trades")

        pd.DataFrame([
            {"Field": "Generated At", "Value": timezone.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Field": "Source File", "Value": source_file},
            {"Field": "Source Rows", "Value": analysis_context.source_row_count},
            {"Field": "Analysed Rows", "Value": analysis_context.filtered_row_count},
            {"Field": "Exported Rows", "Value": len(export_scope_df)},
            {"Field": "Analysis Filters", "Value": analysis_filters},
            {"Field": "Export-only Min RR", "Value": min_rr_meta},
        ]).to_excel(writer, index=False, sheet_name="ExportMeta")

        if include_kpis:
            kpis = _safe_compute_kpis(analysis_df)
            kpi_rows = (
                [{"Metric": k, "Value": v} for k, v in kpis.items()]
                if kpis
                else [{"Metric": "Info", "Value": "No KPI values available for the current analysis scope."}]
            )
            pd.DataFrame(kpi_rows).to_excel(writer, index=False, sheet_name="KPIs")

    output.seek(0)
    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    response = HttpResponse(
        output.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="tradeintel_report_{timestamp}.xlsx"'
    return response


def project_one_plan(request):
    return render(request, "performance/project_one_plan.html")
