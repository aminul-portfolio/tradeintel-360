import csv
import os
from io import BytesIO, StringIO

import pandas as pd
import plotly.express as px
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
from xhtml2pdf import pisa

from .forms import FilterForm, TradingFileForm
from .models import TradingFile
from .utils import compute_kpis


# ─────────────────────────────────────────────
# DARK THEME HELPER
# ─────────────────────────────────────────────

def _apply_dark_theme(fig, title: str = "") -> None:
    """Apply the TradeIntel dark theme to any Plotly figure in-place."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="IBM Plex Mono, monospace", size=11, color="#4a5470"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", color="#8896b3", size=11),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.07)",
            tickcolor="rgba(255,255,255,0.07)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.07)",
            tickcolor="rgba(255,255,255,0.07)",
            zeroline=False,
        ),
        margin=dict(l=8, r=8, t=36 if title else 8, b=8),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(size=10, color="#8896b3"),
        ),
        hoverlabel=dict(
            bgcolor="#1a2236",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(family="IBM Plex Mono, monospace", size=11, color="#f0f4ff"),
        ),
    )


# ─────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────

def _query_without(request, *keys):
    q = request.GET.copy()
    for key in keys:
        q.pop(key, None)
    return q.urlencode()


def _first_present(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def _read_session_df(cleaned_data):
    """Safely load the cleaned session dataset. Returns a DataFrame or None."""
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


def _is_trade_like_df(df):
    if df is None or df.empty:
        return False
    trade_like_markers = ["Open", "Open Time", "Date", "Symbol", "Type", "Side", "Profit"]
    return any(col in df.columns for col in trade_like_markers)


def _smart_filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if df is None or df.empty or not q:
        return df
    q = str(q).strip()
    if not q:
        return df

    mask = pd.Series(False, index=df.index)

    preferred_text_cols = [
        "Symbol", "Type", "Side", "Tag", "Tags", "Notes",
        "Comment", "Comments", "Strategy", "Reason",
    ]
    text_cols = [c for c in preferred_text_cols if c in df.columns]
    if not text_cols:
        text_cols = list(df.select_dtypes(include=["object"]).columns)
    for col in text_cols:
        mask |= df[col].astype(str).str.contains(q, case=False, na=False)

    try:
        q_num = float(str(q).replace(",", ""))
    except ValueError:
        q_num = None

    if q_num is not None:
        numeric_cols = [
            c for c in ["Profit", "Commission", "Swap", "Balance", "Pips", "Size", "Volume"]
            if c in df.columns
        ]
        for col in numeric_cols:
            mask |= pd.to_numeric(df[col], errors="coerce").round(8).eq(q_num)

    for date_col in ["Open", "Open Time", "Date"]:
        if date_col in df.columns:
            date_series = pd.to_datetime(df[date_col], errors="coerce")
            mask |= date_series.dt.strftime("%Y-%m-%d").fillna("").str.contains(q, case=False, na=False)
            mask |= date_series.dt.strftime("%Y-%m").fillna("").str.contains(q, case=False, na=False)

    return df[mask]


# ─────────────────────────────────────────────
# MAIN VIEWS
# ─────────────────────────────────────────────

@login_required
def dashboard(request):
    cleaned_data = request.session.get("cleaned_data")
    df = _read_session_df(cleaned_data)

    df_html        = None
    kpis           = {}
    chart_equity   = None
    chart_profit   = None
    chart_hist     = None
    chart_month    = None
    trade_page     = None
    chart_pie_sections = {}

    filter_form = FilterForm(request.GET or None)
    trade_q     = request.GET.get("q", "").strip()
    file_q      = request.GET.get("file_q", "").strip()
    file_status = request.GET.get("file_status", "").strip()

    if df is not None and not df.empty:
        working_df = df.copy()

        # ── Date / symbol filters ──
        if filter_form.is_valid():
            start_date = filter_form.cleaned_data.get("start_date")
            end_date   = filter_form.cleaned_data.get("end_date")
            symbol     = filter_form.cleaned_data.get("symbol")

            for date_col in ["Open", "Open Time", "Date"]:
                if date_col in working_df.columns:
                    working_df[date_col] = pd.to_datetime(
                        working_df[date_col], errors="coerce", dayfirst=True,
                    )
                    if start_date:
                        working_df = working_df[working_df[date_col] >= pd.to_datetime(start_date)]
                    if end_date:
                        working_df = working_df[working_df[date_col] <= pd.to_datetime(end_date)]
                    break

            if "Symbol" in working_df.columns and symbol:
                working_df = working_df[
                    working_df["Symbol"].astype(str).str.contains(symbol, case=False, na=False)
                ]

        if trade_q:
            working_df = _smart_filter_df(working_df, trade_q)

        # ── Paginated trade table ──
        trade_paginator  = Paginator(working_df.to_dict("records"), 10)
        trade_page       = trade_paginator.get_page(request.GET.get("trade_page"))
        df_page          = pd.DataFrame(trade_page.object_list)

        if not df_page.empty:
            df_html = df_page.to_html(
                classes="table table-striped align-middle",
                index=False,
            )

        kpis = _safe_compute_kpis(working_df)

        # ── Charts ──
        if "Profit" in working_df.columns:
            working_df["Profit"] = pd.to_numeric(working_df["Profit"], errors="coerce")

            # Determine date column for monthly chart
            maybe_date = None
            for col in ["Open", "Open Time", "Date"]:
                if col in working_df.columns and pd.api.types.is_datetime64_any_dtype(working_df[col]):
                    maybe_date = col
                    break

            # Monthly bar chart
            if maybe_date:
                df_month = working_df.dropna(subset=[maybe_date]).copy()
                if not df_month.empty:
                    monthly_profit = (
                        df_month
                        .groupby(df_month[maybe_date].dt.to_period("M").astype(str))["Profit"]
                        .sum()
                        .reset_index(name="Profit")
                    )
                    monthly_profit.rename(columns={monthly_profit.columns[0]: "Month"}, inplace=True)

                    colors_monthly = [
                        "#22c55e" if v >= 0 else "#f43f5e"
                        for v in monthly_profit["Profit"]
                    ]
                    fig_month = go.Figure(
                        data=[go.Bar(
                            x=monthly_profit["Month"],
                            y=monthly_profit["Profit"],
                            marker_color=colors_monthly,
                        )]
                    )
                    _apply_dark_theme(fig_month)
                    chart_month = opy.plot(fig_month, auto_open=False, output_type="div")

            # Equity curve
            working_df["Cumulative Profit"] = working_df["Profit"].fillna(0).cumsum()
            fig_equity = go.Figure(
                data=[go.Scatter(
                    x=list(range(len(working_df))),
                    y=working_df["Cumulative Profit"],
                    mode="lines",
                    line=dict(color="#00e5b4", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(0,229,180,0.07)",
                )]
            )
            _apply_dark_theme(fig_equity)
            chart_equity = opy.plot(fig_equity, auto_open=False, output_type="div")

            # Profit per trade bars
            profit_vals = working_df["Profit"].fillna(0).tolist()
            colors_profit = ["#22c55e" if v >= 0 else "#f43f5e" for v in profit_vals]
            fig_profit = go.Figure(
                data=[go.Bar(
                    x=list(range(len(working_df))),
                    y=profit_vals,
                    marker_color=colors_profit,
                )]
            )
            _apply_dark_theme(fig_profit)
            chart_profit = opy.plot(fig_profit, auto_open=False, output_type="div")

            # Profit distribution histogram
            fig_hist = go.Figure(
                data=[go.Histogram(
                    x=working_df["Profit"].dropna(),
                    nbinsx=30,
                    marker_color="#00e5b4",
                    opacity=0.85,
                )]
            )
            _apply_dark_theme(fig_hist)
            chart_hist = opy.plot(fig_hist, auto_open=False, output_type="div")

            # ── Pie / breakdown charts ──
            sections = [("Overall", working_df)]
            if "Type" in working_df.columns:
                sections.extend([
                    ("Buy",  working_df[working_df["Type"].astype(str).str.lower() == "buy"]),
                    ("Sell", working_df[working_df["Type"].astype(str).str.lower() == "sell"]),
                ])

            def build_pie(title, labels, values):
                fig_pie = go.Figure(
                    data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        hoverinfo="label+percent+value",
                        textinfo="percent",
                        textfont=dict(size=11, color="#8896b3"),
                        marker=dict(
                            colors=px.colors.qualitative.Set3,
                            line=dict(color="rgba(0,0,0,0)", width=0),
                        ),
                    )]
                )
                fig_pie.update_layout(
                    title=dict(
                        text=title,
                        font=dict(family="IBM Plex Mono, monospace", size=11, color="#4a5470"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="IBM Plex Mono, monospace", color="#8896b3", size=10),
                    margin=dict(l=8, r=8, t=40, b=100),
                    legend=dict(
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(255,255,255,0.07)",
                        borderwidth=1,
                        font=dict(size=9, color="#8896b3"),
                        orientation="h",
                        x=0.5, xanchor="center", y=-0.25,
                    ),
                    hoverlabel=dict(
                        bgcolor="#1a2236",
                        bordercolor="rgba(255,255,255,0.12)",
                        font=dict(family="IBM Plex Mono, monospace", size=11, color="#f0f4ff"),
                    ),
                )
                return opy.plot(fig_pie, auto_open=False, output_type="div")

            for label, section_df in sections:
                if (
                    section_df.empty
                    or "Profit" not in section_df.columns
                    or "Symbol" not in section_df.columns
                ):
                    continue

                section_df = section_df.copy()
                section_df["Profit"] = pd.to_numeric(section_df["Profit"], errors="coerce")

                win_loss = (
                    section_df.groupby("Symbol")["Profit"]
                    .apply(lambda x: pd.Series({
                        "Wins":   (x > 0).sum(),
                        "Losses": (x <= 0).sum(),
                    }))
                    .unstack()
                    .fillna(0)
                )

                wins   = list(win_loss["Wins"])   if "Wins"   in win_loss.columns else []
                losses = list(win_loss["Losses"]) if "Losses" in win_loss.columns else []

                chart_pie_sections[f"{label.lower()}_count"] = build_pie(
                    title=f"{label} Win/Loss Count by Pair",
                    labels=(
                        [f"{s} Wins"   for s in win_loss.index] +
                        [f"{s} Losses" for s in win_loss.index]
                    ),
                    values=wins + losses,
                )

    trade_query = _query_without(request, "trade_page")
    files_query = _query_without(request, "page")

    files_qs = TradingFile.objects.filter(user=request.user).order_by("-uploaded_at")
    if file_q:
        files_qs = files_qs.filter(file__icontains=file_q)
    if file_status:
        files_qs = files_qs.filter(status__iexact=file_status)

    files = Paginator(files_qs, 5).get_page(request.GET.get("page"))

    return render(request, "performance/dashboard.html", {
        "df_html":            df_html,
        "kpis":               kpis,
        "chart_equity":       chart_equity,
        "chart_profit":       chart_profit,
        "chart_hist":         chart_hist,
        "chart_month":        chart_month,
        "chart_pie_sections": chart_pie_sections,
        "filter_form":        filter_form,
        "files":              files,
        "trade_page":         trade_page,
        "trade_q":            trade_q,
        "file_q":             file_q,
        "file_status":        file_status,
        "trade_query":        trade_query,
        "files_query":        files_query,
    })


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

@login_required
def upload_file(request):
    if request.method == "POST":
        form = TradingFileForm(request.POST, request.FILES)
        if form.is_valid():
            trading_file = form.save(commit=False)
            trading_file.user   = request.user
            trading_file.status = "pending"
            trading_file.save()
            try:
                df = clean_ftmo_csv(trading_file.file.path)
                request.session["last_uploaded_file"] = trading_file.file.name
                request.session["cleaned_data"] = df.to_json(orient="split", date_format="iso")
                trading_file.status = "processed"
                trading_file.save(update_fields=["status"])
                messages.success(request, "File uploaded and processed successfully.")
                return redirect("performance:dashboard")
            except ValueError as ve:
                messages.error(request, str(ve))
                trading_file.status = "error"
                trading_file.save(update_fields=["status"])
            except Exception:
                messages.error(request, "An error occurred while processing your file.")
                trading_file.status = "error"
                trading_file.save(update_fields=["status"])
    else:
        form = TradingFileForm()

    return render(request, "performance/upload_file.html", {"form": form})


# ─────────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────────

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
    trading_file = get_object_or_404(TradingFile, id=file_id, user=request.user)
    try:
        df = clean_ftmo_csv(trading_file.file.path)
        request.session["last_uploaded_file"] = trading_file.file.name
        request.session["cleaned_data"] = df.to_json(orient="split", date_format="iso")
        messages.success(request, f"Loaded {trading_file.file.name} successfully.")
    except Exception as e:
        messages.error(request, f"Error loading file: {e}")
    return redirect("performance:dashboard")


# ─────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────

@login_required
def download_cleaned_csv(request):
    df = _read_session_df(request.session.get("cleaned_data"))
    if df is None or df.empty:
        return HttpResponse("No cleaned session data is available.", status=400)

    response = HttpResponse(df.to_csv(index=False), content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="cleaned_trading_data.csv"'
    return response


@login_required
def download_excel(request):
    df = _read_session_df(request.session.get("cleaned_data"))
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
    df = _read_session_df(request.session.get("cleaned_data"))
    if df is None or df.empty:
        return HttpResponse("No data available.", status=400)

    is_trade_like = _is_trade_like_df(df)
    kpis          = _safe_compute_kpis(df)

    def choose_report_columns(pdf_df: pd.DataFrame) -> list[str]:
        preferred_trade_cols = [
            "Open Time", "Open", "Date", "Symbol", "Type", "Side",
            "Size", "Volume", "Profit", "Commission", "Swap", "Balance", "Pips",
        ]
        present_trade_cols = [c for c in preferred_trade_cols if c in pdf_df.columns]
        if present_trade_cols:
            return present_trade_cols[:8]

        excluded_keywords = [
            "shipping", "address", "line1", "line2", "city", "postcode", "country",
            "stripe", "intent", "charge", "refund", "email",
        ]
        compact_cols = [
            c for c in pdf_df.columns
            if not any(word in c.lower() for word in excluded_keywords)
        ]
        return compact_cols[:8] if compact_cols else list(pdf_df.columns[:8])

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
    df   = _read_session_df(request.session.get("cleaned_data"))
    kpis = _safe_compute_kpis(df)

    kpi_rows = [{"metric": k, "value": v} for k, v in (kpis or {}).items()]
    kpi_q    = request.GET.get("kpi_q", "").strip()

    if kpi_q:
        kpi_rows = [
            row for row in kpi_rows
            if kpi_q.lower() in str(row["metric"]).lower()
            or kpi_q.lower() in str(row["value"]).lower()
        ]

    kpi_page  = Paginator(kpi_rows, 10).get_page(request.GET.get("kpi_page"))
    kpi_query = _query_without(request, "kpi_page")

    return render(request, "performance/kpi_report.html", {
        "kpis":         kpis,
        "kpi_page":     kpi_page,
        "kpi_q":        kpi_q,
        "kpi_query":    kpi_query,
        "generated_at": timezone.now(),
    })


@login_required
def export_excel(request):
    fallback_columns = [
        "entry_time", "exit_time", "symbol", "side", "qty",
        "entry", "exit", "sl", "tp", "pnl", "rr", "tag", "notes",
    ]

    df = _read_session_df(request.session.get("cleaned_data"))

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
        present_preferred = [c for c in preferred if c in dataframe.columns]
        return present_preferred if present_preferred else list(dataframe.columns)

    available_columns = get_available_columns(df)
    selected_cols     = request.GET.getlist("cols") if request.GET.getlist("cols") else available_columns

    if "download" not in request.GET:
        return render(request, "performance/excel_export.html", {
            "columns":       available_columns,
            "selected_cols": selected_cols,
            "data_ready":    bool(df is not None and not df.empty),
            "available_rows": len(df) if df is not None else 0,
        })

    if df is None or df.empty:
        return HttpResponse(
            "No cleaned session data is available. Upload a trade history file first.",
            status=400,
        )

    working_df = df.copy()

    date_candidates = ["Open Time", "Open", "Date", "entry_time", "exit_time", "created_at", "updated_at"]
    date_col = _first_present(working_df.columns, date_candidates)
    if date_col:
        working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")

    start_date  = (request.GET.get("start_date") or "").strip()
    end_date    = (request.GET.get("end_date")   or "").strip()
    symbol      = (request.GET.get("symbol")     or "").strip()
    min_rr      = (request.GET.get("min_rr")     or "").strip()
    include_kpis = "include_kpis" in request.GET

    if date_col:
        if start_date:
            working_df = working_df[working_df[date_col] >= pd.to_datetime(start_date, errors="coerce")]
        if end_date:
            working_df = working_df[working_df[date_col] <= pd.to_datetime(end_date, errors="coerce")]

    symbol_col = _first_present(working_df.columns, ["Symbol", "symbol"])
    if symbol and symbol_col:
        working_df = working_df[
            working_df[symbol_col].astype(str).str.contains(symbol, case=False, na=False)
        ]

    rr_col = _first_present(working_df.columns, ["RR", "rr", "R:R", "risk_reward", "Risk Reward"])
    if min_rr and rr_col:
        try:
            min_rr_value = float(min_rr)
            working_df[rr_col] = pd.to_numeric(working_df[rr_col], errors="coerce")
            working_df = working_df[working_df[rr_col] >= min_rr_value]
        except ValueError:
            pass

    selected_cols = [c for c in request.GET.getlist("cols") if c in working_df.columns]
    if not selected_cols:
        selected_cols = [c for c in available_columns if c in working_df.columns]
    if not selected_cols:
        selected_cols = list(working_df.columns)

    export_df = working_df[selected_cols].copy()
    for col in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[col]):
            export_df[col] = export_df[col].dt.strftime("%Y-%m-%d %H:%M")
    export_df = export_df.fillna("")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Trades")

        pd.DataFrame([
            {"Field": "Generated At",        "Value": timezone.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Field": "Source Rows",          "Value": len(df)},
            {"Field": "Exported Rows",        "Value": len(export_df)},
            {"Field": "Date Filter Column",   "Value": date_col or "Not available"},
            {"Field": "Symbol Filter",        "Value": symbol or "Not set"},
            {"Field": "Min RR Filter",        "Value": min_rr or "Not set"},
        ]).to_excel(writer, index=False, sheet_name="ExportMeta")

        if include_kpis:
            kpis     = _safe_compute_kpis(working_df)
            kpi_rows = (
                [{"Metric": k, "Value": v} for k, v in kpis.items()]
                if kpis
                else [{"Metric": "Info", "Value": "No KPI values available for the current filtered dataset."}]
            )
            pd.DataFrame(kpi_rows).to_excel(writer, index=False, sheet_name="KPIs")

    output.seek(0)
    timestamp = timezone.now().strftime("%Y%m%d_%H%M")
    response  = HttpResponse(
        output.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="tradeintel_report_{timestamp}.xlsx"'
    return response


def project_one_plan(request):
    return render(request, "performance/project_one_plan.html")


# ─────────────────────────────────────────────
# FILE CLEANING HELPER
# ─────────────────────────────────────────────

def clean_ftmo_csv(file_path: str) -> pd.DataFrame:
    """Reads and cleans a CSV/XLSX trading file. Returns a cleaned DataFrame."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif ext == ".csv":
        encodings = ["utf-8", "ISO-8859-1"]
        delimiter = ","

        with open(file_path, "r", encoding=encodings[0], errors="ignore") as f:
            sample = f.read(2048)
            try:
                dialect   = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc, delimiter=delimiter, on_bad_lines="skip")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("Unable to read the CSV file with supported encodings.")

    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    df.dropna(how="all", inplace=True)
    df.columns = [c.strip() for c in df.columns]

    for date_col in ["Open Time", "Open", "Date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            break

    for col in ["Size", "Profit", "Commission", "Swap", "Balance", "Pips", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df