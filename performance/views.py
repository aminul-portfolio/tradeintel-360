# performance/views.py
import csv
import io
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
from .utils import compute_kpis  # uses your current KPI function


@login_required
def dashboard(request):
    """
    Reads cleaned_data from session -> filters -> shows KPIs, charts, paginated trade table.
    Also lists uploaded files.
    """
    cleaned_data = request.session.get("cleaned_data")

    # placeholders
    df_html = None
    kpis = None
    chart_equity = None
    chart_profit = None
    chart_hist = None
    chart_month = None
    trade_page = None
    chart_pie_sections = {}

    filter_form = FilterForm(request.GET or None)

    if cleaned_data:
        df = pd.read_json(StringIO(cleaned_data), orient="split")

        # Apply filters (date + symbol) based on your FilterForm
        if filter_form.is_valid():
            start_date = filter_form.cleaned_data.get("start_date")
            end_date = filter_form.cleaned_data.get("end_date")
            symbol = filter_form.cleaned_data.get("symbol")

            # try to parse a date column
            for date_col in ["Open", "Open Time", "Date"]:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
                    if start_date:
                        df = df[df[date_col] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df[date_col] <= pd.to_datetime(end_date)]
                    break

            if "Symbol" in df.columns and symbol:
                df = df[df["Symbol"].str.contains(symbol, case=False, na=False)]

        # Paginated trade table
        trade_paginator = Paginator(df.to_dict("records"), 10)
        trade_page_number = request.GET.get("trade_page")
        trade_page = trade_paginator.get_page(trade_page_number)
        df_page = pd.DataFrame(trade_page.object_list)
        df_html = df_page.to_html(classes="table table-striped", index=False) if not df_page.empty else None

        # KPIs (your current utils.compute_kpis)
        try:
            kpis = compute_kpis(df)
        except Exception:
            kpis = None

        # Charts if Profit exists
        if "Profit" in df.columns:
            # Identify any parsed datetime column for monthly grouping
            maybe_date = None
            for col in ["Open", "Open Time", "Date"]:
                if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                    maybe_date = col
                    break

            if maybe_date:
                df_month = df.dropna(subset=[maybe_date]).copy()
                if not df_month.empty:
                    df_month["Month"] = df_month[maybe_date].dt.to_period("M").astype(str)
                    monthly_profit = df_month.groupby("Month")["Profit"].sum().reset_index()
                    chart_month = opy.plot(
                        go.Figure(
                            data=[go.Bar(x=monthly_profit["Month"], y=monthly_profit["Profit"])],
                            layout=go.Layout(title="Monthly Performance"),
                        ),
                        auto_open=False,
                        output_type="div",
                    )

            # Equity curve
            df["Cumulative Profit"] = df["Profit"].cumsum()
            chart_equity = opy.plot(
                go.Figure(
                    data=[go.Scatter(x=list(range(len(df))), y=df["Cumulative Profit"], mode="lines")],
                    layout=go.Layout(title="Equity Curve"),
                ),
                auto_open=False,
                output_type="div",
            )
            # Profit per trade
            chart_profit = opy.plot(
                go.Figure(
                    data=[go.Bar(x=list(range(len(df))), y=df["Profit"])],
                    layout=go.Layout(title="Profit per Trade"),
                ),
                auto_open=False,
                output_type="div",
            )
            # Histogram
            chart_hist = opy.plot(
                go.Figure(
                    data=[go.Histogram(x=df["Profit"], nbinsx=30)],
                    layout=go.Layout(title="Profit Distribution Histogram"),
                ),
                auto_open=False,
                output_type="div",
            )

            # Optional pies by Type if available
            sections = [("Overall", df)]
            if "Type" in df.columns:
                sections.extend([
                    ("Buy", df[df["Type"].astype(str).str.lower() == "buy"]),
                    ("Sell", df[df["Type"].astype(str).str.lower() == "sell"]),
                ])

            def build_pie(title, labels, values, colors):
                return opy.plot(
                    go.Figure(
                        data=[go.Pie(
                            labels=labels, values=values, hole=0.4,
                            hoverinfo="label+percent+value", textinfo="percent",
                            marker=dict(colors=colors),
                        )],
                        layout=go.Layout(
                            title=title, width=370, height=600,
                            margin=dict(t=50, b=120),
                            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.3),
                        ),
                    ),
                    auto_open=False, output_type="div",
                )

            for label, d in sections:
                if d.empty or "Profit" not in d.columns or "Symbol" not in d.columns:
                    continue
                # Win/Loss counts per symbol
                win_loss = d.groupby("Symbol")["Profit"].apply(
                    lambda x: pd.Series({"Wins": (x > 0).sum(), "Losses": (x <= 0).sum()})
                ).unstack().fillna(0)
                chart_pie_sections[f"{label.lower()}_count"] = build_pie(
                    f"{label} Win/Loss Count by Pair",
                    labels=[f"{s} Wins" for s in win_loss.index] + [f"{s} Losses" for s in win_loss.index],
                    values=list(win_loss.get("Wins", pd.Series())) + list(win_loss.get("Losses", pd.Series())),
                    colors=px.colors.qualitative.Set3,
                )

    # Files uploaded by the user (paginate 5 per page)
    files_qs = TradingFile.objects.filter(user=request.user).order_by("-uploaded_at")
    files = Paginator(files_qs, 5).get_page(request.GET.get("page"))

    return render(request, "performance/dashboard.html", {
        "df_html": df_html,
        "kpis": kpis,
        "chart_equity": chart_equity,
        "chart_profit": chart_profit,
        "chart_hist": chart_hist,
        "chart_month": chart_month,
        "chart_pie_sections": chart_pie_sections,
        "filter_form": filter_form,
        "files": files,
        "trade_page": trade_page,
    })


@login_required
def upload_file(request):
    """
    Upload CSV/XLSX, clean it, store cleaned JSON in session, update file status.
    """
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


@staff_member_required
def admin_all_files(request):
    all_files = TradingFile.objects.select_related("user").order_by("-uploaded_at")
    return render(request, "performance/admin_files.html", {"all_files": all_files})


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


@login_required
def download_cleaned_csv(request):
    cleaned_data = request.session.get("cleaned_data")
    if not cleaned_data:
        return HttpResponse("No data to download.", status=400)
    df = pd.read_json(cleaned_data, orient="split")
    csv_content = df.to_csv(index=False)
    response = HttpResponse(csv_content, content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="cleaned_trading_data.csv"'
    return response


@login_required
def download_excel(request):
    cleaned_data = request.session.get("cleaned_data")
    if not cleaned_data:
        return HttpResponse("No data to export.", status=400)
    df = pd.read_json(StringIO(cleaned_data), orient="split")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Trades")
    output.seek(0)
    response = HttpResponse(
        output,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = 'attachment; filename="trades.xlsx"'
    return response


@login_required
def download_pdf(request):
    cleaned_data = request.session.get("cleaned_data")
    if not cleaned_data:
        return HttpResponse("No data available.", status=400)
    df = pd.read_json(StringIO(cleaned_data), orient="split")
    kpis = compute_kpis(df) if df is not None else {}
    template = get_template("performance/pdf_report.html")
    html = template.render({"user": request.user, "kpis": kpis, "df": df, "now": timezone.now()})
    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="trading_report.pdf"'
    pisa_status = pisa.CreatePDF(html, dest=response)
    if pisa_status.err:
        return HttpResponse("Error generating PDF", status=500)
    return response


@login_required
def kpi_report(request):
    cleaned_data = request.session.get("cleaned_data")
    df = pd.read_json(StringIO(cleaned_data), orient="split") if cleaned_data else None
    kpis = compute_kpis(df) if df is not None else None
    return render(request, "performance/kpi_report.html", {"kpis": kpis})


@login_required
def export_excel(request):
    """
    Renders the Excel export UI (GET) and downloads Excel when `download=1`.
    Uses dummy rows for now (replace with DB queries later).
    """
    if "download" not in request.GET:
        columns = "entry_time,exit_time,symbol,side,qty,entry,exit,sl,tp,pnl,rr,tag,notes".split(",")
        selected_cols = request.GET.getlist("cols") if request.GET.getlist("cols") else columns
        return render(
            request,
            "performance/excel_export.html",
            {
                "columns": columns,
                "selected_cols": selected_cols,
            }
        )

    symbol = request.GET.get("symbol") or None
    min_rr = request.GET.get("min_rr") or None
    cols = request.GET.getlist("cols") or [
        "entry_time", "exit_time", "symbol", "side", "qty", "entry", "exit", "pnl", "rr", "tag", "notes"
    ]
    include_kpis = "include_kpis" in request.GET

    rows = [
        {"entry_time": "2025-07-01 09:00", "exit_time": "2025-07-01 11:00", "symbol": "XAUUSD", "side": "LONG",
         "qty": 1.0, "entry": 2350.0, "exit": 2360.0, "pnl": 100.0, "rr": 2.0, "tag": "Breakout", "notes": ""},
        {"entry_time": "2025-07-02 10:00", "exit_time": "2025-07-02 12:30", "symbol": "BTCUSD", "side": "SHORT",
         "qty": 0.2, "entry": 64000.0, "exit": 63500.0, "pnl": 100.0, "rr": 1.8, "tag": "Trend", "notes": "Nice move"},
    ]

    def passes_filters(r):
        if symbol and symbol.strip().upper() not in r["symbol"].upper():
            return False
        if min_rr:
            try:
                if float(r["rr"]) < float(min_rr):
                    return False
            except ValueError:
                pass
        return True

    filtered = [{k: v for k, v in r.items() if k in cols} for r in rows if passes_filters(r)]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df = pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=cols)
        df.to_excel(writer, index=False, sheet_name="Trades")

        if include_kpis:
            kpi_data = [
                {"Metric": "Win Rate", "Value": "—"},
                {"Metric": "Sharpe Ratio", "Value": "—"},
                {"Metric": "Max Drawdown", "Value": "—"},
                {"Metric": "Profit Factor", "Value": "—"},
            ]
            pd.DataFrame(kpi_data).to_excel(writer, index=False, sheet_name="KPIs")

    output.seek(0)
    response = HttpResponse(
        output.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = 'attachment; filename="tradeintel_report.xlsx"'
    return response


def project_one_plan(request):
    return render(request, "performance/project_one_plan.html")


# ---------- Helpers ----------

def clean_ftmo_csv(file_path: str) -> pd.DataFrame:
    """
    Reads and cleans a CSV/XLSX trading file (FTMO-like). Returns a cleaned DataFrame.
    """
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
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc, delimiter=delimiter, on_bad_lines="skip")
                break
            except UnicodeDecodeError:
                continue
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    df.dropna(how="all", inplace=True)
    df.columns = [c.strip() for c in df.columns]

    # parse common date columns
    for date_col in ["Open Time", "Open", "Date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            break

    # numeric
    for col in ["Size", "Profit", "Commission", "Swap", "Balance", "Pips", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import TradingFile  # Replace with your actual file model

@login_required
def admin_all_files(request):
    all_files = TradingFile.objects.all().order_by('-uploaded_at')
    return render(request, "performance/admin_files.html", {"all_files": all_files})
