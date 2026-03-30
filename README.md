# TradeIntel 360

**Post-trade performance analytics built with Python, Django, Pandas, Plotly, and openpyxl.**

Upload a trade history CSV [Comma-Separated Values] or Excel [Microsoft Excel] file and TradeIntel 360 cleans it, computes KPI [Key Performance Indicator] outputs, and surfaces the results across an interactive dashboard, a structured report view, and configurable export outputs. The review workflow runs from a single uploaded file and is designed to minimise manual preparation before analysis.

![Performance Dashboard](docs/screenshots/04.1_performance_dashboard.png)

---

## What this demonstrates

| Area | Detail |
|---|---|
| **Data workflow** | Ingest raw CSV/XLSX → clean → normalise → session-backed analysis workflow |
| **KPI engine** | 17 computed metrics from uploaded data, including win rate, profit factor, Sharpe, max drawdown, and expectancy |
| **Django patterns** | Login-gated views, file upload handling, session state, paginated tables, context-driven reporting views |
| **Data visualisation** | Plotly-rendered equity curve, P&L distribution, monthly performance, and segmented win/loss views |
| **Export workflow** | PDF generation via xhtml2pdf, configurable Excel export with optional KPI sheet via openpyxl, and cleaned CSV export |
| **FinTech domain** | Trade-level data structures, performance metrics, and analyst-facing review workflow presentation |

---

## Core workflow

```text
Upload CSV / XLSX
       ↓
Clean and normalise trade history
       ↓
Store cleaned DataFrame in session
       ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Dashboard   │  KPI Report  │ Excel Export │  PDF Report  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

The workflow is session-driven: upload once, review across all surfaces without re-uploading. Dashboard filters (date range, symbol) update the active dataset and recompute KPIs in real time.

---

## Screenshots

<table>
<tr>
<td><img src="docs/screenshots/02_home_reviewer_path.png" alt="Reviewer path" width="320"/><br><em>Reviewer onboarding path</em></td>
<td><img src="docs/screenshots/03_upload_trade_history.png" alt="Upload" width="320"/><br><em>Trade history upload</em></td>
</tr>
<tr>
<td><img src="docs/screenshots/04.2_performance_dashboard.png" alt="Charts" width="320"/><br><em>Dashboard charts</em></td>
<td><img src="docs/screenshots/05_kpi_report.png" alt="KPI report" width="320"/><br><em>KPI report</em></td>
</tr>
<tr>
<td><img src="docs/screenshots/06_excel_export_configuration.png" alt="Excel export" width="320"/><br><em>Configurable Excel export</em></td>
<td><img src="docs/screenshots/07_trade_review_table_optional.png" alt="Trade table" width="320"/><br><em>Trade review table with search and pagination</em></td>
</tr>
</table>

---

## KPI engine

All metrics are computed from the loaded dataset. Applying a date, symbol, or RR filter recomputes the full suite against the filtered subset.

**Volume & outcome**
- Total trades, wins, losses, break-evens, win rate

**Profit & loss**
- Total profit, average profit, gross profit, gross loss
- Average win, average loss, profit factor

**Risk metrics**
- Expectancy, best trade, worst trade, max drawdown

**Statistical**
- Trade-based Sharpe ratio, per-trade profit volatility

> Note: Sharpe is computed as a trade-series ratio, not an annualised institutional Sharpe. Volatility refers to per-trade profit dispersion.

---

## Export surfaces

**PDF report** — full KPI summary rendered via xhtml2pdf, ready to share or archive.

**Cleaned CSV export** — normalised version of the uploaded dataset.

**Configurable Excel export** — built with openpyxl with the following options:
- column selection
- date range filter
- symbol filter
- minimum RR filter (where RR data is present)
- optional KPI summary sheet
- export metadata sheet

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Django 5 |
| Data processing | Pandas, openpyxl |
| Visualisation | Plotly |
| PDF generation | xhtml2pdf |
| Auth | Django auth, session management |
| Database | SQLite (local), PostgreSQL-ready |
| UI | Django templates, Bootstrap |

---

## Local setup

```bash
git clone https://github.com/aminul-portfolio/tradeintel-360.git
cd tradeintel-360

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then visit `http://127.0.0.1:8000`, log in, and upload a trade history file to begin the review workflow.

**Expected input format:** CSV or XLSX with a `Profit` column and common trade fields such as date/time, symbol, and side/type.

---

## Review checklist

- [ ] Upload a CSV or XLSX file
- [ ] Confirm cleaned dataset loads into the dashboard
- [ ] Apply date and symbol filters — observe KPIs recompute
- [ ] Open the KPI report
- [ ] Configure and download the Excel export with optional KPI sheet
- [ ] Generate the PDF report

---

## Portfolio context

TradeIntel 360 is the **post-trade analytics** product in a four-project FinTech portfolio, each covering a distinct domain:

| Project | Domain |
|---|---|
| DataBridge Market API | Market data ingestion, ETL, API delivery |
| MarketVista Dashboard | Market monitoring and analyst visibility |
| RiskWise Planner | Pre-trade risk planning and scenario modelling |
| **TradeIntel 360** | **Post-trade performance analytics and review** |

---

## Target roles

This project is best aligned to:

- Data Analyst [Finance / Trading]
- Analytics Engineer [FinTech]
- BI / Reporting Analyst
- Python/Django data-product roles
- Performance reporting and trade-review workflows in finance environments
