# TradeIntel 360 — Trading Performance Analytics

**TradeIntel 360** is a post-trade performance analytics data product built with **Python, Django, Pandas, and Plotly**. It turns uploaded trade history into KPI-driven review, dashboard-based analysis, report-style summary surfaces, and export-ready outputs for finance and trading workflows.

---

## What this project is

TradeIntel 360 is designed as a **post-trade analytics and review workflow**, not just a trading journal or a finance-themed Django app.

It is built to support:

- uploaded trade-history review
- KPI-driven performance analysis
- dashboard-based post-trade inspection
- trade-level review surfaces
- export-ready reporting outputs
- analyst-facing workflow presentation

---

## What this project is not

TradeIntel 360 is **not positioned mainly as**:

- a generic trading journal
- a market-data monitoring platform
- a pre-trade risk calculator suite
- a market-data API product
- a chart-only dashboard demo

Those responsibilities are deliberately separated from other projects in the wider portfolio.

---

## Portfolio positioning

**Analytics Engineer | Data Engineer | Python & Django | ETL, KPI Dashboards, FinTech & BI**

TradeIntel 360 is positioned as the **post-trade performance analytics** product in a broader FinTech portfolio:

- **DataBridge Market API** → upstream market-data ingestion, normalization, ETL, ops visibility, API delivery
- **MarketVista Dashboard** → market monitoring and analyst-facing market visibility
- **RiskWise Planner** → pre-trade risk planning and scenario analysis
- **TradeIntel 360** → post-trade performance analytics and review

---

## Core workflow

The strongest reviewer path is:

1. **Upload trade history**
2. **Review the performance dashboard**
3. **Open the KPI report**
4. **Use export surfaces for downstream reporting**

This keeps the project focused on post-trade analytics rather than mixed product identity.

---

## What this project proves

This project demonstrates:

- handling uploaded trading-history files
- cleaning and preparing uploaded data for review
- session-based analysis workflow from uploaded data
- KPI generation from loaded trade history
- dashboard-based performance inspection
- chart-based analysis surfaces
- trade-level review table with pagination
- filtered review workflow
- export surfaces for CSV, Excel, and PDF output
- premium reviewer-facing UI packaging in Django

---

## Proven features in the current repo

Based on the current implementation, the following are supported in the project:

- file upload for CSV/XLSX trade-history review
- session-backed cleaned data workflow
- dashboard filters for date and symbol
- smart search on the trade review table
- KPI summary surface
- chart outputs for:
  - equity curve
  - profit per trade
  - profit distribution histogram
  - monthly performance
  - optional segmented pie sections
- paginated trade review table
- paginated uploaded-file history
- KPI report page
- PDF report generation
- cleaned CSV export
- cleaned Excel export
- configurable Excel export surface

---

## KPI coverage

The project currently displays KPI outputs using the existing KPI computation utility.

**Proven in current UI outputs:**
- Total Trades
- Winning Trades
- Losing Trades
- Win Rate
- Total Profit
- Average Profit

**May be supported depending on the current KPI utility implementation:**
- gross profit
- gross loss
- average win
- average loss
- profit factor
- expectancy
- best trade
- worst trade
- max drawdown

**Unknown unless confirmed in the KPI utility:**
- Sharpe ratio
- volatility
- advanced risk-adjusted metrics

Only claim KPI definitions publicly if they are actually computed in the current codebase.

---

## Screens to review

The best screens for a hiring manager or recruiter are:

- **Home** — product identity and reviewer path
- **Upload Trade History** — entry point for loaded-data workflow
- **Performance Dashboard** — KPI and chart review surface
- **KPI Report** — structured summary surface
- **PDF / Excel / CSV exports** — reporting-oriented outputs

---

## How to review this project

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-project-folder>