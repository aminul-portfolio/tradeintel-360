# TradeIntel 360 — Trading Performance Analytics

**TradeIntel 360** is a post-trade performance analytics data product built with **Python, Django, Pandas, Plotly, and openpyxl**. It turns uploaded trade history into KPI-driven review, dashboard-based analysis, report-style summary surfaces, and export-ready outputs for finance and trading workflows.

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

## Overview

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
- trade-level review tables with filtering and pagination
- export-oriented reporting workflow
- reviewer-facing UI packaging in Django

---

## Proven features in the current repo

Based on the current implementation, the project supports:

- CSV/XLSX trade-history upload
- session-backed cleaned data workflow
- dashboard filters for date and symbol
- smart search across the trade review table
- KPI summary outputs from the loaded dataset
- chart outputs for:
  - equity curve
  - profit per trade
  - profit distribution histogram
  - monthly performance
  - segmented win/loss pie sections where relevant fields are present
- paginated trade review table
- paginated uploaded-file history
- KPI report page
- PDF report generation
- cleaned CSV export
- cleaned Excel export
- configurable Excel export with:
  - selected-column export
  - date filtering
  - symbol filtering
  - minimum RR filtering where RR data exists
  - optional KPI sheet
  - export metadata sheet

---

## KPI definitions

TradeIntel 360 currently computes post-trade KPI outputs from the loaded trade dataset.

### Current KPI coverage

- **Total Trades** — count of rows with valid `Profit` values
- **Winning Trades** — trades where `Profit > 0`
- **Losing Trades** — trades where `Profit < 0`
- **Break-even Trades** — trades where `Profit = 0`
- **Win Rate (%)** — winning trades divided by total trades
- **Total Profit** — sum of `Profit`
- **Average Profit** — average `Profit` per trade
- **Gross Profit** — sum of positive `Profit` values
- **Gross Loss** — absolute sum of negative `Profit` values
- **Average Win** — average positive `Profit`
- **Average Loss** — absolute average negative `Profit`
- **Profit Factor** — gross profit divided by gross loss
- **Expectancy** — currently represented as average profit per trade
- **Best Trade** — maximum `Profit`
- **Worst Trade** — minimum `Profit`
- **Max Drawdown** — maximum peak-to-trough decline from cumulative profit
- **Sharpe** — currently a trade-based ratio derived from average profit and profit volatility
- **Volatility** — currently the standard deviation of per-trade profit

### KPI notes

- KPI outputs are computed from the currently loaded dataset.
- Date, symbol, and export filters can change the dataset being summarized.
- Sharpe in this project should be read as a **trade-based implementation**, not an annualized institutional Sharpe.
- Volatility in this project refers to **per-trade profit dispersion**, not market price volatility.

---

## How to review this project

### 1. Clone the repo

```bash
git clone https://github.com/aminul-portfolio/tradeintel-360.git
cd tradeintel-360