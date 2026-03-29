import math
import pandas as pd


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value):
    return f"{_safe_float(value):.2f}"


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute post-trade KPI metrics from a trade-history DataFrame.

    Required column:
    - Profit

    Notes:
    - Break-even trades (Profit == 0) are excluded from wins and losses.
    - Sharpe and volatility are trade-based, using the ordered Profit series.
    - Max drawdown is calculated from cumulative Profit.
    """
    if df is None or df.empty or "Profit" not in df.columns:
        return {}

    working_df = df.copy()
    working_df["Profit"] = pd.to_numeric(working_df["Profit"], errors="coerce")
    working_df = working_df.dropna(subset=["Profit"]).copy()

    if working_df.empty:
        return {}

    profit = working_df["Profit"]

    total_trades = int(len(profit))
    winning_profit = profit[profit > 0]
    losing_profit = profit[profit < 0]
    breakeven_profit = profit[profit == 0]

    winning_trades = int(len(winning_profit))
    losing_trades = int(len(losing_profit))
    breakeven_trades = int(len(breakeven_profit))

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    total_profit = profit.sum()
    average_profit = profit.mean() if total_trades > 0 else 0.0

    gross_profit = winning_profit.sum() if not winning_profit.empty else 0.0
    gross_loss = abs(losing_profit.sum()) if not losing_profit.empty else 0.0

    average_win = winning_profit.mean() if winning_trades > 0 else 0.0
    average_loss = abs(losing_profit.mean()) if losing_trades > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = None if gross_profit == 0 else math.inf

    expectancy = average_profit if total_trades > 0 else 0.0

    best_trade = profit.max() if total_trades > 0 else 0.0
    worst_trade = profit.min() if total_trades > 0 else 0.0

    cumulative_profit = profit.cumsum()
    running_peak = cumulative_profit.cummax()
    drawdown = cumulative_profit - running_peak
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0

    volatility = profit.std(ddof=1) if total_trades > 1 else 0.0

    if volatility and volatility > 0:
        sharpe = average_profit / volatility
    else:
        sharpe = 0.0

    def fmt_special(value):
        if value is None:
            return "N/A"
        if value == math.inf:
            return "∞"
        return _fmt(value)

    return {
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Break-even Trades": breakeven_trades,
        "Win Rate (%)": _fmt(win_rate),
        "Total Profit": _fmt(total_profit),
        "Average Profit": _fmt(average_profit),
        "Gross Profit": _fmt(gross_profit),
        "Gross Loss": _fmt(gross_loss),
        "Average Win": _fmt(average_win),
        "Average Loss": _fmt(average_loss),
        "Profit Factor": fmt_special(profit_factor),
        "Expectancy": _fmt(expectancy),
        "Best Trade": _fmt(best_trade),
        "Worst Trade": _fmt(worst_trade),
        "Max Drawdown": _fmt(max_drawdown),
        "Sharpe": _fmt(sharpe),
        "Volatility": _fmt(volatility),
    }