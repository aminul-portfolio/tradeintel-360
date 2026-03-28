def compute_kpis(df):
    """
    Example KPI calculations from your dataframe.
    Adjust as you need.
    """
    total_trades = len(df)
    winning_trades = len(df[df['Profit'] > 0])
    losing_trades = len(df[df['Profit'] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    total_profit = df['Profit'].sum()
    average_profit = df['Profit'].mean() if total_trades > 0 else 0

    return {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate (%)': f"{win_rate:.2f}",
        'Total Profit': f"{total_profit:.2f}",
        'Average Profit': f"{average_profit:.2f}"
    }
