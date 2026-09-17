import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase

from .models import TradingFile
from .utils import compute_kpis

User = get_user_model()


class ComputeKpisTests(TestCase):
    def test_compute_kpis_empty_dataframe(self):
        df = pd.DataFrame()

        result = compute_kpis(df)

        self.assertEqual(result, {})

    def test_compute_kpis_missing_profit_column(self):
        df = pd.DataFrame({"Symbol": ["XAUUSD", "BTCUSD"]})

        result = compute_kpis(df)

        self.assertEqual(result, {})

    def test_compute_kpis_all_wins(self):
        df = pd.DataFrame({"Profit": [10, 20, 30]})

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 3)
        self.assertEqual(result["Winning Trades"], 3)
        self.assertEqual(result["Losing Trades"], 0)
        self.assertEqual(result["Win Rate (%)"], "100.00")
        self.assertEqual(result["Gross Loss"], "0.00")
        self.assertEqual(result["Profit Factor"], "∞")

    def test_compute_kpis_all_losses(self):
        df = pd.DataFrame({"Profit": [-10, -20]})

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 2)
        self.assertEqual(result["Winning Trades"], 0)
        self.assertEqual(result["Losing Trades"], 2)
        self.assertEqual(result["Win Rate (%)"], "0.00")
        self.assertEqual(result["Total Profit"], "-30.00")
        self.assertEqual(result["Gross Loss"], "30.00")
        self.assertEqual(result["Profit Factor"], "0.00")

    def test_compute_kpis_mixed(self):
        df = pd.DataFrame({"Profit": [100, -50, 0]})

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 3)
        self.assertEqual(result["Winning Trades"], 1)
        self.assertEqual(result["Losing Trades"], 1)
        self.assertEqual(result["Break-even Trades"], 1)
        self.assertEqual(result["Win Rate (%)"], "33.33")
        self.assertEqual(result["Profit Factor"], "2.00")
        self.assertEqual(result["Expectancy"], "16.67")

    def test_compute_kpis_infinite_profit_factor(self):
        df = pd.DataFrame({"Profit": [50, 25]})

        result = compute_kpis(df)

        self.assertEqual(result["Gross Loss"], "0.00")
        self.assertEqual(result["Profit Factor"], "∞")

    def test_compute_kpis_single_trade_win(self):
        df = pd.DataFrame({"Profit": [10]})

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 1)
        self.assertEqual(result["Winning Trades"], 1)
        self.assertEqual(result["Sharpe"], "0.00")

    def test_compute_kpis_breakeven_excluded_from_win_loss(self):
        df = pd.DataFrame({"Profit": [0, 0]})

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 2)
        self.assertEqual(result["Winning Trades"], 0)
        self.assertEqual(result["Losing Trades"], 0)
        self.assertEqual(result["Break-even Trades"], 2)

    def test_compute_kpis_max_drawdown(self):
        df = pd.DataFrame({"Profit": [100, -40, -80, 50]})

        result = compute_kpis(df)

        self.assertEqual(result["Max Drawdown"], "120.00")

    def test_compute_kpis_non_numeric_profit_rows_are_dropped(self):
        df = pd.DataFrame({"Profit": ["invalid", 10]})

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 1)
        self.assertEqual(result["Total Profit"], "10.00")


class TradingFileTests(TestCase):
    def test_tradingfile_str(self):
        user = User.objects.create_user(
            username="testtrader",
            password="test-password-123",
        )

        trading_file = TradingFile.objects.create(
            user=user,
            file="trading_files/trades.csv",
            description="Test trade history",
            status="processed",
        )

        expected_date = trading_file.uploaded_at.strftime("%Y-%m-%d")
        expected = (
            f"testtrader - trading_files/trades.csv "
            f"({expected_date}) [processed]"
        )

        self.assertEqual(str(trading_file), expected)
# Create your tests here.
