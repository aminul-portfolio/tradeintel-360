import os
import tempfile

import pandas as pd
from django.test import SimpleTestCase

from .ingestion import clean_ftmo_csv
from .utils import compute_kpis


class MaxDrawdownEvidenceTests(SimpleTestCase):
    def test_loss_leading_sequence_includes_zero_baseline(self):
        df = pd.DataFrame(
            {
                "Profit": [-50, 100, 0, 25],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(
            result["Max Drawdown"],
            "50.00",
        )

    def test_all_loss_sequence_measures_drawdown_from_zero(self):
        df = pd.DataFrame(
            {
                "Profit": [-10, -20],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(
            result["Max Drawdown"],
            "30.00",
        )

class CanonicalKpiEvidenceTests(SimpleTestCase):
    def test_canonical_fixture_reconciles_all_18_kpis(self):
        df = pd.DataFrame(
            {
                "Profit": [100, -50, 0, 25],
            }
        )

        result = compute_kpis(df)

        expected = {
            "Total Trades": 4,
            "Winning Trades": 2,
            "Losing Trades": 1,
            "Break-even Trades": 1,
            "Win Rate (%)": "50.00",
            "Total Profit": "75.00",
            "Average Profit": "18.75",
            "Gross Profit": "125.00",
            "Gross Loss": "50.00",
            "Average Win": "62.50",
            "Average Loss": "50.00",
            "Profit Factor": "2.50",
            "Expectancy": "18.75",
            "Best Trade": "100.00",
            "Worst Trade": "-50.00",
            "Max Drawdown": "50.00",
            "Sharpe": "0.30",
            "Volatility": "62.50",
        }

        self.assertEqual(result, expected)

    def test_kpi_key_order_and_output_types_are_stable(self):
        df = pd.DataFrame(
            {
                "Profit": [100, -50, 0, 25],
            }
        )

        result = compute_kpis(df)

        expected_keys = [
            "Total Trades",
            "Winning Trades",
            "Losing Trades",
            "Break-even Trades",
            "Win Rate (%)",
            "Total Profit",
            "Average Profit",
            "Gross Profit",
            "Gross Loss",
            "Average Win",
            "Average Loss",
            "Profit Factor",
            "Expectancy",
            "Best Trade",
            "Worst Trade",
            "Max Drawdown",
            "Sharpe",
            "Volatility",
        ]

        self.assertEqual(
            list(result.keys()),
            expected_keys,
        )

        count_keys = [
            "Total Trades",
            "Winning Trades",
            "Losing Trades",
            "Break-even Trades",
        ]

        for key in count_keys:
            self.assertIsInstance(result[key], int)

        for key in expected_keys:
            if key not in count_keys:
                self.assertIsInstance(result[key], str)

class KpiSemanticEvidenceTests(SimpleTestCase):
    def test_win_rate_includes_breakeven_in_denominator(self):
        df = pd.DataFrame(
            {
                "Profit": [10, 0],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Winning Trades"], 1)
        self.assertEqual(result["Break-even Trades"], 1)
        self.assertEqual(result["Win Rate (%)"], "50.00")

    def test_all_winners_semantics(self):
        df = pd.DataFrame(
            {
                "Profit": [10, 20, 30],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 3)
        self.assertEqual(result["Winning Trades"], 3)
        self.assertEqual(result["Losing Trades"], 0)
        self.assertEqual(result["Break-even Trades"], 0)
        self.assertEqual(result["Win Rate (%)"], "100.00")
        self.assertEqual(result["Gross Profit"], "60.00")
        self.assertEqual(result["Gross Loss"], "0.00")

    def test_all_losses_semantics(self):
        df = pd.DataFrame(
            {
                "Profit": [-10, -20],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 2)
        self.assertEqual(result["Winning Trades"], 0)
        self.assertEqual(result["Losing Trades"], 2)
        self.assertEqual(result["Break-even Trades"], 0)
        self.assertEqual(result["Win Rate (%)"], "0.00")
        self.assertEqual(result["Gross Profit"], "0.00")
        self.assertEqual(result["Gross Loss"], "30.00")

    def test_all_breakeven_semantics(self):
        df = pd.DataFrame(
            {
                "Profit": [0, 0],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 2)
        self.assertEqual(result["Winning Trades"], 0)
        self.assertEqual(result["Losing Trades"], 0)
        self.assertEqual(result["Break-even Trades"], 2)
        self.assertEqual(result["Win Rate (%)"], "0.00")
        self.assertEqual(result["Total Profit"], "0.00")

    def test_single_trade_has_zero_volatility_and_sharpe(self):
        df = pd.DataFrame(
            {
                "Profit": [10],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Volatility"], "0.00")
        self.assertEqual(result["Sharpe"], "0.00")

    def test_invalid_profit_rows_are_excluded(self):
        df = pd.DataFrame(
            {
                "Profit": ["invalid", 10, None],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Total Trades"], 1)
        self.assertEqual(result["Total Profit"], "10.00")

    def test_all_invalid_profit_returns_empty_result(self):
        df = pd.DataFrame(
            {
                "Profit": ["invalid", None],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result, {})

    def test_profit_factor_finite_ratio(self):
        df = pd.DataFrame(
            {
                "Profit": [100, -25],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Profit Factor"], "4.00")

    def test_profit_factor_infinity_when_no_losses(self):
        df = pd.DataFrame(
            {
                "Profit": [10, 20],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Profit Factor"], "∞")

    def test_profit_factor_na_when_no_gains_or_losses(self):
        df = pd.DataFrame(
            {
                "Profit": [0, 0],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Profit Factor"], "N/A")

    def test_gross_profit_and_average_win(self):
        df = pd.DataFrame(
            {
                "Profit": [100, 50, -25],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Gross Profit"], "150.00")
        self.assertEqual(result["Average Win"], "75.00")

    def test_gross_loss_and_average_loss_are_positive_magnitudes(self):
        df = pd.DataFrame(
            {
                "Profit": [100, -50, -25],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Gross Loss"], "75.00")
        self.assertEqual(result["Average Loss"], "37.50")

    def test_expectancy_matches_average_profit(self):
        df = pd.DataFrame(
            {
                "Profit": [100, -50, 0, 25],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Expectancy"], "18.75")
        self.assertEqual(
            result["Expectancy"],
            result["Average Profit"],
        )

    def test_best_and_worst_trade(self):
        df = pd.DataFrame(
            {
                "Profit": [12, -7, 3],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Best Trade"], "12.00")
        self.assertEqual(result["Worst Trade"], "-7.00")

    def test_trade_order_changes_max_drawdown(self):
        first_order = pd.DataFrame(
            {
                "Profit": [100, -80, -30, 50],
            }
        )
        second_order = pd.DataFrame(
            {
                "Profit": [-80, 100, -30, 50],
            }
        )

        first_result = compute_kpis(first_order)
        second_result = compute_kpis(second_order)

        self.assertEqual(
            first_result["Max Drawdown"],
            "110.00",
        )
        self.assertEqual(
            second_result["Max Drawdown"],
            "80.00",
        )
        self.assertNotEqual(
            first_result["Max Drawdown"],
            second_result["Max Drawdown"],
        )

    def test_constant_profit_sequence_has_zero_volatility_and_sharpe(self):
        df = pd.DataFrame(
            {
                "Profit": [10, 10, 10],
            }
        )

        result = compute_kpis(df)

        self.assertEqual(result["Volatility"], "0.00")
        self.assertEqual(result["Sharpe"], "0.00")

    def test_compute_kpis_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame(
            {
                "Profit": ["10", "-5", "invalid"],
                "Symbol": ["EURUSD", "XAUUSD", "BTCUSD"],
            }
        )
        original = df.copy(deep=True)

        compute_kpis(df)

        pd.testing.assert_frame_equal(
            df,
            original,
        )

    def test_formatted_numeric_values_use_two_decimal_strings(self):
        df = pd.DataFrame(
            {
                "Profit": [100, -50, 0, 25],
            }
        )

        result = compute_kpis(df)

        count_keys = {
            "Total Trades",
            "Winning Trades",
            "Losing Trades",
            "Break-even Trades",
        }

        for key, value in result.items():
            if key in count_keys:
                continue

            self.assertIsInstance(value, str)
            self.assertRegex(
                value,
                r"^-?\d+\.\d{2}$",
            )

class IngestionKpiEvidenceTests(SimpleTestCase):
    def test_csv_ingestion_reconciles_to_expected_kpis(self):
        handle = tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        )
        file_path = handle.name

        try:
            handle.write(
                (
                    b"Symbol,Open Time,Profit,Commissions\n"
                    b"EURUSD,2026-09-01 10:00:00,100,-1\n"
                    b"XAUUSD,2026-09-02 10:00:00,-50,-2\n"
                    b"GBPUSD,2026-09-03 10:00:00,0,-1\n"
                    b"BTCUSD,2026-09-04 10:00:00,25,-3\n"
                )
            )
            handle.close()

            df = clean_ftmo_csv(file_path)
            result = compute_kpis(df)

            expected = {
                "Total Trades": 4,
                "Winning Trades": 2,
                "Losing Trades": 1,
                "Break-even Trades": 1,
                "Win Rate (%)": "50.00",
                "Total Profit": "75.00",
                "Average Profit": "18.75",
                "Gross Profit": "125.00",
                "Gross Loss": "50.00",
                "Average Win": "62.50",
                "Average Loss": "50.00",
                "Profit Factor": "2.50",
                "Expectancy": "18.75",
                "Best Trade": "100.00",
                "Worst Trade": "-50.00",
                "Max Drawdown": "50.00",
                "Sharpe": "0.30",
                "Volatility": "62.50",
            }

            self.assertIn("Commission", df.columns)
            self.assertNotIn("Commissions", df.columns)
            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(
                    df["Open Time"]
                )
            )
            self.assertEqual(result, expected)

        finally:
            if not handle.closed:
                handle.close()
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_xlsx_ingestion_reconciles_to_expected_kpis(self):
        handle = tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            delete=False,
        )
        file_path = handle.name
        handle.close()

        try:
            source_df = pd.DataFrame(
                {
                    "Symbol": [
                        "EURUSD",
                        "XAUUSD",
                        "GBPUSD",
                        "BTCUSD",
                    ],
                    "Open Time": [
                        "2026-09-01 10:00:00",
                        "2026-09-02 10:00:00",
                        "2026-09-03 10:00:00",
                        "2026-09-04 10:00:00",
                    ],
                    "Profit": [
                        100,
                        -50,
                        0,
                        25,
                    ],
                    "Commission": [
                        -1,
                        -2,
                        -1,
                        -3,
                    ],
                }
            )

            source_df.to_excel(
                file_path,
                index=False,
                engine="openpyxl",
            )

            df = clean_ftmo_csv(file_path)
            result = compute_kpis(df)

            expected = {
                "Total Trades": 4,
                "Winning Trades": 2,
                "Losing Trades": 1,
                "Break-even Trades": 1,
                "Win Rate (%)": "50.00",
                "Total Profit": "75.00",
                "Average Profit": "18.75",
                "Gross Profit": "125.00",
                "Gross Loss": "50.00",
                "Average Win": "62.50",
                "Average Loss": "50.00",
                "Profit Factor": "2.50",
                "Expectancy": "18.75",
                "Best Trade": "100.00",
                "Worst Trade": "-50.00",
                "Max Drawdown": "50.00",
                "Sharpe": "0.30",
                "Volatility": "62.50",
            }

            self.assertIn("Commission", df.columns)
            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(
                    df["Open Time"]
                )
            )
            self.assertEqual(result, expected)

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
