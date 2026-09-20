import pandas as pd
from django.test import SimpleTestCase

from .analytics import apply_analysis


class AnalysisExperienceTests(SimpleTestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Open Time": pd.to_datetime(
                    [
                        "2026-09-01 10:00:00",
                        "2026-09-02 23:30:00",
                        "2026-09-03 08:00:00",
                        "2026-09-04 12:00:00",
                    ]
                ),
                "Symbol": [
                    "XAUUSD",
                    "EURUSD",
                    "XAUEUR",
                    "BTCUSD",
                ],
                "Type": [
                    "buy",
                    "sell",
                    "buy",
                    "sell",
                ],
                "Profit": [
                    100,
                    -50,
                    25,
                    10,
                ],
                "Notes": [
                    "breakout",
                    "reversal",
                    "continuation",
                    "range",
                ],
            }
        )

    def test_no_filters_preserve_all_rows(self):
        filtered, context = apply_analysis(
            self.df,
            {},
        )

        pd.testing.assert_frame_equal(
            filtered,
            self.df,
        )
        self.assertEqual(
            context.source_row_count,
            4,
        )
        self.assertEqual(
            context.filtered_row_count,
            4,
        )
        self.assertEqual(
            dict(context.active_filters),
            {},
        )
        self.assertFalse(
            context.is_zero_result
        )

    def test_start_date_filter_is_inclusive(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "start_date": "2026-09-03",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUEUR", "BTCUSD"],
        )
        self.assertEqual(
            dict(context.active_filters),
            {
                "start_date": "2026-09-03",
            },
        )

    def test_end_date_includes_full_calendar_day(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "end_date": "2026-09-02",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUUSD", "EURUSD"],
        )
        self.assertEqual(
            context.filtered_row_count,
            2,
        )

    def test_date_range_is_inclusive(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "start_date": "2026-09-02",
                "end_date": "2026-09-03",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["EURUSD", "XAUEUR"],
        )
        self.assertEqual(
            context.filtered_row_count,
            2,
        )

    def test_open_time_has_date_column_precedence(self):
        dataframe = self.df.copy(deep=True)

        dataframe["Open"] = pd.to_datetime(
            ["2026-01-01"] * 4
        )
        dataframe["Date"] = pd.to_datetime(
            ["2027-01-01"] * 4
        )

        filtered, _ = apply_analysis(
            dataframe,
            {
                "start_date": "2026-09-03",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUEUR", "BTCUSD"],
        )

    def test_symbol_filter_is_case_insensitive_substring(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "symbol": "xau",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUUSD", "XAUEUR"],
        )
        self.assertEqual(
            dict(context.active_filters),
            {
                "symbol": "xau",
            },
        )

    def test_combined_date_and_symbol_filters(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "start_date": "2026-09-02",
                "symbol": "xau",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUEUR"],
        )
        self.assertEqual(
            context.filtered_row_count,
            1,
        )

    def test_smart_search_matches_text_columns(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "q": "buy",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUUSD", "XAUEUR"],
        )
        self.assertEqual(
            dict(context.active_filters),
            {
                "q": "buy",
            },
        )

    def test_smart_search_matches_numeric_values(self):
        filtered, _ = apply_analysis(
            self.df,
            {
                "q": "25",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUEUR"],
        )

    def test_smart_search_matches_date_values(self):
        filtered, _ = apply_analysis(
            self.df,
            {
                "q": "2026-09-03",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUEUR"],
        )

    def test_invalid_date_filter_is_ignored(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "start_date": "not-a-date",
            },
        )

        self.assertEqual(
            len(filtered),
            4,
        )
        self.assertEqual(
            dict(context.active_filters),
            {},
        )

    def test_invalid_date_does_not_suppress_valid_symbol(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "start_date": "invalid",
                "symbol": "xau",
            },
        )

        self.assertEqual(
            list(filtered["Symbol"]),
            ["XAUUSD", "XAUEUR"],
        )
        self.assertEqual(
            dict(context.active_filters),
            {
                "symbol": "xau",
            },
        )

    def test_date_filter_is_skipped_when_date_column_missing(self):
        dataframe = self.df.drop(
            columns=["Open Time"]
        )

        filtered, context = apply_analysis(
            dataframe,
            {
                "start_date": "2026-09-03",
            },
        )

        self.assertEqual(
            len(filtered),
            4,
        )
        self.assertEqual(
            dict(context.active_filters),
            {},
        )

    def test_symbol_filter_is_skipped_when_symbol_column_missing(self):
        dataframe = self.df.drop(
            columns=["Symbol"]
        )

        filtered, context = apply_analysis(
            dataframe,
            {
                "symbol": "xau",
            },
        )

        self.assertEqual(
            len(filtered),
            4,
        )
        self.assertEqual(
            dict(context.active_filters),
            {},
        )

    def test_apply_analysis_does_not_mutate_input_dataframe(self):
        original = self.df.copy(deep=True)

        apply_analysis(
            self.df,
            {
                "start_date": "2026-09-02",
                "symbol": "xau",
                "q": "buy",
            },
        )

        pd.testing.assert_frame_equal(
            self.df,
            original,
        )

    def test_zero_result_context_reconciles_row_counts(self):
        filtered, context = apply_analysis(
            self.df,
            {
                "symbol": "NO-SUCH-SYMBOL",
            },
        )

        self.assertTrue(
            filtered.empty
        )
        self.assertEqual(
            context.source_row_count,
            4,
        )
        self.assertEqual(
            context.filtered_row_count,
            0,
        )
        self.assertTrue(
            context.is_zero_result
        )

    def test_posix_source_path_is_reduced_to_basename(self):
        _, context = apply_analysis(
            self.df,
            {},
            source_filename=(
                "/media/trading_files/"
                "user_4/history.csv"
            ),
        )

        self.assertEqual(
            context.source_filename,
            "history.csv",
        )
        self.assertNotIn(
            "/",
            context.source_filename,
        )

    def test_windows_source_path_is_reduced_to_basename(self):
        _, context = apply_analysis(
            self.df,
            {},
            source_filename=(
                r"C:\media\trading_files"
                r"\user_4\history.csv"
            ),
        )

        self.assertEqual(
            context.source_filename,
            "history.csv",
        )
        self.assertNotIn(
            "\\",
            context.source_filename,
        )

    def test_date_column_is_normalised_without_date_filter(self):
        dataframe = self.df.copy(deep=True)
        dataframe["Open Time"] = (
            dataframe["Open Time"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        original = dataframe.copy(deep=True)

        filtered, context = apply_analysis(
            dataframe,
            {},
        )

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                filtered["Open Time"]
            )
        )
        self.assertEqual(
            context.filtered_row_count,
            4,
        )

        pd.testing.assert_frame_equal(
            dataframe,
            original,
        )
