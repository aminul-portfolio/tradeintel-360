import pandas as pd
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

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

class DashboardAnalysisIntegrationTests(TestCase):
    def setUp(self):
        user_model = get_user_model()

        self.user = user_model.objects.create_user(
            username="analysis-user",
            email="analysis@example.com",
            password="test-password-123",
        )

        self.client.force_login(self.user)

        self.df = pd.DataFrame(
            {
                "Open Time": [
                    "2026-09-01 10:00:00",
                    "2026-09-02 11:00:00",
                    "2026-09-03 12:00:00",
                    "2026-09-04 13:00:00",
                ],
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

        session = self.client.session
        session["cleaned_data"] = self.df.to_json(
            orient="split",
            date_format="iso",
        )
        session["last_uploaded_file"] = (
            r"trading_files\user_4\history.csv"
        )
        session.save()

        self.dashboard_url = reverse(
            "performance:dashboard"
        )

    def test_dashboard_shared_analysis_controls_kpi_scope(self):
        response = self.client.get(
            self.dashboard_url,
            {
                "symbol": "xau",
                "q": "buy",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        context = response.context[
            "analysis_context"
        ]

        self.assertEqual(
            context.source_row_count,
            4,
        )
        self.assertEqual(
            context.filtered_row_count,
            2,
        )
        self.assertEqual(
            response.context["kpis"]["Total Trades"],
            2,
        )

        self.assertEqual(
            dict(context.active_filters),
            {
                "symbol": "xau",
                "q": "buy",
            },
        )

    def test_dashboard_analysis_context_is_recomputed_per_request(self):
        first_response = self.client.get(
            self.dashboard_url,
            {
                "symbol": "xau",
            },
        )

        first_context = first_response.context[
            "analysis_context"
        ]

        self.assertEqual(
            first_context.filtered_row_count,
            2,
        )

        second_response = self.client.get(
            self.dashboard_url
        )

        second_context = second_response.context[
            "analysis_context"
        ]

        self.assertEqual(
            second_context.source_row_count,
            4,
        )
        self.assertEqual(
            second_context.filtered_row_count,
            4,
        )
        self.assertEqual(
            dict(second_context.active_filters),
            {},
        )

        self.assertNotIn(
            "analysis_context",
            self.client.session,
        )

    def test_dashboard_exposes_basename_only_source_filename(self):
        response = self.client.get(
            self.dashboard_url
        )

        context = response.context[
            "analysis_context"
        ]

        self.assertEqual(
            context.source_filename,
            "history.csv",
        )

        self.assertContains(
            response,
            "history.csv",
        )

        self.assertNotContains(
            response,
            r"trading_files\user_4\history.csv",
        )

    def test_kpi_report_link_preserves_only_applied_analysis_filters(self):
        response = self.client.get(
            self.dashboard_url,
            {
                "start_date": "2026-09-01",
                "end_date": "2026-09-03",
                "symbol": "xau",
                "q": "buy",
                "file_q": "history",
                "file_status": "processed",
                "page": "3",
                "trade_page": "2",
            },
        )

        analysis_query = response.context[
            "analysis_query"
        ]

        self.assertEqual(
            analysis_query,
            (
                "start_date=2026-09-01"
                "&end_date=2026-09-03"
                "&symbol=xau"
                "&q=buy"
            ),
        )

        expected_href = (
            reverse("performance:kpi_report")
            + "?"
            + analysis_query
        )

        expected_rendered_href = expected_href.replace("&", "&amp;")

        self.assertContains(
            response,
            f'href="{expected_rendered_href}"',
            html=False,
        )

        self.assertNotIn(
            "file_q",
            analysis_query,
        )
        self.assertNotIn(
            "file_status",
            analysis_query,
        )
        self.assertNotIn(
            "page=",
            analysis_query,
        )
        self.assertNotIn(
            "trade_page",
            analysis_query,
        )

    def test_dashboard_zero_result_preserves_analysis_lineage(self):
        response = self.client.get(
            self.dashboard_url,
            {
                "symbol": "NO-SUCH-SYMBOL",
            },
        )

        context = response.context[
            "analysis_context"
        ]

        self.assertEqual(
            context.source_row_count,
            4,
        )
        self.assertEqual(
            context.filtered_row_count,
            0,
        )
        self.assertTrue(
            context.is_zero_result,
        )

        self.assertEqual(
            response.context["kpis"],
            {},
        )

        self.assertContains(
            response,
            "No trades match the current analysis scope",
        )

    def test_trade_pagination_query_preserves_analysis_filters(self):
        response = self.client.get(
            self.dashboard_url,
            {
                "start_date": "2026-09-01",
                "symbol": "usd",
                "q": "buy",
                "trade_page": "2",
            },
        )

        trade_query = response.context[
            "trade_query"
        ]

        self.assertIn(
            "start_date=2026-09-01",
            trade_query,
        )
        self.assertIn(
            "symbol=usd",
            trade_query,
        )
        self.assertIn(
            "q=buy",
            trade_query,
        )
        self.assertNotIn(
            "trade_page",
            trade_query,
        )
