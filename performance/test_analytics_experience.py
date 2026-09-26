import re
from io import BytesIO

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
        self.export_url = reverse(
            "performance:export_excel"
        )

    def _load_export_session(self):
        export_df = pd.DataFrame(
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
                "RR": [
                    2.0,
                    1.0,
                    1.5,
                    3.0,
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
        session["cleaned_data"] = export_df.to_json(
            orient="split",
            date_format="iso",
        )
        session["last_uploaded_file"] = (
            r"trading_files\user_4\history.csv"
        )
        session.save()
        return export_df

    def _read_export_workbook(self, response):
        workbook = pd.ExcelFile(BytesIO(response.content))
        return {
            name: pd.read_excel(workbook, sheet_name=name)
            for name in workbook.sheet_names
        }

    def _export_meta_map(self, sheets):
        meta = sheets["ExportMeta"]
        return {
            str(field): value
            for field, value in zip(
                meta["Field"],
                meta["Value"],
            )
        }

    def _normalise_kpi_value(self, value):
        if value is None or (
            isinstance(value, float)
            and pd.isna(value)
        ):
            return ""
        text = str(value).strip()
        try:
            number = float(text.replace(",", ""))
            return (
                f"{number:.6f}"
                .rstrip("0")
                .rstrip(".")
            )
        except ValueError:
            return text

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

    def test_kpi_report_shared_analysis_controls_kpi_scope(self):
        response = self.client.get(
            reverse("performance:kpi_report"),
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

    def test_dashboard_and_kpi_report_have_exact_analysis_parity(self):
        params = {
            "start_date": "2026-09-01",
            "end_date": "2026-09-03",
            "symbol": "xau",
            "q": "buy",
        }

        dashboard_response = self.client.get(
            self.dashboard_url,
            params,
        )
        report_response = self.client.get(
            reverse("performance:kpi_report"),
            params,
        )

        self.assertEqual(
            dashboard_response.context["kpis"],
            report_response.context["kpis"],
        )
        self.assertEqual(
            dashboard_response.context[
                "analysis_context"
            ].filtered_row_count,
            report_response.context[
                "analysis_context"
            ].filtered_row_count,
        )

    def test_kpi_q_filters_display_only_not_analysis(self):
        analytical_params = {
            "symbol": "xau",
            "q": "buy",
        }

        without_kpi_q = self.client.get(
            reverse("performance:kpi_report"),
            analytical_params,
        )
        with_kpi_q = self.client.get(
            reverse("performance:kpi_report"),
            {
                **analytical_params,
                "kpi_q": "profit",
            },
        )

        self.assertEqual(
            without_kpi_q.context["kpis"],
            with_kpi_q.context["kpis"],
        )
        self.assertEqual(
            without_kpi_q.context[
                "analysis_context"
            ].filtered_row_count,
            with_kpi_q.context[
                "analysis_context"
            ].filtered_row_count,
        )
        self.assertEqual(
            dict(
                without_kpi_q.context[
                    "analysis_context"
                ].active_filters
            ),
            dict(
                with_kpi_q.context[
                    "analysis_context"
                ].active_filters
            ),
        )
        self.assertNotIn(
            "kpi_q",
            with_kpi_q.context[
                "analysis_context"
            ].active_filters,
        )

        without_rows = list(
            without_kpi_q.context[
                "kpi_page"
            ].object_list
        )
        with_rows = list(
            with_kpi_q.context[
                "kpi_page"
            ].object_list
        )

        self.assertNotEqual(
            without_rows,
            with_rows,
        )
        self.assertLess(
            len(with_rows),
            len(without_rows),
        )
        self.assertTrue(
            all(
                "profit" in str(row["metric"]).lower()
                or "profit" in str(row["value"]).lower()
                for row in with_rows
            )
        )

        self.assertContains(
            with_kpi_q,
            'name="symbol"',
        )
        self.assertContains(
            with_kpi_q,
            'value="xau"',
        )
        self.assertContains(
            with_kpi_q,
            'name="q"',
        )
        self.assertContains(
            with_kpi_q,
            'value="buy"',
        )

        analysis_query = with_kpi_q.context[
            "analysis_query"
        ]
        expected_href = (
            reverse("performance:dashboard")
            + "?"
            + analysis_query
        )
        expected_rendered_href = expected_href.replace(
            "&",
            "&amp;",
        )

        self.assertContains(
            with_kpi_q,
            f'href="{expected_rendered_href}"',
            html=False,
        )
        self.assertNotIn(
            "kpi_q",
            analysis_query,
        )
        self.assertNotIn(
            "kpi_page",
            analysis_query,
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

    def test_kpi_report_zero_result_preserves_analysis_lineage(self):
        response = self.client.get(
            reverse("performance:kpi_report"),
            {
                "symbol": "NO-SUCH-SYMBOL",
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
        self.assertNotContains(
            response,
            "Upload trade history first",
        )

    def test_kpi_report_analysis_context_is_recomputed_per_request(self):
        first_response = self.client.get(
            reverse("performance:kpi_report"),
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
            reverse("performance:kpi_report")
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

    def test_kpi_report_exposes_basename_only_source_filename(self):
        response = self.client.get(
            reverse("performance:kpi_report")
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

    def test_export_preview_uses_shared_analysis_scope(self):
        self._load_export_session()

        response = self.client.get(
            self.export_url,
            {
                "preview": "1",
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
            dict(context.active_filters),
            {
                "symbol": "xau",
                "q": "buy",
            },
        )
        self.assertEqual(
            response.context["exported_row_count"],
            2,
        )
        self.assertContains(
            response,
            'name="q"',
        )
        self.assertContains(
            response,
            'value="buy"',
        )
        self.assertNotIn(
            "min_rr",
            context.active_filters,
        )

    def test_export_min_rr_is_export_only_refinement(self):
        self._load_export_session()

        response = self.client.get(
            self.export_url,
            {
                "preview": "1",
                "symbol": "xau",
                "q": "buy",
                "min_rr": "1.75",
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
            2,
        )
        self.assertEqual(
            response.context["exported_row_count"],
            1,
        )
        self.assertNotIn(
            "min_rr",
            context.active_filters,
        )
        self.assertTrue(
            response.context["min_rr_applied"]
        )
        self.assertEqual(
            response.context["applied_min_rr"],
            "1.75",
        )
        self.assertContains(
            response,
            "Shared Analysis Filters",
        )
        self.assertContains(
            response,
            "Export-only Min R/R",
        )
        self.assertContains(
            response,
            "1.75",
        )
        self.assertContains(
            response,
            "Analysed rows: 2",
        )
        self.assertContains(
            response,
            "Exported rows: 1",
        )

    def test_export_workbook_metadata_reconciles_row_lineage(self):
        self._load_export_session()

        response = self.client.get(
            self.export_url,
            {
                "download": "1",
                "symbol": "xau",
                "q": "buy",
                "min_rr": "1.75",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        sheets = self._read_export_workbook(response)
        meta_map = self._export_meta_map(sheets)

        self.assertEqual(
            str(meta_map["Source File"]),
            "history.csv",
        )
        self.assertEqual(
            int(meta_map["Source Rows"]),
            4,
        )
        self.assertEqual(
            int(meta_map["Analysed Rows"]),
            2,
        )
        self.assertEqual(
            int(meta_map["Exported Rows"]),
            1,
        )
        self.assertEqual(
            str(meta_map["Analysis Filters"]),
            "symbol=xau&q=buy",
        )
        self.assertEqual(
            str(meta_map["Export-only Min RR"]),
            "1.75",
        )
        self.assertEqual(
            len(sheets["Trades"]),
            1,
        )
        self.assertNotIn(
            r"trading_files\user_4\history.csv",
            " ".join(str(v) for v in meta_map.values()),
        )
        self.assertNotIn(
            r"trading_files\user_4\history.csv",
            response.content.decode(
                "latin-1",
                errors="ignore",
            ),
        )

    def test_export_kpi_sheet_matches_kpi_report_analysis_scope(self):
        self._load_export_session()

        export_response = self.client.get(
            self.export_url,
            {
                "download": "1",
                "include_kpis": "1",
                "symbol": "xau",
                "q": "buy",
                "min_rr": "1.75",
            },
        )
        report_response = self.client.get(
            reverse("performance:kpi_report"),
            {
                "symbol": "xau",
                "q": "buy",
            },
        )

        sheets = self._read_export_workbook(
            export_response
        )
        kpi_sheet = sheets["KPIs"]
        excel_kpis = {
            str(metric): value
            for metric, value in zip(
                kpi_sheet["Metric"],
                kpi_sheet["Value"],
            )
        }
        report_kpis = report_response.context["kpis"]

        self.assertEqual(
            len(sheets["Trades"]),
            1,
        )
        self.assertEqual(
            set(excel_kpis),
            set(report_kpis),
        )
        for key, expected in report_kpis.items():
            self.assertEqual(
                self._normalise_kpi_value(
                    excel_kpis[key]
                ),
                self._normalise_kpi_value(expected),
                key,
            )
        self.assertEqual(
            self._normalise_kpi_value(
                excel_kpis["Total Trades"]
            ),
            "2",
        )

    def test_export_invalid_or_unavailable_min_rr_is_safe(self):
        self._load_export_session()

        invalid_response = self.client.get(
            self.export_url,
            {
                "preview": "1",
                "symbol": "xau",
                "q": "buy",
                "min_rr": "not-a-number",
            },
        )

        self.assertEqual(
            invalid_response.status_code,
            200,
        )

        invalid_context = invalid_response.context[
            "analysis_context"
        ]

        self.assertEqual(
            invalid_context.filtered_row_count,
            2,
        )
        self.assertEqual(
            invalid_response.context[
                "exported_row_count"
            ],
            invalid_context.filtered_row_count,
        )
        self.assertFalse(
            invalid_response.context["min_rr_applied"]
        )
        self.assertContains(
            invalid_response,
            "Not applied",
        )
        self.assertNotIn(
            "min_rr",
            invalid_context.active_filters,
        )

        session = self.client.session
        session["cleaned_data"] = self.df.to_json(
            orient="split",
            date_format="iso",
        )
        session.save()

        missing_rr_response = self.client.get(
            self.export_url,
            {
                "preview": "1",
                "symbol": "xau",
                "q": "buy",
                "min_rr": "1.75",
            },
        )

        missing_rr_context = missing_rr_response.context[
            "analysis_context"
        ]

        self.assertEqual(
            missing_rr_response.status_code,
            200,
        )
        self.assertEqual(
            missing_rr_context.filtered_row_count,
            2,
        )
        self.assertEqual(
            missing_rr_response.context[
                "exported_row_count"
            ],
            missing_rr_context.filtered_row_count,
        )
        self.assertFalse(
            missing_rr_response.context["min_rr_applied"]
        )
        self.assertContains(
            missing_rr_response,
            "Not applied",
        )
        self.assertNotIn(
            "min_rr",
            missing_rr_context.active_filters,
        )

    def test_export_context_recomputed_with_private_analysis_navigation(self):
        self._load_export_session()

        first_response = self.client.get(
            self.export_url,
            {
                "preview": "1",
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
            self.export_url,
            {
                "preview": "1",
            },
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
        self.assertEqual(
            second_context.source_filename,
            "history.csv",
        )
        self.assertContains(
            second_response,
            "history.csv",
        )
        self.assertNotContains(
            second_response,
            r"trading_files\user_4\history.csv",
        )

        filtered_response = self.client.get(
            self.export_url,
            {
                "start_date": "2026-09-01",
                "symbol": "xau",
                "q": "buy",
                "min_rr": "1.75",
                "include_kpis": "1",
                "preview": "1",
            },
        )

        analysis_query = filtered_response.context[
            "analysis_query"
        ]

        self.assertEqual(
            analysis_query,
            (
                "start_date=2026-09-01"
                "&symbol=xau"
                "&q=buy"
            ),
        )

        dashboard_href = (
            reverse("performance:dashboard")
            + "?"
            + analysis_query
        )
        report_href = (
            reverse("performance:kpi_report")
            + "?"
            + analysis_query
        )

        self.assertContains(
            filtered_response,
            f'href="{dashboard_href.replace("&", "&amp;")}"',
            html=False,
        )
        self.assertContains(
            filtered_response,
            f'href="{report_href.replace("&", "&amp;")}"',
            html=False,
        )

        for leaked in (
            "min_rr",
            "include_kpis",
            "preview",
            "download",
            "cols",
            "kpi_q",
            "trade_page",
            "kpi_page",
            "file_q",
            "file_status",
        ):
            self.assertNotIn(
                leaked,
                analysis_query,
            )
        self.assertNotIn(
            "page=",
            analysis_query,
        )

    def _load_trade_review_session(self, rows=12):
        data = {
            "Ticket": list(range(1, rows + 1)),
            "Open Time": [
                f"2026-09-{(index % 28) + 1:02d} 10:00:00"
                for index in range(rows)
            ],
            "Symbol": [
                "XAUUSD" if index % 2 == 0 else "EURUSD"
                for index in range(rows)
            ],
            "Type": [
                "buy" if index % 2 == 0 else "sell"
                for index in range(rows)
            ],
            "Price": [100 + index for index in range(rows)],
            "Price.1": [
                110 + index if index % 2 == 0 else 90 + index
                for index in range(rows)
            ],
            "Profit": [10 * (rows - index) for index in range(rows)],
            "Pips": [
                10.0 if index < 2 else None
                for index in range(rows)
            ],
            "Notes": ["ok"] * rows,
        }
        if rows:
            data["Symbol"][0] = "<script>alert(1)</script>"
        frame = pd.DataFrame(data)
        session = self.client.session
        session["cleaned_data"] = frame.to_json(
            orient="split",
            date_format="iso",
        )
        session["last_uploaded_file"] = (
            r"trading_files\user_4\history.csv"
        )
        session.save()
        return frame

    def _normalize_chart_html(self, html):
        if not html:
            return html
        return re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "PLOT-ID",
            html,
        )

    def _trade_column_index(self, response, key):
        for index, column in enumerate(
            response.context["trade_review_columns"]
        ):
            if column["key"] == key:
                return index
        self.fail(f"Missing Trade Review column {key}")

    def test_trade_review_sorts_before_pagination(self):
        self._load_trade_review_session(12)

        response = self.client.get(
            self.dashboard_url,
            {
                "trade_sort": "profit",
                "trade_dir": "desc",
                "trade_page": "1",
            },
        )

        page_rows = response.context["trade_page"].object_list
        self.assertEqual(len(page_rows), 10)
        profit_index = self._trade_column_index(
            response,
            "profit",
        )
        first_profits = [
            row["cells"][profit_index]["value"]
            for row in page_rows
        ]
        self.assertEqual(first_profits[0], 120)
        self.assertEqual(first_profits[-1], 30)

    def test_changing_sort_resets_trade_page(self):
        self._load_trade_review_session(12)

        response = self.client.get(
            self.dashboard_url,
            {
                "trade_sort": "profit",
                "trade_dir": "asc",
                "trade_page": "2",
                "symbol": "usd",
            },
        )

        profit_column = None
        for column in response.context["trade_review_columns"]:
            if column["key"] == "profit":
                profit_column = column
                break

        self.assertIsNotNone(profit_column)
        self.assertNotIn(
            "trade_page",
            profit_column["sort_query"],
        )
        self.assertIn(
            "trade_sort=profit",
            profit_column["sort_query"],
        )
        self.assertIn(
            "symbol=usd",
            profit_column["sort_query"],
        )

    def test_pagination_preserves_trade_sort_dir_and_filters(self):
        self._load_trade_review_session(12)

        response = self.client.get(
            self.dashboard_url,
            {
                "start_date": "2026-09-01",
                "end_date": "2026-09-30",
                "symbol": "usd",
                "q": "buy",
                "trade_sort": "profit",
                "trade_dir": "desc",
                "trade_page": "2",
            },
        )

        trade_query = response.context["trade_query"]
        self.assertIn("start_date=2026-09-01", trade_query)
        self.assertIn("end_date=2026-09-30", trade_query)
        self.assertIn("symbol=usd", trade_query)
        self.assertIn("q=buy", trade_query)
        self.assertIn("trade_sort=profit", trade_query)
        self.assertIn("trade_dir=desc", trade_query)
        self.assertNotIn("trade_page", trade_query)

    def test_kpis_and_charts_unchanged_across_trade_sort(self):
        first = self.client.get(self.dashboard_url)
        second = self.client.get(
            self.dashboard_url,
            {
                "trade_sort": "profit",
                "trade_dir": "desc",
            },
        )

        self.assertEqual(
            first.context["kpis"],
            second.context["kpis"],
        )
        self.assertEqual(
            self._normalize_chart_html(
                first.context["chart_equity"]
            ),
            self._normalize_chart_html(
                second.context["chart_equity"]
            ),
        )
        self.assertEqual(
            self._normalize_chart_html(
                first.context["chart_profit"]
            ),
            self._normalize_chart_html(
                second.context["chart_profit"]
            ),
        )
        self.assertEqual(
            first.context["analysis_context"].filtered_row_count,
            second.context["analysis_context"].filtered_row_count,
        )
        self.assertNotIn(
            "trade_sort",
            first.context["analysis_context"].active_filters,
        )
        self.assertNotIn(
            "trade_dir",
            second.context["analysis_context"].active_filters,
        )

    def test_trade_review_cells_are_escaped(self):
        self._load_trade_review_session(4)

        response = self.client.get(self.dashboard_url)
        content = response.content.decode()

        self.assertNotIn("<script>alert(1)</script>", content)
        self.assertIn(
            "&lt;script&gt;alert(1)&lt;/script&gt;",
            content,
        )

    def test_trade_review_page_size_is_ten(self):
        self._load_trade_review_session(12)

        response = self.client.get(self.dashboard_url)

        self.assertEqual(
            response.context["trade_page"].paginator.per_page,
            10,
        )
        self.assertEqual(
            len(response.context["trade_page"].object_list),
            10,
        )
        self.assertEqual(
            response.context["trade_page"].paginator.num_pages,
            2,
        )

    def test_trade_review_zero_result_remains_safe(self):
        response = self.client.get(
            self.dashboard_url,
            {
                "symbol": "NO-SUCH-SYMBOL",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            list(response.context["trade_page"].object_list),
            [],
        )
        self.assertContains(
            response,
            "No trades match the current analysis scope",
        )

    def test_trade_review_renders_broker_pips_and_price_move(self):
        self._load_trade_review_session(4)

        response = self.client.get(self.dashboard_url)
        movement_index = self._trade_column_index(
            response,
            "movement",
        )
        notes = [
            row["cells"][movement_index]["note"]
            for row in response.context["trade_page"].object_list
        ]

        self.assertIn("Pips", notes)
        self.assertIn("Price Move", notes)
        self.assertContains(response, "Realised movement")
