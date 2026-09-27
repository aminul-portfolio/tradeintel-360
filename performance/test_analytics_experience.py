import datetime as dt
import re
import types
import zipfile
from contextlib import contextmanager
from io import BytesIO
from unittest.mock import patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .analytics import apply_analysis
from .excursion import compute_journal_fingerprint
from .excursion_state import EXCURSION_SESSION_KEY
from .forms import DECLARED_EXPORT_CTRADER_CBOT, MarketDataUploadForm
from .market_data import EXCURSION_CONTRACT_VERSION, MAX_MARKET_DATA_BYTES
from .models import TradingFile
from .trade_review import PIPS_MOVEMENT_TOLERANCE

_FIXED_XLSX_DATETIME = dt.datetime(2026, 6, 25, 10, 30, 0)
_FIXED_ZIP_DATE_TIME = (2026, 6, 25, 10, 30, 0)


class _FrozenXlsxDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_XLSX_DATETIME


_FROZEN_XLSX_DATETIME_MODULE = types.SimpleNamespace(
    datetime=_FrozenXlsxDateTime,
    timezone=dt.timezone,
)


@contextmanager
def freeze_xlsx_timestamps():
    real_init = zipfile.ZipInfo.__init__

    def frozen_init(self, filename="NoName", date_time=None):
        real_init(self, filename, _FIXED_ZIP_DATE_TIME)

    with (
        patch("openpyxl.packaging.core.datetime", _FROZEN_XLSX_DATETIME_MODULE),
        patch("openpyxl.writer.excel.datetime", _FROZEN_XLSX_DATETIME_MODULE),
        patch.object(zipfile.ZipInfo, "__init__", frozen_init),
    ):
        yield


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
            "Volume": [
                0.30000000000000004
                if index == 2
                else 100000
                if index == 1
                else 1.0 + index
                for index in range(rows)
            ],
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

    def _trade_review_markup(self, content):
        match = re.search(
            r'<section class="dash-section" id="trade-review">(.*?)</section>',
            content,
            re.S,
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def test_trade_review_volume_display_and_fragment_navigation(self):
        self._load_trade_review_session(12)

        inactive = self.client.get(
            self.dashboard_url,
            {
                "symbol": "usd",
            },
        )
        inactive_html = inactive.content.decode()
        inactive_review = self._trade_review_markup(inactive_html)
        volume_column = None
        for column in inactive.context["trade_review_columns"]:
            if column["key"] == "volume":
                volume_column = column
                break
        self.assertIsNotNone(volume_column)
        volume_href = None
        for match in re.finditer(
            r'<a\s+class="trade-review-sort"\s+href="([^"]+)"\s+aria-label="([^"]+)"',
            inactive_review,
        ):
            href, label = match.groups()
            if "trade_sort=volume" in href.replace("&amp;", "&"):
                volume_href = href
                self.assertEqual(
                    label,
                    "Sort by Volume, ascending",
                )
        self.assertIsNotNone(volume_href)
        self.assertTrue(volume_href.endswith("#trade-review"))
        self.assertEqual(volume_href.count("#trade-review"), 1)
        self.assertNotIn("%23trade-review", volume_href)
        self.assertNotIn("trade_page", volume_href.replace("&amp;", "&"))
        self.assertIn("symbol=usd", volume_href.replace("&amp;", "&"))

        sort_hrefs = re.findall(
            r'class="trade-review-sort"[^>]*href="([^"]+)"',
            inactive_review,
        )
        self.assertTrue(sort_hrefs)
        for href in sort_hrefs:
            self.assertTrue(href.endswith("#trade-review"))
            self.assertEqual(href.count("#trade-review"), 1)
            self.assertNotIn("%23trade-review", href)
            self.assertNotIn("trade_page", href.replace("&amp;", "&"))

        self.assertEqual(inactive_html.count('id="trade-review"'), 1)
        self.assertNotIn(
            "#trade-review",
            re.search(
                r'<form method="get" class="app-filter-bar".*?</form>',
                inactive_html,
                re.S,
            ).group(0),
        )
        reset_hrefs = re.findall(
            r'href="([^"]+)"[^>]*>[\s\S]*?Reset',
            inactive_html,
        )
        for href in reset_hrefs:
            self.assertNotIn("#trade-review", href)

        volume_index = self._trade_column_index(inactive, "volume")
        displayed_volumes = [
            row["cells"][volume_index]["value"]
            for row in inactive.context["trade_page"].object_list
        ]
        self.assertIn("0.3", displayed_volumes)
        self.assertIn("100000", displayed_volumes)
        self.assertNotIn("0.30000000000000004", displayed_volumes)
        self.assertFalse(
            any("e" in str(value).lower() for value in displayed_volumes)
        )

        active_asc = self.client.get(
            self.dashboard_url,
            {
                "symbol": "usd",
                "trade_sort": "volume",
                "trade_dir": "asc",
            },
        )
        active_asc_review = self._trade_review_markup(
            active_asc.content.decode()
        )
        self.assertIn('aria-sort="ascending"', active_asc_review)
        self.assertIn('aria-sort="none"', active_asc_review)
        self.assertIn(
            'aria-label="Sort by Volume, descending"',
            active_asc_review,
        )
        self.assertIn('aria-hidden="true"', active_asc_review)
        self.assertIn(
            'class="trade-review-sort-indicator"',
            active_asc_review,
        )

        active_volume_href = None
        for href in re.findall(
            r'class="trade-review-sort"[^>]*href="([^"]+)"',
            active_asc_review,
        ):
            if "trade_sort=volume" in href.replace("&amp;", "&"):
                active_volume_href = href
        self.assertIsNotNone(active_volume_href)
        self.assertTrue(active_volume_href.endswith("#trade-review"))
        self.assertEqual(active_volume_href.count("#trade-review"), 1)
        self.assertNotIn("%23trade-review", active_volume_href)
        self.assertNotIn(
            "trade_page",
            active_volume_href.replace("&amp;", "&"),
        )

        active_desc = self.client.get(
            self.dashboard_url,
            {
                "symbol": "usd",
                "trade_sort": "volume",
                "trade_dir": "desc",
                "trade_page": "2",
            },
        )
        active_desc_html = active_desc.content.decode()
        active_desc_review = self._trade_review_markup(active_desc_html)
        self.assertIn('aria-sort="descending"', active_desc_review)
        self.assertIn(
            'aria-label="Sort by Volume, ascending"',
            active_desc_review,
        )
        self.assertEqual(
            active_desc.context["trade_page"].paginator.per_page,
            10,
        )

        page_hrefs = re.findall(
            r'<a\s+class="page-link"\s+href="([^"]+)"',
            active_desc_review,
        )
        self.assertTrue(page_hrefs)
        for href in page_hrefs:
            self.assertTrue(href.endswith("#trade-review"))
            self.assertEqual(href.count("#trade-review"), 1)
            self.assertNotIn("%23trade-review", href)
            decoded = href.replace("&amp;", "&")
            self.assertIn("symbol=usd", decoded)
            self.assertIn("trade_sort=volume", decoded)
            self.assertIn("trade_dir=desc", decoded)
        self.assertTrue(
            any("trade_page=1" in href.replace("&amp;", "&") for href in page_hrefs)
        )

        first_page = self.client.get(
            self.dashboard_url,
            {
                "symbol": "usd",
                "trade_sort": "volume",
                "trade_dir": "desc",
                "trade_page": "1",
            },
        )
        next_hrefs = re.findall(
            r'<a\s+class="page-link"\s+href="([^"]+)"',
            self._trade_review_markup(first_page.content.decode()),
        )
        self.assertTrue(
            any(
                href.endswith("#trade-review")
                and "trade_page=2" in href.replace("&amp;", "&")
                for href in next_hrefs
            )
        )

        first = self.client.get(self.dashboard_url)
        paged = self.client.get(
            self.dashboard_url,
            {
                "trade_sort": "volume",
                "trade_dir": "desc",
                "trade_page": "2",
            },
        )
        self.assertEqual(first.context["kpis"], paged.context["kpis"])
        self.assertEqual(
            self._normalize_chart_html(first.context["chart_equity"]),
            self._normalize_chart_html(paged.context["chart_equity"]),
        )
        self.assertEqual(
            PIPS_MOVEMENT_TOLERANCE,
            0.1,
        )


class MarketDataUploadFormTests(SimpleTestCase):
    def _file(self, content=b"TimestampUTC,Symbol,Open,High,Low,Close\n"):
        return SimpleUploadedFile("bars.csv", content)

    def _data(self, **overrides):
        data = {
            "declared_source": "user_declared_export",
            "declared_export_method": DECLARED_EXPORT_CTRADER_CBOT,
            "time_basis_kind": "UTC",
        }
        data.update(overrides)
        return data

    def test_valid_utc_fixed_offset_and_iana_forms(self):
        utc = MarketDataUploadForm(
            data=self._data(),
            files={"market_file": self._file()},
        )
        offset = MarketDataUploadForm(
            data=self._data(
                time_basis_kind="FIXED_OFFSET",
                offset_minutes=60,
            ),
            files={"market_file": self._file()},
        )
        iana = MarketDataUploadForm(
            data=self._data(
                time_basis_kind="IANA",
                iana_zone="Europe/London",
            ),
            files={"market_file": self._file()},
        )
        self.assertTrue(utc.is_valid(), utc.errors)
        self.assertEqual(utc.cleaned_data["time_basis"].kind, "UTC")
        self.assertTrue(offset.is_valid(), offset.errors)
        self.assertEqual(offset.cleaned_data["time_basis"].offset_minutes, 60)
        self.assertTrue(iana.is_valid(), iana.errors)
        self.assertEqual(iana.cleaned_data["time_basis"].zone, "Europe/London")

    def test_missing_or_invalid_time_basis_and_locked_export_method(self):
        missing_offset = MarketDataUploadForm(
            data=self._data(time_basis_kind="FIXED_OFFSET"),
            files={"market_file": self._file()},
        )
        missing_zone = MarketDataUploadForm(
            data=self._data(time_basis_kind="IANA"),
            files={"market_file": self._file()},
        )
        bad_offset = MarketDataUploadForm(
            data=self._data(
                time_basis_kind="FIXED_OFFSET",
                offset_minutes=62,
            ),
            files={"market_file": self._file()},
        )
        bad_zone = MarketDataUploadForm(
            data=self._data(
                time_basis_kind="IANA",
                iana_zone="Not/AZone",
            ),
            files={"market_file": self._file()},
        )
        utc_conflict = MarketDataUploadForm(
            data=self._data(offset_minutes=60),
            files={"market_file": self._file()},
        )
        locked = MarketDataUploadForm(
            data=self._data(declared_export_method="manual_csv"),
            files={"market_file": self._file()},
        )
        missing_source = MarketDataUploadForm(
            data=self._data(declared_source=""),
            files={"market_file": self._file()},
        )
        self.assertFalse(missing_offset.is_valid())
        self.assertFalse(missing_zone.is_valid())
        self.assertFalse(bad_offset.is_valid())
        self.assertFalse(bad_zone.is_valid())
        self.assertFalse(utc_conflict.is_valid())
        self.assertFalse(locked.is_valid())
        self.assertFalse(missing_source.is_valid())

    def test_oversized_file_is_rejected_before_read(self):
        uploaded = SimpleUploadedFile("huge.csv", b"tiny")
        uploaded.size = MAX_MARKET_DATA_BYTES + 1
        form = MarketDataUploadForm(
            data=self._data(),
            files={"market_file": uploaded},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("market_file", form.errors)


class BrokerBarUploadIntegrationTests(TestCase):
    FORBIDDEN_CLAIMS = (
        "Verified FTMO broker bars",
        "Verified broker data",
        "True MFE",
        "True MAE",
        "Exact in-trade high",
        "Exact in-trade low",
    )
    CROSS_SYMBOL_WARNING = (
        "MFE/MAE use each trade's instrument price points. "
        "Values across different symbols are not directly comparable."
    )

    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="broker-bar-user",
            email="broker-bar@example.com",
            password="test-password-123",
        )
        self.client.force_login(self.user)
        self.dashboard_url = reverse("performance:dashboard")
        self.upload_url = reverse("performance:upload_market_data")

    def _journal_frame(self, rows=None):
        if rows is None:
            rows = [
                {
                    "Ticket": 1,
                    "Open Time": "25 Jun 2026 10:30:15",
                    "Close Time": "25 Jun 2026 10:31:20",
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 105.0,
                    "Profit": 10.0,
                    "Volume": 1.0,
                    "Notes": "ok",
                },
                {
                    "Ticket": 2,
                    "Open Time": "25 Jun 2026 10:30:15",
                    "Close Time": "25 Jun 2026 10:31:20",
                    "Symbol": "EURUSD",
                    "Type": "sell",
                    "Entry": 1.20,
                    "Exit": 1.10,
                    "Profit": 5.0,
                    "Volume": 2.0,
                    "Notes": "ok",
                },
            ]
        return pd.DataFrame(rows)

    def _store_journal(self, frame):
        session = self.client.session
        session["cleaned_data"] = frame.to_json(
            orient="split",
            date_format="iso",
        )
        session["last_uploaded_file"] = r"trading_files\user_4\history.csv"
        session.save()
        return frame

    def _market_bytes(self, extra_rows=None):
        rows = [
            "TimestampUTC,Symbol,Open,High,Low,Close",
            "2026-06-25T10:30:00Z,US100.cash,100,110,90,101",
            "2026-06-25T10:31:00Z,US100.cash,101,112,91,102",
            "2026-06-25T10:30:00Z,EURUSD,1.10,1.25,1.05,1.15",
            "2026-06-25T10:31:00Z,EURUSD,1.15,1.22,1.00,1.12",
        ]
        if extra_rows:
            rows.extend(extra_rows)
        return ("\n".join(rows) + "\n").encode("utf-8")

    def _upload_files(self, payload=None, name="bars.csv"):
        return {
            "market_file": SimpleUploadedFile(
                name,
                payload if payload is not None else self._market_bytes(),
            )
        }

    def _upload_data(self, **overrides):
        data = {
            "declared_source": "user_declared_export",
            "declared_export_method": DECLARED_EXPORT_CTRADER_CBOT,
            "time_basis_kind": "UTC",
        }
        data.update(overrides)
        return data

    def _post_upload(self, data=None, files=None, follow=False):
        payload = {}
        payload.update(data if data is not None else self._upload_data())
        payload.update(files if files is not None else self._upload_files())
        return self.client.post(
            self.upload_url,
            data=payload,
            follow=follow,
        )

    def _session_state(self):
        return self.client.session.get(EXCURSION_SESSION_KEY)

    def _seed_bound_state(self, frame, evidence, status_counts=None, sha="abc123"):
        fingerprint = compute_journal_fingerprint(frame)
        payload = {
            "contract_version": EXCURSION_CONTRACT_VERSION,
            "journal_fingerprint": fingerprint,
            "market_file_sha256": sha,
            "time_basis": {"kind": "UTC", "resolution": "USER_DECLARED"},
            "market_provenance": {
                "source_basename": "bars.csv",
                "market_file_sha256": sha,
                "file_byte_size": 128,
                "declared_source": "user_declared_export",
                "declared_export_method": DECLARED_EXPORT_CTRADER_CBOT,
                "bar_timestamp_semantic": "BAR_OPEN_TIME",
                "resolution": "M1",
                "evidence_class": "BROKER_BAR_SOURCE",
                "precision": "M1_BAR_APPROXIMATE",
                "price_basis": "UNSPECIFIED_CTRADER_HISTORICAL_BAR",
                "contract_version": EXCURSION_CONTRACT_VERSION,
                "source_row_count": 4,
                "valid_row_count": 4,
                "symbols": ["EURUSD", "US100.cash"],
                "coverage": [
                    {
                        "symbol": "US100.cash",
                        "coverage_start": "2026-06-25T10:30:00Z",
                        "coverage_end": "2026-06-25T10:31:00Z",
                    }
                ],
            },
            "run_status": "OK",
            "status_counts": status_counts or {"COMPUTED": len(evidence)},
            "reason_counts": {},
            "evidence": evidence,
        }
        session = self.client.session
        session[EXCURSION_SESSION_KEY] = payload
        session.save()
        return payload

    def _evidence_item(self, ticket, **overrides):
        item = {
            "ticket": str(ticket),
            "status": "COMPUTED",
            "reason_code": "OK",
            "interval_high": 110.0,
            "interval_low": 90.0,
            "mfe": 10.0,
            "mae": 4.0,
            "high_from_boundary_bar": False,
            "low_from_boundary_bar": False,
            "entry_outside_first_bar_range": False,
            "exit_outside_last_bar_range": False,
            "realised_outside_interval": False,
        }
        item.update(overrides)
        return item

    def _column_index(self, response, key):
        for index, column in enumerate(response.context["trade_review_columns"]):
            if column["key"] == key:
                return index
        self.fail(f"Missing Trade Review column {key}")

    def _trade_review_markup(self, content):
        match = re.search(
            r'<section class="dash-section" id="trade-review">(.*?)</section>',
            content,
            re.S,
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def _context_payload(self, context):
        return {
            "source_filename": context.source_filename,
            "source_row_count": context.source_row_count,
            "filtered_row_count": context.filtered_row_count,
            "active_filters": dict(context.active_filters),
            "is_zero_result": context.is_zero_result,
        }

    def _assert_trade_review_href(
        self,
        href,
        *,
        sort_key=None,
        trade_dir=None,
        filters=(),
        trade_page=None,
        allow_trade_page=False,
    ):
        decoded = href.replace("&amp;", "&")
        self.assertTrue(decoded.endswith("#trade-review"), decoded)
        self.assertEqual(decoded.count("#trade-review"), 1, decoded)
        self.assertNotIn("%23trade-review", decoded)
        if sort_key is not None:
            self.assertIn(f"trade_sort={sort_key}", decoded)
        if trade_dir is not None:
            self.assertIn(f"trade_dir={trade_dir}", decoded)
        for item in filters:
            self.assertIn(item, decoded)
        if trade_page is not None:
            self.assertIn(f"trade_page={trade_page}", decoded)
        elif not allow_trade_page:
            self.assertNotIn("trade_page", decoded)

    def _sort_href_map(self, review):
        found = {}
        for match in re.finditer(
            r'<a\s+class="trade-review-sort"\s+href="([^"]+)"',
            review,
        ):
            href = match.group(1)
            decoded = href.replace("&amp;", "&")
            key_match = re.search(r"trade_sort=([a-z_]+)", decoded)
            if key_match:
                found[key_match.group(1)] = href
        return found

    def test_route_auth_and_post_only(self):
        self.assertEqual(
            self.upload_url,
            "/performance/market-data/upload/",
        )
        get_response = self.client.get(self.upload_url)
        self.assertEqual(get_response.status_code, 405)
        self.assertIsNone(self._session_state())

        self.client.logout()
        anonymous_payload = {}
        anonymous_payload.update(self._upload_data())
        anonymous_payload.update(self._upload_files())
        anonymous = self.client.post(
            self.upload_url,
            data=anonymous_payload,
        )
        self.assertEqual(anonymous.status_code, 302)
        self.assertIn("/login/", anonymous.url)

    def test_pre_read_limit_rejects_before_parser(self):
        self._store_journal(self._journal_frame())
        with (
            patch("performance.forms.MAX_MARKET_DATA_BYTES", 1),
            patch("performance.views.parse_market_data_bytes") as mocked,
        ):
            response = self._post_upload()
        mocked.assert_not_called()
        self.assertRedirects(response, self.dashboard_url)
        self.assertIsNone(self._session_state())

    def test_no_journal_fails_safely(self):
        response = self._post_upload()
        self.assertRedirects(response, self.dashboard_url)
        self.assertIsNone(self._session_state())
        self.assertEqual(TradingFile.objects.count(), 0)

    def test_invalid_market_data_clears_old_evidence_and_keeps_journal(self):
        frame = self._store_journal(self._journal_frame())
        self._seed_bound_state(
            frame,
            {"1": self._evidence_item(1)},
            sha="old-sha",
        )
        self.assertIsNotNone(self._session_state())

        response = self._post_upload(
            files=self._upload_files(b"not,a,valid,csv\n1,2,3\n")
        )
        self.assertRedirects(response, self.dashboard_url)
        self.assertIsNone(self._session_state())
        self.assertIn("cleaned_data", self.client.session)
        self.assertEqual(TradingFile.objects.count(), 0)

    def test_successful_utc_fixed_offset_and_iana_uploads(self):
        self._store_journal(self._journal_frame())
        utc = self._post_upload()
        self.assertRedirects(utc, self.dashboard_url)
        state = self._session_state()
        self.assertIsNotNone(state)
        self.assertEqual(state["time_basis"]["kind"], "UTC")
        self.assertEqual(
            state["market_provenance"]["declared_source"],
            "user_declared_export",
        )
        self.assertTrue(state["journal_fingerprint"])
        self.assertNotIn("raw_bytes", state)
        self.assertNotIn("raw_csv", self.client.session)
        self.assertNotIn("market_dataframe", self.client.session)
        self.assertEqual(TradingFile.objects.count(), 0)
        session_blob = str(self.client.session.get(EXCURSION_SESSION_KEY))
        self.assertNotIn("TimestampUTC", session_blob)

        offset_journal = self._journal_frame(
            [
                {
                    "Ticket": 10,
                    "Open Time": "25 Jun 2026 11:30:15",
                    "Close Time": "25 Jun 2026 11:31:20",
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 105.0,
                    "Profit": 10.0,
                }
            ]
        )
        self._store_journal(offset_journal)
        offset = self._post_upload(
            data=self._upload_data(
                time_basis_kind="FIXED_OFFSET",
                offset_minutes=60,
            )
        )
        self.assertRedirects(offset, self.dashboard_url)
        self.assertEqual(self._session_state()["time_basis"]["kind"], "FIXED_OFFSET")
        self.assertEqual(self._session_state()["time_basis"]["offset_minutes"], 60)

        iana_journal = self._journal_frame(
            [
                {
                    "Ticket": 11,
                    "Open Time": "25 Jun 2026 11:30:15",
                    "Close Time": "25 Jun 2026 11:31:20",
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 105.0,
                    "Profit": 10.0,
                }
            ]
        )
        self._store_journal(iana_journal)
        iana = self._post_upload(
            data=self._upload_data(
                time_basis_kind="IANA",
                iana_zone="Europe/London",
            )
        )
        self.assertRedirects(iana, self.dashboard_url)
        self.assertEqual(self._session_state()["time_basis"]["kind"], "IANA")
        self.assertEqual(self._session_state()["time_basis"]["zone"], "Europe/London")

    def test_join_key_failure_is_not_stored(self):
        self._store_journal(
            pd.DataFrame(
                {
                    "Open Time": ["25 Jun 2026 10:30:15"],
                    "Close Time": ["25 Jun 2026 10:31:20"],
                    "Symbol": ["US100.cash"],
                    "Type": ["buy"],
                    "Entry": [100.0],
                    "Exit": [105.0],
                    "Profit": [10.0],
                }
            )
        )
        response = self._post_upload()
        self.assertRedirects(response, self.dashboard_url)
        self.assertIsNone(self._session_state())

    def test_successful_replacement_and_failed_replacement(self):
        self._store_journal(self._journal_frame())
        first = self._post_upload(
            files=self._upload_files(self._market_bytes(), name="old.csv")
        )
        self.assertRedirects(first, self.dashboard_url)
        old_state = self._session_state()
        old_sha = old_state["market_file_sha256"]
        old_tickets = set(old_state["evidence"])

        replacement_bytes = self._market_bytes(
            extra_rows=[
                "2026-06-25T10:32:00Z,US100.cash,102,130,80,103",
                "2026-06-25T10:32:00Z,EURUSD,1.12,1.30,0.90,1.10",
            ]
        )
        second = self._post_upload(
            files=self._upload_files(replacement_bytes, name="new.csv")
        )
        self.assertRedirects(second, self.dashboard_url)
        new_state = self._session_state()
        self.assertNotEqual(new_state["market_file_sha256"], old_sha)
        self.assertEqual(new_state["market_provenance"]["source_basename"], "new.csv")
        self.assertTrue(old_tickets)
        self.assertTrue(new_state["evidence"])

        failed = self._post_upload(files=self._upload_files(b"bad-csv"))
        self.assertRedirects(failed, self.dashboard_url)
        self.assertIsNone(self._session_state())
        self.assertIn("cleaned_data", self.client.session)

    def test_dashboard_lazy_binding_and_stale_invalidation(self):
        frame = self._store_journal(self._journal_frame())
        self._seed_bound_state(frame, {"1": self._evidence_item(1), "2": self._evidence_item(2)})
        shown = self.client.get(self.dashboard_url)
        self.assertIsNotNone(shown.context["excursion_state"])
        labels = [column["label"] for column in shown.context["trade_review_columns"]]
        self.assertIn("Approx. MFE (price pts)", labels)

        changed = frame.copy()
        changed.loc[0, "Entry"] = 101.0
        self._store_journal(changed)
        session = self.client.session
        session[EXCURSION_SESSION_KEY] = shown.context["excursion_state"]
        session.save()
        stale = self.client.get(self.dashboard_url)
        self.assertIsNone(stale.context["excursion_state"])
        self.assertIsNone(self._session_state())

        no_fp = self._journal_frame()
        no_fp = no_fp.drop(columns=["Ticket"])
        self._store_journal(no_fp)
        session = self.client.session
        session[EXCURSION_SESSION_KEY] = {
            "contract_version": EXCURSION_CONTRACT_VERSION,
            "journal_fingerprint": "stale",
            "market_file_sha256": "abc",
            "time_basis": {"kind": "UTC", "resolution": "USER_DECLARED"},
            "market_provenance": {"declared_source": "x"},
            "run_status": "OK",
            "status_counts": {},
            "reason_counts": {},
            "evidence": {"1": self._evidence_item(1)},
        }
        session.save()
        missing_fp = self.client.get(self.dashboard_url)
        self.assertIsNone(missing_fp.context["excursion_state"])

        session = self.client.session
        session.pop("cleaned_data", None)
        session[EXCURSION_SESSION_KEY] = {
            "contract_version": EXCURSION_CONTRACT_VERSION,
            "journal_fingerprint": "stale",
            "market_file_sha256": "abc",
            "time_basis": {"kind": "UTC", "resolution": "USER_DECLARED"},
            "market_provenance": {"declared_source": "x"},
            "run_status": "OK",
            "status_counts": {},
            "reason_counts": {},
            "evidence": {"1": self._evidence_item(1)},
        }
        session.save()
        empty = self.client.get(self.dashboard_url)
        self.assertIsNone(empty.context["excursion_state"])
        self.assertIsNone(self._session_state())

    def test_filters_do_not_recompute_and_keep_full_journal_evidence(self):
        frame = self._store_journal(self._journal_frame())
        self._seed_bound_state(
            frame,
            {
                "1": self._evidence_item(1, mfe=21.0),
                "2": self._evidence_item(2, mfe=3.5),
            },
        )
        with patch("performance.views.compute_excursion_evidence") as mocked:
            response = self.client.get(
                self.dashboard_url,
                {"symbol": "US100", "q": "buy"},
            )
        mocked.assert_not_called()
        self.assertEqual(response.context["analysis_context"].filtered_row_count, 1)
        mfe_index = self._column_index(response, "mfe")
        values = [
            row["cells"][mfe_index]["value"]
            for row in response.context["trade_page"].object_list
        ]
        self.assertEqual(values, ["21"])

    def test_trade_review_visibility_blank_vs_zero(self):
        frame = self._store_journal(self._journal_frame())
        absent = self.client.get(self.dashboard_url)
        absent_labels = [column["label"] for column in absent.context["trade_review_columns"]]
        self.assertNotIn("Approx. Interval High", absent_labels)

        self._seed_bound_state(
            frame,
            {
                "1": self._evidence_item(1, mfe=0.0, mae=0.0),
                "2": self._evidence_item(
                    2,
                    status="NO_MARKET_DATA",
                    reason_code="OUTSIDE_FILE_RANGE",
                    interval_high=None,
                    interval_low=None,
                    mfe=None,
                    mae=None,
                ),
            },
        )
        present = self.client.get(self.dashboard_url)
        labels = [column["label"] for column in present.context["trade_review_columns"]]
        self.assertEqual(
            labels[-5:],
            [
                "Approx. Interval High",
                "Approx. Interval Low",
                "Approx. MFE (price pts)",
                "Approx. MAE (price pts)",
                "Bar Evidence",
            ],
        )
        mfe_index = self._column_index(present, "mfe")
        values = [
            row["cells"][mfe_index]["value"]
            for row in present.context["trade_page"].object_list
        ]
        self.assertIn("0", values)
        self.assertIn("", values)

    def test_evidence_sort_before_pagination_and_accessibility(self):
        rows = []
        evidence = {}
        for index in range(1, 13):
            rows.append(
                {
                    "Ticket": index,
                    "Open Time": "25 Jun 2026 10:30:15",
                    "Close Time": "25 Jun 2026 10:31:20",
                    "Symbol": "US100.cash" if index % 2 else "EURUSD",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 105.0,
                    "Profit": float(index),
                    "Volume": 1.0,
                    "Notes": "ok",
                }
            )
            if index == 12:
                evidence[str(index)] = self._evidence_item(
                    index,
                    mfe=None,
                    mae=None,
                    interval_high=None,
                    interval_low=None,
                    status="NO_MARKET_DATA",
                )
            else:
                evidence[str(index)] = self._evidence_item(
                    index,
                    mfe=float(index),
                    mae=float(20 - index),
                    interval_high=100.0 + index,
                    interval_low=80.0 + (index % 3),
                )
        frame = self._store_journal(pd.DataFrame(rows))
        self._seed_bound_state(frame, evidence)

        page_one = self.client.get(
            self.dashboard_url,
            {"trade_sort": "mfe", "trade_dir": "desc", "trade_page": "1"},
        )
        mfe_index = self._column_index(page_one, "mfe")
        page_values = [
            row["cells"][mfe_index]["value"]
            for row in page_one.context["trade_page"].object_list
        ]
        self.assertEqual(len(page_values), 10)
        self.assertEqual(page_values[0], "11")
        self.assertNotIn("", page_values)

        page_two = self.client.get(
            self.dashboard_url,
            {"trade_sort": "mfe", "trade_dir": "desc", "trade_page": "2"},
        )
        second_values = [
            row["cells"][mfe_index]["value"]
            for row in page_two.context["trade_page"].object_list
        ]
        self.assertEqual(second_values[-1], "")

        mfe_column = None
        for column in page_one.context["trade_review_columns"]:
            if column["key"] == "mfe":
                mfe_column = column
                break
        self.assertIsNotNone(mfe_column)
        self.assertNotIn("trade_page", mfe_column["sort_query"])
        self.assertIn("trade_sort=mfe", mfe_column["sort_query"])

        filtered = self.client.get(
            self.dashboard_url,
            {
                "symbol": "usd",
                "q": "buy",
                "trade_sort": "mae",
                "trade_dir": "asc",
            },
        )
        mae_column = None
        for column in filtered.context["trade_review_columns"]:
            if column["key"] == "mae":
                mae_column = column
                break
        self.assertIn("symbol=usd", mae_column["sort_query"])
        self.assertIn("q=buy", mae_column["sort_query"])
        self.assertEqual(filtered.context["trade_query"].count("#"), 0)
        self.assertIn("trade_sort=mae", filtered.context["trade_query"])

        review = self._trade_review_markup(page_one.content.decode())
        self.assertIn('aria-sort="descending"', review)
        self.assertIn('aria-label="Sort by Approx. MFE (price pts), ascending"', review)
        self.assertIn('aria-hidden="true"', review)

        page_one_sorts = self._sort_href_map(review)
        for key in ("interval_high", "interval_low", "mfe", "mae"):
            self.assertIn(key, page_one_sorts)
            self._assert_trade_review_href(
                page_one_sorts[key],
                sort_key=key,
            )

        filtered_review = self._trade_review_markup(filtered.content.decode())
        filtered_sorts = self._sort_href_map(filtered_review)
        analysis_filters = ("symbol=usd", "q=buy")
        for key in ("interval_high", "interval_low", "mfe", "mae"):
            self.assertIn(key, filtered_sorts)
            self._assert_trade_review_href(
                filtered_sorts[key],
                sort_key=key,
                filters=analysis_filters,
            )

        paged = self.client.get(
            self.dashboard_url,
            {
                "q": "buy",
                "trade_sort": "mfe",
                "trade_dir": "desc",
                "trade_page": "1",
            },
        )
        paged_review = self._trade_review_markup(paged.content.decode())
        next_hrefs = re.findall(
            r'<a\s+class="page-link"\s+href="([^"]+)"',
            paged_review,
        )
        page_two_hrefs = [
            href
            for href in next_hrefs
            if "trade_page=2" in href.replace("&amp;", "&")
        ]
        self.assertTrue(page_two_hrefs)
        for href in page_two_hrefs:
            self._assert_trade_review_href(
                href,
                sort_key="mfe",
                trade_dir="desc",
                filters=("q=buy",),
                trade_page="2",
                allow_trade_page=True,
            )

    def test_kpi_chart_and_export_invariance(self):
        frame = self._store_journal(self._journal_frame())
        filters = {"symbol": "US100", "q": "buy"}
        without = self.client.get(self.dashboard_url, filters)
        self._seed_bound_state(
            frame,
            {
                "1": self._evidence_item(1, mfe=21.0),
                "2": self._evidence_item(2, mfe=3.5),
            },
        )
        with_state = self.client.get(self.dashboard_url, filters)

        self.assertEqual(without.context["kpis"], with_state.context["kpis"])
        self.assertEqual(
            self._context_payload(without.context["analysis_context"]),
            self._context_payload(with_state.context["analysis_context"]),
        )
        for key in ("chart_equity", "chart_profit", "chart_hist", "chart_month"):
            self.assertEqual(
                re.sub(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    "PLOT-ID",
                    without.context[key] or "",
                ),
                re.sub(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    "PLOT-ID",
                    with_state.context[key] or "",
                ),
            )
        self.assertEqual(
            without.context["chart_pie_sections"].keys(),
            with_state.context["chart_pie_sections"].keys(),
        )
        for key, html in without.context["chart_pie_sections"].items():
            self.assertEqual(
                re.sub(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    "PLOT-ID",
                    html or "",
                ),
                re.sub(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    "PLOT-ID",
                    with_state.context["chart_pie_sections"][key] or "",
                ),
            )

        csv_url = reverse("performance:download_cleaned_csv")
        excel_url = reverse("performance:download_excel")
        csv_without = self.client.get(csv_url)
        session = self.client.session
        session.pop(EXCURSION_SESSION_KEY, None)
        session.save()
        csv_plain = self.client.get(csv_url)
        self.assertEqual(csv_without.content, csv_plain.content)
        self.assertNotIn(b"Approx. MFE", csv_plain.content)
        self.assertNotIn(b"Approx. MAE", csv_plain.content)

        self._seed_bound_state(
            frame,
            {"1": self._evidence_item(1), "2": self._evidence_item(2)},
        )
        with freeze_xlsx_timestamps():
            self.assertIs(dt.datetime, __import__("datetime").datetime)
            self.assertIsNot(dt.datetime, _FrozenXlsxDateTime)
            excel_with = self.client.get(excel_url)
            session = self.client.session
            session.pop(EXCURSION_SESSION_KEY, None)
            session.save()
            excel_without = self.client.get(excel_url)
        self.assertEqual(excel_with.content, excel_without.content)
        self.assertNotIn(b"Approx. MFE", excel_without.content)
        self.assertNotIn(b"Approx. MAE", excel_without.content)

    def test_provenance_card_claim_safety_and_matched_count(self):
        self._store_journal(self._journal_frame())
        self._post_upload()
        response = self.client.get(self.dashboard_url)
        html = response.content.decode()
        self.assertContains(response, "Uploaded cTrader M1 bars")
        self.assertContains(response, "Declared source")
        self.assertContains(response, "M1 bar approximation")
        self.assertContains(response, self.CROSS_SYMBOL_WARNING)
        for claim in self.FORBIDDEN_CLAIMS:
            self.assertNotIn(claim, html)
        self.assertContains(response, "user_declared_export")
        self.assertContains(response, DECLARED_EXPORT_CTRADER_CBOT)
        self.assertContains(response, "SHA-256")
        self.assertContains(response, "PRICE_COMPARISON_EPSILON")
        self.assertContains(response, "DURATION_TOLERANCE_SECONDS")
        self.assertContains(response, response.context["excursion_summary"]["journal_fingerprint"])

        mixed = self._session_state()
        mixed["status_counts"] = {
            "COMPUTED": 2,
            "INCOMPLETE_COVERAGE": 1,
            "INVARIANT_VIOLATION": 1,
            "NO_MARKET_DATA": 3,
            "INVALID_TRADE_DATA": 2,
            "TIMEZONE_AMBIGUOUS": 1,
            "TIME_BASIS_INCONSISTENT": 1,
        }
        session = self.client.session
        session[EXCURSION_SESSION_KEY] = mixed
        session.save()
        counted = self.client.get(self.dashboard_url)
        self.assertEqual(counted.context["excursion_summary"]["matched_trade_count"], 4)
