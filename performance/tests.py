import os
import tempfile
from io import StringIO

import pandas as pd
from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .ingestion import (
    EmptyDataset,
    IngestionError,
    MissingProfitColumn,
    NoParsableProfit,
    UnreadableFile,
    UnsupportedFileType,
    clean_ftmo_csv,
)
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

class IngestionTests(TestCase):
    def _write_temp_file(self, content: bytes, suffix: str) -> str:
        handle = tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        )
        try:
            handle.write(content)
            return handle.name
        finally:
            handle.close()

    def _remove_temp_file(self, file_path: str) -> None:
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_valid_csv_returns_non_empty_dataframe(self):
        file_path = self._write_temp_file(
            b"Symbol,Profit\nEURUSD,10\nXAUUSD,-5\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertFalse(df.empty)
        self.assertEqual(list(df["Profit"]), [10, -5])

    def test_semicolon_csv_is_supported(self):
        file_path = self._write_temp_file(
            b"Symbol;Profit\nEURUSD;10\nXAUUSD;-5\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertEqual(list(df.columns), ["Symbol", "Profit"])
        self.assertEqual(list(df["Profit"]), [10, -5])

    def test_valid_xlsx_returns_non_empty_dataframe(self):
        handle = tempfile.NamedTemporaryFile(
            suffix=".xlsx",
            delete=False,
        )
        file_path = handle.name
        handle.close()

        try:
            source_df = pd.DataFrame(
                {
                    "Symbol": ["EURUSD", "XAUUSD"],
                    "Profit": [10, -5],
                }
            )
            source_df.to_excel(
                file_path,
                index=False,
                engine="openpyxl",
            )

            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertFalse(df.empty)
        self.assertEqual(list(df["Profit"]), [10, -5])

    def test_legacy_xls_is_rejected(self):
        file_path = self._write_temp_file(
            b"not-used",
            ".xls",
        )
        try:
            with self.assertRaises(UnsupportedFileType):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

    def test_unsupported_extension_is_rejected(self):
        file_path = self._write_temp_file(
            b'{"Profit": 10}',
            ".json",
        )
        try:
            with self.assertRaises(UnsupportedFileType):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

    def test_empty_csv_raises_empty_dataset(self):
        file_path = self._write_temp_file(
            b"",
            ".csv",
        )
        try:
            with self.assertRaises(EmptyDataset):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

    def test_header_only_csv_raises_empty_dataset(self):
        file_path = self._write_temp_file(
            b"Symbol,Profit\n",
            ".csv",
        )
        try:
            with self.assertRaises(EmptyDataset):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

    def test_missing_profit_column_is_rejected(self):
        file_path = self._write_temp_file(
            b"Symbol,Volume\nEURUSD,1\n",
            ".csv",
        )
        try:
            with self.assertRaises(MissingProfitColumn):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

    def test_all_non_numeric_profit_is_rejected(self):
        file_path = self._write_temp_file(
            b"Symbol,Profit\nEURUSD,invalid\nXAUUSD,bad\n",
            ".csv",
        )
        try:
            with self.assertRaises(NoParsableProfit):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

    def test_column_whitespace_is_stripped(self):
        file_path = self._write_temp_file(
            b" Symbol , Profit \nEURUSD,10\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertIn("Symbol", df.columns)
        self.assertIn("Profit", df.columns)
        self.assertNotIn(" Symbol ", df.columns)

    def test_completely_empty_rows_are_removed(self):
        file_path = self._write_temp_file(
            b"Symbol,Profit\nEURUSD,10\n,\nXAUUSD,20\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertEqual(len(df), 2)

    def test_unnamed_export_index_column_is_removed(self):
        file_path = self._write_temp_file(
            b"Unnamed: 0,Symbol,Profit\n0,EURUSD,10\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertNotIn("Unnamed: 0", df.columns)
        self.assertIn("Symbol", df.columns)

    def test_hash_column_is_preserved(self):
        file_path = self._write_temp_file(
            b"Unnamed: 0,#,Profit\n0,ABC,10\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertNotIn("Unnamed: 0", df.columns)
        self.assertIn("#", df.columns)

    def test_commissions_is_renamed_to_commission(self):
        file_path = self._write_temp_file(
            b"Symbol,Profit,Commissions\nEURUSD,10,-2.5\n",
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertIn("Commission", df.columns)
        self.assertNotIn("Commissions", df.columns)
        self.assertEqual(df["Commission"].iloc[0], -2.5)

    def test_existing_commission_prevents_commissions_rename(self):
        file_path = self._write_temp_file(
            (
                b"Symbol,Profit,Commission,Commissions\n"
                b"EURUSD,10,-1.5,-2.5\n"
            ),
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertIn("Commission", df.columns)
        self.assertIn("Commissions", df.columns)
        self.assertEqual(df["Commission"].iloc[0], -1.5)
        self.assertEqual(df["Commissions"].iloc[0], -2.5)

    def test_numeric_columns_are_coerced(self):
        file_path = self._write_temp_file(
            (
                b"Symbol,Profit,Volume,Swap\n"
                b"EURUSD,10.5,2,-1.25\n"
            ),
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertTrue(
            pd.api.types.is_numeric_dtype(df["Profit"])
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(df["Volume"])
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(df["Swap"])
        )

    def test_first_recognised_date_column_is_parsed(self):
        file_path = self._write_temp_file(
            (
                b"Symbol,Open Time,Date,Profit\n"
                b"EURUSD,2026-09-01 10:30:00,not-a-date,10\n"
            ),
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(df["Open Time"])
        )
        self.assertFalse(
            pd.api.types.is_datetime64_any_dtype(df["Date"])
        )

    def test_cp1252_csv_is_supported(self):
        content = (
            "Symbol,Notes,Profit\n"
            "EURUSD,£ gain,10\n"
        ).encode("cp1252")

        file_path = self._write_temp_file(
            content,
            ".csv",
        )
        try:
            df = clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)

        self.assertEqual(df["Notes"].iloc[0], "£ gain")
        self.assertEqual(df["Profit"].iloc[0], 10)

    def test_missing_supported_file_raises_unreadable_file(self):
        missing_path = os.path.join(
            tempfile.gettempdir(),
            "tradeintel-definitely-missing-file.csv",
        )

        if os.path.exists(missing_path):
            os.remove(missing_path)

        with self.assertRaises(UnreadableFile):
            clean_ftmo_csv(missing_path)

    def test_ingestion_errors_share_common_base(self):
        self.assertTrue(
            issubclass(UnsupportedFileType, IngestionError)
        )
        self.assertTrue(
            issubclass(EmptyDataset, IngestionError)
        )
        self.assertTrue(
            issubclass(MissingProfitColumn, IngestionError)
        )
        self.assertTrue(
            issubclass(NoParsableProfit, IngestionError)
        )
        self.assertTrue(
            issubclass(UnreadableFile, IngestionError)
        )

    def test_corrupt_xlsx_raises_unreadable_file(self):
        file_path = self._write_temp_file(
            b"not-a-valid-xlsx-workbook",
            ".xlsx",
        )
        try:
            with self.assertRaises(UnreadableFile):
                clean_ftmo_csv(file_path)
        finally:
            self._remove_temp_file(file_path)


class IngestionViewIntegrationTests(TestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.media_override = override_settings(
            MEDIA_ROOT=self.media_directory.name
        )
        self.media_override.enable()

        self.user = User.objects.create_user(
            username="ingestion-user",
            password="test-password-123",
        )
        self.other_user = User.objects.create_user(
            username="other-ingestion-user",
            password="test-password-123",
        )

        self.client.force_login(self.user)

    def tearDown(self):
        self.media_override.disable()
        self.media_directory.cleanup()

    def _valid_upload(self, name="valid-trades.csv"):
        return SimpleUploadedFile(
            name,
            (
                b"Symbol,Open,Profit,Commissions\n"
                b"EURUSD,2026-09-01 10:00:00,100,-2\n"
                b"XAUUSD,2026-09-02 11:00:00,-40,-3\n"
            ),
            content_type="text/csv",
        )

    def _invalid_upload(self, name="invalid-trades.csv"):
        return SimpleUploadedFile(
            name,
            b"Symbol,Volume\nEURUSD,1\n",
            content_type="text/csv",
        )

    def test_valid_upload_marks_trading_file_processed(self):
        response = self.client.post(
            reverse("performance:upload_file"),
            {"file": self._valid_upload()},
        )

        self.assertEqual(response.status_code, 302)

        trading_file = TradingFile.objects.get(
            user=self.user
        )
        self.assertEqual(
            trading_file.status,
            "processed",
        )

    def test_invalid_upload_marks_trading_file_error(self):
        response = self.client.post(
            reverse("performance:upload_file"),
            {"file": self._invalid_upload()},
        )

        self.assertEqual(response.status_code, 200)

        trading_file = TradingFile.objects.get(
            user=self.user
        )
        self.assertEqual(
            trading_file.status,
            "error",
        )

    def test_valid_upload_stores_cleaned_data_in_session(self):
        self.client.post(
            reverse("performance:upload_file"),
            {"file": self._valid_upload()},
        )

        session = self.client.session

        self.assertIn(
            "cleaned_data",
            session,
        )

        df = pd.read_json(
            StringIO(session["cleaned_data"]),
            orient="split",
        )

        self.assertEqual(len(df), 2)
        self.assertIn("Commission", df.columns)
        self.assertNotIn("Commissions", df.columns)

    def test_failed_upload_clears_previous_cleaned_data(self):
        session = self.client.session
        session["cleaned_data"] = pd.DataFrame(
            {
                "Symbol": ["OLD"],
                "Profit": [999],
            }
        ).to_json(
            orient="split",
            date_format="iso",
        )
        session.save()

        self.client.post(
            reverse("performance:upload_file"),
            {"file": self._invalid_upload()},
        )

        self.assertNotIn(
            "cleaned_data",
            self.client.session,
        )

    def test_ingestion_error_is_shown_as_safe_message(self):
        response = self.client.post(
            reverse("performance:upload_file"),
            {"file": self._invalid_upload()},
        )

        self.assertEqual(response.status_code, 200)

        messages = [
            str(message)
            for message in get_messages(response.wsgi_request)
        ]

        self.assertIn(
            "The uploaded trading file must contain a Profit column.",
            messages,
        )

    def test_own_trading_file_loads_into_session(self):
        trading_file = TradingFile.objects.create(
            user=self.user,
            file=self._valid_upload("owned-trades.csv"),
            status="processed",
        )

        response = self.client.get(
            reverse(
                "performance:load_file",
                args=[trading_file.pk],
            )
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn(
            "cleaned_data",
            self.client.session,
        )

    def test_other_users_trading_file_returns_404(self):
        trading_file = TradingFile.objects.create(
            user=self.other_user,
            file=self._valid_upload("other-user-trades.csv"),
            status="processed",
        )

        response = self.client.get(
            reverse(
                "performance:load_file",
                args=[trading_file.pk],
            )
        )

        self.assertEqual(response.status_code, 404)

    def test_failed_load_clears_previous_cleaned_data(self):
        invalid_file = TradingFile.objects.create(
            user=self.user,
            file=self._invalid_upload("broken-load.csv"),
            status="error",
        )

        session = self.client.session
        session["cleaned_data"] = pd.DataFrame(
            {
                "Symbol": ["OLD"],
                "Profit": [999],
            }
        ).to_json(
            orient="split",
            date_format="iso",
        )
        session.save()

        response = self.client.get(
            reverse(
                "performance:load_file",
                args=[invalid_file.pk],
            )
        )

        self.assertEqual(response.status_code, 302)
        self.assertNotIn(
            "cleaned_data",
            self.client.session,
        )

    def test_upload_session_dashboard_round_trip(self):
        upload_response = self.client.post(
            reverse("performance:upload_file"),
            {"file": self._valid_upload()},
        )
        self.assertEqual(
            upload_response.status_code,
            302,
        )

        dashboard_response = self.client.get(
            reverse("performance:dashboard")
        )

        self.assertEqual(
            dashboard_response.status_code,
            200,
        )
        self.assertEqual(
            dashboard_response.context["kpis"]["Total Trades"],
            2,
        )
        self.assertEqual(
            dashboard_response.context["kpis"]["Total Profit"],
            "60.00",
        )

    def test_upload_session_kpi_report_round_trip(self):
        upload_response = self.client.post(
            reverse("performance:upload_file"),
            {"file": self._valid_upload()},
        )
        self.assertEqual(
            upload_response.status_code,
            302,
        )

        response = self.client.get(
            reverse("performance:kpi_report")
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["kpis"]["Total Trades"],
            2,
        )
        self.assertEqual(
            response.context["kpis"]["Total Profit"],
            "60.00",
        )
