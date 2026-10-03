from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from performance.analytical_queries import (
    ENGINE_CONTRACT_VERSION,
    AnalyticalIssueCode,
    AnalyticalQueryId,
    AnalyticalQueryStatus,
    AnalyticalResultKind,
    AnalyticalWarningCode,
    canonical_result_json,
    execute_analytical_query,
)
from performance.analytical_query_inspector import (
    QUERY_CATALOGUE,
    ROW_IDENTITY_PREFIX,
    STATE_JOURNAL_INVALID,
    STATE_NO_ACTIVE_JOURNAL,
    STATE_READY,
    STATE_REQUEST_INVALID,
    STATE_RESULT_AVAILABLE,
    STATE_WORKFLOW_ERROR,
)

INSPECTOR_PATH = Path(__file__).resolve().with_name("analytical_query_inspector.py")
TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "performance" / "analytical_query_inspector.html"
)


class SecretEngineError(Exception):
    def __str__(self) -> str:
        return "SECRET_ENGINE_FAILURE_TOKEN"


class AnalyticalQueryInspectorTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(
            username="analytical-inspector-user",
            password="test-password-123",
        )
        self.client.force_login(self.user)
        self.url = reverse("performance:analytical_query_inspector")

    def _journal(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open Time": [
                    "2024-01-15 10:00:00",
                    "2024-01-16 11:00:00",
                    "2024-02-01 09:00:00",
                ],
                "Symbol": ["US100", "DE40", "US100"],
                "Profit": [10.0, -3.25, 5.0],
                "Commission": [1.0, 1.0, 1.0],
                "Swap": [0.2, 0.2, 0.2],
            }
        )

    def _store_journal(self, frame: pd.DataFrame | None = None, filename: str | None = None) -> None:
        journal = frame if frame is not None else self._journal()
        session = self.client.session
        session["cleaned_data"] = journal.to_json(orient="split", date_format="iso")
        if filename is not None:
            session["last_uploaded_file"] = filename
        session.save()

    def _run(self, **params):
        return self.client.get(self.url, params)

    def test_anonymous_redirects_to_login(self) -> None:
        self.client.logout()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_authenticated_non_staff_allowed(self) -> None:
        self.assertFalse(self.user.is_staff)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)

    def test_staff_allowed(self) -> None:
        staff = User.objects.create_user(
            username="analytical-staff",
            password="test-password-123",
            is_staff=True,
        )
        self.client.force_login(staff)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)

    def test_post_method_not_allowed(self) -> None:
        response = self.client.post(self.url, {"query_id": "TRADE_COUNT"})
        self.assertEqual(response.status_code, 405)

    def test_named_url_and_catalogue(self) -> None:
        self.assertEqual(self.url, "/performance/analytical-query-inspector/")
        self.assertEqual(
            [item[0] for item in QUERY_CATALOGUE],
            [item.value for item in AnalyticalQueryId],
        )
        self.assertNotIn("PNL_BY_DAY", [item[0] for item in QUERY_CATALOGUE])

    def test_initial_get_with_journal_is_ready_and_does_not_execute(self) -> None:
        self._store_journal()
        with patch("performance.analytical_query_inspector.execute_analytical_query") as mocked:
            response = self.client.get(self.url)
        mocked.assert_not_called()
        self.assertEqual(response.context["inspector_state"], STATE_READY)
        self.assertIsNone(response.context["analytical_result"])
        self.assertContains(response, "READY")

    def test_symbol_only_does_not_execute(self) -> None:
        self._store_journal()
        with patch("performance.analytical_query_inspector.execute_analytical_query") as mocked:
            response = self._run(symbol="US100")
        mocked.assert_not_called()
        self.assertEqual(response.context["inspector_state"], STATE_READY)
        self.assertEqual(response.context["form_symbol"], "US100")

    def test_no_cleaned_data_is_no_active_journal(self) -> None:
        response = self.client.get(self.url)
        self.assertEqual(response.context["inspector_state"], STATE_NO_ACTIVE_JOURNAL)
        self.assertIsNone(response.context["analytical_result"])
        self.assertContains(response, "Load or upload an active journal before running an analytical query.")

    def test_empty_cleaned_data_is_no_active_journal(self) -> None:
        session = self.client.session
        session["cleaned_data"] = "   "
        session.save()
        response = self._run(query_id="TRADE_COUNT")
        self.assertEqual(response.context["inspector_state"], STATE_NO_ACTIVE_JOURNAL)
        self.assertIsNone(response.context["analytical_result"])

    def test_malformed_journal_is_invalid_without_details(self) -> None:
        session = self.client.session
        session["cleaned_data"] = "{not-valid-json"
        session.save()
        response = self._run(query_id="TRADE_COUNT")
        self.assertEqual(response.context["inspector_state"], STATE_JOURNAL_INVALID)
        self.assertIsNone(response.context["analytical_result"])
        body = response.content.decode()
        self.assertNotIn("{not-valid-json", body)
        self.assertNotIn("JSONDecodeError", body)
        self.assertNotIn("Traceback", body)

    def test_duplicate_query_id_is_request_invalid(self) -> None:
        self._store_journal()
        with patch("performance.analytical_query_inspector.execute_analytical_query") as mocked:
            response = self.client.get(
                self.url,
                {"query_id": ["TRADE_COUNT", "PROFIT_BY_DAY"]},
            )
        mocked.assert_not_called()
        self.assertEqual(response.context["inspector_state"], STATE_REQUEST_INVALID)
        self.assertIsNone(response.context["analytical_result"])

    def test_duplicate_symbol_is_request_invalid(self) -> None:
        self._store_journal()
        with patch("performance.analytical_query_inspector.execute_analytical_query") as mocked:
            response = self.client.get(self.url, {"query_id": "TRADE_COUNT", "symbol": ["US100", "DE40"]})
        mocked.assert_not_called()
        self.assertEqual(response.context["inspector_state"], STATE_REQUEST_INVALID)

    def test_unsupported_query_uses_engine_status(self) -> None:
        self._store_journal()
        response = self._run(query_id="PNL_BY_DAY")
        result = response.context["analytical_result"]
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        self.assertEqual(result.status, AnalyticalQueryStatus.UNSUPPORTED_QUERY)
        self.assertContains(response, "UNSUPPORTED_QUERY")

    def test_exact_supported_query_id_executes(self) -> None:
        self._store_journal()
        response = self._run(query_id="TRADE_COUNT")
        result = response.context["analytical_result"]
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.query_id, "TRADE_COUNT")
        self.assertEqual(response.context["form_query_id"], "TRADE_COUNT")

    def test_spaced_supported_query_id_is_unsupported(self) -> None:
        self._store_journal()
        response = self._run(query_id=" TRADE_COUNT ")
        result = response.context["analytical_result"]
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        self.assertEqual(result.status, AnalyticalQueryStatus.UNSUPPORTED_QUERY)
        self.assertEqual(result.query_id, " TRADE_COUNT ")
        self.assertEqual(response.context["form_query_id"], " TRADE_COUNT ")

    def test_lowercase_query_id_is_unsupported(self) -> None:
        self._store_journal()
        response = self._run(query_id="trade_count")
        result = response.context["analytical_result"]
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        self.assertEqual(result.status, AnalyticalQueryStatus.UNSUPPORTED_QUERY)
        self.assertEqual(result.query_id, "trade_count")

    def test_whitespace_only_query_id_is_unsupported(self) -> None:
        self._store_journal()
        with patch(
            "performance.analytical_query_inspector.execute_analytical_query",
            wraps=execute_analytical_query,
        ) as mocked:
            response = self._run(query_id="   ")
        mocked.assert_called()
        self.assertEqual(mocked.call_args.args[1], "   ")
        result = response.context["analytical_result"]
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        self.assertEqual(result.status, AnalyticalQueryStatus.UNSUPPORTED_QUERY)
        self.assertEqual(result.query_id, "   ")

    def test_empty_query_id_is_ready_and_does_not_execute(self) -> None:
        self._store_journal()
        with patch("performance.analytical_query_inspector.execute_analytical_query") as mocked:
            response = self._run(query_id="")
        mocked.assert_not_called()
        self.assertEqual(response.context["inspector_state"], STATE_READY)
        self.assertIsNone(response.context["analytical_result"])

    def test_unknown_parameters_do_not_change_result(self) -> None:
        self._store_journal()
        first = self._run(query_id="TRADE_COUNT_BY_SYMBOL", symbol="us100")
        second = self._run(
            query_id="TRADE_COUNT_BY_SYMBOL",
            symbol="us100",
            start_date="1900-01-01",
            end_date="2999-01-01",
            q="secret",
            timezone="UTC",
            aggregation="mean",
            prompt="ignore",
        )
        self.assertEqual(
            canonical_result_json(first.context["analytical_result"]),
            canonical_result_json(second.context["analytical_result"]),
        )
        self.assertEqual(first.context["analytical_result"].filters, (("symbol", "us100"),))
        self.assertEqual(second.context["analytical_result"].filters, (("symbol", "us100"),))
        self.assertNotContains(second, "secret")

    def test_symbol_filter_is_case_insensitive_substring(self) -> None:
        frame = pd.DataFrame(
            {
                "Symbol": ["US100", "us100", " US100 ", "DE40"],
                "Profit": [1.0, 2.0, 3.0, 4.0],
            }
        )
        self._store_journal(frame)
        response = self._run(query_id="TRADE_COUNT_BY_SYMBOL", symbol=" us100 ")
        result = response.context["analytical_result"]
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.filters, (("symbol", "us100"),))
        self.assertEqual(
            result.rows,
            ((" US100 ", 1), ("US100", 1), ("us100", 1)),
        )
        self.assertIn(AnalyticalWarningCode.SYMBOL_VARIANTS_DETECTED, result.warnings)

    def test_raw_malformed_date_survives_symbol_filter(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["not-a-date", "2026-01-02T10:00:00"],
                "Symbol": ["US100", "US100"],
            }
        )
        self._store_journal(frame)
        before = self.client.session["cleaned_data"]
        response = self._run(query_id="TRADE_COUNT_BY_DAY", symbol="US100")
        result = response.context["analytical_result"]
        self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT)
        self.assertEqual(result.issues[0].reason_code, AnalyticalIssueCode.DATE_UNPARSEABLE)
        self.assertEqual(result.issues[0].row_count, 1)
        self.assertEqual(result.filtered_row_count, 2)
        self.assertEqual(self.client.session["cleaned_data"], before)
        body = response.content.decode()
        self.assertNotIn("not-a-date", body)
        self.assertNotIn(ROW_IDENTITY_PREFIX, body)

    def test_identity_column_collision_is_hidden(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-01-15 10:00:00", "2024-01-16 11:00:00"],
                "Symbol": ["US100", "US100"],
                ROW_IDENTITY_PREFIX: ["LEAK-A", "LEAK-B"],
            }
        )
        self._store_journal(frame)
        response = self._run(query_id="TRADE_COUNT")
        result = response.context["analytical_result"]
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.scalar, 2)
        body = response.content.decode()
        self.assertNotIn(ROW_IDENTITY_PREFIX, body)
        self.assertNotIn("LEAK-A", body)
        self.assertNotIn("LEAK-B", body)

    def test_unmatched_symbol_is_no_data_not_no_active_journal(self) -> None:
        self._store_journal()
        response = self._run(query_id="TRADE_COUNT", symbol="NO_SUCH_SYMBOL")
        result = response.context["analytical_result"]
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        self.assertEqual(result.status, AnalyticalQueryStatus.NO_DATA)
        self.assertGreater(result.source_row_count, 0)
        self.assertEqual(result.filtered_row_count, 0)
        self.assertNotEqual(response.context["inspector_state"], STATE_NO_ACTIVE_JOURNAL)

    def test_each_catalogue_query_returns_ok(self) -> None:
        self._store_journal()
        for query_id, _label in QUERY_CATALOGUE:
            response = self._run(query_id=query_id)
            result = response.context["analytical_result"]
            self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE, query_id)
            self.assertEqual(result.status, AnalyticalQueryStatus.OK, query_id)

    def test_trade_count_scalar_and_units(self) -> None:
        self._store_journal()
        response = self._run(query_id="TRADE_COUNT")
        result = response.context["analytical_result"]
        self.assertEqual(result.result_kind, AnalyticalResultKind.SCALAR)
        self.assertEqual(result.scalar, 3)
        self.assertContains(response, "Scalar value: 3")
        self.assertContains(response, "trades")

    def test_tied_days_show_all_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-01-15 10:00:00", "2024-01-16 11:00:00"],
                "Symbol": ["US100", "DE40"],
            }
        )
        self._store_journal(frame)
        response = self._run(query_id="BUSIEST_TRADING_DAY")
        result = response.context["analytical_result"]
        self.assertTrue(result.tie)
        self.assertEqual(result.rows, (("2024-01-15", 1), ("2024-01-16", 1)))
        self.assertContains(response, "All rows tied for the maximum count are shown.")
        self.assertNotContains(response, "best")
        self.assertNotContains(response, "winner")

    def test_tied_symbols_show_all_rows(self) -> None:
        frame = pd.DataFrame({"Symbol": ["DE40", "US100"]})
        self._store_journal(frame)
        response = self._run(query_id="MOST_TRADED_SYMBOL")
        result = response.context["analytical_result"]
        self.assertTrue(result.tie)
        self.assertEqual(result.rows, (("DE40", 1), ("US100", 1)))
        self.assertContains(response, "All rows tied for the maximum count are shown.")

    def test_profit_display_excludes_commission_and_swap(self) -> None:
        self._store_journal()
        response = self._run(query_id="PROFIT_BY_SYMBOL")
        result = response.context["analytical_result"]
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertIn(AnalyticalWarningCode.PROFIT_EXCLUDES_COMMISSION_AND_SWAP, result.warnings)
        self.assertContains(response, "15.00")
        self.assertContains(response, "Commission and Swap are not added")
        self.assertNotContains(response, "P&L")

    def test_malformed_profit_shows_issue_counts_only(self) -> None:
        frame = pd.DataFrame(
            {
                "Symbol": ["US100"],
                "Profit": pd.Series(["bad-profit-value"], dtype="object"),
            }
        )
        self._store_journal(frame)
        response = self._run(query_id="PROFIT_BY_SYMBOL")
        result = response.context["analytical_result"]
        self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT)
        self.assertEqual(result.issues[0].reason_code, AnalyticalIssueCode.PROFIT_NOT_NUMERIC)
        self.assertContains(response, "PROFIT_NOT_NUMERIC")
        self.assertContains(response, "row_count")
        self.assertNotContains(response, "bad-profit-value")

    def test_date_metadata_and_recorded_wall_clock_disclosure(self) -> None:
        self._store_journal()
        response = self._run(query_id="TRADE_COUNT_BY_DAY")
        result = response.context["analytical_result"]
        self.assertEqual(result.date_column, "Open Time")
        self.assertEqual(result.grouping_basis, "JOURNAL_RECORDED_WALL_CLOCK")
        self.assertContains(response, "Open Time")
        self.assertContains(response, "No timezone conversion is applied")
        self.assertContains(response, "Dates are grouped as recorded in the journal")
        self.assertNotContains(response, "timezone conversion was applied to UTC")

    def test_provenance_hash_is_short_only(self) -> None:
        self._store_journal()
        response = self._run(query_id="TRADE_COUNT")
        result = response.context["analytical_result"]
        full_hash = result.provenance.input_sha256
        self.assertIsNotNone(full_hash)
        assert full_hash is not None
        short_hash = full_hash[:12]
        self.assertContains(response, short_hash)
        self.assertContains(response, ENGINE_CONTRACT_VERSION)
        self.assertContains(response, "Source rows")
        self.assertContains(response, "Filtered rows")
        self.assertNotContains(response, full_hash)
        self.assertNotContains(response, canonical_result_json(result))

    def test_duplicate_warning_and_count_without_row_dump(self) -> None:
        row = {
            "Open Time": "2024-01-15 10:00:00",
            "Symbol": "US100",
            "Profit": 1.5,
            "Note": "duplicate-marker-xyz",
        }
        self._store_journal(pd.DataFrame([row, row]))
        response = self._run(query_id="TRADE_COUNT")
        result = response.context["analytical_result"]
        self.assertIn(AnalyticalWarningCode.DUPLICATE_ROWS_DETECTED, result.warnings)
        self.assertEqual(result.warning_counts, (("DUPLICATE_ROWS_DETECTED", 2),))
        self.assertContains(response, "DUPLICATE_ROWS_DETECTED")
        self.assertContains(response, "Count: 2")
        self.assertNotContains(response, "duplicate-marker-xyz")

    def test_claim_safety_wording(self) -> None:
        self._store_journal()
        response = self._run(query_id="TRADE_COUNT")
        self.assertContains(
            response,
            "Deterministic means the same supported inputs and query contract",
        )
        self.assertContains(response, "It does not verify that the underlying")
        self.assertContains(response, "No LLM, external AI provider, RAG retrieval")
        body = response.content.decode().lower()
        self.assertNotIn("ai insight", body)
        self.assertNotIn("trading signal", body)
        self.assertNotIn("prediction", body)
        self.assertNotIn("verified journal", body)
        self.assertNotIn("financially reliable", body)

    def test_symbol_and_result_values_are_escaped(self) -> None:
        payload = "<script>alert(1)</script>"
        image = "<img src=x onerror=alert(1)>"
        frame = pd.DataFrame({"Symbol": [payload, image], "Profit": [1.0, 2.0]})
        self._store_journal(frame)
        response = self._run(query_id="TRADE_COUNT_BY_SYMBOL")
        body = response.content.decode()
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", body)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", body)
        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertNotIn("<img src=x onerror=alert(1)>", body)

    def test_source_filename_basename_is_escaped(self) -> None:
        self._store_journal(
            filename=r"trading_files\user_4\<img src=x onerror=alert(1)>.csv",
        )
        response = self.client.get(self.url)
        body = response.content.decode()
        self.assertNotIn(r"trading_files\user_4", body)
        self.assertNotIn("trading_files/user_4", body)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;.csv", body)
        self.assertNotIn("<img src=x onerror=alert(1)>.csv", body)

    def test_user_b_does_not_inherit_user_a_journal(self) -> None:
        self._store_journal()
        first = self._run(query_id="TRADE_COUNT_BY_SYMBOL")
        self.assertEqual(first.context["analytical_result"].status, AnalyticalQueryStatus.OK)
        self.assertContains(first, "US100")
        self.client.logout()
        other = User.objects.create_user(
            username="analytical-user-b",
            password="test-password-123",
        )
        self.client.force_login(other)
        second = self._run(query_id="TRADE_COUNT_BY_SYMBOL")
        self.assertEqual(second.context["inspector_state"], STATE_NO_ACTIVE_JOURNAL)
        self.assertIsNone(second.context["analytical_result"])
        self.assertNotContains(second, "US100")

    def test_workflow_error_hides_secret_exception(self) -> None:
        self._store_journal()
        with patch(
            "performance.analytical_query_inspector.execute_analytical_query",
            side_effect=SecretEngineError(),
        ):
            response = self._run(query_id="TRADE_COUNT")
        self.assertEqual(response.context["inspector_state"], STATE_WORKFLOW_ERROR)
        self.assertIsNone(response.context["analytical_result"])
        self.assertContains(response, "WORKFLOW_ERROR")
        self.assertNotContains(response, "SECRET_ENGINE_FAILURE_TOKEN")
        self.assertNotContains(response, "Traceback")

    def test_session_is_read_only(self) -> None:
        self._store_journal(filename="trading_files/user_4/history.csv")
        before = {
            "cleaned_data": self.client.session.get("cleaned_data"),
            "last_uploaded_file": self.client.session.get("last_uploaded_file"),
        }
        keys_before = set(self.client.session.keys())
        response = self._run(query_id="MONTHLY_PROFIT")
        self.assertEqual(response.context["inspector_state"], STATE_RESULT_AVAILABLE)
        after_session = self.client.session
        self.assertEqual(after_session.get("cleaned_data"), before["cleaned_data"])
        self.assertEqual(after_session.get("last_uploaded_file"), before["last_uploaded_file"])
        self.assertNotIn("analytical_query_result", after_session)
        extra = set(after_session.keys()) - keys_before
        self.assertFalse(any("analytical" in key for key in extra))

    def test_template_has_no_safe_bypass(self) -> None:
        source = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("|safe", source)
        self.assertNotIn("mark_safe", source)
        self.assertNotIn("format_html", source)
        self.assertIn("Back to Dashboard", source)
        self.assertIn("Run deterministic query", source)

    def test_ast_import_boundaries(self) -> None:
        tree = ast.parse(INSPECTOR_PATH.read_text(encoding="utf-8"))
        banned_roots = {"anthropic", "requests", "urllib", "http", "socket"}
        banned_modules = {
            "performance.models",
            "performance.views",
            "performance.time_basis",
            "performance.provider_inspector",
        }
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for name in imported:
            root = name.split(".", 1)[0]
            self.assertNotIn(root, banned_roots, name)
            self.assertNotIn(name, banned_modules, name)
            self.assertNotIn("ai_", name)
            self.assertNotIn("rag", name.lower())
        self.assertIn("performance.analytics", imported)
        self.assertIn("performance.analytical_queries", imported)
