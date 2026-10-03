from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from django.test import SimpleTestCase

from performance.analytical_queries import (
    ANALYTICAL_QUERY_RESULT_SCHEMA_VERSION,
    ENGINE_CONTRACT_VERSION,
    FROZEN_ISSUE_CODES,
    FROZEN_QUERY_IDS,
    FROZEN_RESULT_KINDS,
    FROZEN_STATUSES,
    FROZEN_WARNING_CODES,
    GROUPING_BASIS_RECORDED_WALL_CLOCK,
    AnalyticalInputContext,
    AnalyticalIssueCode,
    AnalyticalProvenance,
    AnalyticalQueryId,
    AnalyticalQueryResult,
    AnalyticalQueryStatus,
    AnalyticalResultKind,
    AnalyticalWarningCode,
    canonical_result_json,
    execute_analytical_query,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = Path(__file__).resolve().with_name("analytical_queries.py")


class ExplodingFrame:
    def __len__(self) -> int:
        raise AssertionError("DataFrame accessed")

    @property
    def columns(self) -> object:
        raise AssertionError("DataFrame accessed")

    def copy(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("DataFrame accessed")

    def __getattr__(self, name: str) -> object:
        raise AssertionError("DataFrame accessed: " + name)


def _ctx(frame: pd.DataFrame, source: int | None = None, filters: object = ()) -> AnalyticalInputContext:
    return AnalyticalInputContext(
        source_row_count=len(frame) if source is None else source,
        filtered_row_count=len(frame),
        filters=filters,
    )


def _run(frame: pd.DataFrame, query_id: object, source: int | None = None, filters: object = ()):
    return execute_analytical_query(frame, query_id, _ctx(frame, source=source, filters=filters))


class CatalogueTests(SimpleTestCase):
    def test_exact_eight_ids_and_no_aliases(self) -> None:
        expected = (
            "TRADE_COUNT",
            "TRADE_COUNT_BY_DAY",
            "BUSIEST_TRADING_DAY",
            "PROFIT_BY_DAY",
            "TRADE_COUNT_BY_SYMBOL",
            "MOST_TRADED_SYMBOL",
            "PROFIT_BY_SYMBOL",
            "MONTHLY_PROFIT",
        )
        self.assertEqual(FROZEN_QUERY_IDS, expected)
        self.assertEqual(tuple(item.value for item in AnalyticalQueryId), expected)
        self.assertFalse(hasattr(AnalyticalQueryId, "PNL_BY_DAY"))
        self.assertNotIn("PNL_BY_DAY", FROZEN_QUERY_IDS)
        self.assertNotIn("SUCCESS", FROZEN_STATUSES)
        self.assertNotIn("PARTIAL", FROZEN_STATUSES)
        self.assertNotIn("ERROR", FROZEN_STATUSES)

    def test_unknown_id_does_not_touch_dataframe(self) -> None:
        context = AnalyticalInputContext(source_row_count=4, filtered_row_count=4, filters={"a": "b"})
        result = execute_analytical_query(ExplodingFrame(), "PNL_BY_DAY", context)
        self.assertEqual(result.status, AnalyticalQueryStatus.UNSUPPORTED_QUERY)
        self.assertIsNone(result.result_kind)
        self.assertEqual(result.rows, ())
        self.assertIsNone(result.scalar)
        self.assertFalse(result.tie)
        self.assertEqual(result.provenance.columns_used, ())
        self.assertIsNone(result.provenance.input_sha256)
        self.assertEqual(result.source_row_count, 4)
        self.assertEqual(result.filtered_row_count, 4)
        self.assertEqual(result.filters, (("a", "b"),))


class EnumContractTests(SimpleTestCase):
    def test_frozen_sets(self) -> None:
        self.assertEqual(
            FROZEN_STATUSES,
            (
                "OK",
                "NO_DATA",
                "INVALID_INPUT",
                "REQUIRED_COLUMN_MISSING",
                "UNSUPPORTED_QUERY",
                "AMBIGUOUS_TIME_BASIS",
            ),
        )
        self.assertEqual(FROZEN_RESULT_KINDS, ("SCALAR", "ROWS", "TIED_MAX"))
        self.assertEqual(
            FROZEN_WARNING_CODES,
            (
                "DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION",
                "SYMBOL_VARIANTS_DETECTED",
                "DUPLICATE_ROWS_DETECTED",
                "PROFIT_EXCLUDES_COMMISSION_AND_SWAP",
            ),
        )
        self.assertEqual(
            FROZEN_ISSUE_CODES,
            (
                "PROFIT_MISSING",
                "PROFIT_NOT_NUMERIC",
                "PROFIT_NON_FINITE",
                "DATE_UNPARSEABLE",
                "SYMBOL_MISSING",
            ),
        )


class FilterContractTests(SimpleTestCase):
    def test_mapping_proxy_and_dict_are_sorted(self) -> None:
        payload = {"symbol": "xau", "q": "buy"}
        from_dict = AnalyticalInputContext(source_row_count=1, filtered_row_count=1, filters=payload)
        from_proxy = AnalyticalInputContext(
            source_row_count=1,
            filtered_row_count=1,
            filters=MappingProxyType(payload),
        )
        expected = (("q", "buy"), ("symbol", "xau"))
        self.assertEqual(from_dict.filters, expected)
        self.assertEqual(from_proxy.filters, expected)

    def test_tuple_of_pairs_accepted(self) -> None:
        context = AnalyticalInputContext(
            source_row_count=1,
            filtered_row_count=1,
            filters=(("symbol", "xau"), ("q", "buy")),
        )
        self.assertEqual(context.filters, (("q", "buy"), ("symbol", "xau")))

    def test_non_string_filters_rejected_with_value_error(self) -> None:
        cases = (
            {1: "xau"},
            {"symbol": 123},
            {"symbol": object()},
            (("symbol", "xau", "extra"),),
            (("symbol",),),
            (("symbol", "xau"), ("symbol", "us100")),
        )
        for filters in cases:
            with self.assertRaises(ValueError, msg=repr(filters)):
                AnalyticalInputContext(source_row_count=1, filtered_row_count=1, filters=filters)


class RowCountInvariantTests(SimpleTestCase):
    def test_filtered_cannot_exceed_source(self) -> None:
        with self.assertRaises(ValueError):
            AnalyticalInputContext(source_row_count=2, filtered_row_count=3)

    def test_equal_counts_accepted(self) -> None:
        context = AnalyticalInputContext(source_row_count=2, filtered_row_count=2)
        self.assertEqual(context.source_row_count, 2)
        self.assertEqual(context.filtered_row_count, 2)

    def test_zero_filtered_with_positive_source_accepted(self) -> None:
        context = AnalyticalInputContext(source_row_count=5, filtered_row_count=0)
        self.assertEqual(context.source_row_count, 5)
        self.assertEqual(context.filtered_row_count, 0)


class NoDataTests(SimpleTestCase):
    def test_source_zero_filtered_zero(self) -> None:
        frame = pd.DataFrame()
        result = _run(frame, AnalyticalQueryId.PROFIT_BY_DAY, source=0)
        self.assertEqual(result.status, AnalyticalQueryStatus.NO_DATA)
        self.assertEqual(result.source_row_count, 0)
        self.assertEqual(result.filtered_row_count, 0)
        self.assertIsNone(result.result_kind)
        self.assertEqual(result.rows, ())

    def test_source_positive_filtered_zero(self) -> None:
        frame = pd.DataFrame(columns=["Profit", "Symbol"])
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT, source=20)
        self.assertEqual(result.status, AnalyticalQueryStatus.NO_DATA)
        self.assertEqual(result.source_row_count, 20)
        self.assertEqual(result.filtered_row_count, 0)

    def test_empty_frame_without_required_columns_is_no_data(self) -> None:
        result = _run(pd.DataFrame(), AnalyticalQueryId.MONTHLY_PROFIT, source=3)
        self.assertEqual(result.status, AnalyticalQueryStatus.NO_DATA)


class ColumnRequirementTests(SimpleTestCase):
    def test_trade_count_succeeds_without_analytical_columns(self) -> None:
        frame = pd.DataFrame({"Note": ["a", "b"]})
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.scalar, 2)
        self.assertEqual(result.result_kind, AnalyticalResultKind.SCALAR)
        self.assertEqual(result.provenance.columns_used, ())

    def test_required_columns_missing(self) -> None:
        bare = pd.DataFrame({"Note": ["a"]})
        cases = (
            AnalyticalQueryId.TRADE_COUNT_BY_DAY,
            AnalyticalQueryId.BUSIEST_TRADING_DAY,
            AnalyticalQueryId.PROFIT_BY_DAY,
            AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL,
            AnalyticalQueryId.MOST_TRADED_SYMBOL,
            AnalyticalQueryId.PROFIT_BY_SYMBOL,
            AnalyticalQueryId.MONTHLY_PROFIT,
        )
        for query_id in cases:
            result = _run(bare, query_id)
            self.assertEqual(result.status, AnalyticalQueryStatus.REQUIRED_COLUMN_MISSING, query_id)
            self.assertEqual(result.rows, ())
            self.assertIsNone(result.scalar)

    def test_date_precedence_open_time_over_open_over_date(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-01-15 10:00:00"],
                "Open": ["2024-02-01 10:00:00"],
                "Date": ["2024-03-01 10:00:00"],
            }
        )
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.date_column, "Open Time")
        self.assertEqual(result.rows, (("2024-01-15", 1),))

        without_open_time = frame.drop(columns=["Open Time"])
        result = _run(without_open_time, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.date_column, "Open")
        self.assertEqual(result.rows, (("2024-02-01", 1),))

        date_only = without_open_time.drop(columns=["Open"])
        result = _run(date_only, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.date_column, "Date")
        self.assertEqual(result.rows, (("2024-03-01", 1),))

    def test_mismatch_filtered_count_raises(self) -> None:
        frame = pd.DataFrame({"Note": ["a", "b"]})
        context = AnalyticalInputContext(source_row_count=5, filtered_row_count=5)
        with self.assertRaises(ValueError):
            execute_analytical_query(frame, AnalyticalQueryId.TRADE_COUNT, context)


class TradeCountTests(SimpleTestCase):
    def test_integer_and_shuffle_invariant(self) -> None:
        frame = pd.DataFrame({"Note": ["a", "b", "c"], "Profit": [1, 2, 3]})
        first = _run(frame, AnalyticalQueryId.TRADE_COUNT)
        shuffled = _run(frame.sample(frac=1, random_state=7).reset_index(drop=True), AnalyticalQueryId.TRADE_COUNT)
        self.assertEqual(first.scalar, 3)
        self.assertEqual(first.status, AnalyticalQueryStatus.OK)
        self.assertEqual(canonical_result_json(first), canonical_result_json(shuffled))


class DateContractTests(SimpleTestCase):
    def test_recorded_wall_clock_and_adjacent_split(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": [
                    "2024-01-15 23:59:00",
                    "2024-01-16 00:01:00",
                ]
            }
        )
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.grouping_basis, GROUPING_BASIS_RECORDED_WALL_CLOCK)
        self.assertEqual(result.rows, (("2024-01-15", 1), ("2024-01-16", 1)))
        self.assertIn(
            AnalyticalWarningCode.DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION,
            result.warnings,
        )

    def test_month_boundary(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-01-31 23:59:00", "2024-02-01 00:01:00"],
                "Profit": [10, 5],
            }
        )
        result = _run(frame, AnalyticalQueryId.MONTHLY_PROFIT)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual([row[0] for row in result.rows], ["2024-01", "2024-02"])
        self.assertEqual(result.rows[0][2], "10.00")
        self.assertEqual(result.rows[1][2], "5.00")

    def test_invalid_and_null_dates(self) -> None:
        invalid = pd.DataFrame({"Open Time": ["not-a-date"]})
        result = _run(invalid, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT)
        self.assertEqual(result.rows, ())
        self.assertEqual(result.issues[0].reason_code, AnalyticalIssueCode.DATE_UNPARSEABLE)
        self.assertEqual(result.issues[0].row_count, 1)

        missing = pd.DataFrame({"Open Time": pd.Series([None, ""], dtype="object")})
        result = _run(missing, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT)
        self.assertEqual(result.issues[0].reason_code, AnalyticalIssueCode.DATE_UNPARSEABLE)
        self.assertEqual(result.issues[0].row_count, 2)

    def test_naive_only_accepted(self) -> None:
        frame = pd.DataFrame({"Open Time": [datetime(2024, 1, 15, 10, 0), datetime(2024, 1, 15, 18, 0)]})
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertIsNone(result.date_offset)
        self.assertEqual(result.rows, (("2024-01-15", 2),))

    def test_aware_common_offset_recorded_without_conversion(self) -> None:
        plus_five = timezone(timedelta(hours=5))
        frame = pd.DataFrame(
            {
                "Open Time": [
                    datetime(2024, 1, 15, 2, 0, tzinfo=plus_five),
                    datetime(2024, 1, 15, 3, 0, tzinfo=plus_five),
                ]
            }
        )
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.date_offset, "+05:00")
        self.assertEqual(result.rows, (("2024-01-15", 2),))
        self.assertNotEqual(result.rows[0][0], "2024-01-14")

    def test_mixed_offsets_ambiguous(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": [
                    datetime(2024, 1, 15, 10, 0, tzinfo=timezone(timedelta(hours=1))),
                    datetime(2024, 1, 15, 10, 0, tzinfo=timezone(timedelta(hours=-5))),
                ]
            }
        )
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.AMBIGUOUS_TIME_BASIS)
        self.assertEqual(result.rows, ())
        self.assertIsNone(result.result_kind)

    def test_naive_and_aware_ambiguous(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": [
                    datetime(2024, 1, 15, 10, 0),
                    datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
                ]
            }
        )
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.AMBIGUOUS_TIME_BASIS)
        self.assertIsNone(result.scalar)
        self.assertEqual(result.rows, ())

    def test_no_timezone_conversion_helpers_in_engine(self) -> None:
        source = ENGINE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("utc=True", source)
        self.assertNotIn("tz_convert(", source)
        self.assertNotIn("tz_localize(", source)

    def test_date_warning_on_every_successful_date_query(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-03-01 10:00:00", "2024-03-02 11:00:00"],
                "Symbol": ["US100", "DE40"],
                "Profit": [4.0, 6.0],
            }
        )
        queries = (
            AnalyticalQueryId.TRADE_COUNT_BY_DAY,
            AnalyticalQueryId.BUSIEST_TRADING_DAY,
            AnalyticalQueryId.PROFIT_BY_DAY,
            AnalyticalQueryId.MONTHLY_PROFIT,
        )
        for query_id in queries:
            result = _run(frame, query_id)
            self.assertEqual(result.status, AnalyticalQueryStatus.OK, query_id)
            self.assertIn(
                AnalyticalWarningCode.DATES_AS_RECORDED_NO_TIMEZONE_CONVERSION,
                result.warnings,
                query_id,
            )

    def test_whitespace_only_date_is_unparseable(self) -> None:
        frame = pd.DataFrame({"Open Time": pd.Series(["   "], dtype="object")})
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT)
        self.assertEqual(result.issues[0].reason_code, AnalyticalIssueCode.DATE_UNPARSEABLE)
        self.assertEqual(result.rows, ())
        self.assertIsNone(result.scalar)


class ProfitContractTests(SimpleTestCase):
    def test_valid_numbers_and_numeric_strings(self) -> None:
        frame = pd.DataFrame(
            {
                "Symbol": ["A", "B"],
                "Profit": [10, "2.5"],
                "Commission": [100, 100],
                "Swap": [50, 50],
            }
        )
        result = _run(frame, AnalyticalQueryId.PROFIT_BY_SYMBOL)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.rows[0][0], "A")
        self.assertEqual(result.rows[0][2], "10.00")
        self.assertEqual(result.rows[1][2], "2.50")
        self.assertIn(AnalyticalWarningCode.PROFIT_EXCLUDES_COMMISSION_AND_SWAP, result.warnings)
        self.assertNotIn("P&L", canonical_result_json(result))
        self.assertNotIn("GBP", canonical_result_json(result))
        self.assertNotIn("USD", canonical_result_json(result))

    def test_profit_classifications(self) -> None:
        cases = (
            (None, AnalyticalIssueCode.PROFIT_MISSING),
            (pd.NA, AnalyticalIssueCode.PROFIT_MISSING),
            ("", AnalyticalIssueCode.PROFIT_MISSING),
            ("   ", AnalyticalIssueCode.PROFIT_MISSING),
            ("abc", AnalyticalIssueCode.PROFIT_NOT_NUMERIC),
            (float("nan"), AnalyticalIssueCode.PROFIT_NON_FINITE),
            (float("inf"), AnalyticalIssueCode.PROFIT_NON_FINITE),
            (float("-inf"), AnalyticalIssueCode.PROFIT_NON_FINITE),
        )
        for value, code in cases:
            frame = pd.DataFrame({"Symbol": ["US100"], "Profit": pd.Series([value], dtype="object")})
            result = _run(frame, AnalyticalQueryId.PROFIT_BY_SYMBOL)
            self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT, msg=repr(value))
            self.assertEqual(result.rows, ())
            self.assertIsNone(result.scalar)
            self.assertEqual(result.issues[0].reason_code, code)
            self.assertEqual(result.issues[0].column, "Profit")
            self.assertEqual(result.issues[0].row_count, 1)
            encoded = canonical_result_json(result)
            self.assertNotIn("US100", encoded)
            if isinstance(value, str) and value.strip():
                self.assertNotIn(value, encoded)
            self.assertNotIn("row_index", encoded)

    def test_issue_counts_and_no_partial(self) -> None:
        frame = pd.DataFrame(
            {
                "Symbol": ["A", "B", "C", "D"],
                "Profit": pd.Series([None, "x", float("nan"), 10], dtype="object"),
            }
        )
        result = _run(frame, AnalyticalQueryId.PROFIT_BY_SYMBOL)
        self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT)
        self.assertEqual(result.rows, ())
        codes = {item.reason_code: item.row_count for item in result.issues}
        self.assertEqual(codes[AnalyticalIssueCode.PROFIT_MISSING], 1)
        self.assertEqual(codes[AnalyticalIssueCode.PROFIT_NOT_NUMERIC], 1)
        self.assertEqual(codes[AnalyticalIssueCode.PROFIT_NON_FINITE], 1)
        issue_order = [item.reason_code.value for item in result.issues]
        self.assertEqual(issue_order, sorted(issue_order))

    def test_count_queries_work_without_profit(self) -> None:
        frame = pd.DataFrame({"Symbol": ["US100", "DE40"]})
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(result.rows, (("DE40", 1), ("US100", 1)))

    def test_profit_warning_on_every_successful_profit_query(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-03-01 10:00:00"],
                "Symbol": ["US100"],
                "Profit": [10.0],
                "Commission": [2.0],
                "Swap": [1.0],
            }
        )
        queries = (
            AnalyticalQueryId.PROFIT_BY_DAY,
            AnalyticalQueryId.PROFIT_BY_SYMBOL,
            AnalyticalQueryId.MONTHLY_PROFIT,
        )
        for query_id in queries:
            result = _run(frame, query_id)
            self.assertEqual(result.status, AnalyticalQueryStatus.OK, query_id)
            self.assertIn(
                AnalyticalWarningCode.PROFIT_EXCLUDES_COMMISSION_AND_SWAP,
                result.warnings,
                query_id,
            )
            self.assertEqual(result.rows[0][1], 10.0)
            self.assertEqual(result.rows[0][2], "10.00")
            encoded = canonical_result_json(result)
            self.assertNotIn("P&L", encoded)
            self.assertNotIn("net P&L", encoded)


class FsumTests(SimpleTestCase):
    def test_point_one_two_three_and_half_even(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-01-15 10:00:00"] * 3,
                "Profit": [0.1, 0.2, 0.3],
            }
        )
        result = _run(frame, AnalyticalQueryId.PROFIT_BY_DAY)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        raw = result.rows[0][1]
        expected_raw = math.fsum(
            sorted([0.1, 0.2, 0.3], key=lambda item: (math.copysign(1.0, item), abs(item), repr(item)))
        )
        self.assertEqual(raw, expected_raw)
        display = Decimal(repr(raw)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        self.assertEqual(result.rows[0][2], format(display, "f"))
        self.assertEqual(result.rows[0][2], "0.60")

        shuffled = frame.sample(frac=1, random_state=3).reset_index(drop=True)
        again = _run(shuffled, AnalyticalQueryId.PROFIT_BY_DAY)
        self.assertEqual(result.rows[0][1], again.rows[0][1])
        self.assertEqual(canonical_result_json(result), canonical_result_json(again))

    def test_half_even_tie_boundaries_on_public_result(self) -> None:
        frame = pd.DataFrame(
            {
                "Symbol": ["EVEN", "ODD"],
                "Profit": [1.005, 1.015],
            }
        )
        result = _run(frame, AnalyticalQueryId.PROFIT_BY_SYMBOL)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(repr(result.rows[0][1]), "1.005")
        self.assertEqual(repr(result.rows[1][1]), "1.015")
        self.assertEqual(result.rows[0][2], "1.00")
        self.assertEqual(result.rows[1][2], "1.02")


class SymbolContractTests(SimpleTestCase):
    def test_exact_as_recorded_and_codepoint_order(self) -> None:
        frame = pd.DataFrame({"Symbol": ["us100", " US100 ", "US100", "DE40"]})
        result = _run(frame, AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL)
        self.assertEqual(result.status, AnalyticalQueryStatus.OK)
        self.assertEqual(
            result.rows,
            ((" US100 ", 1), ("DE40", 1), ("US100", 1), ("us100", 1)),
        )
        keys = [row[0] for row in result.rows]
        self.assertEqual(keys, sorted(keys))
        self.assertIn(AnalyticalWarningCode.SYMBOL_VARIANTS_DETECTED, result.warnings)

    def test_symbol_missing_cases(self) -> None:
        for value in (None, "", "   "):
            frame = pd.DataFrame({"Symbol": [value]})
            result = _run(frame, AnalyticalQueryId.MOST_TRADED_SYMBOL)
            self.assertEqual(result.status, AnalyticalQueryStatus.INVALID_INPUT, msg=repr(value))
            self.assertEqual(result.issues[0].reason_code, AnalyticalIssueCode.SYMBOL_MISSING)
            self.assertEqual(result.rows, ())


class DuplicateContractTests(SimpleTestCase):
    def test_duplicates_counted_and_warning(self) -> None:
        row = {"Symbol": "US100", "Profit": 1.5}
        two = pd.DataFrame([row, row])
        result = _run(two, AnalyticalQueryId.TRADE_COUNT)
        self.assertEqual(result.scalar, 2)
        self.assertIn(AnalyticalWarningCode.DUPLICATE_ROWS_DETECTED, result.warnings)
        self.assertEqual(result.warning_counts, (("DUPLICATE_ROWS_DETECTED", 2),))

        three = pd.DataFrame([row, row, row])
        result = _run(three, AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL)
        self.assertEqual(result.rows, (("US100", 3),))
        self.assertEqual(result.warning_counts, (("DUPLICATE_ROWS_DETECTED", 3),))

        one = pd.DataFrame([row])
        self.assertNotEqual(
            _run(two, AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL).provenance.input_sha256,
            _run(one, AnalyticalQueryId.TRADE_COUNT_BY_SYMBOL).provenance.input_sha256,
        )


class TieContractTests(SimpleTestCase):
    def test_tied_and_single_days(self) -> None:
        tied = pd.DataFrame({"Open Time": ["2024-01-15 10:00:00", "2024-01-16 10:00:00"]})
        result = _run(tied, AnalyticalQueryId.BUSIEST_TRADING_DAY)
        self.assertEqual(result.result_kind, AnalyticalResultKind.TIED_MAX)
        self.assertTrue(result.tie)
        self.assertEqual(result.rows, (("2024-01-15", 1), ("2024-01-16", 1)))

        single = pd.DataFrame({"Open Time": ["2024-01-15 10:00:00", "2024-01-15 11:00:00", "2024-01-16 10:00:00"]})
        result = _run(single, AnalyticalQueryId.BUSIEST_TRADING_DAY)
        self.assertFalse(result.tie)
        self.assertEqual(result.rows, (("2024-01-15", 2),))

    def test_tied_and_single_symbols(self) -> None:
        tied = pd.DataFrame({"Symbol": ["DE40", "US100"]})
        result = _run(tied, AnalyticalQueryId.MOST_TRADED_SYMBOL)
        self.assertTrue(result.tie)
        self.assertEqual(result.rows, (("DE40", 1), ("US100", 1)))

        single = pd.DataFrame({"Symbol": ["US100", "US100", "DE40"]})
        result = _run(single, AnalyticalQueryId.MOST_TRADED_SYMBOL)
        self.assertFalse(result.tie)
        self.assertEqual(result.rows, (("US100", 2),))


class HashingTests(SimpleTestCase):
    def test_repeat_and_shuffle_and_irrelevant_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "Open Time": ["2024-01-15 10:00:00", "2024-01-16 11:00:00"],
                "Symbol": ["US100", "DE40"],
                "Profit": [1.25, -0.5],
                "Note": ["a", "b"],
            }
        )
        first = _run(frame, AnalyticalQueryId.PROFIT_BY_DAY)
        second = _run(frame, AnalyticalQueryId.PROFIT_BY_DAY)
        self.assertEqual(first.provenance.input_sha256, second.provenance.input_sha256)
        shuffled = frame.sample(frac=1, random_state=11).reset_index(drop=True)
        shuffled_hash = _run(shuffled, AnalyticalQueryId.PROFIT_BY_DAY).provenance.input_sha256
        self.assertEqual(first.provenance.input_sha256, shuffled_hash)

        changed = frame.copy()
        changed.loc[0, "Profit"] = 9.0
        changed_hash = _run(changed, AnalyticalQueryId.PROFIT_BY_DAY).provenance.input_sha256
        self.assertNotEqual(first.provenance.input_sha256, changed_hash)

        extra = frame.copy()
        extra["Note"] = ["x", "y"]
        extra_hash = _run(extra, AnalyticalQueryId.PROFIT_BY_DAY).provenance.input_sha256
        self.assertEqual(first.provenance.input_sha256, extra_hash)
        self.assertEqual(first.provenance.columns_used, ("Open Time", "Profit"))

    def test_trade_count_hash_tracks_row_count_only(self) -> None:
        two = pd.DataFrame({"Note": ["a", "b"], "Profit": [1, 2]})
        three = pd.DataFrame({"Note": ["a", "b", "c"]})
        same_count_other_values = pd.DataFrame({"Other": [9, 8]})
        self.assertEqual(
            _run(two, AnalyticalQueryId.TRADE_COUNT).provenance.input_sha256,
            _run(same_count_other_values, AnalyticalQueryId.TRADE_COUNT).provenance.input_sha256,
        )
        self.assertNotEqual(
            _run(two, AnalyticalQueryId.TRADE_COUNT).provenance.input_sha256,
            _run(three, AnalyticalQueryId.TRADE_COUNT).provenance.input_sha256,
        )


class SerialisationTests(SimpleTestCase):
    def _fixture(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open Time": ["2024-01-15 10:00:00", "2024-01-16 11:00:00"],
                "Symbol": ["US100", "DE40"],
                "Profit": [10.5, -3.25],
            }
        )

    def test_repeat_and_shuffle_byte_identical(self) -> None:
        frame = self._fixture()
        context_filters = {"zeta": "2", "alpha": "1"}
        first = _run(frame, AnalyticalQueryId.PROFIT_BY_DAY, filters=context_filters)
        second = _run(frame, AnalyticalQueryId.PROFIT_BY_DAY, filters=context_filters)
        encoded = canonical_result_json(first)
        self.assertEqual(encoded, canonical_result_json(second))
        self.assertEqual(encoded, canonical_result_json(first))
        shuffled = frame.sample(frac=1, random_state=5).reset_index(drop=True)
        shuffled_json = canonical_result_json(
            _run(shuffled, AnalyticalQueryId.PROFIT_BY_DAY, filters=context_filters)
        )
        self.assertEqual(encoded, shuffled_json)
        payload = json.loads(encoded)
        self.assertEqual(payload["filters"], [["alpha", "1"], ["zeta", "2"]])
        self.assertNotIn("NaN", encoded)
        warnings = [item.value if hasattr(item, "value") else item for item in first.warnings]
        self.assertEqual(warnings, sorted(warnings))
        issue_keys = [(item.reason_code.value, item.column, item.row_count) for item in first.issues]
        self.assertEqual(issue_keys, sorted(issue_keys))

    def test_fresh_subprocess_canonical_json(self) -> None:
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import pandas as pd\n"
            "from performance.analytical_queries import (\n"
            "    AnalyticalInputContext,\n"
            "    AnalyticalQueryId,\n"
            "    canonical_result_json,\n"
            "    execute_analytical_query,\n"
            ")\n"
            "frame = pd.DataFrame({\n"
            "    'Open Time': ['2024-01-15 10:00:00', '2024-01-16 11:00:00'],\n"
            "    'Symbol': ['US100', 'DE40'],\n"
            "    'Profit': [10.5, -3.25],\n"
            "})\n"
            "context = AnalyticalInputContext(\n"
            "    source_row_count=2,\n"
            "    filtered_row_count=2,\n"
            "    filters={'zeta': '2', 'alpha': '1'},\n"
            ")\n"
            "result = execute_analytical_query(frame, AnalyticalQueryId.PROFIT_BY_DAY, context)\n"
            "sys.stdout.write(canonical_result_json(result))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(REPO_ROOT)],
            check=True,
            capture_output=True,
            text=True,
        )
        local = canonical_result_json(
            _run(self._fixture(), AnalyticalQueryId.PROFIT_BY_DAY, filters={"zeta": "2", "alpha": "1"})
        )
        self.assertEqual(completed.stdout, local)

    def test_versions_and_units(self) -> None:
        result = _run(self._fixture(), AnalyticalQueryId.PROFIT_BY_SYMBOL)
        self.assertEqual(result.schema_version, ANALYTICAL_QUERY_RESULT_SCHEMA_VERSION)
        self.assertEqual(result.provenance.engine_contract_version, ENGINE_CONTRACT_VERSION)
        self.assertEqual(
            result.units,
            (
                ("profit_raw", "account currency (as recorded)"),
                ("profit_display", "account currency (as recorded)"),
            ),
        )
        self.assertIsNone(result.grouping_basis)
        self.assertIsNone(result.date_column)
        self.assertIsNone(result.date_offset)


class ImmutabilityTests(SimpleTestCase):
    def test_frozen_objects_and_input_unchanged(self) -> None:
        frame = pd.DataFrame({"Symbol": ["US100", "DE40"], "Profit": [1.0, 2.0]})
        before = frame.copy(deep=True)
        context = _ctx(frame, filters={"b": "2", "a": "1"})
        result = execute_analytical_query(frame, AnalyticalQueryId.PROFIT_BY_SYMBOL, context)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(list(frame.columns), list(before.columns))
        self.assertEqual(list(frame.index), list(before.index))
        with self.assertRaises(Exception):
            context.source_row_count = 99  # type: ignore[misc]
        with self.assertRaises(Exception):
            result.scalar = 1  # type: ignore[misc]
        with self.assertRaises(Exception):
            result.provenance.columns_used = ()  # type: ignore[misc]
        self.assertIsInstance(result, AnalyticalQueryResult)
        self.assertIsInstance(result.provenance, AnalyticalProvenance)
        self.assertEqual(context.filters, (("a", "1"), ("b", "2")))


class PurityTests(SimpleTestCase):
    def test_ast_import_boundaries(self) -> None:
        tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
        banned_roots = {"django", "requests", "urllib", "http", "socket", "anthropic", "numpy"}
        banned_modules = {
            "performance.models",
            "performance.views",
            "performance.time_basis",
            "performance.analytics",
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
            self.assertNotIn("rag", name.lower() if name != "str" else name)

    def test_engine_source_has_no_forbidden_integrations(self) -> None:
        source = ENGINE_PATH.read_text(encoding="utf-8")
        for token in ("apply_analysis", "request.session", "django.db", "anthropic", "requests."):
            self.assertNotIn(token, source)
        for claim in ("verified", "hallucination-free", "financially reliable", "audited"):
            self.assertNotIn(claim, source.lower())
