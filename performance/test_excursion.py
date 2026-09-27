from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
from django.test import SimpleTestCase

from .excursion import (
    DURATION_TOLERANCE_SECONDS,
    MAX_TRADE_WINDOW_MINUTES,
    PRICE_COMPARISON_EPSILON,
    REASON_AMBIGUOUS,
    REASON_BOUNDARY_BAR_MISSING,
    REASON_INTERIOR_GAP,
    REASON_INVARIANT_FAILED,
    REASON_NONEXISTENT,
    REASON_OUTSIDE_FILE_RANGE,
    REASON_WINDOW_EXCEEDS_LIMIT,
    RUN_STATUS_JOIN_KEY_UNAVAILABLE,
    RUN_STATUS_OK,
    RUN_STATUS_TIME_BASIS_NOT_DECLARED,
    STATUS_COMPUTED,
    STATUS_INCOMPLETE_COVERAGE,
    STATUS_INVALID_TRADE_DATA,
    STATUS_INVARIANT_VIOLATION,
    STATUS_NO_MARKET_DATA,
    STATUS_TIME_BASIS_INCONSISTENT,
    STATUS_TIMEZONE_AMBIGUOUS,
    _index_market,
    compute_excursion_evidence,
    compute_journal_fingerprint,
    resolve_journal_columns,
)
from .time_basis import TIME_BASIS_IANA, TIME_BASIS_UTC, create_time_basis


class ExcursionEngineTests(SimpleTestCase):
    def setUp(self):
        self.utc = create_time_basis(TIME_BASIS_UTC)

    def _bars(self, symbol, start, count, *, high=None, low=None, step_minutes=1):
        rows = []
        cursor = start
        for index in range(count):
            top = 110 + index if high is None else high[index]
            bottom = 90 - index if low is None else low[index]
            rows.append(
                {
                    "TimestampUTC": cursor.replace(tzinfo=timezone.utc),
                    "Symbol": symbol,
                    "Open": 100.0,
                    "High": float(top),
                    "Low": float(bottom),
                    "Close": 101.0,
                }
            )
            cursor = cursor + timedelta(minutes=step_minutes)
        return pd.DataFrame(rows)

    def _journal(self, rows):
        return pd.DataFrame(rows)

    def _trade(self, **overrides):
        row = {
            "Ticket": 1,
            "Open Time": datetime(2026, 6, 25, 10, 30, 15),
            "Close Time": datetime(2026, 6, 25, 10, 32, 47),
            "Symbol": "US100.cash",
            "Type": "buy",
            "Entry": 100.0,
            "Exit": 105.0,
            "Profit": 50.0,
            "Commission": 1.0,
            "Notes": "keep",
        }
        row.update(overrides)
        return row

    def _run(self, journal_rows, market, time_basis=None):
        if time_basis is None:
            time_basis = self.utc
        return compute_excursion_evidence(
            self._journal(journal_rows),
            market,
            time_basis,
        )

    def _evidence(self, result, ticket="1"):
        return result.evidence[ticket]

    def test_locked_literals(self):
        self.assertEqual(MAX_TRADE_WINDOW_MINUTES, 10_080)
        self.assertEqual(DURATION_TOLERANCE_SECONDS, 1)
        self.assertEqual(PRICE_COMPARISON_EPSILON, 1e-9)
        self.assertEqual(RUN_STATUS_OK, "OK")
        self.assertEqual(
            RUN_STATUS_TIME_BASIS_NOT_DECLARED,
            "TIME_BASIS_NOT_DECLARED",
        )
        self.assertEqual(
            RUN_STATUS_JOIN_KEY_UNAVAILABLE,
            "JOIN_KEY_UNAVAILABLE",
        )
        self.assertEqual(STATUS_INVALID_TRADE_DATA, "INVALID_TRADE_DATA")
        self.assertEqual(STATUS_TIMEZONE_AMBIGUOUS, "TIMEZONE_AMBIGUOUS")
        self.assertEqual(STATUS_TIME_BASIS_INCONSISTENT, "TIME_BASIS_INCONSISTENT")
        self.assertEqual(STATUS_NO_MARKET_DATA, "NO_MARKET_DATA")
        self.assertEqual(STATUS_INCOMPLETE_COVERAGE, "INCOMPLETE_COVERAGE")
        self.assertEqual(STATUS_INVARIANT_VIOLATION, "INVARIANT_VIOLATION")
        self.assertEqual(STATUS_COMPUTED, "COMPUTED")
        self.assertEqual(REASON_WINDOW_EXCEEDS_LIMIT, "WINDOW_EXCEEDS_LIMIT")
        self.assertEqual(REASON_OUTSIDE_FILE_RANGE, "OUTSIDE_FILE_RANGE")
        self.assertEqual(REASON_BOUNDARY_BAR_MISSING, "BOUNDARY_BAR_MISSING")
        self.assertEqual(REASON_INTERIOR_GAP, "INTERIOR_GAP")
        self.assertEqual(REASON_AMBIGUOUS, "AMBIGUOUS")
        self.assertEqual(REASON_NONEXISTENT, "NONEXISTENT")

    def test_join_key_rules(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 5)
        missing = compute_excursion_evidence(
            pd.DataFrame([{"Symbol": "US100.cash"}]),
            market,
            self.utc,
        )
        self.assertEqual(missing.run_status, RUN_STATUS_JOIN_KEY_UNAVAILABLE)
        self.assertEqual(dict(missing.evidence), {})
        self.assertIsNone(missing.journal_fingerprint)

        null_ticket = self._run([self._trade(Ticket=None)], market)
        self.assertEqual(null_ticket.run_status, RUN_STATUS_JOIN_KEY_UNAVAILABLE)
        blank = self._run([self._trade(Ticket="  ")], market)
        self.assertEqual(blank.run_status, RUN_STATUS_JOIN_KEY_UNAVAILABLE)
        duplicate = self._run(
            [self._trade(Ticket=1), self._trade(Ticket=1, Symbol="EURUSD")],
            market,
        )
        self.assertEqual(duplicate.run_status, RUN_STATUS_JOIN_KEY_UNAVAILABLE)
        numeric_dup = self._run(
            [self._trade(Ticket=123), self._trade(Ticket=123.0, Symbol="EURUSD")],
            market,
        )
        self.assertEqual(numeric_dup.run_status, RUN_STATUS_JOIN_KEY_UNAVAILABLE)

        alpha = self._run(
            [
                self._trade(Ticket="ABC-123"),
                self._trade(Ticket="abc-124", Symbol="EURUSD"),
            ],
            pd.concat(
                [
                    market,
                    self._bars("EURUSD", datetime(2026, 6, 25, 10, 30), 5),
                ],
                ignore_index=True,
            ),
        )
        self.assertEqual(alpha.run_status, RUN_STATUS_OK)
        self.assertIn("ABC-123", alpha.evidence)
        self.assertIn("abc-124", alpha.evidence)

    def test_column_resolution_precedence(self):
        journal = pd.DataFrame(
            [
                {
                    "Ticket": 9,
                    "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                    "Open": datetime(2026, 6, 25, 11, 30, 0),
                    "Close Time": datetime(2026, 6, 25, 10, 31, 0),
                    "Close": datetime(2026, 6, 25, 11, 31, 0),
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Side": "sell",
                    "Entry": 100.0,
                    "Price": 50.0,
                    "Exit": 101.0,
                    "Price.1": 40.0,
                }
            ]
        )
        columns = resolve_journal_columns(journal)
        self.assertEqual(columns.open_column, "Open Time")
        self.assertEqual(columns.close_column, "Close Time")
        self.assertEqual(columns.side_column, "Type")
        self.assertEqual(columns.entry_column, "Entry")
        self.assertEqual(columns.exit_column, "Exit")
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 3)
        result = compute_excursion_evidence(journal, market, self.utc)
        item = result.evidence["9"]
        self.assertEqual(item["status"], STATUS_COMPUTED)
        self.assertEqual(item["bars_used"], 2)

    def test_time_basis_and_dst(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 5)
        undeclared = compute_excursion_evidence(
            self._journal([self._trade()]),
            market,
            None,
        )
        self.assertEqual(
            undeclared.run_status,
            RUN_STATUS_TIME_BASIS_NOT_DECLARED,
        )
        self.assertEqual(dict(undeclared.evidence), {})
        self.assertIsNotNone(undeclared.journal_fingerprint)

        plus = create_time_basis("FIXED_OFFSET", offset_minutes=120)
        minus = create_time_basis("FIXED_OFFSET", offset_minutes=-300)
        london = create_time_basis(TIME_BASIS_IANA, zone="Europe/London")
        utc_item = self._evidence(self._run([self._trade()], market))
        self.assertEqual(utc_item["status"], STATUS_COMPUTED)
        plus_item = self._evidence(
            compute_excursion_evidence(
                self._journal(
                    [
                        self._trade(
                            **{
                                "Open Time": datetime(2026, 6, 25, 12, 30, 15),
                                "Close Time": datetime(2026, 6, 25, 12, 32, 47),
                            }
                        )
                    ]
                ),
                market,
                plus,
            )
        )
        self.assertEqual(plus_item["status"], STATUS_COMPUTED)
        minus_market = self._bars("US100.cash", datetime(2026, 6, 25, 12, 30), 5)
        minus_item = self._evidence(
            compute_excursion_evidence(
                self._journal(
                    [
                        self._trade(
                            **{
                                "Open Time": datetime(2026, 6, 25, 7, 30, 15),
                                "Close Time": datetime(2026, 6, 25, 7, 32, 47),
                            }
                        )
                    ]
                ),
                minus_market,
                minus,
            )
        )
        self.assertEqual(minus_item["status"], STATUS_COMPUTED)

        london_market = pd.concat(
            [
                self._bars("US100.cash", datetime(2026, 7, 15, 11, 0), 5),
                self._bars("US100.cash", datetime(2026, 3, 29, 0, 0), 4),
                self._bars("US100.cash", datetime(2026, 10, 25, 0, 0), 4),
            ],
            ignore_index=True,
        )
        mixed = compute_excursion_evidence(
            self._journal(
                [
                    self._trade(
                        Ticket=1,
                        **{
                            "Open Time": datetime(2026, 7, 15, 12, 0, 15),
                            "Close Time": datetime(2026, 7, 15, 12, 2, 0),
                        },
                    ),
                    self._trade(
                        Ticket=2,
                        **{
                            "Open Time": datetime(2026, 10, 25, 1, 30, 0),
                            "Close Time": datetime(2026, 10, 25, 1, 31, 0),
                        },
                    ),
                    self._trade(
                        Ticket=3,
                        **{
                            "Open Time": datetime(2026, 3, 29, 1, 30, 0),
                            "Close Time": datetime(2026, 3, 29, 1, 31, 0),
                        },
                    ),
                ]
            ),
            london_market,
            london,
        )
        self.assertEqual(mixed.evidence["1"]["status"], STATUS_COMPUTED)
        self.assertEqual(mixed.evidence["2"]["status"], STATUS_TIMEZONE_AMBIGUOUS)
        self.assertEqual(mixed.evidence["2"]["reason_code"], REASON_AMBIGUOUS)
        self.assertEqual(mixed.evidence["3"]["status"], STATUS_TIMEZONE_AMBIGUOUS)
        self.assertEqual(mixed.evidence["3"]["reason_code"], REASON_NONEXISTENT)
        for ticket in ("2", "3"):
            item = mixed.evidence[ticket]
            self.assertIsNone(item["interval_high"])
            self.assertIsNone(item["interval_low"])
            self.assertIsNone(item["mfe"])
            self.assertIsNone(item["mae"])

    def test_trade_validation(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 5)
        cases = [
            ({"Type": "hold"}, "INVALID_SIDE"),
            ({"Symbol": "   "}, "BLANK_SYMBOL"),
            ({"Entry": "abc"}, "INVALID_ENTRY"),
            ({"Entry": 0}, "NON_POSITIVE_ENTRY"),
            ({"Entry": -1}, "NON_POSITIVE_ENTRY"),
            ({"Exit": "xyz"}, "INVALID_EXIT"),
            ({"Exit": 0}, "NON_POSITIVE_EXIT"),
            ({"Open Time": "not-a-date"}, "INVALID_OPEN"),
            ({"Close Time": "nope"}, "INVALID_CLOSE"),
            (
                {
                    "Open Time": datetime(2026, 6, 25, 10, 32, 0),
                    "Close Time": datetime(2026, 6, 25, 10, 30, 0),
                },
                "CLOSE_BEFORE_OPEN",
            ),
            (
                {
                    "Open Time": datetime(
                        2026, 6, 25, 10, 30, tzinfo=timezone.utc
                    )
                },
                "AWARE_TIMESTAMP",
            ),
        ]
        for overrides, reason in cases:
            item = self._evidence(self._run([self._trade(**overrides)], market))
            self.assertEqual(item["status"], STATUS_INVALID_TRADE_DATA)
            self.assertEqual(item["reason_code"], reason)
            self.assertIsNone(item["mfe"])
            self.assertIsNone(item["mae"])
            self.assertIsNone(item["interval_high"])
            self.assertIsNone(item["interval_low"])

    def test_status_precedence_combinations(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 4)
        long_window = {
            "Open Time": datetime(2026, 1, 1, 0, 0, 0),
            "Close Time": datetime(2026, 1, 8, 0, 1, 0),
        }
        no_market = self._evidence(
            self._run([self._trade(Symbol="NAS100", **long_window)], market)
        )
        self.assertEqual(no_market["status"], STATUS_NO_MARKET_DATA)
        self.assertNotEqual(no_market["status"], STATUS_INCOMPLETE_COVERAGE)

        invalid = self._evidence(
            self._run(
                [self._trade(Type="hold", Symbol="NAS100", **long_window)],
                market,
            )
        )
        self.assertEqual(invalid["status"], STATUS_INVALID_TRADE_DATA)

        duration_row = self._trade(Symbol="NAS100", **long_window)
        duration_row["Trade duration in seconds"] = 10
        duration = self._evidence(self._run([duration_row], market))
        self.assertEqual(duration["status"], STATUS_TIME_BASIS_INCONSISTENT)

        coverage = self._evidence(
            self._run([self._trade(Symbol="US100.cash", **long_window)], market)
        )
        self.assertEqual(coverage["status"], STATUS_INCOMPLETE_COVERAGE)
        self.assertEqual(coverage["reason_code"], REASON_WINDOW_EXCEEDS_LIMIT)

        london = create_time_basis(TIME_BASIS_IANA, zone="Europe/London")
        ambiguous = self._evidence(
            compute_excursion_evidence(
                self._journal(
                    [
                        self._trade(
                            Symbol="NAS100",
                            **{
                                "Open Time": datetime(2026, 10, 25, 1, 30, 0),
                                "Close Time": datetime(2026, 11, 2, 0, 0, 0),
                            },
                        )
                    ]
                ),
                market,
                london,
            )
        )
        self.assertEqual(ambiguous["status"], STATUS_TIMEZONE_AMBIGUOUS)
        self.assertEqual(ambiguous["reason_code"], REASON_AMBIGUOUS)

    def test_duration_rules(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 6)
        base = self._trade(
            **{
                "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                "Close Time": datetime(2026, 6, 25, 10, 35, 0),
            }
        )
        absent = self._evidence(self._run([base], market))
        self.assertEqual(absent["status"], STATUS_COMPUTED)

        def with_duration(value):
            row = dict(base)
            row["Trade duration in seconds"] = value
            return self._evidence(self._run([row], market))

        self.assertEqual(with_duration("  ")["status"], STATUS_COMPUTED)
        self.assertEqual(with_duration(300)["status"], STATUS_COMPUTED)
        self.assertEqual(with_duration(301)["status"], STATUS_COMPUTED)
        mismatch = with_duration(302)
        self.assertEqual(mismatch["status"], STATUS_TIME_BASIS_INCONSISTENT)
        self.assertIsNone(mismatch["mfe"])
        self.assertEqual(with_duration(-1)["status"], STATUS_INVALID_TRADE_DATA)
        self.assertEqual(with_duration("abc")["status"], STATUS_INVALID_TRADE_DATA)
        self.assertEqual(with_duration(float("inf"))["status"], STATUS_INVALID_TRADE_DATA)

    def test_symbol_and_windows(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 6)
        exact = self._evidence(self._run([self._trade()], market))
        self.assertEqual(exact["status"], STATUS_COMPUTED)
        trimmed = self._evidence(
            self._run([self._trade(Symbol=" US100.cash ")], market)
        )
        self.assertEqual(trimmed["status"], STATUS_COMPUTED)
        aliased = self._evidence(self._run([self._trade(Symbol="US100")], market))
        self.assertEqual(aliased["status"], STATUS_NO_MARKET_DATA)
        missing = self._evidence(self._run([self._trade(Symbol="NAS100")], market))
        self.assertEqual(missing["status"], STATUS_NO_MARKET_DATA)

        same = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 15),
                            "Close Time": datetime(2026, 6, 25, 10, 30, 40),
                        }
                    )
                ],
                market,
            )
        )
        self.assertEqual(same["bars_used"], 1)
        self.assertEqual(same["bars_expected"], 1)
        self.assertTrue(same["high_from_boundary_bar"])
        self.assertTrue(same["low_from_boundary_bar"])

        exact_close = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 34, 0),
                        }
                    )
                ],
                market,
            )
        )
        self.assertEqual(exact_close["bars_used"], 5)
        multi = self._evidence(self._run([self._trade()], market))
        self.assertEqual(multi["bars_used"], 3)

        allowed = self._run(
            [
                self._trade(
                    **{
                        "Open Time": datetime(2026, 1, 1, 0, 0, 0),
                        "Close Time": datetime(2026, 1, 8, 0, 0, 0),
                    }
                )
            ],
            market,
        )
        self.assertNotEqual(
            allowed.evidence["1"]["reason_code"],
            REASON_WINDOW_EXCEEDS_LIMIT,
        )
        exceeded = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 1, 1, 0, 0, 0),
                            "Close Time": datetime(2026, 1, 8, 0, 1, 0),
                        }
                    )
                ],
                market,
            )
        )
        self.assertEqual(exceeded["status"], STATUS_INCOMPLETE_COVERAGE)
        self.assertEqual(exceeded["reason_code"], REASON_WINDOW_EXCEEDS_LIMIT)

    def test_coverage_reasons(self):
        market = self._bars("US100.cash", datetime(2026, 6, 25, 10, 0), 6)
        gapped = pd.concat(
            [
                self._bars("US100.cash", datetime(2026, 6, 25, 10, 0), 3),
                self._bars("US100.cash", datetime(2026, 6, 25, 10, 4), 2),
            ],
            ignore_index=True,
        )
        outside_start = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 9, 0, 0),
                            "Close Time": datetime(2026, 6, 25, 9, 1, 0),
                        }
                    )
                ],
                market,
            )
        )
        self.assertEqual(outside_start["reason_code"], REASON_OUTSIDE_FILE_RANGE)
        outside_end = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 10, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 11, 0),
                        }
                    )
                ],
                market,
            )
        )
        self.assertEqual(outside_end["reason_code"], REASON_OUTSIDE_FILE_RANGE)
        start_missing = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 3, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 4, 0),
                        }
                    )
                ],
                gapped,
            )
        )
        self.assertEqual(start_missing["reason_code"], REASON_BOUNDARY_BAR_MISSING)
        end_missing = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 2, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 3, 0),
                        }
                    )
                ],
                gapped,
            )
        )
        self.assertEqual(end_missing["reason_code"], REASON_BOUNDARY_BAR_MISSING)
        interior = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 0, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 4, 0),
                        }
                    )
                ],
                gapped,
            )
        )
        self.assertEqual(interior["reason_code"], REASON_INTERIOR_GAP)
        complete = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 0, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 2, 0),
                        }
                    )
                ],
                market,
            )
        )
        self.assertEqual(complete["status"], STATUS_COMPUTED)
        self.assertEqual(complete["bars_used"], 3)

    def test_market_index_is_bounded_and_monotonic(self):
        market = pd.concat(
            [
                self._bars("EURUSD", datetime(2026, 6, 25, 9, 0), 3),
                self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 4),
            ],
            ignore_index=True,
        )
        indexed = _index_market(market)
        self.assertEqual(set(indexed), {"EURUSD", "US100.cash"})
        us = indexed["US100.cash"]["frame"]
        self.assertIsInstance(us.index, pd.DatetimeIndex)
        self.assertTrue(us.index.is_monotonic_increasing)
        self.assertTrue(us.index.is_unique)
        start = pd.Timestamp(datetime(2026, 6, 25, 10, 31, tzinfo=timezone.utc))
        end = pd.Timestamp(datetime(2026, 6, 25, 10, 32, tzinfo=timezone.utc))
        self.assertIn(start, us.index)
        self.assertNotIn(
            pd.Timestamp(datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)),
            us.index,
        )
        window = us.loc[start:end]
        self.assertEqual(len(window), 2)
        self.assertEqual(
            list(window.index),
            [start, end],
        )
        self.assertTrue((window["Symbol"] == "US100.cash").all())

    def test_math_flags_and_json(self):
        highs = [110, 120, 105]
        lows = [90, 95, 80]
        market = self._bars(
            "US100.cash",
            datetime(2026, 6, 25, 10, 30),
            3,
            high=highs,
            low=lows,
        )
        buy = self._evidence(
            self._run(
                [
                    self._trade(
                        Type="BUY",
                        Entry=100.0,
                        Exit=105.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            )
        )
        self.assertEqual(buy["interval_high"], 120.0)
        self.assertEqual(buy["interval_low"], 80.0)
        self.assertEqual(buy["mfe"], 20.0)
        self.assertEqual(buy["mae"], -20.0)
        self.assertFalse(buy["entry_outside_first_bar_range"])
        self.assertFalse(buy["exit_outside_last_bar_range"])
        self.assertFalse(buy["realised_outside_interval"])
        self.assertFalse(buy["high_from_boundary_bar"])
        self.assertTrue(buy["low_from_boundary_bar"])

        sell = self._evidence(
            self._run(
                [
                    self._trade(
                        Ticket=2,
                        Type="SELL",
                        Entry=100.0,
                        Exit=95.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            ),
            "2",
        )
        self.assertEqual(sell["interval_high"], 120.0)
        self.assertEqual(sell["interval_low"], 80.0)
        self.assertEqual(sell["mfe"], 20.0)
        self.assertEqual(sell["mae"], -20.0)

        clamp_market = self._bars(
            "US100.cash",
            datetime(2026, 6, 25, 10, 30),
            1,
            high=[99],
            low=[101],
        )
        # High < Low triggers invariant after attempted compute.
        invariant = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 30, 30),
                        }
                    )
                ],
                clamp_market,
            )
        )
        self.assertEqual(invariant["status"], STATUS_INVARIANT_VIOLATION)
        self.assertEqual(invariant["reason_code"], REASON_INVARIANT_FAILED)
        self.assertIsNone(invariant["mfe"])

        zero_market = self._bars(
            "US100.cash",
            datetime(2026, 6, 25, 10, 30),
            1,
            high=[100],
            low=[100],
        )
        zero = self._evidence(
            self._run(
                [
                    self._trade(
                        Entry=100.0,
                        Exit=100.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 30, 10),
                        },
                    )
                ],
                zero_market,
            )
        )
        self.assertEqual(zero["mfe"], 0.0)
        self.assertEqual(zero["mae"], 0.0)

        open_high = self._bars(
            "US100.cash",
            datetime(2026, 6, 25, 10, 30),
            3,
            high=[130, 120, 110],
            low=[80, 85, 70],
        )
        open_flags = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        }
                    )
                ],
                open_high,
            )
        )
        self.assertTrue(open_flags["high_from_boundary_bar"])
        self.assertTrue(open_flags["low_from_boundary_bar"])

        close_high = self._bars(
            "US100.cash",
            datetime(2026, 6, 25, 10, 30),
            3,
            high=[110, 115, 140],
            low=[95, 85, 70],
        )
        close_flags = self._evidence(
            self._run(
                [
                    self._trade(
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        }
                    )
                ],
                close_high,
            )
        )
        self.assertTrue(close_flags["high_from_boundary_bar"])
        self.assertTrue(close_flags["low_from_boundary_bar"])

        range_item = self._evidence(
            self._run(
                [
                    self._trade(
                        Entry=80.0,
                        Exit=140.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            )
        )
        self.assertTrue(range_item["entry_outside_first_bar_range"])
        self.assertTrue(range_item["exit_outside_last_bar_range"])
        self.assertTrue(range_item["realised_outside_interval"])

        below_mae = self._evidence(
            self._run(
                [
                    self._trade(
                        Entry=100.0,
                        Exit=50.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            )
        )
        self.assertTrue(below_mae["realised_outside_interval"])

        edge = self._evidence(
            self._run(
                [
                    self._trade(
                        Entry=90.0,
                        Exit=105.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            )
        )
        self.assertFalse(edge["entry_outside_first_bar_range"])

        inside_eps = self._evidence(
            self._run(
                [
                    self._trade(
                        Entry=90.0 - PRICE_COMPARISON_EPSILON,
                        Exit=105.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            )
        )
        self.assertFalse(inside_eps["entry_outside_first_bar_range"])
        outside_eps = self._evidence(
            self._run(
                [
                    self._trade(
                        Entry=90.0 - (2 * PRICE_COMPARISON_EPSILON),
                        Exit=105.0,
                        **{
                            "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                            "Close Time": datetime(2026, 6, 25, 10, 32, 0),
                        },
                    )
                ],
                market,
            )
        )
        self.assertTrue(outside_eps["entry_outside_first_bar_range"])

        encoded = json.dumps(buy, allow_nan=False)
        self.assertIsInstance(encoded, str)
        self.assertNotIn("NaN", encoded)
        self.assertFalse(isinstance(buy["mfe"], pd.Timestamp))
        self.assertTrue(isinstance(buy["mfe"], float))
        self.assertTrue(math.isfinite(buy["mfe"]))

    def test_multi_symbol_counts_and_precedence(self):
        market = pd.concat(
            [
                self._bars("US100.cash", datetime(2026, 6, 25, 10, 30), 4),
                self._bars("EURUSD", datetime(2026, 6, 25, 10, 30), 4),
            ],
            ignore_index=True,
        )
        result = self._run(
            [
                self._trade(Ticket=1, Symbol="US100.cash"),
                self._trade(Ticket=2, Symbol="EURUSD"),
                self._trade(Ticket=3, Symbol="NAS100"),
                self._trade(Ticket=4, Type="hold"),
            ],
            market,
        )
        self.assertEqual(result.evidence["1"]["status"], STATUS_COMPUTED)
        self.assertEqual(result.evidence["2"]["status"], STATUS_COMPUTED)
        self.assertEqual(result.evidence["3"]["status"], STATUS_NO_MARKET_DATA)
        self.assertEqual(result.evidence["4"]["status"], STATUS_INVALID_TRADE_DATA)
        self.assertEqual(result.status_counts[STATUS_COMPUTED], 2)
        self.assertEqual(result.status_counts[STATUS_NO_MARKET_DATA], 1)
        self.assertEqual(result.reason_counts["NO_MARKET_DATA"], 1)
        self.assertEqual(result.reason_counts["INVALID_SIDE"], 1)

    def test_journal_fingerprint(self):
        rows = [
            self._trade(Ticket=2, Profit=9),
            self._trade(Ticket=1, Symbol="EURUSD", Profit=8),
        ]
        first = compute_journal_fingerprint(self._journal(rows))
        shuffled = compute_journal_fingerprint(self._journal(list(reversed(rows))))
        self.assertEqual(first, shuffled)
        self.assertEqual(len(first), 64)
        self.assertEqual(first, first.lower())

        def changed(**overrides):
            altered = [self._trade(Ticket=2), self._trade(Ticket=1, Symbol="EURUSD")]
            altered[0].update(overrides)
            return compute_journal_fingerprint(self._journal(altered))

        self.assertNotEqual(first, changed(Ticket=22))
        self.assertNotEqual(
            first,
            changed(**{"Open Time": datetime(2026, 6, 25, 9, 30, 15)}),
        )
        self.assertNotEqual(
            first,
            changed(**{"Close Time": datetime(2026, 6, 25, 11, 32, 47)}),
        )
        self.assertNotEqual(first, changed(Symbol="XAUUSD"))
        self.assertNotEqual(first, changed(Type="sell"))
        self.assertNotEqual(first, changed(Entry=101.0))
        self.assertNotEqual(first, changed(Exit=106.0))
        self.assertEqual(first, changed(Profit=999))
        self.assertEqual(first, changed(Commission=99))
        self.assertEqual(first, changed(Notes="changed"))

        with_duration = [self._trade(), self._trade(Ticket=2, Symbol="EURUSD")]
        with_duration[0]["Trade duration in seconds"] = 300
        duration_a = compute_journal_fingerprint(self._journal(with_duration))
        with_duration[0]["Trade duration in seconds"] = 301
        duration_b = compute_journal_fingerprint(self._journal(with_duration))
        self.assertNotEqual(duration_a, duration_b)

        open_time = pd.DataFrame([self._trade()])
        open_only = pd.DataFrame(
            [
                {
                    "Ticket": 1,
                    "Open": datetime(2026, 6, 25, 10, 30, 15),
                    "Close": datetime(2026, 6, 25, 10, 32, 47),
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 105.0,
                }
            ]
        )
        self.assertNotEqual(
            compute_journal_fingerprint(open_time),
            compute_journal_fingerprint(open_only),
        )
        self.assertIsNone(
            compute_journal_fingerprint(pd.DataFrame([{"Symbol": "US100"}]))
        )

        def fingerprint_with(**overrides):
            row = self._trade()
            row.update(overrides)
            return compute_journal_fingerprint(self._journal([row]))

        self.assertEqual(
            fingerprint_with(Entry=100),
            fingerprint_with(Entry=100.0),
        )
        self.assertEqual(
            fingerprint_with(Entry=100.0),
            fingerprint_with(Entry="100.00"),
        )
        self.assertEqual(
            fingerprint_with(Exit=105),
            fingerprint_with(Exit=105.0),
        )
        self.assertEqual(
            fingerprint_with(Exit=105.0),
            fingerprint_with(Exit="105.000"),
        )
        duration_int = fingerprint_with(**{"Trade duration in seconds": 300})
        duration_float = fingerprint_with(**{"Trade duration in seconds": 300.0})
        duration_text = fingerprint_with(**{"Trade duration in seconds": "300.000"})
        self.assertEqual(duration_int, duration_float)
        self.assertEqual(duration_float, duration_text)
        self.assertNotEqual(
            duration_int,
            fingerprint_with(**{"Trade duration in seconds": 301}),
        )
        self.assertNotEqual(
            fingerprint_with(Entry=100.0),
            fingerprint_with(Entry=100.01),
        )
