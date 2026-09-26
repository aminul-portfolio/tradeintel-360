import math

import pandas as pd
from django.test import SimpleTestCase

from .trade_review import (
    LABEL_PIPS,
    LABEL_PRICE_MOVE,
    MOVEMENT_SOURCE_BROKER,
    MOVEMENT_SOURCE_PRICE_MOVE,
    MOVEMENT_STATUS_DISAGREEMENT,
    MOVEMENT_STATUS_FALLBACK,
    PIPS_MOVEMENT_TOLERANCE,
    calculated_movement,
    enrich_trade_review,
    is_valid_pips,
    normalize_sort_direction,
    normalize_sort_key,
    resolve_realised_movement,
    sort_trade_review,
)


class TradeReviewMovementTests(SimpleTestCase):
    def test_buy_broker_pips_selected(self):
        result = resolve_realised_movement(
            12.5,
            "buy",
            100,
            112.5,
        )

        self.assertEqual(result["value"], 12.5)
        self.assertEqual(result["label"], LABEL_PIPS)
        self.assertEqual(
            result["source"],
            MOVEMENT_SOURCE_BROKER,
        )
        self.assertFalse(result["disagreement"])

    def test_sell_broker_pips_selected(self):
        result = resolve_realised_movement(
            8.0,
            "sell",
            110,
            102,
        )

        self.assertEqual(result["value"], 8.0)
        self.assertEqual(result["label"], LABEL_PIPS)
        self.assertEqual(
            result["source"],
            MOVEMENT_SOURCE_BROKER,
        )
        self.assertFalse(result["disagreement"])

    def test_missing_pips_buy_price_move(self):
        result = resolve_realised_movement(
            None,
            "buy",
            100,
            107.25,
        )

        self.assertEqual(result["value"], 7.25)
        self.assertEqual(result["label"], LABEL_PRICE_MOVE)
        self.assertEqual(
            result["source"],
            MOVEMENT_SOURCE_PRICE_MOVE,
        )
        self.assertEqual(
            result["status"],
            MOVEMENT_STATUS_FALLBACK,
        )

    def test_missing_pips_sell_price_move(self):
        result = resolve_realised_movement(
            "",
            "sell",
            200,
            194.5,
        )

        self.assertEqual(result["value"], 5.5)
        self.assertEqual(result["label"], LABEL_PRICE_MOVE)
        self.assertEqual(
            result["source"],
            MOVEMENT_SOURCE_PRICE_MOVE,
        )

    def test_invalid_and_non_finite_pips_use_fallback(self):
        invalid = resolve_realised_movement(
            "n/a",
            "buy",
            50,
            55,
        )
        non_finite = resolve_realised_movement(
            math.inf,
            "buy",
            50,
            55,
        )

        self.assertFalse(is_valid_pips("n/a"))
        self.assertFalse(is_valid_pips(math.inf))
        self.assertEqual(invalid["value"], 5)
        self.assertEqual(invalid["label"], LABEL_PRICE_MOVE)
        self.assertEqual(non_finite["value"], 5)
        self.assertEqual(
            non_finite["source"],
            MOVEMENT_SOURCE_PRICE_MOVE,
        )

    def test_pips_disagreement_above_tolerance_flagged(self):
        result = resolve_realised_movement(
            10.2,
            "buy",
            100,
            110,
        )

        self.assertEqual(
            PIPS_MOVEMENT_TOLERANCE,
            0.1,
        )
        self.assertEqual(
            calculated_movement("buy", 100, 110),
            10,
        )
        self.assertEqual(result["value"], 10.2)
        self.assertEqual(result["label"], LABEL_PIPS)
        self.assertTrue(result["disagreement"])
        self.assertEqual(
            result["status"],
            MOVEMENT_STATUS_DISAGREEMENT,
        )

    def test_pips_within_tolerance_not_flagged(self):
        result = resolve_realised_movement(
            10.05,
            "buy",
            100,
            110,
        )

        self.assertFalse(result["disagreement"])
        self.assertEqual(result["label"], LABEL_PIPS)


class TradeReviewSortingTests(SimpleTestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Ticket": [3, 1, 2, 4],
                "Open Time": pd.to_datetime(
                    [
                        "2026-09-03 10:00:00",
                        "2026-09-01 10:00:00",
                        "2026-09-02 10:00:00",
                        "2026-09-01 10:00:00",
                    ]
                ),
                "Symbol": [
                    "btc",
                    "XAUUSD",
                    "eurusd",
                    "XAUUSD",
                ],
                "Profit": [10, 40, None, 40],
            }
        )

    def test_numeric_sort_ascending_and_descending(self):
        ascending = sort_trade_review(
            self.df,
            "profit",
            "asc",
        )
        descending = sort_trade_review(
            self.df,
            "profit",
            "desc",
        )

        self.assertEqual(
            list(ascending["Profit"].dropna()),
            [10, 40, 40],
        )
        self.assertTrue(
            pd.isna(ascending["Profit"].iloc[-1])
        )
        self.assertEqual(
            list(descending["Profit"].iloc[:2]),
            [40, 40],
        )
        self.assertEqual(
            descending["Profit"].iloc[2],
            10,
        )

    def test_date_chronological_sort(self):
        sorted_df = sort_trade_review(
            self.df,
            "open_time",
            "asc",
        )

        self.assertEqual(
            list(sorted_df["Open Time"]),
            list(
                pd.to_datetime(
                    [
                        "2026-09-01 10:00:00",
                        "2026-09-01 10:00:00",
                        "2026-09-02 10:00:00",
                        "2026-09-03 10:00:00",
                    ]
                )
            ),
        )

    def test_text_case_insensitive_sort(self):
        sorted_df = sort_trade_review(
            self.df,
            "symbol",
            "asc",
        )

        self.assertEqual(
            list(sorted_df["Symbol"]),
            ["btc", "eurusd", "XAUUSD", "XAUUSD"],
        )

    def test_invalid_sort_key_and_direction_fallback(self):
        original = self.df.copy(deep=True)

        invalid_key = sort_trade_review(
            self.df,
            "not-a-column",
            "desc",
        )
        invalid_dir = sort_trade_review(
            self.df,
            "profit",
            "sideways",
        )

        pd.testing.assert_frame_equal(
            invalid_key.reset_index(drop=True),
            original.reset_index(drop=True),
        )
        self.assertIsNone(
            normalize_sort_key("not-a-column")
        )
        self.assertEqual(
            normalize_sort_direction("sideways"),
            "asc",
        )
        self.assertEqual(
            list(invalid_dir["Profit"].dropna()),
            [10, 40, 40],
        )

    def test_stable_ticket_tie_break(self):
        sorted_df = sort_trade_review(
            self.df,
            "profit",
            "desc",
        )
        tied = sorted_df[
            sorted_df["Profit"] == 40
        ]

        self.assertEqual(
            list(tied["Ticket"]),
            [1, 4],
        )

    def test_enrich_then_sort_uses_movement_values(self):
        frame = pd.DataFrame(
            {
                "Ticket": [2, 1],
                "Type": ["buy", "sell"],
                "Price": [100, 200],
                "Price.1": [110, 190],
                "Pips": [None, None],
            }
        )
        enriched = enrich_trade_review(frame)
        sorted_df = sort_trade_review(
            enriched,
            "movement",
            "desc",
        )

        self.assertEqual(
            list(sorted_df["_movement_value"]),
            [10, 10],
        )
        self.assertEqual(
            list(sorted_df["Ticket"]),
            [1, 2],
        )
        self.assertEqual(
            list(sorted_df["_movement_label"]),
            [LABEL_PRICE_MOVE, LABEL_PRICE_MOVE],
        )
