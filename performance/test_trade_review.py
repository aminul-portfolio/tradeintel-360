import math

import pandas as pd
from django.test import SimpleTestCase

from .trade_review import (
    BAR_EVIDENCE_LABELS,
    BAR_EVIDENCE_UNBOUND_LABEL,
    LABEL_PIPS,
    LABEL_PRICE_MOVE,
    MOVEMENT_SOURCE_BROKER,
    MOVEMENT_SOURCE_PRICE_MOVE,
    MOVEMENT_STATUS_DISAGREEMENT,
    MOVEMENT_STATUS_FALLBACK,
    PIPS_MOVEMENT_TOLERANCE,
    _cell_alignment_class,
    attach_excursion_evidence,
    bar_evidence_notes,
    build_trade_review_columns,
    build_trade_review_rows,
    calculated_movement,
    enrich_trade_review,
    format_bar_evidence_status,
    format_evidence_value,
    format_movement_value,
    format_volume_value,
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


class TradeReviewVolumeFormattingTests(SimpleTestCase):
    def test_volume_display_rounding_and_special_values(self):
        self.assertEqual(
            format_volume_value(0.30000000000000004),
            "0.3",
        )
        self.assertEqual(format_volume_value(1.2345), "1.23")
        self.assertEqual(format_volume_value(1.235), "1.24")
        self.assertEqual(format_volume_value(0.125), "0.13")
        self.assertEqual(format_volume_value(2.675), "2.68")
        self.assertEqual(format_volume_value(2.0), "2")
        self.assertEqual(format_volume_value(100000), "100000")
        self.assertNotIn("e", format_volume_value(100000).lower())
        self.assertNotIn("e", format_volume_value(1234567).lower())
        self.assertEqual(format_volume_value(0.001), "<0.01")
        self.assertEqual(format_volume_value(-0.001), ">-0.01")
        self.assertEqual(format_volume_value(None), "")
        self.assertEqual(format_volume_value(float("nan")), "")
        self.assertEqual(format_volume_value(float("inf")), "")
        self.assertEqual(format_volume_value(float("-inf")), "")
        self.assertEqual(format_volume_value(-0.0), "0")
        self.assertEqual(format_volume_value("1.235"), "1.24")
        self.assertEqual(format_volume_value(pd.Series([2], dtype="int64").iloc[0]), "2")

    def test_volume_presenter_does_not_mutate_raw_values(self):
        raw_volume = 0.30000000000000004
        frame = pd.DataFrame(
            {
                "Ticket": [1],
                "Volume": [raw_volume],
                "Profit": [10],
            }
        )
        enriched = enrich_trade_review(frame)
        rows = build_trade_review_rows(
            enriched,
            build_trade_review_columns(enriched),
        )
        volume_index = [
            index
            for index, column in enumerate(
                build_trade_review_columns(enriched)
            )
            if column["key"] == "volume"
        ][0]

        self.assertEqual(
            rows[0]["cells"][volume_index]["value"],
            "0.3",
        )
        self.assertEqual(
            enriched["Volume"].iloc[0],
            raw_volume,
        )
        self.assertEqual(
            frame["Volume"].iloc[0],
            raw_volume,
        )

    def test_volume_sort_uses_raw_numeric_not_display(self):
        frame = pd.DataFrame(
            {
                "Ticket": [2, 1, 3],
                "Volume": [1.234, 1.231, 2.0],
            }
        )
        columns = build_trade_review_columns(frame)
        volume_index = [
            index
            for index, column in enumerate(columns)
            if column["key"] == "volume"
        ][0]
        displays = [
            row["cells"][volume_index]["value"]
            for row in build_trade_review_rows(frame, columns)
        ]
        self.assertEqual(displays[:2], ["1.23", "1.23"])

        sorted_df = sort_trade_review(frame, "volume", "asc")
        self.assertEqual(
            list(sorted_df["Volume"]),
            [1.231, 1.234, 2.0],
        )
        self.assertEqual(
            list(sorted_df["Ticket"]),
            [1, 2, 3],
        )
        self.assertEqual(
            sorted_df["Volume"].iloc[0],
            1.231,
        )


class TradeReviewExcursionEvidenceTests(SimpleTestCase):
    EVIDENCE_LABELS = (
        "Approx. Window High",
        "Approx. Window Low",
        "Approx. MFE (price pts)",
        "Approx. MAE (price pts)",
        "Bar Evidence",
    )
    FORBIDDEN_ADVICE = (
        "should have held",
        "should have exited",
        "missed profit",
        "profit left on table",
        "optimal exit",
    )

    def _base_frame(self):
        return pd.DataFrame(
            {
                "Ticket": [1, 2, 3],
                "Open Time": [
                    "2026-06-25 10:30:00",
                    "2026-06-25 10:30:00",
                    "2026-06-25 11:00:00",
                ],
                "Symbol": ["US100.cash", "EURUSD", "US100.cash"],
                "Type": ["buy", "sell", "buy"],
                "Profit": [10, 5, 8],
                "Volume": [1.0, 2.0, 3.0],
            }
        )

    def _item(self, ticket, **overrides):
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

    def _attached(self, frame=None, evidence=None):
        if frame is None:
            frame = self._base_frame()
        if evidence is None:
            evidence = {
                "1": self._item(1, interval_high=120.0, mfe=20.0, mae=2.0),
                "2": self._item(2, interval_high=1.2, interval_low=1.0, mfe=0.0, mae=0.1),
                "3": self._item(3, interval_high=105.0, mfe=5.0, mae=None, status="INCOMPLETE_COVERAGE"),
            }
        return attach_excursion_evidence(frame, evidence)

    def test_attaches_by_ticket_including_canonical_and_alphanumeric(self):
        frame = pd.DataFrame(
            {
                "Ticket": [1, 2.0, "ABC-123"],
                "Symbol": ["US100.cash", "EURUSD", "GBPUSD"],
                "Profit": [1, 2, 3],
            }
        )
        evidence = {
            "1": self._item(1, mfe=11.0),
            "2": self._item(2, mfe=22.0),
            "ABC-123": self._item("ABC-123", mfe=33.0),
        }
        original = frame.copy(deep=True)
        attached = attach_excursion_evidence(frame, evidence)

        self.assertEqual(
            list(attached["_excursion_mfe"]),
            [11.0, 22.0, 33.0],
        )
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_or_wrong_ticket_has_no_symbol_or_time_fallback(self):
        no_ticket = pd.DataFrame(
            {
                "Symbol": ["US100.cash"],
                "Open Time": ["2026-06-25 10:30:00"],
                "Profit": [10],
            }
        )
        wrong_ticket = pd.DataFrame(
            {
                "Ticket": [99],
                "Symbol": ["US100.cash"],
                "Open Time": ["2026-06-25 10:30:00"],
                "Profit": [10],
            }
        )
        evidence = {
            "1": self._item(
                1,
                mfe=99.0,
                interval_high=150.0,
            )
        }

        missing = attach_excursion_evidence(no_ticket, evidence)
        mismatched = attach_excursion_evidence(wrong_ticket, evidence)

        self.assertTrue(pd.isna(missing["_excursion_mfe"]).all())
        self.assertIsNone(mismatched["_excursion_mfe"].iloc[0])
        self.assertIsNone(mismatched["_excursion_interval_high"].iloc[0])

    def test_evidence_columns_always_present_without_internal_evidence(self):
        frame = self._base_frame()
        columns = build_trade_review_columns(frame)
        labels = [column["label"] for column in columns]
        self.assertEqual(labels[-5:], list(self.EVIDENCE_LABELS))
        by_key = {column["key"]: column for column in columns}
        for key in ("interval_high", "interval_low", "mfe", "mae"):
            self.assertEqual(by_key[key]["sort_key"], "")
            self.assertFalse(by_key[key]["evidence_bound"])
            self.assertEqual(
                by_key[key]["header_css_class"],
                "trade-review-cell--numeric",
            )
        self.assertEqual(by_key["bar_evidence"]["sort_key"], "")
        self.assertNotEqual(
            by_key["bar_evidence"]["header_css_class"],
            "trade-review-cell--numeric",
        )

    def test_no_evidence_rows_are_blank_with_unbound_status(self):
        frame = self._base_frame()
        columns = build_trade_review_columns(frame)
        rows = build_trade_review_rows(frame, columns)
        by_key = {column["key"]: index for index, column in enumerate(columns)}
        cell = rows[0]["cells"]
        self.assertEqual(cell[by_key["interval_high"]]["value"], "")
        self.assertEqual(cell[by_key["interval_low"]]["value"], "")
        self.assertEqual(cell[by_key["mfe"]]["value"], "")
        self.assertEqual(cell[by_key["mae"]]["value"], "")
        self.assertEqual(
            cell[by_key["bar_evidence"]]["value"],
            BAR_EVIDENCE_UNBOUND_LABEL,
        )
        self.assertNotEqual(cell[by_key["mfe"]]["value"], "0")
        self.assertNotEqual(
            cell[by_key["bar_evidence"]]["value"],
            BAR_EVIDENCE_LABELS["COMPUTED"],
        )
        self.assertNotEqual(
            cell[by_key["bar_evidence"]]["value"],
            BAR_EVIDENCE_LABELS["NO_MARKET_DATA"],
        )

    def test_no_evidence_sort_keys_disabled(self):
        columns = build_trade_review_columns(self._base_frame())
        sort_keys = {
            column["key"]: column["sort_key"]
            for column in columns
            if column["key"]
            in {"interval_high", "interval_low", "mfe", "mae", "bar_evidence"}
        }
        self.assertEqual(
            sort_keys,
            {
                "interval_high": "",
                "interval_low": "",
                "mfe": "",
                "mae": "",
                "bar_evidence": "",
            },
        )
        sorted_df = sort_trade_review(self._base_frame(), "mfe", "desc")
        self.assertEqual(list(sorted_df["Ticket"]), [1, 2, 3])

    def test_bound_no_market_data_is_not_unbound_placeholder(self):
        attached = self._attached(
            pd.DataFrame({"Ticket": [1], "Profit": [1]}),
            {
                "1": self._item(
                    1,
                    status="NO_MARKET_DATA",
                    reason_code="OUTSIDE_FILE_RANGE",
                    interval_high=None,
                    interval_low=None,
                    mfe=None,
                    mae=None,
                )
            },
        )
        columns = build_trade_review_columns(attached)
        rows = build_trade_review_rows(attached, columns)
        by_key = {column["key"]: index for index, column in enumerate(columns)}
        self.assertTrue(columns[by_key["bar_evidence"]]["evidence_bound"])
        self.assertEqual(columns[by_key["mfe"]]["sort_key"], "mfe")
        self.assertEqual(rows[0]["cells"][by_key["interval_high"]]["value"], "")
        self.assertEqual(
            rows[0]["cells"][by_key["bar_evidence"]]["value"],
            BAR_EVIDENCE_LABELS["NO_MARKET_DATA"],
        )
        self.assertNotEqual(
            rows[0]["cells"][by_key["bar_evidence"]]["value"],
            BAR_EVIDENCE_UNBOUND_LABEL,
        )

    def test_valid_attached_evidence_adds_five_user_facing_columns(self):
        columns = build_trade_review_columns(self._attached())
        labels = [column["label"] for column in columns]
        self.assertEqual(
            labels[-5:],
            list(self.EVIDENCE_LABELS),
        )
        sort_keys = {
            column["key"]: column["sort_key"]
            for column in columns
            if column["label"] in self.EVIDENCE_LABELS
        }
        self.assertEqual(sort_keys["interval_high"], "interval_high")
        self.assertEqual(sort_keys["interval_low"], "interval_low")
        self.assertEqual(sort_keys["mfe"], "mfe")
        self.assertEqual(sort_keys["mae"], "mae")
        self.assertEqual(sort_keys["bar_evidence"], "")
        self.assertTrue(
            all(
                column["evidence_bound"]
                for column in columns
                if column["label"] in self.EVIDENCE_LABELS
            )
        )

    def test_evidence_numeric_sort_nulls_last_and_ticket_tie_break(self):
        frame = pd.DataFrame(
            {
                "Ticket": [2, 1, 4, 3],
                "Profit": [1, 1, 1, 1],
            }
        )
        evidence = {
            "1": self._item(1, interval_high=10.0, interval_low=1.0, mfe=5.0, mae=8.0),
            "2": self._item(2, interval_high=10.0, interval_low=3.0, mfe=1.0, mae=2.0),
            "3": self._item(3, interval_high=None, interval_low=None, mfe=None, mae=None, status="NO_MARKET_DATA"),
            "4": self._item(4, interval_high=30.0, interval_low=2.0, mfe=9.0, mae=1.0),
        }
        attached = attach_excursion_evidence(frame, evidence)

        high_asc = sort_trade_review(attached, "interval_high", "asc")
        high_desc = sort_trade_review(attached, "interval_high", "desc")
        low_asc = sort_trade_review(attached, "interval_low", "asc")
        low_desc = sort_trade_review(attached, "interval_low", "desc")
        mfe_asc = sort_trade_review(attached, "mfe", "asc")
        mfe_desc = sort_trade_review(attached, "mfe", "desc")
        mae_asc = sort_trade_review(attached, "mae", "asc")
        mae_desc = sort_trade_review(attached, "mae", "desc")

        self.assertEqual(list(high_asc["Ticket"]), [1, 2, 4, 3])
        self.assertEqual(list(high_desc["Ticket"]), [4, 1, 2, 3])
        self.assertEqual(list(low_asc["Ticket"]), [1, 4, 2, 3])
        self.assertEqual(list(low_desc["Ticket"]), [2, 4, 1, 3])
        self.assertEqual(list(mfe_asc["Ticket"]), [2, 1, 4, 3])
        self.assertEqual(list(mfe_desc["Ticket"]), [4, 1, 2, 3])
        self.assertEqual(list(mae_asc["Ticket"]), [4, 2, 1, 3])
        self.assertEqual(list(mae_desc["Ticket"]), [1, 2, 4, 3])
        self.assertEqual(normalize_sort_key("profit"), "profit")
        self.assertEqual(normalize_sort_key("ticket"), "ticket")

    def test_evidence_formatting_blank_zero_and_no_scientific_notation(self):
        self.assertEqual(format_evidence_value(None), "")
        self.assertEqual(format_evidence_value(float("nan")), "")
        self.assertEqual(format_evidence_value(float("inf")), "")
        self.assertEqual(format_evidence_value(float("-inf")), "")
        self.assertEqual(format_evidence_value(0.0), "0")
        self.assertEqual(format_evidence_value(-0.0), "0")
        self.assertEqual(format_evidence_value(12.5), "12.5")
        self.assertEqual(format_evidence_value(1000000), "1000000")
        self.assertNotIn("e", format_evidence_value(1000000).lower())
        self.assertNotIn("e", format_evidence_value(0.000125).lower())
        self.assertEqual(format_evidence_value(0.000125), "0.000125")

        tiny = format_evidence_value(0.0000001)
        self.assertNotEqual(tiny, "0")
        self.assertNotIn("e", tiny.lower())
        self.assertEqual(tiny, "0.0000001")

        negative_tiny = format_evidence_value(-0.0000001)
        self.assertNotEqual(negative_tiny, "0")
        self.assertTrue(negative_tiny.startswith("-"))
        self.assertNotIn("e", negative_tiny.lower())
        self.assertEqual(negative_tiny, "-0.0000001")

        self.assertEqual(format_evidence_value(0.30000000000000004), "0.3")
        self.assertNotIn("00000000000000004", format_evidence_value(0.30000000000000004))

    def test_evidence_float_noise_is_suppressed_for_display(self):
        cases = (
            (5.399999999999782, "5.4"),
            (-166.1000000000002, "-166.1"),
            (70.84999999999985, "70.85"),
            (-54.70000000000007, "-54.7"),
            (7.450000000000073, "7.45"),
        )
        for raw, expected in cases:
            self.assertEqual(format_evidence_value(raw), expected)
            self.assertNotIn("9999", format_evidence_value(raw))
            self.assertNotIn("00000", format_evidence_value(raw))

    def test_noisy_evidence_sort_uses_raw_numeric_not_display(self):
        noisy = 5.399999999999782
        tiny = 0.0000001
        frame = pd.DataFrame(
            {
                "Ticket": [3, 1, 2],
                "Profit": [1, 1, 1],
            }
        )
        attached = attach_excursion_evidence(
            frame,
            {
                "1": self._item(1, mfe=tiny),
                "2": self._item(2, mfe=0.3),
                "3": self._item(3, mfe=noisy),
            },
        )
        self.assertEqual(
            attached.loc[attached["Ticket"] == 3, "_excursion_mfe"].iloc[0],
            noisy,
        )
        self.assertEqual(format_evidence_value(noisy), "5.4")
        self.assertEqual(format_evidence_value(tiny), "0.0000001")
        self.assertEqual(format_evidence_value(0.3), "0.3")
        columns = build_trade_review_columns(attached)
        rows = build_trade_review_rows(attached, columns)
        mfe_index = [
            index for index, column in enumerate(columns) if column["key"] == "mfe"
        ][0]
        displays = {
            int(row_ticket): rows[position]["cells"][mfe_index]["value"]
            for position, row_ticket in enumerate(attached["Ticket"])
        }
        self.assertEqual(displays[3], "5.4")
        self.assertEqual(displays[1], "0.0000001")
        self.assertEqual(displays[2], "0.3")
        sorted_df = sort_trade_review(attached, "mfe", "asc")
        self.assertEqual(list(sorted_df["Ticket"]), [1, 2, 3])
        self.assertEqual(sorted_df["_excursion_mfe"].iloc[2], noisy)
        self.assertNotEqual(sorted_df["_excursion_mfe"].iloc[2], "5.4")

    def test_evidence_sort_uses_raw_numeric_not_display(self):
        raw_tiny = 0.0000001
        frame = pd.DataFrame(
            {
                "Ticket": [2, 1],
                "Profit": [1, 1],
            }
        )
        attached = attach_excursion_evidence(
            frame,
            {
                "1": self._item(1, mfe=raw_tiny),
                "2": self._item(2, mfe=0.3),
            },
        )
        self.assertEqual(
            attached.loc[attached["Ticket"] == 1, "_excursion_mfe"].iloc[0],
            raw_tiny,
        )
        self.assertEqual(format_evidence_value(raw_tiny), "0.0000001")
        sorted_df = sort_trade_review(attached, "mfe", "asc")
        self.assertEqual(list(sorted_df["Ticket"]), [1, 2])
        self.assertEqual(sorted_df["_excursion_mfe"].iloc[0], raw_tiny)
        self.assertNotEqual(
            sorted_df["_excursion_mfe"].iloc[0],
            format_evidence_value(raw_tiny),
        )

    def test_bar_evidence_status_flags_and_no_recommendation_language(self):
        for status, label in BAR_EVIDENCE_LABELS.items():
            self.assertEqual(format_bar_evidence_status(status), label)

        computed = {
            "_excursion_status": "COMPUTED",
            "_excursion_reason": "OK",
            "_excursion_high_from_boundary_bar": True,
            "_excursion_low_from_boundary_bar": True,
            "_excursion_entry_outside_first_bar_range": True,
            "_excursion_exit_outside_last_bar_range": True,
            "_excursion_realised_outside_interval": True,
        }
        reason, notes = bar_evidence_notes(computed)
        self.assertEqual(reason, "OK")
        self.assertEqual(
            notes,
            "Boundary extreme; Entry outside first M1 bar range; "
            "Exit outside last M1 bar range; Realised move outside M1 envelope",
        )
        self.assertEqual(notes.count("Boundary extreme"), 1)

        incomplete = {
            "_excursion_status": "INCOMPLETE_COVERAGE",
            "_excursion_reason": "INTERIOR_GAP",
            "_excursion_high_from_boundary_bar": True,
        }
        incomplete_reason, incomplete_notes = bar_evidence_notes(incomplete)
        self.assertEqual(incomplete_reason, "INTERIOR_GAP")
        self.assertEqual(incomplete_notes, "")

        frame = self._attached(
            pd.DataFrame({"Ticket": [1], "Profit": [1]}),
            {
                "1": self._item(
                    1,
                    high_from_boundary_bar=True,
                    entry_outside_first_bar_range=True,
                    exit_outside_last_bar_range=True,
                    realised_outside_interval=True,
                )
            },
        )
        columns = build_trade_review_columns(frame)
        rows = build_trade_review_rows(frame, columns)
        status_index = [
            index for index, column in enumerate(columns) if column["key"] == "bar_evidence"
        ][0]
        cell = rows[0]["cells"][status_index]
        self.assertEqual(cell["value"], "Computed")
        self.assertEqual(cell["note"], "OK")
        self.assertIn("Boundary extreme", cell["flag"])
        self.assertIn("Entry outside first M1 bar range", cell["flag"])
        self.assertIn("Exit outside last M1 bar range", cell["flag"])
        self.assertIn("Realised move outside M1 envelope", cell["flag"])
        rendered = " ".join(
            [
                str(cell["value"]),
                str(cell["note"]),
                str(cell["flag"]),
            ]
        ).lower()
        for phrase in self.FORBIDDEN_ADVICE:
            self.assertNotIn(phrase, rendered)

        blank_frame = self._attached(
            pd.DataFrame({"Ticket": [7], "Profit": [1]}),
            {
                "7": self._item(
                    7,
                    status="NO_MARKET_DATA",
                    reason_code="OUTSIDE_FILE_RANGE",
                    interval_high=None,
                    interval_low=None,
                    mfe=None,
                    mae=None,
                )
            },
        )
        blank_columns = build_trade_review_columns(blank_frame)
        blank_rows = build_trade_review_rows(blank_frame, blank_columns)
        mfe_index = [
            index for index, column in enumerate(blank_columns) if column["key"] == "mfe"
        ][0]
        self.assertEqual(blank_rows[0]["cells"][mfe_index]["value"], "")
        bar_index = [
            index
            for index, column in enumerate(blank_columns)
            if column["key"] == "bar_evidence"
        ][0]
        self.assertEqual(
            blank_rows[0]["cells"][bar_index]["value"],
            BAR_EVIDENCE_LABELS["NO_MARKET_DATA"],
        )
        self.assertNotEqual(
            blank_rows[0]["cells"][bar_index]["value"],
            BAR_EVIDENCE_UNBOUND_LABEL,
        )


class TradeReviewMovementPresentationTests(SimpleTestCase):
    def _rows(self, frame):
        enriched = enrich_trade_review(frame)
        columns = build_trade_review_columns(enriched)
        return enriched, columns, build_trade_review_rows(enriched, columns)

    def _movement_cell(self, frame):
        _enriched, columns, rows = self._rows(frame)
        index = [
            position
            for position, column in enumerate(columns)
            if column["key"] == "movement"
        ][0]
        return rows[0]["cells"][index], _enriched

    def test_positive_valid_broker_pips_is_signed(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["buy"],
                    "Price": [100],
                    "Price.1": [142.3],
                    "Pips": [42.3],
                }
            )
        )
        self.assertEqual(cell["value"], "+42.3 pips")
        self.assertEqual(cell["note"], LABEL_PIPS)
        self.assertIn("trade-review-move--pos", cell["css_class"])
        self.assertEqual(enriched["_movement_value"].iloc[0], 42.3)

    def test_negative_valid_broker_pips_is_signed(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["sell"],
                    "Price": [100],
                    "Price.1": [118.7],
                    "Pips": [-18.7],
                }
            )
        )
        self.assertEqual(cell["value"], "-18.7 pips")
        self.assertEqual(cell["note"], LABEL_PIPS)
        self.assertIn("trade-review-move--neg", cell["css_class"])
        self.assertEqual(enriched["_movement_value"].iloc[0], -18.7)

    def test_zero_valid_broker_pips_is_visible(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["buy"],
                    "Price": [100],
                    "Price.1": [100],
                    "Pips": [0.0],
                }
            )
        )
        self.assertEqual(cell["value"], "0 pips")
        self.assertNotEqual(cell["value"], "")
        self.assertEqual(enriched["_movement_value"].iloc[0], 0.0)

    def test_disagreement_keeps_numeric_value_and_shows_mismatch(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["buy"],
                    "Price": [100],
                    "Price.1": [110],
                    "Pips": [10.2],
                }
            )
        )
        self.assertEqual(PIPS_MOVEMENT_TOLERANCE, 0.1)
        self.assertTrue(bool(enriched["_movement_disagreement"].iloc[0]))
        self.assertEqual(cell["value"], "+10.2 pips")
        self.assertIn("10.2", cell["value"])
        self.assertEqual(cell["note"], "Calculated move: +10 price pts")
        self.assertEqual(cell["flag"], "Mismatch")
        self.assertEqual(
            cell["title"],
            "Broker Pips and calculated price move differ by more than 0.1.",
        )
        self.assertNotIn("should have", cell["value"].lower())
        self.assertNotIn("Pips disagreement", cell["value"])
        self.assertEqual(enriched["_movement_value"].iloc[0], 10.2)
        self.assertEqual(enriched["_movement_source"].iloc[0], MOVEMENT_SOURCE_BROKER)

    def test_price_move_fallback_is_not_labelled_pips(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["buy"],
                    "Price": [100],
                    "Price.1": [112.4],
                    "Pips": [None],
                }
            )
        )
        self.assertEqual(cell["value"], "+12.4 price pts")
        self.assertEqual(cell["note"], LABEL_PRICE_MOVE)
        self.assertNotIn("Calculated move:", cell["note"])
        self.assertNotIn("pips", cell["value"].lower())
        self.assertNotIn("pips", cell["note"].lower())
        self.assertEqual(cell["flag"], "")
        self.assertEqual(enriched["_movement_source"].iloc[0], MOVEMENT_SOURCE_PRICE_MOVE)

    def test_window_high_low_labels_use_existing_evidence_keys(self):
        frame = pd.DataFrame({"Ticket": [1], "Profit": [1]})
        attached = attach_excursion_evidence(
            frame,
            {
                "1": {
                    "ticket": "1",
                    "status": "COMPUTED",
                    "reason_code": "OK",
                    "interval_high": 16172.8,
                    "interval_low": 16135.2,
                    "mfe": 0.0,
                    "mae": 4.0,
                    "high_from_boundary_bar": False,
                    "low_from_boundary_bar": False,
                    "entry_outside_first_bar_range": False,
                    "exit_outside_last_bar_range": False,
                    "realised_outside_interval": False,
                }
            },
        )
        columns = build_trade_review_columns(attached)
        labels = {column["key"]: column["label"] for column in columns}
        self.assertEqual(labels["interval_high"], "Approx. Window High")
        self.assertEqual(labels["interval_low"], "Approx. Window Low")
        self.assertEqual(labels["mfe"], "Approx. MFE (price pts)")
        self.assertEqual(labels["mae"], "Approx. MAE (price pts)")
        evidence_sort_keys = [
            column["sort_key"]
            for column in columns
            if column["key"] in {"interval_high", "interval_low", "mfe", "mae"}
        ]
        self.assertEqual(
            evidence_sort_keys,
            ["interval_high", "interval_low", "mfe", "mae"],
        )
        rows = build_trade_review_rows(attached, columns)
        high_index = [i for i, column in enumerate(columns) if column["key"] == "interval_high"][0]
        low_index = [i for i, column in enumerate(columns) if column["key"] == "interval_low"][0]
        mfe_index = [i for i, column in enumerate(columns) if column["key"] == "mfe"][0]
        self.assertEqual(rows[0]["cells"][high_index]["value"], "16172.8")
        self.assertEqual(rows[0]["cells"][low_index]["value"], "16135.2")
        self.assertEqual(rows[0]["cells"][mfe_index]["value"], "0")
        self.assertEqual(attached["_excursion_interval_high"].iloc[0], 16172.8)
        self.assertEqual(format_movement_value(0.0, LABEL_PIPS), "0 pips")

    def test_non_computed_window_and_mfe_are_blank(self):
        frame = pd.DataFrame({"Ticket": [1], "Profit": [1]})
        attached = attach_excursion_evidence(
            frame,
            {
                "1": {
                    "ticket": "1",
                    "status": "NO_MARKET_DATA",
                    "reason_code": "OUTSIDE_FILE_RANGE",
                    "interval_high": None,
                    "interval_low": None,
                    "mfe": None,
                    "mae": None,
                }
            },
        )
        columns = build_trade_review_columns(attached)
        rows = build_trade_review_rows(attached, columns)
        for key in ("interval_high", "interval_low", "mfe", "mae"):
            index = [i for i, column in enumerate(columns) if column["key"] == key][0]
            self.assertEqual(rows[0]["cells"][index]["value"], "")

    def test_movement_sort_uses_raw_numeric_not_display(self):
        frame = pd.DataFrame(
            {
                "Ticket": [2, 1],
                "Type": ["buy", "sell"],
                "Price": [100, 200],
                "Price.1": [110, 181.3],
                "Pips": [10.0, -18.7],
            }
        )
        enriched = enrich_trade_review(frame)
        sorted_df = sort_trade_review(enriched, "movement", "asc")
        self.assertEqual(list(sorted_df["Ticket"]), [1, 2])
        self.assertEqual(list(sorted_df["_movement_value"]), [-18.7, 10.0])
        self.assertNotEqual(
            sorted_df["_movement_value"].iloc[0],
            format_movement_value(-18.7, LABEL_PIPS),
        )

    def test_zero_filled_pips_mismatch_shows_calculated_secondary(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["sell"],
                    "Price": [43054.86],
                    "Price.1": [43125.04],
                    "Pips": [0.0],
                }
            )
        )
        self.assertEqual(PIPS_MOVEMENT_TOLERANCE, 0.1)
        self.assertTrue(is_valid_pips(0.0))
        self.assertAlmostEqual(
            calculated_movement("sell", 43054.86, 43125.04),
            -70.18,
            places=2,
        )
        self.assertEqual(cell["value"], "0 pips")
        self.assertEqual(cell["note"], "Calculated move: -70.18 price pts")
        self.assertEqual(cell["flag"], "Mismatch")
        self.assertEqual(
            cell["title"],
            "Broker Pips and calculated price move differ by more than 0.1.",
        )
        self.assertEqual(enriched["_movement_value"].iloc[0], 0.0)
        self.assertEqual(enriched["_movement_source"].iloc[0], MOVEMENT_SOURCE_BROKER)
        self.assertTrue(bool(enriched["_movement_disagreement"].iloc[0]))

    def test_genuine_zero_has_no_mismatch_or_calculated_secondary(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["buy"],
                    "Price": [150],
                    "Price.1": [150],
                    "Pips": [0.0],
                }
            )
        )
        self.assertEqual(PIPS_MOVEMENT_TOLERANCE, 0.1)
        self.assertEqual(cell["value"], "0 pips")
        self.assertEqual(cell["note"], LABEL_PIPS)
        self.assertNotIn("Calculated move:", cell["note"])
        self.assertEqual(cell["flag"], "")
        self.assertEqual(cell["title"], "")
        self.assertFalse(bool(enriched["_movement_disagreement"].iloc[0]))
        self.assertEqual(enriched["_movement_value"].iloc[0], 0.0)
        self.assertEqual(enriched["_movement_source"].iloc[0], MOVEMENT_SOURCE_BROKER)

    def test_nonzero_pips_mismatch_shows_calculated_secondary(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["buy"],
                    "Price": [100],
                    "Price.1": [110],
                    "Pips": [10.2],
                }
            )
        )
        self.assertEqual(PIPS_MOVEMENT_TOLERANCE, 0.1)
        self.assertEqual(cell["value"], "+10.2 pips")
        self.assertEqual(cell["note"], "Calculated move: +10 price pts")
        self.assertEqual(cell["flag"], "Mismatch")
        self.assertEqual(
            cell["title"],
            "Broker Pips and calculated price move differ by more than 0.1.",
        )
        self.assertEqual(enriched["_movement_source"].iloc[0], MOVEMENT_SOURCE_BROKER)
        self.assertEqual(enriched["_movement_value"].iloc[0], 10.2)

    def test_fallback_does_not_add_calculated_mismatch_line(self):
        cell, enriched = self._movement_cell(
            pd.DataFrame(
                {
                    "Ticket": [1],
                    "Type": ["sell"],
                    "Price": [200],
                    "Price.1": [194.5],
                    "Pips": [""],
                }
            )
        )
        self.assertEqual(cell["value"], "+5.5 price pts")
        self.assertEqual(cell["note"], LABEL_PRICE_MOVE)
        self.assertNotIn("Calculated move:", cell["note"])
        self.assertNotIn("pips", cell["value"].lower())
        self.assertEqual(cell["flag"], "")
        self.assertEqual(enriched["_movement_source"].iloc[0], MOVEMENT_SOURCE_PRICE_MOVE)
        self.assertEqual(enriched["_movement_value"].iloc[0], 5.5)

    def test_mismatch_secondary_does_not_replace_sort_value(self):
        frame = pd.DataFrame(
            {
                "Ticket": [2, 1],
                "Type": ["buy", "sell"],
                "Price": [100, 43054.86],
                "Price.1": [110, 43125.04],
                "Pips": [10.2, 0.0],
            }
        )
        enriched = enrich_trade_review(frame)
        self.assertEqual(list(enriched["_movement_value"]), [10.2, 0.0])
        sorted_df = sort_trade_review(enriched, "movement", "asc")
        self.assertEqual(list(sorted_df["Ticket"]), [1, 2])
        self.assertEqual(list(sorted_df["_movement_value"]), [0.0, 10.2])
        columns = build_trade_review_columns(sorted_df)
        rows = build_trade_review_rows(sorted_df, columns)
        movement_index = [
            position
            for position, column in enumerate(columns)
            if column["key"] == "movement"
        ][0]
        first = rows[0]["cells"][movement_index]
        self.assertEqual(first["value"], "0 pips")
        self.assertEqual(first["note"], "Calculated move: -70.18 price pts")
        self.assertEqual(sorted_df["_movement_value"].iloc[0], 0.0)
        self.assertNotEqual(
            sorted_df["_movement_value"].iloc[0],
            first["note"],
        )


class TradeReviewHeaderAlignmentTests(SimpleTestCase):
    NUMERIC_CLASS = "trade-review-cell--numeric"

    def test_column_header_alignment_matches_body_contract(self):
        frame = pd.DataFrame(
            {
                "Ticket": [1],
                "Open Time": ["2026-09-01 10:00:00"],
                "Close Time": ["2026-09-01 11:00:00"],
                "Symbol": ["ABC"],
                "Type": ["buy"],
                "Volume": [1.0],
                "Price": [100.0],
                "Price.1": [110.0],
                "SL": [90.0],
                "TP": [120.0],
                "Profit": [10.0],
                "Commission": [0.2],
                "Swap": [0.0],
            }
        )
        columns = build_trade_review_columns(frame)
        by_key = {column["key"]: column for column in columns}
        numeric_keys = (
            "volume",
            "entry",
            "exit",
            "sl",
            "tp",
            "profit",
            "commission",
            "swap",
            "movement",
        )
        for key in numeric_keys:
            self.assertEqual(
                by_key[key]["header_css_class"],
                self.NUMERIC_CLASS,
            )
            self.assertEqual(
                by_key[key]["header_css_class"],
                _cell_alignment_class(key),
            )
        for key in ("ticket", "symbol", "type"):
            self.assertNotEqual(
                by_key[key]["header_css_class"],
                self.NUMERIC_CLASS,
            )
            self.assertEqual(
                by_key[key]["header_css_class"],
                _cell_alignment_class(key),
            )
        self.assertEqual(
            by_key["interval_high"]["header_css_class"],
            self.NUMERIC_CLASS,
        )
        self.assertEqual(by_key["interval_high"]["sort_key"], "")
        self.assertFalse(by_key["interval_high"]["evidence_bound"])
        self.assertNotEqual(
            by_key["bar_evidence"]["header_css_class"],
            self.NUMERIC_CLASS,
        )

    def test_evidence_header_alignment_uses_same_contract(self):
        frame = pd.DataFrame({"Ticket": [1], "Profit": [1]})
        attached = attach_excursion_evidence(
            frame,
            {
                "1": {
                    "ticket": "1",
                    "status": "COMPUTED",
                    "reason_code": "OK",
                    "interval_high": 16172.8,
                    "interval_low": 16135.2,
                    "mfe": 0.0,
                    "mae": 4.0,
                    "high_from_boundary_bar": False,
                    "low_from_boundary_bar": False,
                    "entry_outside_first_bar_range": False,
                    "exit_outside_last_bar_range": False,
                    "realised_outside_interval": False,
                }
            },
        )
        columns = build_trade_review_columns(attached)
        by_key = {column["key"]: column for column in columns}
        for key in ("interval_high", "interval_low", "mfe", "mae"):
            self.assertEqual(
                by_key[key]["header_css_class"],
                self.NUMERIC_CLASS,
            )
            self.assertEqual(
                by_key[key]["header_css_class"],
                _cell_alignment_class(key),
            )
        self.assertNotEqual(
            by_key["bar_evidence"]["header_css_class"],
            self.NUMERIC_CLASS,
        )
        self.assertEqual(
            by_key["bar_evidence"]["header_css_class"],
            _cell_alignment_class("bar_evidence"),
        )
