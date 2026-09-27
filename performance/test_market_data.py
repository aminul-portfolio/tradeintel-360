from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone
from unittest.mock import patch

from django.test import SimpleTestCase

from .market_data import (
    BAR_TIMESTAMP_SEMANTIC,
    EVIDENCE_CLASS,
    EVIDENCE_PRECISION,
    EXCURSION_CONTRACT_VERSION,
    MARKET_DATA_RESOLUTION,
    MAX_MARKET_DATA_BYTES,
    MAX_MARKET_DATA_ROWS,
    MAX_SYMBOL_LENGTH,
    PRICE_BASIS,
    REQUIRED_COLUMNS,
    MarketDataValidationError,
    parse_market_data_bytes,
)


class MarketDataFoundationTests(SimpleTestCase):
    def _csv_bytes(self, rows, columns=None, bom=False):
        fieldnames = columns or list(REQUIRED_COLUMNS)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        payload = buffer.getvalue().encode("utf-8")
        if bom:
            return b"\xef\xbb\xbf" + payload
        return payload

    def _bar(
        self,
        timestamp,
        symbol="US100.cash",
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        **extra,
    ):
        row = {
            "TimestampUTC": timestamp,
            "Symbol": symbol,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
        }
        row.update(extra)
        return row

    def _m1_rows(self, symbol="US100.cash", start_hour=10, count=2):
        rows = []
        for index in range(count):
            rows.append(
                self._bar(
                    f"2026-06-25T{start_hour:02d}:{index:02d}:00Z",
                    symbol=symbol,
                )
            )
        return rows

    def _parse(self, payload, filename="bars.csv"):
        return parse_market_data_bytes(
            payload,
            filename,
            declared_source="user_declared_export",
            declared_export_method="manual_csv",
        )

    def _reject(self, payload, filename="bars.csv"):
        with self.assertRaises(MarketDataValidationError) as caught:
            self._parse(payload, filename=filename)
        return caught.exception

    def test_locked_constants(self):
        self.assertEqual(MAX_MARKET_DATA_BYTES, 16 * 1024 * 1024)
        self.assertEqual(MAX_MARKET_DATA_ROWS, 200_000)
        self.assertEqual(MAX_SYMBOL_LENGTH, 64)
        self.assertEqual(
            REQUIRED_COLUMNS,
            (
                "TimestampUTC",
                "Symbol",
                "Open",
                "High",
                "Low",
                "Close",
            ),
        )
        self.assertEqual(MARKET_DATA_RESOLUTION, "M1")
        self.assertEqual(EVIDENCE_CLASS, "BROKER_BAR_SOURCE")
        self.assertEqual(EVIDENCE_PRECISION, "M1_BAR_APPROXIMATE")
        self.assertEqual(
            PRICE_BASIS,
            "UNSPECIFIED_CTRADER_HISTORICAL_BAR",
        )
        self.assertEqual(BAR_TIMESTAMP_SEMANTIC, "BAR_OPEN_TIME")
        self.assertEqual(
            EXCURSION_CONTRACT_VERSION,
            "EXCURSION_CONTRACT_V1",
        )

    def test_valid_standard_csv_and_provenance(self):
        payload = self._csv_bytes(self._m1_rows())
        result = self._parse(payload)
        frame = result.dataframe
        provenance = result.provenance

        self.assertEqual(len(frame), 2)
        self.assertEqual(
            list(frame.columns),
            list(REQUIRED_COLUMNS),
        )
        self.assertEqual(provenance.source_row_count, 2)
        self.assertEqual(provenance.valid_row_count, 2)
        self.assertEqual(
            provenance.market_file_sha256,
            hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(provenance.file_byte_size, len(payload))
        self.assertEqual(provenance.source_basename, "bars.csv")
        self.assertEqual(provenance.declared_source, "user_declared_export")
        self.assertEqual(
            provenance.declared_export_method,
            "manual_csv",
        )
        self.assertEqual(provenance.resolution, MARKET_DATA_RESOLUTION)
        self.assertEqual(provenance.evidence_class, EVIDENCE_CLASS)
        self.assertEqual(provenance.precision, EVIDENCE_PRECISION)
        self.assertEqual(provenance.price_basis, PRICE_BASIS)
        self.assertEqual(
            provenance.bar_timestamp_semantic,
            BAR_TIMESTAMP_SEMANTIC,
        )
        self.assertEqual(
            provenance.contract_version,
            EXCURSION_CONTRACT_VERSION,
        )
        self.assertEqual(provenance.symbols, ("US100.cash",))
        self.assertEqual(provenance.coverage[0].symbol, "US100.cash")
        self.assertEqual(
            provenance.coverage[0].coverage_start,
            "2026-06-25T10:00:00Z",
        )
        self.assertEqual(
            provenance.coverage[0].coverage_end,
            "2026-06-25T10:01:00Z",
        )
        self.assertTrue(
            str(frame["TimestampUTC"].dtype).startswith("datetime64")
        )
        self.assertEqual(
            frame["TimestampUTC"].dt.tz,
            timezone.utc,
        )

    def test_utf8_bom_and_tickvolume_absent(self):
        payload = self._csv_bytes(self._m1_rows(), bom=True)
        result = self._parse(payload)
        self.assertEqual(len(result.dataframe), 2)
        self.assertEqual(
            result.provenance.market_file_sha256,
            hashlib.sha256(payload).hexdigest(),
        )

    def test_extra_columns_are_ignored(self):
        rows = self._m1_rows()
        for row in rows:
            row["TickVolume"] = 12
            row["Unused"] = "ignore"
        payload = self._csv_bytes(
            rows,
            columns=[*REQUIRED_COLUMNS, "TickVolume", "Unused"],
        )
        result = self._parse(payload)
        self.assertEqual(list(result.dataframe.columns), list(REQUIRED_COLUMNS))
        self.assertNotIn("TickVolume", result.dataframe.columns)

    def test_filename_and_empty_or_invalid_bytes(self):
        valid = self._csv_bytes(self._m1_rows())
        windows = self._parse(
            valid,
            filename=r"C:\Users\sumon\data\bars.csv",
        )
        posix = self._parse(valid, filename="/tmp/private/bars.csv")
        traversal = self._parse(valid, filename="../../bars.csv")
        self.assertEqual(windows.provenance.source_basename, "bars.csv")
        self.assertEqual(posix.provenance.source_basename, "bars.csv")
        self.assertEqual(traversal.provenance.source_basename, "bars.csv")

        self.assertEqual(self._reject(b"", filename="bars.csv").reason, "EMPTY_FILE")
        self.assertEqual(
            self._reject(valid, filename="bars.xlsx").reason,
            "NOT_CSV",
        )
        self.assertEqual(
            self._reject(b"hello \xff", filename="bars.csv").reason,
            "DECODE_ERROR",
        )

    def test_byte_and_row_limits(self):
        with patch("performance.market_data.MAX_MARKET_DATA_BYTES", 8):
            error = self._reject(b"012345678", filename="bars.csv")
        self.assertEqual(error.reason, "FILE_TOO_LARGE")

        rows = self._m1_rows(count=3)
        payload = self._csv_bytes(rows)
        with patch("performance.market_data.MAX_MARKET_DATA_ROWS", 2):
            error = self._reject(payload)
        self.assertEqual(error.reason, "ROW_LIMIT_EXCEEDED")

        too_few = self._csv_bytes(self._m1_rows(count=1))
        self.assertEqual(self._reject(too_few).reason, "TOO_FEW_ROWS")

    def test_missing_each_required_column(self):
        for missing in REQUIRED_COLUMNS:
            columns = [name for name in REQUIRED_COLUMNS if name != missing]
            rows = []
            for row in self._m1_rows():
                rows.append({key: row[key] for key in columns})
            error = self._reject(self._csv_bytes(rows, columns=columns))
            self.assertEqual(error.reason, "MISSING_REQUIRED_COLUMNS")

    def test_symbol_trimming_and_length(self):
        rows = [
            self._bar("2026-06-25T10:00:00Z", symbol=" US100.cash "),
            self._bar("2026-06-25T10:00:00Z", symbol="US100.cash"),
        ]
        error = self._reject(self._csv_bytes(rows))
        self.assertEqual(error.reason, "DUPLICATE_BAR")

        valid = self._parse(
            self._csv_bytes(
                [
                    self._bar("2026-06-25T10:00:00Z", symbol=" US100.cash "),
                    self._bar("2026-06-25T10:01:00Z", symbol=" US100.cash "),
                ]
            )
        )
        self.assertEqual(list(valid.dataframe["Symbol"]), ["US100.cash", "US100.cash"])

        sixty_four = "A" * 64
        sixty_five = "A" * 65
        accepted = self._parse(
            self._csv_bytes(
                [
                    self._bar("2026-06-25T10:00:00Z", symbol=sixty_four),
                    self._bar("2026-06-25T10:01:00Z", symbol=sixty_four),
                ]
            )
        )
        self.assertEqual(accepted.provenance.symbols, (sixty_four,))
        rejected = self._reject(
            self._csv_bytes(
                [
                    self._bar("2026-06-25T10:00:00Z", symbol=sixty_five),
                    self._bar("2026-06-25T10:01:00Z", symbol=sixty_five),
                ]
            )
        )
        self.assertEqual(rejected.reason, "SYMBOL_TOO_LONG")
        empty = self._reject(
            self._csv_bytes(
                [
                    self._bar("2026-06-25T10:00:00Z", symbol="   "),
                    self._bar("2026-06-25T10:01:00Z", symbol="US100"),
                ]
            )
        )
        self.assertEqual(empty.reason, "EMPTY_SYMBOL")

    def test_ohlc_validation(self):
        valid = self._parse(self._csv_bytes(self._m1_rows()))
        self.assertEqual(valid.dataframe["Open"].iloc[0], 100.0)

        cases = [
            ({"Open": "abc"}, "INVALID_OHLC"),
            ({"High": "NaN"}, "INVALID_OHLC"),
            ({"Low": "inf"}, "INVALID_OHLC"),
            ({"Close": "-inf"}, "INVALID_OHLC"),
            ({"Open": 0}, "NON_POSITIVE_OHLC"),
            ({"High": -1}, "NON_POSITIVE_OHLC"),
            ({"High": 90}, "IMPOSSIBLE_OHLC"),
            ({"Low": 110}, "IMPOSSIBLE_OHLC"),
        ]
        for override, reason in cases:
            rows = self._m1_rows()
            rows[1].update(override)
            self.assertEqual(self._reject(self._csv_bytes(rows)).reason, reason)

    def test_timestamps_and_duplicates(self):
        accepted = [
            "2026-06-25T10:30:00Z",
            "2026-06-25T10:30:00.0Z",
            "2026-06-25T10:30:00.000Z",
            "2026-06-25T10:30:00.000000Z",
            "2026-06-25T10:31:00+00:00",
            "2026-06-25T10:31:00.000+00:00",
        ]
        result = self._parse(
            self._csv_bytes(
                [
                    self._bar(accepted[0], symbol="US100"),
                    self._bar(accepted[4], symbol="US100"),
                ]
            )
        )
        self.assertEqual(len(result.dataframe), 2)
        dotted = self._parse(
            self._csv_bytes(
                [
                    self._bar(accepted[2], symbol="NAS100"),
                    self._bar(accepted[5], symbol="NAS100"),
                ]
            )
        )
        self.assertEqual(len(dotted.dataframe), 2)
        zero_fraction = self._parse(
            self._csv_bytes(
                [
                    self._bar("2026-06-25T10:30:00.0Z", symbol="EURUSD"),
                    self._bar("2026-06-25T10:31:00.000000Z", symbol="EURUSD"),
                ]
            )
        )
        self.assertEqual(len(zero_fraction.dataframe), 2)

        rejected = [
            ("2026-06-25 10:30:00", "INVALID_TIMESTAMP"),
            ("2026-06-25T10:30:00+01:00", "NON_UTC_TIMESTAMP"),
            ("2026-06-25T10:30:00-05:00", "NON_UTC_TIMESTAMP"),
            ("2026-06-25T10:30:01Z", "TIMESTAMP_NOT_MINUTE_ALIGNED"),
            ("2026-06-25T10:30:00.500Z", "TIMESTAMP_NOT_MINUTE_ALIGNED"),
            ("2026-06-25T10:30:00.000001Z", "TIMESTAMP_NOT_MINUTE_ALIGNED"),
            ("2026-06-25T10:30:00.0000001Z", "TIMESTAMP_NOT_MINUTE_ALIGNED"),
        ]
        for timestamp, reason in rejected:
            rows = [
                self._bar(timestamp, symbol="US100"),
                self._bar("2026-06-25T10:31:00Z", symbol="US100"),
            ]
            self.assertEqual(self._reject(self._csv_bytes(rows)).reason, reason)

        duplicate = self._csv_bytes(
            [
                self._bar("2026-06-25T10:00:00Z"),
                self._bar("2026-06-25T10:00:00Z"),
            ]
        )
        self.assertEqual(self._reject(duplicate).reason, "DUPLICATE_BAR")

    def test_sort_resolution_and_multi_symbol(self):
        unsorted = self._csv_bytes(
            [
                self._bar("2026-06-25T10:05:00Z"),
                self._bar("2026-06-25T10:00:00Z"),
                self._bar("2026-06-25T10:01:00Z"),
            ]
        )
        result = self._parse(unsorted)
        times = [
            value.to_pydatetime().astimezone(timezone.utc)
            for value in result.dataframe["TimestampUTC"]
        ]
        self.assertEqual(
            times,
            [
                datetime(2026, 6, 25, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 6, 25, 10, 1, tzinfo=timezone.utc),
                datetime(2026, 6, 25, 10, 5, tzinfo=timezone.utc),
            ],
        )

        m5 = self._csv_bytes(
            [
                self._bar("2026-06-25T10:00:00Z"),
                self._bar("2026-06-25T10:05:00Z"),
                self._bar("2026-06-25T10:10:00Z"),
            ]
        )
        self.assertEqual(self._reject(m5).reason, "M1_RESOLUTION_UNCONFIRMED")

        one_row_symbol = self._csv_bytes(
            [
                *self._m1_rows(symbol="US100"),
                self._bar("2026-06-25T10:00:00Z", symbol="EURUSD"),
            ]
        )
        self.assertEqual(
            self._reject(one_row_symbol).reason,
            "M1_RESOLUTION_UNCONFIRMED",
        )

        two_row = self._parse(self._csv_bytes(self._m1_rows(count=2)))
        self.assertEqual(len(two_row.dataframe), 2)

        multi = self._parse(
            self._csv_bytes(
                [
                    *self._m1_rows(symbol="US100"),
                    *self._m1_rows(symbol="EURUSD", start_hour=11),
                ]
            )
        )
        self.assertEqual(multi.provenance.symbols, ("EURUSD", "US100"))
        self.assertEqual(
            [item.symbol for item in multi.provenance.coverage],
            ["EURUSD", "US100"],
        )
        self.assertEqual(
            multi.provenance.coverage[0].coverage_start,
            "2026-06-25T11:00:00Z",
        )
        self.assertEqual(
            multi.provenance.coverage[1].coverage_end,
            "2026-06-25T10:01:00Z",
        )
        self.assertEqual(
            list(multi.dataframe["Symbol"]),
            ["EURUSD", "EURUSD", "US100", "US100"],
        )

    def test_whole_file_rejected_when_one_row_invalid(self):
        rows = self._m1_rows(count=3)
        rows[2]["Close"] = "bad"
        with self.assertRaises(MarketDataValidationError) as caught:
            self._parse(self._csv_bytes(rows))
        self.assertEqual(caught.exception.reason, "INVALID_OHLC")

    def test_malformed_quoted_csv_is_rejected(self):
        payload = (
            "TimestampUTC,Symbol,Open,High,Low,Close\n"
            '2026-06-25T10:00:00Z,"US100"cash,100,101,99,100.5\n'
            "2026-06-25T10:01:00Z,US100,100,101,99,100.5\n"
        ).encode("utf-8")
        error = self._reject(payload)
        self.assertEqual(error.reason, "CSV_PARSE_ERROR")
