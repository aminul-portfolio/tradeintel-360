from __future__ import annotations

import inspect
import json
import math
from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from .excursion import (
    RUN_STATUS_OK,
    STATUS_COMPUTED,
    ExcursionRunResult,
    compute_excursion_evidence,
)
from .excursion_state import (
    EXCURSION_SESSION_KEY,
    MAX_EVIDENCE_SESSION_BYTES,
    ExcursionStateError,
    clear_excursion_state,
    load_bound_excursion_state,
    store_excursion_state,
)
from .market_data import (
    BAR_TIMESTAMP_SEMANTIC,
    EVIDENCE_CLASS,
    EVIDENCE_PRECISION,
    EXCURSION_CONTRACT_VERSION,
    MARKET_DATA_RESOLUTION,
    PRICE_BASIS,
    MarketDataProvenance,
    SymbolCoverage,
)
from .time_basis import TIME_BASIS_UTC, create_time_basis


class ExcursionStateTests(SimpleTestCase):
    def setUp(self):
        self.time_basis = create_time_basis(TIME_BASIS_UTC)
        journal = pd.DataFrame(
            [
                {
                    "Ticket": 1,
                    "Open Time": datetime(2026, 6, 25, 10, 30, 0),
                    "Close Time": datetime(2026, 6, 25, 10, 31, 0),
                    "Symbol": "US100.cash",
                    "Type": "buy",
                    "Entry": 100.0,
                    "Exit": 101.0,
                }
            ]
        )
        market = pd.DataFrame(
            [
                {
                    "TimestampUTC": datetime(2026, 6, 25, 10, 30, tzinfo=timezone.utc),
                    "Symbol": "US100.cash",
                    "Open": 100.0,
                    "High": 110.0,
                    "Low": 90.0,
                    "Close": 101.0,
                },
                {
                    "TimestampUTC": datetime(2026, 6, 25, 10, 31, tzinfo=timezone.utc),
                    "Symbol": "US100.cash",
                    "Open": 101.0,
                    "High": 112.0,
                    "Low": 91.0,
                    "Close": 102.0,
                },
            ]
        )
        self.result = compute_excursion_evidence(
            journal,
            market,
            self.time_basis,
        )
        self.provenance = MarketDataProvenance(
            source_basename="bars.csv",
            market_file_sha256="a" * 64,
            file_byte_size=128,
            declared_source="user_declared_export",
            declared_export_method="manual_csv",
            bar_timestamp_semantic=BAR_TIMESTAMP_SEMANTIC,
            resolution=MARKET_DATA_RESOLUTION,
            evidence_class=EVIDENCE_CLASS,
            precision=EVIDENCE_PRECISION,
            price_basis=PRICE_BASIS,
            contract_version=EXCURSION_CONTRACT_VERSION,
            source_row_count=2,
            valid_row_count=2,
            symbols=("US100.cash",),
            coverage=(
                SymbolCoverage(
                    symbol="US100.cash",
                    coverage_start="2026-06-25T10:30:00Z",
                    coverage_end="2026-06-25T10:31:00Z",
                ),
            ),
        )

    def _store(self, state, result=None):
        return store_excursion_state(
            state,
            result or self.result,
            time_basis=self.time_basis,
            market_provenance=self.provenance,
        )

    def test_literals_and_django_independence(self):
        self.assertEqual(
            EXCURSION_SESSION_KEY,
            "broker_bar_excursion_evidence_v1",
        )
        self.assertEqual(MAX_EVIDENCE_SESSION_BYTES, 512 * 1024)
        source = inspect.getsource(
            __import__(
                "performance.excursion_state",
                fromlist=["store_excursion_state"],
            )
        )
        self.assertNotIn("django.http", source)
        self.assertNotIn("django.contrib.sessions", source)
        self.assertNotIn("request.session", source)

    def test_store_replace_and_bindings(self):
        state = {}
        payload = self._store(state)
        json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertEqual(set(state), {EXCURSION_SESSION_KEY})
        stored = state[EXCURSION_SESSION_KEY]
        self.assertEqual(
            stored["journal_fingerprint"],
            self.result.journal_fingerprint,
        )
        self.assertEqual(stored["market_file_sha256"], "a" * 64)
        self.assertEqual(stored["time_basis"], self.time_basis.as_json())
        self.assertEqual(
            stored["market_provenance"],
            self.provenance.as_json(),
        )
        self.assertEqual(stored["status_counts"], self.result.status_counts)
        self.assertEqual(stored["reason_counts"], self.result.reason_counts)
        self.assertEqual(stored["contract_version"], EXCURSION_CONTRACT_VERSION)
        self.assertEqual(set(stored["evidence"]), {"1"})
        self.assertEqual(stored["evidence"]["1"]["status"], STATUS_COMPUTED)
        self.assertNotIn("TimestampUTC", json.dumps(stored))
        self.assertFalse(any(isinstance(value, pd.DataFrame) for value in stored.values()))
        self.assertNotIsInstance(stored.get("evidence"), pd.DataFrame)

        replacement = ExcursionRunResult(
            run_status=RUN_STATUS_OK,
            reason_code=None,
            journal_fingerprint=self.result.journal_fingerprint,
            evidence=MappingProxyType(
                {
                    "99": {
                        "ticket": "99",
                        "status": STATUS_COMPUTED,
                        "reason_code": None,
                        "interval_high": 1.0,
                        "interval_low": 1.0,
                        "mfe": 0.0,
                        "mae": 0.0,
                        "bars_used": 1,
                        "bars_expected": 1,
                        "high_from_boundary_bar": True,
                        "low_from_boundary_bar": True,
                        "entry_outside_first_bar_range": False,
                        "exit_outside_last_bar_range": False,
                        "realised_outside_interval": False,
                    }
                }
            ),
            status_counts={STATUS_COMPUTED: 1},
            reason_counts={},
        )
        self._store(state, replacement)
        self.assertEqual(set(state[EXCURSION_SESSION_KEY]["evidence"]), {"99"})
        self.assertNotIn("1", state[EXCURSION_SESSION_KEY]["evidence"])

    def test_load_clear_and_lazy_invalidation(self):
        state = {}
        self._store(state)
        loaded = load_bound_excursion_state(
            state,
            current_journal_fingerprint=self.result.journal_fingerprint,
        )
        self.assertEqual(loaded["journal_fingerprint"], self.result.journal_fingerprint)
        self.assertIsNot(loaded, state[EXCURSION_SESSION_KEY])

        mismatched = load_bound_excursion_state(
            state,
            current_journal_fingerprint="b" * 64,
        )
        self.assertIsNone(mismatched)
        self.assertNotIn(EXCURSION_SESSION_KEY, state)

        self._store(state)
        state[EXCURSION_SESSION_KEY] = dict(state[EXCURSION_SESSION_KEY])
        state[EXCURSION_SESSION_KEY].pop("contract_version")
        self.assertIsNone(
            load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        )
        self.assertNotIn(EXCURSION_SESSION_KEY, state)

        self._store(state)
        state[EXCURSION_SESSION_KEY]["contract_version"] = "OTHER"
        self.assertIsNone(
            load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        )

        self._store(state)
        state[EXCURSION_SESSION_KEY] = "not-a-dict"
        self.assertIsNone(
            load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        )

        self._store(state)
        state[EXCURSION_SESSION_KEY]["journal_fingerprint"] = None
        self.assertIsNone(
            load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        )

        self._store(state)
        state[EXCURSION_SESSION_KEY].pop("market_file_sha256")
        self.assertIsNone(
            load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        )

        empty = {}
        self.assertIsNone(
            load_bound_excursion_state(
                empty,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        )
        self.assertEqual(empty, {})

        self._store(state)
        clear_excursion_state(state)
        self.assertNotIn(EXCURSION_SESSION_KEY, state)

    def test_required_package_schema(self):
        def drop_and_load(field):
            state = {}
            self._store(state)
            state[EXCURSION_SESSION_KEY].pop(field)
            loaded = load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
            self.assertIsNone(loaded)
            self.assertNotIn(EXCURSION_SESSION_KEY, state)

        drop_and_load("market_provenance")
        drop_and_load("status_counts")
        drop_and_load("reason_counts")

        def type_and_load(field, value):
            state = {}
            self._store(state)
            state[EXCURSION_SESSION_KEY][field] = value
            loaded = load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
            self.assertIsNone(loaded)
            self.assertNotIn(EXCURSION_SESSION_KEY, state)

        type_and_load("time_basis", "UTC")
        type_and_load("market_provenance", [])
        type_and_load("status_counts", [])
        type_and_load("reason_counts", [])
        type_and_load("evidence", [])

    def test_json_and_size_guards(self):
        state = {}

        def bad_result(value):
            return ExcursionRunResult(
                run_status=RUN_STATUS_OK,
                reason_code=None,
                journal_fingerprint=self.result.journal_fingerprint,
                evidence=MappingProxyType(
                    {
                        "1": {
                            "ticket": "1",
                            "status": STATUS_COMPUTED,
                            "reason_code": None,
                            "interval_high": value,
                            "interval_low": 1.0,
                            "mfe": 0.0,
                            "mae": 0.0,
                            "bars_used": 1,
                            "bars_expected": 1,
                            "high_from_boundary_bar": False,
                            "low_from_boundary_bar": False,
                            "entry_outside_first_bar_range": False,
                            "exit_outside_last_bar_range": False,
                            "realised_outside_interval": False,
                        }
                    }
                ),
                status_counts={STATUS_COMPUTED: 1},
                reason_counts={},
            )

        with self.assertRaises(ExcursionStateError) as nan_error:
            self._store(state, bad_result(float("nan")))
        self.assertEqual(nan_error.exception.reason, "NON_JSON_SAFE_PAYLOAD")
        self.assertNotIn(EXCURSION_SESSION_KEY, state)

        with self.assertRaises(ExcursionStateError) as inf_error:
            self._store(state, bad_result(float("inf")))
        self.assertEqual(inf_error.exception.reason, "NON_JSON_SAFE_PAYLOAD")

        with self.assertRaises(ExcursionStateError) as obj_error:
            self._store(state, bad_result(object()))
        self.assertEqual(obj_error.exception.reason, "NON_JSON_SAFE_PAYLOAD")

        with patch(
            "performance.excursion_state.MAX_EVIDENCE_SESSION_BYTES",
            16,
        ):
            with self.assertRaises(ExcursionStateError) as size_error:
                self._store(state)
        self.assertEqual(size_error.exception.reason, "PAYLOAD_TOO_LARGE")
        self.assertNotIn(EXCURSION_SESSION_KEY, state)

        accepted = self._store({})
        measured = len(
            json.dumps(
                accepted,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        with patch(
            "performance.excursion_state.MAX_EVIDENCE_SESSION_BYTES",
            measured,
        ):
            again = {}
            self._store(again)
            self.assertIn(EXCURSION_SESSION_KEY, again)

        self.assertTrue(math.isfinite(MAX_EVIDENCE_SESSION_BYTES))

    def test_load_strict_json_and_detached_store(self):
        nested = {
            "ticket": "1",
            "status": STATUS_COMPUTED,
            "reason_code": None,
            "interval_high": 1.0,
            "interval_low": 1.0,
            "mfe": 0.0,
            "mae": 0.0,
            "bars_used": 1,
            "bars_expected": 1,
            "high_from_boundary_bar": False,
            "low_from_boundary_bar": False,
            "entry_outside_first_bar_range": False,
            "exit_outside_last_bar_range": False,
            "realised_outside_interval": False,
        }
        manual = ExcursionRunResult(
            run_status=RUN_STATUS_OK,
            reason_code=None,
            journal_fingerprint=self.result.journal_fingerprint,
            evidence=MappingProxyType({"1": nested}),
            status_counts={STATUS_COMPUTED: 1},
            reason_counts={},
        )
        detached_state = {}
        stored = self._store(detached_state, manual)
        nested["interval_high"] = 999.0
        self.assertEqual(
            detached_state[EXCURSION_SESSION_KEY]["evidence"]["1"]["interval_high"],
            1.0,
        )
        self.assertEqual(stored["evidence"]["1"]["interval_high"], 1.0)

        loaded = load_bound_excursion_state(
            detached_state,
            current_journal_fingerprint=self.result.journal_fingerprint,
        )
        json.dumps(loaded, allow_nan=False)
        self.assertEqual(loaded["evidence"]["1"]["interval_high"], 1.0)

        def corrupt(value, path=("evidence", "1", "mfe")):
            state = {}
            self._store(state)
            cursor = state[EXCURSION_SESSION_KEY]
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            result = load_bound_excursion_state(
                state,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
            self.assertIsNone(result)
            self.assertNotIn(EXCURSION_SESSION_KEY, state)

        corrupt(float("nan"))
        corrupt(float("inf"))
        corrupt(object())
        corrupt(object(), path=("time_basis",))

        oversized = {}
        self._store(oversized)
        with patch(
            "performance.excursion_state.MAX_EVIDENCE_SESSION_BYTES",
            16,
        ):
            cleared = load_bound_excursion_state(
                oversized,
                current_journal_fingerprint=self.result.journal_fingerprint,
            )
        self.assertIsNone(cleared)
        self.assertNotIn(EXCURSION_SESSION_KEY, oversized)
