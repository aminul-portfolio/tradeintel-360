from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from django.test import SimpleTestCase

from .time_basis import (
    MAX_OFFSET_MINUTES,
    MIN_OFFSET_MINUTES,
    OFFSET_STEP_MINUTES,
    TIME_BASIS_FIXED_OFFSET,
    TIME_BASIS_IANA,
    TIME_BASIS_RESOLUTION,
    TIME_BASIS_UTC,
    NormalisationStatus,
    TimeBasisValidationError,
    create_time_basis,
    normalise_journal_datetime,
)


class TimeBasisFoundationTests(SimpleTestCase):
    def test_locked_literals(self):
        self.assertEqual(TIME_BASIS_UTC, "UTC")
        self.assertEqual(TIME_BASIS_FIXED_OFFSET, "FIXED_OFFSET")
        self.assertEqual(TIME_BASIS_IANA, "IANA")
        self.assertEqual(TIME_BASIS_RESOLUTION, "USER_DECLARED")

    def test_utc_declaration(self):
        basis = create_time_basis(TIME_BASIS_UTC)
        self.assertEqual(basis.kind, TIME_BASIS_UTC)
        self.assertEqual(basis.resolution, TIME_BASIS_RESOLUTION)
        self.assertIsNone(basis.offset_minutes)
        self.assertIsNone(basis.zone)
        self.assertEqual(
            basis.as_json(),
            {
                "kind": TIME_BASIS_UTC,
                "resolution": TIME_BASIS_RESOLUTION,
            },
        )
        with self.assertRaises(TimeBasisValidationError) as offset_error:
            create_time_basis(TIME_BASIS_UTC, offset_minutes=0)
        self.assertEqual(offset_error.exception.reason, "CONFLICTING_FIELDS")
        with self.assertRaises(TimeBasisValidationError) as zone_error:
            create_time_basis(TIME_BASIS_UTC, zone="Europe/London")
        self.assertEqual(zone_error.exception.reason, "CONFLICTING_FIELDS")

    def test_fixed_offset_declaration(self):
        for offset in (-720, -300, 0, 60, 120, 330, 840):
            basis = create_time_basis(
                TIME_BASIS_FIXED_OFFSET,
                offset_minutes=offset,
            )
            self.assertEqual(basis.kind, TIME_BASIS_FIXED_OFFSET)
            self.assertEqual(basis.offset_minutes, offset)
            self.assertEqual(basis.resolution, TIME_BASIS_RESOLUTION)
            self.assertIsNone(basis.zone)
        self.assertEqual(MIN_OFFSET_MINUTES, -720)
        self.assertEqual(MAX_OFFSET_MINUTES, 840)
        self.assertEqual(OFFSET_STEP_MINUTES, 15)
        self.assertEqual(
            create_time_basis(
                TIME_BASIS_FIXED_OFFSET,
                offset_minutes=120,
            ).as_json(),
            {
                "kind": TIME_BASIS_FIXED_OFFSET,
                "offset_minutes": 120,
                "resolution": TIME_BASIS_RESOLUTION,
            },
        )
        cases = [
            (-721, "OFFSET_OUT_OF_RANGE"),
            (841, "OFFSET_OUT_OF_RANGE"),
            (62, "OFFSET_NOT_15_MINUTE_STEP"),
            ("120", "INVALID_OFFSET"),
        ]
        for value, reason in cases:
            with self.assertRaises(TimeBasisValidationError) as caught:
                create_time_basis(
                    TIME_BASIS_FIXED_OFFSET,
                    offset_minutes=value,
                )
            self.assertEqual(caught.exception.reason, reason)
        with self.assertRaises(TimeBasisValidationError) as zone_error:
            create_time_basis(
                TIME_BASIS_FIXED_OFFSET,
                offset_minutes=60,
                zone="Europe/London",
            )
        self.assertEqual(zone_error.exception.reason, "CONFLICTING_FIELDS")

    def test_iana_declaration_and_environment(self):
        london = create_time_basis(TIME_BASIS_IANA, zone="Europe/London")
        york = create_time_basis(TIME_BASIS_IANA, zone="America/New_York")
        self.assertEqual(london.kind, TIME_BASIS_IANA)
        self.assertEqual(london.zone, "Europe/London")
        self.assertIsNone(london.offset_minutes)
        self.assertEqual(york.zone, "America/New_York")
        self.assertEqual(
            london.as_json(),
            {
                "kind": TIME_BASIS_IANA,
                "zone": "Europe/London",
                "resolution": TIME_BASIS_RESOLUTION,
            },
        )
        ZoneInfo("Europe/London")
        ZoneInfo("America/New_York")
        for invalid_zone in (
            "Not/AZone",
            "/etc/passwd",
            "../Europe/London",
        ):
            with self.assertRaises(TimeBasisValidationError) as missing:
                create_time_basis(TIME_BASIS_IANA, zone=invalid_zone)
            self.assertEqual(missing.exception.reason, "ZONE_NOT_FOUND")
        with self.assertRaises(TimeBasisValidationError) as conflict:
            create_time_basis(
                TIME_BASIS_IANA,
                zone="Europe/London",
                offset_minutes=60,
            )
        self.assertEqual(conflict.exception.reason, "CONFLICTING_FIELDS")

    def test_normalisation_utc_and_fixed_offset(self):
        utc_basis = create_time_basis(TIME_BASIS_UTC)
        naive = datetime(2026, 6, 25, 10, 30, 0)
        utc_result = normalise_journal_datetime(naive, utc_basis)
        self.assertEqual(utc_result.status, NormalisationStatus.VALID.value)
        self.assertEqual(
            utc_result.utc_datetime,
            datetime(2026, 6, 25, 10, 30, tzinfo=timezone.utc),
        )

        plus = create_time_basis(TIME_BASIS_FIXED_OFFSET, offset_minutes=120)
        plus_result = normalise_journal_datetime(
            datetime(2026, 6, 25, 12, 30),
            plus,
        )
        self.assertEqual(
            plus_result.utc_datetime,
            datetime(2026, 6, 25, 10, 30, tzinfo=timezone.utc),
        )

        minus = create_time_basis(TIME_BASIS_FIXED_OFFSET, offset_minutes=-300)
        minus_result = normalise_journal_datetime(
            datetime(2026, 6, 25, 7, 30),
            minus,
        )
        self.assertEqual(
            minus_result.utc_datetime,
            datetime(2026, 6, 25, 12, 30, tzinfo=timezone.utc),
        )

        aware = normalise_journal_datetime(
            datetime(2026, 6, 25, 10, 30, tzinfo=timezone.utc),
            utc_basis,
        )
        self.assertEqual(aware.status, NormalisationStatus.INVALID_INPUT.value)
        self.assertIsNone(aware.utc_datetime)

    def test_iana_dst_classifications(self):
        ZoneInfo("Europe/London")
        ZoneInfo("America/New_York")
        london = create_time_basis(TIME_BASIS_IANA, zone="Europe/London")

        normal_result = normalise_journal_datetime(
            datetime(2026, 7, 15, 12, 0, 0),
            london,
        )
        self.assertEqual(normal_result.status, NormalisationStatus.VALID.value)
        self.assertEqual(
            normal_result.utc_datetime,
            datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
        )

        missing_result = normalise_journal_datetime(
            datetime(2026, 3, 29, 1, 30, 0),
            london,
        )
        self.assertEqual(
            missing_result.status,
            NormalisationStatus.NONEXISTENT.value,
        )
        self.assertIsNone(missing_result.utc_datetime)

        ambiguous_result = normalise_journal_datetime(
            datetime(2026, 10, 25, 1, 30, 0),
            london,
        )
        self.assertEqual(
            ambiguous_result.status,
            NormalisationStatus.AMBIGUOUS.value,
        )
        self.assertIsNone(ambiguous_result.utc_datetime)
