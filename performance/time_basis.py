from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIME_BASIS_UTC = "UTC"
TIME_BASIS_FIXED_OFFSET = "FIXED_OFFSET"
TIME_BASIS_IANA = "IANA"
TIME_BASIS_RESOLUTION = "USER_DECLARED"

MIN_OFFSET_MINUTES = -720
MAX_OFFSET_MINUTES = 840
OFFSET_STEP_MINUTES = 15


class TimeBasisValidationError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


class NormalisationStatus(str, Enum):
    VALID = "VALID"
    AMBIGUOUS = "AMBIGUOUS"
    NONEXISTENT = "NONEXISTENT"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class TimeBasis:
    kind: str
    resolution: str = TIME_BASIS_RESOLUTION
    offset_minutes: int | None = None
    zone: str | None = None

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "resolution": self.resolution,
        }
        if self.kind == TIME_BASIS_FIXED_OFFSET:
            payload["offset_minutes"] = self.offset_minutes
        if self.kind == TIME_BASIS_IANA:
            payload["zone"] = self.zone
        return payload


@dataclass(frozen=True)
class NormalisationResult:
    status: str
    utc_datetime: datetime | None = None


def _raise(reason: str, message: str = "") -> None:
    raise TimeBasisValidationError(reason, message)


def create_time_basis(
    kind: str,
    *,
    offset_minutes: int | None = None,
    zone: str | None = None,
) -> TimeBasis:
    if kind == TIME_BASIS_UTC:
        if offset_minutes is not None or zone is not None:
            _raise(
                "CONFLICTING_FIELDS",
                "UTC declarations cannot include an offset or zone",
            )
        return TimeBasis(
            kind=TIME_BASIS_UTC,
            resolution=TIME_BASIS_RESOLUTION,
        )
    if kind == TIME_BASIS_FIXED_OFFSET:
        if zone is not None:
            _raise(
                "CONFLICTING_FIELDS",
                "FIXED_OFFSET declarations cannot include an IANA zone",
            )
        if not isinstance(offset_minutes, int) or isinstance(offset_minutes, bool):
            _raise("INVALID_OFFSET", "FIXED_OFFSET requires an integer offset")
        if offset_minutes < MIN_OFFSET_MINUTES or offset_minutes > MAX_OFFSET_MINUTES:
            _raise(
                "OFFSET_OUT_OF_RANGE",
                "FIXED_OFFSET is outside the permitted range",
            )
        if offset_minutes % OFFSET_STEP_MINUTES != 0:
            _raise(
                "OFFSET_NOT_15_MINUTE_STEP",
                "FIXED_OFFSET must be a 15-minute step",
            )
        return TimeBasis(
            kind=TIME_BASIS_FIXED_OFFSET,
            resolution=TIME_BASIS_RESOLUTION,
            offset_minutes=offset_minutes,
        )
    if kind == TIME_BASIS_IANA:
        if offset_minutes is not None:
            _raise(
                "CONFLICTING_FIELDS",
                "IANA declarations cannot include a fixed offset",
            )
        if zone is None or str(zone).strip() == "":
            _raise("ZONE_REQUIRED", "IANA declarations require a zone")
        try:
            ZoneInfo(str(zone))
        except (ZoneInfoNotFoundError, ValueError):
            _raise("ZONE_NOT_FOUND", "IANA zone is not available")
        return TimeBasis(
            kind=TIME_BASIS_IANA,
            resolution=TIME_BASIS_RESOLUTION,
            zone=str(zone),
        )
    _raise("INVALID_KIND", "Time basis kind is not supported")
    raise AssertionError("unreachable")


def _iana_utc_mappings(naive: datetime, zone_name: str) -> list[datetime]:
    zone = ZoneInfo(zone_name)
    mappings: list[datetime] = []
    for fold in (0, 1):
        local = naive.replace(tzinfo=zone, fold=fold)
        utc_value = local.astimezone(timezone.utc)
        wall = utc_value.astimezone(zone).replace(tzinfo=None)
        if wall != naive.replace(tzinfo=None):
            continue
        if utc_value not in mappings:
            mappings.append(utc_value)
    return mappings


def normalise_journal_datetime(
    value: Any,
    time_basis: TimeBasis,
) -> NormalisationResult:
    if not isinstance(value, datetime):
        return NormalisationResult(status=NormalisationStatus.INVALID_INPUT.value)
    if value.tzinfo is not None:
        return NormalisationResult(status=NormalisationStatus.INVALID_INPUT.value)

    if time_basis.kind == TIME_BASIS_UTC:
        return NormalisationResult(
            status=NormalisationStatus.VALID.value,
            utc_datetime=value.replace(tzinfo=timezone.utc),
        )
    if time_basis.kind == TIME_BASIS_FIXED_OFFSET:
        offset = timedelta(minutes=time_basis.offset_minutes or 0)
        utc_value = value.replace(tzinfo=timezone.utc) - offset
        return NormalisationResult(
            status=NormalisationStatus.VALID.value,
            utc_datetime=utc_value,
        )
    if time_basis.kind == TIME_BASIS_IANA and time_basis.zone:
        mappings = _iana_utc_mappings(value, time_basis.zone)
        if len(mappings) == 0:
            return NormalisationResult(status=NormalisationStatus.NONEXISTENT.value)
        if len(mappings) == 1:
            return NormalisationResult(
                status=NormalisationStatus.VALID.value,
                utc_datetime=mappings[0],
            )
        return NormalisationResult(status=NormalisationStatus.AMBIGUOUS.value)
    return NormalisationResult(status=NormalisationStatus.INVALID_INPUT.value)
