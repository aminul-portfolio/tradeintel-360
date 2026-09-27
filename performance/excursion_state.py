from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import Any

from .excursion import ExcursionRunResult
from .market_data import EXCURSION_CONTRACT_VERSION, MarketDataProvenance
from .time_basis import TimeBasis

EXCURSION_SESSION_KEY = "broker_bar_excursion_evidence_v1"
MAX_EVIDENCE_SESSION_BYTES = 512 * 1024

REQUIRED_BINDING_FIELDS = (
    "contract_version",
    "journal_fingerprint",
    "market_file_sha256",
    "time_basis",
    "market_provenance",
    "run_status",
    "status_counts",
    "reason_counts",
    "evidence",
)


class ExcursionStateError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


def _encode_json(payload: Any) -> bytes:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return encoded.encode("utf-8")


def _serialise_payload(payload: Any) -> bytes:
    try:
        return _encode_json(payload)
    except (TypeError, ValueError) as exc:
        raise ExcursionStateError(
            "NON_JSON_SAFE_PAYLOAD",
            "Excursion evidence payload is not JSON-safe",
        ) from exc


def _detach_json(encoded: bytes) -> Any:
    return json.loads(encoded.decode("utf-8"))


def build_excursion_state_payload(
    result: ExcursionRunResult,
    *,
    time_basis: TimeBasis,
    market_provenance: MarketDataProvenance,
) -> dict[str, Any]:
    return {
        "contract_version": EXCURSION_CONTRACT_VERSION,
        "journal_fingerprint": result.journal_fingerprint,
        "market_file_sha256": market_provenance.market_file_sha256,
        "time_basis": time_basis.as_json(),
        "market_provenance": market_provenance.as_json(),
        "run_status": result.run_status,
        "status_counts": dict(result.status_counts),
        "reason_counts": dict(result.reason_counts),
        "evidence": dict(result.evidence),
    }


def store_excursion_state(
    state: MutableMapping[str, Any],
    result: ExcursionRunResult,
    *,
    time_basis: TimeBasis,
    market_provenance: MarketDataProvenance,
) -> dict[str, Any]:
    payload = build_excursion_state_payload(
        result,
        time_basis=time_basis,
        market_provenance=market_provenance,
    )
    encoded = _serialise_payload(payload)
    if len(encoded) > MAX_EVIDENCE_SESSION_BYTES:
        raise ExcursionStateError(
            "PAYLOAD_TOO_LARGE",
            "Excursion evidence payload exceeds the session size limit",
        )
    detached = _detach_json(encoded)
    state[EXCURSION_SESSION_KEY] = detached
    return detached


def clear_excursion_state(state: MutableMapping[str, Any]) -> None:
    state.pop(EXCURSION_SESSION_KEY, None)


def _payload_is_bound(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for field in REQUIRED_BINDING_FIELDS:
        if field not in payload:
            return False
    if payload.get("contract_version") != EXCURSION_CONTRACT_VERSION:
        return False
    if not isinstance(payload.get("contract_version"), str):
        return False
    if not isinstance(payload.get("journal_fingerprint"), str) or not payload["journal_fingerprint"]:
        return False
    if not isinstance(payload.get("market_file_sha256"), str) or not payload["market_file_sha256"]:
        return False
    if not isinstance(payload.get("time_basis"), dict):
        return False
    if not isinstance(payload.get("market_provenance"), dict):
        return False
    if not isinstance(payload.get("run_status"), str) or not payload["run_status"]:
        return False
    if not isinstance(payload.get("status_counts"), dict):
        return False
    if not isinstance(payload.get("reason_counts"), dict):
        return False
    if not isinstance(payload.get("evidence"), dict):
        return False
    return True


def load_bound_excursion_state(
    state: MutableMapping[str, Any],
    *,
    current_journal_fingerprint: str | None,
) -> dict[str, Any] | None:
    if EXCURSION_SESSION_KEY not in state:
        return None
    payload = state.get(EXCURSION_SESSION_KEY)
    try:
        encoded = _encode_json(payload)
        detached = _detach_json(encoded)
    except (TypeError, ValueError):
        clear_excursion_state(state)
        return None
    if len(encoded) > MAX_EVIDENCE_SESSION_BYTES:
        clear_excursion_state(state)
        return None
    if current_journal_fingerprint in {None, ""} or not _payload_is_bound(detached):
        clear_excursion_state(state)
        return None
    if detached.get("journal_fingerprint") != current_journal_fingerprint:
        clear_excursion_state(state)
        return None
    return detached
