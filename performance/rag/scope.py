from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..excursion import compute_journal_fingerprint


class RetrievalScopeError(Exception):
    def __init__(self, reason: str, message: str = "") -> None:
        self.reason = reason
        super().__init__(message or reason)


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    owner_id: int
    journal_fingerprint: str


def resolve_retrieval_scope(user: Any, journal: pd.DataFrame | None) -> RetrievalScope:
    if user is None or not bool(getattr(user, "is_authenticated", False)):
        raise RetrievalScopeError(
            "UNAUTHENTICATED_SCOPE",
            "Retrieval scope requires an authenticated user",
        )
    owner_id = getattr(user, "pk", None)
    if owner_id is None:
        raise RetrievalScopeError(
            "UNAUTHENTICATED_SCOPE",
            "Authenticated user has no stable identity",
        )
    if journal is None or getattr(journal, "empty", True):
        raise RetrievalScopeError(
            "INVALID_JOURNAL_FINGERPRINT",
            "Active journal is missing or empty",
        )
    fingerprint = compute_journal_fingerprint(journal)
    if not fingerprint:
        raise RetrievalScopeError(
            "INVALID_JOURNAL_FINGERPRINT",
            "Journal fingerprint could not be produced",
        )
    return RetrievalScope(
        owner_id=int(owner_id),
        journal_fingerprint=str(fingerprint),
    )
