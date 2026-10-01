from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .codes import (
    FieldType,
    GroundingError,
    LimitationCode,
    RejectionCode,
    ValidationStatus,
)
from .schema import (
    CLAIMABLE_TYPES,
    GROUNDING_RESPONSE_SCHEMA_VERSION,
    GroundedAIRequest,
    GroundedEvidenceItem,
    ModelFacingField,
    ServerGroundingContext,
    request_sha256,
    validate_request_shape,
)

RESPONSE_REQUIRED_FIELDS = ("schema_version", "answer", "claims", "citations", "limitation_codes")
CLAIM_REQUIRED_FIELDS = ("evidence", "field", "value")
LEAKAGE_TOKENS = (
    "request_sha256",
    "journal_fingerprint",
    "alias_to_document_id",
    "owner_id",
    "user_id",
    "content_sha256",
)
INTERNAL_ID_PREFIXES = (
    "TRADE_EVIDENCE:ticket:",
    "KPI_EVIDENCE:kpi:",
    "DATASET_CONTEXT:",
)
INLINE_CITATION_RE = re.compile(r"\[E[1-9][0-9]*\]")
NUMBER_BOUNDARY_LOOKAHEAD = r"(?![A-Za-z0-9_-]|\.[A-Za-z0-9._-])"
ANSWER_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9._-])([+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)(%)?" + NUMBER_BOUNDARY_LOOKAHEAD
)
IDENTIFIER_TICKET_RE = re.compile(
    r"\bticket\s*#?\s*([A-Za-z0-9._-]{1,32})",
    re.IGNORECASE,
)
IDENTIFIER_SYMBOL_RE = re.compile(
    r"\bsymbol\s+([A-Za-z0-9._-]{1,32})",
    re.IGNORECASE,
)
PROHIBITED_RECOMMENDATION_RE = (
    re.compile(r"\byou should buy\b", re.IGNORECASE),
    re.compile(r"\byou should sell\b", re.IGNORECASE),
    re.compile(r"\bi recommend\b", re.IGNORECASE),
    re.compile(r"\benter now\b", re.IGNORECASE),
    re.compile(r"\bexit now\b", re.IGNORECASE),
)
PROHIBITED_PREDICTION_RE = (
    re.compile(r"\bwill rise\b", re.IGNORECASE),
    re.compile(r"\bwill fall\b", re.IGNORECASE),
    re.compile(r"\blikely to rise\b", re.IGNORECASE),
    re.compile(r"\blikely to fall\b", re.IGNORECASE),
    re.compile(r"\bpredict(?:ion|ing|ed)?\b", re.IGNORECASE),
    re.compile(r"\bforecast(?:s|ed|ing)?\b", re.IGNORECASE),
)
PROHIBITED_PRECISION_RE = (
    re.compile(r"\bexact mfe\b", re.IGNORECASE),
    re.compile(r"\bexact mae\b", re.IGNORECASE),
    re.compile(r"\btrue mfe\b", re.IGNORECASE),
    re.compile(r"\btrue mae\b", re.IGNORECASE),
    re.compile(r"\bverified\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\b", re.IGNORECASE),
    re.compile(r"\bprecisely identifies\b", re.IGNORECASE),
)
PROHIBITED_COUNTERFACTUAL_RE = (
    re.compile(r"\bwould have made\b", re.IGNORECASE),
    re.compile(r"\bcould have made\b", re.IGNORECASE),
    re.compile(r"\bif you had entered\b", re.IGNORECASE),
    re.compile(r"\bif you had exited\b", re.IGNORECASE),
    re.compile(r"\bmissed profit\b", re.IGNORECASE),
)
LIMITATION_VALUES = {code.value: code for code in LimitationCode}


@dataclass(frozen=True, slots=True)
class GroundingValidationResult:
    status: ValidationStatus
    rejection_codes: tuple[RejectionCode, ...]
    review_required: bool


def validate_grounded_response(
    request: GroundedAIRequest,
    context: ServerGroundingContext,
    candidate_json: str,
) -> GroundingValidationResult:
    try:
        return _validate_grounded_response(request, context, candidate_json)
    except GroundingError as exc:
        return _result({exc.code})
    except Exception:
        return _result({RejectionCode.SCHEMA_TYPE_ERROR})


def _validate_grounded_response(
    request: GroundedAIRequest,
    context: ServerGroundingContext,
    candidate_json: str,
) -> GroundingValidationResult:
    codes: set[RejectionCode] = set()
    validated_request: GroundedAIRequest | None = None
    if not isinstance(request, GroundedAIRequest):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
    else:
        try:
            validate_request_shape(request)
            validated_request = request
        except GroundingError as exc:
            codes.add(exc.code)
        except Exception:
            codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
    context_object = context if isinstance(context, ServerGroundingContext) else None
    if context_object is None:
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
    elif validated_request is not None and request_sha256(validated_request) != context_object.request_sha256:
        codes.add(RejectionCode.REQUEST_MISMATCH)
    if not isinstance(candidate_json, str):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        return _result(codes)
    _collect_raw_leakage(codes, candidate_json, context_object)
    try:
        payload = json.loads(candidate_json)
    except json.JSONDecodeError:
        codes.add(RejectionCode.SCHEMA_INVALID_JSON)
        return _result(codes)
    if not isinstance(payload, dict):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        return _result(codes)
    _collect_decoded_leakage(codes, payload, context_object)
    _validate_payload(codes, validated_request, payload)
    return _result(codes)


def _result(codes: set[RejectionCode]) -> GroundingValidationResult:
    ordered = tuple(sorted(codes, key=lambda item: item.value))
    status = (
        ValidationStatus.PASSED_DETERMINISTIC_CHECKS
        if not ordered
        else ValidationStatus.REJECTED
    )
    return GroundingValidationResult(
        status=status,
        rejection_codes=ordered,
        review_required=True,
    )


def _context_secrets(context: ServerGroundingContext | None) -> tuple[str, ...]:
    if context is None:
        return ()
    secrets = [context.request_sha256, context.journal_fingerprint]
    secrets.extend(document_id for _alias, document_id in context.alias_to_document_id)
    return tuple(item for item in secrets if item)


def _text_has_leakage(text: str, secrets: Sequence[str]) -> bool:
    if any(token in text for token in LEAKAGE_TOKENS):
        return True
    return any(secret in text for secret in secrets)


def _collect_raw_leakage(
    codes: set[RejectionCode],
    candidate_json: str,
    context: ServerGroundingContext | None,
) -> None:
    if _text_has_leakage(candidate_json, _context_secrets(context)):
        codes.add(RejectionCode.LEAKAGE_DETECTED)


def _collect_decoded_leakage(
    codes: set[RejectionCode],
    payload: Any,
    context: ServerGroundingContext | None,
) -> None:
    if _decoded_has_leakage(payload, _context_secrets(context)):
        codes.add(RejectionCode.LEAKAGE_DETECTED)


def _decoded_has_leakage(value: Any, secrets: Sequence[str]) -> bool:
    if isinstance(value, str):
        return _text_has_leakage(value, secrets)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _text_has_leakage(key, secrets):
                return True
            if _decoded_has_leakage(item, secrets):
                return True
        return False
    if isinstance(value, list):
        return any(_decoded_has_leakage(item, secrets) for item in value)
    return False


def _validate_payload(
    codes: set[RejectionCode],
    request: GroundedAIRequest | None,
    payload: dict[str, Any],
) -> None:
    extras = set(payload) - set(RESPONSE_REQUIRED_FIELDS)
    missing = [key for key in RESPONSE_REQUIRED_FIELDS if key not in payload]
    if extras:
        codes.add(RejectionCode.SCHEMA_UNKNOWN_FIELD)
    if missing:
        codes.add(RejectionCode.SCHEMA_MISSING_FIELD)
    schema_version = payload.get("schema_version")
    if "schema_version" in payload and not isinstance(schema_version, str):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
    elif isinstance(schema_version, str) and schema_version != GROUNDING_RESPONSE_SCHEMA_VERSION:
        codes.add(RejectionCode.SCHEMA_VERSION_MISMATCH)
    answer = payload.get("answer")
    claims = payload.get("claims")
    citations = payload.get("citations")
    limitations = payload.get("limitation_codes")
    constraints = request.constraints if request is not None else None
    if "answer" in payload and not isinstance(answer, str):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        answer = None
    if isinstance(answer, str):
        if not answer.strip():
            codes.add(RejectionCode.ANSWER_EMPTY)
        max_chars = constraints.max_answer_chars if constraints is not None else 2000
        if len(answer) > max_chars:
            codes.add(RejectionCode.SCHEMA_LIMIT_EXCEEDED)
        _collect_prohibited_language(codes, answer)
    if "claims" in payload and not isinstance(claims, list):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        claims = None
    if "citations" in payload and not isinstance(citations, list):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        citations = None
    if "limitation_codes" in payload and not isinstance(limitations, list):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        limitations = None
    if isinstance(claims, list) and constraints is not None and len(claims) > constraints.max_claims:
        codes.add(RejectionCode.SCHEMA_LIMIT_EXCEEDED)
    if isinstance(citations, list) and constraints is not None and len(citations) > constraints.max_citations:
        codes.add(RejectionCode.SCHEMA_LIMIT_EXCEEDED)
    if (
        isinstance(limitations, list)
        and constraints is not None
        and len(limitations) > constraints.max_limitation_codes
    ):
        codes.add(RejectionCode.SCHEMA_LIMIT_EXCEEDED)
    citation_aliases = _validate_citations(codes, request, answer, citations)
    validated_numeric = _validate_claims(codes, request, claims, citation_aliases)
    _validate_limitations(codes, request, limitations)
    if request is not None and isinstance(answer, str):
        _validate_answer_numbers(codes, request, answer, validated_numeric)
        _validate_explicit_identifiers(codes, request, answer)


def _validate_citations(
    codes: set[RejectionCode],
    request: GroundedAIRequest | None,
    answer: str | None,
    citations: list[Any] | None,
) -> set[str]:
    if citations is None:
        return set()
    if any(not isinstance(item, str) for item in citations):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        return set()
    aliases = {item.alias for item in request.evidence} if request is not None else set()
    seen: set[str] = set()
    for value in citations:
        if _is_internal_document_id(value):
            codes.add(RejectionCode.CITATION_INTERNAL_ID)
        elif aliases and value not in aliases:
            codes.add(RejectionCode.CITATION_UNKNOWN_ALIAS)
        if value in seen:
            codes.add(RejectionCode.CITATION_DUPLICATE)
        seen.add(value)
    if isinstance(answer, str) and answer.strip() and not citations:
        codes.add(RejectionCode.CITATION_MISSING)
    if isinstance(answer, str):
        inline = {match[1:-1] for match in INLINE_CITATION_RE.findall(answer)}
        if inline != seen:
            codes.add(RejectionCode.CITATION_INLINE_MISMATCH)
    return seen


def _validate_claims(
    codes: set[RejectionCode],
    request: GroundedAIRequest | None,
    claims: list[Any] | None,
    citation_aliases: set[str],
) -> tuple[tuple[Decimal, str], ...]:
    if claims is None:
        return ()
    bind = request is not None
    evidence_by_alias: dict[str, GroundedEvidenceItem] = {}
    if bind:
        evidence_by_alias = {item.alias: item for item in request.evidence}
    validated: list[tuple[Decimal, str]] = []
    for claim in claims:
        bound = _validate_one_claim(codes, claim, evidence_by_alias, citation_aliases, bind=bind)
        if bound is not None:
            validated.append(bound)
    return tuple(validated)


def _validate_one_claim(
    codes: set[RejectionCode],
    claim: Any,
    evidence_by_alias: Mapping[str, GroundedEvidenceItem],
    citation_aliases: set[str],
    *,
    bind: bool,
) -> tuple[Decimal, str] | None:
    if not isinstance(claim, dict):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        return None
    extras = set(claim) - set(CLAIM_REQUIRED_FIELDS)
    missing = [key for key in CLAIM_REQUIRED_FIELDS if key not in claim]
    if extras:
        codes.add(RejectionCode.SCHEMA_UNKNOWN_FIELD)
    if missing:
        codes.add(RejectionCode.SCHEMA_MISSING_FIELD)
        return None
    evidence = claim["evidence"]
    field_key = claim["field"]
    value = claim["value"]
    if not isinstance(evidence, str) or not isinstance(field_key, str) or not isinstance(value, str):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        return None
    if not bind:
        return None
    item = evidence_by_alias.get(evidence)
    valid = True
    if item is None:
        if _is_internal_document_id(evidence):
            codes.add(RejectionCode.CITATION_INTERNAL_ID)
        else:
            codes.add(RejectionCode.CITATION_UNKNOWN_ALIAS)
        valid = False
    if evidence not in citation_aliases:
        codes.add(RejectionCode.CLAIM_ALIAS_NOT_CITED)
        valid = False
    if item is None:
        return None
    by_key = {field.field_key: field for field in item.fields}
    bound = by_key.get(field_key)
    if bound is None:
        codes.add(RejectionCode.CLAIM_FIELD_UNAVAILABLE)
        return None
    if field_key not in item.claimable_field_keys:
        codes.add(RejectionCode.CLAIM_FIELD_NOT_CLAIMABLE)
        return None
    if bound.canonical_value != value:
        codes.add(RejectionCode.CLAIM_VALUE_MISMATCH)
        return None
    if not valid or bound.value_type not in CLAIMABLE_TYPES:
        return None
    try:
        return Decimal(bound.canonical_value), bound.unit
    except (InvalidOperation, ValueError):
        return None


def _validate_limitations(
    codes: set[RejectionCode],
    request: GroundedAIRequest | None,
    limitations: list[Any] | None,
) -> None:
    if limitations is None:
        return
    if any(not isinstance(item, str) for item in limitations):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
        return
    if len(set(limitations)) != len(limitations):
        codes.add(RejectionCode.SCHEMA_TYPE_ERROR)
    parsed: set[LimitationCode] = set()
    for value in set(limitations):
        code = LIMITATION_VALUES.get(value)
        if code is None:
            codes.add(RejectionCode.LIMITATION_UNKNOWN_CODE)
        else:
            parsed.add(code)
    if request is None:
        return
    required = set(request.required_limitation_codes)
    if required - parsed:
        codes.add(RejectionCode.LIMITATION_REQUIRED_MISSING)
    if parsed - required:
        codes.add(RejectionCode.STATUS_CONTRADICTION)


def _collect_prohibited_language(codes: set[RejectionCode], answer: str) -> None:
    if any(pattern.search(answer) for pattern in PROHIBITED_RECOMMENDATION_RE):
        codes.add(RejectionCode.PROHIBITED_RECOMMENDATION)
    if any(pattern.search(answer) for pattern in PROHIBITED_PREDICTION_RE):
        codes.add(RejectionCode.PROHIBITED_PREDICTION)
    if any(pattern.search(answer) for pattern in PROHIBITED_PRECISION_RE):
        codes.add(RejectionCode.PROHIBITED_PRECISION_CLAIM)
    if any(pattern.search(answer) for pattern in PROHIBITED_COUNTERFACTUAL_RE):
        codes.add(RejectionCode.PROHIBITED_COUNTERFACTUAL)


def _validate_answer_numbers(
    codes: set[RejectionCode],
    request: GroundedAIRequest,
    answer: str,
    validated_numeric: Sequence[tuple[Decimal, str]],
) -> None:
    masked = _mask_non_numeric_tokens(answer, request)
    for match in ANSWER_NUMBER_RE.finditer(masked):
        try:
            number = Decimal(match.group(1))
        except (InvalidOperation, ValueError):
            codes.add(RejectionCode.NUMBER_UNGROUNDED)
            return
        percent = match.group(2) == "%"
        if not _number_is_bound(number, percent, validated_numeric):
            codes.add(RejectionCode.NUMBER_UNGROUNDED)
            return


def _number_is_bound(
    number: Decimal,
    percent: bool,
    validated_numeric: Sequence[tuple[Decimal, str]],
) -> bool:
    for claim_number, unit in validated_numeric:
        if number != claim_number:
            continue
        if percent and unit != "%":
            continue
        return True
    return False


def _mask_non_numeric_tokens(answer: str, request: GroundedAIRequest) -> str:
    masked = INLINE_CITATION_RE.sub(" ", answer)
    for field in _iter_fields(request):
        if field.value_type == FieldType.TIMESTAMP:
            masked = masked.replace(field.canonical_value, " ")
    for field in _iter_fields(request):
        if field.value_type != FieldType.IDENTIFIER:
            continue
        token = re.escape(field.canonical_value)
        masked = re.sub(
            rf"(?<![A-Za-z0-9._-]){token}{NUMBER_BOUNDARY_LOOKAHEAD}",
            " ",
            masked,
        )
    return masked


def _validate_explicit_identifiers(
    codes: set[RejectionCode],
    request: GroundedAIRequest,
    answer: str,
) -> None:
    tickets = {field.canonical_value for field in _iter_fields(request) if field.field_key == "ticket"}
    symbols = {field.canonical_value for field in _iter_fields(request) if field.field_key == "symbol"}
    for match in IDENTIFIER_TICKET_RE.finditer(answer):
        if _canonical_identifier(match.group(1), tickets) not in tickets:
            codes.add(RejectionCode.IDENTIFIER_UNGROUNDED)
            return
    for match in IDENTIFIER_SYMBOL_RE.finditer(answer):
        if _canonical_identifier(match.group(1), symbols) not in symbols:
            codes.add(RejectionCode.IDENTIFIER_UNGROUNDED)
            return


def _canonical_identifier(extracted: str, allowed: set[str]) -> str:
    token = extracted
    while token.endswith(".") and token not in allowed:
        token = token[:-1]
    return token


def _iter_fields(request: GroundedAIRequest) -> Iterable[ModelFacingField]:
    for item in request.evidence:
        yield from item.fields


def _is_internal_document_id(value: str) -> bool:
    return value.startswith(INTERNAL_ID_PREFIXES)


__all__ = [
    "ANSWER_NUMBER_RE",
    "GroundingValidationResult",
    "validate_grounded_response",
]
