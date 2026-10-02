from __future__ import annotations

from dataclasses import dataclass

from ..ai_grounding.codes import FROZEN_REJECTION_CODES


@dataclass(frozen=True, slots=True)
class MutationUnit:
    check_id: str
    module_name: str
    attribute: str
    owned_codes: tuple[str, ...]
    killing_cases: tuple[str, ...]
    stub_kind: str


@dataclass(frozen=True, slots=True)
class DifferentialPair:
    code: str
    attack_case_id: str
    control_case_id: str


@dataclass(frozen=True, slots=True)
class SecondaryCodeEmission:
    check_id: str
    codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecondaryBuildTimePath:
    code: str
    path: str


VALIDATOR_MUTATION_UNITS: tuple[MutationUnit, ...] = (
    MutationUnit(
        "VAL-01",
        "performance.ai_grounding.validation",
        "_collect_raw_leakage",
        ("LEAKAGE_DETECTED",),
        ("S7-LEAKAGE-002",),
        "noop",
    ),
    MutationUnit(
        "VAL-02",
        "performance.ai_grounding.validation",
        "_collect_decoded_leakage",
        ("LEAKAGE_DETECTED",),
        ("S7-LEAKAGE-003",),
        "noop",
    ),
    MutationUnit(
        "VAL-03",
        "performance.ai_grounding.validation",
        "_collect_prohibited_language",
        (
            "PROHIBITED_RECOMMENDATION",
            "PROHIBITED_PREDICTION",
            "PROHIBITED_PRECISION_CLAIM",
            "PROHIBITED_COUNTERFACTUAL",
        ),
        (
            "S7-LANGUAGE-001",
            "S7-LANGUAGE-002",
            "S7-LANGUAGE-003",
            "S7-LANGUAGE-004",
        ),
        "noop",
    ),
    MutationUnit(
        "VAL-04",
        "performance.ai_grounding.validation",
        "_validate_citations",
        (
            "CITATION_MISSING",
            "CITATION_UNKNOWN_ALIAS",
            "CITATION_DUPLICATE",
            "CITATION_INLINE_MISMATCH",
            "CITATION_INTERNAL_ID",
        ),
        (
            "S7-CITATION-001",
            "S7-CITATION-002",
            "S7-CITATION-003",
            "S7-CITATION-004",
            "S7-CITATION-005",
        ),
        "scratch_codes",
    ),
    MutationUnit(
        "VAL-05",
        "performance.ai_grounding.validation",
        "_validate_one_claim",
        (
            "CLAIM_ALIAS_NOT_CITED",
            "CLAIM_FIELD_UNAVAILABLE",
            "CLAIM_FIELD_NOT_CLAIMABLE",
            "CLAIM_VALUE_MISMATCH",
        ),
        (
            "S7-CLAIM-001",
            "S7-CLAIM-002",
            "S7-CLAIM-003",
            "S7-CLAIM-004",
        ),
        "scratch_codes",
    ),
    MutationUnit(
        "VAL-06",
        "performance.ai_grounding.validation",
        "_validate_limitations",
        (
            "LIMITATION_REQUIRED_MISSING",
            "LIMITATION_UNKNOWN_CODE",
            "STATUS_CONTRADICTION",
        ),
        (
            "S7-LIMITATION-001",
            "S7-LIMITATION-002",
            "S7-LIMITATION-003",
        ),
        "noop",
    ),
    MutationUnit(
        "VAL-07",
        "performance.ai_grounding.validation",
        "_validate_answer_numbers",
        ("NUMBER_UNGROUNDED",),
        ("S7-NUMBER-001",),
        "noop",
    ),
    MutationUnit(
        "VAL-08",
        "performance.ai_grounding.validation",
        "_validate_explicit_identifiers",
        ("IDENTIFIER_UNGROUNDED",),
        ("S7-IDENTIFIER-001",),
        "noop",
    ),
)

BUILD_TIME_MUTATION_UNITS: tuple[MutationUnit, ...] = (
    MutationUnit(
        "BLD-01",
        "performance.ai_grounding.packet",
        "_validate_symbol",
        ("EVIDENCE_FIELD_INVALID",),
        ("S7-INJECTION-002",),
        "identity_raw",
    ),
)

INLINE_DIFFERENTIAL_PAIRS: tuple[DifferentialPair, ...] = (
    DifferentialPair("REQUEST_MISMATCH", "S7-STRUCTURE-003", "S7-CONTROL-001"),
    DifferentialPair("SCHEMA_INVALID_JSON", "S7-PARSE-001", "S7-CONTROL-011"),
    DifferentialPair("SCHEMA_TYPE_ERROR", "S7-PARSE-002", "S7-CONTROL-011"),
    DifferentialPair("SCHEMA_UNKNOWN_FIELD", "S7-PARSE-004", "S7-CONTROL-001"),
    DifferentialPair("SCHEMA_MISSING_FIELD", "S7-PARSE-003", "S7-CONTROL-001"),
    DifferentialPair("SCHEMA_VERSION_MISMATCH", "S7-PARSE-005", "S7-CONTROL-001"),
    DifferentialPair("SCHEMA_LIMIT_EXCEEDED", "S7-PARSE-006", "S7-CONTROL-024"),
    DifferentialPair("ANSWER_EMPTY", "S7-MIXED-001", "S7-CONTROL-025"),
)

BUILD_TIME_INLINE_PAIRS: tuple[DifferentialPair, ...] = (
    DifferentialPair("REQUEST_STATE_NOT_OK", "S7-STRUCTURE-001", "S7-CONTROL-001"),
    DifferentialPair("REQUEST_EMPTY_EVIDENCE", "S7-STRUCTURE-002", "S7-CONTROL-001"),
)

MUTATION_OWNED_CODES = (
    "LEAKAGE_DETECTED",
    "PROHIBITED_RECOMMENDATION",
    "PROHIBITED_PREDICTION",
    "PROHIBITED_PRECISION_CLAIM",
    "PROHIBITED_COUNTERFACTUAL",
    "CITATION_MISSING",
    "CITATION_UNKNOWN_ALIAS",
    "CITATION_DUPLICATE",
    "CITATION_INLINE_MISMATCH",
    "CITATION_INTERNAL_ID",
    "CLAIM_ALIAS_NOT_CITED",
    "CLAIM_FIELD_UNAVAILABLE",
    "CLAIM_FIELD_NOT_CLAIMABLE",
    "CLAIM_VALUE_MISMATCH",
    "LIMITATION_REQUIRED_MISSING",
    "LIMITATION_UNKNOWN_CODE",
    "STATUS_CONTRADICTION",
    "NUMBER_UNGROUNDED",
    "IDENTIFIER_UNGROUNDED",
)

INLINE_DIFFERENTIAL_CODES = tuple(item.code for item in INLINE_DIFFERENTIAL_PAIRS)
BUILD_TIME_CODES = (
    "REQUEST_STATE_NOT_OK",
    "REQUEST_EMPTY_EVIDENCE",
    "EVIDENCE_FIELD_INVALID",
)

VALIDATOR_SECONDARY_EMISSIONS: tuple[SecondaryCodeEmission, ...] = (
    SecondaryCodeEmission("VAL-02", ("LEAKAGE_DETECTED",)),
    SecondaryCodeEmission("VAL-04", ("SCHEMA_TYPE_ERROR",)),
    SecondaryCodeEmission(
        "VAL-05",
        (
            "SCHEMA_TYPE_ERROR",
            "SCHEMA_UNKNOWN_FIELD",
            "SCHEMA_MISSING_FIELD",
            "CITATION_INTERNAL_ID",
            "CITATION_UNKNOWN_ALIAS",
        ),
    ),
    SecondaryCodeEmission("VAL-06", ("SCHEMA_TYPE_ERROR",)),
)

BUILD_TIME_SECONDARY_PATHS: tuple[SecondaryBuildTimePath, ...] = (
    SecondaryBuildTimePath("REQUEST_MISMATCH", "rank-sequence mismatch"),
    SecondaryBuildTimePath("REQUEST_MISMATCH", "mixed journal fingerprints"),
    SecondaryBuildTimePath("REQUEST_MISMATCH", "content/provenance symbol disagreement"),
    SecondaryBuildTimePath("SCHEMA_LIMIT_EXCEEDED", "packet evidence-item limit"),
    SecondaryBuildTimePath(
        "EVIDENCE_FIELD_INVALID",
        "_validate_side",
    ),
    SecondaryBuildTimePath(
        "EVIDENCE_FIELD_INVALID",
        "_project_trade malformed/unavailable computed excursion evidence",
    ),
    SecondaryBuildTimePath("EVIDENCE_FIELD_INVALID", "_project_kpi unknown field"),
    SecondaryBuildTimePath(
        "EVIDENCE_FIELD_INVALID",
        "_project_result unsupported document/status paths",
    ),
    SecondaryBuildTimePath(
        "EVIDENCE_FIELD_INVALID",
        "downstream schema symbol re-check relevant to BLD-01 redundancy",
    ),
    SecondaryBuildTimePath(
        "LEAKAGE_DETECTED",
        "packet canonical-request leakage assertion",
    ),
)

SECONDARY_EMISSIONS = tuple(
    (code, item.check_id) for item in VALIDATOR_SECONDARY_EMISSIONS for code in item.codes
) + tuple((item.code, item.path) for item in BUILD_TIME_SECONDARY_PATHS)

REQUIRED_NEAR_MISS_CATEGORIES = (
    "STRUCTURE",
    "PARSE",
    "CITATION",
    "ALIAS_FORMAT",
    "NUMBER",
    "NUMBER_FORMAT",
    "IDENTIFIER",
    "LIMITATION",
    "CROSS_EVIDENCE",
    "MIXED_GROUNDING",
    "ENCODING",
    "CONFIDENCE",
    "UNSUPPORTED_CLAIM",
    "CLAIM_LAUNDERING",
    "SELF_CERTIFICATION",
    "INJECTION",
)

RENDERING_PAIRING_EXEMPTION = "RENDERING"


def frozen_code_buckets() -> dict[str, str]:
    buckets: dict[str, str] = {}
    for code in MUTATION_OWNED_CODES:
        buckets[code] = "MUTATION"
    for code in INLINE_DIFFERENTIAL_CODES:
        buckets[code] = "INLINE_DIFFERENTIAL"
    for code in BUILD_TIME_CODES:
        buckets[code] = "BUILD_TIME"
    return buckets


def assert_manifest_coverage() -> tuple[str, ...]:
    buckets = frozen_code_buckets()
    missing = tuple(code for code in FROZEN_REJECTION_CODES if code not in buckets)
    extra = tuple(code for code in sorted(buckets) if code not in FROZEN_REJECTION_CODES)
    return missing + extra
