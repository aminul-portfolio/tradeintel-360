from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from unittest.mock import patch

from ..ai_grounding import (
    FROZEN_REJECTION_CODES,
    GroundingError,
    GroundingValidationResult,
    ValidationStatus,
    build_evidence_packet,
    validate_grounded_response,
)
from .check_manifest import (
    BUILD_TIME_INLINE_PAIRS,
    BUILD_TIME_MUTATION_UNITS,
    INLINE_DIFFERENTIAL_PAIRS,
    VALIDATOR_MUTATION_UNITS,
    MutationUnit,
    assert_manifest_coverage,
    frozen_code_buckets,
)
from .corpus import (
    QUESTION,
    EvaluationCase,
    ExpectedOutcome,
    load_corpus,
    mutates_context_sha,
    retrieval_for,
)

SCHEMA_TYPE_ERROR = "SCHEMA_TYPE_ERROR"


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    expected_outcome: str
    actual_status: str
    actual_rejection_codes: tuple[str, ...]
    required_codes: tuple[str, ...]
    forbidden_codes: tuple[str, ...]
    matched: bool
    review_required: bool
    packet_error: bool
    used_production_packet: bool


@dataclass(frozen=True, slots=True)
class MutationUnitResult:
    check_id: str
    killed: bool
    survived: bool
    mutation_invalid: bool
    survived_redundant: bool
    spy_calls: tuple[tuple[str, int], ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    evaluations: tuple[CaseEvaluation, ...]
    total_cases: int
    expect_reject_cases: int
    expect_pass_cases: int
    known_limitation_cases: int
    matched_expectations: int
    unexpected_outcomes: int
    false_accept_style_mismatches: int
    false_reject_style_mismatches: int
    frozen_rejection_code_count: int
    kill_matrix_killed: int
    kill_matrix_total: int
    survived: tuple[str, ...]
    mutation_invalid: tuple[str, ...]
    build_time_killed: int
    build_time_total: int
    build_time_survived_redundant: tuple[str, ...]
    inline_differential_passing: int
    inline_differential_total: int
    build_time_inline_passing: int
    build_time_inline_total: int
    expectation_sensitivity_covered: int
    expectation_sensitivity_total: int
    expectation_sensitivity_misses: tuple[str, ...]
    manifest_coverage_covered: int
    manifest_coverage_total: int
    category_counts: tuple[tuple[str, int], ...]
    unexpected_cases: tuple[CaseEvaluation, ...]
    known_limitation_rows: tuple[CaseEvaluation, ...]
    mutation_results: tuple[MutationUnitResult, ...]
    build_time_results: tuple[MutationUnitResult, ...]


def _code_values(result: GroundingValidationResult) -> tuple[str, ...]:
    return tuple(code.value for code in result.rejection_codes)


def _packet_error_result(exc: GroundingError) -> GroundingValidationResult:
    return GroundingValidationResult(
        status=ValidationStatus.REJECTED,
        rejection_codes=(exc.code,),
        review_required=True,
    )


def reject_expectation_holds(case: EvaluationCase, codes: Sequence[str]) -> bool:
    actual = set(codes)
    required = set(case.required_rejection_codes)
    forbidden = set(case.forbidden_rejection_codes)
    return required <= actual and actual.isdisjoint(forbidden)


def expectation_holds(case: EvaluationCase, result: GroundingValidationResult) -> bool:
    codes = _code_values(result)
    if case.expected_outcome == ExpectedOutcome.EXPECT_PASS:
        return (
            result.status == ValidationStatus.PASSED_DETERMINISTIC_CHECKS
            and codes == ()
            and result.review_required is True
        )
    if case.expected_outcome == ExpectedOutcome.EXPECT_REJECT:
        return result.status == ValidationStatus.REJECTED and reject_expectation_holds(case, codes)
    if case.documented_status:
        return result.status.value == case.documented_status
    return True


def evaluate_case(case: EvaluationCase) -> CaseEvaluation:
    retrieval = retrieval_for(case.fixture_id)
    packet_error = False
    used_production_packet = True
    try:
        request, context = build_evidence_packet(retrieval, QUESTION)
    except GroundingError as exc:
        result = _packet_error_result(exc)
        packet_error = True
    else:
        if mutates_context_sha(case.fixture_id):
            context = replace(context, request_sha256="a" * 64)
        result = validate_grounded_response(request, context, case.candidate_json)
    return CaseEvaluation(
        case_id=case.case_id,
        expected_outcome=case.expected_outcome.value,
        actual_status=result.status.value,
        actual_rejection_codes=_code_values(result),
        required_codes=case.required_rejection_codes,
        forbidden_codes=case.forbidden_rejection_codes,
        matched=expectation_holds(case, result),
        review_required=result.review_required,
        packet_error=packet_error,
        used_production_packet=used_production_packet,
    )


def evaluate_corpus(cases: Sequence[EvaluationCase] | None = None) -> tuple[CaseEvaluation, ...]:
    corpus = tuple(cases) if cases is not None else load_corpus()
    return tuple(evaluate_case(case) for case in corpus)


def _expectation_sensitivity(
    corpus: Sequence[EvaluationCase],
    evaluations: Sequence[CaseEvaluation],
) -> tuple[int, tuple[str, ...]]:
    by_id = {item.case_id: item for item in evaluations}
    covered = 0
    misses: list[str] = []
    for code in FROZEN_REJECTION_CODES:
        owners = [
            case
            for case in corpus
            if case.expected_outcome == ExpectedOutcome.EXPECT_REJECT and code in case.required_rejection_codes
        ]
        if not owners:
            misses.append(code)
            continue
        observed = by_id[owners[0].case_id]
        if not observed.matched:
            misses.append(code)
            continue
        mutated = tuple(item for item in observed.actual_rejection_codes if item != code)
        if reject_expectation_holds(owners[0], mutated):
            misses.append(code)
            continue
        covered += 1
    return covered, tuple(misses)


def _make_stub(unit: MutationUnit, original: Callable[..., object], raised_flag: list[bool]) -> Callable[..., object]:
    if unit.stub_kind == "noop":

        def noop(*args: object, **kwargs: object) -> None:
            return None

        return noop
    if unit.stub_kind == "scratch_codes":

        def scratch_codes(*args: object, **kwargs: object) -> object:
            try:
                return original(set(), *args[1:], **kwargs)
            except Exception:
                raised_flag.append(True)
                raise

        return scratch_codes
    if unit.stub_kind == "identity_raw":

        def identity_raw(raw: object) -> object:
            return raw

        return identity_raw
    raise ValueError("Unknown mutation stub kind: " + unit.stub_kind)


def _owned_required(case: EvaluationCase, unit: MutationUnit) -> tuple[str, ...]:
    owned = set(unit.owned_codes)
    return tuple(code for code in case.required_rejection_codes if code in owned)


def evaluate_mutation_unit(
    unit: MutationUnit,
    corpus: Sequence[EvaluationCase],
    baseline_by_id: dict[str, CaseEvaluation],
    *,
    stub_factory: Callable[[MutationUnit, Callable[..., object], list[bool]], Callable[..., object]] | None = None,
) -> MutationUnitResult:
    cases_by_id = {case.case_id: case for case in corpus}
    if unit.module_name == "performance.ai_grounding.validation":
        module = importlib.import_module("performance.ai_grounding.validation")
    elif unit.module_name == "performance.ai_grounding.packet":
        module = importlib.import_module("performance.ai_grounding.packet")
    else:
        module = importlib.import_module(unit.module_name)
    original = getattr(module, unit.attribute)
    spy_calls: list[tuple[str, int]] = []
    for case_id in unit.killing_cases:
        with patch.object(module, unit.attribute, wraps=original) as spy:
            evaluate_case(cases_by_id[case_id])
            spy_calls.append((case_id, spy.call_count))
    raised_flag: list[bool] = []
    factory = stub_factory or _make_stub
    stub = factory(unit, original, raised_flag)
    pass_cases = [case for case in corpus if case.expected_outcome == ExpectedOutcome.EXPECT_PASS]
    mutated_killing: dict[str, CaseEvaluation] = {}
    mutated_controls: dict[str, CaseEvaluation] = {}
    with patch.object(module, unit.attribute, stub):
        for case_id in unit.killing_cases:
            mutated_killing[case_id] = evaluate_case(cases_by_id[case_id])
        for case in pass_cases:
            mutated_controls[case.case_id] = evaluate_case(case)
    spy_tuple = tuple(spy_calls)
    if raised_flag or any(count == 0 for _, count in spy_calls):
        return MutationUnitResult(
            check_id=unit.check_id,
            killed=False,
            survived=False,
            mutation_invalid=True,
            survived_redundant=False,
            spy_calls=spy_tuple,
            notes="stub raised or spy observed zero calls",
        )
    for case_id, mutated in mutated_killing.items():
        baseline = baseline_by_id[case_id]
        mutated_has_schema_type = SCHEMA_TYPE_ERROR in mutated.actual_rejection_codes
        baseline_has_schema_type = SCHEMA_TYPE_ERROR in baseline.actual_rejection_codes
        if mutated_has_schema_type and not baseline_has_schema_type:
            return MutationUnitResult(
                check_id=unit.check_id,
                killed=False,
                survived=False,
                mutation_invalid=True,
                survived_redundant=False,
                spy_calls=spy_tuple,
                notes="SCHEMA_TYPE_ERROR appeared only under mutation",
            )
    controls_stable = True
    for case in pass_cases:
        baseline = baseline_by_id[case.case_id]
        mutated = mutated_controls[case.case_id]
        if (
            mutated.actual_status != baseline.actual_status
            or mutated.actual_rejection_codes != baseline.actual_rejection_codes
        ):
            controls_stable = False
            break
    owned_lost = True
    still_has_owned = False
    for case_id in unit.killing_cases:
        case = cases_by_id[case_id]
        baseline = baseline_by_id[case_id]
        mutated = mutated_killing[case_id]
        required_owned = _owned_required(case, unit)
        if not required_owned:
            owned_lost = False
            break
        if any(code not in baseline.actual_rejection_codes for code in required_owned):
            owned_lost = False
            break
        if any(code in mutated.actual_rejection_codes for code in required_owned):
            owned_lost = False
            still_has_owned = True
            break
    if unit.check_id.startswith("BLD-") and still_has_owned and controls_stable:
        return MutationUnitResult(
            check_id=unit.check_id,
            killed=False,
            survived=False,
            mutation_invalid=False,
            survived_redundant=True,
            spy_calls=spy_tuple,
            notes="owned code retained by a later production re-check",
        )
    killed = owned_lost and controls_stable
    return MutationUnitResult(
        check_id=unit.check_id,
        killed=killed,
        survived=not killed,
        mutation_invalid=False,
        survived_redundant=False,
        spy_calls=spy_tuple,
        notes="" if killed else "owned code retained or control drifted",
    )


def _differential_passing(
    pairs: Sequence,
    corpus: Sequence[EvaluationCase],
    baseline_by_id: dict[str, CaseEvaluation],
) -> int:
    cases_by_id = {case.case_id: case for case in corpus}
    passing = 0
    for item in pairs:
        attack_case = cases_by_id[item.attack_case_id]
        control_case = cases_by_id[item.control_case_id]
        attack = baseline_by_id[item.attack_case_id]
        control = baseline_by_id[item.control_case_id]
        attack_ok = (
            attack_case.expected_outcome == ExpectedOutcome.EXPECT_REJECT
            and item.code in attack.actual_rejection_codes
            and attack.matched
        )
        control_ok = (
            control_case.expected_outcome == ExpectedOutcome.EXPECT_PASS
            and item.code not in control.actual_rejection_codes
            and control.matched
            and control.actual_status == ValidationStatus.PASSED_DETERMINISTIC_CHECKS.value
        )
        if attack_ok and control_ok:
            passing += 1
    return passing


def run_evaluation(cases: Sequence[EvaluationCase] | None = None) -> EvaluationReport:
    corpus = tuple(cases) if cases is not None else load_corpus()
    evaluations = evaluate_corpus(corpus)
    reject_cases = [case for case in corpus if case.expected_outcome == ExpectedOutcome.EXPECT_REJECT]
    pass_cases = [case for case in corpus if case.expected_outcome == ExpectedOutcome.EXPECT_PASS]
    known_cases = [case for case in corpus if case.expected_outcome == ExpectedOutcome.KNOWN_LIMITATION]
    by_id = {item.case_id: item for item in evaluations}

    scored = [
        by_id[case.case_id]
        for case in corpus
        if case.expected_outcome != ExpectedOutcome.KNOWN_LIMITATION
    ]
    unexpected = tuple(item for item in scored if not item.matched)
    false_accept = 0
    false_reject = 0
    for item in unexpected:
        if item.expected_outcome == ExpectedOutcome.EXPECT_REJECT:
            if item.actual_status == ValidationStatus.PASSED_DETERMINISTIC_CHECKS.value:
                false_accept += 1
        elif item.expected_outcome == ExpectedOutcome.EXPECT_PASS:
            false_reject += 1

    known_drift = tuple(by_id[case.case_id] for case in known_cases if not by_id[case.case_id].matched)
    unexpected_all = tuple(sorted(unexpected + known_drift, key=lambda item: item.case_id))
    sensitivity_covered, sensitivity_misses = _expectation_sensitivity(corpus, evaluations)
    mutation_results = tuple(evaluate_mutation_unit(unit, corpus, by_id) for unit in VALIDATOR_MUTATION_UNITS)
    build_time_results = tuple(evaluate_mutation_unit(unit, corpus, by_id) for unit in BUILD_TIME_MUTATION_UNITS)
    killed_ids = tuple(item.check_id for item in mutation_results if item.killed)
    survived = tuple(item.check_id for item in mutation_results if item.survived)
    mutation_invalid = tuple(
        item.check_id
        for item in (*mutation_results, *build_time_results)
        if item.mutation_invalid
    )
    build_killed = tuple(item.check_id for item in build_time_results if item.killed)
    build_redundant = tuple(item.check_id for item in build_time_results if item.survived_redundant)
    inline_passing = _differential_passing(INLINE_DIFFERENTIAL_PAIRS, corpus, by_id)
    build_inline_passing = _differential_passing(BUILD_TIME_INLINE_PAIRS, corpus, by_id)
    coverage_gaps = assert_manifest_coverage()
    buckets = frozen_code_buckets()
    manifest_covered = len(buckets) - len(coverage_gaps)
    counts: Counter[str] = Counter()
    for case in corpus:
        counts.update(case.categories)
    category_counts = tuple(sorted(counts.items()))
    known_rows = tuple(by_id[case.case_id] for case in known_cases)
    matched = sum(1 for item in scored if item.matched)
    return EvaluationReport(
        evaluations=evaluations,
        total_cases=len(corpus),
        expect_reject_cases=len(reject_cases),
        expect_pass_cases=len(pass_cases),
        known_limitation_cases=len(known_cases),
        matched_expectations=matched,
        unexpected_outcomes=len(unexpected_all),
        false_accept_style_mismatches=false_accept,
        false_reject_style_mismatches=false_reject,
        frozen_rejection_code_count=len(FROZEN_REJECTION_CODES),
        kill_matrix_killed=len(killed_ids),
        kill_matrix_total=len(VALIDATOR_MUTATION_UNITS),
        survived=survived,
        mutation_invalid=mutation_invalid,
        build_time_killed=len(build_killed),
        build_time_total=len(BUILD_TIME_MUTATION_UNITS),
        build_time_survived_redundant=build_redundant,
        inline_differential_passing=inline_passing,
        inline_differential_total=len(INLINE_DIFFERENTIAL_PAIRS),
        build_time_inline_passing=build_inline_passing,
        build_time_inline_total=len(BUILD_TIME_INLINE_PAIRS),
        expectation_sensitivity_covered=sensitivity_covered,
        expectation_sensitivity_total=len(FROZEN_REJECTION_CODES),
        expectation_sensitivity_misses=sensitivity_misses,
        manifest_coverage_covered=manifest_covered,
        manifest_coverage_total=len(FROZEN_REJECTION_CODES),
        category_counts=category_counts,
        unexpected_cases=unexpected_all,
        known_limitation_rows=known_rows,
        mutation_results=mutation_results,
        build_time_results=build_time_results,
    )


def _or_none(items: Sequence[str]) -> str:
    return ",".join(items) if items else "NONE"


def format_report(report: EvaluationReport) -> str:
    lines = [
        "TOTAL_CASES=" + str(report.total_cases),
        "EXPECT_REJECT_CASES=" + str(report.expect_reject_cases),
        "EXPECT_PASS_CASES=" + str(report.expect_pass_cases),
        "KNOWN_LIMITATION_CASES=" + str(report.known_limitation_cases),
        "MATCHED_EXPECTATIONS=" + str(report.matched_expectations),
        "UNEXPECTED_OUTCOMES=" + str(report.unexpected_outcomes),
        "FALSE_ACCEPT_STYLE_MISMATCHES=" + str(report.false_accept_style_mismatches),
        "FALSE_REJECT_STYLE_MISMATCHES=" + str(report.false_reject_style_mismatches),
        "FROZEN_REJECTION_CODE_COUNT=" + str(report.frozen_rejection_code_count),
        "KILL_MATRIX=" + str(report.kill_matrix_killed) + "/" + str(report.kill_matrix_total),
        "SURVIVED=" + _or_none(report.survived),
        "MUTATION_INVALID=" + _or_none(report.mutation_invalid),
        "BUILD_TIME_UNITS=" + str(report.build_time_killed) + "/" + str(report.build_time_total),
        "BUILD_TIME_SURVIVED_REDUNDANT=" + _or_none(report.build_time_survived_redundant),
        "INLINE_DIFFERENTIAL="
        + str(report.inline_differential_passing)
        + "/"
        + str(report.inline_differential_total),
        "BUILD_TIME_INLINE_DIFFERENTIAL="
        + str(report.build_time_inline_passing)
        + "/"
        + str(report.build_time_inline_total),
        "EXPECTATION_SENSITIVITY="
        + str(report.expectation_sensitivity_covered)
        + "/"
        + str(report.expectation_sensitivity_total),
        "MANIFEST_COVERAGE=" + str(report.manifest_coverage_covered) + "/" + str(report.manifest_coverage_total),
        "CATEGORY_COUNTS",
    ]
    for name, count in report.category_counts:
        lines.append(name + "=" + str(count))
    lines.append("UNEXPECTED_CASES")
    if not report.unexpected_cases:
        lines.append("NONE")
    else:
        for item in report.unexpected_cases:
            lines.extend(
                [
                    "case_id=" + item.case_id,
                    "expected_outcome=" + item.expected_outcome,
                    "actual_status=" + item.actual_status,
                    "actual_rejection_codes=" + ",".join(item.actual_rejection_codes),
                    "required_codes=" + ",".join(item.required_codes),
                    "forbidden_codes=" + ",".join(item.forbidden_codes),
                ]
            )
    lines.append("KNOWN_LIMITATION_CASES")
    for item in report.known_limitation_rows:
        lines.extend(
            [
                "case_id=" + item.case_id,
                "actual_status=" + item.actual_status,
                "actual_rejection_codes=" + ",".join(item.actual_rejection_codes),
            ]
        )
    lines.append(
        str(report.unexpected_outcomes)
        + " unexpected outcomes across "
        + str(report.total_cases)
        + " hand-authored synthetic cases covering the listed categories; "
        + str(report.known_limitation_cases)
        + " documented known limitations; kill matrix "
        + str(report.kill_matrix_killed)
        + "/"
        + str(report.kill_matrix_total)
        + " checks killed."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run_evaluation()
    print(format_report(report), end="")


if __name__ == "__main__":
    main()
