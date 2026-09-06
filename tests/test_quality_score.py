"""Tests for quality_score.py — the document-level quality aggregate."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quality_score import compute_quality_score


def _cov(status, required, confidence=0.8, risk=0.1):
    return {"status": status, "required": required, "confidence": confidence, "risk": risk}


def test_all_required_sections_covered_scores_high():
    coverage = {
        "a": _cov("covered", True, confidence=0.9, risk=0.05),
        "b": _cov("covered", True, confidence=0.9, risk=0.05),
    }
    result = compute_quality_score(coverage, [], [])
    assert result["overall_score"] > 80
    assert result["grade"] in ("A", "B")


def test_missing_required_section_scores_much_worse_than_missing_optional():
    coverage_required_missing = {
        "a": _cov("missing", True),
        "b": _cov("covered", True),
    }
    coverage_optional_missing = {
        "a": _cov("missing", False),
        "b": _cov("covered", True),
    }
    required_missing_score = compute_quality_score(coverage_required_missing, [], [])["overall_score"]
    optional_missing_score = compute_quality_score(coverage_optional_missing, [], [])["overall_score"]
    assert required_missing_score < optional_missing_score


def test_validation_warnings_reduce_score():
    coverage = {"a": _cov("covered", True)}
    clean = compute_quality_score(coverage, [], [])["overall_score"]
    with_warnings = compute_quality_score(coverage, [], ["warning 1", "warning 2"])["overall_score"]
    assert with_warnings < clean


def test_validation_penalty_is_capped():
    coverage = {"a": _cov("covered", True)}
    many_warnings = [f"warning {i}" for i in range(100)]
    result = compute_quality_score(coverage, [], many_warnings)
    assert result["validation_penalty"] <= 0.30 + 1e-9
    assert result["overall_score"] >= 0.0


def test_empty_coverage_does_not_crash():
    # No sections at all is a degenerate case that shouldn't happen in
    # practice (every real job's coverage dict always has every schema
    # section, even if "missing") — the key requirement is just that it
    # doesn't crash (e.g. divide by zero) and produces a well-formed result.
    result = compute_quality_score({}, [], [])
    assert 0.0 <= result["overall_score"] <= 100.0
    assert result["grade"] in {"A", "B", "C", "D", "F"}


def test_falls_back_to_dynamic_schema_for_required_flag():
    # coverage entries here don't carry their own "required" key — must look
    # it up from dynamic_schema instead of defaulting everything to optional.
    coverage = {"a": {"status": "missing", "confidence": 0.0, "risk": 1.0}}
    schema = [{"id": "a", "required": True}]
    with_schema = compute_quality_score(coverage, schema, [])["overall_score"]
    without_schema = compute_quality_score(coverage, [], [])["overall_score"]
    assert with_schema <= without_schema


def test_grade_bands():
    assert compute_quality_score({"a": _cov("covered", True, confidence=1.0, risk=0.0)}, [], [])["grade"] == "A"
    assert compute_quality_score({"a": _cov("missing", True)}, [], [])["grade"] == "F"


def test_result_includes_dimension_breakdown():
    coverage = {"a": _cov("covered", True)}
    result = compute_quality_score(coverage, [], [])
    assert set(result["dimensions"].keys()) == {"coverage", "confidence", "risk"}
