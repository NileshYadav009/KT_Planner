"""
quality_score.py
=================
Aggregates a KT run's per-section coverage/confidence/risk (already computed
by context_mapper.py's SectionCoverage) plus validation.py's structural
warnings into a single document-level quality score. Nothing here recomputes
coverage — it only weighs numbers that already exist.

Deliberately separate from `kt.overall_coverage_percent` (context_mapper.py),
which is a raw coverage percentage currently reused by pipeline.py as the
job's processing "progress" value — that number treats every section equally
and knows nothing about validation issues. A document can have 100% section
coverage and still have real structural problems (a mismatched field id, an
out-of-range confidence value); this score is meant to reflect that.
"""

from typing import Any, Dict, List

# How much each status is "worth" toward coverage_score, separately for
# required vs. optional sections — a missing required section is a much
# bigger problem than a missing optional one, so the weights aren't symmetric.
_REQUIRED_STATUS_WEIGHTS = {"covered": 1.0, "weak": 0.5, "missing": 0.0}
_OPTIONAL_STATUS_WEIGHTS = {"covered": 1.0, "weak": 0.7, "missing": 0.3}

# Deduction per validation warning, capped so a document with many minor
# warnings doesn't score negative — this is a penalty, not the whole score.
_VALIDATION_PENALTY_PER_WARNING = 0.03
_VALIDATION_PENALTY_CAP = 0.30

_DIMENSION_WEIGHTS = {"coverage": 0.5, "confidence": 0.25, "risk": 0.25}

_GRADE_BANDS = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
]


def _grade_for(score: float) -> str:
    for threshold, grade in _GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def compute_quality_score(
    coverage: Dict[str, Any],
    dynamic_schema: List[Dict[str, Any]],
    validation_warnings: List[str],
) -> Dict[str, Any]:
    """Compute a 0-100 document-level quality score with a letter grade and
    a per-dimension breakdown. `coverage` is the pipeline's coverage dict
    (section_id -> {"status", "confidence", "risk", "required", ...});
    `dynamic_schema` is used to look up each section's `required` flag when
    coverage doesn't already carry it.
    """
    required_by_id = {s.get("id"): bool(s.get("required")) for s in (dynamic_schema or [])}

    coverage_terms: List[float] = []
    confidence_values: List[float] = []
    risk_values: List[float] = []

    for section_id, info in (coverage or {}).items():
        status = info.get("status", "missing")
        is_required = info.get("required")
        if is_required is None:
            is_required = required_by_id.get(section_id, False)

        weights = _REQUIRED_STATUS_WEIGHTS if is_required else _OPTIONAL_STATUS_WEIGHTS
        coverage_terms.append(weights.get(status, 0.0))

        if status in ("covered", "weak"):
            confidence_values.append(float(info.get("confidence", 0.0) or 0.0))
        risk_values.append(float(info.get("risk", 0.0) or 0.0))

    coverage_score = sum(coverage_terms) / len(coverage_terms) if coverage_terms else 0.0
    confidence_score = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    risk_score = 1.0 - (sum(risk_values) / len(risk_values) if risk_values else 0.0)

    validation_penalty = min(
        len(validation_warnings or []) * _VALIDATION_PENALTY_PER_WARNING,
        _VALIDATION_PENALTY_CAP,
    )

    weighted = (
        coverage_score * _DIMENSION_WEIGHTS["coverage"]
        + confidence_score * _DIMENSION_WEIGHTS["confidence"]
        + risk_score * _DIMENSION_WEIGHTS["risk"]
    )
    overall = max(0.0, weighted - validation_penalty)
    overall_100 = round(overall * 100, 1)

    return {
        "overall_score": overall_100,
        "grade": _grade_for(overall_100),
        "dimensions": {
            "coverage": round(coverage_score, 3),
            "confidence": round(confidence_score, 3),
            "risk": round(risk_score, 3),
        },
        "validation_warning_count": len(validation_warnings or []),
        "validation_penalty": round(validation_penalty, 3),
    }
