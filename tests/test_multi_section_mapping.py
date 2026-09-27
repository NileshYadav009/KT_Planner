"""Controlled multi-section assignment (Principle #3), behind
ENABLE_MULTI_SECTION_MAPPING and OFF by default.

The classifier produces a ranked top-k for every sentence, so adding
secondaries unconditionally would copy most of a transcript into several
sections at once -- which is exactly why single-section placement was
deliberate (see ClassifiedSentence.__post_init__'s original comment). These
tests pin the two things that make re-enabling it safe:

  1. With the flag OFF, behaviour is byte-for-byte what it was.
  2. With it ON, an extra section needs a stated justification, unrelated
     sections are still rejected, and every extra placement stays traceable.

See REPOSITORY_AUDIT.md §9tt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import context_mapper as cm
from context_mapper import (
    Classification,
    ClassifiedSentence,
    Sentence,
    justify_extra_sections,
)


def _classification(section_id, confidence, reason="embedding"):
    return Classification(
        section_id=section_id,
        section_title=section_id.replace("_", " ").title(),
        confidence=confidence,
        reason=reason,
        similarity_score=confidence,
    )


def _classified(text, primary, secondary):
    return ClassifiedSentence(
        sentence=Sentence(text=text, start=0.0, end=1.0),
        primary_classification=primary,
        secondary_classifications=secondary,
    )


# ---------------------------------------------------------------------------
# Flag OFF: existing behaviour must be unchanged.
# ---------------------------------------------------------------------------

def test_flag_is_off_by_default():
    assert cm.ENABLE_MULTI_SECTION_MAPPING is False


def test_with_flag_off_only_the_primary_section_is_assigned(monkeypatch):
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", False)

    cs = _classified(
        "For an order alert, first check Grafana for request rate and pod health.",
        _classification("monitoring_observability", 0.70),
        # A secondary that WOULD qualify if the flag were on.
        [_classification("day1_survival_checklist", 0.66)],
    )

    assert cs.multi_section_assignments == ["monitoring_observability"]
    assert cs.multi_section_evidence == []


# ---------------------------------------------------------------------------
# Flag ON: legitimate multi-section mapping.
# ---------------------------------------------------------------------------

def test_a_close_strong_secondary_is_added_with_a_stated_reason(monkeypatch):
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "Service Bus backlog can temporarily increase during large deployments.",
        _classification("common_failures", 0.72),
        [_classification("deployment_and_rollback", 0.68)],
    )

    assert cs.multi_section_assignments == ["common_failures", "deployment_and_rollback"]
    assert len(cs.multi_section_evidence) == 1
    record = cs.multi_section_evidence[0]
    assert record["section_id"] == "deployment_and_rollback"
    assert record["confidence"] == 0.68
    assert record["primary_confidence"] == 0.72
    assert record["reason"]  # a justification is always recorded


def test_a_rule_backed_secondary_counts_as_evidence_on_its_own(monkeypatch):
    # A section's own deterministic rule firing is stronger evidence than an
    # embedding score, so it qualifies even below the relative bar.
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "Do not flush the Redis cluster as an emergency action.",
        _classification("common_failures", 0.80),
        [_classification("danger_zones", 0.50, reason="Rule match: do not flush")],
    )

    assert "danger_zones" in cs.multi_section_assignments
    assert "section rule fired" in cs.multi_section_evidence[0]["reason"]


# ---------------------------------------------------------------------------
# Flag ON: unrelated sections must still be rejected.
# The four scenarios the change was scoped against.
# ---------------------------------------------------------------------------

def test_a_clear_single_home_gains_no_extra_section(monkeypatch):
    # "Do not manually modify production Kubernetes configuration" ->
    # Danger Zones only. A distant runner-up is not "genuinely relevant".
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "Production Kubernetes configuration must not be changed manually.",
        _classification("danger_zones", 0.88),
        [_classification("deployment_and_rollback", 0.31)],
    )

    assert cs.multi_section_assignments == ["danger_zones"]
    assert cs.multi_section_evidence == []


def test_rollback_does_not_leak_into_unrelated_sections(monkeypatch):
    # "Rollback uses Flux to synchronize the previous version" ->
    # Deployment & Rollback, nothing else.
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "Rollback is performed by reverting the Git deployment configuration "
        "and allowing Flux to synchronize the previous version.",
        _classification("deployment_and_rollback", 0.84),
        [
            _classification("disaster_recovery", 0.38),
            _classification("cost_optimization", 0.12),
        ],
    )

    assert cs.multi_section_assignments == ["deployment_and_rollback"]


def test_a_secondary_close_to_primary_but_weak_in_absolute_terms_is_rejected(monkeypatch):
    # Both scores low: the runner-up is "close" only because neither is a
    # real match. Relative closeness alone must not justify a placement.
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "We reviewed a few odds and ends at the end of the call.",
        _classification("system_overview", 0.22),
        [_classification("known_bad_days", 0.21)],
    )

    assert cs.multi_section_assignments == ["system_overview"]


def test_a_strong_secondary_far_below_the_primary_is_rejected(monkeypatch):
    # Clears the absolute floor but the primary is a much better home.
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "Azure SQL is the primary database.",
        _classification("system_overview", 0.90),
        [_classification("disaster_recovery", 0.50)],
    )

    assert cs.multi_section_assignments == ["system_overview"]


# ---------------------------------------------------------------------------
# Invariants that hold regardless of the flag.
# ---------------------------------------------------------------------------

def test_the_primary_section_is_never_displaced(monkeypatch):
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "Check Kafka consumer lag and the order-state processing metrics.",
        _classification("monitoring_observability", 0.70),
        [_classification("common_failures", 0.69)],
    )

    # Additive only: the primary stays, and stays first.
    assert cs.multi_section_assignments[0] == "monitoring_observability"


def test_extra_sections_are_capped(monkeypatch):
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)
    monkeypatch.setattr(cm, "MULTI_SECTION_MAX_EXTRA", 1)

    cs = _classified(
        "A sentence that straddles several sections.",
        _classification("common_failures", 0.80),
        [
            _classification("monitoring_observability", 0.78),
            _classification("danger_zones", 0.77),
            _classification("tribal_knowledge", 0.76),
        ],
    )

    assert len(cs.multi_section_assignments) == 2  # primary + at most 1 extra


def test_the_same_section_is_never_added_twice(monkeypatch):
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)
    monkeypatch.setattr(cm, "MULTI_SECTION_MAX_EXTRA", 3)

    cs = _classified(
        "Duplicate candidates for one section.",
        _classification("common_failures", 0.80),
        [
            _classification("danger_zones", 0.78),
            _classification("danger_zones", 0.77),
            _classification("common_failures", 0.76),  # same as primary
        ],
    )

    assert cs.multi_section_assignments == ["common_failures", "danger_zones"]


def test_justification_is_a_pure_function_and_needs_no_llm():
    # No provider, no model, no network -- the gate reads only what the
    # classifier already produced.
    primary = _classification("monitoring_observability", 0.70)
    secondary = [_classification("day1_survival_checklist", 0.64)]

    first = justify_extra_sections(primary, secondary)
    second = justify_extra_sections(primary, secondary)

    assert first == second
    assert first and first[0]["section_id"] == "day1_survival_checklist"


def test_no_primary_means_no_assignments():
    cs = _classified("Unclassifiable filler.", None, [_classification("system_overview", 0.9)])
    assert cs.multi_section_assignments == []
    assert cs.multi_section_evidence == []


def test_the_floor_is_the_real_similarity_threshold_not_the_retention_bar():
    # Regression guard for two measured mis-calibrations (see §9tt):
    #   0.45 -> unreachable, the gate became dead code;
    #   0.18 -> the candidate-RETENTION bar, which fires on noise.
    # A census of 140 real sentences found secondaries at 0.13-0.18 scoring
    # 0.98-1.00 of their primary (0.143 vs 0.144). At that spread the
    # classifier cannot tell the sections apart, so promoting them would
    # duplicate the least-confident sentences.
    assert cm.MULTI_SECTION_MIN_CONFIDENCE >= 0.30


def test_two_indistinguishable_low_scores_are_never_promoted(monkeypatch):
    # The exact shape observed in the census.
    monkeypatch.setattr(cm, "ENABLE_MULTI_SECTION_MAPPING", True)

    cs = _classified(
        "The first is to review and document the external market-data certificate renewal.",
        _classification("day1_survival_checklist", 0.144),
        [_classification("handover_completion", 0.143)],
    )

    assert cs.multi_section_assignments == ["day1_survival_checklist"]
