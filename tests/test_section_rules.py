"""Tests for section_rules.py's deterministic classification patterns."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from section_rules import match_section_rules


def test_explicit_danger_zones_phrase_is_classified_as_danger_zones():
    # Regression test for a real bug found on a live Azure Banking
    # transcript: a sentence that explicitly announces itself as naming
    # the danger zones ("The main danger zones are SQL service tier
    # changes, Service bus retention and Dead-letter configuration, And
    # manual Kubernetes changes outside GitOps.") matched none of the
    # section's existing patterns (which only covered "dangerous area",
    # "do not touch", "terraform state files manually", "autoscaler
    # configuration") because none of them recognize the literal phrase
    # "danger zone(s)" -- the section's own name. The sentence fell
    # through to a different section, silently dropping 2 of its 3 named
    # items from the rendered Danger Zones section.
    text = (
        "The main danger zones are SQL service tier changes, Service bus "
        "retention and Dead-letter configuration, And manual Kubernetes "
        "changes outside GitOps."
    )
    match = match_section_rules(text)
    assert match is not None
    assert match.section_id == "danger_zones"


def test_sensitive_area_phrase_is_classified_as_danger_zones():
    text = (
        "The production Kubernetes cluster autoscaler is another sensitive "
        "area, because incorrect changes can affect the platform."
    )
    match = match_section_rules(text)
    assert match is not None
    assert match.section_id == "danger_zones"
