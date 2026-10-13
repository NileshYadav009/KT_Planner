"""Tests for the lossless-polish guard in ai.py.

The polish pass rewrites a section wholesale and its output then REPLACES
the raw fragments as that section's content, so anything the model omits is
gone from the document with nothing reporting a loss. Observed live: a
Danger Zones polish returned only one of two prohibitions and "Production
Kubernetes configuration must not be changed manually." vanished from the KT.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai import _fragments_missing_from

FRAGMENTS = [
    "Production Kubernetes configuration must not be changed manually.",
    "Do not manually modify Bicep-managed infrastructure without coordinating with platform engineering.",
]


def test_detects_a_fragment_the_polish_pass_dropped():
    lossy = "- Do not manually modify Bicep-managed infrastructure without coordinating with platform engineering."
    missing = _fragments_missing_from(FRAGMENTS, lossy)
    assert missing == [FRAGMENTS[0]]


def test_accepts_a_polish_that_kept_everything():
    faithful = (
        "- Production Kubernetes configuration must not be changed manually.\n"
        "- Do not manually modify Bicep-managed infrastructure without coordinating with platform engineering."
    )
    assert _fragments_missing_from(FRAGMENTS, faithful) == []


def test_tolerates_legitimate_rewording_and_inflection():
    # Polishing is SUPPOSED to rephrase; only an actual omission may be
    # flagged. Words are compared on a short stem so "changed"/"change" and
    # dropped modals ("must not" -> "never") don't read as a loss.
    reworded = (
        "- Never change production Kubernetes configuration manually.\n"
        "- Never modify Bicep-managed infrastructure without coordinating with the platform engineering team."
    )
    assert _fragments_missing_from(FRAGMENTS, reworded) == []


def test_ignores_fragments_too_short_to_judge():
    assert _fragments_missing_from(["Okay so."], "completely unrelated text") == []


def test_handles_empty_inputs():
    assert _fragments_missing_from([], "anything") == []
    assert _fragments_missing_from(FRAGMENTS, "") == FRAGMENTS


def test_local_cleanup_keeps_the_space_before_a_dotted_name():
    """'The .NET services' printed as 'The.NET services'."""
    from ai import _local_cleanup

    assert _local_cleanup("our AKS cluster . The .NET services use SQL , Redis .") == \
        "Our AKS cluster. The .NET services use SQL, Redis."


def test_danger_zones_are_never_reworded(monkeypatch):
    """A rewrite turned 'because they are the only copy we can replay from' into
    'as they are the only copy available for replay': safety warnings keep the
    speaker's words, and no LLM call is spent on them."""
    import ai

    class Rewriter:
        model = "stub"

        def __init__(self):
            self.prompts = []

        def generate(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return "Reworded text."

    stub = Rewriter()
    monkeypatch.setattr(ai, "get_llm_provider", lambda: stub)
    warning = "Never delete raw events during an active investigation, because they are the only copy we can replay from."
    out = ai.polish_coverage_sections({"danger_zones": {"title": "Danger Zones", "fragments": [warning]}})
    assert out["danger_zones"] == [warning]
    assert stub.prompts == []
