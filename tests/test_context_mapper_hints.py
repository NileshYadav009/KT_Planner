"""Tests for context_mapper.py's schema-hint keyword matching —
_compile_hint_patterns() — in isolation, with no model loading required.

Regression guard for a real classification miss: the schema hint
"replacement can deploy safely" (handover_completion) never matched the
transcript sentence "The replacements can deploy safely." because the old
matcher did exact substring/`\\b<word>\\b` matching with no tolerance for
the most ordinary English variation (a trailing "s"). That silently zeroed
out the keyword boost for an explicit, clearly-relevant sentence, and the
same gap applies to every hint in every section's schema, not just this one.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_mapper import _compile_hint_patterns, is_kt_session_meta_commentary


def _matches(patterns, text):
    text_lower = text.lower()
    return any(pattern.search(text_lower) for pattern, _ in patterns)


def test_multi_word_hint_tolerates_plural_mismatch():
    patterns = _compile_hint_patterns(["replacement can deploy safely"])
    assert _matches(patterns, "The replacements can deploy safely.")
    assert _matches(patterns, "The replacement can deploy safely.")


def test_single_word_hint_tolerates_plural_mismatch():
    patterns = _compile_hint_patterns(["secret"])
    assert _matches(patterns, "Verify the secrets before deploying.")
    assert _matches(patterns, "Verify the secret before deploying.")


def test_hint_pattern_does_not_match_unrelated_text():
    patterns = _compile_hint_patterns(["replacement can deploy safely"])
    assert not _matches(patterns, "The rollback took fifteen minutes.")


def test_multi_word_hint_requires_words_in_order_and_adjacent():
    # Not a full paraphrase matcher — still requires the hint's words to
    # appear together in order, just tolerant of trailing pluralization.
    patterns = _compile_hint_patterns(["replacement can deploy safely"])
    assert not _matches(patterns, "Safely, the replacement can deploy.")


def test_empty_and_blank_hints_are_skipped():
    patterns = _compile_hint_patterns(["", "   ", "danger zone"])
    assert len(patterns) == 1
    assert _matches(patterns, "This is a danger zone.")


def test_boost_weight_higher_for_multi_word_hints():
    single = _compile_hint_patterns(["secret"])
    multi = _compile_hint_patterns(["replacement can deploy safely"])
    assert single[0][1] == 0.05
    assert multi[0][1] == 0.08


# Regression guard: these exact sentences (from real user transcripts)
# ended up polluting Architecture Reference, Disaster Recovery, and System
# Overview's Customer Reach field with tool self-promotion / scene-setting
# filler instead of real content, because nothing else claimed them.
META_COMMENTARY_EXAMPLES = [
    "Today this KT is about DevOps.",
    "This is a small overview of the planned work, and it serves as a small sample KT using the Continuum application as the first KT planner.",
    "Continuum is a good application.",
    "One last thing, continuum is a good application.",
    "I think this is enough.",
    "And let me know if you can help me improve it.",
    "Please feel free to reach out to me and test the application as much as possible.",
    "Let me start with the system overview and the system in five lines.",
    # Found live (job 8185143D): a generic transition sentence with no
    # named entity of its own was winning Environments' Production row by
    # default, since nothing else disqualified it.
    "Second, the third part we will be discussing about is environments and technologies.",
    "Let's talk about environments.",
    "We will be talking about business purpose and criticality.",
    "There would be no gaps.",
    "There will be no gaps.",
]

# Sentences that OPEN with transition-sounding phrasing but continue into a
# real, comma-separated fact in the SAME sentence — must survive, since
# these patterns are end-anchored specifically so a longer sentence isn't
# silently truncated along with its meta-sounding prefix.
TRANSITION_PREFIX_WITH_REAL_CONTENT_EXAMPLES = [
    "Now, come into the business purpose, the problem this system solves is reliable scalable order processing across multiple sales channel.",
    "Let's talk about environments, which closely mirror production configs.",
]


def test_transition_sentences_with_real_trailing_content_are_not_flagged():
    for text in TRANSITION_PREFIX_WITH_REAL_CONTENT_EXAMPLES:
        assert not is_kt_session_meta_commentary(text), f"should NOT be flagged (has real content): {text!r}"

REAL_CONTENT_EXAMPLES = [
    "All services run on Amazon EKS.",
    "The staging environment closely mirrors production.",
    "Never modify Terraform state files manually.",
    "The replacements can deploy safely.",
    "Devops own staging.",
    "We use Terraform for infrastructure provisioning.",
]


def test_meta_commentary_sentences_are_flagged():
    for text in META_COMMENTARY_EXAMPLES:
        assert is_kt_session_meta_commentary(text), f"should be flagged as meta: {text!r}"


def test_real_content_sentences_are_not_flagged():
    for text in REAL_CONTENT_EXAMPLES:
        assert not is_kt_session_meta_commentary(text), f"should NOT be flagged as meta: {text!r}"
