"""Tests for devops_transcription.py's fuzzy term-correction step.

Regression guard for a real data-corruption bug found via a 4-transcript
adversarial test pass: apply_fuzzy_term_corrections() tokenized on
`\\b[\\w/]+\\b`, which doesn't include apostrophes, so any contraction
("it's", "that's", "there's") split into two tokens ("it" + "s"). The
resulting bare "s" token then formed n-grams like "s the" that fuzzy-
matched the known glossary term "s three" (used to correct mis-heard
"S3") with a very high jaro-winkler score, silently corrupting ordinary
sentences like "It's the old claims processing system" into "It's three
old claims processing system". This wasn't a rare edge case — "it's
the"/"that's the"/"there's a" are among the most common contraction
patterns in spoken English, so this could corrupt real transcript content
on a routine basis.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from devops_transcription import clean_transcript, apply_devops_corrections


def test_apostrophe_contraction_not_corrupted_by_s3_fuzzy_match():
    text = "It's the old claims processing system, it's been around forever."
    result = clean_transcript(text)
    assert "three" not in result.lower()
    assert "It's the old claims processing system" in result


def test_various_common_contractions_survive_fuzzy_correction():
    for contraction_sentence in [
        "It's the primary database for the platform.",
        "That's the escalation path we use.",
        "There's the danger zone everyone talks about.",
        "What's the rollback procedure look like?",
    ]:
        corrected, _ = apply_devops_corrections(contraction_sentence)
        assert "three" not in corrected.lower(), f"corrupted: {contraction_sentence!r} -> {corrected!r}"


def test_genuine_s3_mention_without_apostrophe_is_unaffected():
    # This fix only removes the spurious apostrophe-splitting trigger — a
    # literal "s three" (no apostrophe involved at all) tokenizes and is
    # evaluated exactly as before.
    text = "We store all our backups in s three for durability."
    corrected, _ = apply_devops_corrections(text)
    assert "s three" in corrected.lower() or "s3" in corrected.lower()
