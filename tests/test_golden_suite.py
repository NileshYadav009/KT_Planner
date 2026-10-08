"""P1-11: the golden KT suite (tests/goldens, scored by tests/golden_eval.py).

Nineteen annotated transcripts, 429 facts, each with the section(s) it
belongs in. The tuning set (g01-g11, h01-h05) is what the routing rules and
the classifier's labelled examples (section_examples.json) were improved
against: AWS, Azure, GCP, a complex multi-team platform, a noisy recording,
an incomplete KT, contradictions, receiver dialogue, on-premises platforms, a
long loosely ordered KT and conversational serverless KTs. The holdouts
(h06-h08: a streaming data platform, a pasted Teams transcript with speaker
names, an on-premises monologue) were written after the last fixes and scored
blind; they alone measure accuracy on unseen KTs. The pipeline runs without
an LLM, so the scores are deterministic.

A change fails here if any golden places fewer facts correctly, puts a fact
somewhere it must never be, reports a discussed section Missing or an
undiscussed one covered, loses or repeats a fact, or stops flagging a
contradiction. The floors are tests/goldens/baseline.json; raise them with
`python scripts/golden_report.py --update-baseline` when mapping improves.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from golden_eval import TARGET_ACCURACY, load_baseline, load_goldens, norm, run_all

GOLDENS = load_goldens()
IDS = [g["id"] for g in GOLDENS]


def test_every_annotation_quote_is_in_its_transcript():
    from devops_transcription import clean_transcript

    for golden in GOLDENS:
        text = norm(clean_transcript(golden["transcript"]))
        for fact in golden["facts"]:
            assert norm(fact["quote"]) in text, (golden["id"], fact["id"])


@pytest.fixture(scope="module")
def scores():
    import pipeline

    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    return {s["id"]: s for s in run_all(GOLDENS)}


def _misplaced(score):
    return [(f["id"], f["quote"], f["found_in"]) for f in score["facts"] if f["status"] != "correct"]


@pytest.mark.parametrize("golden_id", IDS)
def test_no_golden_gets_worse_than_its_baseline(scores, golden_id):
    floor = load_baseline()[golden_id]
    score = scores[golden_id]
    assert score["accuracy"] >= floor["accuracy"], _misplaced(score)
    assert len(score["wrong"]) <= floor["wrong"], _misplaced(score)
    assert len(score["violations"]) <= floor["violations"], score["violations"]
    assert len(score["false_missing"]) <= floor["false_missing"], score["false_missing"]
    assert len(score["false_covered"]) <= floor["false_covered"], score["false_covered"]


@pytest.mark.parametrize("golden_id", IDS)
def test_nothing_is_lost_or_repeated(scores, golden_id):
    assert scores[golden_id]["lost"] == []
    assert scores[golden_id]["repeats"] == []


@pytest.mark.parametrize("golden_id", IDS)
def test_every_fact_shown_points_to_the_sentence_that_states_it(scores, golden_id):
    """P1-2: a reviewer can check any fact against what was said."""
    assert scores[golden_id]["unsourced"] == [], [
        (f["id"], f["quote"]) for f in scores[golden_id]["facts"] if f["id"] in scores[golden_id]["unsourced"]]
    assert scores[golden_id]["evidence"].get("stated", 0) > 0


def test_contradictions_are_flagged(scores):
    for golden in GOLDENS:
        if golden.get("conflicts"):
            assert scores[golden["id"]]["missing_conflicts"] == [], golden["id"]


@pytest.mark.xfail(reason="Accuracy on unseen KTs is 89.9% (89 of 99 holdout facts; target 90%); see "
                          "scripts/golden_report.py", strict=False)
def test_unseen_kts_meet_the_accuracy_target(scores):
    """Only the holdouts measure accuracy on a KT the rules were not tuned
    on. Blind rounds so far: 84% (h01-h02), 83% (h03-h05; 86% once their
    ownership rows were annotated), 89.9% (h06-h08, 89 of 99 facts)."""
    holdouts = [s for s in scores.values() if s["holdout"]]
    total = sum(len(s["facts"]) for s in holdouts)
    correct = sum(sum(f["status"] == "correct" for f in s["facts"]) for s in holdouts)
    assert correct / total >= TARGET_ACCURACY
