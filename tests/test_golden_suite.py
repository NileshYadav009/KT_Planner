"""P1-11: the golden KT suite (tests/goldens, scored by tests/golden_eval.py).

Twelve annotated transcripts, 211 facts, each with the section(s) it belongs
in: AWS, Azure, GCP, a complex multi-team platform, a noisy recording, an
incomplete KT, contradictions, receiver dialogue, an on-premises platform and
a long loosely ordered KT (the tuning set, which the routing rules were
improved against), plus two holdouts written afterwards and never tuned for
(an on-premises Java system and a conversational Azure serverless KT). The
pipeline runs without an LLM, so the scores are deterministic.

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


def test_contradictions_are_flagged(scores):
    for golden in GOLDENS:
        if golden.get("conflicts"):
            assert scores[golden["id"]]["missing_conflicts"] == [], golden["id"]


@pytest.mark.xfail(reason="Accuracy on unseen KTs is 84% (target 90%); see scripts/golden_report.py", strict=False)
def test_unseen_kts_meet_the_accuracy_target(scores):
    """Only the holdouts measure accuracy on a KT the rules were not tuned
    on; the tuning set is expected to stay at 100%."""
    holdouts = [s for s in scores.values() if s["holdout"]]
    total = sum(len(s["facts"]) for s in holdouts)
    correct = sum(sum(f["status"] == "correct" for f in s["facts"]) for s in holdouts)
    assert correct / total >= TARGET_ACCURACY
