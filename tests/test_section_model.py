"""section_model.npz: the classifier's learned section model
(scripts/build_section_model.py, audit 9abd). No embedding model loads here:
the scoring tests use small stand-in embeddings."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_section_model  # noqa: E402
import context_mapper as cm  # noqa: E402
from golden_eval import load_goldens  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402


def _about():
    with np.load(cm._SECTION_MODEL_PATH, allow_pickle=False) as data:
        return json.loads(str(data["about"]))


def test_the_model_loads_and_knows_only_schema_sections():
    model = cm.load_section_model()
    assert model is not None, "run python scripts/build_section_model.py"
    sections, coef, intercept = model
    assert set(sections) <= {s["id"] for s in SCHEMA}
    # The fact's embedding and its context's (bge-large: 1024 each).
    assert coef.shape == (len(sections), 2048)
    assert intercept.shape == (len(sections),)


def test_no_holdout_blind_or_customer_transcript_is_trained_on():
    """Holdouts and blind KTs must stay unseen to measure accuracy on new
    KTs, and a customer's KT never ships inside the product."""
    trained = set(_about()["trained_on"])
    assert trained
    for golden in load_goldens():
        if not build_section_model.usable(golden):
            assert golden["id"] not in trained, golden["id"]
    blind = {name[:-5] for folder in ("blind", "blind3", "blind4")
             for name in os.listdir(os.path.join(ROOT, "evaluation", folder)) if name.endswith(".json")}
    assert not trained & blind


def test_a_missing_or_unreadable_file_leaves_the_scores_as_they_were(tmp_path):
    assert cm.load_section_model(str(tmp_path / "none.npz")) is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a model")
    assert cm.load_section_model(str(bad)) is None


def _stand_in_model():
    """Section "a" reads the fact's first dimension, "b" the context's second."""
    coef = np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]], dtype="float32")
    return ["a", "b"], coef, np.zeros(2, dtype="float32")


def test_the_probability_reads_the_fact_and_its_context():
    clf = cm.ContextClassifier.__new__(cm.ContextClassifier)
    clf._section_model = _stand_in_model()
    p = clf._learned_probabilities(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert p["a"] > p["b"] and sum(p.values()) == pytest.approx(1.0)
    p = clf._learned_probabilities(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert p["b"] > p["a"]
    # No context: the fact stands in for it.
    p = clf._learned_probabilities(np.array([0.0, 1.0]), None)
    assert p["b"] > p["a"]
    # Embeddings of another size than the model's: no probability, no error.
    assert clf._learned_probabilities(np.array([1.0, 0.0, 0.0]), None) == {}


def test_weight_zero_or_no_model_turns_it_off(monkeypatch):
    clf = cm.ContextClassifier.__new__(cm.ContextClassifier)
    assert clf._learned_probabilities(np.array([1.0, 0.0]), None) == {}
    clf._section_model = _stand_in_model()
    monkeypatch.setattr(cm, "SECTION_MODEL_WEIGHT", 0.0)
    assert clf._learned_probabilities(np.array([1.0, 0.0]), None) == {}


class _Util:
    @staticmethod
    def cos_sim(a, b):
        a, b = np.atleast_2d(np.asarray(a, dtype=float)), np.atleast_2d(np.asarray(b, dtype=float))
        return (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :])


class _NoEntities:
    def get_context_entities(self, text):
        return {}


def test_the_learned_probability_decides_a_near_tie_and_says_so(monkeypatch):
    monkeypatch.setattr(cm, "util", _Util)
    monkeypatch.setattr(cm, "RECOVERY_WITHOUT_EVIDENCE_PENALTY", 0.0)
    clf = cm.ContextClassifier.__new__(cm.ContextClassifier)
    clf.section_embeddings = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    clf.section_metadata = {sid: {"title": sid} for sid in clf.section_embeddings}
    clf._section_hint_patterns = {}
    clf._example_embeddings = None
    clf._example_sections = []
    clf.entity_extractor = _NoEntities()
    clf._section_model = _stand_in_model()
    sentence = cm.Sentence(text="It settles down after a while.", start=0.0, end=0.0)
    # The embeddings lean slightly to "b"; the model reads "a" in the fact.
    candidates, _, _ = clf._score_sentence_candidates(
        sentence, sent_embedding=np.array([0.70, 0.72]), context_embedding=np.array([0.70, 0.72]))
    assert candidates[0].section_id == "a"
    assert "Learned=+" in candidates[0].reason
    monkeypatch.setattr(cm, "SECTION_MODEL_WEIGHT", 0.0)
    candidates, _, _ = clf._score_sentence_candidates(
        sentence, sent_embedding=np.array([0.70, 0.72]), context_embedding=np.array([0.70, 0.72]))
    assert candidates[0].section_id == "b"
