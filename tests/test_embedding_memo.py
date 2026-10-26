"""Each text embedded once per model (embedding_memo.py, audit P1-5): field
population encoded a section's sentences again for every field, though
classification had just encoded them with the same model."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import embedding_memo  # noqa: E402


class Counting:
    """Encodes a text as [its length, its word count], counting the texts."""

    def __init__(self, scale=1.0):
        self.texts, self.scale = [], scale

    def encode(self, texts, convert_to_tensor=False, convert_to_numpy=True, normalize_embeddings=False):
        self.texts += list(texts)
        return np.array([[len(t) * self.scale, len(t.split())] for t in texts], dtype=np.float64)


def test_each_text_is_encoded_once():
    model = Counting()
    first = embedding_memo.encode(model, ["a b", "c", "a b"])
    again = embedding_memo.encode(model, ["c", "d e f"])
    assert model.texts == ["a b", "c", "d e f"]
    assert first.shape == (3, 2) and first.dtype.is_floating_point
    assert first[0].tolist() == [3.0, 2.0] and again[1].tolist() == [5.0, 3.0]
    assert embedding_memo.encode(model, "c", convert_to_tensor=False).tolist() == [1.0, 1.0]   # one text: 1-D


def test_models_and_normalisation_are_kept_apart():
    a, b = Counting(), Counting(scale=10.0)
    assert embedding_memo.encode(a, "same text")[0].item() == 9.0
    assert embedding_memo.encode(b, "same text")[0].item() == 90.0
    embedding_memo.encode(a, "same text", normalize_embeddings=False)
    assert a.texts == ["same text", "same text"]


def test_a_stand_in_that_is_not_an_encoder_is_asked_as_before():
    class StandIn:
        def encode(self, value, convert_to_tensor=False, normalize_embeddings=False):
            return value

    assert embedding_memo.encode(StandIn(), ["x", "y"]) == ["x", "y"]


def test_the_memo_keeps_at_most_max_texts(monkeypatch):
    monkeypatch.setattr(embedding_memo, "MAX_TEXTS", 3)
    embedding_memo.clear()
    model = Counting()
    embedding_memo.encode(model, ["1", "2", "3", "4"])
    embedding_memo.encode(model, ["1"])                       # the oldest was dropped
    assert model.texts == ["1", "2", "3", "4", "1"]
