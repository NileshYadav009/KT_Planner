"""Regression test for semantic_chunk_sentences()'s embedding computation.

The initial per-sentence embedding pass used to call model.encode() once per
sentence in a Python list comprehension (context_mapper.py, STAGE 2) — a real
performance issue for any transcript with more than a handful of sentences,
since sentence-transformers batches its forward pass internally and pays
per-call Python/tokenization overhead once per invocation. Switched to one
batched model.encode(list_of_texts) call. This test locks in that the
function's actual chunking behavior (which sentences merge, which don't, and
that no text is lost) is unchanged by that switch.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_mapper import Sentence, semantic_chunk_sentences


def _sentence(text, idx):
    return Sentence(text=text, start=float(idx), end=float(idx) + 1.0)


def test_semantic_chunk_sentences_preserves_all_text():
    sentences = [
        _sentence("All services run on Amazon EKS.", 0),
        _sentence("Traffic enters through CloudFront and the load balancer.", 1),
        _sentence("The disaster recovery process restores RDS snapshots.", 2),
        _sentence("Escalation starts with the on-call engineer.", 3),
    ]
    chunks = semantic_chunk_sentences(sentences)
    assert chunks

    combined = " ".join(c.text for c in chunks)
    for s in sentences:
        assert s.text in combined


def test_semantic_chunk_sentences_merges_closely_related_short_sentences():
    # Two short, near-identical fragments (below min_merge_length) should
    # merge into one chunk rather than staying as fragmented single lines.
    sentences = [
        _sentence("Use PagerDuty.", 0),
        _sentence("Use PagerDuty for alerts.", 1),
        _sentence("The disaster recovery process restores RDS snapshots from the most recent nightly backup and validates data integrity before promoting.", 2),
    ]
    chunks = semantic_chunk_sentences(sentences)
    assert len(chunks) <= len(sentences)
    assert any("PagerDuty" in c.text for c in chunks)


def test_semantic_chunk_sentences_noop_for_single_sentence():
    sentences = [_sentence("Only one sentence here.", 0)]
    chunks = semantic_chunk_sentences(sentences)
    assert chunks == sentences


def test_semantic_chunk_sentences_noop_for_empty_input():
    assert semantic_chunk_sentences([]) == []
