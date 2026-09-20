"""Tests for ContextClassifier._verify_classification_with_llm() and
_maybe_verify_with_llm() (context_mapper.py) — the selective LLM fallback
used to break ties for genuinely borderline section classifications.

Constructed without a real ContextClassifier instance (which would load the
heavy BAAI/bge-large-en-v1.5 embedding model) — these methods only touch
self.llm_fallback_fn and self.similarity_threshold, so a lightweight
SimpleNamespace stands in as `self` for the unbound method call.
"""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_mapper import ContextClassifier, Classification


def _classifier(llm_fallback_fn=None, similarity_threshold=0.15):
    fake_self = SimpleNamespace(llm_fallback_fn=llm_fallback_fn, similarity_threshold=similarity_threshold)
    # _maybe_verify_with_llm calls self._verify_classification_with_llm(...)
    # internally — bind the real unbound method onto this fake `self` too,
    # so testing _maybe_verify_with_llm doesn't require a real (heavy)
    # ContextClassifier instance.
    fake_self._verify_classification_with_llm = ContextClassifier._verify_classification_with_llm.__get__(fake_self)
    return fake_self


def _candidate(section_id, confidence, title=None):
    return Classification(
        section_id=section_id,
        section_title=title or section_id,
        confidence=confidence,
        reason="test",
        similarity_score=confidence,
    )


def test_verify_classification_includes_context_when_it_adds_information():
    # The classifier's own embedding scoring already sees a ±2-sentence
    # window around the target sentence — verification (which only fires for
    # the hardest, already-ambiguous cases) must not see strictly less
    # context than that, or it's working with less information than what
    # produced the ambiguous candidates in the first place.
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "system_overview"

    fake_self = _classifier(llm_fallback_fn=fake_llm)
    candidates = [_candidate("system_overview", 0.5), _candidate("architecture_reference", 0.48)]

    result = ContextClassifier._verify_classification_with_llm(
        fake_self, "It happens during flash sales.", candidates,
        context_text="The most common production issue is database connection pool exhaustion. It happens during flash sales. This causes order failures.",
    )
    assert result == "system_overview"
    assert "Surrounding context" in captured["prompt"]
    assert "flash sales" in captured["prompt"]


def test_verify_classification_omits_context_block_when_no_context_given():
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "NONE"

    fake_self = _classifier(llm_fallback_fn=fake_llm)
    candidates = [_candidate("system_overview", 0.5), _candidate("architecture_reference", 0.48)]

    ContextClassifier._verify_classification_with_llm(fake_self, "A short sentence.", candidates)
    assert "Surrounding context" not in captured["prompt"]


def test_verify_classification_omits_context_block_when_context_equals_sentence():
    # A single-sentence window (no real neighbors) degenerates to the same
    # text as the sentence itself — must not pad the prompt with a
    # "context" block that says nothing new.
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "NONE"

    fake_self = _classifier(llm_fallback_fn=fake_llm)
    candidates = [_candidate("system_overview", 0.5)]

    ContextClassifier._verify_classification_with_llm(
        fake_self, "Only sentence.", candidates, context_text="Only sentence.",
    )
    assert "Surrounding context" not in captured["prompt"]


def test_maybe_verify_with_llm_forwards_context_text_to_verification():
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "architecture_reference"

    fake_self = _classifier(llm_fallback_fn=fake_llm, similarity_threshold=0.15)
    primary = _candidate("system_overview", 0.10)
    secondary = [_candidate("architecture_reference", 0.095)]

    new_primary, new_secondary, note = ContextClassifier._maybe_verify_with_llm(
        fake_self, "It runs on managed infrastructure.", primary, secondary,
        context_text="All services run on Amazon EKS. It runs on managed infrastructure. Traffic enters through CloudFront.",
    )
    assert new_primary.section_id == "architecture_reference"
    assert "Surrounding context" in captured["prompt"]
    assert note is not None


def test_maybe_verify_with_llm_defaults_to_no_context_for_callers_without_one():
    # classify_sentence()'s single-sentence path has no neighbor window
    # available — must keep working exactly as before with an empty context.
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "NONE"

    fake_self = _classifier(llm_fallback_fn=fake_llm, similarity_threshold=0.15)
    primary = _candidate("system_overview", 0.10)
    secondary = [_candidate("architecture_reference", 0.095)]

    ContextClassifier._maybe_verify_with_llm(fake_self, "A sentence.", primary, secondary)
    assert "Surrounding context" not in captured["prompt"]
