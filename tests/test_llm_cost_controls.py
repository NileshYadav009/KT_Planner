"""LLM cost controls: exact-response cache, per-KT usage accounting, and the
deterministic skips. None of these may change what a KT contains; the tests
use stub providers only (no network, no tokens)."""
import contextvars
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm import cache
from llm.cached_provider import CachedLLMProvider
from llm.usage import LLMUsageTracker, track_job


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "llm_cache.sqlite"))
    monkeypatch.setenv("LLM_CACHE", "readwrite")
    monkeypatch.delenv("LLM_CACHE_NAMESPACE", raising=False)


class CountingProvider:
    model = "stub-model"

    def __init__(self, reply="answer"):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, *args, **kwargs):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# Named like the real call site so the wrapper labels it "section_polish".
def _polish_one(provider, prompt, **kwargs):
    return provider.generate(prompt, **kwargs)


def test_identical_request_is_served_from_cache_with_identical_text():
    inner = CountingProvider("Polished section text.")
    provider = CachedLLMProvider(inner)
    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        first = _polish_one(provider, "prompt A", temperature=0.2, max_output_tokens=1024, system_prompt="sys")
        second = _polish_one(provider, "prompt A", temperature=0.2, max_output_tokens=1024, system_prompt="sys")
    assert first == second == "Polished section text."
    assert inner.calls == 1
    usage = tracker.summary()
    assert usage["llm_calls"] == 1 and usage["cache_hits"] == 1
    assert usage["stages"][0]["stage"] == "section_polish"
    assert usage["tokens_saved_by_cache"] > 0


@pytest.mark.parametrize("change", [
    {"prompt": "prompt B"},
    {"temperature": 0.1},
    {"max_output_tokens": 128},
    {"system_prompt": "other"},
])
def test_any_change_to_the_request_misses_the_cache(change):
    inner = CountingProvider()
    provider = CachedLLMProvider(inner)
    base = {"prompt": "prompt A", "temperature": 0.2, "max_output_tokens": 1024, "system_prompt": "sys"}
    provider.generate(base["prompt"], **{k: v for k, v in base.items() if k != "prompt"})
    req = {**base, **change}
    provider.generate(req["prompt"], **{k: v for k, v in req.items() if k != "prompt"})
    assert inner.calls == 2


def test_a_different_model_or_namespace_misses_the_cache(monkeypatch):
    inner = CountingProvider()
    CachedLLMProvider(inner).generate("p")
    inner.model = "another-model"
    CachedLLMProvider(inner).generate("p")
    monkeypatch.setenv("LLM_CACHE_NAMESPACE", "tenant-b")
    CachedLLMProvider(inner).generate("p")
    assert inner.calls == 3


def test_failures_and_empty_replies_are_never_cached():
    failing = CountingProvider(RuntimeError("429"))
    provider = CachedLLMProvider(failing)
    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        for _ in range(2):
            with pytest.raises(RuntimeError):
                provider.generate("p")
    assert failing.calls == 2
    assert tracker.summary()["failed"] == 2

    empty = CountingProvider("   ")
    provider = CachedLLMProvider(empty)
    provider.generate("q")
    provider.generate("q")
    assert empty.calls == 2


def test_cache_off_and_read_only_modes(monkeypatch):
    inner = CountingProvider()
    monkeypatch.setenv("LLM_CACHE", "off")
    CachedLLMProvider(inner).generate("p")
    CachedLLMProvider(inner).generate("p")
    assert inner.calls == 2
    monkeypatch.setenv("LLM_CACHE", "read")
    CachedLLMProvider(inner).generate("p")
    CachedLLMProvider(inner).generate("p")
    assert inner.calls == 4  # read-only never stores


def test_expired_entries_are_not_reused(monkeypatch):
    inner = CountingProvider()
    CachedLLMProvider(inner).generate("p")
    monkeypatch.setattr(cache, "_ttl_seconds", lambda: 1e-9)
    CachedLLMProvider(inner).generate("p")
    assert inner.calls == 2


def test_calls_on_worker_threads_count_against_the_same_kt():
    provider = CachedLLMProvider(CountingProvider())
    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(contextvars.copy_context().run, _polish_one, provider, f"p{i}") for i in range(5)]
            [f.result() for f in futures]
    assert tracker.summary()["llm_calls"] == 5


def test_provider_reported_tokens_are_used_when_available():
    import llm_provider

    class ReportingProvider(CountingProvider):
        def generate(self, prompt, *args, **kwargs):
            llm_provider._note_usage(123, 45)
            return super().generate(prompt, *args, **kwargs)

    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        CachedLLMProvider(ReportingProvider()).generate("p")
    usage = tracker.summary()
    assert (usage["input_tokens"], usage["output_tokens"], usage["estimated_tokens"]) == (123, 45, False)


def test_groq_usage_block_is_read(monkeypatch):
    import llm_provider
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )
    groq = llm_provider.GroqProvider.__new__(llm_provider.GroqProvider)
    groq.model, groq.api_key, groq.base_url = "m", "test-key", "http://localhost"
    groq.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    monkeypatch.setattr(llm_provider, "_throttle", lambda *a, **k: None)
    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        assert CachedLLMProvider(groq).generate("hello") == "ok"
    usage = tracker.summary()
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_cost_is_reported_only_when_prices_are_configured(monkeypatch):
    tracker = LLMUsageTracker("job")
    tracker.record_call("field_fill", input_tokens=1_000_000, output_tokens=500_000, estimated=False, seconds=1)
    assert tracker.summary()["estimated_cost_usd"] is None
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "0.5")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "1.0")
    assert tracker.summary()["estimated_cost_usd"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Deterministic skips
# --------------------------------------------------------------------------

def test_url_and_date_prechecks_only_skip_when_nothing_could_be_a_literal():
    from field_populator import _may_contain_literal
    assert not _may_contain_literal("url", "The architecture is explained verbally during onboarding.")
    for text in ("See https://wiki.example.com/x", "It is on Confluence", "docs live at runbooks.internal",
                 "the diagram is in the repo"):
        assert _may_contain_literal("url", text), text
    assert not _may_contain_literal("date", "Nobody mentioned when the diagram was refreshed.")
    for text in ("Updated in March", "last week", "on 2026-01-04", "Q3"):
        assert _may_contain_literal("date", text), text


def test_url_field_without_a_candidate_link_makes_no_llm_call():
    from field_populator import populate_fields
    inner = CountingProvider("NOT_MENTIONED\nEXPLICIT")
    provider = CachedLLMProvider(inner)
    schema = [{"id": "architecture_reference", "title": "Architecture", "fields": [
        {"id": "architecture_link", "label": "Link to architecture documentation", "type": "url"},
    ]}]
    coverage = {"architecture_reference": {"content": ["The service reads from the queue and writes to the database."]}}
    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        out = populate_fields(schema, coverage, llm_provider=provider, embedding_model=None)
    assert inner.calls == 0
    assert not (out["architecture_reference"]["architecture_link"].get("value"))
    assert tracker.summary()["skipped"] == 1


def _verify_self(answer="deployment_and_rollback"):
    from context_mapper import ContextClassifier
    calls = []

    def fallback(prompt, **kwargs):
        calls.append(prompt)
        return answer

    fake = SimpleNamespace(llm_fallback_fn=fallback, similarity_threshold=0.15)
    fake._verify_classification_with_llm = lambda *a, **k: ContextClassifier._verify_classification_with_llm(fake, *a, **k)
    return fake, calls


def _borderline():
    from context_mapper import Classification
    primary = Classification("danger_zones", "Danger", 0.18, "", 0.18)
    secondary = [Classification("deployment_and_rollback", "Deploy", 0.17, "", 0.17)]
    return primary, secondary


@pytest.mark.parametrize("mode,expect_call", [("shadow", True), ("skip", False), ("off", True)])
def test_rule_decided_classification_check_modes(monkeypatch, mode, expect_call):
    from context_mapper import ContextClassifier
    monkeypatch.setenv("LLM_VERIFY_RULE_DECIDED", mode)
    fake, calls = _verify_self()
    primary, secondary = _borderline()
    tracker = LLMUsageTracker("job")
    with track_job(tracker):
        ContextClassifier._maybe_verify_with_llm(fake, "Never touch prod.", primary, secondary, rule_decided=True)
    assert bool(calls) is expect_call
    usage = tracker.summary()
    assert usage["skipped"] == (1 if mode == "skip" else 0)
    assert usage["would_skip"] == (1 if mode == "shadow" else 0)


def test_sentences_without_a_deciding_rule_are_always_checked(monkeypatch):
    from context_mapper import ContextClassifier
    monkeypatch.setenv("LLM_VERIFY_RULE_DECIDED", "skip")
    fake, calls = _verify_self()
    primary, secondary = _borderline()
    new_primary, _, note = ContextClassifier._maybe_verify_with_llm(fake, "Some sentence.", primary, secondary)
    assert calls and new_primary.section_id == "deployment_and_rollback" and note
