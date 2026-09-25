"""Tests for llm_provider.py's rate-limit handling: the proactive
sliding-window throttle (_throttle) and the retry-classification helpers
(_is_rate_limit_error, _extract_retry_delay_seconds). See
REPOSITORY_AUDIT.md §9m for why the throttle exists — retry-with-backoff
alone wasn't enough to stop live 429s.

The throttle tests monkeypatch time.monotonic/time.sleep with a controllable
fake clock so they run in milliseconds instead of waiting out real 60s
windows.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time as time_module

import pytest

import llm_provider as lp


class _FakeClock:
    def __init__(self, start=1000.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _patched_clock(monkeypatch, start=1000.0):
    clock = _FakeClock(start)
    monkeypatch.setattr(time_module, "monotonic", clock.monotonic)
    monkeypatch.setattr(time_module, "sleep", clock.sleep)
    return clock


def test_throttle_allows_calls_up_to_the_limit_without_waiting(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 3)
    lp._call_timestamps.clear()
    clock = _patched_clock(monkeypatch)

    for _ in range(3):
        lp._throttle("Groq")

    assert clock.sleeps == []


def test_throttle_blocks_the_call_over_the_limit(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 3)
    lp._call_timestamps.clear()
    clock = _patched_clock(monkeypatch)

    for _ in range(3):
        lp._throttle("Groq")
    lp._throttle("Groq")

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] > 0


def test_throttle_gemini_and_groq_buckets_are_independent(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 2)
    lp._call_timestamps.clear()
    clock = _patched_clock(monkeypatch)

    lp._throttle("Gemini")
    lp._throttle("Gemini")
    # Gemini bucket is now full — but Groq's is untouched and should not wait.
    lp._throttle("Groq")

    assert clock.sleeps == []


def test_throttle_gemini_labels_share_one_bucket(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 2)
    lp._call_timestamps.clear()
    clock = _patched_clock(monkeypatch)

    lp._throttle("Gemini")
    lp._throttle("Gemini (HTTP fallback)")
    # Bucket is now full via the combined count of both labels.
    lp._throttle("Gemini")

    assert len(clock.sleeps) == 1


def test_throttle_old_timestamps_age_out_of_the_window(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 1)
    lp._call_timestamps.clear()
    clock = _patched_clock(monkeypatch)

    lp._throttle("Groq")
    clock.now += 61  # advance past the 60s window without going through _throttle's own sleep
    lp._throttle("Groq")

    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Tokens-per-minute throttle. Measured live on a real Groq account: the tier
# allowed 1,000 requests/DAY but only 8,000 tokens/MINUTE, so a request-count
# throttle of 20/min sent several times the token budget and every call 429'd
# (23 consecutive rate-limit retries, zero structured extractions completed).
# See REPOSITORY_AUDIT.md §9qq.
# ---------------------------------------------------------------------------

def _clear_throttle_state():
    lp._call_timestamps.clear()
    lp._token_timestamps.clear()
    lp._learned_token_limits.clear()


def test_token_throttle_is_inert_when_no_budget_is_known(monkeypatch):
    # No env override and nothing learned from a provider => uncapped, so a
    # deployment that never sees a token limit behaves exactly as before.
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 0)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    for _ in range(5):
        lp._throttle("Groq", estimated_tokens=100000)

    assert clock.sleeps == []


def test_token_throttle_blocks_when_the_token_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    # 3 x 2500 = 7500 fits under 8000; the 4th would reach 10000 and must wait
    # even though the REQUEST count (4) is nowhere near its own limit of 100.
    for _ in range(3):
        lp._throttle("Groq", estimated_tokens=2500)
    assert clock.sleeps == []

    lp._throttle("Groq", estimated_tokens=2500)
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] > 0


def test_token_throttle_window_ages_out(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    lp._throttle("Groq", estimated_tokens=7000)
    clock.now += 61
    lp._throttle("Groq", estimated_tokens=7000)

    assert clock.sleeps == []


def test_token_throttle_does_not_deadlock_on_an_oversized_prompt(monkeypatch):
    # A single prompt larger than the whole per-minute budget can never fit.
    # It must still be sent (and be rejected by the provider, which the retry
    # path handles) rather than block the run forever.
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    lp._throttle("Groq", estimated_tokens=50000)

    assert clock.sleeps == []


def test_token_throttle_buckets_are_independent(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    lp._throttle("Groq", estimated_tokens=7900)
    lp._throttle("Gemini", estimated_tokens=7900)

    assert clock.sleeps == []


def test_token_limit_is_recorded_from_provider_headers(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 0)
    _clear_throttle_state()

    lp._note_token_limit("Groq", {"x-ratelimit-limit-tokens": "8000"})

    # Recorded per bucket...
    assert lp._learned_token_limits.get("groq") == 8000
    assert lp._learned_token_limits.get("gemini") is None


def test_a_learned_limit_alone_does_not_switch_throttling_on(monkeypatch):
    # Deliberate: the throttle is opt-in. Learning that a tokens/minute bucket
    # EXISTS is not evidence it is the binding constraint -- on the account
    # that motivated this code the real limit was tokens-per-DAY and the
    # per-minute bucket never filled, so auto-activating would have added
    # latency to every run for nothing. See REPOSITORY_AUDIT.md §9qq.
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 0)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    lp._note_token_limit("Groq", {"x-ratelimit-limit-tokens": "8000"})
    assert lp._token_budget("groq") == 0

    # Way over the learned 8000, and still not throttled.
    lp._throttle("Groq", estimated_tokens=6000)
    lp._throttle("Groq", estimated_tokens=6000)
    assert clock.sleeps == []


def test_explicit_env_budget_is_what_enables_the_throttle(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 1234)
    _clear_throttle_state()
    lp._note_token_limit("Groq", {"x-ratelimit-limit-tokens": "8000"})

    # The env var wins over the learned value, in both directions.
    assert lp._token_budget("groq") == 1234


def test_note_token_limit_ignores_junk_headers(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 0)
    for headers in (None, {}, {"x-ratelimit-limit-tokens": ""},
                    {"x-ratelimit-limit-tokens": "not-a-number"},
                    {"x-ratelimit-limit-tokens": "0"},
                    {"x-ratelimit-limit-tokens": "-5"}):
        _clear_throttle_state()
        lp._note_token_limit("Groq", headers)
        assert lp._learned_token_limits.get("groq") is None, headers


def test_actual_usage_corrects_an_under_estimate(monkeypatch):
    # The estimate is a heuristic and under-counts on jargon-dense prose.
    # Measured live: throttling purely on the estimate still drew 35 retries
    # because the shortfall accumulated across dozens of calls. The provider's
    # own usage.total_tokens is exact and must reconcile the window.
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    # This call reserved 1000 but really cost 7500.
    lp._throttle("Groq", estimated_tokens=1000)
    lp.reconcile_token_usage("Groq", estimated_tokens=1000, actual_tokens=7500)

    # The next call must now wait, even though the local estimate alone would
    # have happily allowed another 7000 tokens.
    lp._throttle("Groq", estimated_tokens=1000)
    assert len(clock.sleeps) == 1


def test_actual_usage_shortfall_is_the_difference_not_the_total(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    _patched_clock(monkeypatch)

    lp._throttle("Groq", estimated_tokens=1000)
    lp.reconcile_token_usage("Groq", estimated_tokens=1000, actual_tokens=1500)

    # 1000 reserved + 500 shortfall, NOT 1000 + 1500.
    assert sum(tokens for _, tokens in lp._token_timestamps["groq"]) == 1500


def test_actual_usage_never_refunds_an_over_estimate(monkeypatch):
    # An over-estimate is deliberately left in place: trusting the heuristic
    # further is precisely how the estimate-only version failed.
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    _patched_clock(monkeypatch)

    lp._throttle("Groq", estimated_tokens=3000)
    lp.reconcile_token_usage("Groq", estimated_tokens=3000, actual_tokens=800)

    assert sum(tokens for _, tokens in lp._token_timestamps["groq"]) == 3000


def test_reconcile_uses_the_right_bucket(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_TOKENS_PER_MINUTE", 8000)
    _clear_throttle_state()
    _patched_clock(monkeypatch)

    lp.reconcile_token_usage("Groq", estimated_tokens=0, actual_tokens=5000)

    assert sum(tokens for _, tokens in lp._token_timestamps["groq"]) == 5000
    assert sum(tokens for _, tokens in lp._token_timestamps["gemini"]) == 0


def test_estimate_tokens_counts_prompt_and_output_budget():
    # Output budget must be included: providers metering TPM charge for the
    # completion too, so costing only the prompt under-reserves by exactly
    # the max_output_tokens the call asked for.
    prompt_only = lp.estimate_tokens("x" * 3500)
    with_output = lp.estimate_tokens("x" * 3500, max_output_tokens=1024)

    assert prompt_only == 1000
    assert with_output == 1000 + 1024


def test_estimate_tokens_tolerates_none_and_empty_parts():
    assert lp.estimate_tokens(None, "", max_output_tokens=0) == 0
    assert lp.estimate_tokens(None, "x" * 350) == 100


# ---------------------------------------------------------------------------
# Per-DAY quota exhaustion must fail fast, not retry.
#
# Measured live: a Groq free tier refused every call with "on tokens per day
# (TPD): Limit 200000, Used 199517" while the per-MINUTE bucket sat completely
# full (x-ratelimit-remaining-tokens: 8000). Retrying that 5 times at 60s
# apiece turned each refusal into a 5-minute stall and still failed, so a KT
# run ground for over an hour and produced nothing. Retries cannot clear a
# daily quota. See REPOSITORY_AUDIT.md §9qq.
# ---------------------------------------------------------------------------

class _QuotaResponse:
    def __init__(self, headers=None, status_code=429):
        self.headers = headers or {}
        self.status_code = status_code


class _QuotaRateLimitError(Exception):
    def __init__(self, message, headers=None):
        super().__init__(message)
        self.status_code = 429
        self.response = _QuotaResponse(headers)


_GROQ_TPD_BODY = (
    "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` "
    "service tier `on_demand` on tokens per day (TPD): Limit 200000, "
    "Used 199517, Requested 1450. Please try again in 6m57.744s."
)


def test_daily_quota_error_is_recognized():
    assert lp._is_daily_quota_error(_QuotaRateLimitError(_GROQ_TPD_BODY))
    # Gemini names the window in its quota metric instead.
    assert lp._is_daily_quota_error(_QuotaRateLimitError(
        "quota metric GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded"
    ))


def test_per_minute_rate_limit_is_not_treated_as_a_daily_quota():
    exc = _QuotaRateLimitError(
        "Rate limit reached on tokens per minute (TPM): Limit 8000, Used 7900.",
        headers={"retry-after": "12"},
    )
    assert not lp._is_daily_quota_error(exc)


def test_an_unsatisfiably_long_retry_after_counts_as_a_daily_quota(monkeypatch):
    # Even when the provider does not name the window, a suggested delay far
    # beyond the retry cap cannot be satisfied by any retry this loop makes.
    monkeypatch.setattr(lp, "LLM_RETRY_MAX_DELAY_SECONDS", 60)
    exc = _QuotaRateLimitError("rate_limit_exceeded", headers={"retry-after": "418"})
    assert lp._is_daily_quota_error(exc)


def test_daily_quota_raises_immediately_without_retrying(monkeypatch):
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    _clear_throttle_state()
    clock = _patched_clock(monkeypatch)

    attempts = []

    def _boom():
        attempts.append(1)
        raise _QuotaRateLimitError(_GROQ_TPD_BODY)

    with pytest.raises(Exception):
        lp._call_with_rate_limit_retry(_boom, "Groq", 100)

    # Called exactly once, and never slept -- the whole point of failing fast.
    assert len(attempts) == 1
    assert clock.sleeps == []


def test_per_minute_rate_limit_still_retries(monkeypatch):
    # The fail-fast path must not regress the behaviour it sits next to.
    monkeypatch.setattr(lp, "LLM_MAX_CALLS_PER_MINUTE", 100)
    monkeypatch.setattr(lp, "LLM_RETRY_MAX_ATTEMPTS", 3)
    _clear_throttle_state()
    _patched_clock(monkeypatch)

    attempts = []

    def _flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise _QuotaRateLimitError(
                "Rate limit reached on tokens per minute (TPM): Limit 8000.",
                headers={"retry-after": "2"},
            )
        return "recovered"

    assert lp._call_with_rate_limit_retry(_flaky, "Groq", 100) == "recovered"
    assert len(attempts) == 3


class _FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeHTTPError(Exception):
    def __init__(self, response):
        self.response = response


def test_is_rate_limit_error_detects_status_code_429():
    exc = _FakeHTTPError(_FakeResponse(429))
    assert lp._is_rate_limit_error(exc) is True


def test_is_rate_limit_error_detects_status_code_attribute_directly():
    exc = Exception("boom")
    exc.status_code = 429
    assert lp._is_rate_limit_error(exc) is True


def test_is_rate_limit_error_false_for_unrelated_error():
    assert lp._is_rate_limit_error(ValueError("connection reset by peer")) is False


def test_is_rate_limit_error_detects_resource_exhausted_text():
    assert lp._is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED")) is True


def test_extract_retry_delay_prefers_retry_after_header():
    exc = _FakeHTTPError(_FakeResponse(429, headers={"retry-after": "12"}))
    assert lp._extract_retry_delay_seconds(exc, fallback=99) == 12.0


def test_extract_retry_delay_parses_gemini_retry_delay_in_message():
    exc = Exception("...'retryDelay': '40s'...")
    assert lp._extract_retry_delay_seconds(exc, fallback=99) == 40.0


def test_extract_retry_delay_falls_back_when_nothing_found():
    exc = Exception("no delay info here")
    assert lp._extract_retry_delay_seconds(exc, fallback=7.5) == 7.5
