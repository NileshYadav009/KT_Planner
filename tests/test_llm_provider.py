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
