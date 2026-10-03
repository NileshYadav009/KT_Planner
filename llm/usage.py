"""Per-KT accounting of LLM calls: how many calls each pipeline stage made,
the tokens they used, time spent waiting, retries, and how many were answered
from the cache or skipped. Shown to the user next to the generated KT.

One tracker per job, made current with `track_job()`. The pipeline's thread
pools copy the context into their workers, so calls made there are counted
against the same job. With no current tracker nothing is recorded.
"""
import contextvars
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from llm.cache import cache_mode

# Human-readable names for the call sites, in pipeline order.
STAGE_LABELS = {
    "classification_check": "Classification check",
    "sentence_repair": "Sentence repair",
    "section_polish": "Section polish",
    "structured_extraction": "Structured extraction",
    "field_fill": "Field fill",
    "other": "Other",
}

_CURRENT: contextvars.ContextVar[Optional["LLMUsageTracker"]] = contextvars.ContextVar(
    "continuum_llm_usage", default=None
)


class LLMUsageTracker:
    def __init__(self, job_id: str = ""):
        self.job_id = job_id
        self.started = time.time()
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self._lock = threading.Lock()
        self._stages: Dict[str, Dict[str, Any]] = {}

    def _stage(self, call_site: str) -> Dict[str, Any]:
        return self._stages.setdefault(call_site, {
            "llm_calls": 0, "cache_hits": 0, "skipped": 0, "would_skip": 0, "failed": 0,
            "input_tokens": 0, "output_tokens": 0, "estimated_tokens": False,
            "llm_seconds": 0.0, "retries": 0, "skip_reasons": {},
        })

    def record_call(self, call_site: str, *, input_tokens: int, output_tokens: int,
                    estimated: bool, seconds: float, retries: int = 0, failed: bool = False,
                    provider: Optional[str] = None, model: Optional[str] = None) -> None:
        with self._lock:
            st = self._stage(call_site)
            st["llm_calls"] += 1
            st["failed"] += 1 if failed else 0
            st["input_tokens"] += int(input_tokens or 0)
            st["output_tokens"] += int(output_tokens or 0)
            st["estimated_tokens"] = st["estimated_tokens"] or bool(estimated)
            st["llm_seconds"] += float(seconds or 0.0)
            st["retries"] += int(retries or 0)
            self.provider = provider or self.provider
            self.model = model or self.model

    def record_cache_hit(self, call_site: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            st = self._stage(call_site)
            st["cache_hits"] += 1
            st.setdefault("tokens_saved", 0)
            st["tokens_saved"] += int(input_tokens or 0) + int(output_tokens or 0)

    def record_skip(self, call_site: str, reason: str, *, shadow: bool = False) -> None:
        """A call that deterministic logic made unnecessary. In shadow mode the
        call is still made; it is only counted as one that could be skipped."""
        with self._lock:
            st = self._stage(call_site)
            st["would_skip" if shadow else "skipped"] += 1
            key = reason + (" (shadow)" if shadow else "")
            st["skip_reasons"][key] = st["skip_reasons"].get(key, 0) + 1

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            order = list(STAGE_LABELS)
            stages = []
            for call_site in sorted(self._stages, key=lambda s: order.index(s) if s in order else len(order)):
                st = dict(self._stages[call_site])
                st["skip_reasons"] = dict(st["skip_reasons"])
                st["llm_seconds"] = round(st["llm_seconds"], 2)
                st["stage"] = call_site
                st["label"] = STAGE_LABELS.get(call_site, call_site)
                stages.append(st)
            total = lambda k: sum(s.get(k, 0) for s in stages)
            # Cost only when the operator has said what tokens cost; never a
            # guessed price.
            cost = None
            try:
                price_in = os.getenv("LLM_PRICE_INPUT_PER_MTOK")
                price_out = os.getenv("LLM_PRICE_OUTPUT_PER_MTOK")
                if price_in and price_out:
                    cost = round(
                        total("input_tokens") / 1e6 * float(price_in)
                        + total("output_tokens") / 1e6 * float(price_out), 6,
                    )
            except ValueError:
                cost = None
            return {
                "estimated_cost_usd": cost,
                "cache_mode": cache_mode(),
                "provider": self.provider,
                "model": self.model,
                "llm_calls": total("llm_calls"),
                "cache_hits": total("cache_hits"),
                "skipped": total("skipped"),
                "would_skip": total("would_skip"),
                "failed": total("failed"),
                "input_tokens": total("input_tokens"),
                "output_tokens": total("output_tokens"),
                "tokens_saved_by_cache": total("tokens_saved"),
                "estimated_tokens": any(s["estimated_tokens"] for s in stages),
                "llm_seconds": round(total("llm_seconds"), 2),
                "retries": total("retries"),
                "stages": stages,
            }


def current_tracker() -> Optional[LLMUsageTracker]:
    return _CURRENT.get()


@contextmanager
def track_job(tracker: LLMUsageTracker):
    token = _CURRENT.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT.reset(token)


def start_tracking(tracker: LLMUsageTracker):
    """Make `tracker` current; returns a token for stop_tracking()."""
    return _CURRENT.set(tracker)


def stop_tracking(token) -> None:
    _CURRENT.reset(token)


def record_skip(call_site: str, reason: str, *, shadow: bool = False) -> None:
    tracker = current_tracker()
    if tracker is not None:
        tracker.record_skip(call_site, reason, shadow=shadow)
