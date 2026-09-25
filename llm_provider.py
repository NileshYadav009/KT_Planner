import os
import json
import logging
import re
import threading
import time
from collections import defaultdict, deque
from typing import Optional

LOGGER = logging.getLogger(__name__)

# Load .env before reading any provider config below. This module must not
# rely on some other module (e.g. ai.py, which also calls load_dotenv but only
# after already importing this module) happening to load the environment
# first — that ordering bug meant GEMINI_API_KEY/GROQ_API_KEY were silently
# read as empty on every fresh process start regardless of what .env
# contained, disabling every LLM-dependent feature in the app.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Groq/OpenAI-compatible SDK support
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Gemini support via Google GenAI
try:
    from google import genai
except Exception:
    genai = None

import requests

LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llama-3.3-70b-versatile has been decommissioned on Groq (confirmed live via
# this account's /models list, 2026-09-05 — it's no longer in the catalog).
# qwen/qwen3.8-27b verified as a working replacement: non-reasoning-leaking
# (unlike openai/gpt-oss-* and qwen/qwen3.6-27b, which spend/leak tokens on
# chain-of-thought and return empty/garbled content at this codebase's
# existing short max_output_tokens budgets — as low as 20 tokens in
# context_mapper.py's classification verification), handles real structured
# JSON extraction cleanly, and is the larger/more capable of the two verified
# non-reasoning options on this account (the other being allam-2-7b, 7B vs 27B).
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ============================================================================
# Rate-limit retry
# ============================================================================
#
# A single KT run makes 15-25+ LLM calls in quick succession (polish, gap-fill,
# structured extraction, classification verification), which bursts past both
# Gemini's and Groq's free-tier per-minute limits partway through — observed
# live: 35 succeeded / 17 hit 429 out of 52 requests in one minute on Groq.
# Every real call site already wraps provider.generate() in a try/except that
# just logs and falls back to non-LLM behavior on ANY failure — which turns a
# transient, recoverable rate limit into a permanent quality loss for that
# field/section. Retrying here, inside the provider, fixes it for every call
# site at once without any of them needing to know rate limits exist.
LLM_RETRY_MAX_ATTEMPTS = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "5"))
LLM_RETRY_MAX_DELAY_SECONDS = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "60"))

# Retrying after a 429 (above) doesn't reduce how many requests were already
# in flight — a single KT run's 15-25+ calls can burst well past a real
# per-minute quota regardless of how well failures are retried afterward
# (observed live: repeated ~35-41 req/min bursts still drawing a growing
# fraction of 429s even with retry-with-backoff already in place). This is
# the proactive half: cap how many requests THIS PROCESS sends per minute,
# before they go out, shared across all concurrent KT jobs (pipeline.py's
# process_upload_task runs via FastAPI BackgroundTasks, so more than one job's
# calls can legitimately overlap in the same window).
LLM_MAX_CALLS_PER_MINUTE = int(os.getenv("LLM_MAX_CALLS_PER_MINUTE", "20"))

# A single KT run's 15-25+ LLM calls are spread across several independent
# per-section loops (structured extraction, prose polish, field gap-fill
# across different sections) that used to run strictly one call at a time —
# the dominant cost was simply waiting for each network round trip to finish
# before starting the next, unrelated one. _throttle() above is already
# thread-safe (a lock-guarded sliding window shared by every caller), so
# dispatching independent calls concurrently is safe: it can only make those
# calls start sooner, never exceed LLM_MAX_CALLS_PER_MINUTE — the throttle
# still enforces that ceiling exactly as before, across however many threads
# are calling it. Kept modest by default to avoid hammering a provider with
# simultaneous connections beyond what the per-minute quota already implies.
LLM_PARALLEL_WORKERS = int(os.getenv("LLM_PARALLEL_WORKERS", "4"))

# The request-count throttle above is necessary but NOT sufficient: several
# provider tiers meter TOKENS per minute, not requests. Measured live on a
# real Groq account for qwen/qwen3.8-27b: 1,000 requests/DAY but only
# 8,000 tokens/MINUTE (the account's own x-ratelimit-limit-tokens header).
# A KT run's polish and structured-extraction prompts each carry a slice of
# the transcript, so 20 requests/minute of multi-thousand-token prompts is
# several times over a budget the request counter cannot see: every call
# 429s, retries 5x at 60s apiece, and the run grinds for hours while still
# losing the content those calls were supposed to produce. Confirmed live on
# a ~2,600-word transcript — 23 consecutive rate-limit retries, zero
# structured extractions completed, with the request throttle working
# exactly as designed. This is the token half of the same throttle.
#
# Default 0 (no token ceiling) so no already-working deployment slows down:
# the real budget is LEARNED from the provider's own response headers
# (_note_token_limit below), which is exact, per-account and needs no
# configuration. An explicit env var overrides the learned value.
LLM_MAX_TOKENS_PER_MINUTE = int(os.getenv("LLM_MAX_TOKENS_PER_MINUTE", "0"))

# Characters per token, for costing a prompt BEFORE sending it. Deliberately
# below the ~4 real English average: over-estimating only makes the throttle
# slightly more patient, whereas under-estimating re-creates the 429 storm
# this exists to prevent.
_CHARS_PER_TOKEN = 3.5

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "ratelimit", "quota")

_throttle_lock = threading.Lock()
_call_timestamps: dict = defaultdict(deque)
# bucket -> deque[(monotonic_ts, tokens_reserved)], the token half of the
# same sliding 60s window as _call_timestamps.
_token_timestamps: dict = defaultdict(deque)
# bucket -> tokens/minute as reported by the provider itself. Populated from
# response headers (success or 429), so the ceiling matches the real account
# tier instead of a guess baked into this file.
_learned_token_limits: dict = {}


def _bucket_for(provider_label: str) -> str:
    """Gemini's two labels ("Gemini" / "Gemini (HTTP fallback)") share one
    bucket since both hit the same quota."""
    return "gemini" if provider_label.startswith("Gemini") else "groq"


def _token_budget(bucket: str) -> int:
    """Tokens/minute ceiling for this bucket, or 0 (= uncapped).

    Deliberately gated on the EXPLICIT env var only. The value learned from
    provider headers (_learned_token_limits) is recorded and logged, but does
    NOT switch throttling on by itself.

    Why: this throttle was originally built to auto-activate from the learned
    header, which meant it silently started throttling every Groq run. On the
    account that motivated it, tokens-per-minute was never the binding limit
    at all (the real one was tokens-per-DAY — see REPOSITORY_AUDIT.md §9qq),
    so auto-activation added latency to every run to protect against a ceiling
    that was never being hit. A throttle that slows down real work should be
    turned on by someone who has measured that they need it, not inferred from
    a header that merely advertises a bucket's existence.
    """
    return LLM_MAX_TOKENS_PER_MINUTE if LLM_MAX_TOKENS_PER_MINUTE > 0 else 0


def estimate_tokens(*texts: Optional[str], max_output_tokens: int = 0) -> int:
    """Conservative token cost of a call: its prompt text plus the completion
    budget it reserves. Providers that meter TPM count both halves, so
    costing only the prompt under-reserves by exactly the output budget."""
    chars = sum(len(t) for t in texts if t)
    return int(chars / _CHARS_PER_TOKEN) + max(int(max_output_tokens), 0)


def _header_int(headers, *names) -> Optional[int]:
    """First of `names` present in `headers` as an int, or None. Tolerates any
    header mapping shape and any junk value — this must never be the reason a
    successful call is treated as failed."""
    if not headers:
        return None
    for name in names:
        try:
            raw = headers.get(name) or headers.get(name.title())
        except Exception:
            return None
        if raw in (None, ""):
            continue
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            continue
    return None


def _note_token_limit(provider_label: str, headers) -> None:
    """Learn this account's real tokens/minute ceiling from a provider
    response.

    Called for both successes and 429s — Groq sends the header on either, so
    the very first call of a run is enough to calibrate.
    """
    bucket = _bucket_for(provider_label)

    limit = _header_int(headers, "x-ratelimit-limit-tokens")
    if limit and limit > 0 and _learned_token_limits.get(bucket) != limit:
        with _throttle_lock:
            _learned_token_limits[bucket] = limit
        LOGGER.info("%s token budget learned from provider: %d tokens/minute", provider_label, limit)


def reconcile_token_usage(provider_label: str, estimated_tokens: int, actual_tokens: int) -> None:
    """Book the shortfall when a call really cost more than it reserved.

    estimate_tokens() is a heuristic, and on jargon-dense technical prose it
    under-counts (acronyms and punctuation tokenize worse than the ~3.5
    chars/token English average). A few percent per call is invisible, but it
    ACCUMULATES across dozens of calls in the same window until the window is
    genuinely over budget — measured live as a run that still drew 35 retries
    while throttled purely on the estimate.

    `actual_tokens` is the provider's own `usage.total_tokens` for the call:
    exact, and — unlike the `x-ratelimit-remaining-tokens` header — a fixed
    figure rather than one that decays as the provider's bucket refills, so it
    is directly comparable to what this window reserved. (Reconciling against
    the remaining-tokens header was tried first and is the wrong quantity: it
    recovers within seconds while the local 60s window is still holding the
    reservation, so the shortfall it computes is almost always <= 0.)

    Only under-estimates are corrected. An over-estimate is never refunded,
    because trusting the heuristic further is precisely how the estimate-only
    version failed.
    """
    shortfall = int(actual_tokens) - int(estimated_tokens)
    if shortfall <= 0:
        return
    bucket = _bucket_for(provider_label)
    with _throttle_lock:
        tq = _token_timestamps[bucket]
        now = time.monotonic()
        while tq and now - tq[0][0] >= 60:
            tq.popleft()
        tq.append((now, shortfall))


def _throttle(provider_label: str, estimated_tokens: int = 0) -> None:
    """Block until sending another request stays within BOTH
    LLM_MAX_CALLS_PER_MINUTE and this bucket's tokens/minute budget, over the
    same sliding 60s window.

    `estimated_tokens` of 0 skips the token check entirely, so callers that
    genuinely don't know a call's size behave exactly as before.
    """
    bucket = _bucket_for(provider_label)
    while True:
        with _throttle_lock:
            now = time.monotonic()
            dq = _call_timestamps[bucket]
            while dq and now - dq[0] >= 60:
                dq.popleft()
            tq = _token_timestamps[bucket]
            while tq and now - tq[0][0] >= 60:
                tq.popleft()

            wait = 0.0
            if len(dq) >= LLM_MAX_CALLS_PER_MINUTE:
                wait = 60 - (now - dq[0]) + 0.05

            budget = _token_budget(bucket)
            if budget > 0 and estimated_tokens > 0 and tq:
                # `and tq` matters: a single prompt larger than the entire
                # per-minute budget can never "fit", so gating it on an empty
                # window would deadlock the run forever. Let it through, and
                # let the existing 429 retry path deal with the rejection.
                used = sum(tokens for _, tokens in tq)
                if used + estimated_tokens > budget:
                    wait = max(wait, 60 - (now - tq[0][0]) + 0.05)

            if wait <= 0:
                dq.append(now)
                if estimated_tokens > 0:
                    tq.append((now, estimated_tokens))
                return
        time.sleep(wait)


def _is_rate_limit_error(exc: Exception) -> bool:
    # openai.RateLimitError has .status_code directly; requests.HTTPError has
    # it nested under .response.status_code — check both shapes.
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429:
        return True
    text = str(exc)
    return any(marker.lower() in text.lower() for marker in _RATE_LIMIT_MARKERS)


# A per-DAY quota exhaustion is not a transient rate limit: nothing this
# process does in the next few minutes will clear it. Retrying it 5 times at
# 60s apiece turns one refusal into a 5-minute stall per call and still ends
# in the same failure — measured live as a KT run that ground for over an hour
# producing nothing, while the per-minute bucket sat completely full. Groq
# names the quota in the 429 body ("on tokens per day (TPD): Limit 200000,
# Used 199517"); Gemini names it in its quota metric
# ("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), which lowercases to
# contain "perday". See REPOSITORY_AUDIT.md §9qq.
_DAILY_QUOTA_MARKERS = (
    "tokens per day", "requests per day", "(tpd)", "(rpd)", "per day", "perday", "per-day",
)


def _is_daily_quota_error(exc: Exception) -> bool:
    """True when a 429 is a per-day quota exhaustion rather than a per-minute
    rate limit, so the caller can fail fast instead of retrying pointlessly."""
    text = str(exc).lower()
    if any(marker in text for marker in _DAILY_QUOTA_MARKERS):
        return True
    # Providers that don't name the window still signal it: a suggested delay
    # far longer than the retry cap cannot be satisfied by any retry this loop
    # would make, so treating it as transient is wrong regardless of wording.
    suggested = _extract_retry_delay_seconds(exc, fallback=0.0)
    return suggested > max(LLM_RETRY_MAX_DELAY_SECONDS * 2, 120)


def _extract_retry_delay_seconds(exc: Exception, fallback: float) -> float:
    """Best-effort: honor whatever delay the provider actually suggested,
    rather than always guessing with backoff."""
    # OpenAI-compatible (Groq): Retry-After header on the HTTP response.
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        header_val = headers.get("retry-after") or headers.get("Retry-After")
        if header_val:
            try:
                return max(float(header_val), 0.5)
            except (TypeError, ValueError):
                pass
    # Gemini: retryDelay embedded in the error's JSON body / string form,
    # e.g. "'retryDelay': '40s'" — seen verbatim in this app's own logs.
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        return max(float(match.group(1)), 0.5)
    return fallback


def _call_with_rate_limit_retry(call_fn, provider_label: str, estimated_tokens: int = 0):
    """Call call_fn() (a zero-arg callable making the real API request),
    retrying on rate-limit errors with the provider's own suggested delay
    (falling back to capped exponential backoff), up to LLM_RETRY_MAX_ATTEMPTS
    times. Re-raises the last error if every attempt is exhausted, so the
    caller's existing fallback-on-exception behavior still applies as the
    final safety net.

    Every attempt (the original call and every retry) is throttled via
    _throttle() first — the retry backoff alone only reacts after a 429; the
    throttle is what actually keeps outbound request volume under the
    account's real per-minute quota. `estimated_tokens` additionally keeps it
    under a tokens-per-minute quota, which on some tiers (see
    LLM_MAX_TOKENS_PER_MINUTE) binds long before the request count does.
    """
    last_exc = None
    for attempt in range(LLM_RETRY_MAX_ATTEMPTS + 1):
        _throttle(provider_label, estimated_tokens)
        try:
            return call_fn()
        except Exception as exc:
            # A 429 body carries the account's real ceilings — learn from it
            # even though this attempt failed, so the throttle stops
            # over-sending for the rest of the run.
            _note_token_limit(provider_label, getattr(getattr(exc, "response", None), "headers", None))
            if not _is_rate_limit_error(exc) or attempt == LLM_RETRY_MAX_ATTEMPTS:
                raise
            if _is_daily_quota_error(exc):
                # Fail fast: the caller's existing fallback-on-exception path
                # produces the deterministic result immediately, instead of
                # every remaining call in the run stalling for minutes first.
                LOGGER.warning(
                    "%s per-day quota exhausted — not retrying (retries cannot clear a "
                    "daily quota). Falling back to the non-LLM path. Provider said: %s",
                    provider_label, exc,
                )
                raise
            backoff = min(2 ** attempt, LLM_RETRY_MAX_DELAY_SECONDS)
            delay = min(_extract_retry_delay_seconds(exc, fallback=backoff), LLM_RETRY_MAX_DELAY_SECONDS)
            # The exception text is included deliberately: a bare
            # "rate-limited" line is un-diagnosable, and a whole session was
            # lost to inferring the wrong quota dimension from advertised
            # headers because this log never said which limit the provider
            # actually named.
            LOGGER.warning(
                "%s rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                provider_label, attempt + 1, LLM_RETRY_MAX_ATTEMPTS, delay, exc,
            )
            time.sleep(delay)
            last_exc = exc
    raise last_exc


class LLMProvider:
    def generate(self, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError("LLMProvider subclasses must implement generate()")


class GeminiProvider(LLMProvider):
    def __init__(self):
        self.model = GEMINI_MODEL
        self.api_key = GEMINI_API_KEY
        self.client = None
        if genai is not None and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                LOGGER.warning("Gemini client initialization failed: %s", exc)
                self.client = None

    def generate(self, prompt: str, *args, **kwargs) -> str:
        if not self.api_key and not self.client:
            raise RuntimeError("Gemini provider is not configured. Set GEMINI_API_KEY or install google-genai.")

        if self.client:
            config_kwargs = dict(
                temperature=kwargs.get("temperature", 0.2),
                max_output_tokens=kwargs.get("max_output_tokens", 1024),
                stop_sequences=kwargs.get("stop_sequences", ["\n\n"]),
            )
            # Every structured/polish call site in this codebase passes
            # system_prompt expecting it to matter (see GroqProvider, which
            # already honors it) — this silently dropped it before.
            system_prompt = kwargs.get("system_prompt")
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            def _do_call():
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(**config_kwargs)
                )

            response = _call_with_rate_limit_retry(
                _do_call,
                "Gemini",
                estimate_tokens(
                    system_prompt,
                    prompt,
                    max_output_tokens=config_kwargs.get("max_output_tokens", 0),
                ),
            )
            text = getattr(response, "text", None) or str(response)
            return text.strip()

        url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "prompt": {"text": prompt},
            "temperature": kwargs.get("temperature", 0.2),
            "maxOutputTokens": kwargs.get("max_output_tokens", 1024),
            "stop_sequences": kwargs.get("stop_sequences", ["\n\n"])
        }
        def _do_call():
            resp = requests.post(url, json=payload, timeout=(10, 120))
            resp.raise_for_status()
            return resp

        r = _call_with_rate_limit_retry(
            _do_call,
            "Gemini (HTTP fallback)",
            estimate_tokens(prompt, max_output_tokens=payload.get("maxOutputTokens", 0)),
        )
        data = r.json()
        text = ""
        if isinstance(data, dict):
            if "candidates" in data and data["candidates"]:
                candidates = data["candidates"]
                if isinstance(candidates, list):
                    parts = []
                    for c in candidates:
                        if isinstance(c, dict):
                            parts.append(c.get("content") or c.get("output") or c.get("text", ""))
                        else:
                            parts.append(str(c))
                    text = "\n".join([p for p in parts if p])
            elif "output" in data:
                text = data.get("output")
            elif "response" in data and isinstance(data.get("response"), dict):
                text = data.get("response", {}).get("output", "") or json.dumps(data.get("response", {}))
            else:
                text = data.get("text") or json.dumps(data)
        else:
            text = str(data)
        return str(text).strip()


class GroqProvider(LLMProvider):
    def __init__(self):
        self.model = GROQ_MODEL
        self.api_key = GROQ_API_KEY
        self.base_url = GROQ_BASE_URL
        self.client = None
        if OpenAI is not None and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as exc:
                LOGGER.warning("Groq OpenAI client initialization failed: %s", exc)
                self.client = None

    def generate(self, prompt: str, *args, **kwargs) -> str:
        if not self.api_key:
            raise RuntimeError("Groq provider is not configured. Set GROQ_API_KEY.")
        if self.client is None:
            raise RuntimeError("OpenAI SDK is not available. Install openai.")

        # Every call site in this codebase passes max_output_tokens (Gemini's
        # kwarg name) — this used to only check max_tokens, so a Groq-routed
        # call silently ignored the caller's requested limit and always used
        # the 1024 default instead. Accept max_tokens too for any caller using
        # OpenAI-native naming directly.
        max_tokens = kwargs.get("max_output_tokens", kwargs.get("max_tokens", 1024))

        completion_kwargs = dict(
            model=self.model,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "You are an expert technical writer.")},
                {"role": "user", "content": prompt}
            ]
        )
        # stop_sequences=[] is passed explicitly by some call sites to mean
        # "no stop sequence" — only forward it when non-empty.
        stop_sequences = kwargs.get("stop_sequences")
        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences

        def _do_call():
            # Plain create(), unchanged from before this file gained token
            # accounting. An earlier version routed this through
            # with_raw_response.create() purely to read rate-limit headers on
            # success — reverted: it put an SDK-version-dependent accessor in
            # the path of EVERY Groq call to feed a throttle that is now
            # opt-in, and the headers are still available on 429s where they
            # actually matter. Not worth the exposure for an off-by-default
            # feature.
            return self.client.chat.completions.create(**completion_kwargs)

        estimated = estimate_tokens(
            completion_kwargs["messages"][0]["content"],
            prompt,
            max_output_tokens=max_tokens,
        )
        response = _call_with_rate_limit_retry(_do_call, "Groq", estimated)

        # Correct the throttle's window with what the call actually cost, so a
        # systematic under-estimate can't accumulate into a 429 storm. See
        # reconcile_token_usage().
        try:
            actual = getattr(getattr(response, "usage", None), "total_tokens", None)
            if actual is None and isinstance(response, dict):
                actual = (response.get("usage") or {}).get("total_tokens")
            if actual:
                reconcile_token_usage("Groq", estimated, int(actual))
        except Exception:
            pass

        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = getattr(choice, "message", None) or choice.get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", "")).strip()
            return str(getattr(message, "content", "")).strip()
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return str(message.get("content", "")).strip()
        raise RuntimeError("Groq response did not contain a valid completion.")


class FallbackLLMProvider(LLMProvider):
    def __init__(self, providers):
        self.providers = providers

    def generate(self, prompt: str, **kwargs) -> str:
        last_error = None
        for provider in self.providers:
            try:
                return provider.generate(prompt, **kwargs)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("LLM provider %s failed: %s", provider.__class__.__name__, exc)
                continue
        raise RuntimeError("All LLM providers failed.") from last_error


def create_llm_provider() -> LLMProvider:
    provider_name = LLM_PROVIDER_NAME
    if provider_name == "groq":
        return GroqProvider()
    if provider_name in {"gemini", "google", "genai"}:
        return GeminiProvider()
    if provider_name == "fallback":
        return FallbackLLMProvider([GeminiProvider(), GroqProvider()])
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider_name}'. Use 'gemini', 'groq', or 'fallback'.")


def get_llm_provider() -> Optional[LLMProvider]:
    try:
        return create_llm_provider()
    except Exception as e:
        LOGGER.warning("Failed to initialize LLM provider: %s", e)
        return None
