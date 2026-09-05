import os
import json
import logging
import re
import time
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

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "ratelimit", "quota")


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


def _call_with_rate_limit_retry(call_fn, provider_label: str):
    """Call call_fn() (a zero-arg callable making the real API request),
    retrying on rate-limit errors with the provider's own suggested delay
    (falling back to capped exponential backoff), up to LLM_RETRY_MAX_ATTEMPTS
    times. Re-raises the last error if every attempt is exhausted, so the
    caller's existing fallback-on-exception behavior still applies as the
    final safety net.
    """
    last_exc = None
    for attempt in range(LLM_RETRY_MAX_ATTEMPTS + 1):
        try:
            return call_fn()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == LLM_RETRY_MAX_ATTEMPTS:
                raise
            backoff = min(2 ** attempt, LLM_RETRY_MAX_DELAY_SECONDS)
            delay = min(_extract_retry_delay_seconds(exc, fallback=backoff), LLM_RETRY_MAX_DELAY_SECONDS)
            LOGGER.warning(
                "%s rate-limited (attempt %d/%d), retrying in %.1fs",
                provider_label, attempt + 1, LLM_RETRY_MAX_ATTEMPTS, delay,
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

            response = _call_with_rate_limit_retry(_do_call, "Gemini")
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

        r = _call_with_rate_limit_retry(_do_call, "Gemini (HTTP fallback)")
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
            return self.client.chat.completions.create(**completion_kwargs)

        response = _call_with_rate_limit_retry(_do_call, "Groq")
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
