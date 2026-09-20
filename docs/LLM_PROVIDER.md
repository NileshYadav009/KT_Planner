# LLM Provider

`llm_provider.py` is the single abstraction every LLM-dependent part of the
pipeline (`field_populator.py`'s gap-fill, `ai.py`'s polish/structured
extraction, `context_mapper.py`'s classification verification) calls through
— no call site talks to Gemini or Groq's SDKs directly.

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini`, `groq`, or `fallback` | `gemini` |
| `GEMINI_API_KEY` | Gemini API key | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `GROQ_API_KEY` | Groq API key | — |
| `GROQ_MODEL` | Groq model name | `qwen/qwen3.8-27b` |
| `GROQ_BASE_URL` | Groq's OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `LLM_RETRY_MAX_ATTEMPTS` | Reactive retry attempts on a 429 | `5` |
| `LLM_RETRY_MAX_DELAY_SECONDS` | Cap on any single retry's backoff | `60` |
| `LLM_MAX_CALLS_PER_MINUTE` | Proactive throttle ceiling per provider | `20` |

**Setting `GROQ_API_KEY` alone does not route calls to Groq** — `LLM_PROVIDER`
must also be set to `groq`, or `create_llm_provider()` defaults to Gemini
regardless of which keys happen to be present.

`llm_provider.py` self-loads `.env` at import time (`load_dotenv()`) —
deliberately, because a real bug this session found (`REPOSITORY_AUDIT.md`
§6/Phase 6) was `ai.py` importing `llm_provider` *before* calling its own
`load_dotenv()`, silently baking in empty API keys for the process's entire
lifetime. Any module that imports `llm_provider` gets a correctly-loaded
environment regardless of its own import order.

## Model choice: why `qwen/qwen3.8-27b`

`llama-3.3-70b-versatile` (an earlier default) is confirmed decommissioned on
Groq. Live-tested 5 replacement candidates at this codebase's tightest real
token budget (20 tokens, `context_mapper.py`'s classification-verification
call): `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and `qwen/qwen3.6-27b` all
failed — they're reasoning models that spend/leak tokens on hidden
chain-of-thought, returning empty or garbled output at a tight budget.
`qwen/qwen3.8-27b` and `allam-2-7b` both work cleanly (no reasoning-token
leakage); `qwen/qwen3.8-27b` was chosen as the default for being the larger,
more capable of the two (27B vs. 7B). See `REPOSITORY_AUDIT.md` §9g for the
full empirical comparison.

## Rate-limit handling: two layers, and why both exist

**Layer 1 — reactive retry** (`_call_with_rate_limit_retry()`): on a 429,
honors the provider's own suggested delay (`Retry-After` header for Groq,
the `retryDelay` field embedded in Gemini's error body) with a capped
exponential-backoff fallback, up to `LLM_RETRY_MAX_ATTEMPTS` times.

**Layer 2 — proactive throttle** (`_throttle()`): a sliding-window limiter
that caps outbound requests *before* they're sent, shared process-wide
across all concurrent jobs (one bucket per real quota — Gemini's two
internal labels share a bucket since both hit the same API; Groq is
independent).

Layer 1 alone was tried first and wasn't enough — a single KT run fires
15-25+ LLM calls in quick succession (more with each later phase's
additions), which can burst well past a real per-minute quota regardless of
how well individual failures are retried afterward. Retrying a failed call
doesn't reduce how many requests were already in flight; if the whole burst
already exceeds the quota, retries just mean the same volume gets resent,
which can still collide with an already-saturated window. Live-verified:
before Layer 2, real burst load reliably produced some 429s even with Layer
1 active; after adding Layer 2, an equivalent run produced zero rate-limit
warnings in the server log (`REPOSITORY_AUDIT.md` §9m). The tradeoff is
latency — a 40-call burst under a 20/min cap takes about 2 minutes instead of
finishing fast with roughly half the calls failing. Correct tradeoff for a
background job the client polls (`/status/{job_id}`), not something with a
synchronous timeout.

**Known gap**: `_call_with_rate_limit_retry()` doesn't currently distinguish
a per-minute rate limit (worth waiting ~60s) from a per-*day* quota
exhaustion (Gemini's free tier: 20 requests/day for `gemini-2.5-flash`,
confirmed live via the actual 429 body's
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` quota metric) — both get
the same bounded-but-still-long retry treatment. Not urgent since Groq is
the account's active provider, but worth hardening if Gemini use increases.

## `FallbackLLMProvider`

`LLM_PROVIDER=fallback` wraps multiple providers (tries each in order) —
implemented but not exercised by this session's verification; treat as
unverified if depended on.
