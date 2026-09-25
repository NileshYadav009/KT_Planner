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
| `LLM_MAX_CALLS_PER_MINUTE` | Proactive throttle ceiling, **requests**/min per provider | `20` |
| `LLM_MAX_TOKENS_PER_MINUTE` | Proactive throttle ceiling, **tokens**/min per provider. `0` = off (opt-in; see below) | `0` |

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

## Rate-limit handling: three layers, and why each exists

**Layer 1 — reactive retry** (`_call_with_rate_limit_retry()`): on a 429,
honors the provider's own suggested delay (`Retry-After` header for Groq,
the `retryDelay` field embedded in Gemini's error body) with a capped
exponential-backoff fallback, up to `LLM_RETRY_MAX_ATTEMPTS` times.

**Layer 2 — proactive request throttle** (`_throttle()`): a sliding-window
limiter that caps outbound requests *before* they're sent, shared
process-wide across all concurrent jobs (one bucket per real quota — Gemini's
two internal labels share a bucket since both hit the same API; Groq is
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

**Layer 3 — proactive token throttle** (the `estimated_tokens` half of
`_throttle()`): the same sliding 60s window, metering **tokens** instead of
requests. Layers 1 and 2 were both present and healthy when a real Groq
account still failed every call on a ~2,600-word transcript: that tier
allows 1,000 requests per *day* but only **8,000 tokens per minute**, and a
KT run's prompts each carry a slice of the transcript, so 20 requests/minute
of multi-thousand-token prompts is several times over a budget a request
counter is structurally unable to see. The result was 23 consecutive
rate-limit retries and zero completed extractions — hours of backoff, with
the content those calls would have produced lost anyway
(`REPOSITORY_AUDIT.md` §9qq).

`estimate_tokens()` costs each call before sending it, at a deliberately
conservative 3.5 chars/token, and **includes `max_output_tokens`** — a
TPM-metered provider charges for the completion too, so costing only the
prompt under-reserves by exactly the output budget requested.

The estimate alone is not sufficient, though: it under-counts jargon-dense
technical prose (acronyms and punctuation tokenize worse than the English
average), and a few percent per call *accumulates* across a window until the
window really is over budget — measured live as 35 retries over ~30 minutes.
So `reconcile_token_usage()` books the shortfall from each completed call's
own `usage.total_tokens`. Estimate error becomes self-correcting rather than
cumulative. The reverse is deliberately not applied — an over-estimate is
never refunded — because inflating trust in the heuristic is exactly how the
estimate-only version failed.

Do **not** reconcile against `x-ratelimit-remaining-tokens` instead: that
header is the provider's current bucket headroom, which refills continuously
and recovers within seconds while the local 60s window still holds the
reservation. `budget - remaining` is therefore not comparable to the window's
total, and the correction silently never fires (tried, measured, reverted —
`REPOSITORY_AUDIT.md` §9qq).

**This layer is opt-in: set `LLM_MAX_TOKENS_PER_MINUTE` to enable it.** The
default `0` means no token ceiling, so behaviour matches what it was before
this layer existed.

`_note_token_limit()` still records `x-ratelimit-limit-tokens` when a provider
sends it (Groq does, on 429s), and logs it — useful for deciding what to set
the env var *to* — but a learned value deliberately does **not** switch
throttling on by itself. An earlier version did auto-activate from that
header, and it was wrong: on the account that motivated this code,
tokens-per-minute was never the binding limit (the real one was
tokens-per-**day**), so auto-activation added latency to every run to guard a
ceiling that was never being hit. Learning that a bucket exists is not
evidence it is the constraint. A throttle that slows real work down should be
switched on by someone who measured that they need it.

A prompt larger than the entire per-minute budget can never "fit", so the gate
only applies to a non-empty window — such a call is sent and Layer 1 handles
the rejection, rather than blocking the run forever.

## Per-day quotas fail fast (this gap was real, and it bit)

`_call_with_rate_limit_retry()` distinguishes a per-minute rate limit (worth
waiting) from a per-**day** quota exhaustion (not worth waiting — no retry in
this loop can clear it). `_is_daily_quota_error()` recognizes one from the
provider's wording (`tokens per day`, `(TPD)`, and Gemini's
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, which lowercases to
contain `perday`) or from a suggested `Retry-After` far beyond
`LLM_RETRY_MAX_DELAY_SECONDS` — a delay no retry could satisfy is not
transient regardless of wording. On a match it re-raises immediately, and the
caller's existing fallback produces the deterministic result at once.

This was previously listed here as a non-urgent "known gap". It then cost a
whole session: a Groq tier with **200,000 tokens/day** refused every call
while the per-minute bucket sat completely full, and each refusal was retried
5 × 60s, so a KT run ground for over an hour and produced nothing
(`REPOSITORY_AUDIT.md` §9qq).

**The retry warning logs the provider's own message**, and must keep doing so.
It previously logged only `"rate-limited, retrying in 60s"`, which made the
above undiagnosable from logs alone and sent three rounds of fixes at the
wrong dimension. A rate-limit log without the provider's text is not
actionable.

## `FallbackLLMProvider`

`LLM_PROVIDER=fallback` wraps multiple providers (tries each in order) —
implemented but not exercised by this session's verification; treat as
unverified if depended on.
