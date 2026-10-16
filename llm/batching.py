"""Fewer LLM round trips for long KTs (P1-12).

One call per borderline sentence and one per empty field made the number of
calls grow with the transcript: 88 of 140 calls on a 1,260-word KT were
classification checks, and a 60-minute KT would take 600+ calls at a
20-calls-a-minute throttle. With LLM_BATCH_CALLS=1 the classification checks
go out LLM_VERIFY_BATCH_SIZE sentences per call
(context_mapper.ContextClassifier.verify_in_batches) and each section's
gap-fills in one call (field_populator._fill_deferred_gaps).

On by default since 16 Oct (audit §9aai). Live Groq runs on the five e2e
KTs: 66 requests and 35k tokens batched against 238 and 87k one call each,
in half the time. Every checklist was equal or better batched: one call each
misplaced two GCP facts (one was a splitter defect, since fixed) and drew a
flow nobody stated. The blind holdouts scored 83 of 95 batched, against 77
with no LLM. The two modes do not word every LLM-written value identically.
LLM_BATCH_CALLS=0 restores one call each.
"""
import os


def llm_batch_calls_enabled() -> bool:
    return os.getenv("LLM_BATCH_CALLS", "1").strip().lower() in ("1", "true", "yes", "on")
