"""Fewer LLM round trips for long KTs (P1-12).

One call per borderline sentence and one per empty field made the number of
calls grow with the transcript: 88 of 140 calls on a 1,260-word KT were
classification checks, and a 60-minute KT would take 600+ calls at a
20-calls-a-minute throttle. With LLM_BATCH_CALLS=1 the classification checks
go out LLM_VERIFY_BATCH_SIZE sentences per call
(context_mapper.ContextClassifier.verify_in_batches) and each section's
gap-fills in one call (field_populator._fill_deferred_gaps).

Off by default until LLM runs on the golden set show the same placements and
values as one call each.
"""
import os


def llm_batch_calls_enabled() -> bool:
    return os.getenv("LLM_BATCH_CALLS", "0").strip().lower() in ("1", "true", "yes", "on")
