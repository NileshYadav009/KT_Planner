# PHASE 2 — DEAD CODE REPORT

> Each entry cites the symbol and proves it is unreachable from the live request path
> (`POST /upload` → `process_upload_task` → `GET /status`). Verified against current head.

## D1 — `ai.py` legacy classification layer (entire module partially dead)

| Symbol | Location | Evidence of dead |
|---|---|---|
| `assess_audio_quality()` | ai.py:208 | grep across main.py / context_mapper.py: no caller. Only self-referential. |
| `validate_sentence_quality()` | ai.py:279 | No caller in live path. |
| `classify_with_confidence()` | ai.py:250 | No caller. |
| `estimate_transcription_accuracy()` | ai.py:108 | No caller. |
| `analyze_transcript()` | ai.py:537 | Called only by `generate_report` (ai.py:744) and `KTSessionAggregator.add_transcript` (ai.py:712). Neither is called from main.py. |
| `classify_transcript()` | ai.py:520 | Imported in main.py:14 but **never invoked** there (the only reference is commented-out screenshot code at main.py:357). |
| `generate_report()` | ai.py:743 | Called only by `test_pipeline.py:249` (test) and itself. Not in live path. |
| `KTSessionAggregator` | ai.py:706 | No endpoint exposes it. No caller. |
| `process_coverage_with_gemini()` | ai.py:433 | No caller anywhere. |
| `chunk_text()` | ai.py:374 | Self-documented "no longer used"; no caller. |

**Live symbols from ai.py:** `gemini_refiner`, `GEMINI_ENABLED`, `warmup_models`, `get_sentence_model`, `get_section_embeds`, `SECTION_HINTS`, `build_section_paragraphs`, `map_analysis_to_fields`, `summarize_coverage`, `explainability_logs`, `[NEW] infer_open_risks`.

## D2 — Three parallel classification engines, only one live

| Engine | Location | Status |
|---|---|---|
| `ContextClassifier` (bge + cross-encoder + topic memory) | context_mapper.py:413 | ✅ **LIVE** — used by `ContextMappingPipeline` |
| `EnterpriseSemanticMapper` (MiniLM + clause split + registry) | enterprise_semantic_mapper.py:680 | ◐ Partial — used only by `build_section_paragraphs` for paragraph reconstruction, **re-classifies from scratch** ignoring Stage 3 |
| `ai._classify_transcript_sentences` (MiniLM) | ai.py:495 | ❌ Dead — only feeds dead `analyze_transcript` |

**Duplicate classification** = two independent models (bge-large vs all-MiniLM-L6-v2) producing two independent section mappings for the same transcript, neither reconciled.

## D3 — Duplicate logic

| Logic | Copy 1 | Copy 2 | Impact |
|---|---|---|---|
| Schema load | main.py:43 (`json.load kt_schema_new.json`) | ai.py:169 (same) | Two reads; can diverge if file changes between them |
| Section hints build | context_mapper.py:483 (`index_schema`) | ai.py:176 (`_build_section_hints`) | Built twice, two representations |
| Coverage status | context_mapper.py:1173 (SCS) | main.py never recomputes — but `serialize_kt` (context_mapper.py:2300) emits a *different* shape than main.py:262 | Two coverage shapes for same data |
| Grammar fix | context_mapper.py:869 (`_basic_grammar_fix`) | enterprise_semantic_mapper.py:373 (`_local_professionalize`) | Near-identical; two sources of truth |
| `confidence=0.85` literal | context_mapper.py:872 (RepairAction) | (coincides with audio-conf value, see Phase 3) | Misleading — looks intentional, is accidental |

## D4 — Unreachable / stub code

| Symbol | Location | State |
|---|---|---|
| `_infer_from_context()` | context_mapper.py:904 | Function body is a no-op (`pass` after comment "stick with original") |
| OCR extraction branch | context_mapper.py:1322 | `pass` block, never implemented |
| Screenshot capture | main.py:308-367 | Entirely commented out |
| `merge_incremental_kt()` | context_mapper.py:2367 | Implemented, no caller, no endpoint |
| `apply_human_feedback()` | context_mapper.py:2449 | "For now, just record the feedback" — mutates nothing structural |
| `templates.py` router | templates.py (whole file) | Never `app.include_router()` in main.py |
| PII anonymization | pii_anonymizer.py:201 | `anonymize_transcript` returns original text; redaction commented out |
| `devops_transcription.clean_transcript` PII step | devops_transcription.py:450 | Commented out |

## D5 — Abandoned experiment markers

- `TOPIC_BLOCKS_FIX.md`, `archive/TODO.md`, `demo_topic_blocks.py`, `diagnostics_runner.py` — leftover from prior iteration; `diagnostics_runner.py` is not imported anywhere.
- `__pycache__/*.pyc` tracked in git (visible in branch diff) — should be gitignored.

## Summary count
- **Fully dead functions (ai.py):** 10
- **Partially-dead engines:** 1 (EnterpriseSemanticMapper used only for paragraph text, classification half dead)
- **Duplicate logic pairs:** 5
- **Stubs / commented-out paths:** 7
