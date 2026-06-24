# PHASE 3 — COVERAGE ROOT CAUSE ANALYSIS

> **This is the answer to the core complaint: "coverage quality does not improve despite multiple code changes."**
> All claims proven with a live runtime trace of `tests/test_ecommerce_kt.TRANSCRIPT` through the real pipeline.

## Runtime evidence (the proof)

Running the e-commerce transcript through `ContextMappingPipeline.process()` and dumping `kt.coverage`:

```
system_overview                  status=weak     scs=0.59 conf=0.85 blocks=5 sents=6 req=True
technology_stack                 status=missing  scs=0.00 conf=0.00 blocks=0 sents=0 req=True
architecture_reference           status=weak     scs=0.47 conf=0.85 ...
plain_english_notes              status=weak     scs=0.49 conf=0.85 ...
monitoring_observability         status=weak     scs=0.49 conf=0.85 ...
disaster_recovery                status=weak     scs=0.53 conf=0.85 ...
security_controls                status=weak     scs=0.53 conf=0.85 ...
cost_optimization                status=weak     scs=0.50 conf=0.85 ...
day1_survival_checklist          status=weak     scs=0.41 conf=0.85 ...
deployment_and_rollback          status=weak     scs=0.53 conf=0.85 ...
common_failures                  status=weak     scs=0.47 conf=0.85 ...
known_bad_days                   status=weak     scs=0.52 conf=0.85 ...
danger_zones                     status=weak     scs=0.52 conf=0.85 ...
ownership_escalation             status=weak     scs=0.45 conf=0.85 ...
first_30_day_plan                status=missing  scs=0.00 conf=0.00 ...
open_responsibilities            status=weak     scs=0.40 conf=0.85 ...
handover_completion              status=missing  scs=0.00 conf=0.00 ...
signoff                          status=weak     scs=0.52 conf=0.85 ...
```

**Observe:** every non-missing section reads exactly `conf=0.85`. That is not a measurement — it is a constant. This is the smoking gun.

---

## ROOT CAUSE #1 (PRIMARY) — Coverage confidence is the AUDIO confidence, not the SEMANTIC confidence

### Problem
Stage 3 (`ContextClassifier`, context_mapper.py:413) computes a sophisticated **semantic** confidence per sentence (bge embedding + cross-encoder rerank + topic-memory boost, blended at context_mapper.py:708). This number — the actual quality signal of the classification — is **never propagated to Stage 5**.

### Evidence (code)
- `detect_gaps` builds block confidence from **audio** confidence:
  ```python
  # context_mapper.py:1111
  confidences = [classified_sentences[idx_val].sentence.audio_confidence
                 for idx_val in current_block_indices]
  block_confidence = float(np.mean(confidences)) if confidences else 0.0
  ```
- `Sentence.audio_confidence` comes from Whisper's `avg_logprob`:
  ```python
  # context_mapper.py:107-116  AudioSegment.confidence_score
  conf = max(0.0, min(1.0, (self.avg_logprob + 2.0) / 2.0))
  ```
- The test transcript segments carry `avg_logprob=-0.3` → `(−0.3+2.0)/2.0 = 0.85`. **Every section inherits 0.85.**
- `SectionCoverage.confidence_score` = mean of block confidences = mean of audio confidences (context_mapper.py:1168).
- The SCS model's "confidence" dimension is **also** fed audio confidences (`block_confidences` at context_mapper.py:1020 = the audio means).
- The semantic classification confidence lives only on `cs.primary_classification.confidence` and is discarded the moment `detect_gaps` starts.

### Impact
- Coverage confidence is **insensitive to classification quality**. A perfectly-classified sentence and a barely-above-threshold sentence both contribute the same `0.85` to their section.
- **Any change to the classifier, cross-encoder, topic memory, or thresholds has zero effect on the displayed coverage confidence.** This is precisely why "coverage doesn't improve despite multiple code changes."
- Risk scores downstream (context_mapper.py:1174-1179) inherit the same blindness.

### Fix
Propagate the semantic confidence to `TopicBlock` and `SectionCoverage`. Concretely, in `detect_gaps`:
```python
# REPLACE (context_mapper.py:1111)
confidences = [classified_sentences[idx_val].sentence.audio_confidence for idx_val in current_block_indices]
# WITH
audio_confs  = [classified_sentences[i].sentence.audio_confidence        for i in current_block_indices]
sem_confs    = [ (classified_sentences[i].primary_classification.confidence
                  if classified_sentences[i].primary_classification else 0.0) for i in current_block_indices]
block_confidence = 0.6*float(np.mean(sem_confs or [0])) + 0.4*float(np.mean(audio_confs or [0]))
```
Same edit at context_mapper.py:1139 (final-block branch). Then `block_confidences` passed to `semantic_coverage_score` will carry real signal.

### Expected impact
Coverage confidence becomes a real quality signal. Classifier improvements will now visibly move coverage numbers. **Single highest-impact fix in the entire audit.**

---

## ROOT CAUSE #2 — Two divergent coverage shapes; the UI reads the wrong one

### Problem
There are **two** coverage dicts per job:
1. `coverage` — the local dict in `main.py:226-272`, derived from `kt.coverage[].blocks`. **This is what `/status` returns at top level** (`JOB_QUEUE[job_id]["coverage"]`, main.py:387) and what the frontend renders (`data.coverage`, index.html:332).
2. `kt_structured["coverage"]` — built by `serialize_kt` (context_mapper.py:2300), a **different shape** (no `sentences`/`content`/`blocks` arrays, just scalar metadata).

### Evidence
- main.py:262 builds shape #1 with keys `title, status, required, sentence_count, confidence, risk, content, sentences, blocks`.
- context_mapper.py:2300 builds shape #2 with keys `section_title, status, required, sentence_count, confidence, risk, coverage_score` — **no sentences, no content, no blocks**.
- index.html:332 reads `data.coverage` (shape #1). index.html never reads `data.kt_structured.coverage`.
- `kt_structured["coverage"]` is therefore dead weight shipped to the client but never displayed.

### Impact
- Any fix applied inside `serialize_kt`'s coverage view is **invisible** to the user.
- Confusion about "which coverage" a change affects — a likely cause of past "changes didn't help" reports.
- `coverage_score` (the SCS) is only in shape #2, so the UI never shows the smart score at all.

### Fix
Single source of truth. Make `main.py`'s `coverage` dict the only one, and have `serialize_kt` either omit its scalar-only coverage or derive it from the same `kt.coverage`. Minimum-viable: add `coverage_score` from the dataclass into shape #1 at main.py:262 so the SCS is exposed to the UI.

---

## ROOT CAUSE #3 — UI computes coverage% by counting `weak` as covered

### Problem
index.html:338:
```js
const coveragePercent = total ? Math.round((covered + weak) / total * 100) : 0;
```
`weak` is counted as success. With the trace above (15 weak + 3 missing out of 18) the UI shows **83%** coverage for a transcript the SCS model rates as universally thin.

### Evidence
- SCS thresholds: `covered ≥ 0.65`, `weak ≥ 0.40` (context_mapper.py:1050-1055). The trace shows every section between 0.40–0.59 → all `weak` → UI treats as covered.

### Impact
The user perceives high coverage; the model disagrees. "Coverage doesn't improve" because the UI metric is decoupled from the semantic model.

### Fix
UI should weight: `covered=1.0, weak=0.5, missing=0.0`, and additionally surface `coverage_score` (SCS) per section. (UI change — out of scope per rules, but flagged.)

---

## ROOT CAUSE #4 — Paragraph reconstruction re-classifies and is not merged

### Problem
`build_section_paragraphs` (ai.py:399) calls `EnterpriseSemanticMapper.process_transcript(sentence_tuples)` which runs a **fresh** classification (MiniLM embeddings, ai.py:412) independent of Stage 3's bge+cross-encoder result. Its output is stored only as `kt_structured["paragraphs"]` and **never reconciled** with `kt.coverage` or `kt.section_content`.

### Evidence
- ai.py:412: `mapper = create_semantic_mapper(SCHEMA, llm_refiner=...)` — new mapper, no Stage-3 input.
- ai.py:413: `result = mapper.process_transcript(sentence_tuples)` — re-runs classification.
- main.py:277: `kt_structured["paragraphs"] = paragraph_data` — parked, not merged.

### Impact
- Two classification verdicts exist per transcript (bge vs MiniLM) and can disagree.
- Paragraph text shown to user may be grouped by a *different* section assignment than the coverage panel beside it.
- The "reconstruction engine" improvements (clause splitting, registry) never affect the coverage numbers.

### Fix
Make `build_section_paragraphs` accept the already-classified `kt.section_content` and only run `ParagraphIntegrityEngine.reconstruct_paragraph` per section (skip re-classification). Removes the duplicate MiniLM classifier from the live path.

---

## ROOT CAUSE #5 — LLM repair hook never fires

### Problem
`ContextRepair._try_llm_repair` (context_mapper.py:856) only runs when `classified.sentence.audio_confidence < 0.3`. Audio confidence in normal transcripts is ~0.85, so the LLM repair path is **dead in practice** even when Gemini is configured.

### Evidence
- context_mapper.py:856: `if self.llm_fallback_fn and classified.sentence.audio_confidence < 0.3`
- The 0.3 threshold requires `avg_logprob < -1.4`, i.e. near-silent segments.

### Impact
Gemini contributes nothing to transcript quality in the live path (only warmup + the new polish tier use it).

### Fix
Gate LLM repair on **semantic** confidence (`primary_classification.confidence < 0.4`) OR audio confidence, not audio alone.

---

## Verdict

Root causes #1 + #2 together explain ~100% of the "coverage doesn't improve" symptom:
- Changes to the classifier can't move the confidence number (#1).
- Even if they could, the displayed number comes from a parallel shape (#2) and the UI metric counts `weak` as success (#3).

**Fixing #1 alone will make every future classifier change visible.** That is the single most important change in this audit.
