# PHASE 1 — SYSTEM FLOW

> Verified against `feature/evolution_setup` head. Every node cites the code that implements it.

## End-to-end request flow

```
POST /upload (main.py:393)
   │  writes uploaded bytes to temp file, creates job_id, queues background task
   ▼
process_upload_task (main.py:142)                          ← runs in FastAPI BackgroundTasks thread
   │
   ├─ ffmpeg: extract WAV 16kHz mono (main.py:154)
   ├─ ffmpeg: silenceremove trim        (main.py:168)
   │
   ├─ WhisperModel.transcribe           (main.py:183)      ← BLOCKING, no timeout, single-thread
   │      → segments[{text,start,end,avg_logprob,...}]
   │
   ├─ clean_transcript per segment      (main.py:193)      ← devops_transcription.py
   │
   ▼
MAPPER_PIPELINE.process(job_id, transcript, segments)     (main.py:218)
   = ContextMappingPipeline.process  (context_mapper.py:1793)
   │
   │  Stage 1   AudioSegment construction          (context_mapper.py:1811)
   │  Stage 1.5 clean segments                     (context_mapper.py:1823)
   │  Stage 2   segment_sentences + semantic_chunk (context_mapper.py:1826, 145, 303)
   │  Stage 3   ContextClassifier (embed+cross-enc+topic-mem) (context_mapper.py:1864-1973)
   │  Stage 3.5 apply_rule_overrides               (context_mapper.py:1976)  ← section_rules.py
   │  Stage 3.6 policy gating (CONFIDENCE_ACCEPT_THRESHOLD=0.42)
   │  Stage 4   ContextRepair (grammar/glossary/LLM)        (context_mapper.py:2066)
   │  Stage 5   detect_gaps → TopicBlocks + SCS coverage    (context_mapper.py:2076, 1069)
   │  Stage 6   extract_urls_and_assets                     (context_mapper.py:2080)
   │  Stage 7   assemble_kt → StructuredKT                  (context_mapper.py:2087)
   │
   ▼  returns StructuredKT  (kt)
   │
   ├─ build_section_paragraphs(transcript)          (main.py:222, ai.py:399)
   │      ⚠ INDEPENDENT re-classification via EnterpriseSemanticMapper (ai.py:412)
   │        → kt_structured["paragraphs"]           (NOT merged back into coverage)
   │
   ├─ build user-visible coverage dict              (main.py:226-272)
   │      reads kt.coverage[].blocks sentences; sets status/confidence from dataclass
   │
   ├─ serialize_kt(kt) → kt_structured              (main.py:276)
   │      produces SEPARATE kt_structured["coverage"] view (context_mapper.py:2300)
   │
   ├─ [NEW] polish_all_sections → kt_structured["polished_sections"]  (main.py:285)
   ├─ [NEW] infer_open_risks      → kt_structured["open_risks"]       (main.py:297)
   │
   ├─ map_analysis_to_fields(field_analysis)        (main.py:381)   ← uses local `coverage`, not kt_structured
   │
   ▼
JOB_QUEUE[job_id] = { coverage, kt_structured, mapped_fields, ... }   (main.py:384)
   │
   ▼
GET /status/{job_id}  (main.py:424)  → returns the whole job dict
   │
   ▼
static/index.html reads data.coverage (NOT data.kt_structured.coverage)
   computes coverage% = (covered + weak) / total        (index.html:338)
```

## Key flows (sub-pipelines)

### Coverage extraction flow
```
Stage 3  ContextClassifier.classify_sentence  →  cs.primary_classification.confidence   (semantic score)
                                                        │
                                                        ▼  DISCARDED — never stored on the sentence for Stage 5
Stage 5  detect_gaps  builds TopicBlock.confidence_score from sentence.audio_confidence   (context_mapper.py:1111)
         semantic_coverage_score(block_sentences, block_confidences=audio, sec)           (context_mapper.py:1164)
              → status ∈ {missing, weak, covered}
         SectionCoverage.confidence_score = mean(block.confidence_score) = AUDIO mean      (context_mapper.py:1168)
```
**The semantic classification confidence computed in Stage 3 is not used by Stage 5.** See Phase 3.

### Section mapping flow
```
sentence → _score_sentence_candidates   (context_mapper.py:604)
             base_sim (embed) + context_sim + keyword_boost + overview_penalty
         → _rerank_candidates            (context_mapper.py:689)
             cross-encoder top-5 + rule-match reinsert
         → TopicMemory.get_topic_boost   (context_mapper.py:1949)
         → apply_rule_overrides          (context_mapper.py:1976 / section_rules.py:304)
         → policy gate (0.42)            (context_mapper.py:2022)
```

### Reconstruction flow (two parallel paths)
```
PATH A (live, feeds UI coverage):    TopicBlocks from Stage 5 → kt.coverage → main.py coverage dict
PATH B (live, feeds kt_structured):  build_section_paragraphs → EnterpriseSemanticMapper.process_transcript
                                      (re-classifies from scratch, ignores Stage 3 results)
PATH C (live, NEW):                  polish_all_sections → kt_structured["polished_sections"]
PATH D (dead):                       ai.generate_report / analyze_transcript / KTSessionAggregator
```

### Gemini flow
```
ai.py:64   GEMINI_CLIENT = genai.Client(api_key=...)   (startup)
ai.py:153  warmup_models() sends "hello"               (startup cost, no downstream use)
ai.py:73   gemini_refiner(prompt)                       ← stop_sequences=["\n\n"]  TRUNCATES multi-line
   consumed by:
     • build_section_paragraphs → mapper llm_refiner    (ai.py:411)  → kt_structured["paragraphs"]
     • ContextRepair._try_llm_repair                    (context_mapper.py:856)  ← only if audio_conf<0.3 (never fires)
     • [NEW] kt_gemini_polisher (calls gemini_fn directly)  → kt_structured["polished_sections"]
```
