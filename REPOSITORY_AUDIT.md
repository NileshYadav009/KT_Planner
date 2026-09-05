# REPOSITORY AUDIT — Continuum KT Planner

Date: 2026-09-03
Branch audited: `feature/Dynamic_Schema_Builder` (includes uncommitted working-tree changes)
Method: full read of every non-vendor `.py` file, cross-referenced with `git grep`/`Grep` for actual callers — no claim below is asserted without a verified import or call site.

> Prior audit docs (`DEAD_CODE_REPORT.md`, `SYSTEM_FLOW.md`, `CODE_ORGANIZATION.md`) exist from an earlier pass on a different branch head (`feature/evolution_setup`). Their line numbers are stale and several of their "dead" verdicts are now **wrong** because a new layer (dynamic schema → field populator → knowledge object → section renderers → PDF export) was built on top of the pipeline they described. This document supersedes them for current `main.py`. Where they're still correct I say so explicitly.

---

## 1. What this product currently does (verified, not aspirational)

```
POST /upload  →  job queued, background task starts
  │
  ├─ ffmpeg: convert to WAV → trim silence
  ├─ faster-whisper transcribe → segments
  ├─ devops_transcription.clean_transcript() per segment (vocabulary fixes, filler removal)
  │
  ├─ ContextMappingPipeline.process()          [context_mapper.py, 7-stage pipeline]
  │     → StructuredKT (kt): sentences, coverage-by-section, topic blocks, risk scores
  │
  ├─ build_section_paragraphs(transcript)      [ai.py → enterprise_semantic_mapper.py]
  │     → RE-CLASSIFIES the transcript from scratch (2nd independent model pass)
  │     → stored as kt_structured["paragraphs"]   ⚠ see §4.1 — not read by anything
  │
  ├─ coverage dict assembled from kt.coverage[].blocks (main.py:549-598)
  │
  ├─ polish_coverage_sections()                [ai.py, batched Gemini/Groq call]
  │     → rewrites each section's raw fragments into 1 clean paragraph, 1 LLM call total
  │
  ├─ _extract_structured_section() for "monitoring" only [ai.py, NEW on this branch]
  │     → 1 extra LLM call, asks for strict JSON {tools, first_response_steps, alert_routing}
  │
  ├─ generate_dynamic_schema(coverage, SCHEMA)  [schema_generator.py]
  │     → drops sections with no evidence, adds a handful of conditional fields
  │       (e.g. "cache_layer" field only if Redis was mentioned)
  │
  ├─ populate_fields(dynamic_schema, coverage)  [field_populator.py]
  │     → per field: regex pattern → semantic-similarity sentence match → LLM gap-fill → give up
  │
  ├─ build_knowledge_object(...)                [knowledge/knowledge_builder.py]
  │     → canonical per-section dict: facts, entities, evidence, relationships, fields
  │       (knowledge/{facts,entities,evidence,relationships}.py are thin, mostly reshape populated_fields)
  │
  ├─ build_rendered_sections(knowledge_object)  [main.py]
  │     → for each section: try its dedicated renderer (renderers/sections/*.py);
  │       if it produces nothing meaningful, fall back to a generic narrative block
  │
  └─ JOB_QUEUE[job_id] = {status, coverage, knowledge_object, kt_structured, mapped_fields, ...}

GET /status/{job_id}        → whole job dict (frontend polls this)
GET /schema/{job_id}        → dynamic_schema + populated_fields + knowledge_object
GET /export/pdf/{job_id}    → Jinja2 template + WeasyPrint, using knowledge_object.rendered_sections
POST /feedback               → mutates coverage in place (reclassify a sentence to another section)
POST /semantic-placement     → standalone endpoint, calls ai.generate_report() (a 3rd, independent
                                classification path — see §4.2)
```

**Two document representations exist for every job**, and only one reaches the PDF:
- **Live path**: `coverage` → `dynamic_schema` → `populated_fields` → `knowledge_object.rendered_sections` → PDF. This is the actual source of truth today, closer to what the 26-phase brief calls for than the old docs suggest.
- **Legacy path**: `kt_structured` (via `serialize_kt`) with `kt_structured["paragraphs"]` from a second, independent semantic re-classification (`enterprise_semantic_mapper.py`). Computed on **every** upload, returned by `/status/{job_id}`, but **not read by `static/index.html`** (verified: zero references to `kt_structured` or `paragraphs` in the frontend) and not used by PDF export. This is pure wasted compute + a second LLM-adjacent model load per job.

---

## 2. File inventory by role

| Role | Files | Status |
|---|---|---|
| **API/orchestration** | `main.py` (908 lines) | Live, actively edited this branch |
| **Transcription cleanup** | `devops_transcription.py` | Live — vocabulary correction, filler removal, fuzzy term matching |
| **7-stage context pipeline** | `context_mapper.py` (2315 lines) | Live — classification, repair, gap detection, asset extraction, assembly |
| ↳ supporting | `policy.py`, `runtime_policy.py`, `entity_extractor.py` (GLiNER, optional dep not in requirements.txt — degrades to no-op), `section_rules.py` | Live, imported by context_mapper.py |
| **LLM orchestration / polish** | `ai.py` (1090+ lines) | **Mixed** — see §4 for live vs dead symbol list |
| **2nd classification engine** | `enterprise_semantic_mapper.py` | Live but only feeds the unused `kt_structured["paragraphs"]` path (§4.1) |
| **LLM provider abstraction** | `llm_provider.py` | Live. Already close to Phase 16/17's ask: `GeminiProvider`, `GroqProvider`, `FallbackLLMProvider`, all behind one `LLMProvider` interface. Provider choice is env-driven (`LLM_PROVIDER`), not automatic primary→fallback→local unless `LLM_PROVIDER=fallback` is set explicitly. |
| **Dynamic schema** | `schema_generator.py` | Live — coverage-gated section inclusion + conditional field injection from tech-stack regex hits |
| **Field extraction** | `field_populator.py` | Live — pattern → semantic → LLM cascade per field, actively being refined this branch |
| **Knowledge model** | `knowledge/{__init__,knowledge_builder,facts,entities,evidence,relationships}.py` | Live — this *is* Phase 6/7/8's "structured facts + evidence" model, already built, just thinner than the brief describes (facts/entities/relationships mostly reshape `populated_fields` rather than doing independent NLP extraction) |
| **Section renderers** | `renderers/sections/*.py` (16 files) + `renderers/blocks/*.py` (8 block builders) + `renderers/base.py` | Live — this *is* Phase 12's "section-specific renderers," already implemented per-section (system_overview, monitoring, deployment, security, ownership_escalation, disaster_recovery, cost_optimization, danger_zones, known_bad_days, open_responsibilities, handover_completion, first_30_day_ownership, common_failures, architecture, architecture_reference, day1) |
| **PDF export** | `main.py: render_pdf_html/export_pdf` + `pdf/templates/kt_document.html` + `.css` | Live — Jinja2 + WeasyPrint, matches Phase 14's recommended approach already |
| **PII handling** | `pii_anonymizer.py` | **Present but disabled** — its call site in `devops_transcription.clean_transcript` is commented out (§4.4) |
| **Frontend** | `static/index.html` (1724 lines), `static/index_old.html` (927 lines), `static/review.html` (39 lines) | `index.html` live; `index_old.html` appears to be a superseded prior version (not referenced from any route or the live HTML) |
| **Root-level scripts** | `upload_and_get_kt.py`, `generate_audio.py`, `check_kt.py` | Stale duplicates — see §5.3 |
| **Scripts (correct location)** | `scripts/upload_and_get_kt.py`, `scripts/generate_audio.py`, `scripts/check_kt.py`, `scripts/debug_glossary.py` | Live-ish dev tooling |
| **Root-level ad hoc tests** | `test_pipeline.py`, `test_coverage_extraction.py`, `test_presidio_integration.py`, `demo_topic_blocks.py`, `demo_coverage_polish.py` | Manual/demo scripts, not pytest-discovered, inconsistent with `tests/` |
| **Formal tests** | `tests/{run_test_pipeline,test_ecommerce_kt,test_dynamic_schema,test_pdf_export}.py` | Live, pytest-style |
| **Schema config** | `kt_schema_new.json` (17 sections) | Live, loaded once at import time in `main.py` |
| **Docs** | 15+ root-level `.md` files, `archive/` (already contains 9 old docs + old audio + old log) | See §6 |

---

## 3. Duplicate implementations (verified, safe to consolidate)

| Pair | Winner (imported) | Loser (orphaned) | Evidence |
|---|---|---|---|
| `renderers/sections/system_overview.py` (35 lines) vs `renderers/system_overview.py` (25 lines) | `sections/` version — `renderers/sections/__init__.py:3` | top-level file | `renderers/__init__.py` only does `from renderers.sections import get_renderer`; no import anywhere reaches `renderers.system_overview` |
| Same pattern for `deployment.py`, `ownership.py`, `day1.py`, `security.py` | `renderers/sections/*` | `renderers/{deployment,ownership,day1,security}.py` | Same grep — zero references outside their own files |
| `renderers/sections/failures.py` (29 lines, defines nothing imported) | `renderers/sections/common.py::render_failures` (line 35) is what's actually registered for `common_failures` | `renderers/sections/failures.py` | grep for `sections.failures` / `render_failures` shows only the `common.py` definition ever used |
| `knowledge_builder.py` (root, 2-line shim: `from knowledge import build_knowledge_object`) | `knowledge/knowledge_builder.py` (real implementation) | root shim | `main.py:21` imports `from knowledge import build_knowledge_object` directly — the root shim has no importers anywhere |

**Correction to an earlier draft of this audit**: `upload_and_get_kt.py`, `generate_audio.py`, `check_kt.py` at repo root are **not** duplicated by their `scripts/` namesakes — `scripts/*.py` are thin `importlib.import_module()` wrapper stubs that call into the root files, so the root files are the real implementations and must stay. Only `check_kt.py` had a real problem: it called `GET /kt/{job_id}`, an endpoint that no longer exists — fixed to hit `GET /status/{job_id}` instead (both status update: 2026-09-03).

---

## 4. Dead / disabled code (verified against current `main.py`, not the old report)

### 4.1 `kt_structured["paragraphs"]` + `EnterpriseSemanticMapper` — computed, never read
`build_section_paragraphs()` (`ai.py:503`) runs a full second semantic classification of the transcript via `enterprise_semantic_mapper.py` on every single upload. Its output is stashed into `kt_structured["paragraphs"]` (`main.py:687-688`) and returned by `/status/{job_id}`. **`static/index.html` never reads `kt_structured` or `.paragraphs`** (verified by grep — zero hits). This is a full model pass wasted on every request. Candidate for removal once confirmed no external consumer depends on it (it's not referenced from `/schema/{job_id}` or PDF export either).

### 4.2 Three independent classification paths, one of them fully redundant now
- `ContextClassifier` (`context_mapper.py:415`) — **live**, feeds the real coverage/knowledge-object path.
- `EnterpriseSemanticMapper` (`enterprise_semantic_mapper.py`) — **live but wasted**, see 4.1.
- `ai.analyze_transcript` / `ai.classify_transcript` / `ai.generate_report` (`ai.py:918,901,1124`) — **live**, but only via `POST /semantic-placement`, a side-endpoint that duplicates classification logic independently of the main upload pipeline and isn't wired into the KT document at all. (Correction to the old `DEAD_CODE_REPORT.md`, which called this dead — it now has a live caller, just an isolated one.) `classify_transcript` itself is imported in `main.py:15` but never called outside the commented-out screenshot block — still genuinely dead as an import.
- `KTSessionAggregator` (`ai.py:1087`) — still no caller anywhere. Still dead.

### 4.3 Screenshot capture — fully commented out
`main.py:694-756` — the entire segment→screenshot pipeline is commented out. `static/screenshots/` still holds **141 tracked JPEG files** from when this was active. Dead feature, dead artifacts.

### 4.4 PII anonymization — implemented, wired, but disabled at the call site
`devops_transcription.py:445-458` — the call to `anonymize_transcript_before_classification` is commented out with a note that it "was introducing blocking placeholders." `pii_anonymizer.py` is otherwise a complete, importable module. `test_presidio_integration.py` still exists and exercises `clean_transcript()`, but since the anonymization step is dead, the test no longer actually tests PII redaction — it's silently testing nothing.

### 4.5 `templates.py`
Never `app.include_router()`'d, never imported by `main.py`. Matches old report — still true.

### 4.6 Tracked `__pycache__/*.pyc`
`__pycache__/ai.cpython-313.pyc`, `context_mapper.cpython-313.pyc`, `main.cpython-313.pyc` are tracked in git (shown as `M` in `git status`) despite `.gitignore` already excluding `__pycache__/`. They were committed before the ignore rule existed.

---

## 5. Correctness risk found while auditing (not cosmetic — worth fixing before Phase 3+)

### 5.1 `validate_renderer_registry()` doesn't validate against the schema it's given
`renderers/sections/__init__.py` (new on this branch): the function signature takes `schema_sections`, computes `schema_ids = {sec["id"] for sec in schema_sections}` — **and then never uses `schema_ids`**. The actual check only verifies that a hardcoded set (`sections_with_renderers`) is a subset of `RENDERER_REGISTRY.keys()`. It never checks whether the *schema's own* section ids match a registry key.

This matters because **they currently don't match for one section**: `kt_schema_new.json` has `"id": "first_30_day_plan"`, but `RENDERER_REGISTRY` (and the file `renderers/sections/first_30_day_ownership.py`) key it as `"first_30_day_ownership"`. `get_renderer("first_30_day_plan")` silently falls through to `render_default`, producing generic narrative text instead of the dedicated checklist-style renderer — for every KT generated today. The startup assertion this branch just added does not catch it, because it validates the wrong thing.

**Fix is a one-line rename** (either the schema id or the registry key) plus fixing the validator to actually check `schema_ids` against `registry_keys` for every section that has a `renderers/sections/<id>.py` file. Flagging this now since it's a live rendering bug, not a hypothetical.

### 5.2 Two coverage/status shapes for the same data
`main.py`'s local `coverage` dict (used for schema generation, field population, knowledge object) and `context_mapper.serialize_kt()`'s `kt_structured["coverage"]` are built independently from the same `kt.coverage` dataclasses, with different field names and status logic. Only the first is live; the second is dead weight per §4.1. Once §4.1 is removed this resolves itself.

---

## 6. Documentation sprawl

15 markdown files at repo root plus `archive/` (which already holds 9 older docs — a previous cleanup pass). Several actively contradict each other or the current code:
- `DOCUMENTATION_INDEX.md` links to endpoints (`/expert-correction`, `/training-stats`, `/quality-report/{job_id}`, `/enterprise-status`) that **do not exist** in current `main.py`.
- `REQUIREMENTS_AUDIT.md` documents endpoints (`/kt/{job_id}`, `/coverage/{job_id}`, `/explainability/{job_id}`, `/multi-section/{job_id}`, `/incremental-kt`, `/templates`) that **do not exist** in current `main.py`. Only `/upload`, `/status/{job_id}`, `/schema`, `/schema/{job_id}`, `/export/pdf/{job_id}`, `/semantic-placement`, `/feedback`, `/` are real today.
- `DEAD_CODE_REPORT.md` and `SYSTEM_FLOW.md` predate the dynamic-schema/knowledge-object/renderer/PDF layer entirely and describe line numbers that no longer exist.

This isn't a blocker, but it means anyone (human or agent) reading these docs today gets actively misled about what API surface exists. Recommend archiving the stale ones alongside a note, per Phase 25/26's ask for updated `ARCHITECTURE.md`/`KT_PIPELINE.md`.

---

## 7. Proposed architecture vs. current reality

The Phase 3 target layout (`api/`, `transcription/`, `classification/`, `knowledge/`, `schema/`, `llm/`, `document/`, `rendering/`) is **largely already expressed**, just flat at repo root instead of nested under a package, and with some naming drift:

| Target | Current equivalent |
|---|---|
| `transcription/` | `main.py` (Whisper call inline) + `devops_transcription.py` |
| `classification/` | `context_mapper.py` + `section_rules.py` + `policy.py`/`runtime_policy.py` + `entity_extractor.py` |
| `knowledge/` | `knowledge/` package — already matches almost exactly |
| `schema/` | `schema_generator.py`, `kt_schema_new.json` |
| `llm/` | `llm_provider.py`, `ai.py` (prompts are inline dicts in `ai.py`, not a `prompts/` submodule) |
| `document/` | `knowledge/knowledge_builder.py` (composer role) + `renderers/blocks/` (block model) |
| `rendering/` | `renderers/sections/`, `pdf/templates/` |
| `api/` | `main.py` (everything — routes, background task, startup, all in one file) |

**Recommendation for Phase 3 (when we get there): don't do a big-bang directory move.** The logical separation already exists at the module level; the highest-value structural change is splitting `main.py` (908 lines doing routing + orchestration + HTML rendering + PDF templating) into an `api/routes.py` + `pipeline/orchestrator.py` split, and extracting `ai.py`'s prompt strings into a `llm/prompts/` submodule. A full directory-tree rewrite would touch every import in the repo for cosmetic benefit and is the kind of blind restructure the brief itself warns against.

---

## 8. Cleanup candidates for Phase 2

**Done (2026-09-03):**
- Fixed the §5.1 `first_30_day_plan`/`first_30_day_ownership` schema-vs-registry mismatch, and fixed `validate_renderer_registry` to actually check schema ids against registry keys (verified: re-ran the validator against the live schema, passes)
- Untracked `__pycache__/*.pyc` (3 files) from git
- Deleted `renderers/system_overview.py`, `renderers/deployment.py`, `renderers/ownership.py`, `renderers/day1.py`, `renderers/security.py` — superseded duplicates, verified zero importers
- Deleted `renderers/sections/failures.py` — orphaned, superseded by `common.py::render_failures`
- Deleted `knowledge_builder.py` (root shim) — verified zero importers
- Fixed `check_kt.py` to call `GET /status/{job_id}` instead of the removed `GET /kt/{job_id}`

**Not deleted — corrected assumption:** `upload_and_get_kt.py`, `generate_audio.py`, `check_kt.py` are not duplicated by `scripts/`; the `scripts/` versions are `importlib`-based wrapper stubs that call into these root files. Deleting the root files would have broken the wrappers.

**Also done (2026-09-03):**
- Deleted `static/index_old.html` (superseded, not routed) and `static/screenshots/*.jpg` (141 orphaned files from the disabled screenshot feature, tracked via `git rm -r`)
- Removed the wasted-compute path from the live request: `build_section_paragraphs()` (a full second semantic re-classification via `enterprise_semantic_mapper.py`) is no longer called per-upload, and `kt_structured` is no longer stored on the job or returned by `/status/{job_id}` — verified zero in-repo consumers first (frontend, tests, scripts). The underlying `ai.build_section_paragraphs` / `enterprise_semantic_mapper.py` code itself was left in place, just no longer invoked from the hot path.
- Removed the fully-commented-out screenshot-capture implementation block in `main.py` (~60 lines); `screenshots = []` is kept since the frontend reads `job.screenshots` unconditionally
- Removed now-dead imports from `main.py`: `classify_transcript`, `analyze_transcript`, `get_sentence_model`, `SECTION_HINTS`, `serialize_kt`, `sentence_transformers.util` — each verified to have zero live call sites (only referenced inside the deleted commented block or nowhere at all)
- Removed confirmed-dead symbols from `ai.py` (zero callers anywhere in the repo, verified by grep, not just old docs): `KTSessionAggregator`, `assess_audio_quality`, `classify_with_confidence`, `validate_sentence_quality`, `chunk_text`, `process_coverage_with_gemini`
- Removed now-unused optional-dependency import blocks from `ai.py`: `librosa`/`librosa_feature`, `scipy.signal`, and `nltk` (including its `nltk.download('punkt'/'stopwords')` calls, which were running on **every process start** for a dependency nothing used)
- Verified throughout: `main.py`/`ai.py` still import cleanly end-to-end, and the full `tests/` suite (7 tests, including the heavier `test_ecommerce_kt.py` pipeline test) passes

**Decided (2026-09-03): PII anonymization removed, not fixed.** Product decision was to remove rather than re-enable this session. Deleted `pii_anonymizer.py` and `test_presidio_integration.py` (the latter only tested the now-gone feature); removed the dead `HAS_PRESIDIO` import block, the commented-out call site, and the now-meaningless `anonymize_pii` parameter from `devops_transcription.clean_transcript()`; removed `presidio-analyzer`/`presidio-anonymizer` from `requirements.txt`; archived `PRESIDIO_IMPLEMENTATION.md` to `archive/` since it now describes a fully removed feature, not a disabled one. Verified: `devops_transcription.py` compiles and runs, `main.py` imports cleanly, full `tests/` suite still passes (7/7). **If PII redaction is needed later, it should be built fresh** rather than resurrecting this version — the "blocking placeholder" bug was never diagnosed.

**Decided (2026-09-03): `templates.py` left unwired, untouched.** It's a real, complete KT-template CRUD/versioning/lock router, not dead scaffolding — but its "RBAC" trusts a raw client-supplied `X-User-Role` header with no authentication behind it, so anyone could claim `Admin` and get write/lock access. Wiring it into `main.py` as-is would ship that hole; fixing the auth gap is real feature work, not a cleanup-pass side effect. Decision: leave it exactly as found (not routed, not deleted) until real auth exists. Relevant if this project resumes the original brief's Phase 9 (dynamic/user-supplied KT templates) — this file is most of that groundwork already.

**Also done (2026-09-03):** removed `devops_transcription.py`'s own dead `validate_transcription_quality()` and `estimate_transcription_accuracy()` (zero callers anywhere, confirmed by repo-wide grep), which let the file drop its unused `nltk`, `scipy.signal`, and `numpy` imports too. Verified: compiles, `clean_transcript()` still runs correctly, `main.py` still imports, full `tests/` suite still 7/7.

**Remaining candidates not yet acted on — need a product decision, not just cleanup:**
- `ai.analyze_transcript`/`generate_report`/`POST /semantic-placement` — decide if this stays as an intentionally separate preview/debug endpoint or should be removed
- Root-level ad hoc test/demo scripts (`test_pipeline.py`, `test_coverage_extraction.py`, `demo_topic_blocks.py`, `demo_coverage_polish.py`) vs. the formal `tests/` package — pick one convention and consolidate

**MARK_FOR_REVIEW (needs a product decision, not just a dependency check):**
- `kt_structured`/`build_section_paragraphs`/`enterprise_semantic_mapper.py` codepath (§4.1) — deleting saves real compute per request, but confirm nothing outside this repo (e.g. a future frontend) is meant to consume `kt_structured.paragraphs` before removing
- `ai.analyze_transcript`/`classify_transcript`/`generate_report`/`KTSessionAggregator` and the `/semantic-placement` endpoint (§4.2) — decide if this is a kept debugging/preview endpoint or dead weight
- `pii_anonymizer.py` + its disabled call site (§4.4) — re-enable properly (fix the "blocking placeholder" bug) or remove; leaving it half-wired is the worst of both options
- `templates.py` — finish wiring it in (it looks like an intended template-management router) or remove
- Root-level `test_pipeline.py`, `test_coverage_extraction.py`, `test_presidio_integration.py`, `demo_topic_blocks.py`, `demo_coverage_polish.py` — consolidate into `tests/` or keep as documented manual/demo scripts, but pick one convention

---

## 9a. Phase 3 — architecture split (2026-09-03)

Executed the surgical split plan (approved via plan mode) instead of the original
brief's full 9-folder directory move. `main.py` (939 lines mixing FastAPI routes,
PDF/HTML rendering, and upload orchestration) is now a 32-line entrypoint. New files:

- `kt_schema_loader.py` (14 lines) — single source of truth for `SCHEMA`; fixed the
  real duplicate-load bug where `main.py` and `ai.py` each read `kt_schema_new.json`
  independently
- `pdf_rendering.py` (241 lines) — HTML/PDF composition, moved verbatim from `main.py`
- `pipeline.py` (349 lines) — job state (`JOB_QUEUE`/`JOB_LOCK`/`MODEL`/`MAPPER_PIPELINE`)
  and `process_upload_task`, moved verbatim from `main.py`; dropped `build_coverage()`
  and `deduplicate_analysis()` (found dead — zero callers — while doing this move)
- `api/routes.py` (306 lines) — all 8 HTTP endpoints, moved verbatim onto an `APIRouter`
- `llm/prompts.py` (129 lines) — the two prompt-template dicts extracted out of `ai.py`
  (802 lines now, down from 923)

`context_mapper.py`, `knowledge/`, `renderers/`, `schema_generator.py`,
`field_populator.py`, `llm_provider.py`, `devops_transcription.py` were left
untouched — already correctly separated by responsibility, moving them would only
have been folder-renaming.

**Verified**: all files compile; `import main` resolves the full graph with no
circular imports; all 8 routes present (`GET /schema`, `GET /schema/{job_id}`,
`GET /export/pdf/{job_id}`, `POST /upload`, `GET /status/{job_id}`,
`POST /semantic-placement`, `GET /`, `POST /feedback`); full `tests/` suite 7/7
(one test's import path updated: `tests/test_pdf_export.py` now imports from
`pdf_rendering` instead of `main`); **live smoke test** — booted the real server
with `uvicorn main:app`, hit `GET /` (200, full `index.html`), `GET /schema` (17
sections), `GET /export/pdf/{bad-id}` and `GET /status/{bad-id}` (404 as expected),
and `POST /semantic-placement` with a real transcript (200, real classification
result) — confirms the `os.path.dirname` fixups needed because `api/routes.py` now
sits one directory deeper than the old `main.py` were done correctly for both the
static-file-serving route and the PDF export's WeasyPrint `base_url`. `uvicorn
main:app --reload` still works unchanged as the startup command.

## 9b. Phase 4 — vocabulary expansion + self-learning + transcript test path (2026-09-03)

**Transcript-in test feature**: `pipeline.process_upload_task` was split so classification-through-rendering lives in a new `pipeline.run_kt_pipeline(job_id, transcript, segments=None)`, callable independently of audio/Whisper. New `POST /kt-from-transcript` endpoint (`api/routes.py`) and `scripts/test_kt_from_transcript.py` CLI (mirrors the polling pattern in `upload_and_get_kt.py`, `--file`/`--text`/`--pdf` flags) let a transcript go straight to a KT without recording audio or touching the UI. Note: while building this, found `upload_and_get_kt.py` itself is *also* stale (calls the removed `/kt/{job_id}` and `/coverage/{job_id}` endpoints, same problem `check_kt.py` had) — not fixed this pass since it wasn't in scope, but it's a live gap: right now there is no working reference script for polling a real audio-upload job's structured output beyond `/status/{job_id}`'s raw JSON.

**Vocabulary consolidation + expansion**: found two overlapping vocabulary-correction systems (`devops_transcription.DEVOPS_VOCABULARY`, always-on; `glossary.py`'s `glossary.json`-backed `DEFAULT_GLOSSARY`, previously only used in `context_mapper.py`'s low-confidence repair path). Consolidated onto `glossary.py`'s persistent store rather than building a third system. `DEVOPS_VOCABULARY` moved to `devops_vocabulary.py` and expanded from ~60 to ~230 canonical terms (AWS/Azure/GCP, Kubernetes objects, CI/CD, IaC, observability, messaging, databases, service mesh, security, concepts). `glossary.py`'s bootstrap `DEFAULT_GLOSSARY` similarly expanded. `devops_transcription.get_known_terms()` now merges both, and the phrase-correction step also applies `glossary`'s approved `phrase_corrections` — known-terms count went from ~200 to ~695.

**Self-learning loop**: new `vocabulary_learning.py` — `detect_vocabulary_candidates()` (near-miss fuzzy matches + novel-technical-token heuristics, pure function) runs automatically once per job inside `run_kt_pipeline`, recording candidates to `glossary_candidates.json` via `record_candidates()`. Nothing is ever auto-applied — `scripts/review_vocabulary.py` (list / `--approve` / `--reject` / `--interactive`) is the only path that writes into `glossary.json`, via `approve_candidate()`.

**Three real bugs found and fixed while verifying this, not just added feature work:**
1. **Near-miss detector was unusably noisy at first pass** — jaro-winkler scores ordinary English words surprisingly high against short tech acronyms ("our" vs "cloudfront" ~0.77). Fixed with stopword/min-length filtering on the candidate n-gram plus raising the gray-zone floor to 0.85 — cut one test sentence's candidates from 24 down to 4.
2. **Pre-existing data-loss bug in `apply_fuzzy_term_corrections`, made much worse by the vocabulary expansion**: multi-word fuzzy matching compared whole-phrase jaro-winkler only, so a phrase like "RabbitMQ heavily" could score 0.89 against the known alias "rabbit mq" purely from a shared prefix and silently replace *both* words — dropping "heavily" entirely and lowercasing "RabbitMQ". A second instance ("code is"/"code should" scoring ~0.9 against the newly-added alias "code build", replacing "is"/"should" with "build") was directly caused by this session's new aliases. Fixed in `apply_fuzzy_term_corrections` with two guards: skip any n-gram containing a word that's already an exact known term, and require every individual word to independently meet a minimum per-word similarity (not just the whole-phrase score) against its target-word counterpart. Also fixed, as a side effect, a third pre-existing instance that was already in the file's own demo output ("Devons are..." → "environment variables" dropping "are").
3. **Cross-process staleness**: `glossary.GLOSSARY` was a load-once module global — `save_glossary()` from a separate CLI process (`scripts/review_vocabulary.py`) updated `glossary.json` on disk but never reached an already-running server's private in-memory copy of the module, so an approval silently didn't take effect until restart. This is exactly the scenario the whole self-learning feature exists for, so it was caught by testing the real cross-process flow (CLI approves → already-running `uvicorn` server corrects on the next request) rather than only testing within one Python process. Fixed with `glossary.get_glossary()`, an mtime-checked accessor that transparently reloads from disk when the file has changed since this process last read it; `devops_transcription.py` and `glossary.py`'s own internal functions were switched to call it instead of reading the bare `GLOSSARY` name.

**Known, not fixed (flagged, out of scope for this pass)**: `PHRASE_CORRECTIONS` has a pre-existing rule `r"is\s+done": "goes down"` that mangles unrelated sentences (e.g. "Monitoring is done through Grafana" → "Monitoring goes down through Grafana") — surfaced during smoke testing, not introduced by this session's changes, left alone since fixing the broader fuzzy/phrase-correction rule quality is a larger undertaking than this pass's scope.

**Verified**: all files compile; `import main` resolves; full `tests/` suite 7/7 (including after both fuzzy-matcher fixes); live smoke test — booted the real server, submitted a transcript with mishearings via the new endpoint, confirmed corrections applied and candidates recorded, approved/rejected via the CLI from a *separate process*, and confirmed the still-running server (not restarted) applied the newly-approved correction on the next request — the actual end-to-end proof of the "self-learning, no restart" requirement.

## 9c. Vocabulary bulk expansion + a real performance fix (2026-09-03)

Follow-up to 9b: pushed the static vocabulary further so real-world coverage doesn't depend solely on the self-learning loop catching things over time. `devops_vocabulary.py` grew from ~230 to 482 canonical terms (round 2: more AWS/Azure/GCP services, the wider Kubernetes ecosystem — kubectl, Rancher, OpenShift, Cilium, Calico, etc. — more CI/CD/IaC tools, more observability/security tooling, and a large batch of SRE/DevOps process vocabulary: DORA metrics, MTTR/MTTA/MTBF, error budget, zero trust, blast radius, trunk-based development, etc.). `glossary.py`'s `DEFAULT_GLOSSARY` — the code-committed bootstrap for a fresh `glossary.json`, which matters more now that the file is gitignored (see below) — grew from 11 terms/3 acronyms to 180 terms/59 acronyms. Combined known-terms count: ~200 at session start → 1080 now. One duplicate dict key (`"auto scaling"`, harmless — Python just kept the last occurrence) found and removed while verifying.

**`glossary.json` was a tracked file with stale content, now properly untracked**: found mid-task that `glossary.json` was still committed to git from before this session, and a `rm -f` cleanup step during earlier testing had staged it as deleted. Restored its original content first (nothing unique was at risk — it was a strict subset of the new bootstrap), then untracked it and added both `glossary.json` and `glossary_candidates.json` to `.gitignore` alongside the already-ignored `job_queue_state.json`/`session_state.json` — this is per-machine runtime state for the self-learning feature now, not static bootstrap content, so it shouldn't be tracked at all going forward.

**Found and fixed a real performance regression this expansion caused, not just added data**: profiling showed `clean_transcript()` — called once per Whisper segment during real transcription — went from fast to ~970ms/call after the vocabulary grew ~5x, because `apply_fuzzy_term_corrections`'s brute-force n-gram-vs-every-same-length-term comparison scales with vocabulary size, and (separately) `get_known_terms()` was rebuilding the full ~1080-entry merged set from scratch on every single call. Fixed both:
- `get_known_terms()` now caches its result, invalidated by the identity of the object `glossary.get_glossary()` returns (which itself only changes when the file's mtime moves) — so a human approval still takes effect immediately cross-process (the Phase 4 requirement), but the expensive merge only reruns when the glossary actually changed, not on every correction call.
- `bucket_terms_by_word_count()` was extended to also bucket by first letter (a standard fuzzy-matching "blocking" technique — jaro-winkler is prefix-weighted, so a target starting with a different letter essentially never scores near the 0.88 threshold anyway), cutting the comparison pool roughly 15-20x.

Net result verified by profiling before/after: **~970ms → ~24ms per `clean_transcript()` call (~40x)**, with identical output on every correctness case re-checked (the RabbitMQ, "code is/should", and "Devons are" cases from 9b's bug fixes, plus the module's own built-in test cases). Full `tests/` suite stayed 7/7 throughout.

## 9d. Phase 5 — Section mapping: closed two real signal gaps (2026-09-04)

The prior status summary called Phase 5 "already substantially satisfied" by
`context_mapper.ContextClassifier`. Verified that claim properly this round and
found it was mostly right, but two of the brief's seven required signals were
computed and displayed without actually affecting the classification decision:

1. **Entities extracted but decorative** — `_score_sentence_candidates` called
   `EntityExtractor.get_context_entities()` and attached the result to every
   `Classification` for explainability, but the score formula
   (`alpha*base_sim + beta*context_sim + keyword_boost - overview_penalty`) never
   referenced it. Fixed: new `ENTITY_TYPE_SECTION_AFFINITY` table in
   `section_rules.py` (deliberately small — only `monitoring`→`monitoring_observability`,
   `escalation`/`owner`→`ownership_escalation`; generic types like `tool`/`platform`
   are excluded on purpose, not selective enough) feeds a capped `entity_boost`
   (max 0.10) into the formula, and the contribution is now visible in the
   `reason` string. **Verified live, not just by inspection**: a sentence with a
   GLiNER-extracted `monitoring` entity now shows `Entities=+0.10(monitoring:Alerts)`
   in its reason and a measurably higher combined confidence than before.
2. **No LLM verification anywhere in classification** — `llm_fallback_fn` existed
   and was wired into `ContextRepair` (Stage 4, audio-confidence text repair) but
   never into `ContextClassifier` (Stage 3, section classification) — the brief's
   "optional LLM verification" signal didn't exist for classification decisions at
   all. Fixed, scoped to selective/bounded volume by design (confirmed with the
   user given the parallel free-tier-tokens conversation): `ContextClassifier` now
   accepts `llm_fallback_fn`, and a new shared method `_maybe_verify_with_llm()`
   only fires when a classification is genuinely borderline (top-2 candidates
   within 0.05 confidence, or top confidence below `similarity_threshold*1.3`) —
   not on every sentence. The LLM can only pick among the classifier's own
   already-computed candidates (prompt explicitly forbids inventing a new one);
   a malformed/invented response or a thrown exception both fall back silently to
   the classifier's original pick.
3. **Found while wiring #2**: `ContextMappingPipeline.process()`'s real per-sentence
   loop calls `ContextClassifier._score_sentence_candidates()` **directly**, not
   `classify_sentence()` — it's a separate, batched reimplementation of the same
   primary/secondary selection logic (kept separate for cross-encoder batch
   performance). Adding Part B only inside `classify_sentence()` would have made
   it dead code in the real pipeline. Fixed by extracting the borderline-check +
   verification logic into `_maybe_verify_with_llm()`, called from **both**
   `classify_sentence()` and the batched loop, so the two paths can't drift apart
   on what counts as borderline or how a verified pick gets applied.
4. **Also fixed**: `gliner` (entity_extractor.py's real, in-use dependency) and
   `openai` (llm_provider.GroqProvider's dependency) were both actually installed
   and in active use in this environment but undeclared in `requirements.txt` — a
   fresh install would have silently degraded entity extraction to a no-op and
   made Groq unavailable. Declared both. Also removed `librosa` and `nltk` from
   `requirements.txt` — genuinely unused now (their last call sites were removed
   during Phase 4's `ai.py`/`devops_transcription.py` cleanup), confirmed by
   repo-wide grep before removing.

Everything else on the brief's Phase 5 checklist was already genuinely present
and needed no changes: semantic similarity + embeddings, keyword/rule signals
(`section_rules.SECTION_RULES`), neighboring sentence context, section
definitions, and per-mapping storage of sentence_id/section/confidence/reason/
alternative_section (`ExplainabilityLog`).

**Verified**: all files compile; full `tests/` suite 7/7 (including
`test_ecommerce_kt.py`'s real end-to-end classification); Part A confirmed live
via a direct `_score_sentence_candidates()` call showing the entity boost in both
the reason string and the numeric confidence; Part B confirmed via 5 direct
`_maybe_verify_with_llm()` scenarios — not-borderline (LLM not called),
borderline-with-real-pick (promotes correctly, demotes old primary, note
generated), invented-section response (discarded), thrown exception (swallowed),
and no `llm_fallback_fn` configured (never called) — all 5 behaved exactly as
designed.

## 9e. Phase 6 — Knowledge extraction was silently empty for 8 of 17 sections (2026-09-04)

Checked "knowledge extraction is thinner than the brief describes" precisely
rather than taking the earlier characterization at face value. It was worse than
"thin": **8 of 17 schema sections have no `fields` array at all**
(`plain_english_notes`, `monitoring_observability`, `disaster_recovery`,
`security_controls`, `cost_optimization`, `common_failures`, `danger_zones`,
`ownership_escalation`). `field_populator.populate_fields()` skips any section
without `fields`, so these 8 never got an entry in `populated_fields` at all —
which cascades into `knowledge_builder.py`'s `build_facts`/`build_entities`/
`build_relationships` always receiving `{}` and returning `[]`. Zero facts, zero
entities, zero relationships for these sections, not "thin" — genuinely empty.

Worse: `renderers/sections/security.py`, `disaster_recovery.py`, and
`ownership_escalation.py` all check specific field ids (`security_scan_config`,
`rto_steps`, `escalation_chain`, etc.) that **don't exist anywhere in
`kt_schema_new.json`** — these specialized renderers (which the audit had
credited as "already matching Phase 12") always silently fell through to generic
narrative rendering, never their intended technology-grid/checklist/ownership-table
output.

**Fixed** by extending the proven `SECTION_STRUCTURED_PROMPTS`/
`_extract_structured_section` mechanism (built in Phase 4 for monitoring) to
`security_controls`, `disaster_recovery`, `ownership_escalation`, feeding the
LLM-extracted JSON into `populated_fields` using the exact field ids the
renderers already check (new `ai.wrap_structured_as_fields` helper) — one
mechanism fixes both the renderer gap and the knowledge-object gap, no separate
plumbing needed for each. Also upgraded `knowledge/entities.py` to type entities
by field-id (reusing `field_populator.py`'s own dispatch convention) instead of
wrapping every field as an untyped blob, and replaced `knowledge/relationships.py`'s
dead rule (checked `dependent_service`/`service_name` — fields that don't exist
anywhere in the schema, always `[]`) with real ownership/escalation-chain
relationships built from data that now actually exists.

**Found the same class of bug as the `first_30_day_plan` one from earlier this
session, predating this session** (already in the user's uncommitted WIP at the
very start, preserved verbatim through Phase 3/4 without being caught):
`pipeline.py` checked `if section_id in ("monitoring",):` and
`SECTION_STRUCTURED_PROMPTS` was keyed `"monitoring"` — the real schema id is
`monitoring_observability`. **Monitoring's structured extraction, built in Phase
4, had never actually fired in the real pipeline.** Fixed the key.

**Found and fixed something bigger while trying to verify the above live**: `ai.py`
imports `llm_provider` (which reads `GEMINI_API_KEY`/`GROQ_API_KEY` from the
environment at *import time*) several lines *before* calling `load_dotenv()` —
so `llm_provider`'s module-level config constants get permanently baked in as
empty on every fresh process start, regardless of what `.env` contains. **This
silently disabled every Gemini/Groq-dependent call in the entire app**
(coverage polish, field gap-fill, structured extraction, Phase 5's classification
verification) whenever started via `uvicorn main:app` — not something introduced
this session, but not caught either until a live test showed "Gemini provider is
not configured" errors despite `.env` having a real key. Fixed at the root:
`llm_provider.py` now calls `load_dotenv()` itself at the top, before reading any
config — self-sufficient regardless of what imports it or in what order.
`ai.py`'s own import order was also fixed for consistency.

**A second real finding surfaced immediately once the dotenv fix let real API
calls through**: this Gemini API key's free tier is capped at **5 requests per
minute** (`429 RESOURCE_EXHAUSTED`, confirmed from live error responses, not
speculation) — and a single KT run needs 15-25+ Gemini calls across polish/
gap-fill/structured-extraction/classification-verification. Most silently fall
back to non-LLM behavior once the budget is exhausted a few seconds into
processing. This is concrete, measured evidence for the paused "is Gemini
optimal" conversation — worth resuming with the user now that it's not
speculative.

**Verified**: full `tests/` suite 7/7 throughout (including after the dotenv
fix). Direct unit tests of `wrap_structured_as_fields`/`build_entities`/
`build_relationships` with a realistic ownership_escalation payload — correct
output. Live end-to-end run (server up, real transcript, real Gemini calls)
confirmed the full chain for `disaster_recovery`: `populated_fields` gets
real `llm_structured`-sourced values in exactly the shape the renderer expects,
and the knowledge object's `facts` array (with evidence pointing back to the
source sentences) is populated — previously would have been `[]`. Server logs
confirm all 4 target sections (monitoring_observability, security_controls,
disaster_recovery, ownership_escalation) are correctly identified and attempted
by the fixed section-id check in a larger transcript run — the 3 that didn't
complete there were blocked by the external rate limit, not a code defect.

**Explicitly scoped out** (noted for later, not fixed): `cost_optimization`
(renderer expects differently-formatted `coverage_content`, not a `fields` dict)
and `common_failures` (renderer never attempts structured rendering at all,
needs new renderer logic, not just data) — both lower-confidence, more invasive
fixes than the three done here. `danger_zones` and `plain_english_notes` already
render correctly as-is, nothing to fix.

## 9f. LLM provider parameter mismatch — Gemini/Groq weren't actually interchangeable (2026-09-05)

User is running with Groq configured. Investigated "will the prompts built for
Gemini work on Groq" precisely rather than assuming the provider abstraction
was airtight just because it compiled.

**Prompt text itself: yes, fully portable** — every prompt in `llm/prompts.py`
and the ad-hoc prompts in `ai.py`/`field_populator.py`/`context_mapper.py` is
plain text with no Gemini-specific syntax; Groq's OpenAI-compatible chat
endpoint wraps it as a standard user message.

**The generation parameters around it were not portable — a real, two-way bug**:
every one of the 5 real call sites in this codebase (`ai.py`×2,
`field_populator.py`, `context_mapper.py`×2) passes `max_output_tokens=`
(Gemini's kwarg name). `GroqProvider.generate()` only checked `max_tokens=`, so
every Groq-routed call silently ignored the caller's requested limit (512, 256,
20, …) and always used the 1024 default instead — no error, just quietly wrong
behavior. `GroqProvider` also never forwarded `stop_sequences` to the Groq API
at all, so `field_populator.py`'s gap-fill (which relies on `stop_sequences=["\n"]`
to keep an answer to one line) had no such guardrail under Groq. Mirror-image
gap going the other way: `GeminiProvider.generate()` silently dropped
`system_prompt` entirely, even though every structured-extraction/polish call
site passes one expecting it to shape the response — Groq was honoring it,
Gemini wasn't.

**Fixed** in `llm_provider.py`: `GroqProvider` now reads `max_output_tokens`
(falling back to `max_tokens` for any future OpenAI-native caller) and forwards
non-empty `stop_sequences` as the API's `stop` parameter. `GeminiProvider` now
forwards `system_prompt` as `system_instruction` in `GenerateContentConfig`.
Both providers now honor the same canonical kwarg set — the actual promise of
having an abstraction in the first place.

**Also worth knowing (not a bug, a config trap)**: setting `GROQ_API_KEY` alone
does **not** route calls to Groq. `create_llm_provider()` only returns
`GroqProvider()` when the `LLM_PROVIDER` environment variable is literally set
to `"groq"` — it defaults to `"gemini"` otherwise, regardless of whether a Groq
key exists. Confirmed via `.env`: only `GEMINI_API_KEY` is set there; if
`LLM_PROVIDER=groq` and `GROQ_API_KEY` are being set as real shell/OS
environment variables outside this file, that's outside what this session can
see or verify directly.

**Verified**: both fixes confirmed via mocked-client unit tests (no real API
calls needed) — `GroqProvider` now correctly passes `max_tokens=512` (was
silently 1024) and `stop=['\n']` (was never passed) to the API call; empty
`stop_sequences=[]` correctly omits the parameter rather than passing an empty
list; `GeminiProvider` now correctly passes `system_instruction` when a
`system_prompt` is given, and omits it when none is given. Full `tests/` suite
still 7/7.

## 9g. Groq model selection — llama-3.3-70b-versatile decommissioned, reasoning models don't fit this codebase's token budgets (2026-09-05)

User confirmed `llama-3.3-70b-versatile` (this repo's hardcoded `GROQ_MODEL`
default) is decommissioned on Groq. Verified live against the account's actual
`/models` catalog rather than guessing from general knowledge — confirmed it's
genuinely gone, and empirically tested the realistic replacement candidates at
this codebase's tightest real budget (20 tokens, `context_mapper.py`'s
classification verification):

| Model | Result at 20-token budget |
|---|---|
| `openai/gpt-oss-120b` | Empty — all budget consumed by hidden reasoning tokens |
| `openai/gpt-oss-20b` | Empty — same failure, smaller model didn't help |
| `qwen/qwen3.6-27b` | Garbled — leaks `<think>...` chain-of-thought directly into content, truncated mid-thought |
| `qwen/qwen3.8-27b` | **Clean `'OK'` in 2 tokens** |
| `allam-2-7b` | Clean `'OK'` in 3 tokens (works, but 7B — smaller/less capable) |
| `groq/compound-mini` | Clean output, but 52 tokens for a 1-word answer — an agentic/tool-orchestration model, unpredictable for strict "return only this JSON" tasks |

`qwen/qwen3.8-27b` also verified on a realistic structured-JSON-extraction
prompt (disaster_recovery's actual `SECTION_STRUCTURED_PROMPTS` template) —
clean, correctly-keyed JSON output, no reasoning leakage, 73 tokens with
headroom to spare inside the 512-token budget that call site uses.

**Fixed**: `llm_provider.py`'s `GROQ_MODEL` default changed from the
decommissioned `llama-3.3-70b-versatile` to `qwen/qwen3.8-27b` — verified
end-to-end through the actual `GroqProvider.generate()` code path (not just raw
API calls) at the 20-token classification-verification budget, confirmed
non-empty, well-formed output.

**General lesson for anyone changing `GROQ_MODEL` later**: this codebase's LLM
call sites use short token budgets (20-1024) written for fast, direct-answer
models. A reasoning model (anything emitting hidden chain-of-thought — the
`openai/gpt-oss-*` family, `qwen/qwen3.6-27b` in thinking mode) will silently
return empty or garbled content at these budgets unless either the model's
reasoning is disabled/low-effort, or every `max_output_tokens` value in the
codebase is substantially raised to leave headroom. Test any new model choice
against a real call site's actual budget before assuming it works — a
"successful" trivial test can still fail once the token budget gets tight.

## 9h. Rate-limit retry — turn transient 429s into eventual success instead of silent fallback (2026-09-05)

Follow-up to §9g: user's live Groq dashboard showed 35 HTTP 200 / 17 HTTP 429
out of 52 requests in one burst minute. Every real call site already wraps
`provider.generate()` in try/except that just logs and falls back to non-LLM
behavior on any failure — turning a transient, recoverable rate limit into a
permanent quality loss for that field/section, one out of every ~3 calls in
that burst.

**Fixed** in `llm_provider.py`: both `GeminiProvider` and `GroqProvider` (SDK
and raw-HTTP paths) now retry through a shared `_call_with_rate_limit_retry()`
helper on rate-limit errors — honoring the provider's own suggested delay when
available (`Retry-After` header for Groq/OpenAI-compatible responses,
`retryDelay` parsed out of Gemini's error body) and falling back to capped
exponential backoff (1s, 2s, 4s, ... up to `LLM_RETRY_MAX_DELAY_SECONDS`, default
60s) otherwise. Up to `LLM_RETRY_MAX_ATTEMPTS` (default 5) retries before
re-raising — so the existing fallback-on-exception behavior at each call site
still applies as the final safety net if a provider is *genuinely* down, not
just rate-limited.

**Verified** with the real exception types, not approximations: constructed an
actual `openai.RateLimitError` (real `httpx.Response`, real `retry-after`
header) and confirmed the retry logic correctly detects it, extracts the exact
header value, waits, and returns the second call's result. Also verified the
Gemini-style `retryDelay: '7s'` string-embedded format parses correctly, and
that a persistently-failing call exhausts all attempts with the expected
backoff sequence (1/2/4/8/16s) and cleanly re-raises rather than hanging.
Full `tests/` suite still 7/7.

**Trade-off worth knowing**: this makes the pipeline *slower under load*
instead of *silently degraded* — a heavily rate-limited run can now take
significantly longer (worst case ~31s of added wait per call that hits the
default 5-retry ceiling) rather than finishing fast with missing LLM-derived
content. That's the intended trade for "all requests eventually succeed."

## 9i. Phase 7 — Evidence/traceability: precise per-fact evidence was silently never computed (2026-09-05)

Checked "why did Continuum put this information into the KT" precisely.
`knowledge_builder._collect_evidence()` was clearly *written* expecting precise
per-field evidence — it checks `populated_field.get("source_chunk_index")` and,
if present, returns just that one sentence; otherwise it falls back to a
generic "first 3 sentences of the section" for every fact. **Nothing in the
pipeline that actually feeds it ever set `source_chunk_index`** — confirmed by
grep: the only place that ever computed one (`ai.map_analysis_to_fields()`) is
a completely separate, parallel field-extraction system whose output
(`mapped_fields`) is stored in the job dict but never passed into
`build_knowledge_object`. Every fact in every section was silently getting the
identical generic evidence, regardless of which sentence actually supported it.

**Fixed** in two places, since sections split across two different
field-producing code paths (found this the hard way — first live test targeted
`disaster_recovery`, which turned out to be entirely on the *other* path):
- `field_populator.py`: new public `find_source_sentence_index(value, raw_sentence_texts)`
  — best-effort substring match of an extracted value against the section's
  raw, timestamped sentences (handles comma-joined multi-value matches too).
  `populate_fields()` now accepts `section_content` (threaded from
  `pipeline.py`'s `kt.section_content`) and attaches `source_chunk_index` to
  every pattern/semantic/LLM-derived field via a shared `_emit()` helper.
- `ai.wrap_structured_as_fields()` (the Phase 6 mechanism that feeds
  `security_controls`/`disaster_recovery`/`ownership_escalation`/
  `monitoring_observability`, which have no `fields` array in the schema and so
  never go through `populate_fields()` at all) now also accepts
  `raw_sentence_texts` and calls the same shared `find_source_sentence_index()`.

**Critical alignment detail**: the index computed must refer to the *exact*
list `_collect_evidence()` later indexes into (`kt.section_content[id]['sentences']`)
— not `coverage[id]['content']`, which `field_populator.py` already had access
to and initially seemed like the natural thing to search, but is a different,
polished/reordered view with a different item count. Computing an index against
the wrong list would silently attach *wrong* evidence rather than just fall
back to the generic (and safe) default — worse than the bug being fixed. Both
new call sites explicitly source `raw_sentence_texts` from `kt.section_content`.

**Verified**: unit tests on `find_source_sentence_index()` — correct index for
values genuinely present in a sentence, honest `None` (not a guess) for values
that aren't. Full `tests/` suite 7/7. Live end-to-end run (real Groq calls)
before/after comparison on `disaster_recovery`'s 4 facts: before the second fix,
all 4 shared identical evidence (the generic fallback); after, each fact
correctly points to its own distinct, accurate source sentence (verified by
reading the actual returned text, not just checking indices differ).

### 9j. Phase 8 — knowledge model verification: 2 real defects found and fixed

Audited `knowledge/knowledge_builder.py` and its four builders
(`facts.py`/`entities.py`/`evidence.py`/`relationships.py`) against the brief's
"genuine facts, not invented content" principle, now that Phase 6/7 have made
the underlying field data real. Two defects found:

1. **`build_facts()` didn't filter unfilled placeholder fields** (`knowledge/facts.py`).
   `field_populator._populate_fields_recursive()` emits an entry for *every*
   declared field, filled or not — unfilled ones are
   `{"value": "", "confidence": 0.0, "source": "unfilled"}` (see
   `field_populator.py:376`). `build_entities()` already correctly skips these
   (`if not value: continue`), but `build_facts()` had no such guard, so the
   knowledge object's `facts` array — and by extension the raw `/schema/{job_id}`
   API response — carried a noise entry per unfilled field, each with
   `_collect_evidence()`'s generic first-3-sentences fallback attached as if it
   were real evidence for a fact that doesn't exist. `pdf_rendering.py`'s
   `_build_fallback_paragraphs()` happened to filter these at render time
   (`if isinstance(value, str) and value.strip()`), so the PDF itself was never
   affected — this was purely an API/knowledge-object correctness issue, not a
   visible rendering bug. Fixed by adding the same filter `build_entities()`
   already uses: `if field.get("value") not in (None, "", [])`.
   Verified: unit-checked with a 3-field input (1 real, 2 unfilled) — `build_facts`
   now returns exactly the 1 real fact.

2. **Escalation-chain separator inconsistency** (`field_populator.py` /
   `llm/prompts.py` / `knowledge/relationships.py`). Three producers/consumers of
   the `escalation_chain` field id disagreed on delimiter: the pattern-extraction
   fallback `_extract_escalation_chain()` joined steps with `" → "` (unicode
   arrow), while the Phase 6 LLM structured-extraction prompt instructs `"->"`
   (ASCII) and `knowledge/relationships.py`'s `build_relationships()` only ever
   split on `"->"`. Traced whether this is live: `ownership_escalation` (the
   section that owns this field conceptually) has no `fields` array in
   `kt_schema_new.json`, so it's populated exclusively via the Phase 6
   LLM-structured path (`ai.wrap_structured_as_fields`, which already uses
   `"->"`) — never via `_populate_fields_recursive`'s pattern dispatch. The only
   schema field matching the `"escalation" in field_id` check is
   `handover_completion.escalation_clear`, which is `type: "boolean"` and returns
   from the boolean branch before reaching the escalation-chain dispatch. **So
   `_extract_escalation_chain()` is current dead code** — the mismatch has no
   live effect today, but is a landmine (same class as the id-mismatch bugs
   found in Phases 2/5/6/7): the moment any schema field routes a text value
   through that function, `relationships.py` would silently produce zero
   "escalates to" relationships instead of erroring. Standardized on `"->"`
   (matching the two call sites that are actually live) in
   `_extract_escalation_chain()`. Verified: fed a realistic escalation sentence
   through `_extract_escalation_chain()` → `build_relationships()` end-to-end,
   confirmed 2 correct "escalates to" relationship triples now round-trip
   instead of silently 0.

**No other defects found** in the knowledge-model layer this pass —
`build_entities`, `build_evidence`, `_infer_system_name`, and the `_structured`
passthrough (still needed for `renderers/sections/monitoring.py`'s legacy
technology-grid path, not dead) all check out correctly against their actual
call sites.

### 9k. Phase 12 — cost_optimization / common_failures structured extraction (the 2 sections Phase 6 scoped out)

Picked up the 2 sections Phase 6 explicitly deferred as "different shape, more
invasive." Confirmed precisely why: `renderers/sections/cost_optimization.py`
and `common_failures.py` already had real `TechnologyGrid`/`DecisionTable`
rendering code, but it only activates on pipe-delimited (`"label | value"`)
`coverage_content` strings — nothing in the pipeline ever produces that format
(the LLM polish step writes prose), so both sections always fell back to
generic narrative, same failure mode as the 4 sections fixed in Phase 6.

Unlike those 4 (flat scalar fields, fit `ai.wrap_structured_as_fields()`'s
`{field_id: value}` shape), these 2 need a **list of records** (multiple
levers / multiple failures). Reused the existing, already-wired `_structured`
passthrough instead (`coverage[sid]['_structured']`, already forwarded by
`knowledge_builder.py` and already consumed this way by
`renderers/sections/monitoring.py`) rather than forcing list data through the
flat-field mechanism.

Changes: 2 new `SECTION_STRUCTURED_PROMPTS` entries (`llm/prompts.py`);
`_extract_structured_section`'s token budget raised 512→768 (`ai.py`, headroom
for multi-item lists, applies uniformly — safe for the existing 4 sections
too); `pipeline.py`'s extraction-trigger tuple extended to include both new
ids, and the `populated_fields` merge loop explicitly restricted to the
original 4 flat-shaped sections (a `FLAT_STRUCTURED_SECTIONS` guard) so the 2
list-shaped ones don't get wrongly flattened into a single nonsense "fact"
holding a raw Python list; both renderers updated to read `_structured` first,
falling back to their original pipe-parsing/narrative paths unchanged if
absent. `common_failures.py`'s table also widened from a legacy 3-column shape
to the full 5 columns `kt_schema_new.json` actually declares for this section
(`Issue/Symptom, Likely Cause, How to Fix, Frequency, KEDB / Ticket Link`) —
Frequency and Ticket Link were being structurally discarded even on the rare
pipe-format hit.

**Found and fixed one more defect while unit-verifying the renderer change**
(not introduced by this change — pre-existing): `cost_optimization.py`'s pipe
parser synthesized a `{"label": text, "value": ""}` row for *any* non-pipe
text, and `renderers/blocks/technology_grid.py`'s `build_block()` silently
drops rows where `value` is empty (requires both `label` AND `value`
truthy) — so plain narrative coverage content with no `_structured` data
rendered as a **silently empty TechnologyGrid** (worse: `pdf_rendering.py`'s
`_has_meaningful_rendered_blocks()` treats any `TechnologyGrid` block as
"meaningful" unconditionally, so it never fell through to the narrative
fallback either — a genuinely invisible section in the PDF). Fixed by having
the pipe parser skip non-pipe text entirely instead of synthesizing a
value-less row, letting `render()`'s existing `if rows: ... else: narrative`
correctly choose the narrative fallback.

**Verified**: `python -m py_compile` on all 5 touched files; full `pytest
tests/` 7/7; pure-Python renderer unit checks covering 6 cases (structured
present for both sections; pipe fallback; the now-fixed narrative fallback;
common_failures' non-pipe fallback; empty content) — all correct. Live
end-to-end run (real Groq calls) with a transcript covering 3 cost levers and
3 distinct failures: `/schema/{job_id}` confirmed `_structured.levers`
(3 items, correctly detailed) and `_structured.failures` (3 items, correctly
including one accurately-`null` frequency/ticket pair) exactly matching the
source content with no invention; `rendered_sections` showed `TechnologyGrid`/
`DecisionTable` blocks (not `NarrativeBlock`) with the full 5-column shape for
`common_failures`; PDF export succeeded (valid `%PDF`, non-trivial size).

### 9l. PDF enterprise redesign + section-mapping fixes (user-provided real PDF review)

The user shared a real generated PDF (job 85F99846) and asked for an enterprise
visual redesign plus "proper mapping" — read it page-by-page against the live
renderer code and found the root cause of nearly every visual problem, several
of which were genuine data bugs, not cosmetic:

1. **`monitoring_observability` was wired to the wrong renderer.** The
   registry (`renderers/sections/__init__.py`) pointed at `common.py`'s naive
   keyword-substring `render_monitoring`, which dumped raw whole sentences as
   "Tool" values (visible in the shared PDF as a garbled "Tool" row). A
   correct, complete renderer already existed at `renderers/sections/monitoring.py`
   (reads `_structured.tools`/`first_response_steps`/`alert_routing`, the
   actual Phase 6 data) but was never registered — same bug class as every
   id-mismatch bug found earlier this session. Fixed by rewiring the registry.
   **While auditing the registry for this, found 3 more orphaned duplicate
   renderer files that were never reachable at all**: `renderers/sections/
   architecture.py`, `danger_zones.py`, `ownership.py` — each shadowed by a
   different file with the same purpose that the registry actually used.
   Deleted all three (confirmed zero other references first).
2. **`day1_survival_checklist` had field-id mismatches** (`safe_first_actions`/
   `dont_do_day1` vs the real schema ids `first_safe_actions`/
   `actions_not_to_perform`) and never read `required_access` (a `type: table`
   field) at all — always fell to generic narrative. Fixed the ids and added a
   `DecisionTable` for `required_access`, reusing a newly-shared
   `renderers/blocks/table.py:parse_table_rows()` (extracted from
   `deployment.py`'s previously-local `_parse_table_rows`, now used by both).
3. **`deployment_and_rollback` showed the same content 2-3 times** — `pre_deployment_checks`,
   `post_deployment_validation`, and the `deployment_window` timeline entry
   independently resolved to overlapping/identical text (a short transcript's
   semantic/LLM-gap-fill matching collapsing multiple fields onto the same
   broad chunk). Fixed with a rendering-level dedup (`_claim()`/`seen_keys` in
   `deployment.py`, same normalize-and-compare approach `pdf_rendering.py`'s
   fallback-paragraph builder already used) — the underlying field-population
   collision itself is a deeper, separate problem, not tackled here.
   **Also found and fixed a real live bug while touching this file**:
   `deployment.py` used `re.match(...)` in one branch but never imported `re` —
   a `NameError` waiting to fire whenever `deployment_steps` itself was
   unpopulated but other coverage_content existed. Because `pipeline.py` wraps
   `build_rendered_sections()` in a blanket try/except, this wouldn't have
   crashed the app — it would have silently discarded `rendered_sections` for
   **every section in the document**, not just deployment, degrading the whole
   PDF to the generic coverage-based fallback path with no visible error.
4. **`handover_completion` and `signoff` never read their real schema fields**
   (6 boolean/select fields; 4 sign-off fields respectively) — both just
   printed a canned "being derived from KT coverage" / "Thank you." string
   regardless of what was actually populated. Built a real checklist for the
   former and a new `renderers/sections/signoff.py` (key-value `TechnologyGrid`)
   for the latter, registered in place of the generic fallback.
5. **`open_responsibilities` was hardcoded to always render as a red
   `WarningBlock`** regardless of content — misleading for a purely
   informational transition-plan section — and never read its real fields
   (`open_tasks`, `recurring_responsibilities`, both `type: table`). Fixed to
   render real tables when populated, plain narrative otherwise.
6. **The CSS "danger" styling used a `:has(p strong)` content-sniffing
   selector** meant for `WarningBlock`s, but it fired on *any* paragraph
   containing bold text — and `SECTION_POLISH_PROMPTS` (llm/prompts.py)
   formats many sections' narrative with `**Label:**` markdown, which becomes
   `<strong>` after markdown rendering. Result: `DAY-1 SURVIVAL CHECKLIST`,
   `DEPLOYMENT & ROLLBACK` etc. all got flagged red/"dangerous" in the shared
   PDF even though they weren't. Fixed by having `pdf_rendering.py`'s
   `render_section_blocks()` assign an explicit `warning-card` class only when
   `block_type == "WarningBlock"`, and moved the CSS rule onto that class.
7. **Every single-block section repeated its own title twice** (section `<h2>`
   immediately followed by an identical block `<h3>`) — visible throughout the
   shared PDF. Fixed by skipping the block heading when it case-insensitively
   matches the section title; sections whose blocks have genuinely distinct
   titles (e.g. "Recovery actions", "Security tools") keep them.
8. **At least 7 different ad hoc "being extracted/assembled/synthesized/
   inferred/derived" placeholder strings** were scattered across renderer
   files, all doing the same job (zero coverage) but reading like debug output
   in a real deliverable. Replaced all of them with one shared, professionally
   worded helper: `renderers/blocks/common.py:no_coverage_block()` — "This
   section was not covered in the KT session. Flag it for follow-up with the
   outgoing owner." Put in a new `renderers/blocks/common.py` rather than
   `pdf_rendering.py` specifically to avoid a circular import (`pdf_rendering`
   → `renderers` → `renderers.sections.*` → back to `pdf_rendering` if the
   helper lived there). `pdf_rendering.py`'s own top-level fallback
   (`_build_fallback_paragraphs`'s "Rendered content not available.") and its
   `_has_meaningful_rendered_blocks()` detection (previously a fragile 4-phrase
   substring guess) were both updated to use the same shared constant.
9. **Found a structural dead-code bug while fixing #8**, in 3 files
   (`first_30_day_ownership.py`, `cost_optimization.py`, `common_failures.py`):
   each had a `... else: blocks.append(narrative_with_possibly_empty_paragraphs)`
   branch that ran unconditionally, so the file's own trailing
   `if not blocks: append_fallback()` was unreachable dead code — `blocks` was
   never empty by that point, even when the narrative it just appended had zero
   real paragraphs. This is exactly why the shared PDF showed the generic
   top-level "Rendered content not available." for `first_30_day_ownership`
   instead of that file's own (now-removed) ad hoc string: `pdf_rendering.py`'s
   `_has_meaningful_rendered_blocks()` correctly rejected the empty-paragraph
   block, forcing a fall-through to the top-level fallback one layer up. Fixed
   all 3 by only appending the narrative block when paragraphs is non-empty.
10. Added a section-number badge (`01`, `02`, ...) to each `<h2>` matching the
    TOC's ordering, one accent color (`#2563eb`) for section rules/table
    headers, red reserved strictly for the now-correctly-scoped warning cards.
11. Removed `plain_english_notes` (the user's own suggestion — `required: false`,
    scope substantially overlapping `danger_zones`/`common_failures`/day1's
    "actions not to perform") from `kt_schema_new.json`, its registry entry,
    its `SECTION_RULES` force-classification rule (`section_rules.py` — left in
    place it would have force-routed matching sentences to a section id that
    no longer exists), and its `OPTIONAL_SECTION_INCLUSION_RULES` entry
    (`schema_generator.py`). Updated `tests/test_ecommerce_kt.py`'s `EXPECTED`
    mapping to drop the now-removed section's assertion (the transcript's
    "tribal knowledge... cache invalidation" sentence is no longer asserted to
    land anywhere specific).

**One real defect found live-testing but explicitly NOT fixed in this pass**
(flagged for a dedicated follow-up, not silently dropped): `field_populator.py`'s
`type == "table"` extraction (`_extract_by_pattern`) falls back to "first 10
non-empty lines of the whole section's raw/polished text" when it finds no
pipe-delimited or numbered-list lines — with no awareness of which specific
field it's populating. Live-tested with a transcript that clearly stated
day-1 access items ("request access to the cloud console, the git repository,
the CI/CD tool, and the monitoring dashboards") *and* separate safe-first-actions
guidance: `required_access` ended up populated with the safe-first-actions
text instead (duplicating `first_safe_actions`'s own, separately-correct
value), because the "first 10 lines" heuristic isn't field-aware. The new
`day1.py` renderer (item 2 above) faithfully displays whatever it's given —
unit-tested independently with clean synthetic data and confirmed correct — so
this is purely an upstream extraction-quality gap, not a rendering bug. Likely
affects other `type: table` fields across the schema whenever their
pattern/numbered detection fails to find real tabular structure (e.g.
`open_tasks`, `recurring_responsibilities`). Also observed a related, milder
case: `ownership_escalation`'s `oncall_tool` field (pattern-matches specific
tool brand names only) fell to a weak semantic match that picked the whole
escalation-chain sentence when no tool brand appeared in that section's own
text. Both are the same underlying issue — the semantic/table fallback paths
in `field_populator.py` have no per-field precision guarantee — and would need
a proper design pass (e.g. routing through the structured-JSON-extraction
pattern already proven in Phases 6/12, rather than continuing to strengthen
ad hoc string heuristics) rather than another point-fix.

**Verified**: `python -m py_compile` on all ~24 touched/new files; full
`pytest tests/` 7/7 (including the updated ecommerce classification test, which
correctly no longer asserts anything about the removed section); registry
sanity check (`RENDERER_REGISTRY` keys exactly match schema ids, including the
new `signoff` entry, via `validate_renderer_registry()`); pure-Python unit
checks on every fixed renderer (monitoring/day1/deployment/handover_completion/
signoff/open_responsibilities) with synthetic data, including the dedup and
warning-card/title-dedup HTML generation logic in `pdf_rendering.py` directly.
Live end-to-end run (real Groq calls) against a purpose-built transcript
covering every fixed section, followed by a full visual read of the exported
PDF: confirmed clean monitoring tool lists, a real day-1 access table (data
quality caveat above), deduped deployment content, real handover/sign-off
tables, correctly-scoped warning styling (danger_zones/known_bad_days red;
day1/deployment/ownership_escalation — all containing markdown bold — correctly
NOT red), no doubled headings anywhere in 10 rendered pages, consistent
"not covered" messaging for the 2 genuinely-uncovered sections, and
`plain_english_notes` absent from both TOC and body.

## 9. Fix from this audit already worth doing next

The §5.1 renderer/schema id mismatch (`first_30_day_plan` vs `first_30_day_ownership`) is a live, silent rendering bug on the branch currently being worked. Recommend fixing it in the same session as this audit, before moving on to any of Phases 4–26, since it directly undermines the very validation check this branch just introduced.
