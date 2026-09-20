# Architecture

Continuum KT Planner turns a recorded (or pasted) knowledge-transfer session
into a structured, enterprise-formatted Knowledge Transfer document (PDF +
JSON API). This is the module map and request flow — for the detailed
technical history behind why things are shaped this way, see
`REPOSITORY_AUDIT.md`.

## Request flow

```
Client (static/index.html, or any HTTP client)
    │
    ▼
api/routes.py                  FastAPI endpoints: /upload, /kt-from-transcript,
                                /status/{id}, /schema/{id}, /export/pdf/{id},
                                /feedback, /schema
    │
    ▼
pipeline.py                    Job queue (JOB_QUEUE/JOB_LOCK) + orchestration.
                                process_upload_task() (audio) and the
                                /kt-from-transcript route both converge on
                                run_kt_pipeline() — everything past
                                transcription is shared.
    │
    ▼
context_mapper.py              The 7-stage classification pipeline (see
                                KT_PIPELINE.md) — turns a transcript into
                                per-section coverage + section_content
                                (raw, timestamped sentences per section).
    │
    ▼
schema_generator.py            Builds the dynamic schema for this run (adds
                                detected technology fields, includes/excludes
                                optional sections based on coverage).
    │
    ▼
field_populator.py             Fills each schema field from section_content's
                                raw sentences (pattern match -> semantic match
                                -> LLM gap-fill, in that order) — plus
                                ai.wrap_structured_as_fields() for the sections
                                that have no fields array and rely on LLM
                                structured-JSON extraction instead
                                (llm/prompts.py's SECTION_STRUCTURED_PROMPTS).
    │
    ▼
knowledge/                     build_knowledge_object() assembles the final
                                per-section facts/entities/evidence/
                                relationships from populated fields.
    │
    ▼
validation.py                  Non-fatal structural checks on the assembled
                                knowledge object and populated-fields-vs-schema
                                consistency. Logs warnings, never blocks.
    │
    ▼
quality_score.py               Aggregates per-section coverage/confidence/risk
                                + validation warnings into one document-level
                                score/grade.
    │
    ▼
pdf_rendering.py + renderers/  build_rendered_sections() dispatches each
                                section to a specialized renderer (or a
                                shared narrative fallback), producing typed
                                "blocks" (NarrativeBlock, DecisionTable,
                                TechnologyGrid, etc. — see PDF_RENDERING.md).
                                render_pdf_html() turns those blocks into the
                                final HTML WeasyPrint renders to PDF.
```

`llm_provider.py` sits underneath `field_populator.py`, `ai.py`, and
`context_mapper.py`'s classification-verification step — see
`LLM_PROVIDER.md` for the provider abstraction and rate-limit handling.

## Key modules

| Module | Responsibility |
|---|---|
| `api/routes.py` | HTTP surface. Reads/writes `pipeline.JOB_QUEUE` directly (module-attribute access, not `from pipeline import JOB_QUEUE`, so writes from the background task are always visible). |
| `pipeline.py` | Job lifecycle, model loading (`load_models()`), and `run_kt_pipeline()` — the shared orchestration both entry points converge on. |
| `context_mapper.py` | The 7-stage transcript-to-coverage pipeline. See `KT_PIPELINE.md`. |
| `devops_transcription.py` / `devops_vocabulary.py` / `glossary.py` / `vocabulary_learning.py` | Transcript vocabulary correction and the self-learning glossary loop (candidates recorded, approved via `scripts/review_vocabulary.py`). |
| `schema_generator.py` | Builds the per-run dynamic schema from `kt_schema_new.json` (via `kt_schema_loader.py`) plus detected technology and coverage-driven optional-section inclusion. |
| `field_populator.py` | Per-field extraction: pattern regexes first, then semantic similarity against real transcript sentences, then LLM gap-fill as a last resort. |
| `ai.py` | LLM-facing helpers: coverage narrative polish, structured JSON extraction for the schema sections with no `fields` array, legacy field-mapping helpers. |
| `llm/prompts.py` | All LLM prompt templates — polish prompts and structured-extraction prompts, kept as pure data separate from the calling logic. |
| `llm_provider.py` | Provider abstraction (Gemini/Groq/fallback), rate-limit retry + proactive throttle. See `LLM_PROVIDER.md`. |
| `knowledge/` | `build_knowledge_object()` and its four builders (`facts.py`, `entities.py`, `evidence.py`, `relationships.py`) — turn populated fields into the knowledge object's structured arrays. |
| `validation.py` | Structural consistency checks (Phase 19). |
| `quality_score.py` | Document-level quality aggregate (Phase 22). |
| `renderers/` | `sections/` — one renderer per schema section (or a shared fallback), each producing typed blocks. `blocks/` — small builders for each block shape (table, checklist, warning, etc.) plus `common.py`'s shared "not covered" fallback message. |
| `pdf_rendering.py` | Turns typed blocks into the final HTML (`render_section_blocks()`) and hands it to WeasyPrint via `pdf/templates/kt_document.html`/`.css`. |
| `kt_schema_new.json` | The base schema — section definitions, fields, columns, hints/sub_topics used for classification. |
| `static/index.html` | The frontend — upload/paste a transcript, poll status, view coverage + the rendered knowledge document, export PDF, submit sentence-reclassification feedback. |
| `templates.py` | A working template CRUD router, deliberately left unwired (see `DELIVERABLE_SUMMARY.md` — its RBAC trusts a spoofable header with no real auth behind it). |

## State

Job state lives in an in-memory dict (`pipeline.JOB_QUEUE`, guarded by
`pipeline.JOB_LOCK`) — there is no database. This means job history doesn't
survive a process restart; acceptable for the current single-process
deployment model, worth revisiting if this needs to run behind multiple
worker processes or survive restarts.
