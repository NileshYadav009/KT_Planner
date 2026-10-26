# System flow

How a KT goes from upload to PDF, as the code stands (26 Oct 2026). Every
step names the function that does it; line numbers are left out because they
go stale (the earlier version of this file cited `main.py:142` when `main.py`
had 94 lines). `docs/KT_PIPELINE.md` describes each pipeline stage in detail,
and the pipeline map in `CLAUDE.md` is the short version.

## Request to queue (API process)

```
Browser (static/index.html) or an API client
   │
   ▼  main.py middleware, outermost first:
   │    security_headers           CSP with the pages' inline-script hashes, HSTS on HTTPS,
   │                               X-Frame-Options and others (browser_security.py)
   │    refuse_oversized_uploads   413 from Content-Length: uploads (media_guard) and pastes
   │    request_id                 X-Request-ID on every log line
   │
   ▼  api/routes.py, each route behind auth.require_permission (key or session cookie,
   │  CSRF header for changes, idle sign-out after CONTINUUM_SESSION_IDLE)
   │
   ├─ POST /upload                 media_guard.save_upload + probe (ffprobe allow-list, size cap)
   │                               → TASKS.enqueue(job, "upload")
   ├─ POST /kt-from-transcript     _paste_task on a thread: size cap (input_gate.MAX_PASTE_CHARS),
   │                               captions.to_turns (WebVTT/SRT), clean_transcript, assess_transcript
   │                               → 422 with the reasons, or TASKS.enqueue(job, "transcript");
   │                               a paste that needs the punctuation model is queued as it came
   │                               ("clean": true) and cleaned by the worker
   ├─ POST /kt/{id}/sessions       the same, as a follow-up session ("session")
   ├─ POST /kt/{id}/giver, /speakers   a rebuild with the chosen giver or speaker names
   └─ Idempotency-Key              TASKS.reserve_key: a repeated submission returns the first job
```

The queue is the `tasks` table in the job database (`job_queue.TaskQueue`):
a worker claims a task with a lease (`LEASE_SECONDS`, 120), renews it while
the KT runs, and a task whose worker died is claimed again when the lease
runs out.

## Worker

```
worker.Worker._loop → run_next → TASKS.claim → run_task → _run
   │
   ├─ "upload"      pipeline.process_upload_task
   │                  transcribe_recording: ffmpeg audio, faster-whisper, diarisation (sherpa-onnx),
   │                  screen capture (RapidOCR), clean_transcript per segment, assess_transcript
   │                  → run_kt_pipeline
   ├─ "transcript"  pipeline.clean_queued_paste (only a paste queued as it came) → run_kt_pipeline
   ├─ "session"     sessions.add_session: the sessions combined → run_kt_pipeline, a new version
   ├─ "giver"       sessions.set_giver          → run_kt_pipeline, a new version
   └─ "speakers"    sessions.rename_speakers    → run_kt_pipeline, a new version
   │
   ▼  render_and_store_document: the PDF stored with the job version it shows
   TASKS.finish

Heartbeat thread (every LEASE_SECONDS / 4): lease renewal, worker heartbeat,
queue-age alert, review reminders (staleness.check_due_reviews) and
retention (pipeline.apply_retention_when_due, hourly).
```

## run_kt_pipeline (pipeline.py)

```
redact_secrets (transcript and segments)    credentials never reach an LLM or the store
speaker_roles.turn_segments                 pasted "Name:" turns
   │
   ▼  MAPPER_PIPELINE.process  (context_mapper.ContextMappingPipeline)
   │    clean segments; segment_sentences (fact_units.split_into_facts; corrections kept whole)
   │    speaker roles (detect_roles, apply_roles): facts from the giver
   │    merge_spoken_corrections, resolve_self_corrections, copyedit.copyedit_sentences
   │    ContextClassifier: bge-large embeddings, examples, hints, GLiNER, topic segments,
   │      the learned section model; close calls to the LLM (verify_in_batches)
   │    section_rules.apply_rule_overrides; restore_sections_emptied_by_topic; rejoin_classified
   │    coverage per section; assets; assemble_kt → StructuredKT
   │
   ▼  Polishing            ai.polish_coverage_sections (LLM, guarded: grounding, negations, withdrawn values)
   ▼  Structured extraction ai._extract_structured_section (LLM, JSON, ground_structured)
   ▼  Field population     field_populator.populate_fields (patterns, semantic match, LLM gap fill)
   ▼  Document assembly    knowledge_builder.build_knowledge_object, apply_conflicts,
   │                       apply_llm_conflict_check, append_coverage_matrix_section
   ▼  Rendering            pdf_rendering.build_rendered_sections (renderers/sections/*),
   │                       verify_document_coverage (no fact lost), lead_with_quick_reference,
   │                       knowledge.evidence.attach_evidence (sources)
   ▼  Post-processing      vocabulary candidates for the workspace (vocabulary_learning)
   ▼  Validation and scoring, then the result in JOB_QUEUE (job_store.PersistentJobs → SQLite)
```

Every stage that can fail records a stage warning and the KT continues
without it (`_stage_failed`); any LLM failure falls back to the rules-only
result. Each LLM call and each KT's LLM time and calls are bounded
(`llm/cached_provider.CachedLLMProvider`).

## Reading the result

```
GET /status/{job_id}               the job (tenant-checked: auth.ensure_job_access)
GET /export/pdf/{job_id}           the stored PDF, or rendered now (WeasyPrint, data: URLs only)
GET /export/markdown|runbook|gaps  other forms of the same knowledge object
GET /documents/{id}/versions       version history; /diff between versions
POST /documents/{id}/edits         an edit, saved as a new version
/signoff/{id}/...                  three-party sign-off that locks a version
```

The page reads the knowledge object's `rendered_sections`, `_coverage_rows`
and `_knowledge_gaps` (the same coverage matrix as the PDF) and fills its form
from `populated_fields`, the PDF's own field values.

## Where data lives

| Data | Where | Backed up by `scripts/backup.py` |
|---|---|---|
| Jobs, document versions, the queue, people, API keys, sessions, audit log, vocabulary candidates | `CONTINUUM_DB_PATH` (SQLite, WAL) | Yes |
| Screenshots | `KT_ASSETS_DIR` | Yes |
| LLM response cache | `LLM_CACHE_PATH` | No: it refills itself |
| Uploaded recordings | a temporary file, removed when the KT is built | No |
