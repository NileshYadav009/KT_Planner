# Testing

## Running the suite

```
.venv-1/Scripts/python -m pytest tests/ -q
```

Use `.venv-1` specifically — it's the environment with actual dependencies
installed (`.venv` appears to lack `fastapi` and others). Full run takes
several minutes: most of that is one-time embedding/cross-encoder model
loading shared implicitly across the classification-heavy tests, not the
assertions themselves.

## What's covered

| File | What it tests |
|---|---|
| `test_dynamic_schema.py` | `schema_generator.generate_dynamic_schema()` — dynamic field/section inclusion. |
| `test_ecommerce_kt.py` | Stage 3-5 classification against a fixed, realistic transcript (`test_ecommerce_kt_mapping`) — including a regression check that `system_overview` doesn't become a dumping ground for specialized content. Also `clean_transcript()` regex-fix regressions. |
| `test_golden_kt.py` | The fuller pipeline, end to end: `pipeline.run_kt_pipeline()` (the real orchestration `/kt-from-transcript` uses) through knowledge-object assembly, validation, quality scoring, rendering, and PDF export — deterministically, with the LLM provider stubbed out (see below). |
| `test_pdf_export.py` | `render_pdf_html()` + WeasyPrint produce a valid PDF; fallback-paragraph deduplication. |
| `test_validation.py` | `validation.py`'s structural checks — knowledge-object shape, populated-fields-vs-schema id consistency (including nested `type: group` fields). |
| `test_quality_score.py` | `quality_score.py` — required-vs-optional weighting, validation-warning penalty, grade bands. |
| `test_field_populator.py` | The cross-field-contamination fix and its underlying blocks-fallback (`REPOSITORY_AUDIT.md` §9n), `find_source_sentence_index()`. |
| `test_llm_provider.py` | The proactive throttle's sliding-window logic (via a monkeypatched fake clock — no real 60s waits) and the rate-limit-detection/retry-delay helpers. |
| `test_pdf_rendering_helpers.py` | `_render_inline_text()`'s markdown handling, `kv-table`/`grid-table` class tagging, block-title dedup, warning-card CSS-class scoping. |
| `test_knowledge_builders.py` | `build_facts()`'s unfilled-field exclusion, `build_entities()`, the escalation-chain extractor/parser separator agreement. |

## Keeping tests deterministic

Most of this suite runs with **no live LLM calls** — either because the code
path being tested doesn't need one (pure functions like `validation.py`,
`quality_score.py`, `pdf_rendering.py`'s helpers), or because the test
explicitly stubs the provider out. `test_golden_kt.py` is the pattern to
follow for anything that needs to exercise real pipeline orchestration
without live API calls or model downloads beyond the embedding/cross-encoder
models already needed for classification:

```python
monkeypatch.setattr(pipeline, "get_llm_provider", lambda: None)
monkeypatch.setattr(ai, "get_llm_provider", lambda: None)
```

Both are needed — `pipeline.py` and `ai.py` each do
`from llm_provider import get_llm_provider`, creating their own bound
reference; patching `llm_provider.get_llm_provider` itself would not affect
either already-imported name.

`test_golden_kt.py` also constructs `pipeline.MAPPER_PIPELINE` directly
rather than calling `pipeline.load_models()`, since that function
additionally loads a Whisper speech-to-text model no text-only test needs.

## What's explicitly not covered

Stated plainly rather than implied by omission:

- **`pipeline.run_kt_pipeline`/`process_upload_task` with a real LLM
  provider** — the LLM-dependent structured-extraction sections
  (`security_controls`, `disaster_recovery`, `ownership_escalation`,
  `monitoring_observability`, `cost_optimization`, `common_failures`) are
  only verified via this session's documented live manual runs
  (`REPOSITORY_AUDIT.md`), not a checked-in automated test. Golden-testing
  these would need recorded/mocked API responses — a separate effort.
- **`process_upload_task`'s audio path specifically** (Whisper transcription
  itself) — no test exercises real or synthetic audio.
- **`vocabulary_learning.py`** (candidate detection/approval loop) — no
  dedicated tests.
- **The API layer itself** (`api/routes.py`) — no `TestClient`-based
  endpoint tests; the underlying functions they call are tested directly
  instead.
- **`static/index.html`** — no browser-based/interactive testing exists or
  is currently possible in this environment (no browser-automation tooling
  available). Its JS is verified by code review against the actual API
  response shapes it consumes, not by clicking through it.
- **`templates.py`** — untested; deliberately unwired (see
  `DELIVERABLE_SUMMARY.md`).

## Startup validation (not a pytest test, but part of the safety net)

`renderers/sections/validate_renderer_registry()` runs at server startup
(`pipeline.load_models()`) and raises if a section claiming a "dedicated
renderer" is missing from the registry or points at a schema id that doesn't
exist — catches one whole class of silent renderer-wiring bug before the
server even starts accepting requests.
