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

## End-to-end KT suite (tests/e2e_kts)

Four short synthetic handovers (112-160 words): AWS e-commerce on EKS, an
Azure payment platform on AKS, a GCP data platform (Pub/Sub, Dataflow,
BigQuery) and an ambiguous, incomplete KT with one misheard term ("Argo City"
for ArgoCD). Each file holds the transcript and a checklist written from the
transcript before any output was looked at: the facts and the sections they
belong in, safety warnings, the connections the KT states and ones it does not,
values the speaker called unknown, claims that would be invented, and the
procedures the KT leaves incomplete.

`tests/test_e2e_kts.py` runs each one through `pipeline.run_kt_pipeline` (what
`POST /kt-from-transcript` queues) and the real PDF renderer with **no LLM**
(zero requests), then checks the PDF text (`tests/e2e_eval.py`), per dataset:
no fact lost; safety warnings in Danger Zones and incidents not; no value for
anything the speaker said is undocumented; no invented commands, tools,
durations, products or owners; no greeting or hand-over framing; no claim the
handover is complete; every stated connection drawn and none of the wrong
ones; every incomplete procedure flagged in its own section; the coverage
matrix agrees with what is printed; every fact shown has its source sentence;
misheard terms corrected; placement at or above `tests/e2e_kts/floors.json`.
False negatives and false positives are reported separately.

The LLM path is run by `scripts/e2e_kts.py`, which labels every result with
its mode and never mixes them:

```
# no LLM, in-process (the pytest mode)
python scripts/e2e_kts.py --llm off --out e2e_out
# a running server exactly as a client uses it (POST, poll, export PDF), with
# whatever LLM it is configured for; its per-KT usage report is included
python scripts/e2e_kts.py --api http://127.0.0.1:8011 --key-file key.txt --out e2e_live
# recorded LLM output, offline: nothing is sent; unrecorded requests are skipped
python scripts/e2e_kts.py --llm replay --recording tests/fixtures/e2e_kts_llm_recording.json --out e2e_replay
```

`tests/fixtures/e2e_kts_llm_recording.json` holds the 45 Groq
`qwen/qwen3.8-27b` responses (no prompts, no keys) the current code asks for
on these four KTs, recorded on 12 Oct through the API with `LLM_BATCH_CALLS=1`.
A prompt change makes those requests unrecorded; re-record with `--api` and
`--export-recording CACHE.sqlite JSON --since <start of the run>`.

Measured on 12 Oct (problems = checklist failures; see the report):

| | AWS | Azure | GCP | Ambiguous |
|---|---|---|---|---|
| No LLM, before fixes | 9 | 8 | 8 | 13 |
| No LLM, after | 0 | 1 | 0 | 0 |
| Groq through the API, before fixes | 9 | 12 | 8 | 8 |
| Groq (recorded answers, final code), after | 0 | 1 | 0 | 0 |

The one remaining failure: a sentence that states two facts ("Infrastructure
is written in Bicep, and all secrets are stored in Key Vault") is not split,
so the Bicep half is printed under Security. Two connections are inferred from
passive sentences and drawn as the system's ("Events are published to Pub/Sub"
becomes "Streamline publishes to Pub/Sub"); the checklist does not rule them out.

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
