# Continuum: engineering record

This is the living record for anyone, person or agent, changing the KT pipeline.
`docs/REPOSITORY_AUDIT.md` holds the evidence for each change, by section number
(§9aao is the latest batch). `docs/KT_PIPELINE.md` describes each stage in
detail. `docs/progress.md` is the plain-language log.

## What must hold on every KT

1. **No fact lost.** Every classified fact appears in the document. A fact
   the renderers dropped is added back under Additional Notes, and every fact
   has a ledger record (`knowledge_builder.verify_document_coverage`).
2. **Nothing invented.** LLM output is grounded against the transcript or
   refused (`grounding.py`). Inferred values are marked as inferred.
3. **No meaning changed.** Rewrites keep every number, tool, negation, speaker
   and tense. See `copyedit.problems`, `grounding.negations_flipped` and the
   guards in `ai._polish_one`. Safety rules stay word for word.
4. **Gaps are honest.** A section is Covered only when it was discussed
   (`coverage_topics.py`, `knowledge_builder.append_coverage_matrix_section`).
5. **Every job finishes.** Each LLM request is bounded
   (`LLM_REQUEST_TIMEOUT_SECONDS`, 60), and so is each KT's LLM time
   (`LLM_KT_DEADLINE_SECONDS`, 900, counted from the KT's first call) and call
   count (`LLM_MAX_CALLS_PER_KT`). Any LLM failure falls back to the
   rules-only result.

Placement accuracy is a quality target, not an invariant: 90% or better on
unseen KTs. Without an LLM it is 86/95 on the holdouts and 95/114 on the blind
set. With Groq it was about 94% (audit §9aal).

## Pipeline map (a pasted transcript; a recording first goes through `pipeline.transcribe_recording`)

| Stage | Code |
|---|---|
| Upload, queue | `api/routes.py`, `job_queue.py`, `worker.py` |
| Clean (vocabulary, fillers) | `devops_transcription.clean_transcript` |
| Sentences, facts | `context_mapper.segment_sentences`, `fact_units.split_into_facts` |
| Self-corrections, copy edit | `devops_transcription.resolve_self_corrections`, `copyedit.copyedit_sentences` |
| Classify | `context_mapper.ContextClassifier._score_sentence_candidates` (bge-large, examples, hints, GLiNER, topic segments) |
| Close calls to the LLM | `ContextClassifier.verify_in_batches` |
| Routing rules | `section_rules.SECTION_RULES`, `apply_rule_overrides` |
| Regroup facts, coverage | `rejoin_classified`, `restore_sections_emptied_by_topic`, `assemble_kt` |
| Polish, structured extraction | `ai.polish_coverage_sections`, `ai._extract_structured_section` |
| Fields | `field_populator.populate_fields` (patterns, semantic, LLM gap fill) |
| Knowledge object | `knowledge_builder.build_knowledge_object`, `apply_conflicts`, coverage matrix, `verify_document_coverage` |
| Render | `renderers/sections/*`, then `pdf_rendering.build_job_pdf` (WeasyPrint) |

Templates: the pipeline uses one fixed 17-section schema (`kt_schema_new.json`
via `kt_schema_loader.SCHEMA`). `templates.py` has template routes, but nothing
mounts them and the pipeline never reads a stored template.

## LLM calls

The provider is `llm_provider.GroqProvider` (OpenAI SDK, `GROQ_MODEL` in
`.env`). It is wrapped by `llm/cached_provider.CachedLLMProvider`, which adds
the exact-response cache, per-KT usage, the call and time budgets, and the
tenant "no external LLM" policy.

| Stage label | Caller | Asks for | Checked by | On failure |
|---|---|---|---|---|
| `copy_edit` | `copyedit._rewrite_remote` (`SYSTEM_PROMPT`, 15 sentences a call) | Each spoken sentence as written English | `copyedit.problems` per sentence | Sentence kept as said; local Qwen takes sentences Groq could not |
| `classification_check` | `ContextClassifier._verify_batch_with_llm` (15 a call) | A section id from that sentence's own top 3, or NONE. Ids and titles only, no descriptions. | Answer must be one of its candidates | Classifier's pick kept |
| `sentence_repair` | `ContextMappingPipeline._try_llm_repair` | One garbled sentence fixed | `_is_faithful_repair` | Original kept |
| `section_polish` | `ai._polish_one` (`llm/prompts.SECTION_POLISH_PROMPTS`, 7 sections plus a default) | A section's fragments as formatted text | Added specifics, flipped negations, dropped fragments | Raw fragments kept |
| `structured_extraction` | `ai._extract_structured_section` (`SECTION_STRUCTURED_PROMPTS`, 7 sections, JSON, first 20 fragments) | Table and field values | `_extract_json_response`, then `ground_structured` drops unsupported values | Plain sentences |
| `field_fill` | `field_populator._fill_deferred_gaps` (one call per section, the default) or one call per field (`LLM_BATCH_CALLS=0`) | Value plus EXPLICIT / INFERRED, or NOT_MENTIONED | Grounding; structural types need a literal value | Field left unfilled |

## Testing

Don't run the full suite by default. It takes hours on this laptop; run it
overnight and only when asked. Use targeted runs, from the repo root with
`.venv-1`:

| Area | Command (`python -m pytest -q -p no:cacheprovider ...`) | Time |
|---|---|---|
| Routing rules, topics, coverage topics | `tests/test_section_rules.py tests/test_topic_segments.py tests/test_coverage_topics.py` | seconds |
| Cleaner, copy edit, polish guards | `tests/test_devops_transcription.py tests/test_copyedit.py tests/test_polish_integrity.py` | about 30 s |
| LLM controls, no network | `tests/test_llm_cost_controls.py tests/test_llm_provider.py tests/test_llm_batching.py` | about 25 s |
| Fields | `tests/test_field_populator.py` | about 15 s |
| Knowledge and coverage | `tests/test_knowledge_builders.py tests/test_fact_units.py tests/test_dialogue.py` | about 3 min |
| End to end, with recorded Groq replay | `tests/test_e2e_kts.py` | about 13 min |

Accuracy: `python evaluation/accuracy.py LABEL tuning,dev,holdout,blind2,e2e`
(about 4 min; see `evaluation/README.md`). Every placement or coverage change
is measured on all five sets. The LLM-mode version is
`evaluation/groq_eval.py`. It spends Groq quota, so run it only when agreed;
`GROQ_BUDGET` caps its live requests.

Constraints:

- **Groq:**
  - Never call it in tests; `tests/conftest.py` removes the keys.
  - An LLM-mode evaluation spends quota and must be agreed first.
  - A changed prompt invalidates the recorded replay (`tests/e2e_kts`) until
    it is re-recorded.
- **Data:** tests use scratch `CONTINUUM_DB_PATH` and `KT_ASSETS_DIR`. Open
  `data/continuum.sqlite` read-only (`mode=ro`) only.
- **Secrets:** the Groq key is only in `.env` (git-ignored). Never print it,
  log it or commit it.
- **Line endings:** the repository mixes CRLF and LF. Keep each file's own
  endings. `sed -i` strips CRLF, and Git Bash heredocs collapse `\\`, so use
  an editor or a Python byte replacement.

## Issue ledger

Open items, most consequential first. Fixed items say where the evidence is.

| ID | Sev | Issue | Status |
|---|---|---|---|
| R1 | High | A stalled Groq request could hold a KT for about 30 min (SDK: 600 s timeout, 2 retries), and a KT's LLM time had no cap | Fixed, §9aao |
| C1 | High | "By the end of the month you should ..." counted as an Operational Calendar month-end window, so an undiscussed section showed as Partial (hidden gap) | Fixed, §9aao |
| M1 | High | An escalation path ("if they do not answer, call Maria, the engineering manager") went to Monitoring, and Ownership reported "Escalation path: not discussed" | Fixed, §9aao |
| M2 | Med | "We run in one region", and planned or unstarted work, were misfiled | Fixed, §9aao |
| T1 | Low | 4 field-populator tests assumed one call per field after batching became the default | Fixed, §9aao |
| L1 | Med | A correction split across sentences ("The limit is 500. Sorry, not 500, ... 300.") shows both values | Fixed, §9aap (`merge_spoken_corrections`; sources quote both sentences) |
| L2 | Med | A sentence of two commands ("get access ..., and spend the first weeks watching ...") is one fact, so its second topic is unplaced | Open: only 2 more such sentences in the visible KTs, both correct unsplit; splitting commands risks scattering procedures |
| L3 | Med | The classifier makes 21 of the 22 placement errors on unseen KTs; the LLM check sees only ids and titles, and only the top 3 | Dev A/B +1/224 (§9aap). Off by default (`LLM_VERIFY_CANDIDATES=5`, `LLM_VERIFY_PURPOSE=1`). Next: confirm on holdouts and blind with fresh quota, then switch and re-record the replay. |
| L4 | Med | With Groq: repeated lines and unsourced facts | Fixed, §9aap. The calendar copied every cost lever; a paragraph or row cited too few sources. Dev 4 to 0 and 2 to 0, by replay. |
| L5 | Low | Topic memory can raise a second candidate above the chosen section without swapping them | Closed: logs only. Alternatives are read only by multi-section mapping, which is off (`ENABLE_MULTI_SECTION_MAPPING`). |
| P1 | Product | Custom templates are not supported (routes not mounted; rules and prompts name the default section ids) | Needs a product decision |
