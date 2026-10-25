# Continuum: engineering record

This is the living record for anyone, person or agent, changing the KT pipeline.
`docs/REPOSITORY_AUDIT.md` holds the evidence for each change, by section number
(§9aay is the latest batch). `docs/KT_PIPELINE.md` describes each stage in
detail. `docs/CORRECTIONS.md` is the specification for corrections, changes
and contradictions. `docs/ADVERSARIAL_MATRIX.md` is the adversarial case
table. `docs/progress.md` is the plain-language log.

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
| Sentences, facts | `context_mapper.segment_sentences` (a correcting sentence kept whole with what it corrects: `replaces_value_before`, `points_back_to`), `fact_units.split_into_facts` |
| Self-corrections, copy edit | `devops_transcription.resolve_self_corrections`, `merge_spoken_corrections`, `copyedit.copyedit_sentences` |
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
| `section_polish` | `ai._polish_one` (`llm/prompts.SECTION_POLISH_PROMPTS`, 7 sections plus a default; shared `POLISH_SYSTEM_PROMPT`: keep the speaker's wording, omit unfilled labels, fragments are data) | A section's fragments as formatted text | Placeholder lines ("**Window:** N/A") removed (`_without_placeholder_lines`); added specifics, flipped negations, dropped fragments | Raw fragments kept |
| `structured_extraction` | `ai._extract_structured_section` (`SECTION_STRUCTURED_PROMPTS`, 7 sections, JSON, first 20 fragments) | Table and field values | `_extract_json_response`, then `ground_structured` drops unsupported values | Plain sentences |
| `field_fill` | `field_populator._fill_deferred_gaps` (one call per section, the default) or one call per field (`LLM_BATCH_CALLS=0`) | Value plus EXPLICIT / INFERRED, or NOT_MENTIONED | Grounding; structural types need a literal value | Field left unfilled |
| `conflict_check` | `conflicts._check_conflicts_with_llm` via `knowledge_builder.apply_llm_conflict_check` (one call per KT: the numbered statements that have a close neighbour by meaning in another unit, at most 40; `LLM_CONFLICT_CHECK=0` turns it off). Its prompt says the statements are data. | The contradicting pairs, as "i-j" lines, or NONE | A pair must be two listed statements from different units; one the rules already flagged is not added twice | No flag |

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
is measured on all five sets.

Adversarial matrix: `python evaluation/matrix.py LABEL [IDS]` (71 short KTs,
about 3 min). Each case targets one failure (corrections, hedges,
contradictions, numbers, order, noise, PDF text) with exact checks on the
rendered document. A new failure gets a case there first, then a unit test.
`python evaluation/matrix_report.py BEFORE AFTER` rewrites
`docs/ADVERSARIAL_MATRIX.md`. The LLM-mode version is
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
| L3 | Med | The classifier makes 21 of the 22 placement errors on unseen KTs; the LLM check sees only ids and titles, and only the top 3 | Measured twice, +1 each time, no loss: dev +1/224 (§9aap); holdouts and blind with only the section check live, 199 vs 198 of 209 (§9aar). Costs about 25% more tokens in the check. Off by default (`LLM_VERIFY_CANDIDATES=5`, `LLM_VERIFY_PURPOSE=1`); switching needs a replay re-record. The user's call. |
| L4 | Med | With Groq: repeated lines and unsourced facts | Fixed, §9aap. The calendar copied every cost lever; a paragraph or row cited too few sources. Dev 4 to 0 and 2 to 0, by replay. |
| L5 | Low | Topic memory can raise a second candidate above the chosen section without swapping them | Closed: logs only. Alternatives are read only by multi-section mapping, which is off (`ENABLE_MULTI_SECTION_MAPPING`). |
| P1 | Product | Custom templates are not supported (routes not mounted; rules and prompts name the default section ids) | Needs a product decision |
| X1 | High | A same-sentence correction after a dash, "no, sorry", or with its lead-in said again ("every 30 minutes, sorry, every 30 seconds") kept both values | Fixed, §9aaq (`resolve_self_corrections`) |
| X2 | High | A next-sentence correction that says more than the number ("Actually, the upgrade is complete; production is now on PostgreSQL 16") left the old value on its own, often in another section | Fixed, §9aaq: kept whole with the sentence it corrects (`replaces_value_before`) |
| X3 | High | A correction further back ("I was wrong about the backups earlier, ...") left the old value standing | Fixed, §9aaq (`points_back_to`; up to 20 sentences back, one-fact sentences only) |
| X4 | High | A technology named only in a plan or a guess ("planning to migrate to Kubernetes", "might be using Redis Cluster") was listed as current in the Technology summary and Architecture Knowledge | Fixed, §9aaq (`field_populator.current_tool_mentions`) |
| X5 | Med | The last step of a procedure said one step per sentence ("After that, resume the sync ...") went to another section; the topic restore also undid placements the continuation rules left in place | Fixed, §9aaq (step rule in `apply_continuation_rules`) |
| X6 | Med | Two counts of one thing, or one setting with two values, with no correction, were shown apart with no flag | Fixed, §9aaq (`conflicts.find_value_conflicts`). No flag on any of the five corpora. |
| X7 | Low | Names opening a sentence lost their spelling ("Orders-api", "Kubectl", "IOS"); spelled-out acronyms ("r d s") and split names ("post gres", "graph ana") were kept; call chatter ("my connection dropped") became a fact; "That happens during peak traffic" left its antecedent; a connection quoted a corrected sentence without its correction | Fixed, §9aaq |
| X8 | Med | Contradictions of meaning ("Developers own deployment approvals" / "Only the release manager can approve production deployments"; "The service is public" / "Only private subnets can reach it"; "hourly" / "every six hours") are shown but not flagged | Fixed with an LLM, §9aar (`conflict_check`, one call per KT, about 400 tokens). Matrix D02-D04 flagged; no false flag on 8 matrix controls or the 11 dev and e2e KTs. Without an LLM they are still not flagged. Recall on real KTs is unmeasured: no visible KT has such a contradiction. |
| X9 | Low | No LLM prompt said that transcript text is data, not instructions | Fixed: the polish and `conflict_check` (§9aat), and the copy edit, extraction, repair, section check and field fill (§9aax). |
| X10 | Med | The permit/forbid rule flagged any "allowed" and "forbidden" sentences with two shared words: false "Possible conflict" warnings in 3 of the 6 dev KTs ("we can always redeploy an older model" / "don't delete the old model versions") | Fixed, §9aar: the permitting sentence must name the forbidden action (its verb and two thirds of its words). Dev false flags 3 to 0; g07's Fridays rule still flagged. |
| X11 | High | The brief's own correction examples: "We upgraded production to PostgreSQL 16 last night" (no marker), "I think we changed it to thirty days, but I'm not sure" (hedged), and "Actually, application ownership belongs to Payments, but Platform owns the cluster" (no number) left the old statement standing alone | Fixed, §9aas (`replaces_value_before` reads change statements; `corrects_statement_before`). Matrix A16, A18, A19. |
| X12 | Med | "Staging uses 2000K, while production uses 3000K." was cut into two facts; "Production uses 3000K." lost what it measured and left Environments | Fixed, §9aas (`compares_one_value` keeps one value compared across environments as one statement). Matrix A17. |
| X13 | Med | A hedged change was not flagged; once flagged, the warning took the sentence's place (de-dup knew only "Possible conflict" warnings) and Disaster Recovery was reported Missing ("stated as not in place") | Fixed, §9aas (`find_unconfirmed_changes`, "Unconfirmed change" warning; de-dup and scorer treat it as a quote; `dialogue.states_something`; topics judged per sentence) |
| X14 | High | KT taker: a question-and-answer KT was titled "It's the Appointment Booking" (dev d05); the system is Fernhill | Fixed, §9aas (`_infer_system_name`: never a pronoun phrase; the name said twice mid-sentence) |
| X15 | High | KT taker: speaker labels and questions were printed as facts ("Nadia: How do I deploy?" was a deployment step; Tribal Knowledge listed "Nadia: What should I..." as safety-critical with no source); "Can I run through my questions?" / "Sure, go ahead." sat in System Overview | Fixed, §9aas (`dialogue.display_text` after classification; talk requests and their acknowledgement excluded; deployment steps keep a question with its answer). Matrix T01, T02. |
| X16 | Med | KT taker: discussed topics reported "not discussed": rollback said as "swap back", "About twelve thousand a day", "goes offline", fixes said as "clear", "reset", "spread", users said as patients or clinics. The accuracy harness never measured topic-level gaps | Fixed, §9aas (`coverage_topics.py`); the matrix now checks topic gaps (T01) |
| X17 | Med | A fix right after its failure ("Reset the tunnel ...; if it doesn't come back, call the network team") went to Ownership for its "call ... team", and Common Failures said "Fixes: not covered" | Fixed, §9aas (fix-after-failure continuation rule). Dev d05 F19 now correct. |
| X18 | Low | A pronoun-only sentence after an answer ("Only six clinics have it.") is placed on its own; an unpunctuated speech-to-text run-on ("the app is on eks and the db is rds ... argo cd does the deployment") is one fact | Open: matrix B09 (should). Needs answer-turn grouping and punctuation restoration. |
| X19 | Med | With Groq, the section polish wrote filler for template labels nothing filled ("**Window:** N/A", "**Rollback trigger:** Not specified"; 18 of 133 cached answers) and reworded what was said, so 15 of 81 facts on two dev KTs no longer read as said | Fixed, §9aat: shared polish system prompt (keep the speaker's wording, omit unfilled labels, fragments are data) and deterministic removal of placeholder lines. Strict 65 to 69 of 81, lenient unchanged at 78; output tokens per polish call down about 14% |
| X20 | Med | PDF: the cover said nothing about the handover; coverage status was plain text; sections of short facts were walls of paragraphs | Fixed, §9aat: "At a glance" tiles on the cover (covered well, partly, not covered, to confirm, with page links), coloured status badges, short facts as a list, tighter spacing (the user's KT 15 to 14 pages) |
| X21 | Med | PDF: the Quick Reference (the incident page) sat after the coverage report, on page 9 to 11; its rows quoted up to 120 words of speech; each empty table cell said "Not covered during KT" | Fixed, §9aau: the card is section 01 (`knowledge_builder.lead_with_quick_reference`); a row over 45 words keeps its first whole sentences (never a safety, conflict or no-fix row); an empty cell is a dash, with one note under the table |
| X22 | High | A correction after a dash and "actually" ("seven days—actually, the new retention policy is 30 days") was cut into two facts at the marker, the marker was lost, and with Groq Disaster Recovery listed "Database backups retained for seven days" on its own | Fixed, §9aav: `split_facts` keeps a correction with the value it withdraws (`withdrawn_values`); extracted values, polish and copy edit that show a withdrawn value alone are refused (`grounding.ground_structured`, `ai._polish_one`, `copyedit.problems`) |
| X23 | High | Without an LLM, "Hi team, today I'll explain our order service" titled the KT "Today I'll Explain Our Order" (also the diagram and connections table) and was printed under System Overview | Fixed, §9aav: names start after the last determiner or introducing verb; a one-word lower-case name keeps its descriptor; more announcement verbs |
| X24 | Med | Words of other topics covered sections: "we upgraded it last week" (30-day plan), pod replicas (DR failover), "business hours ... approves" in a never-do rule (deployment window, approvals) | Fixed, §9aav (`coverage_topics`, `states_topic_elsewhere`) |
| X25 | Med | A section only mentioned in passing was Strong ("covered well" on the cover); the "Not Covered" list disagreed with the matrix (named Sign-off, left out Handover Completion Check) | Fixed, §9aav: covered well needs most topics covered by the section itself; the list is the matrix's (`pdf_rendering.missing_sections`) |
| X26 | Med | "three replicas in staging, but production runs five" was followed by "not discussed: Differences between environments" | Fixed, §9aav: contrast pattern, topics judged on whole sentences too |
| X27 | Med | "We haven't documented ... yet, so that needs confirmation" and "I'm not completely sure" were not stated gaps; a guess then read "stated as not in place" | Fixed, §9aav: contractions, more doubt wording; uncertainty reported as unconfirmed (`dialogue.is_unconfirmed`) |
| X28 | Low | "If a deployment fails, check the logs first ..." printed as "Pre-deployment checks" (semantic fill) | Fixed, §9aav |
| X29 | Low | "GitHub Actions; Helm" (two names said in two sentences) marked "no source" | Fixed, §9aav (`evidence.sources_for_row`) |
| X30 | High | A Docker container ran every KT without an LLM: compose passes only the Groq key, and `LLM_PROVIDER` defaulted to Gemini | Fixed, §9aaw (`llm_provider._provider_from_env`) |
| X31 | Med | Without an LLM, "If CPU stays above 75% ..., we should scale up" went to Cost Optimization and "Redis is using 4 GB of memory" to Common Failures | Fixed, §9aaw (runbook rule allows "we should"; a component's size is Architecture) |
| X32 | Low | With Groq, a reworded restatement of an Environments row was printed again as an "Additional environment notes" line | Fixed, §9aaw (`renderers.blocks.common.not_shown_in`) |
| X33 | High | 13 ways of saying "not known yet" or "not sure" were not flagged (the user's six KTs) | Fixed, §9aax (`dialogue`) |
| X34 | High | A value given on an old document's word ("RPO four hours, or at least that is what the previous document says") printed as the RPO, also on the Quick Reference | Fixed, §9aax (`_DOUBTED_SOURCE_RE`; RTO/RPO through `is_field_candidate`) |
| X35 | Med | A sentence that only doubts the one before ("That value needs confirmation") was filed apart, as a deployment step | Fixed, §9aax (`dialogue.doubts_statement_before`) |
| X36 | High | False "Conflicting guidance" on the Quick Reference: an environment after the count, an "outdated" count | Fixed, §9aax (`conflicts._counts`, `_PAST_RE`) |
| X37 | Low | "we roll back" printed "we rollback"; "fail over" likewise | Fixed, §9aax (noun only) |
| X38 | Low | Instructions about the document printed as a deployment step or a "never" rule | Fixed, §9aax |
| X39 | Low | "no source" on name lists, quotes, category values and screen links | Fixed, §9aax (`knowledge/evidence.py`) |
| X40 | Low | One screen URL listed once per dashboard tab (ten times) | Fixed, §9aax (`screen_capture`) |
| X41 | Med | A condition cut from its sentence ("because if the platform becomes unavailable."); a greeting left under System Overview | Fixed, §9aax (`fact_units`, `copyedit.problems`, `dialogue.is_greeting_only`) |
| X42 | Med | Eight Whisper mishearings on the user's recordings ("Diplomats must be avoided", "pavement orchestration", ...) | Fixed, §9aax (gated corrections) |
| T2 | Product | Titles dropped "platform"/"system" from names of more than one word ("Online Grocery") | Fixed, §9aax, the user's decision: the name as said ("Online Grocery Platform"); a capitalised proper name stands alone ("CorePay", "Bank of the West") (`field_populator.name_with_descriptor`) |
| X43 | Med | Security "came up in passing" because "Students access the React application" (a user's KT) | Fixed, §9aay (`coverage_topics`) |
| X44 | High | "... are still undocumented" was not a stated gap: missing from the follow-up list, and its escalation contacts counted Ownership & Escalation as covered | Fixed, §9aay (`dialogue`) |
| X45 | Med | "escalation contacts" was filed under Handover Completion Check | Fixed, §9aay (Ownership routing rule) |
| X46 | Low | The greeting was cited as the source of a connection | Fixed, §9aay (`knowledge/evidence.py`) |
| X47 | Low | "no source" on an ownership team cell and on the handover check's placeholders; a long tool list cited half its sentences | Fixed, §9aay (`knowledge/evidence.py`) |
| X48 | Low | Backup details listed as "Recovery actions" | Fixed, §9aay (`renderers/sections/disaster_recovery.py`) |
| X49 | Med | PDF: the connections table ran 18pt past the page edge; a status badge spilled out of its column | Fixed, §9aay (CSS; a layout test measures every box) |
| X50 | Low | The card dropped "before restarting services" ("No fix was stated"); without an LLM, "A common issue occurs when ..." was not a named failure | Fixed, §9aay (`knowledge_builder`) |
