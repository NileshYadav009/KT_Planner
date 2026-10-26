# Continuum: engineering record

This is the living record for anyone, person or agent, changing the KT pipeline.
`docs/REPOSITORY_AUDIT.md` holds the evidence for each change, by section number
(§9abf is the latest batch). `docs/KT_PIPELINE.md` describes each stage in
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
unseen KTs. Without an LLM it is 88/95 on the holdouts, 109/114 on the blind
set, 118/132 on the second blind set (b04-b07) and 151/160 on the sealed
third (b08-b11): 466/501, 93.0% (§9abd; 441/501 before; unchanged by §9abe).
With Groq's section check, 93/95, 110/114, 125/132 and 154/160: 482/501, 96.2%
(§9abd; 321/341 on the first three before). On the 36 visible KTs as speech to
text without punctuation (`evaluation/noisy_copies.py`), 1057/1085 (97.4%), with
1 discussed section reported Missing (§9abf; 1024/1085 and 13 before §9abe,
1048/1085 before the punctuation model).

## Pipeline map (a pasted transcript; a recording first goes through `pipeline.transcribe_recording`)

| Stage | Code |
|---|---|
| Upload, queue | `api/routes.py`, `job_queue.py`, `worker.py` |
| Clean (vocabulary, fillers) | `devops_transcription.clean_transcript` (a run of 120+ words with almost no punctuation first gets sentence ends from a small model: `punctuation.restore_punctuation`, `PUNCTUATION_MODEL=0` turns it off) |
| Speakers and roles (a conversation) | `captions.to_turns` (a Teams or Zoom WebVTT/SRT file as "Name:" turns), `speaker_roles.turn_segments` (pasted "Name:" turns), `detect_roles`, `apply_roles` (in `segment_sentences`): facts from the giver; the receiver's words only where the giver confirmed them. Speakers are named afterwards with `POST /kt/{id}/speakers` (`sessions.rename_speakers`) |
| Sentences, facts | `context_mapper.segment_sentences` (a correcting sentence kept whole with what it corrects: `replaces_value_before`, `points_back_to`, `withdraws_said_before`; a long unpunctuated run cut by `fact_units.even_pieces`), `fact_units.split_into_facts` (a later onboarding time frame opens a fact: `_time_frame_pieces`) |
| Self-corrections, copy edit | `devops_transcription.resolve_self_corrections`, `merge_spoken_corrections`, `copyedit.copyedit_sentences` |
| Classify | `context_mapper.ContextClassifier._score_sentence_candidates` (bge-large, examples, hints, GLiNER, topic segments, and the learned section model `section_model.npz`: `SECTION_MODEL_WEIGHT` x its probability; rebuilt by `scripts/build_section_model.py`) |
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
| Cleaner, punctuation, copy edit, polish guards | `tests/test_devops_transcription.py tests/test_punctuation.py tests/test_copyedit.py tests/test_polish_integrity.py` | about 30 s |
| LLM controls, no network | `tests/test_llm_cost_controls.py tests/test_llm_provider.py tests/test_llm_batching.py` | about 25 s |
| Fields | `tests/test_field_populator.py` | about 15 s |
| Knowledge and coverage | `tests/test_knowledge_builders.py tests/test_fact_units.py tests/test_dialogue.py` | about 3 min |
| Web page structure (ids, tabs, coverage source) | `tests/test_frontend_accessibility.py` | seconds |
| Learned section model, examples | `tests/test_section_model.py tests/test_section_examples.py` | about 20 s |
| Speaker names, caption files, giver choice | `tests/test_speaker_names.py tests/test_giver_choice.py` | about 5 min |
| End to end, with recorded Groq replay | `tests/test_e2e_kts.py` | about 13 min |

Accuracy: `python evaluation/accuracy.py LABEL tuning,dev,holdout,blind2,blind3,e2e`
(about 5 min; see `evaluation/README.md`). Every placement or coverage change
is measured on all six sets; holdout, blind2 and blind3 are read in
aggregate only. `blind4` (b08-b11) is sealed: read once, at the end of a
batch of changes, never while choosing between designs. A choice between
designs (a weight, a threshold, a learned model) is made leave-one-KT-out
on the visible tuning and dev KTs. `python evaluation/noisy_copies.py LABEL`
(about 2 min) runs the same visible KTs as speech to text without
punctuation, on their own keys.

Quality gates: `python evaluation/quality_gates.py LABEL --replay --accuracy ACC
--matrix M` (about 25 min): the release thresholds counted on the PDFs of the
six e2e KTs, with no LLM and from the recorded Groq answers. `--report BEFORE
AFTER` rewrites `docs/QUALITY_GATES.md`. A failing gate is a defect, never a
reason to loosen an answer key.

Adversarial matrix: `python evaluation/matrix.py LABEL [IDS]` (80 short KTs,
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
| L3 | Med | The classifier makes 21 of the 22 placement errors on unseen KTs; the LLM check sees only ids and titles, and only the top 3 | Measured twice, +1 each time, no loss: dev +1/224 (§9aap); holdouts and blind with only the section check live, 199 vs 198 of 209 (§9aar). Costs about 25% more tokens in the check. Measured a third time with the user's go-ahead (§9aba, section check only): 322 vs 321 of 341 unseen facts, one more section wrongly Missing, 37% more tokens on the check. Kept off (`LLM_VERIFY_CANDIDATES=5`, `LLM_VERIFY_PURPOSE=1` to switch; a replay re-record follows). The classifier change that paid is the learned section model (X83, §9abd). |
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
| X18 | Low | A pronoun-only sentence after an answer ("Only six clinics have it.") is placed on its own; an unpunctuated speech-to-text run-on ("the app is on eks and the db is rds ... argo cd does the deployment") is one fact | Open: matrix B09 (should). Needs answer-turn grouping and punctuation restoration. Parts done in §9abe: a later onboarding time frame is cut out (X87), a long run is cut evenly (X89); see X90. |
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
| X51 | Med | Missing architecture dependency: "The backend uses Python FastAPI, PostgreSQL on Amazon RDS, and Redis" drew no RDS line (the workload heads the list; the engine stopped naming its managed database) | Fixed, §9aaz (`architecture_diagram`) |
| X52 | Med | A store opening its sentence ("PostgreSQL on Amazon RDS stores order and payment records", "Redis caches ...", "Amazon MSK carries ... events", "... is stored in Redis") drew no line (QuickBite) | Fixed, §9aaz (`architecture_diagram`; not object storage) |
| X53 | Med | Quick Reference: "An alert fires" quoted a sentence that only says where alerts come from as the response | Fixed, §9aaz: "Alerting (no first step stated)" |
| X54 | Med | Quick Reference without an LLM: failures said as a list ("The two incidents ... are X and Y", "The common problems are X, and Y"), after a time ("During dinner peaks, the most common issue is ..."), by their effect, or as a past incident were missing or merged; "If you see that again" became a situation; "Avoid production deployments during ..." was not a freeze | Fixed, §9aaz (`knowledge_builder._known_failures`) |
| X55 | Med | A failure named without its fix, beside one with a fix: Common Failures "covered well" and no gap listed | Fixed, §9aaz: Partial, the failure on the follow-up list and in the section's note |
| X56 | Low | The rollback row ("Rolling back is usually quick") did not say no procedure was given | Fixed, §9aaz |
| X57 | Med | False "no source" on values in the document's form ("DR testing frequency: quarterly", "30-day retention"; LLM mode) | Fixed, §9aaz (`knowledge/evidence.py`: a labelled value is sourced by its value; value words) |
| X58 | Low | The diagram's users box said "customers" whatever the KT called its users | Fixed, §9aaz |
| X59 | Med | No release gates: layout checked in one test only; false "no source" marks and layout not counted by the e2e harness; the facts ledger had no provenance, score or "shown in its own section" | Fixed, §9aaz (`evaluation/quality_gates.py`, `pdf_rendering.layout_overflows` on every PDF, ledger fields) |
| X60 | Low | Web page: progress and toasts not announced to screen readers | Fixed, §9aaz (live regions, progressbar) |
| X61 | Low | A deployment chain said in one sentence ("GitHub Actions runs tests, Amazon ECR stores images, and ArgoCD deploys to EKS") is one step, with and without an LLM | Open: splitting it drops the joining words, which breaks the verbatim checks of every golden set (holdouts and blind included); the steps are the speaker's words in order |
| X62 | Low | The Architecture Knowledge name list repeats the Technology summary and the diagram | Open: component-name facts may be found only through it; measure before removing |
| P2 | Product | Optional sections (Environments, Cost Optimization) not discussed were left out of the coverage report and the follow-up list | Done, §9aaz, the user's decision: a "Not discussed" row each (`OPTIONAL_NOT_DISCUSSED`, `full_schema`) and a follow-up line asking whether it applies; not counted as a section not covered; no section added to the body. Matrix R06 |
| X63 | Med | A topic the speaker put at the head of a sentence ("Security wise, ... and the API sits behind Identity-Aware Proxy ...", "security, ... and the certs are rotated every year by the roadside team") was lost by the second fact cut from it, which went to Architecture or Ownership | Fixed, §9aba (`apply_continuation_rules`: the fronted topic holds for every fact of its sentence unless a strong rule of the fact's own claims it) |
| X64 | Med | A sentence about one environment ("staging uses recorded gantry traffic from last year") went to Deployment | Fixed, §9aba (environment-subject rule, 0.945, below recovery, monitoring and security rules; it does not move the topic) |
| X65 | Low | The fix after a failure said without a failure word ("... grows and slows logins. Run the purge job, it is safe in business hours.") went to the calendar | Fixed, §9aba (`_FAILURE_RE`: slows, throttled, stalls, saturated, runs out of, stale, degraded) |
| X66 | Med | The KT's opening definition ("Brightline is our data pipeline that loads ... into Snowflake") and who builds on it ("... teams build all their reports on top of it") went to Architecture | Fixed, §9aba (only before any topic is named, so a component defined later stays Architecture) |
| X67 | Med | A fresh blind set (b04-b07, 132 facts) reports 4 discussed sections as Missing without an LLM | Partly fixed, §9abd: four dev KTs in its styles (d07-d10) showed one cause (X77, fixed); b04-b07 now report 3 (with the learned model; 4 with the rule fixes alone). §9abe: four more dev KTs in its styles (d11-d14) reproduced no false Missing; noisy copies of the visible KTs showed two causes (X87, X88, fixed: 13 to 1 there), but b04-b07 still report 3. The rest is open: aggregate only, never inspected |
| X68 | High | A conversation's receiver was a source like the giver: "So it stores invoices in MySQL, right?" printed as "It stores invoices in MySQL." after "No, it's PostgreSQL"; the receiver's corrected "30 days" stayed; the giver's "three nodes" stayed after he agreed it was five; an unanswered "Isn't it PagerDuty now?" made PagerDuty the alerting tool | Fixed, §9abb (`speaker_roles.py`: roles, then what counts; matrix S01) |
| X69 | High | Pasted "Name:" labels were stripped for display but never recorded: no speaker in Sources, so nothing could tell giver from receiver | Fixed, §9abb (`turn_segments`, also on a session rebuild) |
| X70 | High | The cleaner turned a tag question into a statement ("..., right?" to "."), so a receiver's wrong guess read as a fact | Fixed, §9abb: in a conversation the question keeps its mark (`remove_fillers(keep_tag_questions)`) |
| X71 | Med | Small talk ("How was the offsite?" / "Good, the food was great.") described the Production environment; "Can you repeat that?" was joined to the answer after it; "Let's not go down that path" was a fact | Fixed, §9abb (phrasing, not nouns: a food-delivery KT keeps "food delivery platform") |
| X72 | Med | "? No, it's X" was read as a stated gap; a tool guessed in a question turned down, or "the old system", was listed as current | Fixed, §9abb (`dialogue`, `field_populator.current_tool_mentions`) |
| X73 | Low | "You'll own the on-call rotation" was carded as who to page ("call Rotation", the pattern ignoring case) | Fixed, §9abb |
| T3 | Product | The user's decision: the giver is detected, shown on the KT page and in the document, and can be changed, which rebuilds the document as a new version | Done, §9abb (`POST /kt/{job_id}/giver`, "Who is handing over" card) |
| X74 | Med | The web page counted the classifier's per-section statuses, not the coverage matrix: a KT's page said 13 sections "weak" and none covered while its PDF said 4 covered well, 9 partly and 2 not; Sign-off was listed as a missing critical section | Fixed, §9abc (`static/index.html` reads `_coverage_rows` and `_knowledge_gaps`, as `pdf_rendering.handover_glance`) |
| X75 | Low | Web page: one long page (the upload card above every KT, empty cards, the job ID, schema and form buttons); no link to a KT or a tab; at phone width the KT list was hidden with no way to open it | Fixed, §9abc (start page and one page per KT with tabs; `?job=` and `#tab` links; menu button) |
| X76 | High | The cleaner changed meaning: "Dev uses Stripe's test mode" became "Dev uses stress test mode" (the fuzzy term corrector scored "stripe's test" 0.92 against "stress test") | Fixed, §9abd: a word with an apostrophe is never a garbled term (`apply_fuzzy_term_corrections`) |
| X77 | Med | A sentence opening with the new owner's time frame went elsewhere and First 30-Day Ownership was reported Missing: "In the first month take over the black friday preparation" (calendar), "Within the first month, the new owner should ..." (Ownership), "First day get access to ..." (dev d08, d09; the X67 symptom on visible KTs) | Fixed, §9abd (`section_rules.ONBOARDING_MONTH_RX`, `ONBOARDING_DAY_RX`, 0.97; not "the first week of January") |
| X78 | Med | A numbered heading read aloud ("Section eleven, costs.") named no topic; the cost after it went to System Overview | Fixed, §9abd (`topic_anchor` drops the number) |
| X79 | Low | "Twilio rate limits us when there's a big batch like on black friday" went to the Operational Calendar | Fixed, §9abd (common-failures rule: "rate limits us", "throttles us") |
| X80 | Med | Open Responsibilities holding only "The failover procedure and recovery runbook are still missing" was reported Missing beside it (QuickBite e2e, once the learned model filed the sentence there) | Fixed, §9abd: there a stated gap is the section's content, its "Open work" topic (`coverage_topics.GAPS_ARE_CONTENT_SECTIONS`), and it stays on the follow-up list ("still to do") |
| X81 | Low | `section_examples.json` was out of date with the goldens since 5c0e5d7 ("roll back" respelled); its test failed | Fixed, §9abd (rebuilt) |
| X82 | Low | A heading read aloud ("Known problems.", "Section two, architecture.") is printed as a bullet in its section | Open: leaving headings out changes which topics count as discussed; measure first |
| X83 | High | Without an LLM, placement on unseen KTs was 86.8% (296/341), under the 90% target; the classifier's own first choice made most of the errors (L3) | Fixed, §9abd: a learned section model (`section_model.npz`: logistic regression over the fact's and its context's embeddings, trained on the synthetic tuning and dev KTs; `SECTION_MODEL_WEIGHT` 0.1), chosen leave-one-KT-out on visible KTs. Unseen 441 to 466 of 501 (93.0%) with X76-X79; the sealed set b08-b11 145 to 151 of 160 |
| X84 | Med | Deployment steps cut a corrected unit into one step per sentence, so "The deploy freeze starts at 6 pm on Fridays." stood alone before its correction (matrix A10, once the learned model filed the pair under Deployment); "deploy freeze" was not a calendar freeze | Fixed, §9abd (`renderers/sections/deployment._steps` keeps a correcting sentence with its step; the freeze rule reads "deploy freeze") |
| T4 | Product | A recording's speakers showed as "Speaker 1, 2"; a Teams or Zoom transcript file was pasted with its timings and tags and nobody named | Done, §9abd, the user's decision: name speakers on the KT page (`POST /kt/{job_id}/speakers`, a new version), WebVTT and SRT files read as speaker turns (`captions.py`), people listed on upload offered as names. Fetching them through the Teams or Zoom API is not done (needs an app registration) |
| X85 | High | A correction that says again what it withdraws ("Actually, I said earlier we deploy on Tuesdays. That's wrong, it's Wednesdays now"; "Earlier I said the cluster has three nodes; that's not right, it has five") left the old statement alone in Deployment, its two halves in two other sections (dev d14); a day corrected ("Releases go out on Thursdays. Sorry, Fridays.") showed both days | Fixed, §9abe (`withdraws_said_before`, `restates`; days of the week are values: `_WEEKDAY` in `_CORRECTED_VALUE`). Matrix A22-A24 |
| X86 | Med | Where credentials are kept ("The Stripe and Postmark keys are in Heroku config vars") went to Environments; "Card details never touch our servers" to Danger Zones (the "never touch" rule); "Phone numbers are personal data so we delete message logs after ninety days" and "undoing it means a database restore" to Disaster Recovery; "The thing you really must not do is run ..." matched no rule (dev d07, d08, d14) | Fixed, §9abe (`section_rules`) |
| X87 | High | Speech to text without full stops ran an onboarding plan's two sentences together ("in the first week get an account ... by the end of the month run a thursday deployment yourself"); the month's task went to the Day-1 checklist and First 30-Day Ownership was reported Missing on 9 of 36 noisy copies of the visible KTs | Fixed, §9abe (`fact_units._time_frame_pieces`; `ONBOARDING_MONTH_RX` reads "by the end of the month", "by week three", "within thirty days" followed by words to the new owner) |
| X88 | Med | A control stated before a gap with no contrast word ("We take nightly backups to S3 and we have never tested a restore"; a noisy run) was read as a gap alone, and Disaster Recovery reported Missing, "stated as not in place" (3 noisy copies); a sentence-final "though" was read as a contrast mid-sentence | Fixed, §9abe (`dialogue.states_something_before_gap`: not after a doubt, a topic put off, a relative clause or a short question that lost its mark) |
| X89 | Low | A run of more than 240 characters with no punctuation was filled up to the limit: cut mid-phrase ("nightly indexer \| job that pulls ..."), its last piece a stub dropped as under four words ("task definition revision."); its source quote was the run cut to 280 characters, so a fact from its end cited a quote that did not hold it | Fixed, §9abe (`fact_units.even_pieces` for the units and for the Sources list, each piece marked "…") |
| X90 | Med | A whole KT with no punctuation and no paragraph breaks (the noisy copy of g11) places 46% of its facts; cut points inside an unpunctuated run are guesses | Partly fixed, §9abf, the user's decision (install, 210 MB): a run of 120+ words with almost no punctuation gets its sentence ends from a small English model (`punctuation.py`, `punctuators` "pcs_en", pinned); g11 46% to 83%. Paragraph-length runs are left to `fact_units` (the model was no better there) |
| X91 | Low | "Changes to the MQ bridge configuration require approval from the mainframe team" went to Ownership (the named team); "Scale the web dynos to twenty before the appeal starts" to Cost Optimization (dev d13, d14) | Open: one sentence each, and each could belong to two sections |
