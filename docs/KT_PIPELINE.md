# The KT Pipeline

This is `context_mapper.py`'s 7-stage classification pipeline plus the
field-population and knowledge-object-assembly stages that follow it — the
part of the system that turns a transcript into structured KT content. For
where this fits in the overall request flow, see `ARCHITECTURE.md`.

## Stage 1 — Audio confidence

Each transcribed segment carries Whisper's `avg_logprob`. `confidence_score()`
normalizes this into a 0-1 range (logprob -2.0 -> 0%, 0.0 -> 100%) — used
later to weight section-level confidence and to trigger LLM-based repair for
low-confidence sentences (`ContextRepair`, Stage 4).

## Stage 2 — Segmentation into facts

Raw segments are joined and split into sentences, keeping each sentence's
time from the Whisper segments it came from. A run-on (speech with no full
stops) is cut at its sentence openers.

**Each sentence is then cut into the facts it states**
(`fact_units.split_into_facts`, via `context_mapper.split_facts`). The unit
the rest of the pipeline classifies, fills fields from and counts is a fact,
not a spoken sentence: "Developers own business services, Platform
Engineering owns EKS and deployment tooling, and the Data Platform team
manages MSK" is three ownership facts, and "Recovery testing is performed
twice a year, but the RTO and RPO have not been documented" is a recovery
fact and a stated gap. A cut is made at ", and", ", but", ", while",
", whereas", ";" or the comma of a list of clauses, and only when the pieces
on both sides are statements of their own (an explicit subject and a finite
verb). Clauses that lean on each other stay one fact: a pronoun or connective
opener ("and it is owned by..."), a predicate with no subject ("..., pushes
them to ECR"), an explanation ("..., and the fix is to..."), the result of
the step before ("..., and the results land in BigQuery"), a rule and its
reason ("Never deploy on Fridays, we had an outage"), a sentence that opens
with a condition, and lists of nouns. The rules are grammatical; no section,
tool or transcript is named in them.

Every fact keeps its source: `Sentence.source_id` (the spoken sentence's
position), `source_text` (the sentence verbatim) and `source_span` (the
fact's characters in it). Statements about the session itself are left out
here and recorded with the reason (`StructuredKT.excluded_statements`).

Nothing is merged before classification except a question with its answer
(`dialogue.merge_question_answers`); an answer is not cut into facts, so the
question's topic reaches all of it. Sentences used to be merged when they
read alike (`semantic_chunk_sentences`, removed 15 Oct): the merged unit got
one section, so a system's purpose went to System Overview with its request
flow, and a scheduler with the build, deploy and rollback chain. A sentence
that depends on the one before ("The fix is to ...") follows it by the
continuation rules in `section_rules.py`, and each fact is scored with the
spoken sentences around it (two either side, counted in sentences, not
facts) as context.

**A number corrected in the same sentence** keeps only the corrected one
(`devops_transcription.resolve_self_corrections`, audit §9aan), before the
copy edit and with or without a model: "the order limit is 4000 or sorry not
4000 but 2000" becomes "the order limit is 2000". It needs a number on both
sides of a marker that only means a correction ("sorry", "I mean", "or
rather", "no wait", "correction"); "actually" and a bare "no" do not count.
The Sources appendix still quotes the sentence as said.

**A value corrected by a later sentence** is never shown on its own. A
sentence that only restates the number is rewritten into the one before
(`merge_spoken_corrections`, §9aap). One that says more ("Actually, the
upgrade is complete; production is now on PostgreSQL 16") is kept whole with
the sentence it corrects, as one unit, and one that points back ("I was wrong
about the backups earlier, ...") takes the sentence it corrects along
(`segment_sentences`, §9aaq). The rules, and what counts as a correction, are
in `docs/CORRECTIONS.md`.

**Stage 3.6 — one unit per sentence and section** (`rejoin_classified`):
once every fact has its section, the facts of one sentence that sit side by
side in one section become one unit again, shown as the spoken sentence.
Facts of one sentence placed in different sections stay apart. The section
text, its fields and its renderer then read the same unit; the atomic facts
are kept for the ledger (`StructuredKT.facts`, `fact_sections`).

## Stage 3 — Semantic classification

Each sentence is embedded (`BAAI/bge-large-en-v1.5` by default) and compared
against every schema section's own embedding (built from its title/
sub_topics/hints) and its labelled examples (`section_examples.json`) via
cosine similarity; the five best are kept. Produces a primary
classification plus (for genuinely close calls) secondary candidates. The
cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) was removed in audit §9aam:
against a section title it scored about -10 for nearly every pair and added
nothing measurable. Scores keep the 0.4 scale its blend gave them
(`CLASSIFIER_SCORE_SCALE`), which the thresholds were set on.

**Topic segments** (`section_rules.segment_topics()`, audit §9aaj): every
fact gets the topic the speaker last named, and that section gets
`TOPIC_SEGMENT_BOOST` (0.04). A topic is named by a short question
("What breaks most often?"), a heading read aloud ("Known problems."), a
fronted phrase ("Costs, ...", "Security wise, ...", "Unfinished: ..."), or
a sentence a routing rule decides at 0.94 or above. Before any topic is
named, the topic is System Overview. The boost decides a sentence that says
little by itself ("Only six clinics have it." after "Anything
unfinished?"); a sentence with clear evidence of its own still goes
elsewhere, and routing rules still decide over it. Disaster Recovery also
loses `RECOVERY_WITHOUT_EVIDENCE_PENALTY` (0.03) for a sentence with no
recovery word (`RECOVERY_EVIDENCE_RE`): its examples are full of durations,
so a fix "for an hour" used to score closest to it.

**Entity-affinity boost** (`section_rules.py`'s `entity_affinity_boost()`):
sentences containing entities of a type strongly associated with a section
(e.g. a tool name for `monitoring_observability`) get a scoring nudge toward
that section — this exists so classification isn't purely lexical-similarity
driven.

**Selective LLM verification** (`_maybe_verify_with_llm()`): only for
genuinely borderline cases (close-scoring candidates), not every sentence —
an LLM call asks which of the top candidates is actually correct. Skipped
entirely when no LLM provider is available.

## Stage 4 — Rule overrides + context repair

`section_rules.py`'s `SECTION_RULES` — a small set of high-confidence regex
patterns (e.g. "never modify" + "terraform state files" -> `danger_zones`
at 0.97 confidence) that override a weak semantic match. Exists specifically
to stop generic-sounding but specialized sentences from being absorbed by
`system_overview` (the "dumping ground" problem `tests/test_ecommerce_kt.py`
explicitly regression-tests against).

`ContextRepair` then optionally rewrites very-low-confidence
(`audio_confidence < 0.3`) or garbled sentences via an LLM, when available.

## Stage 5 — Gap detection / topic blocks

`detect_gaps()` groups consecutive same-section facts into `TopicBlock`s
(preserving paragraph continuity rather than treating every sentence in
isolation). Facts of one spoken sentence that sit side by side in a block are
shown as that sentence again (`rejoin_facts`); facts of one sentence placed in
different sections are shown apart, each in its own section. Each section is
scored with `semantic_coverage_score()` — a
4-dimension score (confidence / depth / actionability / completeness) that
favors coherent, complete coverage over raw sentence count. Status
(`missing`/`weak`/`covered`) comes from this score, not a simple sentence
count threshold.

**Known subtlety** (see `REPOSITORY_AUDIT.md` §9n): `section_content[id]`'s
`sentences` list (built by a separate multi-label classification loop) and
its `blocks` list (built by this stage's single-label grouping) can disagree
— a section can have real coverage in `blocks` while `sentences` stays
empty. Code that needs a section's raw sentences (`field_populator.py`,
`knowledge/knowledge_builder.py`) must fall back to flattening `blocks` when
`sentences` is empty, or it will silently see no content for a section that
actually has some.

## Stage 6 — Screenshot/URL extraction

Extracts URLs mentioned in the transcript as `ExtractedAsset`s. Screenshot
capture itself is disabled (see `REPOSITORY_AUDIT.md` §4.3) — `screenshots`
is kept as an always-empty list since the frontend reads it unconditionally.

## Stage 7 — Evidence / KT assembly

Produces the final `KTResult` (`kt.coverage`, `kt.section_content`,
`kt.overall_coverage_percent`, `kt.missing_required_sections`, etc.) that
`pipeline.py` consumes.

## After context_mapper.py: field population and knowledge assembly

1. **`schema_generator.generate_dynamic_schema()`** — builds this run's
   schema: adds detected-technology fields, includes/excludes optional
   sections based on `OPTIONAL_SECTION_INCLUSION_RULES` thresholds.
2. **`field_populator.populate_fields()`** — for each field in each section
   with a `fields` array: try a pattern extractor first (`PATTERN_EXTRACTORS`
   + field-id-specific regexes), then semantic similarity against the
   section's real sentences, then an LLM gap-fill prompt as a last resort.
   Sources from `section_content`'s raw per-sentence transcript text (falling
   back to flattened topic blocks — see the Stage 5 subtlety above), not the
   LLM-polished narrative, specifically so different fields in the same
   section don't collide onto the same broad text (`REPOSITORY_AUDIT.md` §9n).
3. **`ai.wrap_structured_as_fields()`** — for the schema sections with no
   `fields` array at all (`monitoring_observability`, `security_controls`,
   `disaster_recovery`, `ownership_escalation`, `cost_optimization`,
   `common_failures`), an LLM structured-JSON prompt
   (`llm/prompts.py`'s `SECTION_STRUCTURED_PROMPTS`) extracts the section's
   data directly, merged into `populated_fields` the same way.
4. **`knowledge.build_knowledge_object()`** — turns `populated_fields` into
   each section's `facts`/`entities`/`evidence`/`relationships` arrays
   (`knowledge/facts.py`, `entities.py`, `evidence.py`, `relationships.py`).
   Evidence attachment uses `field_populator.find_source_sentence_index()` to
   point each fact at the specific sentence that produced it, not a generic
   "first 3 sentences of the section" fallback.
5. **`validation.validate_pipeline_run()`** — non-fatal structural checks
   (Phase 19).
6. **`quality_score.compute_quality_score()`** — document-level score
   (Phase 22).
7. **`pdf_rendering.build_rendered_sections()`** — dispatches each section to
   its renderer, producing the typed blocks `render_pdf_html()` turns into
   the final PDF. See `PDF_RENDERING.md`.
8. **`knowledge.verify_document_coverage()`** — the fact ledger and the
   knowledge-coverage figures. Every fact the mapper classified is checked
   against the rendered document and recorded in
   `knowledge_object["_facts_ledger"]`: its text, source sentence and time,
   the section it was placed in, the sections it appears in, whether it
   states a gap, and its status — `mapped` (shown in a section), `unmapped`
   (shown under Additional Notes), `recovered` (shown nowhere, so added to
   Additional Notes by this check) or `excluded` (greeting, remark about the
   session, under four words, or a repeat), with the reason. The KT Coverage
   page's "Knowledge coverage" line is computed from this ledger, so a fact
   cannot disappear without a record. The Coverage matrix below it is
   separate: per section, which of its expected topics the session covered
   (`coverage_topics.py`). Each gap the speaker stated ("the failover
   runbook is still missing") is listed, quoted, under the Knowledge Gaps
   checklist.
