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

## Stage 2 — Sentence segmentation

Raw segments are split into individual sentences, preserving timestamps by
splitting each segment's char range proportionally across its sentences.
Adjacent very-short fragments get merged (`_merge_sentences`) so a
mid-sentence Whisper segment break doesn't produce two artificial sentences.

## Stage 3 — Semantic classification

Each sentence is embedded (`BAAI/bge-large-en-v1.5` by default) and compared
against every schema section's own embedding (built from its title/
sub_topics/hints) via cosine similarity, then reranked with a cross-encoder
(`ms-marco-MiniLM-L-6-v2`) for precision. Produces a primary classification
plus (for genuinely close calls) secondary candidates.

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

`detect_gaps()` groups consecutive same-section sentences into `TopicBlock`s
(preserving paragraph continuity rather than treating every sentence in
isolation) and scores each section with `semantic_coverage_score()` — a
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
