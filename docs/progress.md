# Continuum KT Planner — Progress

Handoff doc for picking this work up in a new session. Written 2026-09-04,
updated 2026-09-19, end of a long working session on
`feature/Dynamic_Schema_Builder`. For the detailed technical narrative behind
every item below — file paths, line numbers, before/after, verification steps —
see **`REPOSITORY_AUDIT.md`**, sections §1-9dd. **For a 5-minute overview
instead of either of these two detailed files, read `DELIVERABLE_SUMMARY.md`**
(not yet updated with the §9q-§9dd work below — still describes the
state through §9p). This file remains the quick-orientation index; the audit
is the full record.

**Update 2026-09-19, later same day (full architectural spec supplied;
fact-checked, one more bug fixed, gaps ranked)**: user supplied a complete
31-section "Enterprise KT Document Architect" spec — the same
knowledge-first vision as §9s/§9x, now written out in full (evidence
rules, fact-ledger coverage, one-sentence-to-multiple-knowledge-objects,
section-specific rendering, validation gates). Verified it against real
code rather than reacting wholesale: several things it assumes are already
true (Not-discussed posture, per-field evidence tracking, Day-1-as-
checklist, Danger-Zones-as-warnings, common_failures' anti-hallucination
guard) already match with no changes needed. Found and fixed one more
concrete bug the spec's §17 predicts almost exactly: a real transcript's
general escalation guidance ("If you are unsure about a change, involve
the appropriate owner") was rendering as if it were the On-call tool's
NAME, because `ownership_escalation`'s structured-extraction schema had
no field for general guidance separate from a tool name/ownership
statement — same schema-gap class as §9cc's DR-testing-frequency bug.
Fixed with a dedicated `operational_escalation_guidance` field, rendered
in its own block. Full suite: **168 passed, 0 failed**, up from 165.
Ranked the spec's 3 real remaining gaps honestly instead of starting a
rewrite blind: (1) one sentence producing multiple, section-independent
knowledge objects isn't implemented — confirmed the `multi_section_
assignments` field exists but is never actually populated beyond one
section, the single largest gap; (2) no fact-ledger dual-metric coverage
model (today's coverage is template-field-count only); (3) section-
specific visual rendering (architecture flow diagrams, deployment as a
timeline) is partial — a timeline block type exists but isn't wired to
deployment_and_rollback. Asked the user to pick one to scope properly
next rather than starting any of the three unprompted. See audit §9dd.

**Update 2026-09-19, later still (Architecture Reference: knowledge vs.
metadata split)**: user picked gap #2 from §9dd concretely — "don't let
the template define what knowledge exists" — pointing at a real PDF
showing "Architecture Reference: 0 of 3 fields" despite the transcript
clearly describing a real component stack (EKS, React, CloudFront, ALB,
FastAPI, RDS PostgreSQL, Redis, SQS, ECR). Root cause: `architecture_
reference`'s schema only ever had 3 administrative fields (doc link, last
updated, verified-by) — no field for the actual component stack — and the
coverage matrix's assessment collapsed to a raw field-count that read as
"nothing captured" when real knowledge existed, just not shaped to fit
those 3 fields. Fixed by reusing the tools-detection regex already proven
in `enrich_technology_summary()` (added one gap: SQS wasn't in it) in a
new `enrich_architecture_knowledge()` that scans every section's raw
content and attaches the deduped component list to `architecture_
reference` as `_architecture_components`, independent of which section a
sentence classified into. Tagged the section `"fields_role": "metadata"`
in the schema (generic, reusable) so the coverage matrix now reports
knowledge-captured and metadata-discussed as two separate numbers instead
of one misleading field count. Renderer now always shows two blocks:
"Architecture Knowledge" (detected components, falling back to raw
content when none detected) and "Architecture Metadata" (all 3 admin
fields explicit, "Not discussed" when genuinely absent). Full suite:
**178 passed, 0 failed**, up from 168, golden end-to-end test included.
Scoped narrowly to `architecture_reference` only (the section the user
pointed at), not a full fact-ledger rewrite — mechanism is generic enough
to extend to another metadata-only section later. See audit §9ee.

**Update 2026-09-19, later still (performance: batched embeddings,
parallel LLM calls, context-aware verification — plus a mid-round
revert)**: user asked whether section mapping/data capture and PDF
generation speed could both improve. Found and fixed 3 real things:
batched `semantic_chunk_sentences()`'s per-sentence embedding calls into
one call (same embeddings, less per-call overhead); parallelized 3
independent per-section LLM call loops (prose polish, structured
extraction, and initially field gap-fill) via a thread-safe rate limiter
that was already in place (`llm_provider.py`'s `_throttle()`), so
concurrent dispatch sends no more requests per minute than sequential did;
gave the classification-verification LLM prompt the same ±2-sentence
context window the classifier itself already uses, instead of a bare
sentence, for the hardest/most ambiguous cases only. Shipped all three,
then found a real problem via testing: full suite time more than doubled
(523.90s -> 1129.52s) and a rerun of the heaviest 4 test files threw a
flaky `RuntimeError`. Root cause: `field_populator.py`'s per-section loop
does real CPU-bound embedding work (not just the LLM call) before ever
reaching the network, so parallelizing it across sections meant genuine
concurrent CPU compute (PyTorch releases the GIL for this) — a real
oversubscription risk on this project's resource-constrained laptop
target. Reverted that one call site back to sequential; kept the other two
(prose polish, structured extraction), which only wrap `provider.generate()`
itself with no CPU-bound work of their own. Re-verified: full suite back
to **187 passed, 0 failed, 635.23s** (in line with the 523.90s baseline
once the 9 new tests' own real cost is counted), no more flakiness. See
audit §9ff.

**Update 2026-09-20 (Architecture Reference: in-depth detail + generated
flow diagram)**: user's concrete follow-up on the earlier knowledge/
metadata split — the flat component list was right but had no depth (a
bare "Redis" says nothing about its role), and wanted an actual generated
architecture diagram, giving two mockups (linear chain fanning out at the
compute hub to backing services, registry shown separately). Added: (1)
`_architecture_sentences` — the real transcript sentences (verbatim,
never paraphrased) that named each component, sourced from raw per-
sentence `section_content` with a `coverage_content` fallback, rendered as
a new "Architecture Details" block under the unchanged flat list; (2) new
`architecture_diagram.py` — classifies detected components into coarse
layers (frontend/cdn/load_balancer/compute/service/database/cache/queue/
registry) via a small extensible lookup, deliberately NOT a general
sentence-relationship parser (too fragile to generalize), and renders a
top-down ASCII tree diagram as a new "High-Level Architecture" block,
reusing an already-built-but-unused `CodeBlock` type
(`renderers/base.py`/`pdf_rendering.py`) — no new rendering machinery
needed. Returns no diagram at all when nothing resembling a request-flow
position was named, and falls back to a flat chain (no fan-out) when no
compute/orchestration hub was ever stated, rather than fabricating either.
Caught and fixed a real formatting bug (a stray arrow before the fan-out
branches) by manually checking generated output against the user's own
mockup before calling it done. Full suite: **199 passed, 0 failed, 9:00**
— normal timing, no flakiness. See audit §9gg.

**Update 2026-09-20, later same day (two real live bugs in the diagram
feature, found via real KTs — fixed, plus diagram expanded)**: user
supplied two real generated KTs (AWS E-Commerce, Azure Banking) with their
source transcripts — the AWS diagram was missing FastAPI/RDS/SQS entirely
despite being clearly stated early in the transcript, and the Azure
diagram rendered as just "Customer -> Kubernetes", nearly the whole real
stack invisible. Root-caused two real bugs, not guesses: (1)
`enrich_architecture_knowledge()`'s `coverage_content` scan was gated on a
global "nothing found anywhere yet" check instead of always running —
since a section's raw per-sentence data and its `coverage_content` are
populated by two independent mechanisms that can disagree (already known
from field_populator.py, should have been applied here from the start),
the instant ANY section's raw sentences matched something, every OTHER
section's real content was silently skipped for the rest of the run; (2)
the tools-detection regex only recognized "Amazon RDS"/"Amazon ECR" with
the prefix (bare "RDS" mentioned later in speech never matched) and had
zero Azure vocabulary — added bare acronyms plus the full Azure
equivalent set (AKS, Azure SQL, Service Bus, Blob Storage, Front Door,
Application Gateway, ACR, Bicep, Azure DevOps, Key Vault, Azure Monitor,
Application Insights). Also fixed two follow-on cosmetic issues caught
while verifying: acronym/branded-name duplicates ("RDS" and "Amazon RDS"
as two entries) now canonicalize to one, and a term first seen in
lowercase mid-sentence no longer permanently wins over a later properly-
capitalized mention. Also expanded the diagram itself per the user's
requested shape: separate CI/CD pipeline, IaC, secrets, observability, and
alerting flows alongside the main request-flow tree, each shown only when
actually named. Verified end-to-end by reconstructing both real failing
transcripts and running them through the actual code path, not just
synthetic fixtures — both now render fully. Full suite: **206 passed, 0
failed, 10:51** — normal timing. See audit §9hh.

**Update 2026-09-19 (fact-checked a detailed external re-review; fixed 2
real bugs, correctly rejected 1 false claim, precisely scoped and deferred
1 real architectural gap)**: user pasted their own structured, numbered
review of the fresh post-§9bb AWS PDF against the source transcript.
Verified every claim against real code and a real Groq-backed pipeline run
before touching anything (established practice this session). Rejected
one claim as not a bug: "Customer Reach" marked missing is correct
behavior — the field is explicitly defined as geographic/market scope
("regional or global"), and the transcript never discusses that; the
critique conflated it with delivery channel (web/mobile), a different
fact entirely. Fixed two real, precisely root-caused bugs: (1) dynamic
schema fields (`schema_generator.py`'s tech-triggered fields, e.g.
`cache_layer`) are detected from the WHOLE transcript but attached to one
fixed "home" section per technology — when the classifier legitimately
routes the actual sentence to a different section (confirmed: the Redis
sentence correctly lands in `architecture_reference`, not
`system_overview`), the field never sees it and stays permanently
unfilled; fixed generically in `field_populator.py` with a cross-section
fallback pool used only by dynamic fields as a last resort, with no
`source_chunk_index` attached (a wrong index would misattribute evidence,
worse than none); (2) `disaster_recovery`'s structured-extraction JSON
schema had no field at all for "how often DR testing happens" (only
backup/RTO/contact fields existed), so "DR testing is performed
quarterly" — sharing a sentence with a real backup fact — had nowhere to
go and was silently dropped; added a dedicated `dr_testing_frequency`
field end to end. Investigated and deliberately reverted a single-word
fuzzy-correction attempt for LLM-introduced typos (Trivy -> Trivi,
confirmed via live Groq output) after it corrupted "scanning" into
"scaling" in its very first test — shipped only the safe multi-word
version instead, plus prompt-level instructions (spell proper nouns
exactly, don't drop compound-sentence detail) as a lower-risk mitigation.
Precisely scoped but deliberately did NOT fix a real architectural gap:
compound sentences touching multiple sibling fields (e.g. Environments'
Production/Staging/Non-production) use a "first sufficient source wins"
cascade per field, so a field that finds a good match on one sentence
never gets to also pull its own relevant clause out of a second,
compound sentence — confirmed via LLM-free reproduction exactly which
sentence and which extraction stage is responsible; redesigning this
touches the core extraction cascade every field goes through, too large a
regression surface to rush. Also hit and fixed (in tooling, not the
product) the same stuck-Groq-run signature as before — this time
traced to the diagnostic script itself deleting the just-set Groq env
vars before they were read, not a real `.env`/quota issue; documented so
it doesn't recur. Full suite: **165 passed, 0 failed**, up from 163. See
audit §9cc.

**Update 2026-09-18 (the real fix for §9aa's section-mapping bug: a
hardcoded rule, not orchestration)**: user re-supplied the same 3 real
transcripts and asked to actually fix the mapping this time. §9aa's
narrower single-neighbor test had wrongly concluded the defect lived in
`ContextMappingPipeline.process()`'s block-building/topic-continuity
orchestration; re-testing with the pipeline's REAL ±2-sentence context
window immediately surfaced the true cause instead — a hardcoded
`SECTION_RULES` regex (`section_rules.py`) matching `amazon ecr` /
`container images...` and force-routing to `security_controls` at 0.95
confidence, completely bypassing the classifier (which was already
correctly picking `architecture_reference` on its own). Traced to a single
synthetic test transcript where an ECR mention happened to share a
sentence with a security-scanning statement, over-generalized into a rule
that broke every real transcript where ECR/container-image mentions are
just ordinary architecture facts. Fixed by removing the two overly broad
patterns from both places they appeared (`SECTION_RULES` and
`find_overview_reassignment()`'s exclusion list) — the original test still
passes via its other, genuinely-security pattern (`trevi`). Verified full
suite (**163 passed, 0 failed**) and, more importantly, end-to-end with a
real Groq LLM (`qwen/qwen3.8-27b`, user-supplied key used only as a
transient env var, never persisted to any file): both AWS's and Azure's
architecture/tech-stack sentences now land in `architecture_reference`
instead of vanishing into `security_controls`. Also hit and worked around
an operational issue: a separate LLM-free verification attempt hung for
~110 minutes due to a leftover Gemini API key causing rate-limit retry
storms across every sentence — killed once the Groq run gave cleaner,
more realistic confirmation anyway. See audit §9bb.


**Update 2026-09-15 (ground-truth line-by-line audit with real source
transcripts)**: user supplied the actual source transcripts (not just the
generated PDFs) for AWS/Azure/GCP KT documents and asked for a rigorous
line-by-line audit against real ground truth (spelling, data loss,
section-mapping errors), a precise hardcoded-vs-dynamic answer, and a
judgment call on which sections are overhead (explicitly: ask before
removing any). Ran all 3 real transcripts through the actual pipeline
(LLM-free) and inspected real pre-polish per-sentence classification data
rather than reverse-engineering rendered PDF prose. Found and fixed a
second real fuzzy-correction data-corruption bug (same class as the S3 bug
in §9v): jaro-winkler's heavy prefix-weighting let "providers" score 0.83
against the unrelated glossary term "process" (shared "pro..." prefix),
silently corrupting "Some payment providers are mocked in staging" into
"...payment process are mocked...". Fixed with an added Levenshtein
similarity guard (0.56 for this pair vs. 0.75 for genuine corrections)
that isn't fooled by a shared prefix alone. Also found and precisely
root-caused (but deliberately did NOT blind-fix, given regression risk) a
more serious bug: dense architecture/tech-stack sentences that are
immediately followed by a secrets-management sentence get misclassified
into `security_controls` — confirmed in 2 of 3 real transcripts, and
confirmed as the exact cause of "Cache Layer" always showing "missing" in
the coverage matrix even when Redis is explicitly named; for AWS this
sentence (naming the primary database, cache layer, and async queue)
doesn't appear ANYWHERE in the 13-page rendered PDF at all — a genuine,
severe, silent total-content-loss bug. Isolated testing proved the defect
is NOT in the classifier's scoring formula (both the raw scorer and full
`classify_sentence()` correctly pick the right section in isolation, with
or without neighbor context) — it lives in `ContextMappingPipeline
.process()`'s block-building/topic-continuity orchestration layer, which
needs a proper eval harness before it's safe to touch (ties to the
feedback-persistence + eval-harness recommendation from the "how do we
improve section-mapping accuracy" discussion earlier this session).
Documented with full reproduction detail for a future pass. Answered the
hardcoded-vs-dynamic question precisely: the section skeleton is a static
template, but a technology-triggered field set (`schema_generator
.generate_dynamic_schema()`, confirmed working correctly — e.g. GCP's
matrix correctly has no "Cache Layer" field since no Redis is mentioned)
and 100% of section content are genuinely dynamic per-transcript. Full
suite: **163 passed, 0 failed**, up from 161. See audit §9aa. The
"which sections are overhead" judgment call and any section removal are
intentionally left for the user's explicit decision, not made
unilaterally.

**Update 2026-09-14, later same day (3 targeted fixes from an external
critique of 3 real generated PDFs)**: user pasted a lengthy architectural
critique of fresh AWS/Azure/GCP KT PDFs proposing a full rewrite around
typed "Knowledge Objects"; verified its specific claims against the code
rather than taking them at face value. The proposed rewrite is the same
idea already evaluated and deliberately deferred earlier this session
(§9s) — declined to reopen that call — but 3 of its specific findings
were independently confirmed with concrete root causes and fixed: (1)
`common_failures`'s "How to Fix" column was being fabricated by the LLM,
not extracted — the structured-extraction prompt guarded `resolution`/
`preventive_action` against invented content but not `cause`/`fix`, which
was even declared non-nullable; fixed by extending the same grounding
instruction to all four fields; (2) `day1_survival_checklist`'s required-
access table dumped a whole raw sentence ("For new team members, review
Pub/Sub, Dataflow, BigQuery, GKE, ...") into a single cell instead of one
row per tool — the schema's only field declaring fixed row labels, but
that metadata was never actually used; fixed with a new sentence-splitter
scoped specifically to fields declaring `rows`, so no other table field's
behavior changes; (3) `handover_completion` could show only "KT status:
Complete" with none of the 5 substantive readiness checks visible at all
when they were never captured, reading as a false all-clear even in
documents whose own coverage matrix listed that section as Missing —
fixed to always list all 5 checks (defaulting to "Not covered during KT"),
relabel the closing line so it can't read as a computed verdict, and add
an explicit caveat when none of the checks were confirmed. Full suite:
**161 passed, 0 failed**, up from 153. See audit §9z.

**Update 2026-09-14 (MP3 upload transcribing almost nothing — silence-trim
bug)**: user reported uploading an `.mp3` produced no usable transcript.
Reproduced end-to-end with a genuine MP3 run through the real `/upload`
code path (not guessed): the job "completed" but returned a 12-character
transcript out of ~20s of real speech. Root cause: the silence-trimming
step in `pipeline.py` used `ffmpeg`'s `silenceremove` with `stop_periods=1`
on a single forward pass — that setting doesn't trim trailing silence, it
stops the entire filtered output at the FIRST silence gap found anywhere
in the stream and discards everything after it. Confirmed in isolation: a
19.53s clip with one natural pause between sentences came out as 1.43s.
Not mp3-specific — this silently truncates any real recording with a
normal pause between sentences, which unit tests never exercised since
they call the pipeline with transcript text directly, bypassing audio
trimming entirely. Fixed by extracting the step into
`pipeline.trim_leading_trailing_silence()` using the standard
reverse/trim/reverse/trim/reverse technique, which only ever touches
leading/trailing silence. Re-verified with the same real MP3: full
241-character transcript now comes back correctly. New regression test
file `tests/test_audio_trimming.py` (synthetic ffmpeg-generated clips, no
TTS/Whisper needed) guards both the "mid-stream pause must survive" case
and the "leading/trailing silence must still get trimmed" case. Full
suite: **153 passed, 0 failed**, up from 151. See audit §9y.

**Update 2026-09-14 (enterprise UI redesign, static/index.html)**: user
asked for the app's single-page UI to look "Enterprise level," with light
and dark mode, keeping all useful features. Added a full CSS
custom-property theme-token system (light default, dark via both OS
preference and a manual `#themeToggle` persisted to `localStorage`,
applied pre-paint to avoid a flash of the wrong theme). Replaced all emoji
icons with inline SVGs. Fixed a genuinely dead sidebar — 6 nav links
pointed at sections that don't exist in this app — with real anchors into
sections that do, plus scroll-spy active-state tracking. Wired up
drag-and-drop on the upload zone (CSS for it already existed; the JS
listeners never had). Enhanced toasts with 4 auto-inferred types. All
existing functional JS (polling, rendering, the dynamic form, exports, PDF
download) preserved verbatim — verified via JS syntax parsing, a full
DOM-id cross-reference (38 refs, 0 missing), HTML/CSS balance checks, and
headless-Chrome screenshots in both themes plus a populated-data state via
an iframe test harness. No backend or renderer files touched. See audit
§9x.

**Update 2026-09-13, third pass same day (title bug + full section-mapping
audit against a live app-generated PDF)**: user generated a real PDF
through the running app after §9v's fixes and found the document title
itself broken ("Is Named The Cloud Nat Order Processing"), then asked for
a full top-to-bottom section-mapping re-evaluation. Fixed the title bug
precisely (the fallback's stopword-rejection guard correctly skipped a
bare "The" but then accepted "is named the Cloud NAT Order Processing"
wholesale since it has real words in it — added `_trim_name_capture()` to
strip leading verb/filler words off ANY captured name, shared by both the
primary and fallback extraction paths, plus a proper "system name IS X"
direct-statement pattern for the primary path). Also confirmed the
generic-transition-sentence bug flagged unfixed in §9v was still live
("Let's talk about environments." winning Environments' Production row)
and fixed it properly this time, with careful end-anchoring specifically
to avoid a worse regression — an earlier, unanchored version would have
also silently dropped a REAL business-purpose fact that happens to share
an opening phrase with a pure transition sentence in the same
comma-joined utterance; explicitly tested that this doesn't happen. Full
suite: **151 passed, 0 failed**, up from 146. Then ran a complete
section-by-section audit of the live PDF against where each sentence
ideally belongs (full table in audit §9w) — found the remaining gaps
cluster into 4 recognizable bug classes rather than being scattered/
random: (1) real, on-topic content still landing in Unmapped Findings
instead of its correct section (a classification-precision gap, the
deepest item here), (2) LLM-structured-extraction field/content mismatches
confined to 3 sections sharing flat-field prompts (Ownership & Escalation,
Sign-off, Open Responsibilities), (3) a literal "preamble restates the
section's own topic" clutter pattern (Day-1 checklist), (4) one specific
3-word junk fragment appearing in two places (Danger Zones + Tribal
Knowledge). None of these four fixed this pass — each documented with
enough specificity that a future pass doesn't need to re-diagnose from
scratch. See audit §9w for the full before/after table.

**Update 2026-09-13, later same day (adversarial end-to-end pass with a
working LLM)**: user supplied a working `GROQ_API_KEY` (unblocking live LLM
verification after §9u's work was constrained by an exhausted Gemini
quota) and asked for 4 self-authored, deliberately adversarial DevOps KT
transcripts to be run fully end-to-end to find weaknesses. Found and fixed
2 real bugs: (1) a genuine **data-corruption** bug (worse than data loss)
— `devops_transcription.py`'s fuzzy term-correction step tokenized on a
regex that splits contractions on the apostrophe ("it's" → "it" + "s"),
and the resulting bare "s" token fuzzy-matched the glossary term "s three"
(used to fix mis-heard "S3") with a very high score, silently corrupting
ordinary sentences like "It's the old claims processing system" into
"It's **three** old claims processing system" — not a rare edge case,
since "it's the"/"that's the" are among the most common contraction
patterns in English; (2) `system_name` inference was wrong or
meaningless in 3 of 4 transcripts (a single word "The"; the actively wrong
word "Site" grabbed from an unrelated sentence; "KT Document" when a real
name was stated) — two regexes both counted bare "."/"," as valid name
terminators (so *any* "for the X." sentence could be mistaken for a name
introduction) and couldn't handle compound intro phrasing ("this is the
handover for the X platform"). Both fixed at the root, with a shared
stopword-rejection guard so a fallback match is never worse than the
intended "KT Document" default. Full suite: **146 passed, 0 failed**, up
from 137. See audit §9v for the full before/after and the residual,
not-yet-fixed vocabulary-gap finding (GCP/legacy-Java/ML-pipeline content
landing in Unmapped Findings because the schema's hints are AWS/K8s-biased
— the concrete, now-confirmed version of §9u's "fixed section catalog"
design-question answer, but narrower and more tractable than a full
topic-discovery rewrite).

**Update 2026-09-13 (cross-transcript bug hunt: two new real transcripts)**:
user supplied two brand-new transcripts (AWS E-Commerce; a "Cloud NAT Order
Processing Platform" DevOps KT) plus their generated PDFs, asked for a
priority bug pass on section mapping/clutter, and a design answer on making
the template dynamic rather than hardcoded. Found and fixed 6 generic bugs:
(1) a regression from the previous day's evidence-marker work — the
paraphrase-leniency in the LLM gap-fill prompt let a structurally-typed
field (`architecture_link`, type `url`) accept "Confluence" as if it were a
URL, and that one bad value was enough to make the renderer skip its whole
5-bullet fallback, collapsing the section to one line; fixed both at the
prompt (structurally-typed fields now get a strict rule) and the renderer
(never let a thin field hide a richer fallback); (2) Environments
Production/Staging duplication was *still* reproducing despite the prior
sibling-awareness fix — root cause was presence-only checking, not
position; fixed with a "which sibling entity is mentioned first"
positional signal; (3) keyword-hint matching had zero plural tolerance
(`"replacement can deploy safely"` never matched "...replacements can
deploy safely..."), a real, high-value, schema-wide miss found via
Handover Completion scoring 0/6 fields despite explicit transcript
statements; (4) KT-session meta-commentary ("This is a...sample KT using
the Continuum application...", "Today this KT is about DevOps...") was
winning real sections by default — now filtered out before sentences are
even classifiable; (5) `first_30_day_ownership`'s renderer never read real
field data, only a pipe-delimited format natural speech never produces;
(6) found while fixing #5 — `build_knowledge_object()` never actually
recovered a populated field's real schema label/type, silently defaulting
every field's label to its bare id and every type to `"text"` (masked
everywhere else because renderers hardcode their own labels). Full suite:
**137 passed, 0 failed**, up from 120. Live-verified LLM-free (today's
Gemini free-tier quota was already exhausted) against both real
transcripts — see audit §9u for the complete before/after and the
residual, not-yet-fixed issues found along the way.

**Update 2026-09-11, later same day (missing-data pass + embedding-model
reuse)**: user asked what's still missing and whether a "library version"
upgrade could help. Found a real, concrete answer: `field_populator.py`'s
semantic field-matching was loading a separate, materially weaker embedding
model (`all-MiniLM-L6-v2`) instead of reusing the classification stage's own
already-loaded, much stronger `BAAI/bge-large-en-v1.5` — fixed by reusing
the same model instance (zero extra load cost). Also fixed a real gap:
System Overview's Technology Summary table only reflected tools mentioned
in that section's own text, missing tools correctly routed to Monitoring/
Security's own sections — added `enrich_technology_summary()` plus a
fallback to raw sentence text when LLM structured extraction isn't
available (this environment still has no working `GROQ_API_KEY`). Verified
live: Technology Summary went from 3 populated categories to 6. Full
suite: **112 passed, 0 failed**, up from 108. One root cause identified but
**not yet fixed** — see item 6 below. See audit §9t for full detail.

**Update 2026-09-11 (enterprise-review P0 fixes + P1 scope)**: user supplied
a 26-phase "enterprise product review" spec asking for a full rebuild toward
a typed fact-level knowledge model, meaning-first mapping, contradiction
detection, and fully dynamic section generation — reviewed it honestly as a
multi-session architectural rewrite rather than pretending to build it all
in one pass. User chose P1-only scope on top of 3 committed P0 items.
Investigated a reported Environments duplicate-row bug and could not
reproduce it with current code (documented, not silently dropped — see
§9s); fixed 2 real P0 issues (blank table cells now read "Not covered
during KT"; KT Coverage matrix recalibrated so "Strong" actually fires,
was stuck showing "Partial" for everything). Built the approved P1 scope:
an evidence-state marker distinguishing genuinely-inferred field values
from transcript-grounded ones, a Knowledge Gaps list kept structurally
separate from Open Tasks, and Core/Conditional section tiering so
template-boilerplate sections (First 30-Day Plan, Handover Completion
Check) disappear entirely when empty instead of always showing "not
covered" placeholder text. Full suite: **108 passed, 0 failed**, up from
87. Live-verified against a fresh job — see §9s for the complete before/
after.

**Update 2026-09-07 (golden-reference structural parity)**: user supplied a
"golden reference" KT PDF for the same AWS E-Commerce transcript and asked
for the actual output to match it. Found and fixed a real renderer bug
(`system_overview.py` was reading field ids that don't exist in the schema,
so System Overview rendered as one sentence despite the schema already
capturing much richer data) and added 5 new/restructured sections —
Environments, Tribal Knowledge, Operational Calendar (merged with cost
patterns), a Historical Incident sub-block in Common Failures, and a KT
Coverage & Knowledge Gaps matrix plus Quick Reference cheat-sheet — all
generic (schema-agnostic), none hardcoded to this transcript. Full suite:
**87 passed, 0 failed**. Live-verified against a fresh job: all new sections
render with real content, `validation_warnings == []`, valid PDF exported.
Two honest caveats documented in audit §9r: this environment has no
`GROQ_API_KEY` configured, so LLM-structured-extraction-dependent pieces
(Historical Incident block, some Quick Reference rows) weren't re-confirmed
against this exact live transcript (though they are covered by unit tests
and by the pytest suite's own real-Groq golden tests); and `system_overview`'s
attribute table came back thinner than designed on this run because several
overlapping text fields compete for the same 1-2 standout sentences under
the existing cross-field dedup mechanism — content isn't lost, just not
always attributed to its intended field. See §9r for full detail.

**Update 2026-09-06 (fact-fidelity fixes, post-26-phase-brief)**: the 26-phase
brief was complete, but a real transcript/PDF comparison (job 7E30E3D5) found
two more defects in how facts flow through the pipeline: sentences with no
matching schema section vanished silently, and two table fields in one
section could duplicate the same fallback content. Fixed both generically
(no transcript-specific hardcoding) — see audit §9q for the full story,
including a second instance of the duplicate-value bug the new golden-test
invariant caught mid-verification (semantic text-field selection, not just
table fallback) and a live-verification detour into a Gemini daily-quota
stall that required switching the local server to Groq.

**Update 2026-09-05 (post-handoff)**: resolved the LLM-provider items that were
open when this file was first written. See §9f-9h in the audit and the note
under "Must-know" item 1 below — `llama-3.3-70b-versatile` is confirmed
decommissioned on Groq, replaced with `qwen/qwen3.8-27b` (empirically verified
against 5 candidates), a real two-way parameter bug between `GeminiProvider`/
`GroqProvider` was fixed, and rate-limit retry-with-backoff was added to both
providers after the user's live Groq dashboard showed real 429s under burst load.

**Update 2026-09-05 (later same day)**: the retry-with-backoff fix above
turned out to be necessary but not sufficient — user's dashboard kept showing
429s across 3 more bursts. Added a **proactive** sliding-window throttle
(`LLM_MAX_CALLS_PER_MINUTE`, default 20) that actually caps outbound call
volume rather than just reacting after the fact; live-verified zero rate-limit
warnings on a full pipeline run that previously drew some. Also fixed literal
`**bold**` markdown showing up in tables/checklists (a rendering gap, not
LLM-related) and a real PDF whitespace/table-width problem. See audit §9m.

Everything through the §9s update was committed by the user (`05adb17
"updates"`). The §9t work above (model reuse + Technology Summary
enrichment) is not yet committed as of this update — run `git status`
before doing anything else to confirm current state.

---

## How this session was working (context for continuing the same way)

The original ask was a 26-phase "make this an enterprise KT platform" brief.
Approach adopted and validated by the user: **audit first, then work phase by
phase, verify claims instead of trusting old docs or my own prior summaries,
and use plan mode for anything touching core/tested logic.** Several "already
done" claims from earlier passes turned out to be wrong on inspection — worth
continuing that skepticism rather than assuming phase N's audit note is still
accurate.

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Understand product | Done | Implicit in the audit |
| 1 — Full audit | **Done** | `REPOSITORY_AUDIT.md` |
| 2 — Cleanup | **Done** | Dead code, duplicate files, `.gitignore`, PII removed (user's call) |
| 3 — Clean architecture | **Done** | `main.py` split into `api/`, `pipeline.py`, `pdf_rendering.py`, `kt_schema_loader.py`, `llm/prompts.py` |
| 4 — Transcription quality | **Done** | Vocabulary 200→1080 terms, self-learning glossary loop, transcript-test CLI, 3 real bugs found+fixed (fuzzy-matcher data loss, cross-process staleness, 40x perf regression) |
| 5 — Section mapping | **Done** | Entity scoring made real (was decorative), selective LLM verification added, a real dead-path bug found in the process |
| 6 — Knowledge extraction | **Done** | 8/17 sections had zero facts/entities — fixed 3 of them (security_controls, disaster_recovery, ownership_escalation) + monitoring's already-built-but-never-firing extraction. **Found and fixed a session-wide dotenv/import-order bug that silently disabled every Gemini call in the whole app** — see "Must-know" below. |
| 7 — Evidence/traceability | **Done** | Found every fact in every section was silently getting identical generic "first 3 sentences" evidence — `source_chunk_index` was never actually set by either field-producing path. Fixed both (`field_populator.py` and `ai.wrap_structured_as_fields`); verified live that facts now point to distinct, accurate source sentences. See audit §9i. |
| 8 — Knowledge model | **Done** | Verified `build_facts`/`build_entities`/`build_evidence`/`build_relationships`. Found and fixed 2 real defects: unfilled placeholder fields were leaking into the `facts` array with phantom evidence attached (PDF was safe via a render-time filter, but the raw knowledge object/API wasn't); an escalation-chain separator mismatch (` → ` vs `->`) between the pattern-fallback extractor and `relationships.py`'s parser — currently dead code, but same landmine class as prior phases' id-mismatch bugs. See audit §9j. |
| 9 — Dynamic/user templates | **Decided against, not built** | `templates.py` is a complete, working CRUD router — deliberately left unwired because its "RBAC" trusts a spoofable `X-User-Role` header with no real auth. Asked directly and explicitly declined again (2026-09-06) rather than building auth unprompted. See `DELIVERABLE_SUMMARY.md`. |
| 10 — Coverage engine | **Verified, no defect** | `context_mapper.py`'s `semantic_coverage_score()`/`detect_gaps()` — real 4-dimension scoring (confidence/depth/actionability/completeness), not decorative. Fixed one stale docstring that described an old block-count-based rule the code no longer follows. |
| 11 — Document composer | **Verified, no defect** | `pdf_rendering.py`'s `build_rendered_sections()`/`_build_fallback_paragraphs()`/`render_pdf_html()` — renderer dispatch with a real "meaningful blocks vs. fallback" check and deduped fallback paragraphs. Benefits from the Phase 8 `build_facts()` fix (fallback path reads `section["facts"]`). |
| 12 — Section renderers | **Done** | All 8 previously-broken sections fixed (security/DR/ownership in Phase 6; `cost_optimization`/`common_failures`, §9k). Registry `monitoring_observability` id-mismatch, `day1_survival_checklist`/`deployment_and_rollback`/`handover_completion`/`signoff`/`open_responsibilities` all fixed (§9l). The `field_populator.py` field-population cross-contamination flagged here as "needs its own design pass" is now **fixed** — see §9n (two layers: raw-sentence sourcing + a topic-block fallback for a second bug found underneath the first). Remaining known gap: a section-*classification* precision issue found while verifying that fix (a sentence landing in `deployment_and_rollback` instead of `day1_survival_checklist`) — that's Phase 5 territory, not fixed here. |
| 13 — PDF visual design/polish | **Done** | User provided 2 real generated PDFs for review across 2 passes. First pass (§9l): broken `:has(p strong)` CSS heuristic mis-flagging non-warning sections red, doubled headings, ~7 ad hoc placeholder strings consolidated. Second pass (§9m, from a follow-up PDF + Groq dashboard): literal `**markdown**` showing up in tables/checklists (a separate rendering gap — only `NarrativeBlock` ran text through markdown conversion), and page margins consuming ~44% of content width plus a blanket table-column-width rule cramping wide tables. Both live-verified via full visual reads of freshly exported PDFs. |
| 14 — PDF tech (HTML+WeasyPrint) | Already matches brief | No changes needed |
| 15 — UI | **Done** | Turned out to already be a complete, ~1700-line, working frontend wired to every endpoint — not untouched/broken. Fixed a literal-markdown rendering bug matching the one fixed in `pdf_rendering.py` (§9m). No browser-automation tool is available in this environment — verified via server response checks (endpoint correctness, JS content), not interactive click-through. See audit §9p. |
| 16 — LLM provider abstraction | Exists, hardened | `llm_provider.py` — Groq is the active provider (Gemini still supported); `fallback` mode still built but unused/unverified. Rate-limit handling substantially hardened this session (retry-with-backoff, then a proactive sliding-window throttle, §9g/9h/9m) — this abstraction got far more real-world exercise than a typical "exists but unused" rating implies. |
| 17 — Free LLM API strategy | **Resolved** | Groq (`qwen/qwen3.8-27b`) confirmed working in production use this session, including under real burst load — see "Must-know" below. No longer theoretical. |
| 18 — LLM prompting rules (no invention) | Followed throughout, not separately audited | Every prompt in `llm/prompts.py` already has "do not invent" language |
| 19 — Validation layer | **Done** | New `validation.py` — non-fatal structural checks on the knowledge object and populated-fields-vs-schema consistency (the automated version of the by-hand id-mismatch hunting that found several bugs this session). Wired into `pipeline.py`, exposed via `/schema/{job_id}`'s new `validation_warnings` key. **Found and fixed a false-positive in its own field-id check** (§9p) — sections with no `fields` array (or only a dynamic bonus field) were being wrongly flagged on every run. See audit §9o/§9p. |
| 20 — Testing | **Substantially closed** | 65 new tests across 7 new files (validation layer, field_populator regression, LLM throttle/retry, PDF rendering helpers, knowledge builders, quality score, golden KT test) — full suite now 74/74. Still not covered: `pipeline.run_kt_pipeline`/`process_upload_task` with a real LLM provider, `vocabulary_learning.py`, the API endpoints themselves, `static/index.html` (no browser tool available). See `TESTING.md` for the honest map. |
| 21 — Golden KT test | **Done** | New `tests/test_golden_kt.py` — the real `pipeline.run_kt_pipeline()` orchestration end to end (classification → field population → knowledge object → validation → quality score → rendering → PDF export), LLM-stubbed for determinism. First attempt failed on `system_overview` scoring "missing" — a test-transcript-tuning issue (too thin to clear the coverage threshold without LLM tie-breaking), not a product bug; fixed by strengthening the transcript. See audit §9p. |
| 22 — Quality score tracking | **Done** | New `quality_score.py` — required-vs-optional-weighted coverage + confidence + risk + a validation-warning penalty, aggregated into a 0-100 score and letter grade. Wired into `pipeline.py`, exposed via `/schema/{job_id}`. See audit §9p. |
| 23 — Final PDF quality review | **Done** | Same activity as 13 — see above. Both passes were driven by the user sharing an actual rendered PDF, not a synthetic check. |
| 24 — Don't game the system | Followed as a principle throughout | Not a discrete deliverable |
| 25 — Final docs | **Done** | `ARCHITECTURE.md`, `KT_PIPELINE.md`, `PDF_RENDERING.md`, `LLM_PROVIDER.md`, `TESTING.md` created — concise references, not a restatement of the audit's narrative. |
| 26 — Final deliverable summary | **Done** | `DELIVERABLE_SUMMARY.md` — the 5-minute read: what was asked, what was delivered, decisions made, and the 2 explicitly-tracked open issues found but not fixed. |

---

## Must-know before continuing

1. **Gemini free tier on the `.env`-configured key is capped at 5 requests/minute**
   (confirmed live via `429 RESOURCE_EXHAUSTED` responses, not speculation). A
   single KT run needs 15-25+ LLM calls (polish, gap-fill, structured
   extraction, Phase 5's classification verification). Most silently fall back
   to non-LLM behavior once the budget is exhausted a few seconds in.
   **Update**: user is now running with Groq configured instead. Investigated
   whether the Gemini-authored prompts actually work on Groq — prompt text is
   fully portable, but found and fixed a real two-way parameter bug in
   `llm_provider.py` (`GroqProvider` was silently ignoring the `max_output_tokens`/
   `stop_sequences` every call site actually passes; `GeminiProvider` was
   silently dropping `system_prompt`) — see audit §9f. **Important config trap
   also flagged**: setting `GROQ_API_KEY` alone does not route calls to Groq —
   `LLM_PROVIDER=groq` must also be set as an actual env var, or
   `create_llm_provider()` defaults to Gemini regardless. `.env` in this repo
   only has `GEMINI_API_KEY`; if Groq is active it's via shell/OS env vars this
   session couldn't see directly — worth the user double-checking `LLM_PROVIDER`
   is actually `groq` in whatever environment runs the server.
   **Resolved**: user confirmed a real Groq key + `LLM_PROVIDER=groq`; verified
   end-to-end. `llama-3.3-70b-versatile` (the old `GROQ_MODEL` default) is
   confirmed decommissioned on Groq. Live-tested 5 replacement candidates at
   this codebase's tightest real token budget (20 tokens) —
   `openai/gpt-oss-120b`/`-20b` and `qwen/qwen3.6-27b` all fail (reasoning-token
   overhead eats the whole budget, empty/garbled output); `qwen/qwen3.8-27b`
   and `allam-2-7b` both work cleanly. Set `qwen/qwen3.8-27b` (27B, more
   capable) as the new `GROQ_MODEL` default. Full details: audit §9g.
   **Further resolved**: retry-with-backoff alone wasn't enough — added a
   proactive `LLM_MAX_CALLS_PER_MINUTE` throttle (default 20) in
   `llm_provider.py`; live-verified zero 429s on a run that previously drew
   some. Raise the env var if the account's real tier allows more than 20/min.
   Full details: audit §9m.
2. **The dotenv/import-order bug** (`ai.py` imported `llm_provider` before
   calling `load_dotenv()`, so API keys were read as empty on every fresh
   process start) is fixed (`llm_provider.py` now self-loads `.env`), but it
   was silently active for this entire session before being caught in Phase 6.
   Any conclusion drawn earlier in the session about "the LLM polish step
   works" should be treated with suspicion unless it was verified *after* this
   fix (Phase 6's `disaster_recovery` live test is the first verified-good run).
3. **`glossary.json`/`glossary_candidates.json`** are gitignored (Phase 4) —
   local runtime state for the self-learning vocabulary, regenerated from
   `glossary.DEFAULT_GLOSSARY` on first use. Don't be surprised they're absent
   from a fresh clone; that's intended.
4. **Two venvs exist locally** (`.venv`, `.venv-1`) — `.venv-1` is the one with
   actual dependencies installed and is what all this session's verification
   used (`.venv-1/Scripts/python`). `.venv` appears to lack `fastapi` etc.

---

## Working tree — uncommitted (review before committing/pushing)

~43 modified/new/deleted tracked files plus 138 deleted files under
`static/screenshots/` (orphaned artifacts from a disabled feature, Phase 2).
Run `git status` for the live list. Notable new files: `api/`, `pipeline.py`,
`pdf_rendering.py`, `kt_schema_loader.py`, `llm/`, `devops_vocabulary.py`,
`vocabulary_learning.py`, `scripts/review_vocabulary.py`,
`scripts/test_kt_from_transcript.py`, `REPOSITORY_AUDIT.md`.

Full test suite (`pytest tests/`, 7 tests) was green after every phase this
session, most recently after Phase 6's dotenv fix.

## Where to pick up

**19 of 26 phases are done.** Phase 9 is deliberately skipped (asked
directly, declined — needs real auth this codebase doesn't have; see
`DELIVERABLE_SUMMARY.md`). Everything else — 0-8, 10-13, 15-26 excluding
9 — is done as of this update. Read `DELIVERABLE_SUMMARY.md` for the
5-minute version of the whole session; this file and the audit remain the
detailed references.

Two things were found but deliberately **not** fixed — both surfaced through
this session's own live verification, both explicitly tracked rather than
left as silent gaps:

1. **A section-classification precision issue** (`context_mapper.py`, Phase
   5 territory): a clearly Day-1-relevant sentence ("request access to the
   cloud console, the git repository, the CI/CD tool, and the monitoring
   dashboards") was classified into `deployment_and_rollback` instead of
   `day1_survival_checklist` in one live run. Worth a dedicated
   investigation if it recurs — this is the kind of thing that quietly
   degrades document accuracy without ever throwing an error.
2. **`field_populator.py`'s `type: "table"` fallback extraction isn't
   field-aware** — grabs "the first available *unused* lines" regardless of
   which specific field it's filling (still no semantic matching between a
   field's meaning and which lines it gets). The specific cross-contamination
   bug this caused is fixed twice now — §9n fixed collision between a
   table field and a text field reading the same broad content; §9q fixed
   two table fields (or a table + semantic-text field) in one section
   claiming the identical fallback slice — but the underlying "just grab
   whatever's left" heuristic's precision is still coarse. Likely needs
   routing through the structured-JSON-extraction pattern already proven for
   `security_controls`/`disaster_recovery`/etc.
3. **`system_overview`'s text fields over-compete for the same 1-2
   sentences** (found in §9r's live verification): `system_in_5_lines.
   business_impact`/`worst_case`, `impact_if_down.what_breaks`/
   `who_affected`, `customer_reach` all semantically target very similar
   content, and the cross-field dedup mechanism (§9q/§9r) means only the
   first-declared field claims a shared standout sentence — the rest fall
   below the semantic-match threshold and end up unfilled (content still
   visible via the "Additional context" fallback, just not attributed to
   the specific field it was meant for). Not a regression, but worth
   revisiting if System Overview's attribute table keeps coming back thin.
4. **This local environment has no `GROQ_API_KEY` configured** — only
   `GEMINI_API_KEY` is in `.env`, and Gemini's daily free-tier quota (§9g/9h)
   makes it unreliable for ad-hoc verification. Whoever picks this up next
   should set `GROQ_API_KEY` before doing any live LLM-structured-extraction
   verification (common_failures/security_controls/disaster_recovery/
   ownership_escalation/cost_optimization all depend on it, as does LLM
   gap-fill for any pattern/semantic-unfilled field).
5. **The remaining P2/P3 enterprise-review scope from §9s is genuinely
   open**: cross-section semantic deduplication (catching paraphrased
   duplicates like "check Grafana first" vs. "start with Grafana," which
   today's exact-normalized-text dedup misses), contradiction detection
   (no mechanism exists at all), and the larger P3 ask — an independent,
   typed fact-extraction layer (fact_id/type/domain/importance/
   evidence_state/confidence/relationships) computed *before* section
   mapping, plus fully dynamic/emergent section naming beyond the current
   fixed schema + 4 digest sections. All three are real, multi-session
   architectural investments, not something to bolt on incrementally —
   scope them as their own planning pass, don't assume they're small.
6. **`key_technologies`'s pattern-match misses sentences that are visibly
   present in the same section's rendered "Additional context"** (found in
   §9t): "The platform consists of React frontend applications..." shows up
   in System Overview's narrative fallback but never reaches
   `key_technologies`'s own regex pass, so Frontend/Backend/Database/Cache/
   Compute/Edge never appear in the Technology Summary even though the
   sentence is clearly classified to `system_overview`. Root cause
   pattern already documented elsewhere (§9n/§9q) — `field_populator.py`'s
   `section_text` (built from `section_content[id]['sentences']`) and
   `coverage_content` (the "Additional context" fallback's source) are two
   different data views of the same section that can diverge. Not yet
   root-caused for this specific instance or fixed — worth a dedicated
   trace of exactly which sentences land in which of the two views for a
   section like `system_overview` that has many competing fields.
7. **Disaster Recovery / production_notes still occasionally pick up
   generic filler** (found in §9u's live verification, after the §9u meta-
   commentary and sibling-identity fixes): a 4-word fragment ("There would
   be no gaps.") and a generic transition sentence ("Let's talk about
   environments.") respectively — much smaller misses than before those
   fixes, but not zero. The identity-word filter only disqualifies a
   sentence that names a *different* sibling; a sentence naming *no*
   sibling at all still passes to whichever field asks first. Tightening
   that trades recall for precision and deserves its own pass, not a
   rushed addition.
8. **Sign-off and `ownership_escalation`'s `oncall_tool` field
   mis-mapping** (found in §9u): Sign-off's Outgoing/Incoming owner fields
   picked up unrelated sentences (a task-decision-authority remark, a bare
   "Thank you much."), Approved-by picked up a spurious "Yes" from an
   unrelated "say yes/no" sentence elsewhere in the transcript, and
   `oncall_tool` picked up escalation-channel prose instead of a tool name.
   All are `llm/prompts.py` structured-extraction precision issues in
   prompts shared across many sections — not safely fixable without live-
   testing against a working LLM quota (see item 4).
9. **`_infer_system_name()` can produce a document title unrelated to the
   transcript's explicitly stated system name** (found in §9u): one live
   run titled the document "Business Purpose And Criticality" (a spoken
   sub-topic transition) instead of "Cloud Nat Order Processing Platform"
   (the transcript's own explicit "The system name is..." statement).
   Noticed, not root-caused.
10. **A junk fragment can appear as a standalone fact** (found in §9u):
    "The danger zone." rendered as both a stray bullet in Danger Zones and
    a fake Tribal Knowledge row. A safe, generic fix needs a "this fragment
    has no verb, it's just a restated topic label" detector, which needs
    POS tagging — this codebase has no such infrastructure anywhere yet
    (checked; no spaCy/nltk.pos_tag usage exists). Deferred rather than
    built on a fragile keyword-blocklist approximation that would likely
    misfire on legitimate short facts elsewhere.
11. **True dynamic topic discovery is still out of scope, and was asked
    about directly in §9u**: the pipeline is already dynamic in section
    *inclusion* (which of the ~20 known `kt_schema_new.json` sections
    appear depends on transcript coverage) and in tech-stack-triggered
    field addition, but the section *catalog* itself is fixed —
    classification maps each sentence to the best-scoring section among a
    pre-defined list, so a transcript's major topic that isn't already one
    of those ~20 has nowhere accurate to go. Real topic-discovery (segment
    the transcript into topic-coherent chunks independently of the known
    catalog first, then match each chunk to the closest known section or
    mint an ad-hoc one) is the same class of work as the P2/P3 typed-fact-
    model rewrite in item 5 — a dedicated multi-session architecture effort,
    not an incremental add.
12. **Vendor/stack vocabulary gap in Architecture Reference and Deployment
    & Rollback hints, now empirically confirmed** (found in §9v via 3
    non-AWS adversarial transcripts): GCP terms (Pub/Sub, Cloud Run,
    BigQuery, GKE), legacy Java deployment terms (WAR file, Tomcat), and
    ML-pipeline terms (Airflow DAG, data lake, model promotion) all landed
    in Unmapped Findings instead of their obviously-correct real section,
    because those two sections' hints in `kt_schema_new.json` are written
    almost entirely around AWS/Kubernetes/container vocabulary. Narrower
    and more tractable than item 11's full topic-discovery question —
    widening the hint lists to be more vendor-agnostic doesn't need an
    architecture change, just more complete hints — but real content loss
    on any transcript describing a non-AWS/non-container stack. Not
    attempted this pass; a good, contained next piece of work.
13. **`system_name`'s fallback path can't distinguish a real proper name
    from a generic descriptive phrase** (found in §9v): the regex-level
    bugs (bare punctuation as a false terminator, single-"the" skip limit,
    unconditional leftmost match) are fixed, but on a transcript with no
    catchy proper name, the fallback's ceiling is "a coherent if awkward
    description" (e.g. "Is The Legacy Claims Processing"), not a real
    name — that's a semantic-judgment problem (effectively needs NER),
    not something a regex can fully solve. Also worth checking: in one
    live re-run (transcript D, which does explicitly state a real name,
    "CorePay") the fix verified correct in isolation didn't take effect
    end-to-end — the primary field-level extraction path came back empty
    before ever reaching the now-fixed regex in that specific run, so it
    fell to the coverage_content fallback where the LLM's own polishing
    pass had already reworded the sentence past recognition. Worth tracing
    exactly why the field-level path came back empty there if this
    recurs.
14. **Real, on-topic content still lands in Unmapped Findings instead of
    its obviously-correct section** (found in §9w via a live app-generated
    PDF, not a synthetic test): "In terms of business criticality, this
    system is high." and "The reason why the business depends on it is
    that every revenue generating flow passes through this platform."
    are both clearly System Overview content, explicitly stated, yet
    ended up unassigned — which is *why* business_criticality rendered as
    "(inferred)" rather than explicit: the LLM gap-fill never saw the
    explicit sentence because classification routed it elsewhere first.
    Same underlying gap explains Handover Completion Check scoring 0/6
    despite the transcript explicitly stating all 5 confirmations
    ("The replacements can deploy safely...") in one place, and
    Architecture Reference picking up "Artifacts are built and deployed
    in Kubernetes." (which is actually Deployment step 3) while the
    numbered deployment steps it belongs with are scattered across
    Unmapped Findings (Step 1's trigger, orphaned) and nowhere visible
    (Step 2). This is a genuine `context_mapper.py` classification-
    precision gap, not a clutter or evidence-labeling issue — the deepest
    and least-tractable item in this list. Worth a dedicated investigation
    with real transcript examples in hand (this file has several now) if
    picked up.
15. **LLM-structured-extraction field/content mismatches confined to
    sections with flat-field prompts** (found in §9w, same class flagged
    in §9u): Ownership & Escalation's `oncall_tool` picks up escalation-
    *channel* prose ("The escalation channel is the on-call slack
    channel.") instead of a tool name; Sign-off's Outgoing/Incoming owner
    fields pick up unrelated sentences (a task-decision-authority remark,
    a bare "Thank you much.") and Approved-by picks up a spurious "Yes"
    from an unrelated "say yes/no" sentence elsewhere; Open
    Responsibilities' own static "rules" boilerplate text still renders as
    if it were a real task in at least one live case despite §9u's
    structured-extraction fix (which should exclude exactly this) — worth
    a live LLM-available re-check to see whether the fix simply isn't
    firing or the prompt's judgment call is genuinely borderline on this
    phrasing. All three share `llm/prompts.py`'s structured-extraction
    prompts — not safely fixable without live-testing against a working
    LLM quota.
16. **A literal "preamble restates the section's own topic" clutter
    pattern** (found in §9w): Day-1 Survival Checklist's captured value is
    "Day 1 survival checklist. Now the day 1 survival checklist requires
    access including..." — the speaker re-announcing the current topic
    before giving the actual content, with the announcement baked
    verbatim into the extracted field value instead of being stripped. A
    generalizable fix would detect and strip a leading clause that
    fuzzy-matches the section's own title/hints from any extracted field
    value — not attempted this pass.
17. **One specific 3-word junk fragment appearing in two places** (same
    item as the "danger zone." fragment noted in earlier sessions, now
    confirmed still live in §9w): "The danger zone." renders as both a
    stray bullet in Danger Zones and a fake Tribal Knowledge row. Needs a
    "this fragment has no verb, it's just a restated topic label"
    detector — this codebase still has no POS-tagging infrastructure
    (checked again this pass; no spaCy/nltk.pos_tag usage anywhere),
    deliberately not built on a fragile keyword-blocklist approximation
    that would likely misfire on legitimate short facts elsewhere.

If picking Phase 9 back up: decide the auth approach deliberately (a
lightweight shared-API-key-per-role model vs. real user accounts) rather
than defaulting to whichever is faster to build — see
`DELIVERABLE_SUMMARY.md`'s note on this.

Nothing from this entire session is committed yet. Run `git status` before
doing anything else.
