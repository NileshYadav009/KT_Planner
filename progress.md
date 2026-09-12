# Continuum KT Planner — Progress

Handoff doc for picking this work up in a new session. Written 2026-09-04,
updated 2026-09-11, end of a long working session on
`feature/Dynamic_Schema_Builder`. For the detailed technical narrative behind
every item below — file paths, line numbers, before/after, verification steps —
see **`REPOSITORY_AUDIT.md`**, sections §1-9t. **For a 5-minute overview
instead of either of these two detailed files, read `DELIVERABLE_SUMMARY.md`**
(not yet updated with the §9q/§9r/§9s/§9t work below — still describes the
state through §9p). This file remains the quick-orientation index; the audit
is the full record.

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

If picking Phase 9 back up: decide the auth approach deliberately (a
lightweight shared-API-key-per-role model vs. real user accounts) rather
than defaulting to whichever is faster to build — see
`DELIVERABLE_SUMMARY.md`'s note on this.

Nothing from this entire session is committed yet. Run `git status` before
doing anything else.
