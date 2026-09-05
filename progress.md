# Continuum KT Planner — Progress

Handoff doc for picking this work up in a new session. Written 2026-09-04,
updated 2026-09-05, end of a long working session on
`feature/Dynamic_Schema_Builder`. For the detailed technical narrative behind
every item below — file paths, line numbers, before/after, verification steps —
see **`REPOSITORY_AUDIT.md`**, sections §1-9l. This file is the
quick-orientation index; that file is the record.

**Update 2026-09-05 (post-handoff)**: resolved the LLM-provider items that were
open when this file was first written. See §9f-9h in the audit and the note
under "Must-know" item 1 below — `llama-3.3-70b-versatile` is confirmed
decommissioned on Groq, replaced with `qwen/qwen3.8-27b` (empirically verified
against 5 candidates), a real two-way parameter bug between `GeminiProvider`/
`GroqProvider` was fixed, and rate-limit retry-with-backoff was added to both
providers after the user's live Groq dashboard showed real 429s under burst load.

**Nothing from this session is committed yet.** Everything below is sitting in
the working tree. Run `git status` before doing anything else.

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
| 9 — Dynamic/user templates | **Decided against, not built** | `templates.py` is a complete, working CRUD router — deliberately left unwired because its "RBAC" trusts a spoofable `X-User-Role` header with no real auth. Real feature work if resumed, not a quick wire-up. |
| 10 — Coverage engine | **Verified, no defect** | `context_mapper.py`'s `semantic_coverage_score()`/`detect_gaps()` — real 4-dimension scoring (confidence/depth/actionability/completeness), not decorative. Fixed one stale docstring that described an old block-count-based rule the code no longer follows. |
| 11 — Document composer | **Verified, no defect** | `pdf_rendering.py`'s `build_rendered_sections()`/`_build_fallback_paragraphs()`/`render_pdf_html()` — renderer dispatch with a real "meaningful blocks vs. fallback" check and deduped fallback paragraphs. Benefits from the Phase 8 `build_facts()` fix (fallback path reads `section["facts"]`). |
| 12 — Section renderers | **Done** | All 8 previously-broken sections fixed (security/DR/ownership in Phase 6; `cost_optimization`/`common_failures` in an earlier pass today, §9k). **This pass (§9l)** found the registry was pointing `monitoring_observability` at the wrong (broken) renderer the whole time — a correct one already existed, unregistered — plus fixed `day1_survival_checklist` (field-id mismatches, missing table), `deployment_and_rollback` (triplicated content + a real `NameError` landmine that could silently kill rendering for the whole document), `handover_completion`/`signoff` (never read their real fields, showed canned placeholder text), `open_responsibilities` (wrongly always red/warning-styled). Deleted 3 confirmed-dead orphaned renderer files found along the way. One real gap explicitly deferred: `field_populator.py`'s `type: table` fallback extraction isn't field-aware and can cross-contaminate fields (e.g. `required_access` picking up `first_safe_actions`' text) — needs its own design pass, not a point-fix. |
| 13 — PDF visual design/polish | **Done** | User provided a real generated PDF for review — found and fixed a broken `:has(p strong)` CSS heuristic that mis-flagged non-warning sections red, doubled section/block headings on nearly every page, and ~7 scattered ad hoc placeholder strings (consolidated into one shared, professional "not covered" message). Added section numbering + a single accent color. `plain_english_notes` removed (user's suggestion — low unique value, overlapped other sections). Live-verified via full visual read of a freshly exported 10-page PDF. See audit §9l. |
| 14 — PDF tech (HTML+WeasyPrint) | Already matches brief | No changes needed |
| 15 — UI | Not touched | `static/index.html` untouched all session |
| 16 — LLM provider abstraction | Exists | `llm_provider.py` — Gemini default, Groq coded but inactive, `fallback` mode built but unused |
| 17 — Free LLM API strategy | **Paused, now has real data** | See "Must-know" below — this is no longer a theoretical question |
| 18 — LLM prompting rules (no invention) | Followed throughout, not separately audited | Every prompt in `llm/prompts.py` already has "do not invent" language |
| 19 — Validation layer | Not started | No dedicated schema/knowledge-object/coverage validation layer |
| 20 — Testing | Partial gap | Existing 7-test suite kept green all session; **no new tests added** for this session's new code (`pipeline.run_kt_pipeline`, `vocabulary_learning.py`, new endpoints, Phase 5/6 logic) — real gap |
| 21 — Golden KT test | Not started | |
| 22 — Quality score tracking | Not started | |
| 23 — Final PDF quality review | Not started | Same as 13 |
| 24 — Don't game the system | Followed as a principle throughout | Not a discrete deliverable |
| 25 — Final docs | Partial | Only `REPOSITORY_AUDIT.md` exists; brief's named docs (`ARCHITECTURE.md`, `KT_PIPELINE.md`, `PDF_RENDERING.md`, `LLM_PROVIDER.md`, `TESTING.md`) not created |
| 26 — Final deliverable summary | Not started | This file + the audit partially cover it |

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

Phases 7, 8, 10, 11, 12, 13 are now done (see table above). Natural next
candidates:
- **`field_populator.py`'s `type: table` extraction isn't field-aware**
  (audit §9l, found live-testing the day-1 fix) — its no-pipe/no-numbered
  fallback just grabs "first 10 lines of section text" regardless of which
  field it's filling, so multiple table/text fields in the same section can
  end up with duplicated/wrong content. Same root cause likely affects
  `open_tasks`/`recurring_responsibilities` and contributed to a milder miss
  in `ownership_escalation`'s `oncall_tool`. Real, user-visible (a live PDF
  test showed it), but needs its own design pass (probably routing through
  the structured-JSON-extraction pattern already proven in Phases 6/12) —
  not a quick point-fix.
- **Phase 9's real feature** (user-configurable KT templates) — the user
  asked for this as explicit follow-on work after the mapping fixes, but it
  needs a real auth mechanism first; `templates.py`'s existing RBAC trusts a
  spoofable `X-User-Role` header. Not started.
- Phase 20 (add tests for this session's new code — still no coverage for
  `pipeline.run_kt_pipeline`, the renderer fixes, or the knowledge-model
  fixes) remains an open gap, arguably more overdue with each phase added.
- The Gemini rate-limit decision (item 1 above) is also still open whenever
  the user wants to resume it.
