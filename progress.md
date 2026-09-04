# Continuum KT Planner — Progress

Handoff doc for picking this work up in a new session. Written 2026-09-04, end
of a long working session on `feature/Dynamic_Schema_Builder`. For the detailed
technical narrative behind every item below — file paths, line numbers, before/
after, verification steps — see **`REPOSITORY_AUDIT.md`**, sections §1-9e. This
file is the quick-orientation index; that file is the record.

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
| 7 — Evidence/traceability | Not started | `knowledge/evidence.py` exists and is reasonably real (see audit §6 investigation) — not deeply re-verified |
| 8 — Knowledge model | Mostly exists | `knowledge_builder.build_knowledge_object` — untouched beyond Phase 6's fixes |
| 9 — Dynamic/user templates | **Decided against, not built** | `templates.py` is a complete, working CRUD router — deliberately left unwired because its "RBAC" trusts a spoofable `X-User-Role` header with no real auth. Real feature work if resumed, not a quick wire-up. |
| 10 — Coverage engine | Exists, not touched | `context_mapper.py` coverage/confidence/risk scoring |
| 11 — Document composer | Exists, not touched beyond Phase 6 | `knowledge_builder.py` |
| 12 — Section renderers | Partially fixed | 3 of the 8 broken ones fixed in Phase 6 (security/DR/ownership). `cost_optimization` and `common_failures` explicitly scoped out — see audit §9e "Explicitly scoped out" |
| 13 — PDF visual design/polish | Not started | No visual QA pass on an actual rendered PDF yet |
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

1. **Gemini free tier on the current `.env` key is capped at 5 requests/minute**
   (confirmed live via `429 RESOURCE_EXHAUSTED` responses, not speculation). A
   single KT run needs 15-25+ Gemini calls (polish, gap-fill, structured
   extraction, Phase 5's classification verification). Most silently fall back
   to non-LLM behavior once the budget is exhausted a few seconds in. **This
   conversation was paused mid-decision** — options on the table: enable the
   already-built `LLM_PROVIDER=fallback` mode (Gemini→Groq, needs a free Groq
   key from console.groq.com), switch primary providers, or accept the
   degradation. Nothing has been decided or changed yet.
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

Natural next step in brief order is Phase 7/8 (evidence/knowledge model —
likely a quick verify-then-move-on given Phase 6 already touched this area
heavily) or Phase 20 (add tests for this session's new code, arguably overdue
given how much new logic has no test coverage yet). The Gemini rate-limit
decision (item 1 above) is also still open whenever the user wants to resume it.
