# Continuum: complete audit report (3 Nov 2026)

Engineering, product, QA, KT-receiver and application-security audit of the
Continuum KT application, with the improvements implemented and tested.
Companion documents: `SECURITY_AUDIT.md`, `TEST_CASES.md`,
`KT_RECEIVER_FEEDBACK.md`, `FEATURE_ROADMAP.md`. The evidence for each code
change is in `docs/REPOSITORY_AUDIT.md` §9abj; the issue ledger is in
`CLAUDE.md` (X95, X109-X119).

## 1. Executive summary

Continuum is further along than a transcript-to-PDF tool. Before this audit
it already detected gaps per section and topic, kept corrections and doubts
honest, cited the transcript for every fact, produced an incident card, and
supported editing, version history, sign-off, follow-up sessions, Ask the KT,
and publishing to Confluence and Jira. On written KTs it has never seen, it
puts 93.0% of facts in the right section without an LLM and 96.2% with Groq.
The release gates (no lost facts, nothing invented, every fact sourced, PDFs
that lay out) pass.

The audit still found real defects, and fixed them:

- **A correction that corrupted the document** (X109). Moving a sentence from
  the Transcript tab moved a different sentence, from another section, and
  saved it as a new version.
- **Nine kinds of bad input crashed a request** (X110). Upload errors leaked
  server details, and most requests had no size limit (X111).
- **Vulnerable dependencies on the input path** (X112): 28 of 164 packages
  with advisories. The input parsers were upgraded, which broke the PDF's
  fonts and screenshots (X113, caught by the tests, fixed).
- **Approved vocabulary lost in Docker** and missing from backups (X95).
- **A receiver's reading of a Q&A KT** found six generic problems that make
  the document harder to act on (X114): questions printed before answers, an
  answered question reported as a gap, an action for the new owner dropped,
  and an incident card row that gave a fix without naming the failure.
- **The table editor** on the web page was squeezed into one column (X119),
  found by a new browser test.

None of the fixes moved an accuracy figure on any evaluation set. Opening a
finished KT now sends about a seventh of the data and is prepared about
seven times faster.

What limits the product most is unchanged: **recorded meetings** place 73%
of key facts (written KTs 93%), because speech-to-text wording moves facts
between sections. That, not features, is where the next effort should go.

## 2. Architecture (verified against the code)

| Layer | What it is |
|---|---|
| Frontend | `static/index.html` (about 3,250 lines): one page, vanilla JavaScript, no build step. A start page and one page per KT with tabs (Document, Gaps & coverage, Sign-off, Transcript, Details); `?job=` and `#tab` links; light and dark themes; works at phone width. `static/login.html`: API key or SSO. |
| Backend | FastAPI (`main.py`): request ids, size limits, security headers, compression, a lifespan that loads the models and starts a worker thread. `api/routes.py`: 45 endpoints. |
| Work queue | `job_queue.py`: a tasks table with leases, retries and idempotency keys; `worker.py`: threads in the API process or separate processes on the same database. |
| Database | SQLite (`job_store.py`): jobs (compressed JSON), stored PDFs, document versions, tasks, runs, tenants, API keys, sessions, people, audit log, SSO, integrations, review reminders, vocabulary candidates, a full-text search index; migrations (`migrations.py`). The LLM response cache is a second SQLite file. |
| Recording to text | ffmpeg (after `media_guard` checks), faster-whisper "small" with voice detection, speaker diarization, screen capture of shared dashboards. |
| Cleaning | `devops_transcription.py`: DevOps vocabulary, approved glossary, fillers, self-corrections; a punctuation model for text without punctuation; `redaction.py` removes credentials before anything else. |
| Facts and roles | `speaker_roles.py` (who hands over; the receiver's words count only when confirmed), `context_mapper.segment_sentences`, `fact_units.py`, `dialogue.py` (questions, gaps, doubts, remarks with no content). |
| Classification | `context_mapper.ContextClassifier`: bge-large embeddings against section examples, GLiNER entities, topic segments, a learned section model, routing rules (`section_rules.py`); close calls to Groq's section check. |
| Gaps | `coverage_topics.py` (topics per section), stated gaps, the coverage matrix and follow-up list (`knowledge/knowledge_builder.py`). |
| LLM | Groq (`llm_provider.py`) through `llm/cached_provider.py`: cache, per-KT budgets (60 s a request, 900 s and a call cap per KT), token pacing, a tenant "no external LLM" policy. Seven call sites (copy edit, section check, repair, polish, structured extraction, field fill, conflict check), each checked; every failure falls back to the rules-only result. |
| Document | `knowledge_builder.build_knowledge_object`: sections, fields, conflicts, the incident card, the coverage matrix, a facts ledger proving no fact was lost; `renderers/sections/*` turn it into blocks. |
| PDF | `pdf_rendering.py`: a Jinja2 template rendered by WeasyPrint with a fetcher that loads only inline data and the bundled fonts; cover tiles, contents, an architecture diagram, a Sources appendix; every box measured for overflow. |
| Auth | `auth.py`, `sso.py`: hashed API keys, sessions, OpenID Connect, four roles, tenants. |
| Tests | 75 test files, 1,396 tests; evaluation harnesses: accuracy on six sets plus a sealed one, noisy speech-to-text copies, an adversarial matrix (now 90 cases), release gates on six end-to-end KTs with a recorded Groq replay, real recordings (outside the repo), and now a browser journey. |

**KT generation, end to end:** upload or paste (validated, queued) → worker →
transcribe (recording) → clean and redact → speakers and roles → sentences
and facts (corrections kept with what they correct) → classify → routing
rules → regroup and coverage → polish and structured extraction (LLM) →
fields → knowledge object (conflicts, card, matrix, ledger) → render →
PDF, stored for download.

## 3. Critical bugs identified

| ID | Sev | Bug | State |
|---|---|---|---|
| X109 | High | A section correction moved another section's sentence and saved it as a new version; the sentence stayed listed twice; an unknown sentence answered "applied" | Fixed |
| X113 | High | WeasyPrint 70 refused every font and embedded screenshot (after the dependency upgrade) | Fixed before it reached anyone |
| X112 | High | 28 packages with published advisories, several on the request and upload path | Fixed for the input path; the model stack left by decision |
| X95 | Med | Approved vocabulary written outside the Docker data volume, not backed up, saved in place (a crash lost it all) | Fixed |
| X110 | Med | Nine server errors on wrong-typed input | Fixed |
| X111 | Med | Exception text in upload errors; no size limit on most request bodies | Fixed |
| X119 | Med | Table-row editor 177 px wide, text cut off | Fixed |
| X114 | Med | Six receiver-facing problems in Q&A KTs (section 9) | Fixed |

Two failing tests in the first full suite since §9abg were stale tests, not
bugs, and were updated.

## 4. Performance improvements

| What | Before | After | How measured |
|---|---|---|---|
| Encoding a finished KT's `/status` (largest local KT, 170 KB) | 27.5 ms | 4.0 ms | Same JSON both ways, 30 runs each, read-only copy of the KT |
| `/status` on the wire | 169.6 KB | 26.6 KB | GZip level 5 |
| `/schema` on the wire | 128.5 KB | 21.2 KB | GZip level 5 |
| Signing screenshot links | A deep copy of the whole job on every request | Only when the job holds a screenshot path | |

Not changed, with reasons: pipeline speed was worked on in §9abg (embedding
memo, GLiNER batches, cleaning in linear time); classification and the LLM
are the cost and are budgeted. On this laptop (12 GB) three model-loading
processes at once swap; running one at a time is faster than overlapping.

## 5. Code refactoring completed

Kept small on purpose: working modules were not rewritten.

- `document_model.version_number`: one validation for every version number a client sends.
- `/feedback`: the sentence lookup rewritten around one list in page order.
- `_job_json` in `api/routes.py`: one JSON writer for the two large responses.
- `pdf_rendering`: the fetcher split into the allow-list (`_allowed`), WeasyPrint's fetcher and the document fetcher, so it works across WeasyPrint versions.
- `main.py`: a lifespan handler instead of deprecated startup and shutdown events.
- `glossary.default_path`: one place that says where the vocabulary lives.
- Lint (ruff, real-bug rules only): two repeated dictionary keys with different values fixed. The other findings are unused imports in old code, left alone.

## 6. New features introduced

- **Browser journey** (`evaluation/browser_journey.py`): the whole user journey in Chromium, with an LLM-down mode.
- **Q&A-aware display and gap detection**: yes/no answers stand alone; corrected assumptions are not gaps; actions on the sentence before are kept (section 9).
- **Incident card** rows for failures said by what goes wrong.
- **Vocabulary in backups**, with a checksum in the manifest.
- **Matrix case T04**, a question-and-answer KT read by its receiver.

No feature was added for its own sake: most of the brief's feature list
already exists (see `FEATURE_ROADMAP.md`, "What already exists").

## 7. Features recommended for the future

In order (detail in `FEATURE_ROADMAP.md`):

1. Recording accuracy: measure a larger Whisper model; record the four mock-console KTs.
2. Readiness questions: run the standard Day-1 questions through Ask the KT and show which the document answers.
3. Deployment and Monitoring topics a receiver asks first: how to tell a release failed, and which dashboard to open.
4. A glossary of the KT's own terms, from definitions actually said.
5. Regenerate one section (LLM mode); Teams and Zoom transcript fetch.

Not recommended: decision trees and a clarity score, which would mostly be
invented. Custom templates stay a product decision.

## 8. End-to-end testing results

Full detail in `TEST_CASES.md`. In short:

| Area | Result |
|---|---|
| Full suite, 2 Nov | 1392 passed, 2 failed (stale tests, fixed), 1 skipped |
| Targeted suites after each change | Every run passed (web, auth, sign-off, sessions, uploads, PDF, cards, dialogue, coverage, fields, backups); the full suite below is the check on the final code |
| Browser journey | 19/19 checks passed (sign-in to sign-out, a receiver, three KTs at once, a restart, phone width, no script errors) |
| LLM unreachable | 6/6: the KT completed with warnings, 18 failed and 2 skipped LLM calls all fell back to the rules-only result |
| Input probe (every write route) | 21 server errors before (9 real), 0 after |
| Accuracy, no LLM | Unchanged on every set (table in section 14) |
| Adversarial matrix | 90 cases, 324/324 must, 56/61 should |
| Release gates | Every gate passes, without an LLM and on the recorded Groq answers |
| Sealed set b08-b11 | 151/160, as before (read once, at the end) |
| Full suite after the changes | 1416 passed, 0 failed, 1 skipped (the e2e replay with no recorded answer, X106), 1 xpassed (the holdout target test, now met), in 1 h 35 min |

## 9. KT receiver evaluation

A visible dev KT (Quayside, a port booking system on AWS ECS, told as
questions from the new owner and answers from the outgoing one) was built
without an LLM and read as the receiver would, with no transcript. Detail
and the twelve Day-1 questions in `KT_RECEIVER_FEEDBACK.md`.

It let a new owner say what the system does and what it is built on, roll
back (redeploy the previous task definition), fix the two common failures,
know what never to do (purge the SQS queue) and who to call. It did not let
them tell a failed release, find a named dashboard, see a request's path, or
plan beyond the first day and month. Those were not said in the session, and
the document says so on its follow-up list, which is the honest outcome.

What made it harder to use, now fixed: the receiver's questions printed
before answers; "Can we start with ...? Sure." as a fact; an answered
question ("Backups are nightly, I assume? No, ...") reported as a gap; the
untested restore reported "not discussed"; the new owner's action ("That's
something you should schedule.") dropped; the card's fix shown without its
failure; Tribal Knowledge showing a cut-off question; a "thanks" to confirm.

## 10. PDF quality evaluation

| Question | Finding |
|---|---|
| Easy to navigate? | Yes: contents with page numbers, the incident card first, cover tiles saying where to go during an incident and before sign-off |
| Headings consistent? | Yes: numbered sections from one schema |
| Tables readable? | Yes; the release gates measure every box for overflow (0) |
| Excessive whitespace or length? | 13 pages for a 600-word KT (14 before the fixes): the body is 8 pages, the coverage report and the Sources appendix the last 3 to 4 |
| Important information buried? | Less now: the card names each failure; DR's real gap is on the follow-up list |
| Generic AI text? | None in rules-only mode; with Groq, polish is refused when it adds specifics or flips a negation |
| Assumptions as facts? | No: hedges are "To confirm", a receiver's unconfirmed words are listed, not printed as facts |

Recommendations, not yet done: move the Sources appendix and the full
coverage matrix to an appendix after Sign-off (they are reference, not
reading); drop "Architecture Metadata" rows that are all "Not discussed" into
the follow-up list. See `KT_RECEIVER_FEEDBACK.md`.

## 11. Application security findings

See `SECURITY_AUDIT.md`. No cross-tenant read, no injection path, and no
way for transcript text to make the PDF renderer or the LLM do anything
outside its checks was found. Eight findings were fixed; 16 packages keep
advisories (the model stack by decision, tooling, test-only pypdf).

## 12. Security improvements implemented

The `/feedback` integrity fix; input validation on version numbers and ids;
no exception text in responses; a 1 MB limit on JSON bodies and an early
limit on session recordings; a bound on Ask questions; 12 packages upgraded;
the WeasyPrint 70 fetcher; a lifespan handler.

## 13. Remaining known issues

| Issue | Note |
|---|---|
| Recording accuracy 73% (X108, X101, X103 on written text) | The main limit; next steps in section 15 |
| X117: "Is there a queue? Yes, RabbitMQ ..." keeps its question | The answer never repeats the question's word |
| X118: a topic listed "not discussed" next to a stated gap about it | Wording only |
| X18, L2, X67 | Measured before, every fix tried cost more than it gained |
| X106: the Groq replay misses 29 requests | Re-record needs about 49k Groq tokens and your go-ahead |
| Model-stack advisories (torch, transformers, sentence-transformers, nltk) | Upgrade with a full accuracy re-run |
| python-jose (SSO) has an advisory with no fix | Not exploitable here (algorithm allow-list); move to a maintained library |
| No sign-in throttling, chunked bodies not limited early | For the proxy when hosting moves |
| Docker not started in this audit | Your rule; the X95 fix is unit-tested, not tested in a container |

## 14. Before and after

| Measure | Before (2 Nov) | After (3 Nov) |
|---|---|---|
| Written KTs never seen, no LLM | 466/501 (93.0%) | 466/501 (93.0%); the sealed set, read once at the end, 151/160 as before |
| Visible KTs, no LLM | 1085/1085 | 1085/1085 |
| Noisy speech-to-text copies | 1057/1085 | 1057/1085 |
| e2e KTs | 100%, no problems | 100%, no problems |
| Adversarial matrix | 89 cases, 310/310 must, 54/58 should | 90 cases, 324/324 must, 56/61 should |
| Server errors from bad input | 9 causes | 0 |
| Packages with advisories | 28 (249) | 16 (148) |
| `/status` encode, wire size | 27.5 ms, 170 KB | 4.0 ms, 27 KB |
| Browser journey | none | 19/19, and 6/6 with the LLM down |

## 15. Recommended next priorities

1. **Recordings.** Measure Whisper large-v3-turbo on the worked-on recordings overnight, with the laptop kept awake on its charger; record the four mock-console KTs; then decide.
2. **Readiness questions** (P1 in the roadmap), measured on the visible KTs before it is shown.
3. **The two receiver topics** (failed-release signal, named dashboards), measured on all six sets.
4. **Model-stack upgrade** with a full accuracy re-run, then the remaining tooling packages.
5. **The PDF appendix move** (Sources and the full coverage matrix after Sign-off).
6. Commit this batch (`git status` lists the files; nothing is committed).

## 16. Files modified

38 files changed, 6 added, nothing committed; every file keeps its own line
endings.

| Area | Files |
|---|---|
| Request layer | `main.py` (body limit, GZip, lifespan), `api/routes.py` (`/feedback`, upload errors, Ask limit, `_job_json`) |
| Documents and sign-off | `document_model.py` (version numbers), `signoff.py` (acknowledge) |
| Vocabulary and backups | `glossary.py`, `vocabulary_learning.py`, `scripts/backup.py`, `docs/OPERATIONS.md` |
| PDF | `pdf_rendering.py` (WeasyPrint 70 fetcher) |
| KT content | `dialogue.py`, `context_mapper.py`, `coverage_topics.py`, `speaker_roles.py`, `knowledge/knowledge_builder.py`, `field_populator.py`, `renderers/sections/system_overview.py` |
| Web page | `static/index.html` (the move sends its text, errors shown, the table-row editor) |
| Dependencies | `requirements.txt`, `requirements.lock`, `.gitignore` (`.venv-tools/`) |
| Tests | `tests/test_ask.py`, `test_audit_regressions.py`, `test_backup.py`, `test_coverage_topics.py`, `test_data_handling.py`, `test_dialogue.py`, `test_ecommerce_kt.py`, `test_incident_card.py`, `test_review_signoff.py`, `test_speaker_roles.py`, `test_sso.py`, `test_upload_hardening.py`; `evaluation/matrix/cases.json` (T04) |
| New | `evaluation/browser_journey.py`; the five reports at the root |
| Records | `CLAUDE.md` (ledger, testing), `docs/REPOSITORY_AUDIT.md` §9abj, `docs/progress.md`, `docs/ADVERSARIAL_MATRIX.md` and `docs/QUALITY_GATES.md` (regenerated) |

Earlier the same day the user committed the Whisper vocabulary follow-up
and the two stale-test fixes from the 2 Nov suite (cead29f).
