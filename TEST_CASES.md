# Continuum test cases and results (3 Nov 2026)

Every result below was produced by running the test, on this laptop, on the
date shown. Nothing is reported as passed that was not run. Where a test was
not run, it says so and why.

How to re-run each part is in `CLAUDE.md` ("Testing").

## 1. Browser journey (real Chromium, the real app)

`evaluation/browser_journey.py`, run from the tools environment
(`.venv-tools`: Playwright and Chromium). It starts the app on a throwaway
database with sign-in required and no LLM, signs in with a scratch API key,
and drives the page as a user would. Run j5 (3 Nov, about 20 minutes),
on the code before the last two content changes of the day (the failure
topic of X114 and X121's language detection), which change document text,
not any flow it checks.

| Case | Expected | Actual | Result |
|---|---|---|---|
| J01 Sign in with an API key | Signed-out visitors get the sign-in page; a wrong key is refused with a message; the right key opens the app | wrong key refused ('That API key was not accepted.'); right key opens the app | Pass |
| J02 Submit an empty transcript | A message asks for a transcript; no KT is created | message shown, 0 KT created | Pass |
| J03 Submit small talk | Refused with the reason (not enough operational content); no KT | 'Not a usable KT transcript: The transcript has only 11 words; a KT document needs at least 30.'; 0 KT created | Pass |
| J04 Paste a 284-word KT and double-click Build | One KT (not two) is built and opens when done | completed; 1 KT for a double click | Pass |
| J05 Read the document | The document names QuickBite, has every section, the incident card first | 16 sections, first '01 Quick Reference', names QuickBite: True | Pass |
| J06 Gaps & coverage tab | Shows coverage per section and a follow-up list naming what was not discussed (cost, environments, security in this KT) | follow-up items 16; undiscussed named {'Cost': True, 'Environments': True, 'Security': True} | Pass |
| J07 Ask the KT | A covered question gets statements with sources; a topic never discussed is said to be missing, not answered | 5 answer(s) to a covered question; unknown topic: 'The KT does not say. Ask the outgoing owner, or add it in a follow-up session.' | Pass |
| J08 Edit a line and save | The edit is saved as version 2; version 1 is kept | 2 versions after one save | Pass |
| J09 Move a sentence from the Transcript tab | The sentence picked (not one from another section) moves, and is listed once | sentence 6 ('Requests enter through Cloudflare and an AWS Appli…') to system_overview: there True, left behind 0 | Pass |
| J10 Download the PDF | A PDF that opens, names the system and carries the saved edit | 12 pages, names QuickBite, carries the edit: True | Pass |
| J11 Export Markdown, gaps CSV and runbooks | Each downloads with content | 12,914, 4,742 and 1,072 bytes | Pass |
| J12 Refresh, home, browser back | The KT link survives a refresh and the back button | after reload the KT shows; the back button returns to it | Pass |
| J13 Restart the server | Sign-in, the KT, its versions and the PDF survive a restart | after a restart: signed in, PDF 200 | Pass |
| J14 A receiver opens the KT | Can read and download; cannot edit or create (no button, API 403) | Edit button shown: False; edit API 403; create API 403 | Pass |
| J15 Paste a transcript over the limit | Refused (413) with the limit and what to do | 413: "The transcript has 200,000 characters; the limit is 150,000 (about 25,000 words). Split it into sessions …" | Pass |
| J16 Three KTs at once | All are queued and all finish | 3 sent at once, all completed, no errors, 70 s | Pass |
| J17 Phone width (390 px) | No sideways scrolling on the start page or a KT | scroll width 390 = client width 390 on both | Pass |
| J18 Sign out | Back to the sign-in page; the back button does not show the KT | app not shown after back; API without a key 401 | Pass |
| J19 No script errors or blocked scripts | No console error, uncaught exception or CSP violation during the journey | CSP violations 0, console errors 0, page errors 0 (1 network error while J13 had the server stopped, expected) | Pass |
| | | **19/19 passed** | |

The same journey with an LLM that refuses every connection (a closed local
port as Groq's address, so nothing left the machine), run j6-llmdown. It
runs the checks that matter for the fallback: a KT is built and read, and
its PDF downloads.

| Case | Expected | Actual | Result |
|---|---|---|---|
| J01 Sign in (LLM down) | As J01 | as J01 | Pass |
| J04 Build a KT (LLM down) | The KT is built | completed_with_warnings; 1 KT for a double click | Pass |
| J05 Read the document (LLM down) | As J05 | 16 sections, the card first, names QuickBite | Pass |
| J10 Download the PDF (LLM down) | A PDF that opens and names the system | 12 pages, names QuickBite | Pass |
| J19 No script errors (LLM down) | As J19 | 0, 0, 0 | Pass |
| J20 LLM unreachable | The KT still finishes with the rules-only result, and the LLM failures are counted | completed_with_warnings; 18 calls failed, 2 skipped, every one fell back | Pass |
| | | **6/6 passed** | |

**Defects the journey found, and their retests.**

| Run | Finding | Severity | Cause | Fix | Retest |
|---|---|---|---|---|---|
| j1 | Server never ready | (harness) | Three model-loading processes swapped the laptop; the script left its server running | The script always stops its server | j2 onwards |
| j2, j3 | J08 failed | (harness) | The first Edit button is a table row, which opens inputs, not a textarea | The test opens a text editor | j5 pass |
| j3 debug | **Table-row editor 177 px wide, text cut off** | Med (X119) | The editor was a `<div>` inserted after a `<tr>`, laid out in the first column | A row of its own across the table, a text box per cell | Seen in a screenshot, 540 px; j5 J08 and J10 pass |
| j4 | J08 failed with an EvalError | (harness) | The test waited with a JavaScript string; the page's CSP forbids eval, correctly | A locator wait | j5 pass |
| j4 | One console error | None | During J13 the server is stopped on purpose and a request in flight fails | Counted apart as expected | j5 pass |
| j5-llmdown | J04 failed | (harness) | That run skips J02, which opens the paste form | J04 opens it if needed | j6-llmdown pass |

## 2. Robustness probe (every write route)

A script sent wrong-typed JSON (null, numbers, strings, lists, objects,
booleans, huge numbers, a 400-digit string) to every write route of a
finished KT, and odd version numbers to every read route.

| | Server errors (500) |
|---|---|
| Before | 21: 9 real (four causes, X110) and 12 from the probe's own sample data (its coverage records lacked a field real KTs always have, checked on the local database) |
| After | 0 |

## 3. Defects found and the tests added for them

| ID | Sev | Test case | Expected | Before | After |
|---|---|---|---|---|---|
| X109 | High | `test_review_signoff.py::test_a_move_picks_the_sentence_the_transcript_tab_numbered` | Sentence 2 of the tab (Danger Zones' first) moves; listed once; a stale page gets 409; an unknown sentence 404; a KT being built 409 | Another section's sentence moved; listed twice; "applied" for nothing | Pass |
| X110 | Med | `test_review_signoff.py::test_wrong_typed_version_numbers_are_refused_not_server_errors`, `::test_an_acknowledgement_of_a_wrong_typed_version_is_refused_not_a_server_error` | 400 or 404 | 500 | Pass |
| X111 | Med | `test_upload_hardening.py::test_an_unexpected_upload_failure_does_not_echo_server_details`, `::test_other_request_bodies_have_a_size_limit`, the session-recording limit in `::test_oversized_uploads_are_refused`; `test_ask.py` (long question) | No exception text; 413; 400 | Exception text; read whole | Pass |
| X95 | Med | `test_data_handling.py::test_approved_vocabulary_is_kept_beside_the_database`, `::test_a_failed_vocabulary_save_leaves_the_old_file_whole`; `test_backup.py::test_the_approved_vocabulary_is_backed_up_and_restored` | Beside the database, old file adopted, atomic save, in backups | In the working directory, not backed up | Pass |
| X113 | High | `test_pdf_security.py::test_the_pdf_is_set_in_the_bundled_typeface` (existing) | Inter font, data: URIs load | Failed after the WeasyPrint 70 upgrade (Segoe UI) | Pass |
| X114 | Med | `test_dialogue.py::test_an_answer_after_yes_or_no_stands_without_its_question` (9 cases), `::test_a_corrected_assumption_is_not_a_stated_gap`, `::test_an_action_on_the_sentence_before_stays_with_it`; `test_incident_card.py::test_failures_said_by_what_goes_wrong_are_carded_with_their_fixes`; `test_speaker_roles.py::test_a_receivers_thanks_is_not_listed_to_confirm`; `test_coverage_topics.py` (DR testing wordings, `::test_a_failure_said_by_what_goes_wrong_is_a_known_failure`); matrix case T04 | As in `KT_RECEIVER_FEEDBACK.md` | Each failed | Pass (T04: 14 of 14 must, 2 of 3 should) |
| X116 | Low | `test_ecommerce_kt.py` with its duplicate key removed | Monitoring shows "PagerDuty alert" | The expectation was never checked | Pass |
| X121 | Low | `test_audit_regressions.py::test_platform_and_external_services_are_recognised` | "a worker in Python" is Python; "experience in Java" is not a tool | Missing | Pass |
| — | — | Two stale tests from the 2 Nov suite (`test_renderer_sections`, `test_observability`) | Match the deliberate X62 change; read UTF-8 output as UTF-8 | Failed | Pass |

## 4. Automated suites

| Run | When | Result |
|---|---|---|
| Full suite (`pytest`, 75 files) | 2 Nov, 3 h 38 min | 1392 passed, 2 failed (the stale tests above), 1 skipped, 1 xpassed |
| Targeted suites after each change | 3 Nov | Every targeted run passed. The last runs: web, auth, sign-off, sessions, job store, observability, headers, page structure (107 passed); PDF and screenshots (50); card, dialogue, coverage topics, speaker names (132); coverage topics (14); field population (52) |
| Full suite on the final code | 3-4 Nov night, 1 h 35 min | 1416 passed, 0 failed, 1 skipped (`test_e2e_kts` LLM replay for `e2e_f_online_learning`: no recorded answer matches the current prompts, X106), 1 xpassed (`test_golden_suite` holdout target, non-strict xfail) |

## 5. Evaluation suites (no LLM)

| Suite | 2 Nov (before) | 3 Nov (after) |
|---|---|---|
| Tuning and dev KTs (visible) | 1085/1085 facts in an allowed section | 1085/1085 |
| Holdouts h12-h14 | 88/95 | 88/95 |
| Blind b01-b03 | 109/114 | 109/114 |
| Blind b04-b07 | 118/132, 3 discussed sections shown Missing | 118/132, 3 |
| Sealed b08-b11 (read once at the end of a batch) | 151/160 (2 Nov) | 151/160, no discussed section shown Missing |
| e2e KTs (six, PDF) | 100%, no problems | 100%, no problems |
| Noisy speech-to-text copies | 1057/1085, 1 Missing | 1057/1085, 1 Missing |
| Adversarial matrix | 89 cases, 310/310 must, 54/58 should | 90 cases (T04 new), 324/324 must, 56/61 should |
| Release gates (six e2e KTs: no LLM, and the recorded Groq answers) | pass | Every gate passes in both modes (retention 100%, nothing lost, nothing unsupported, PDF layout 0 overflows); `docs/QUALITY_GATES.md` |

The five "should" misses: D04 (needs the LLM by design), B09 (X18), T01 (a
fragment answer keeps its question), L06 (a code walkthrough's Ownership is
Partial), T04's RabbitMQ question (X117).

## 6. Performance

| Measure | Before | After |
|---|---|---|
| Encoding the largest local KT's `/status` | 27.5 ms | 4.0 ms (identical JSON) |
| `/status` on the wire | 169.6 KB | 26.6 KB |
| `/schema` on the wire | 128.5 KB | 21.2 KB |

## 7. Security tooling

| Tool | Result |
|---|---|
| pip-audit, the app's environment | Before: 28 of 164 packages, 249 advisories. After the input-path upgrade: 16 packages, 148 advisories (`SECURITY_AUDIT.md`) |
| ruff, real-bug rules (pyflakes, bugbear) | No undefined name; two repeated dictionary keys with different values, fixed; the rest unused imports in old code |

## 8. Not run, and why

| What | Why |
|---|---|
| Docker image | You asked that Docker Desktop not be started without asking. The X95 fix is covered by unit tests, not a container run. |
| LLM-mode evaluation against live Groq | It spends quota; the gates' recorded Groq replay covers the LLM path offline (with X106's 29 missing requests). |
| Real recordings after today's changes | 9 hours on this laptop; the last run (2-3 Nov, before these changes) is 163/222 key facts. The changes touch Q&A wording and the card, which recordings rarely use; to re-run overnight when wanted. |
| Frontend build | There is none: the page is plain HTML and JavaScript, checked by the browser journey and `test_frontend_accessibility.py`. |
| Type checking | No type checker is set up in the project; adding one was not asked for. |
