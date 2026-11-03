# Continuum feature roadmap (3 Nov 2026)

Continuum turns a KT session (a recording, a Teams or Zoom transcript, or
pasted text) into a handover document that a new owner can operate from.
This roadmap starts from what the code does today, checked feature by
feature, then lists what is worth building next and why. Priorities: P0
critical, P1 high value, P2 useful, P3 later.

## What already exists (verified in the code, not the docs)

| Need | Where it is | State |
|---|---|---|
| Transcript to sections, with or without an LLM | `context_mapper.py` (bge-large, learned section model, routing rules), Groq section check | 93.0% of facts placed right on unseen written KTs without an LLM, 96.2% with Groq |
| Mixed-topic sentences | `fact_units.split_into_facts`, continuation rules | A sentence with two facts is cut; two commands in one sentence stay together on purpose (L2) |
| Duplicates | `knowledge/document_dedup.py` | Repeats removed; the release gate counts printed duplicates (0) |
| Contradictions and changes | `conflicts.py`: values that differ, "only X can" against another subject, schedules with two periods; with an LLM, contradictions of meaning | Flagged on the card and the follow-up list |
| Corrections said aloud | `devops_transcription.resolve_self_corrections`, `replaces_value_before`, `withdraws_said_before` | The withdrawn value is never shown alone |
| Confirmed vs assumed | Hedges and "I think" are marked "To confirm"; inferred values marked inferred; a receiver's unconfirmed claim listed to confirm | `dialogue.is_unconfirmed`, `speaker_roles.py` |
| Missing information | Coverage matrix per section and topic (Strong / Partial / Missing / Not discussed); stated gaps; a follow-up list for the outgoing owner | `coverage_topics.py`, `knowledge_builder.append_coverage_matrix_section` |
| Completeness at a glance | Cover tiles: sections covered well, partly, not covered, items to confirm | `pdf_rendering.handover_glance` |
| Day-1 checklist, 30-day plan, open work | Sections of the fixed 17-section schema | Filled from what was said; empty parts reported, never invented |
| Incident card (quick reference) | Section 01: what to do per known failure, rollback, who to page, never-do rules, DR targets | `knowledge_builder.append_quick_reference_section` |
| Deployment and rollback steps | Deployment section, one step per stated step | `renderers/sections/deployment.py` |
| Escalation | Ownership & Escalation table and the card's who-to-page row | |
| Architecture diagram | Drawn from connections stated in the KT only, with a table quoting each | `architecture_diagram.py` |
| Source references | Every fact cites the transcript sentence it came from (speaker, and time for a recording); click to play the moment | Sources appendix, web page chips |
| Ask the KT | Answers from the document's own statements, with sections and sources; says when a topic was never discussed | `ask.py`, `POST /kt/{id}/ask` |
| Draft, review, signed | Sign-off: name giver, receiver and approver; resolve or accept each gap; all three acknowledge one version; it locks | `signoff.py` |
| Version history and compare | Every edit is a version; diff between versions or against another KT | `document_model.py`, `document_diff.py` |
| Manual editing before export | Approve, edit, delete, move, add lines; saved as a version; the PDF follows | Web page Edit mode |
| Unresolved questions | Knowledge gaps per KT; in sign-off each is closed or accepted as a risk | |
| Follow-up sessions | Add a session to fill gaps; the agenda is the open gaps | `sessions.py` |
| Exports and publishing | PDF, Markdown, runbooks (one per failure), gaps as CSV for Jira; Confluence page and Jira issues | `markdown_export.py`, `runbook.py`, `atlassian.py` |
| Review reminders | "Review due" after 180 quiet days | `staleness.py` |
| Shared components across KTs | Components several systems depend on | `dependencies.py` |
| Screen share | Dashboards and links captured from a shared screen, as screenshots in the document | `screen_capture.py` |
| Conversations | Who is handing over is detected; the receiver's guesses count only when the giver confirms them | `speaker_roles.py` |

## Built in this audit (3 Nov)

From reading a generated PDF as its receiver (`KT_RECEIVER_FEEDBACK.md`):

| Change | Value |
|---|---|
| A yes/no answer stands without its question ("Is there a cache? Yes, ElastiCache Redis holds …" becomes the statement) | Less noise in Q&A-style KTs, the most common way a receiver runs a handover |
| Talk requests ("Can we start with what X does? Sure.") left out | No conversation printed as a fact |
| A corrected assumption is not a gap ("Backups are nightly, I assume? No, Aurora takes continuous backups …") | The follow-up list no longer asks about something answered |
| "Not this year" is a stated gap; "tested a restore" is DR testing | The untested restore is on the follow-up list as what it is |
| An action on the sentence before stays with it ("Not this year. That's something you should schedule.") | The new owner's action is in the document; it was dropped as empty |
| Failures said by what goes wrong are carded with their fix ("The slot worker falls behind when …"; "The customs feed: it sometimes sends …") | The incident card names each failure instead of "Known failure" |
| A receiver's thanks is not "to confirm" | A cleaner follow-up list |

## Next, by priority

| Pri | Feature | User problem | Complexity | Value | Depends on |
|---|---|---|---|---|---|
| P1 | **Readiness questions.** Run the standard Day-1 questions (what it does, how a request flows, how to deploy, roll back, spot a failed release, where to look when it is down, common failures, critical dashboards, never-do, who owns, how to escalate, first week) through Ask the KT and show which the document answers, and from which section | The coverage matrix is per section; a receiver thinks in questions. A KT can be "Strong" on Monitoring and still not say which dashboard to open first | Medium: `ask.py` exists; needs a fixed question set, a threshold per answer, and a block in the Gaps tab and the PDF | High | Measure on the visible KTs first: a wrong "answered" is worse than none |
| P1 | **Recording accuracy.** Recordings place 73% of key facts (written transcripts 93%). Measure a larger Whisper model on the worked-on recordings (X108) | Most real KTs are recorded meetings | Low to run, slow on this laptop | High | Overnight run with the laptop awake |
| P1 | **Mock-console KT recordings** (four briefs are ready) | Real infra KTs over a cloud console, a dashboard and a pipeline are the target, and none of the public recordings is one | Recording time | High | The user records them |
| P2 | **Failed-deployment signal** as a topic of Deployment & Rollback ("how you know a release failed") | Receivers ask it first; the matrix does not, so its absence is not reported | Low: a topic pattern and a follow-up line | Medium | Accuracy sets re-run |
| P2 | **Dashboards named** as a Monitoring topic (which dashboard, where) | "CloudWatch dashboards" covers Monitoring today without saying which to open | Low | Medium | As above |
| P2 | **Glossary of the KT's own terms**: names and acronyms the KT defines ("Patroni is our failover manager") collected in one list with their source | New owners meet internal names first | Medium; only definitions actually said, never written by a model | Medium | Grounding rules |
| P2 | **Regenerate one section** (LLM mode): re-run polish for one section after an edit or a follow-up session | Today the whole document rebuilds | Medium: per-section polish exists; needs a route, a version and guards | Low to medium | LLM budget per KT |
| P2 | **Teams and Zoom transcript fetch** through their APIs | Upload of a .vtt works; fetching needs no download step | Medium | Medium | App registration (T4) |
| P3 | Troubleshooting decision trees | Runbooks are linear; a tree needs branches the speaker rarely states | High, and mostly invented unless said | Low | Not recommended without a speaker who states the branches |
| P3 | Technical clarity score | A number per KT invites gaming; the gaps and to-confirm items already say what is unclear | Medium | Low | |
| P3 | Custom templates (P1 in the ledger) | Teams with their own handover structure | High: rules and prompts name the default sections | Medium | A product decision |

Not recommended: a model-written summary or "AI insights" section. Every
line in the document today is something the speaker said, with its source;
that is what makes it trustworthy to a receiver who cannot ask the person
who left.
