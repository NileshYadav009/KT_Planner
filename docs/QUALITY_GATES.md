# Quality gates

The thresholds a KT document must meet, measured on the PDF a reader gets.
Regenerate with `python evaluation/quality_gates.py LABEL --replay --accuracy ACC --matrix M`,
then `python evaluation/quality_gates.py --report acc g9abd`.

## Method

Each KT in `tests/e2e_kts` runs through the real pipeline and the real PDF
renderer (`tests/e2e_eval.py`). Its answer key was written from the
transcript: the facts and the sections they belong in, the safety rules,
the connections the KT states and those a careless reading would draw,
the values it says are unknown, the procedures it leaves incomplete, and
claims a model might invent. Every count is on the rendered document:

- **retention**: the fact's words (or its answer-key terms on one page) are in the PDF text;
- **mapping**: the fact is shown in a section the answer key allows;
- **evidence**: of the document's fact units that need a source (inferred values excluded), the share
  that cite a transcript sentence; **misattributed**: a fact shown only where the cited sentences do
  not say it;
- **unsupported**: an answer-key forbidden claim, a value the KT called unknown, or a duration or
  product the transcript never names;
- **pdf**: the PDF opens, and no table, cell or image runs past the page or its cell
  (`pdf_rendering.layout_overflows`, on the laid-out document).

Two modes: **no LLM** (rules and local models), and **LLM replay**: the
Groq answers recorded on a live run, replayed with nothing sent. A KT with
no recorded answer is listed as blocked in that mode, not scored. No live
model was called for this page.

## no-llm

6 KTs.

| Gate | Target | Before (acc) | After (g9abd) | |
|---|---|---|---|---|
| Answer-key facts shown in the PDF text | ≥ 95% | 100.0% | 100.0% | pass |
| Facts shown nowhere in the document | 0 | 0 | 0 | pass |
| Claims the KT does not support (invented commands, tools, durations, values said to be unknown, connections not stated) | 0 | 0 | 0 | pass |
| Facts whose cited sentences do not say them | 0 | 0 | 0 | pass |
| Facts in a section they belong to | ≥ 90% | 100.0% | 100.0% | pass |
| Fact units with a transcript source (inferred values excluded) | ≥ 95% | 100.0% | 100.0% | pass |
| Stated connections drawn | 100% | 100.0% | 100.0% | pass |
| Connections drawn that the KT does not state | 0 | 0 | 0 | pass |
| Answer-key facts marked “no source” | 0 | 0 | 0 | pass |
| Claims that the handover is complete or signed | 0 | 0 | 0 | pass |
| Safety rules outside Danger Zones, or incidents inside it | 0 | 0 | 0 | pass |
| Gaps misreported (discussed shown as missing, undiscussed as covered, incomplete procedures not flagged, coverage disagreeing with the content) | 0 | 0 | 0 | pass |
| PDFs that fail to open, or boxes past the page or their table cell | 0 | 0 | 0 | pass |
| Fact sentences printed twice | tracked | 0 | 0 | tracked |
| Facts the ledger shows in the section they were classified into (of those shown) | tracked | 100.0% | 100.0% | tracked |

| KT | Pages | Facts in PDF | Placement | Relations | Sources | Problems |
|---|---|---|---|---|---|---|
| e2e_a_aws_ecommerce | 11 | 15/15 | 100% | 6/6 | 37/37 | 0 |
| e2e_b_azure_payments | 11 | 17/17 | 100% | 7/7 | 39/39 | 0 |
| e2e_c_gcp_data_platform | 10 | 12/12 | 100% | 3/3 | 33/33 | 0 |
| e2e_d_ambiguous_handover | 8 | 11/11 | 100% | 3/3 | 23/23 | 0 |
| e2e_e_quickbite_food_delivery | 12 | 32/32 | 100% | 7/7 | 56/56 | 0 |
| e2e_f_online_learning | 11 | 20/20 | 100% | 5/5 | 42/42 | 0 |

## llm-replay

5 KTs; blocked (no recorded answers): e2e_f_online_learning.

| Gate | Target | Before (acc) | After (g9abd) | |
|---|---|---|---|---|
| Answer-key facts shown in the PDF text | ≥ 95% | 100.0% | 100.0% | pass |
| Facts shown nowhere in the document | 0 | 0 | 0 | pass |
| Claims the KT does not support (invented commands, tools, durations, values said to be unknown, connections not stated) | 0 | 0 | 0 | pass |
| Facts whose cited sentences do not say them | 0 | 0 | 0 | pass |
| Facts in a section they belong to | ≥ 90% | 100.0% | 100.0% | pass |
| Fact units with a transcript source (inferred values excluded) | ≥ 95% | 99.5% | 99.5% | pass |
| Stated connections drawn | 100% | 100.0% | 100.0% | pass |
| Connections drawn that the KT does not state | 0 | 0 | 0 | pass |
| Answer-key facts marked “no source” | 0 | 0 | 0 | pass |
| Claims that the handover is complete or signed | 0 | 0 | 0 | pass |
| Safety rules outside Danger Zones, or incidents inside it | 0 | 0 | 0 | pass |
| Gaps misreported (discussed shown as missing, undiscussed as covered, incomplete procedures not flagged, coverage disagreeing with the content) | 0 | 0 | 0 | pass |
| PDFs that fail to open, or boxes past the page or their table cell | 0 | 0 | 0 | pass |
| Fact sentences printed twice | tracked | 0 | 0 | tracked |
| Facts the ledger shows in the section they were classified into (of those shown) | tracked | 100.0% | 100.0% | tracked |

| KT | Pages | Facts in PDF | Placement | Relations | Sources | Problems |
|---|---|---|---|---|---|---|
| e2e_a_aws_ecommerce | 11 | 15/15 | 100% | 6/6 | 42/42 | 0 |
| e2e_b_azure_payments | 11 | 17/17 | 100% | 7/7 | 41/41 | 0 |
| e2e_c_gcp_data_platform | 11 | 12/12 | 100% | 3/3 | 36/37 | 0 |
| e2e_d_ambiguous_handover | 8 | 11/11 | 100% | 3/3 | 26/26 | 0 |
| e2e_e_quickbite_food_delivery | 13 | 32/32 | 100% | 7/7 | 61/61 | 0 |

A blocked KT has no answers in `tests/fixtures/e2e_kts_llm_recording.json`; recording them
spends Groq quota and must be agreed first. One recorded from a user's own run can instead be
replayed from the app's LLM cache and scored with `e2e_eval.evaluate` (see the audit).

## Placement on other KTs (no LLM)

From `evaluation/accuracy.py` run `final9abd2`. Holdouts and blind KTs are unseen and are
reported in aggregate only. The target is 90% on unseen KTs. These are without an LLM; with
Groq's section check, as the app runs, see the latest accuracy section of the audit.

| Set | Correct | Facts | Placement | Lost | Unseen: 90% |
|---|---|---|---|---|---|
| tuning | 534 | 534 | 100.0% | 0 | tuned on |
| dev | 373 | 375 | 99.5% | 0 | tuned on |
| holdout | 88 | 95 | 92.6% | 0 | pass |
| blind2 | 109 | 114 | 95.6% | 0 | pass |
| blind3 | 118 | 132 | 89.4% | 4 | FAIL |

## Adversarial matrix

Run `m9abd2`: 80 cases, must checks 284/284, should checks 44/49
(`docs/ADVERSARIAL_MATRIX.md`).
