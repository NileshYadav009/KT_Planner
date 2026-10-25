# Accuracy evaluation

Measures how the KT pipeline places facts on KTs with a written answer key.
It runs with no LLM, the copy edit off, and a throwaway database, so it costs
no quota and touches no user data. Run it from the repository root with the
project environment (`.venv-1`).

```
python evaluation/accuracy.py LABEL SETS
```

`SETS` is a comma list. A run of all five takes about 4 minutes once
`evaluation/.cache/` is warm.

| Set | What | Detail printed |
|---|---|---|
| `tuning` | `tests/goldens` without the holdouts | per fact |
| `dev` | `evaluation/dev`: 6 KTs in styles the goldens lack (formal, meeting, noisy speech-to-text, read-aloud notes, Q&A, rambling) | per fact |
| `holdout` | `tests/goldens` h12-h14 | **aggregate only** |
| `blind2` | `evaluation/blind`: b01-b03 | **aggregate only** |
| `blind3` | `evaluation/blind3`: b04-b07, written 25 Oct before any result on them (`blind3_kts.py` regenerates them): question and answer, noisy speech, a formal document, a rambling KT with a correction | **aggregate only** |
| `e2e` | `tests/e2e_kts` (six KTs: AWS, Azure, GCP, an ambiguous handover, and the user's QuickBite and online learning KTs) through `tests/e2e_eval.py` | problems only |

The results go to `evaluation/results/LABEL.json`, which is git-ignored.

Each set reports:

- `correct`, `wrong` and `lost` facts;
- `false_missing`: a discussed section reported Missing;
- `false_covered`: an undiscussed section reported Covered or Partial;
- `repeats` and `unsourced` lines.

## Rules

- **Blind and holdout KTs** are never inspected or tuned on one fact at a time.
  Their aggregate is the only honest estimate of accuracy on unseen KTs. Once
  a fix has been tuned on them, they stop being blind.
- **A fix must be generic.** Nothing may name a KT, a system or a transcript.
  Measure every fix on all five sets.
- **No live LLM runs here.** An LLM-mode run needs a key, spends quota and
  must be agreed first.

## Tools

| Script | Use |
|---|---|
| `quality_gates.py LABEL [--replay] [--accuracy ACC] [--matrix M]` | The release gates on the e2e KTs, counted on the PDF: fact retention, silent loss, unsupported claims, misattributed sources, placement, sources, stated connections, false “no source” marks, false sign-off, safety rules, honest gaps, PDF layout (`pdf_rendering.layout_overflows`) and repeats. No LLM, and with `--replay` the recorded Groq answers (nothing sent; a KT with none is listed as blocked). `--accuracy` and `--matrix` add an earlier run's figures. Results: `results/gates-LABEL.json`; `--report BEFORE AFTER` writes `docs/QUALITY_GATES.md`. About 25 minutes for both modes. |
| `kodiag.py KT_ID SECTION[,...]` | One dev or tuning KT: what a section contains and why the coverage table rates it so. Refuses holdout and blind KTs. |
| `adversarial.py` | A probe KT with distinctions the goldens under-test, listed below. |
| `matrix.py LABEL [IDS] [--dump]` | The adversarial matrix: short KTs in `matrix/cases.json`, one per failure (corrections, ambiguity, hedges, contradictions, several facts in a sentence, duplicates, negation, numbers, order, coverage, injection, noise, PDF text), each with exact checks on the rendered document. `must` checks are a wrong document; `should` checks are behaviour not promised yet. About 2 to 6 minutes for all cases. Results: `results/matrix-LABEL.json`. |
| `trace.py MODULE.VAR [FILTER]` | Mapper decision path per fact for a probe transcript, e.g. `adversarial_text.TRANSCRIPT`. |
| `dev_kts.py OUTDIR` / `blind_kts.py OUTDIR` | Regenerate the dev and blind JSON and check every quote is in its transcript. |
| `groq_diag.py KT_ID` | Replays a dev KT's LLM run from the response cache, sending nothing, and shows its repeated, unsourced, wrong and lost facts. Use the settings of the run being diagnosed. |
| `groq_eval.py LABEL dev,holdout,blind2` | The same sets with the LLM. It spends Groq quota and must be agreed first. `GROQ_BUDGET` (default 150) caps live requests. Unchanged prompts are answered from `evaluation/.cache/llm_cache.sqlite`. Scoring is strict and lenient (`lenient.py`), and holdout and blind results are aggregate only. `GROQ_EVAL_ONLY=classify` sends only the section check (polish, extraction and field fill run without the LLM): an A/B of that check costs a fraction of a full run, and scoring is exact. |
| `conflict_eval.py LABEL matrix,dev,e2e` | The LLM conflict check alone, after a no-LLM pipeline run: the matrix's contradiction cases and controls (a flag on a control, or a wrong pair, is a false flag), and every flag on the dev and e2e KTs. Spends quota; `GROQ_BUDGET` (default 40) caps it. About 400 tokens a KT. |

The adversarial probe covers:

- architecture vs deployment;
- monitoring vs remediation;
- ownership vs escalation;
- current state vs plans;
- a correction vs a contradiction.

`enccache.py` memoises embedding, cross-encoder and entity-model outputs on
disk (`evaluation/.cache/enc_cache.sqlite`). Runs are then minutes, not tens
of minutes, on a slow machine. It is for experiments only, never the app. It
stores pickles, so never use a cache file from anyone else.
