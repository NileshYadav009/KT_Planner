"""Write the adversarial matrix table (docs/ADVERSARIAL_MATRIX.md) from two runs.

    python evaluation/matrix_report.py BEFORE_LABELS AFTER_LABEL

BEFORE_LABELS is a comma list of earlier runs, oldest first: a case's
"before" is its result in the first of them that ran it (cases are added in
batches). Reads evaluation/results/matrix-LABEL.json (from
evaluation/matrix.py) and evaluation/matrix/cases.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
before_labels, after_label = sys.argv[1].split(","), sys.argv[2]


def load(label):
    with open(os.path.join(HERE, "results", f"matrix-{label}.json"), encoding="utf-8") as fh:
        return {c["id"]: c for c in json.load(fh)["cases"]}


def verdict(case):
    if case is None:
        return "not run"
    must = [r for r in case["checks"] if r["level"] == "must"]
    should = [r for r in case["checks"] if r["level"] == "should"]
    text = f"{sum(r['ok'] for r in must)}/{len(must)} must"
    if should:
        text += f", {sum(r['ok'] for r in should)}/{len(should)} should"
    return text


def expected(exp):
    kind, what = exp["check"], exp.get("text") or exp.get("texts") or exp.get("pattern") or exp.get("section")
    if isinstance(what, list):
        what = " / ".join(what)
    words = {
        "shows": "shows", "shows_any": "shows one of", "verbatim": "exactly", "absent": "never shows",
        "absent_re": "never shows", "never_alone": "never shows alone", "in_section": "in", "order": "in order",
        "flagged": "flags", "not_flagged": "does not flag", "coverage": "coverage of", "completes": "the job completes", "system_name": "title names", "pdf_shows": "PDF shows",
        "pdf_absent": "PDF never shows", "gaps_consistent": "the Not Covered list matches the coverage matrix",
        "sourced": "no fact marked “no source”"}
    detail = f"{words.get(kind, kind)} “{what}”" if what else words.get(kind, kind)
    if kind == "never_alone":
        companions = exp["with"] if isinstance(exp["with"], list) else [exp["with"]]
        detail += " (without " + " or ".join(f"“{w}”" for w in companions) + ")"
    if kind == "in_section":
        detail += " in " + ", ".join(exp["sections"])
    if kind == "coverage":
        detail += " is " + "/".join(exp["is"]) if "is" in exp else " is not " + "/".join(exp.get("not", []))
    return detail + ("" if exp["level"] == "must" else " (should)")


with open(os.path.join(HERE, "matrix", "cases.json"), encoding="utf-8") as fh:
    cases = json.load(fh)["cases"]
after = load(after_label)
before = {}
for label in before_labels:
    for cid, case in load(label).items():
        before.setdefault(cid, (label, case))

lines = [
    "# Adversarial KT matrix",
    "",
    "Short KTs built to break the pipeline, one failure each, run with no LLM",
    "through the real pipeline to the rendered document and, where marked, the",
    "PDF text. The transcripts and checks are in `evaluation/matrix/cases.json`.",
    "Run them with `python evaluation/matrix.py LABEL`, and regenerate this page",
    f"with `python evaluation/matrix_report.py {','.join(before_labels)} {after_label}`.",
    "",
    "- **must**: a wrong document (a lost fact, a withdrawn value shown as",
    "  current, a guess shown as fact, a changed number or name).",
    "- **should**: behaviour the product does not promise yet.",
    "",
    "Before: each case's first run, before any fix for it (run named in",
    f"brackets). After: run `{after_label}`.",
    "",
    "Some first-run failures were the harness's, not the product's, and were",
    "corrected: D02 expected a sentence the document shows as an ownership row",
    "(“Deployment approvals | Developers”); I02 expected “Argo CD” where the",
    "document uses the project's spelling, “ArgoCD” (I02 also had a real",
    "defect: the last step went to another section); the PDF checks of A01",
    "and A05 read the PDF text only up to the cover's list of sections; and in",
    "the first run of T01 and T02 the title check errored (the harness gained it",
    "after that run started), so it counts as failed there.",
    "",
    "| ID | Category | Scenario | Expected | Before | After | Regression test |",
    "|---|---|---|---|---|---|---|",
]
for case in cases:
    label, first = before.get(case["id"], ("", None))
    expects = "; ".join(expected(e) for e in case["expect"])
    lines.append(f"| {case['id']} | {case['category']} | {case['title']} | {expects} | {verdict(first)} ({label}) | "
                 f"{verdict(after.get(case['id']))} | {case.get('regression', 'matrix only')} |")

open_rows = [(c["id"], r) for c in cases for r in (after.get(c["id"]) or {}).get("checks", []) if not r["ok"]]
lines += ["", "## Still failing after the fixes", ""]
if not open_rows:
    lines.append("Nothing.")
for cid, r in open_rows:
    what = r["what"] if not isinstance(r["what"], list) else " / ".join(r["what"])
    lines.append(f"- **{cid}** ({r['level']}) {r['check']} “{what}”: {json.dumps(r['detail'], ensure_ascii=False)[:200]}")

with open(os.path.join(REPO, "docs", "ADVERSARIAL_MATRIX.md"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote docs/ADVERSARIAL_MATRIX.md:", len(cases), "cases,", len(open_rows), "open checks")
