"""Quality gates: the thresholds a KT document must meet, measured on the PDF.

    python evaluation/quality_gates.py LABEL [--replay] [--accuracy ACC_LABEL] [--matrix MATRIX_LABEL]
    python evaluation/quality_gates.py --report BEFORE AFTER

Every KT in tests/e2e_kts goes through the real pipeline and the real PDF
renderer (tests/e2e_eval.py) with no LLM; with --replay also through the
LLM path, answered only from the recorded Groq responses
(tests/fixtures/e2e_kts_llm_recording.json, nothing sent). Each KT has an
answer key written from its transcript: facts and where they belong, safety
rules, stated connections and connections a careless reading would draw,
values said to be unknown, procedures left incomplete, and claims a model
might invent. The gates are counted on the document a reader gets.

--accuracy adds placement on the unseen KTs from an evaluation/accuracy.py
run (holdouts and blind KTs, aggregate only); --matrix adds the adversarial
matrix's must checks from an evaluation/matrix.py run. Results go to
evaluation/results/gates-LABEL.json (git-ignored); --report writes
docs/QUALITY_GATES.md from two runs.
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Throwaway state, as in the tests: the job store, vocabulary candidates and
# LLM cache are real files, and no key from .env may reach a provider.
_state = tempfile.mkdtemp(prefix="gates-")
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(_state, "continuum.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(_state, "glossary_candidates.json"),
    "KT_ASSETS_DIR": os.path.join(_state, "kt_assets"),
    "CONTINUUM_AUTH": "disabled",
    "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE_PATH": os.path.join(_state, "llm_cache.sqlite"),
})
for _key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[_key] = ""
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(encoding="utf-8")
if os.environ.get("NO_ENC_CACHE") != "1":
    # Memoised model outputs (evaluation/enccache.py), as accuracy.py uses:
    # the same numbers, minutes faster. Experiments only, never the app.
    sys.path.insert(0, HERE)
    import enccache  # noqa: E402

    enccache.install(os.environ.get("ENC_CACHE_PATH") or os.path.join(HERE, ".cache", "enc_cache.sqlite"))
RESULTS = os.path.join(HERE, "results")
RECORDING = os.path.join(REPO, "tests", "fixtures", "e2e_kts_llm_recording.json")

# (id, what is measured, target, kind): "min" is a share that must reach the
# target, "zero" a count that must be 0, "track" a count reported only.
GATES = [
    ("retention", "Answer-key facts shown in the PDF text", 0.95, "min"),
    ("silent_loss", "Facts shown nowhere in the document", 0, "zero"),
    ("unsupported", "Claims the KT does not support (invented commands, tools, durations, values said to be "
                    "unknown, connections not stated)", 0, "zero"),
    ("misattributed", "Facts whose cited sentences do not say them", 0, "zero"),
    ("mapping", "Facts in a section they belong to", 0.90, "min"),
    ("evidence", "Fact units with a transcript source (inferred values excluded)", 0.95, "min"),
    ("relations", "Stated connections drawn", 1.0, "min"),
    ("wrong_relations", "Connections drawn that the KT does not state", 0, "zero"),
    ("false_no_source", "Answer-key facts marked “no source”", 0, "zero"),
    ("false_signoff", "Claims that the handover is complete or signed", 0, "zero"),
    ("safety", "Safety rules outside Danger Zones, or incidents inside it", 0, "zero"),
    ("gaps", "Gaps misreported (discussed shown as missing, undiscussed as covered, incomplete procedures not "
             "flagged, coverage disagreeing with the content)", 0, "zero"),
    ("pdf", "PDFs that fail to open, or boxes past the page or their table cell", 0, "zero"),
    ("duplicates", "Fact sentences printed twice", None, "track"),
    ("own_section", "Facts the ledger shows in the section they were classified into (of those shown)", None,
     "track"),
]


def _measure(reports):
    """Gate values over a set of e2e reports (tests/e2e_eval.evaluate)."""
    def count(side, *keys):
        return sum(len(r[side][k]) for r in reports for k in keys)

    facts = sum(r["facts"] for r in reports) or 1
    stated = sum(r["relations"]["stated"] for r in reports)
    units = sum((r.get("evidence") or {}).get("units", 0) - (r.get("evidence") or {}).get("inferred", 0)
                for r in reports)
    sourced = sum((r.get("evidence") or {}).get("stated", 0) for r in reports)
    correct = sum(sum(1 for p in r["placement"].values() if p["status"] == "correct") for r in reports)
    return {
        "retention": round(sum(r["retained_in_pdf"] for r in reports) / facts, 4),
        "silent_loss": count("false_negatives", "lost_facts"),
        "unsupported": count("false_positives", "undocumented_given_a_value", "forbidden_claims",
                             "durations_not_in_transcript", "products_not_in_transcript"),
        "misattributed": count("false_positives", "facts_without_source"),
        "mapping": round(correct / facts, 4),
        "evidence": round(sourced / units, 4) if units else None,
        "relations": round(sum(r["relations"]["found"] for r in reports) / stated, 4) if stated else None,
        "wrong_relations": count("false_positives", "wrong_relations"),
        "false_no_source": count("false_positives", "known_facts_marked_no_source"),
        "false_signoff": count("false_positives", "false_signoff"),
        "safety": count("false_negatives", "safety_warnings_misplaced") + count("false_positives",
                                                                               "incidents_in_danger_zones"),
        "gaps": count("false_negatives", "false_missing_sections", "incomplete_procedures_not_flagged")
        + count("false_positives", "false_covered_sections", "coverage_disagrees_with_content"),
        "pdf": sum(1 for r in reports if not r["pdf"]["opens"]) + count("false_positives", "pdf_boxes_past_their_edge"),
        "duplicates": count("false_positives", "repeated_sentences"),
        "own_section": _share(reports, "shown_in_own_section", "mapped"),
    }


def _share(reports, part, whole):
    """A ledger ratio over the KTs whose ledger records it (older runs do not)."""
    rows = [r.get("knowledge_coverage") or {} for r in reports]
    rows = [k for k in rows if part in k]
    total = sum(k.get(whole, 0) for k in rows)
    return round(sum(k[part] for k in rows) / total, 4) if total else None


def _verdicts(values):
    out = {}
    for gid, _, target, kind in GATES:
        value = values.get(gid)
        if kind == "track" or value is None:
            out[gid] = "tracked" if kind == "track" else "not measured"
        elif kind == "min":
            out[gid] = "pass" if value >= target else "FAIL"
        else:
            out[gid] = "pass" if value == 0 else "FAIL"
    return out


def _trim(report):
    """What a later reader needs from one KT's report."""
    keep = ("id", "mode", "status", "system_name", "pdf", "facts", "retained_in_pdf", "placement_accuracy",
            "relations", "evidence", "knowledge_coverage", "llm")
    out = {k: report.get(k) for k in keep}
    out["problems"] = {side: {k: v for k, v in report[side].items() if v}
                       for side in ("false_negatives", "false_positives")}
    out["misplaced"] = {k: v for k, v in report["placement"].items() if v["status"] != "correct"}
    return out


def _mode(reports, recorded_only=False):
    if recorded_only:
        blocked = [r["id"] for r in reports if not (r["llm"] or {}).get("cache_hits")]
        reports = [r for r in reports if r["id"] not in blocked]
    else:
        blocked = []
    values = _measure(reports) if reports else {}
    return {"kts": [_trim(r) for r in reports], "blocked": blocked, "values": values, "verdicts": _verdicts(values)}


def run(label, replay=False, accuracy=None, matrix=None, out_dir=None):
    import e2e_eval
    import pipeline

    started = time.time()
    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    result = {"label": label, "modes": {}}
    result["modes"]["no-llm"] = _mode(e2e_eval.run_all(out_dir=os.path.join(out_dir, "no-llm") if out_dir else None))
    if replay:
        reports = e2e_eval.run_replay(RECORDING, out_dir=os.path.join(out_dir, "replay") if out_dir else None)
        # A KT with no recorded answer would be scored as if it ran without
        # an LLM: it is reported as blocked, not counted.
        result["modes"]["llm-replay"] = _mode(reports, recorded_only=True)
    if accuracy:
        with open(os.path.join(RESULTS, f"{accuracy}.json"), encoding="utf-8") as fh:
            acc = json.load(fh)
        result["accuracy"] = {"label": accuracy, **{k: acc[k] for k in ("tuning", "dev", "holdout", "blind2", "blind3")
                                                    if k in acc}}
    if matrix:
        with open(os.path.join(RESULTS, f"matrix-{matrix}.json"), encoding="utf-8") as fh:
            result["matrix"] = {"label": matrix, **json.load(fh)["summary"]}
    result["seconds"] = round(time.time() - started)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"gates-{label}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def _fmt(gid, value):
    if value is None:
        return "–"
    kind = next(g[3] for g in GATES if g[0] == gid)
    return f"{value:.1%}" if kind == "min" or isinstance(value, float) else str(value)


def table(result):
    lines = []
    for mode, data in result["modes"].items():
        lines.append(f"{mode}: {len(data['kts'])} KTs" + (f"; blocked: {', '.join(data['blocked'])}" if data["blocked"] else ""))
        for gid, what, target, kind in GATES:
            lines.append(f"  {gid:16} {_fmt(gid, data['values'].get(gid)):>8}  {data['verdicts'][gid]:12} {what}")
    return "\n".join(lines)


def report(before_label, after_label):
    def load(label):
        with open(os.path.join(RESULTS, f"gates-{label}.json"), encoding="utf-8") as fh:
            return json.load(fh)

    before, after = load(before_label), load(after_label)
    lines = [
        "# Quality gates",
        "",
        "The thresholds a KT document must meet, measured on the PDF a reader gets.",
        "Regenerate with `python evaluation/quality_gates.py LABEL --replay --accuracy ACC --matrix M`,",
        f"then `python evaluation/quality_gates.py --report {before_label} {after_label}`.",
        "",
        "## Method",
        "",
        "Each KT in `tests/e2e_kts` runs through the real pipeline and the real PDF",
        "renderer (`tests/e2e_eval.py`). Its answer key was written from the",
        "transcript: the facts and the sections they belong in, the safety rules,",
        "the connections the KT states and those a careless reading would draw,",
        "the values it says are unknown, the procedures it leaves incomplete, and",
        "claims a model might invent. Every count is on the rendered document:",
        "",
        "- **retention**: the fact's words (or its answer-key terms on one page) are in the PDF text;",
        "- **mapping**: the fact is shown in a section the answer key allows;",
        "- **evidence**: of the document's fact units that need a source (inferred values excluded), the share",
        "  that cite a transcript sentence; **misattributed**: a fact shown only where the cited sentences do",
        "  not say it;",
        "- **unsupported**: an answer-key forbidden claim, a value the KT called unknown, or a duration or",
        "  product the transcript never names;",
        "- **pdf**: the PDF opens, and no table, cell or image runs past the page or its cell",
        "  (`pdf_rendering.layout_overflows`, on the laid-out document).",
        "",
        "Two modes: **no LLM** (rules and local models), and **LLM replay**: the",
        "Groq answers recorded on a live run, replayed with nothing sent. A KT with",
        "no recorded answer is listed as blocked in that mode, not scored. No live",
        "model was called for this page.",
        "",
    ]
    for mode in after["modes"]:
        a, b = after["modes"][mode], before["modes"].get(mode)
        lines += [f"## {mode}", "",
                  f"{len(a['kts'])} KTs" + (f"; blocked (no recorded answers): {', '.join(a['blocked'])}" if a["blocked"] else "")
                  + ".", "",
                  f"| Gate | Target | Before ({before_label}) | After ({after_label}) | |", "|---|---|---|---|---|"]
        for gid, what, target, kind in GATES:
            goal = "tracked" if kind == "track" else ("100%" if kind == "min" and target >= 1 else
                                                       f"≥ {target:.0%}" if kind == "min" else "0")
            was = _fmt(gid, b["values"].get(gid)) if b else "not run"
            lines.append(f"| {what} | {goal} | {was} | {_fmt(gid, a['values'].get(gid))} | {a['verdicts'][gid]} |")
        lines += ["", "| KT | Pages | Facts in PDF | Placement | Relations | Sources | Problems |", "|---|---|---|---|---|---|---|"]
        for kt in a["kts"]:
            ev = kt.get("evidence") or {}
            problems = sum(len(v) for side in kt["problems"].values() for v in side.values())
            lines.append(f"| {kt['id']} | {kt['pdf']['pages']} | {kt['retained_in_pdf']}/{kt['facts']} | "
                         f"{kt['placement_accuracy']:.0%} | {kt['relations']['found']}/{kt['relations']['stated']} | "
                         f"{ev.get('stated', 0)}/{ev.get('units', 0) - ev.get('inferred', 0)} | {problems} |")
        if a["blocked"]:
            lines += ["", "A blocked KT has no answers in `tests/fixtures/e2e_kts_llm_recording.json`; recording them",
                      "spends Groq quota and must be agreed first. One recorded from a user's own run can instead be",
                      "replayed from the app's LLM cache and scored with `e2e_eval.evaluate` (see the audit)."]
        open_items = [(kt["id"], side, k, v) for kt in a["kts"] for side, checks in kt["problems"].items()
                      for k, v in checks.items()]
        if open_items:
            lines += ["", "Open problems after:", ""]
            lines += [f"- {kt}: {k.replace('_', ' ')}: {', '.join(map(str, v))}" for kt, _, k, v in open_items]
        lines.append("")
    if after.get("accuracy"):
        acc = after["accuracy"]
        lines += ["## Placement on other KTs (no LLM)", "",
                  f"From `evaluation/accuracy.py` run `{acc['label']}`. Holdouts and blind KTs are unseen and are",
                  "reported in aggregate only. The target is 90% on unseen KTs. These are without an LLM; with",
                  "Groq's section check, as the app runs, see the latest accuracy section of the audit.", "",
                  "| Set | Correct | Facts | Placement | Lost | Unseen: 90% |", "|---|---|---|---|---|---|"]
        for name in ("tuning", "dev", "holdout", "blind2", "blind3"):
            if name in acc:
                s = acc[name]
                verdict = ("pass" if s["accuracy"] >= 0.9 else "FAIL") if name in ("holdout", "blind2", "blind3")                     else "tuned on"
                lines.append(f"| {name} | {s['correct']} | {s['facts']} | {s['accuracy']:.1%} | {s['lost']} | {verdict} |")
        lines.append("")
    if after.get("matrix"):
        m = after["matrix"]
        lines += ["## Adversarial matrix", "",
                  f"Run `{m['label']}`: {m['cases']} cases, must checks {m['must']}, should checks {m['should']}",
                  "(`docs/ADVERSARIAL_MATRIX.md`).", ""]
    with open(os.path.join(REPO, "docs", "QUALITY_GATES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote docs/QUALITY_GATES.md")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--report":
        report(args[1], args[2])
        sys.exit(0)

    def opt(name):
        return args[args.index(name) + 1] if name in args else None

    res = run(args[0], replay="--replay" in args, accuracy=opt("--accuracy"), matrix=opt("--matrix"),
              out_dir=opt("--out"))
    print(table(res))
