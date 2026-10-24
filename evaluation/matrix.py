"""Adversarial KT matrix runner (no LLM, copy edit off, scratch state).

    python evaluation/matrix.py LABEL [IDS] [--dump]

IDS is an optional comma list of case ids (default: all cases in
evaluation/matrix/cases.json). --dump prints every document unit of each case.
Results go to evaluation/results/matrix-LABEL.json (git-ignored).

Each case's transcript runs through the real pipeline, and the checks read the
rendered document (knowledge_object["rendered_sections"], what the PDF prints)
and, for cases marked "pdf", the text of the PDF itself without its Sources
appendix. See the "about" line in cases.json for what each check means.
"""
import json
import os
import re
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
label = sys.argv[1]
wanted = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
dump = "--dump" in sys.argv

state = tempfile.mkdtemp(prefix="matrix-")
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "c.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "g.json"),
    "KT_ASSETS_DIR": os.path.join(state, "a"),
    "CONTINUUM_AUTH": "disabled", "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE_PATH": os.path.join(state, "llm.sqlite"),
})
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[key] = ""
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
import enccache  # noqa: E402
enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

import ai  # noqa: E402
import golden_eval  # noqa: E402
import llm_provider  # noqa: E402
import pipeline  # noqa: E402
from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402

pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
pipeline.record_candidates = lambda *a, **k: None
pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)

norm = golden_eval.norm
_FLAG_TITLE = re.compile(r"conflict|confirm|gap|clarif|verify|question", re.IGNORECASE)
_SKIP_KEYS = {"type", "id", "section_id", "anchor", "sources", "level", "style", "variant"}


def _join(value):
    if isinstance(value, dict):
        return " ".join(_join(v) for k, v in value.items() if k not in _SKIP_KEYS)
    if isinstance(value, (list, tuple)):
        return " ".join(_join(v) for v in value)
    return "" if value is None or isinstance(value, bool) else str(value)


def document_units(ko):
    """(section_id, block type, block title, text) for every piece a reader
    sees: each paragraph, item, step, table row and field."""
    units = []
    for sec in ko.get("rendered_sections") or []:
        sid = sec.get("section_id")
        for block in sec.get("blocks") or []:
            btype, title = block.get("type"), str(block.get("title") or "")
            if title:
                units.append((sid, btype, title, title))
            for key, value in block.items():
                if key in _SKIP_KEYS or key == "title":
                    continue
                items = value if isinstance(value, list) else [value]
                for item in items:
                    text = _join(item).strip()
                    if text:
                        units.append((sid, btype, title, text))
    return units


def is_flag(unit):
    return unit[1] == "WarningBlock" or bool(_FLAG_TITLE.search(unit[2] or ""))


def has_word(needle, text):
    n = norm(needle)
    return bool(n) and re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", norm(text)) is not None


def pdf_text(job_id, result):
    import io

    from pdf_rendering import build_job_pdf
    from pypdf import PdfReader

    data = build_job_pdf(job_id, result)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    # The Sources appendix quotes the transcript as said: cut it off at its
    # intro (a short document also lists "Sources" on its cover).
    cut = re.search(r"Each\s+fact\s+in\s+this\s+document\s+is\s+followed\s+by", text)
    return text[:cut.start()] if cut else text


def check(exp, ko, units, status, pdf):
    kind = exp["check"]
    plain = [u for u in units if not is_flag(u)]
    if kind == "completes":
        return status == "completed", status
    if kind == "system_name":
        got = str(ko.get("system_name") or "")
        return norm(exp["text"]) in norm(got), got
    if kind == "shows":
        hits = [u for u in units if norm(exp["text"]) in norm(u[3])]
        return bool(hits), sorted({u[0] for u in hits})
    if kind == "shows_any":
        hits = [u for u in units for t in exp["texts"] if norm(t) in norm(u[3])]
        return bool(hits), sorted({u[0] for u in hits})
    if kind == "verbatim":
        hits = [u for u in units if exp["text"] in u[3]]
        near = [u[3][:160] for u in units if norm(exp["text"]) in norm(u[3])][:2]
        return bool(hits), sorted({u[0] for u in hits}) or near
    if kind == "absent":
        hits = [u for u in plain if has_word(exp["text"], u[3])]
        return not hits, [f"{u[0]}: {u[3][:160]}" for u in hits]
    if kind == "absent_re":
        hits = [u for u in plain if re.search(exp["pattern"], u[3].lower())]
        return not hits, [f"{u[0]}: {u[3][:160]}" for u in hits]
    if kind == "never_alone":
        companions = exp["with"] if isinstance(exp["with"], list) else [exp["with"]]
        hits = [u for u in plain if has_word(exp["text"], u[3]) and not any(norm(w) in norm(u[3]) for w in companions)]
        return not hits, [f"{u[0]}: {u[3][:160]}" for u in hits]
    if kind == "in_section":
        texts = golden_eval.section_texts(ko)
        found = sorted(s for s, items in texts.items() if any(norm(exp["text"]) in norm(t) for t in items))
        return bool(set(found) & set(exp["sections"])), found or "LOST"
    if kind == "order":
        by_section = {}
        for u in units:
            by_section.setdefault(u[0], []).append(norm(u[3]))
        for sid, texts in by_section.items():
            joined = " | ".join(texts)
            pos = [joined.find(norm(t)) for t in exp["texts"]]
            if all(p >= 0 for p in pos) and pos == sorted(pos):
                return True, sid
        where = {t[:30]: sorted(s for s, ts in by_section.items() if any(norm(t) in x for x in ts)) for t in exp["texts"]}
        return False, where
    if kind == "flagged":
        hits = [u for u in units if is_flag(u) and all(norm(t) in norm(u[3]) for t in exp["texts"])]
        return bool(hits), [u[2] for u in hits]
    if kind == "not_flagged":
        hits = [u for u in units if u[1] == "WarningBlock" and all(norm(t) in norm(u[3]) for t in exp["texts"])]
        return not hits, [f"{u[0]}: {u[3][:160]}" for u in hits]
    if kind == "coverage":
        got = golden_eval.coverage_buckets(ko).get(exp["section"])
        ok = (got in exp["is"]) if "is" in exp else (got not in exp.get("not", []))
        return ok, got
    if kind in ("pdf_shows", "pdf_absent"):
        found = has_word(exp["text"], pdf)
        return found == (kind == "pdf_shows"), "in PDF" if found else "not in PDF"
    if kind == "gaps_consistent":
        # The PDF's "Not Covered in This KT" list names exactly the sections
        # the coverage matrix (and the cover's "not covered" tile) call Missing.
        # A section discussed only to say it is not in place keeps its own
        # content and is not in the list ("Nothing in the KT session covered").
        listed = set(not_covered_list(ko))
        rows = next((s.get("_coverage_rows") or [] for s in ko.get("sections") or [] if s.get("id") == "kt_coverage"), [])
        missing = {r.get("Domain") for r in rows
                   if r.get("Coverage") == "Missing" and str(r.get("Assessment") or "").startswith("Not covered")}
        return listed == missing, {"listed only": sorted(listed - missing), "Missing only": sorted(missing - listed)}
    if kind == "sourced":
        # No fact printed with a "no source" mark.
        marked = [f"{sec.get('section_id')}: {block.get('title')}" for sec in ko.get("rendered_sections") or []
                  for block in sec.get("blocks") or [] if block.get("unsourced")]
        return not marked, marked
    raise ValueError(kind)


def not_covered_list(ko):
    """The section titles the PDF lists under "Not Covered in This KT"."""
    import pdf_rendering
    rendered = list(ko.get("rendered_sections") or [])
    if hasattr(pdf_rendering, "not_covered_titles"):
        return pdf_rendering.not_covered_titles(rendered, ko)
    out = pdf_rendering.collapse_not_covered_sections(rendered)
    sec = next((s for s in out if s.get("section_id") == pdf_rendering.NOT_COVERED_SECTION_ID), None)
    return [str(i) for b in (sec or {}).get("blocks") or [] for i in b.get("items") or []]


with open(os.path.join(HERE, "matrix", "cases.json"), encoding="utf-8") as fh:
    cases = json.load(fh)["cases"]
if wanted:
    cases = [c for c in cases if c["id"] in wanted]

t0 = time.time()
report = []
for case in cases:
    job_id = f"matrix-{case['id']}"
    transcript = "\n".join(case["transcript"])
    started = time.time()
    try:
        result = pipeline.run_kt_pipeline(job_id, clean_transcript(transcript))
    except Exception as exc:  # the pipeline must never raise; record it if it does
        result = {"status": f"raised {type(exc).__name__}: {exc}", "knowledge_object": {}}
    ko = result.get("knowledge_object") or {}
    units = document_units(ko)
    expects = list(case["expect"])
    pdf = ""
    if case.get("pdf") and result.get("status") == "completed":
        pdf = pdf_text(job_id, result)
        for exp in case["expect"]:
            if exp["check"] in ("shows", "absent"):
                expects.append({"check": "pdf_" + exp["check"], "text": exp["text"], "level": exp["level"]})
    rows = []
    for exp in expects:
        try:
            ok, detail = check(exp, ko, units, result.get("status"), pdf)
        except Exception as exc:
            ok, detail = False, f"check error {exc}"
        rows.append({"check": exp["check"], "level": exp["level"], "ok": ok, "detail": detail,
                     "what": exp.get("text") or exp.get("texts") or exp.get("pattern") or exp.get("section")})
        mark = "pass" if ok else ("FAIL" if exp["level"] == "must" else "miss")
        what = rows[-1]["what"]
        print(f"{case['id']:4} {mark:4} {exp['level']:6} {exp['check']:11} {str(what)[:58]:58} {'' if ok else detail}")
    if dump:
        for u in units:
            print(f"    [{u[0]} | {u[1]} | {u[2][:30]}] {u[3][:220]}")
    report.append({"id": case["id"], "category": case["category"], "title": case["title"],
                   "status": result.get("status"), "seconds": round(time.time() - started, 1), "checks": rows,
                   "units": [list(u) for u in units] if not all(r["ok"] for r in rows) else []})
    pipeline.JOB_STORE.delete(job_id)

must = [r for c in report for r in c["checks"] if r["level"] == "must"]
should = [r for c in report for r in c["checks"] if r["level"] == "should"]
summary = {"cases": len(report),
           "cases_all_must_pass": sum(all(r["ok"] for r in c["checks"] if r["level"] == "must") for c in report),
           "must": f"{sum(r['ok'] for r in must)}/{len(must)}", "should": f"{sum(r['ok'] for r in should)}/{len(should)}",
           "seconds": round(time.time() - t0)}
print(json.dumps(summary))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", f"matrix-{label}.json"), "w", encoding="utf-8") as fh:
    json.dump({"label": label, "summary": summary, "cases": report}, fh, indent=1)
