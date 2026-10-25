"""End-to-end KT suite: four short synthetic handovers (tests/e2e_kts) run
through the real pipeline and the real PDF renderer, checked on the PDF a
reader gets.

Each dataset is a transcript plus a checklist written from the transcript
before any output was looked at:

    facts         golden_eval format (quote, in, alt, terms, not_in) plus a kind
    safety        fact ids that are safety warnings (must reach Danger Zones)
    covered       sections the KT discussed (must not be reported Missing)
    missing       sections it did not discuss (must be reported Missing)
    relations     connections the KT states (from, to, kind: flow/dependency/deploy)
    wrong_relations  connections a careless reading would draw but the KT does not state
    undocumented  values the KT says are unknown: a regex that must not match
    forbidden     unsupported claims a model or rule might invent: regexes
    asr           a misheard term and the term it must become
    incomplete    procedures the KT leaves incomplete (read by the reviewer)

Every check is reported on its own; nothing is folded into a single score.
False negatives (knowledge lost) and false positives (knowledge invented or
misplaced) are listed separately. Runs without an LLM unless the caller
provides one; the result records which.
"""
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_eval import norm, score as golden_score  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_kts")
FLOORS_PATH = os.path.join(DATA_DIR, "floors.json")

# Names the document may use for what the transcript calls by a short name.
_ALIASES = {
    "eks": ["amazon eks", "elastic kubernetes service"], "rds": ["amazon rds"], "ecr": ["amazon ecr"],
    "aks": ["azure kubernetes service"], "acr": ["azure container registry"], "service bus": ["azure service bus"],
    "gke": ["google kubernetes engine"], "redis": ["azure cache for redis", "memorystore"],
    "application load balancer": ["alb"], "azure sql": ["azure sql database"],
}
_FRAMING = [
    r"\b(?:hi|hello|hey)\s+(?:everyone|team|all|folks)\b", r"\bthanks?\s+(?:you\s+)?for\s+joining\b",
    r"\bhanding\s+over\b", r"\bthis\s+is\s+(?:a\s+(?:quick\s+)?|the\s+)handover\b", r"\bthat'?s\s+most\s+of\s+it\b",
    r"\bthis\s+document\s+serves\s+as\b", r"\bum\b", r"\buh\b",
]
_SIGNOFF = re.compile(
    r"\b(?:handover|hand-over|kt|knowledge\s+transfer)\s+(?:is|was|has\s+been)\s+(?:now\s+)?(?:complete|completed|"
    r"signed\s+off|accepted|finished)\b|\bsigned\s+off\s+(?:by|on)\b|\bsign-?off\s*:\s*(?:complete|done|approved|yes)\b"
    r"|\bkt\s+status\s*:?\s*complete\b", re.IGNORECASE)
_SECRET = re.compile(r"gsk_[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]+|"
                     r"ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|\bpassword\s*[:=]\s*\S+", re.IGNORECASE)
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60,
              "ninety": 90}
_DURATION = re.compile(r"\b(\d+(?:\.\d+)?|" + "|".join(_NUM_WORDS) + r")[\s-]*(seconds?|secs?|minutes?|mins?|hours?|"
                       r"hrs?|days?|weeks?|months?|years?)\b", re.IGNORECASE)
# Template wording that contains a duration but states nothing about the system
# ("First 30-Day Ownership Plan", the numbered heading "06 Day-1 Survival Checklist").
_TEMPLATE_DURATIONS = re.compile(r"first\s+30[\s-]day\s+ownership(?:\s+plan)?|\b\d{1,2}\s+day-1\b", re.IGNORECASE)
# How a section tells the reader that something was not given.
# Including the speaker's own words for it: "have not been formally
# documented", "the runbook is still missing".
_GAP_MARKER = re.compile(r"not covered|not stated|no fix was stated|not (?:been )?(?:\w+ly )?documented|"
                         r"never been documented|(?:is|are) (?:still )?missing|\bundocumented\b|"
                         r"can'?t remember|cannot remember|not discussed|confirm (?:\w+ )?with the outgoing owner",
                         re.IGNORECASE)
# A source number after a word, before a space, the end, or a list bullet the
# text layer puts beside it ("Database backups run daily12•").
_SUPERSCRIPT = re.compile(r"(?<=[A-Za-z.)”\"])\d{1,2}(?:,\d{1,2})*(?=[\s•·]|$)")
# Two table cells side by side come out of the text layer joined
# ("Delayed video processingWhen background jobs ..."): read apart as well.
_JOINED_CELLS = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")


def load_datasets() -> List[Dict[str, Any]]:
    out = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".json") and name != "floors.json":
            with open(os.path.join(DATA_DIR, name), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def load_floors() -> Dict[str, Any]:
    if not os.path.exists(FLOORS_PATH):
        return {}
    with open(FLOORS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def pdf_pages(pdf_bytes: bytes) -> List[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages]


def _body(pages: List[str]) -> str:
    """The document without its Sources appendix (which quotes the transcript
    verbatim on purpose) and without page footers."""
    text = "\n".join(pages)
    cut = text.find("§ Sources")
    text = text[:cut] if cut >= 0 else text
    return re.sub(r"Page \d+ of \d+", " ", text)


def _durations(text: str) -> set:
    found = set()
    for m in _DURATION.finditer(_TEMPLATE_DURATIONS.sub(" ", text)):
        n = m.group(1).lower()
        value = float(_NUM_WORDS.get(n, n)) if n in _NUM_WORDS else float(n)
        unit = re.sub(r"s$", "", m.group(2).lower()).replace("min", "minute").replace("sec", "second")
        unit = {"minuteute": "minute", "secondond": "second", "hr": "hour"}.get(unit, unit)
        found.add((value, unit))
    return found


def _products(text: str) -> set:
    from component_catalog import CATALOG_REGEX, lookup

    out = set()
    for m in CATALOG_REGEX.finditer(text):
        product = lookup(m.group(1))
        out.add(product.name if product else m.group(1).lower())
    return out


def _names(name: str) -> List[str]:
    low = name.lower()
    return [low] + _ALIASES.get(low, [])


def _node_matches(name: str, label: str) -> bool:
    label = (label or "").lower()
    return any(n in label for n in _names(name))


def _graph(ko: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for sec in ko.get("sections") or []:
        if sec.get("id") == "architecture_reference":
            return sec.get("_architecture_graph")
    return None


def _labels(graph: Dict[str, Any]) -> Dict[str, str]:
    labels = {n["id"]: n["label"] for key in ("entry", "data", "external", "delivery", "operations")
              for n in graph.get(key) or []}
    central = graph.get("central") or {}
    labels["central"] = " ".join(str(central.get(k) or "") for k in ("label", "header", "caption"))
    return labels


def check_relations(ds: Dict[str, Any], ko: Dict[str, Any]) -> Dict[str, List[str]]:
    graph = _graph(ko)
    missing, wrong = [], []
    if not graph:
        return {"missing": [f"{r['from']} -> {r['to']}" for r in ds.get("relations", [])], "wrong": []}
    labels = _labels(graph)
    edges = [(labels.get(e["from"], e["from"]), e["label"], labels.get(e["to"], e["to"])) for e in graph["edges"]]
    entry = [n["label"] for n in graph.get("entry") or []] + [labels["central"]]
    for rel in ds.get("relations", []):
        src, dst, kind = rel["from"], rel["to"], rel["kind"]
        ok = any(_node_matches(src, a) and _node_matches(dst, b) for a, _, b in edges)
        if kind == "flow":
            ok = ok or any(_node_matches(dst, a) and _node_matches(src, b) for a, _, b in edges)
            ok = ok or any(_node_matches(src, entry[i]) and _node_matches(dst, entry[i + 1]) for i in range(len(entry) - 1))
        if not ok:
            missing.append(f"{src} -[{kind}]-> {dst}")
    for rel in ds.get("wrong_relations", []):
        src = labels["central"] if rel["from"] == "SYSTEM" else rel["from"]
        for a, label, b in edges:
            if (_node_matches(src, a) if rel["from"] != "SYSTEM" else a == labels["central"]) \
                    and _node_matches(rel["to"], b) and re.search(rel["label"], label, re.IGNORECASE):
                wrong.append(f"{a} {label} {b}")
    return {"missing": missing, "wrong": wrong}


def _section_units(ko: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per section: fact-bearing units a reader sees (status lines excluded)
    and pointers to the same topic under another section."""
    from knowledge.evidence import _is_fact, block_units

    counts: Dict[str, Dict[str, int]] = {}
    for sec in ko.get("rendered_sections") or []:
        own = pointers = 0
        for block in sec.get("blocks") or []:
            title = str(block.get("title") or "")
            if title == "Gaps to confirm":
                continue
            n = sum(1 for u in (block_units(block) or []) if _is_fact(str(u)))
            if title == "Mentioned elsewhere in the KT":
                pointers += n
            else:
                own += n
        counts[sec.get("section_id")] = {"own": own, "pointers": pointers}
    return counts


def _section_text(ko: Dict[str, Any], section_id: str) -> str:
    from golden_eval import _block_strings

    for sec in ko.get("rendered_sections") or []:
        if sec.get("section_id") == section_id:
            return " ".join(" ".join([str(b.get("title") or "")] + _block_strings(b)) for b in sec.get("blocks") or [])
    return ""


def _marked_no_source(ds: Dict[str, Any], ko: Dict[str, Any]) -> tuple:
    """(how many units the document marks "no source", the answer-key facts
    shown in one of them). A known fact so marked tells the reader to
    distrust something the speaker said."""
    from knowledge.evidence import block_units

    marked = []
    for sec in ko.get("rendered_sections") or []:
        for block in sec.get("blocks") or []:
            units = block_units(block) or []
            marked += [norm(_SUPERSCRIPT.sub(" ", str(units[p]))) for p in block.get("unsourced") or [] if p < len(units)]
    found = []
    for fact in ds["facts"]:
        needles = [n for n in [norm(fact["quote"])] + [norm(a) for a in fact.get("alt", [])] if n]
        terms = [norm(t) for t in fact.get("terms", [])]
        if any(any(n in m for n in needles) or (terms and all(t in m for t in terms)) for m in marked):
            found.append(fact["id"])
    return len(marked), found


def evaluate(ds: Dict[str, Any], result: Dict[str, Any], pdf_bytes: bytes, mode: str = "no-llm") -> Dict[str, Any]:
    from devops_transcription import clean_transcript
    from kt_schema_loader import SCHEMA

    ko = result.get("knowledge_object") or {}
    pages = pdf_pages(pdf_bytes)
    body = _body(pages)
    body_norm = norm(body)
    body_norm_plain = norm(_SUPERSCRIPT.sub(" ", body))
    body_norm_cells = norm(_JOINED_CELLS.sub(" ", _SUPERSCRIPT.sub(" ", body)))
    transcript = clean_transcript(ds["transcript"])
    g = golden_score(ds, result)

    # Retention on the PDF itself: the quote (or an alt), or all terms on one page.
    page_norms = [norm(_SUPERSCRIPT.sub(" ", p)) for p in pages]
    retention = {}
    for fact in ds["facts"]:
        needles = [norm(fact["quote"])] + [norm(a) for a in fact.get("alt", [])]
        terms = [norm(t) for t in fact.get("terms", [])]
        in_pdf = any(n and (n in body_norm or n in body_norm_plain or n in body_norm_cells) for n in needles) or \
            bool(terms and any(all(t in p for t in terms) for p in page_norms))
        doc = next(f for f in g["facts"] if f["id"] == fact["id"])
        retention[fact["id"]] = "pdf" if in_pdf else ("document_only" if doc["status"] != "lost" else "lost")

    placed = {f["id"]: f for f in g["facts"]}
    safety_misplaced = [fid for fid in ds.get("safety", []) if placed[fid]["status"] != "correct"]
    # Incomplete procedures: the section must say what is missing.
    unflagged = [f"{item['topic']} ({item['section']})" for item in ds.get("incomplete", [])
                 if item.get("section") and not _GAP_MARKER.search(_section_text(ko, item["section"]))]
    undocumented = [u["topic"] for u in ds.get("undocumented", []) if re.search(u["forbid"], body, re.IGNORECASE)]
    forbidden = [f"{f['why']}: {m.group(0)}" for f in ds.get("forbidden", [])
                 for m in [re.search(f["pattern"], body, re.IGNORECASE)] if m]
    invented_durations = sorted(f"{v:g} {u}" for v, u in _durations(body) - _durations(transcript))
    invented_products = sorted(_products(body) - _products(transcript))
    framing = [m.group(0) for rx in _FRAMING for m in [re.search(rx, body, re.IGNORECASE)] if m]
    signoff = [m.group(0) for m in _SIGNOFF.finditer("\n".join(pages))]
    asr = []
    for item in ds.get("asr", []):
        if re.search(re.escape(item["heard"]), body, re.IGNORECASE):
            asr.append(f"'{item['heard']}' still printed")
        if not re.search(re.escape(item["expected"]), body, re.IGNORECASE):
            asr.append(f"'{item['expected']}' missing")
    relations = check_relations(ds, ko)

    # Coverage: the matrix against the content actually printed.
    from golden_eval import coverage_buckets

    buckets = coverage_buckets(ko)
    units = _section_units(ko)
    title_to_id = {s["title"]: s["id"] for s in SCHEMA}
    disagreements = []
    for sid, bucket in buckets.items():
        sid = title_to_id.get(sid, sid)
        seen = units.get(sid, {"own": 0, "pointers": 0})
        if bucket == "Missing" and seen["own"] > 0:
            disagreements.append(f"{sid}: reported Missing but shows {seen['own']} statement(s)")
        if bucket == "Strong" and seen["own"] == 0:
            disagreements.append(f"{sid}: reported Strong but shows no statement of its own")
        if bucket == "Partial" and seen["own"] == 0 and seen["pointers"] == 0:
            disagreements.append(f"{sid}: reported Partial but shows no statement")
    knowledge_summary = ko.get("_knowledge_coverage_summary") or {}

    placeholders = len(re.findall(r"This section was not covered in the KT session", body))
    no_source_marks, known_marked = _marked_no_source(ds, ko)
    layout = result.get("_pdf_layout") or {}
    overflows = [f"page {p}: {what} {points}pt" for p, what, points in layout.get("overflows") or []]
    secrets = [m.group(0)[:12] + "…" for m in _SECRET.finditer(ds["transcript"] + "\n".join(pages))]
    pdf_ok = bool(pages) and all(len(p.strip()) > 30 for p in pages)

    false_negatives = {
        "lost_facts": [fid for fid, s in retention.items() if s == "lost"],
        "facts_not_in_pdf_text": [fid for fid, s in retention.items() if s == "document_only"],
        "safety_warnings_misplaced": safety_misplaced,
        "relations_missing": relations["missing"],
        "false_missing_sections": g["false_missing"],
        "asr_not_fixed": asr,
        "incomplete_procedures_not_flagged": unflagged,
    }
    false_positives = {
        "misplaced_facts": g["wrong"],
        "incidents_in_danger_zones": g["violations"],
        "undocumented_given_a_value": undocumented,
        "forbidden_claims": forbidden,
        "durations_not_in_transcript": invented_durations,
        "products_not_in_transcript": invented_products,
        "wrong_relations": relations["wrong"],
        "session_framing_printed": framing,
        "false_signoff": signoff,
        "false_covered_sections": g["false_covered"],
        "coverage_disagrees_with_content": disagreements,
        "repeated_sentences": g["repeats"],
        "facts_without_source": g["unsourced"],
        "known_facts_marked_no_source": known_marked,
        "pdf_boxes_past_their_edge": overflows,
    }
    usage = result.get("llm_usage") or {}
    return {
        "id": ds["id"],
        "mode": mode,
        "status": result.get("status"),
        "system_name": ko.get("system_name"),
        "pdf": {"opens": pdf_ok, "pages": len(pages), "placeholder_sections": placeholders,
                "words_in": len(ds["transcript"].split()), "layout_checked": "overflows" in layout},
        "evidence": dict(ko.get("_evidence_stats") or {}, no_source_marks=no_source_marks),
        "facts": len(ds["facts"]),
        "retained_in_pdf": sum(1 for s in retention.values() if s == "pdf"),
        "retained_in_document": sum(1 for s in retention.values() if s != "lost"),
        "placement_accuracy": g["accuracy"],
        "relations": {"stated": len(ds.get("relations", [])), "found": len(ds.get("relations", [])) - len(relations["missing"])},
        "knowledge_coverage": knowledge_summary,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "secrets": secrets,
        "llm": {k: usage.get(k) for k in ("provider", "model", "cache_mode", "llm_calls", "cache_hits", "skipped",
                                          "failed", "rejected", "input_tokens", "output_tokens", "estimated_tokens",
                                          "llm_seconds", "retries")},
        "placement": {f["id"]: {"status": f["status"], "found_in": f["found_in"], "expected": f["expected"]}
                      for f in g["facts"]},
    }


def run_dataset(ds: Dict[str, Any], job_prefix: str = "e2e") -> tuple:
    """(pipeline result, PDF bytes) through pipeline.run_kt_pipeline, the
    orchestration POST /kt-from-transcript queues, and the real renderer.
    The caller decides whether an LLM is available (see run_all)."""
    import time

    import pipeline
    from devops_transcription import clean_transcript
    from pdf_rendering import build_job_pdf

    job_id = f"{job_prefix}-{ds['id']}"
    layout: Dict[str, Any] = {}
    try:
        result = pipeline.run_kt_pipeline(job_id, clean_transcript(ds["transcript"]))
        pdf = build_job_pdf(job_id, result, time.time(), layout=layout)
    finally:
        pipeline.JOB_STORE.delete(job_id)
    return dict(result, _pdf_layout=layout), pdf


def run_all(datasets=None, out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every dataset with no LLM (deterministic, no quota)."""
    import ai
    import llm_provider
    import pipeline
    from context_mapper import ContextMappingPipeline
    from kt_schema_loader import SCHEMA

    # llm_provider's own lookup too: the copy edit (copyedit.py) asks it
    # directly, so a key in .env would otherwise still send requests.
    saved = (pipeline.get_llm_provider, ai.get_llm_provider, llm_provider.get_llm_provider,
             pipeline.record_candidates, pipeline.MAPPER_PIPELINE)
    pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
    pipeline.record_candidates = lambda *a, **k: None
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    try:
        reports = []
        for ds in datasets or load_datasets():
            result, pdf = run_dataset(ds)
            if out_dir:
                save_outputs(out_dir, ds["id"], result, pdf)
            reports.append(evaluate(ds, result, pdf, mode="no-llm"))
        return reports
    finally:
        (pipeline.get_llm_provider, ai.get_llm_provider, llm_provider.get_llm_provider,
         pipeline.record_candidates, pipeline.MAPPER_PIPELINE) = saved


def run_replay(recording_path: str, datasets=None, out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every dataset in LLM mode, answered only from a recording of a live run
    (scripts/e2e_kts.py --export-recording): LLM_CACHE=replay, no API key, so
    nothing can be sent and an unrecorded request is skipped and counted. The
    process's settings are restored afterwards."""
    import tempfile
    import time

    import llm_provider
    import pipeline
    from context_mapper import ContextMappingPipeline
    from kt_schema_loader import SCHEMA
    from llm import cache

    with open(recording_path, encoding="utf-8") as fh:
        recording = json.load(fh)
    meta = recording["recorded_with"]
    env = {"LLM_CACHE": "replay",
           "LLM_CACHE_PATH": os.path.join(tempfile.mkdtemp(prefix="e2e_replay_"), "replay.sqlite"),
           **{k: str(v) for k, v in (meta.get("settings") or {}).items()}}
    saved_env = {k: os.environ.get(k) for k in env}
    names = ("LLM_PROVIDER_NAME", "GROQ_MODEL", "GROQ_API_KEY", "GEMINI_API_KEY")
    saved_llm = {n: getattr(llm_provider, n) for n in names}
    saved_pipeline = (pipeline.MAPPER_PIPELINE, pipeline.record_candidates)
    try:
        os.environ.update(env)
        llm_provider.LLM_PROVIDER_NAME = meta["provider_setting"]
        llm_provider.GROQ_MODEL = meta["model"]
        llm_provider.GROQ_API_KEY = llm_provider.GEMINI_API_KEY = None
        conn = cache._connect(cache.cache_path())
        conn.executemany(
            "INSERT OR REPLACE INTO llm_responses (key, call_site, provider, model, response, input_tokens,"
            " output_tokens, created_at, hit_count, tenant) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            [(r["key"], r["call_site"], r["provider"], r["model"], r["response"], r["input_tokens"],
              r["output_tokens"], time.time(), r.get("tenant")) for r in recording["responses"]])
        conn.commit()
        conn.close()
        provider = llm_provider.get_llm_provider()
        pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=provider.generate if provider else None)
        pipeline.record_candidates = lambda *a, **k: None
        reports = []
        for ds in datasets or load_datasets():
            # Recorded answers are kept per tenant: run as the recording's tenant.
            pipeline.JOB_QUEUE[f"e2e-replay-{ds['id']}"] = {"status": "queued", "tenant_id": meta.get("tenant") or "local"}
            result, pdf = run_dataset(ds, job_prefix="e2e-replay")
            if out_dir:
                save_outputs(out_dir, ds["id"], result, pdf)
            reports.append(evaluate(ds, result, pdf, mode="llm-replay (recorded responses, nothing sent)"))
        return reports
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for n, v in saved_llm.items():
            setattr(llm_provider, n, v)
        pipeline.MAPPER_PIPELINE, pipeline.record_candidates = saved_pipeline


def save_outputs(out_dir: str, name: str, result: Dict[str, Any], pdf: bytes) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".pdf"), "wb") as fh:
        fh.write(pdf)
    with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as fh:
        for i, text in enumerate(pdf_pages(pdf), 1):
            fh.write(f"\n===== PAGE {i} =====\n{text}")


def problems(report: Dict[str, Any]) -> int:
    return sum(len(v) for v in report["false_negatives"].values()) + \
        sum(len(v) for v in report["false_positives"].values())
