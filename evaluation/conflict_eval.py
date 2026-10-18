"""The LLM conflict check (conflicts.find_conflicts_with_llm) on KTs: SPENDS QUOTA.
Run only when agreed.

    python evaluation/conflict_eval.py LABEL SETS     # SETS: matrix,dev,e2e

Each KT runs through the pipeline without an LLM, as everywhere else in
evaluation/; then only the conflict check is sent, through the configured
provider and the response cache (evaluation/.cache/llm_cache.sqlite), so a
re-run of unchanged pairs costs nothing. GROQ_BUDGET (default 40) caps live
requests. Holdout and blind KTs are not run.

matrix  the contradiction cases (D01-D04, A14) and controls whose pairs must
        not be flagged (environments, corrections, exceptions, hedges)
dev     evaluation/dev: every flag is printed, to read for false flags
e2e     tests/e2e_kts: the same
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
label, sets = sys.argv[1], sys.argv[2].split(",")
BUDGET = int(os.environ.get("GROQ_BUDGET", "40"))

state = tempfile.mkdtemp(prefix="conflict-eval-")
os.makedirs(os.path.join(HERE, ".cache"), exist_ok=True)
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "continuum.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "glossary_candidates.json"),
    "KT_ASSETS_DIR": os.path.join(state, "kt_assets"),
    "CONTINUUM_AUTH": "disabled",
    "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE": "readwrite",
    "LLM_CACHE_PATH": os.path.join(HERE, ".cache", "llm_cache.sqlite"),
    "LLM_MAX_TOKENS_PER_MINUTE": os.environ.get("LLM_MAX_TOKENS_PER_MINUTE", "7000"),
})
TENANT = "t_eced34cf4400"
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
import enccache  # noqa: E402
enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

import ai  # noqa: E402
import llm_provider  # noqa: E402
import pipeline  # noqa: E402
from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from knowledge.knowledge_builder import apply_llm_conflict_check  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402
from llm.tenant_context import CURRENT_TENANT  # noqa: E402
from llm.usage import LLMUsageTracker, start_tracking, stop_tracking  # noqa: E402

provider = llm_provider.get_llm_provider()
assert provider is not None, "no LLM provider configured"
print(f"provider={llm_provider.LLM_PROVIDER_NAME} model={llm_provider.GROQ_MODEL} budget={BUDGET}", flush=True)
# The pipeline runs without an LLM; only the conflict check below is sent.
pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
pipeline.record_candidates = lambda *a, **k: None
pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
encode = pipeline.MAPPER_PIPELINE.classifier._encode_texts

EXPECTED_PAIR = {}
MATRIX_EXPECT = {"D01": True, "D02": True, "D03": True, "D04": True, "A14": True,
                 "A06": False, "A12": False, "A13": False, "C03": False, "E03": False, "F01": False, "G01": False,
                 "B03": False}


def kts(name):
    if name == "matrix":
        with open(os.path.join(HERE, "matrix", "cases.json"), encoding="utf-8") as fh:
            cases = [c for c in json.load(fh)["cases"] if c["id"] in MATRIX_EXPECT]
        for c in cases:
            EXPECTED_PAIR[c["id"]] = next((e["texts"] for e in c["expect"] if e["check"] == "flagged"), None)
        return [(c["id"], "\n".join(c["transcript"])) for c in cases]
    folder = {"dev": os.path.join(HERE, "dev"), "e2e": os.path.join(REPO, "tests", "e2e_kts")}[name]
    out = []
    for n in sorted(os.listdir(folder)):
        if n.endswith(".json") and n != "floors.json":
            d = json.load(open(os.path.join(folder, n), encoding="utf-8"))
            if d.get("transcript"):
                out.append((d.get("id") or n, d["transcript"]))
    return out


report = {"label": label}
sent = tokens_in = tokens_out = 0
t0 = time.time()
for name in sets:
    rows = []
    for kt_id, transcript in kts(name):
        if sent >= BUDGET:
            print("budget reached", flush=True)
            break
        job_id = f"conflict-{kt_id}"
        result = pipeline.run_kt_pipeline(job_id, clean_transcript(transcript))
        pipeline.JOB_STORE.delete(job_id)
        ko, coverage = result.get("knowledge_object") or {}, result.get("coverage") or {}
        before = len(ko.get("_conflicts") or [])
        tracker = LLMUsageTracker(job_id)
        token, tenant = start_tracking(tracker), CURRENT_TENANT.set(TENANT)
        try:
            apply_llm_conflict_check(ko, coverage, provider.generate, encode)
        finally:
            CURRENT_TENANT.reset(tenant)
            stop_tracking(token)
        usage = tracker.summary()
        sent += usage.get("llm_calls") or 0
        tokens_in += usage.get("input_tokens") or 0
        tokens_out += usage.get("output_tokens") or 0
        flags = [(c["a"], c["b"]) for c in (ko.get("_conflicts") or [])[before:]]
        row = {"id": kt_id, "llm_flags": flags, "rule_flags": before, "llm_calls": usage.get("llm_calls"),
               "cache_hits": usage.get("cache_hits")}
        if name == "matrix":
            # Right: some flag (rules or LLM) holds the expected pair. False:
            # an LLM flag that does not.
            want = [t.lower() for t in EXPECTED_PAIR.get(kt_id) or []]

            def holds(a, b):
                return bool(want) and all(t in f"{a} {b}".lower() for t in want)

            right = any(holds(c["a"], c["b"]) for c in ko.get("_conflicts") or [])
            false_flags = [f for f in flags if not holds(*f)]
            row.update(expected_flag=MATRIX_EXPECT[kt_id], right=right, false_flags=len(false_flags),
                       ok=(right or not MATRIX_EXPECT[kt_id]) and not false_flags)
        rows.append(row)
        mark = "" if name != "matrix" else ("ok   " if row["ok"] else "WRONG")
        print(f"{name:6} {kt_id:34} {mark} rules={before} llm={len(flags)} calls={usage.get('llm_calls')}", flush=True)
        for a, b in flags:
            print(f"         “{a[:110]}”  <>  “{b[:110]}”", flush=True)
    report[name] = rows
report.update(live_requests=sent, input_tokens=tokens_in, output_tokens=tokens_out, seconds=round(time.time() - t0))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", f"conflicts-{label}.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1)
print(json.dumps({k: report[k] for k in ("live_requests", "input_tokens", "output_tokens", "seconds")}), flush=True)
