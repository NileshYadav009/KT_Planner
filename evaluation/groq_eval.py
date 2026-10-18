"""LLM-mode accuracy with the configured provider (Groq): SPENDS QUOTA.
Run only when agreed.

    python evaluation/groq_eval.py LABEL SETS     # SETS: dev,holdout,blind2

dev      per fact, saved to evaluation/results/LABEL.json
holdout  h12-h14: AGGREGATE ONLY
blind2   evaluation/blind (b01-b03): AGGREGATE ONLY

Stops before a KT once GROQ_BUDGET live requests (default 150) were sent.
Unchanged prompts are answered from evaluation/.cache/llm_cache.sqlite and
cost nothing. Scratch database and assets; copy edit off, like every other
accuracy figure; throttled under the account's tokens-per-minute limit. The
key is read by llm_provider from .env and never printed. Lenient scoring
counts a fact the polish reworded (lenient.py).

GROQ_EVAL_ONLY=classify sends only the mapper's calls (the section check and
a sentence repair); polish, structured extraction and field fill run without
the LLM. For an A/B of the section check alone: the arms then differ only
where the check does, scoring is exact (strict equals lenient), and the cost
is a fraction of a full run.
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
label, sets = sys.argv[1], sys.argv[2].split(",")
BUDGET = int(os.environ.get("GROQ_BUDGET", "150"))

state = tempfile.mkdtemp(prefix="groq-eval-")
os.makedirs(os.path.join(HERE, ".cache"), exist_ok=True)
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "continuum.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "glossary_candidates.json"),
    "KT_ASSETS_DIR": os.path.join(state, "kt_assets"),
    "CONTINUUM_AUTH": "disabled",
    "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE": "readwrite",
    "LLM_CACHE_PATH": os.path.join(HERE, ".cache", "llm_cache.sqlite"),
    "LLM_BATCH_CALLS": os.environ.get("LLM_BATCH_CALLS", "1"),
    "LLM_MAX_CALLS_PER_KT": os.environ.get("LLM_MAX_CALLS_PER_KT", "60"),
    "LLM_MAX_TOKENS_PER_MINUTE": os.environ.get("LLM_MAX_TOKENS_PER_MINUTE", "7000"),
})
# The tenant earlier recordings were made for, so their cached answers apply.
TENANT = "t_eced34cf4400"

sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")

if os.environ.get("NO_ENC_CACHE") != "1":
    import enccache  # noqa: E402
    enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

import ai  # noqa: E402
import golden_eval  # noqa: E402
import llm_provider  # noqa: E402
import pipeline  # noqa: E402
from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402
from lenient import lenient_tally  # noqa: E402

provider = llm_provider.get_llm_provider()
assert provider is not None, "no LLM provider configured"
print(f"provider={llm_provider.LLM_PROVIDER_NAME} model={llm_provider.GROQ_MODEL} "
      f"candidates={os.environ.get('LLM_VERIFY_CANDIDATES', '3')} purpose={os.environ.get('LLM_VERIFY_PURPOSE', '0')} "
      f"budget={BUDGET}", flush=True)
pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=provider.generate)
pipeline.record_candidates = lambda *a, **k: None
ONLY = os.environ.get("GROQ_EVAL_ONLY", "")
if ONLY == "classify":
    # The mapper holds the provider itself; every later stage looks it up.
    pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
print(f"stages={'mapper only' if ONLY == 'classify' else 'all'}", flush=True)

USAGE = ("llm_calls", "cache_hits", "skipped", "failed", "rejected", "input_tokens", "output_tokens", "retries",
         "llm_seconds")


def load_dir(path):
    return [json.load(open(os.path.join(path, n), encoding="utf-8")) for n in sorted(os.listdir(path))
            if n.endswith(".json")]


goldens = golden_eval.load_goldens()
docs = {"dev": load_dir(os.path.join(HERE, "dev")),
        "holdout": [g for g in goldens if g.get("holdout")],
        "blind2": load_dir(os.path.join(HERE, "blind"))}
out = {"label": label, "budget": BUDGET, "only": ONLY, "variant": {k: os.environ.get(k) for k in
                                                                   ("LLM_VERIFY_CANDIDATES", "LLM_VERIFY_PURPOSE")}}
sent = 0
t0 = time.time()
for name in sets:
    agg = {"facts": 0, "correct": 0, "wrong": 0, "lost": 0, "unsourced": 0, "repeats": 0, "false_missing": 0,
           "false_covered": 0, "lenient_correct": 0, "lenient_lost": 0, "kts": 0}
    usage = {k: 0 for k in USAGE}
    detail = []
    for g in docs[name]:
        if sent >= BUDGET:
            print(f"budget reached; {name}: stopping before {g['id'] if name == 'dev' else 'a KT'}", flush=True)
            break
        job_id = f"golden-{g['id']}"
        pipeline.JOB_QUEUE[job_id] = {"status": "queued", "tenant_id": TENANT}
        try:
            result = pipeline.run_kt_pipeline(job_id, clean_transcript(g["transcript"]))
        finally:
            pipeline.JOB_STORE.delete(job_id)
        s = golden_eval.score(g, result)
        lenient = lenient_tally(g, result)
        u = result.get("llm_usage") or {}
        for k in USAGE:
            usage[k] += u.get(k) or 0
        sent += u.get("llm_calls") or 0
        agg["kts"] += 1
        agg["facts"] += len(s["facts"])
        agg["correct"] += sum(f["status"] == "correct" for f in s["facts"])
        for k in ("wrong", "lost", "unsourced", "repeats", "false_missing", "false_covered"):
            agg[k] += len(s[k])
        agg["lenient_correct"] += lenient["correct"]
        agg["lenient_lost"] += lenient["lost"]
        if name == "dev":
            detail.append({"id": g["id"], "accuracy": s["accuracy"],
                           "problems": [f for f in s["facts"] if f["status"] != "correct"],
                           "false_missing": s["false_missing"], "false_covered": s["false_covered"],
                           "repeats": s["repeats"], "unsourced": s["unsourced"], "lenient": lenient,
                           "usage": {k: u.get(k) for k in USAGE}})
        del s, result
    out[name] = {**agg, "usage": usage}
    if detail:
        out[name + "_detail"] = detail
    print(name, json.dumps(out[name]), flush=True)
out["live_requests"] = sent
out["seconds"] = round(time.time() - t0)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", label + ".json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)
print("done", label, "live requests", sent, "seconds", out["seconds"], flush=True)
