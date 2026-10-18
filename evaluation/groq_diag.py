"""Replay one dev KT's LLM run from evaluation/.cache/llm_cache.sqlite
(LLM_CACHE=replay: no request is ever sent, no quota) and show its repeated
and unsourced lines. Dev and tuning KTs only, never holdout or blind ones.

    python evaluation/groq_diag.py KT_ID

Run it with the settings of the run being diagnosed (LLM_VERIFY_* etc.):
a request the run did not make is unrecorded and takes the rules-only path.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
kt_id = sys.argv[1]
if kt_id.startswith(("h12", "h13", "h14", "b0")):
    raise SystemExit("holdout/blind KT: aggregate only")

state = tempfile.mkdtemp(prefix="groq-diag-")
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "continuum.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "glossary_candidates.json"),
    "KT_ASSETS_DIR": os.path.join(state, "kt_assets"),
    "CONTINUUM_AUTH": "disabled",
    "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE": "replay",
    "LLM_CACHE_PATH": os.path.join(HERE, ".cache", "llm_cache.sqlite"),
    "LLM_BATCH_CALLS": os.environ.get("LLM_BATCH_CALLS", "1"),
    "LLM_MAX_CALLS_PER_KT": os.environ.get("LLM_MAX_CALLS_PER_KT", "60"),
})
TENANT = "t_eced34cf4400"
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
import enccache  # noqa: E402
enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

import golden_eval  # noqa: E402
import llm_provider  # noqa: E402
import pipeline  # noqa: E402
from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402

golden = json.load(open(os.path.join(HERE, "dev", kt_id + ".json"), encoding="utf-8"))
provider = llm_provider.get_llm_provider()
assert provider is not None
pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=provider.generate)
pipeline.record_candidates = lambda *a, **k: None
job = "golden-" + kt_id
pipeline.JOB_QUEUE[job] = {"status": "queued", "tenant_id": TENANT}
try:
    result = pipeline.run_kt_pipeline(job, clean_transcript(golden["transcript"]))
finally:
    pipeline.JOB_STORE.delete(job)
ko = result["knowledge_object"]
score = golden_eval.score(golden, result)
usage = result.get("llm_usage") or {}
print("replayed:", usage.get("cache_hits"), "| not recorded (rules-only):", usage.get("skipped"),
      "| live:", usage.get("llm_calls"))
texts = golden_eval.section_texts(ko)
print("REPEATS:")
for rep in score["repeats"]:
    print("  -", str(rep)[:200])
    key = golden_eval.norm(str(rep))[:60]
    for sid, items in texts.items():
        for t in items:
            if key and key in golden_eval.norm(str(t)):
                print(f"      in {sid}: {str(t)[:220]}")
print("UNSOURCED:")
units = golden_eval.sourced_units(ko)
for fid in score["unsourced"]:
    spec = next(s for s in golden["facts"] if s["id"] == fid)
    needle = golden_eval.norm(spec["quote"])
    print(f"  {fid}: {spec['quote'][:120]}")
    for text, quotes in units:
        if needle in text:
            print(f"      shown as: {text[:220]}")
            print(f"      sources:  {[q[:120] for q in quotes] or 'none'}")
print("WRONG / LOST:")
for f in score["facts"]:
    if f["status"] != "correct":
        spec = next(s for s in golden["facts"] if s["id"] == f["id"])
        print(f"  {f['id']} {f['status']:6} want {spec['in']} found {f['found_in']} | {spec['quote'][:90]}")
