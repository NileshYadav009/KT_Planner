"""One visible KT (dev d0x or a tuning golden; never a holdout/blind KT),
no LLM: what chosen sections contain and why the coverage table rates them.

    python evaluation/kodiag.py KT_ID SECTION[,SECTION...]
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
kt_id, sections = sys.argv[1], sys.argv[2].split(",")

state = tempfile.mkdtemp(prefix="kodiag-")
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

import golden_eval  # noqa: E402


def find(kt):
    for path in [os.path.join(HERE, "dev", kt + ".json")]:
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))
    for g in golden_eval.load_goldens():
        if g["id"] == kt:
            return g
    raise SystemExit("not found: " + kt)


golden = find(kt_id)
if golden.get("holdout") or kt_id.startswith(("h12", "h13", "h14", "b0", "b1")):
    raise SystemExit("holdout/blind KT: aggregate only")

import ai  # noqa: E402
import llm_provider  # noqa: E402
import pipeline  # noqa: E402
from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402

pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
pipeline.record_candidates = lambda *a, **k: None
pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
job = "kodiag-" + kt_id
result = pipeline.run_kt_pipeline(job, clean_transcript(golden["transcript"]))
ko = result["knowledge_object"]
texts = golden_eval.section_texts(ko)
rows = {}
for sec in ko.get("sections") or []:
    if sec.get("id") == "kt_coverage":
        rows = {r.get("Domain"): r for r in sec.get("_coverage_rows") or []}
titles = {s["id"]: s["title"] for s in SCHEMA}
print("golden missing:", golden.get("missing"))
for sid in sections:
    print("=" * 70)
    print(sid, "| coverage row:", json.dumps(rows.get(titles.get(sid)), default=str)[:600])
    for t in texts.get(sid, []):
        print("   -", str(t)[:260])
    for sec in ko.get("sections") or []:
        if sec.get("id") == sid:
            print("   source sentences:", [str(x)[:160] for x in (sec.get("_source_sentences") or [])])
            print("   status:", sec.get("status"), "| fields filled:",
                  [k for k, v in (sec.get("fields") or {}).items() if (v or {}).get("source") not in (None, "unfilled")])
pipeline.JOB_STORE.delete(job)
