"""Mapper trace for a probe transcript (not a holdout): each shown unit, its
section, the decision path and reasoning.

    python evaluation/trace.py MODULE.VAR [FILTER]
e.g. python evaluation/trace.py adversarial_text.TRANSCRIPT region
"""
import importlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
state = tempfile.mkdtemp(prefix="trace-")
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "c.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "g.json"),
    "KT_ASSETS_DIR": os.path.join(state, "a"),
    "CONTINUUM_AUTH": "disabled", "CONTINUUM_COPYEDIT": "0",
})
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[key] = ""
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
import enccache  # noqa: E402
enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

mod, var = sys.argv[1].rsplit(".", 1)
text = getattr(importlib.import_module(mod), var)
flt = sys.argv[2].lower() if len(sys.argv) > 2 else ""

from context_mapper import ContextMappingPipeline  # noqa: E402
from devops_transcription import clean_transcript  # noqa: E402
from kt_schema_loader import SCHEMA  # noqa: E402

import context_mapper as cm  # noqa: E402
captured = {}
_orig = cm.restore_sections_emptied_by_topic


def _spy(units, picks):
    captured["units"], captured["picks"] = units, picks
    return _orig(units, picks)


cm.restore_sections_emptied_by_topic = _spy
mapper = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
cleaned = clean_transcript(text)
kt = mapper.process("trace", cleaned, [{"start": 0.0, "end": 0.0, "text": cleaned, "avg_logprob": -0.3}])
logs = kt.explainability_logs or []
print("units", len(kt.sentences), "logs", len(logs), "facts", len(kt.facts or []))
pairs = list(zip(kt.facts or [], kt.fact_sections or []))
for i, (f, sid) in enumerate(pairs):
    if flt and flt not in f.text.lower():
        continue
    print(f"[{i}] {str(sid):28} {f.text[:120]}")
for log in logs:
    if flt and flt not in str(log.reasoning).lower() and flt not in str(getattr(log, "sentence_text", "")).lower():
        pass
for s, log in zip(kt.sentences, logs if len(logs) == len(kt.sentences) else []):
    if flt and flt not in s.text.lower():
        continue
    print(f"   UNIT {log.action:16} {log.section_id:26} {log.confidence:.3f} {str(log.reasoning)[:170]} || {s.text[:70]}")

for i, cs in enumerate(captured.get("units", [])):
    if flt and flt not in cs.sentence.text.lower():
        continue
    log = cs.explainability_log
    sec = [f"{c.section_id}:{c.confidence:.3f}" for c in (cs.secondary_classifications or [])[:3]]
    print(f"   FACT {cs.primary_classification.section_id:26} {cs.primary_classification.confidence:.3f} "
          f"{log.action:16} {str(log.reasoning)[:150]} | 2nd {sec} || {cs.sentence.text[:60]}")
