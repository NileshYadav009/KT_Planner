"""Speech to text without punctuation, on the visible KTs (audit §9abe).

    python evaluation/noisy_copies.py LABEL [flat|lower]

Every tuning golden and dev KT is run as an unpunctuated, lower-case
transcript and scored on its own answer key: the scorer ignores case and
punctuation, so the keys still apply. Holdout and blind KTs are never read.

flat   lower case; no punctuation inside a paragraph, a full stop at its end;
       speaker labels dropped (the default)
lower  lower case only

Some speech-to-text engines return text like this, and a noisy KT was the
blind set's weakest (X67). Prints the aggregate and, per KT, the facts not
placed and the sections wrongly reported Missing or covered. Results:
results/noisy-LABEL.json. About 2 minutes once evaluation/.cache/ is warm.
"""
import glob
import json
import os
import re
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
label = sys.argv[1]
style = sys.argv[2] if len(sys.argv) > 2 else "flat"

state = tempfile.mkdtemp(prefix="noisy-")
os.environ.update({
    "CONTINUUM_DB_PATH": os.path.join(state, "continuum.sqlite"),
    "CONTINUUM_VOCAB_CANDIDATES_PATH": os.path.join(state, "glossary_candidates.json"),
    "KT_ASSETS_DIR": os.path.join(state, "kt_assets"),
    "CONTINUUM_AUTH": "disabled",
    "CONTINUUM_COPYEDIT": "0",
    "LLM_CACHE_PATH": os.path.join(state, "llm_cache.sqlite"),
})
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[key] = ""
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
if os.environ.get("NO_ENC_CACHE") != "1":
    import enccache  # noqa: E402
    enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

import golden_eval  # noqa: E402

_HIDDEN = re.compile(r"(h1[2-4]|b\d)")


def visible_kts():
    out = []
    for path in sorted(glob.glob("tests/goldens/*.json")) + sorted(glob.glob(os.path.join(HERE, "dev", "*.json"))):
        with open(path, encoding="utf-8") as fh:
            kt = json.load(fh)
        if not isinstance(kt, dict) or "transcript" not in kt or "facts" not in kt:
            continue
        if kt.get("holdout") or _HIDDEN.match(kt["id"]):
            continue
        out.append(kt)
    return out


def noisy(text: str) -> str:
    paragraphs = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    if style == "lower":
        return "\n\n".join(p.lower() for p in paragraphs)
    flat = []
    for p in paragraphs:
        p = re.sub(r"^[A-Z][a-z]+:\s*", "", p.strip())
        p = re.sub(r"(?<!\d)[.,;:!?](?!\d)", " ", p.lower())
        p = re.sub(r"[—–]", " ", p)
        flat.append(re.sub(r"\s+", " ", p).strip() + ".")
    return "\n\n".join(flat)


t0 = time.time()
scores = golden_eval.run_all([dict(kt, transcript=noisy(kt["transcript"])) for kt in visible_kts()])
agg = {"facts": 0, "correct": 0, "wrong": 0, "lost": 0, "false_missing": 0, "false_covered": 0}
detail = []
for s in scores:
    correct = sum(f["status"] == "correct" for f in s["facts"])
    agg["facts"] += len(s["facts"])
    agg["correct"] += correct
    for key in ("wrong", "lost", "false_missing", "false_covered"):
        agg[key] += len(s[key])
    detail.append({"id": s["id"], "accuracy": round(correct / max(len(s["facts"]), 1), 3),
                   "false_missing": s["false_missing"], "false_covered": s["false_covered"],
                   "facts": [f for f in s["facts"] if f["status"] != "correct"]})
agg["accuracy"] = round(agg["correct"] / max(agg["facts"], 1), 4)
print(json.dumps(agg))
for d in detail:
    if d["false_missing"] or d["false_covered"] or d["facts"]:
        print(f"{d['id']}  {d['accuracy']:.0%}  false Missing {d['false_missing']}  false covered {d['false_covered']}")
        for f in d["facts"]:
            print(f"  {f['id']} {f['status']:5}  exp {f['expected']} found {f['found_in']}  << {f['quote']}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", f"noisy-{label}.json"), "w", encoding="utf-8") as fh:
    json.dump({"label": label, "style": style, "aggregate": agg, "detail": detail}, fh, indent=1)
print("seconds", round(time.time() - t0))
