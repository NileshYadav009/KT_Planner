"""Round-6 accuracy runner (no LLM, copy edit off, scratch state).

    python evaluation/accuracy.py LABEL SETS     # SETS: comma list of dev,tuning,e2e,holdout,blind2,blind3,blind4

dev      the six round-6 KTs in ./dev (per-fact detail)
tuning   non-holdout goldens in tests/goldens (per-fact detail)
e2e      tests/e2e_kts via e2e_eval (problems only)
holdout  h12-h14: AGGREGATE ONLY. Nothing per fact or per document is
         printed or saved.

Writes ./LABEL.json (dev + tuning per fact, holdout aggregate) and prints a
summary.
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTINUUM_REPO", os.path.dirname(HERE))
label = sys.argv[1]
sets = sys.argv[2].split(",")

state = tempfile.mkdtemp(prefix="acc90-")
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
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, HERE)
if os.environ.get("NO_ENC_CACHE") != "1":
    import enccache  # noqa: E402
    enccache.install(os.path.join(HERE, ".cache", "enc_cache.sqlite"))

import golden_eval  # noqa: E402
import pipeline  # noqa: E402

# No load_models(): golden_eval.run_all builds its own mapper; Whisper is not needed.

t0 = time.time()
out = {"label": label}


def load_dir(path):
    docs = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            with open(os.path.join(path, name), encoding="utf-8") as fh:
                docs.append(json.load(fh))
    return docs


def tally(scores):
    facts = sum(len(s["facts"]) for s in scores)
    correct = sum(sum(f["status"] == "correct" for f in s["facts"]) for s in scores)
    wrong = sum(len(s["wrong"]) for s in scores)
    lost = sum(len(s["lost"]) for s in scores)
    return {"facts": facts, "correct": correct, "wrong": wrong, "lost": lost,
            "accuracy": round(correct / facts, 4) if facts else None,
            "false_missing": sum(len(s["false_missing"]) for s in scores),
            "false_covered": sum(len(s["false_covered"]) for s in scores),
            "violations": sum(len(s["violations"]) for s in scores),
            "repeats": sum(len(s["repeats"]) for s in scores),
            "unsourced": sum(len(s["unsourced"]) for s in scores)}


def detail(scores):
    rows = []
    for s in scores:
        probs = [f for f in s["facts"] if f["status"] != "correct" or f["violations"]]
        rows.append({"id": s["id"], "accuracy": s["accuracy"], "problems": probs,
                     "false_missing": s["false_missing"], "false_covered": s["false_covered"],
                     "repeats": s["repeats"], "unsourced": s["unsourced"],
                     "facts": [{k: f[k] for k in ("id", "status", "found_in", "expected")} for f in s["facts"]]})
    return rows


goldens = golden_eval.load_goldens()
if "dev" in sets:
    scores = golden_eval.run_all(load_dir(os.path.join(HERE, "dev")))
    out["dev"] = tally(scores)
    out["dev_detail"] = detail(scores)
    print("dev", out["dev"], flush=True)
if "tuning" in sets:
    scores = golden_eval.run_all([g for g in goldens if not g.get("holdout")])
    out["tuning"] = tally(scores)
    out["tuning_detail"] = detail(scores)
    print("tuning", out["tuning"], flush=True)
if "blind2" in sets:
    # Round-6 blind KTs (./blind): aggregate only, nothing per fact.
    scores = golden_eval.run_all(load_dir(os.path.join(HERE, "blind")))
    out["blind2"] = tally(scores)
    del scores
    print("blind2 (aggregate only)", out["blind2"], flush=True)
if "blind3" in sets:
    # Round-7 blind KTs (./blind3, written 25 Oct): aggregate only, nothing per fact.
    scores = golden_eval.run_all(load_dir(os.path.join(HERE, "blind3")))
    out["blind3"] = tally(scores)
    del scores
    print("blind3 (aggregate only)", out["blind3"], flush=True)
if "blind4" in sets:
    # Round-8 blind KTs (./blind4, written 26 Oct, sealed for §9abd): aggregate only.
    scores = golden_eval.run_all(load_dir(os.path.join(HERE, "blind4")))
    out["blind4"] = tally(scores)
    del scores
    print("blind4 (aggregate only)", out["blind4"], flush=True)
if "holdout" in sets:
    scores = golden_eval.run_all([g for g in goldens if g.get("holdout")])
    out["holdout"] = tally(scores)
    del scores
    print("holdout (aggregate only)", out["holdout"], flush=True)
if "e2e" in sets:
    import e2e_eval  # noqa: E402
    reports = e2e_eval.run_all()
    out["e2e"] = [{"id": r["id"], "placement_accuracy": r["placement_accuracy"], "retained_in_pdf": r["retained_in_pdf"],
                   "relations": r["relations"], "problems": e2e_eval.problems(r)} for r in reports]
    for e in out["e2e"]:
        print("e2e", e["id"], e["placement_accuracy"], e["retained_in_pdf"], "problems:", e["problems"], flush=True)

out["seconds"] = round(time.time() - t0)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", label + ".json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)

for key in ("dev", "tuning"):
    for row in out.get(key + "_detail", []):
        if row["problems"] or row["false_missing"] or row["false_covered"]:
            print(f"\n{row['id']}  {row['accuracy']:.0%}")
            for f in row["problems"]:
                print(f"  {f['id']} {f['status']:6} exp {f['expected']} found {f['found_in']}  << {f['quote']}")
            for k in ("false_missing", "false_covered"):
                if row[k]:
                    print(f"  {k}: {row[k]}")
print("\nseconds", out["seconds"])
