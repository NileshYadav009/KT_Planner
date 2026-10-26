#!/usr/bin/env python3
"""Write section_model.npz: a small learned section model for the classifier.

    python scripts/build_section_model.py               # rewrite the file
    python scripts/build_section_model.py --enc-cache   # reuse evaluation/.cache embeddings (faster)

The classifier scores each fact against every section from the section's
title and hints, its labelled examples (section_examples.json), the spoken
context, hint words, entities and the running topic
(context_mapper.ContextClassifier). This adds one more signal: a logistic
regression over the fact's embedding and its context's embedding (the two
the classifier already computes), trained on the annotated facts of the
synthetic tuning goldens (tests/goldens) and the development KTs
(evaluation/dev). The classifier adds SECTION_MODEL_WEIGHT x its probability
to each section's score.

Holdouts and the blind sets are left out, so they still measure KTs the
model has not seen, and so is any transcript that is not synthetic (a
customer's KT never ships inside the product).

Each KT runs through the pipeline without an LLM; every unit the classifier
scores is labelled with the first section of the annotated fact it states.
A unit stating facts of different sections is left out. Rebuild after
adding training KTs or changing how transcripts are cut into facts.
"""
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "tests", "goldens")
DEV_DIR = os.path.join(ROOT, "evaluation", "dev")
OUT_PATH = os.path.join(ROOT, "section_model.npz")
# Inverse regularisation, chosen leave-one-KT-out on the training KTs with
# the weight (audit 9abd): 4 against 1 and 10.
C = 4.0


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def usable(kt) -> bool:
    return not kt.get("holdout") and "(synthetic)" in str(kt.get("source", ""))


def training_kts():
    kts = []
    for folder in (GOLDEN_DIR, DEV_DIR):
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json") or name == "baseline.json":
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                kt = json.load(fh)
            if usable(kt):
                kts.append(kt)
    return kts


def label_units(kt, units):
    """(features, section) for each unit that states annotated facts of one
    section: it contains a fact's quote, or (four words or more) lies within
    one."""
    facts = [([norm(f["quote"])] + [norm(a) for a in f.get("alt", [])], f["in"]) for f in kt["facts"]]
    rows = []
    for text, features in units:
        t = norm(text)
        hits = [allowed for needles, allowed in facts
                if any(n and (n in t or (len(t.split()) >= 4 and t in n)) for n in needles)]
        if not hits:
            continue
        common = set(hits[0]).intersection(*map(set, hits[1:]))
        firsts = [h[0] for h in hits if h[0] in common]
        if firsts:
            rows.append((features, firsts[0]))
    return rows


def collect(kts, enc_cache: bool):
    """Each KT through the pipeline (no LLM, scratch state): the units the
    classifier scores, with [fact embedding, context embedding]."""
    import numpy as np

    state = tempfile.mkdtemp(prefix="section-model-")
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
    sys.path[:0] = [ROOT, os.path.join(ROOT, "evaluation")]
    if enc_cache:
        import enccache
        enccache.install(os.path.join(ROOT, "evaluation", ".cache", "enc_cache.sqlite"))

    import ai
    import llm_provider
    import pipeline
    from context_mapper import ContextMappingPipeline
    from devops_transcription import clean_transcript
    from kt_schema_loader import SCHEMA

    pipeline.get_llm_provider = ai.get_llm_provider = llm_provider.get_llm_provider = lambda: None
    pipeline.record_candidates = lambda *a, **k: None
    mapper = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    pipeline.MAPPER_PIPELINE = mapper
    classifier = mapper.classifier
    score = classifier._score_sentence_candidates
    captured = []

    def as_array(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def capturing(sentence, sent_embedding=None, context_embedding=None, **kwargs):
        if sent_embedding is not None:
            sent = as_array(sent_embedding)
            ctx = as_array(context_embedding) if context_embedding is not None else sent
            captured.append((sentence.text, np.concatenate([sent, ctx])))
        return score(sentence, sent_embedding=sent_embedding, context_embedding=context_embedding, **kwargs)

    classifier._score_sentence_candidates = capturing
    rows = []
    for kt in kts:
        captured.clear()
        job_id = f"section-model-{kt['id']}"
        try:
            pipeline.run_kt_pipeline(job_id, clean_transcript(kt["transcript"]))
        finally:
            pipeline.JOB_STORE.delete(job_id)
        rows.extend(label_units(kt, list(captured)))
    return rows


def build(enc_cache: bool = False) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    kts = training_kts()
    rows = collect(kts, enc_cache)
    X = np.stack([f for f, _ in rows])
    y = np.array([s for _, s in rows])
    model = LogisticRegression(C=C, max_iter=3000).fit(X, y)
    about = {"trained_on": [kt["id"] for kt in kts], "units": len(rows), "C": C,
             "features": "fact embedding + context embedding (BAAI/bge-large-en-v1.5)",
             "built_by": "scripts/build_section_model.py"}
    return {"classes": np.array([str(c) for c in model.classes_]),
            "coef": model.coef_.astype(np.float32),
            "intercept": model.intercept_.astype(np.float32),
            "about": np.array(json.dumps(about))}


def main(argv) -> int:
    import numpy as np

    data = build(enc_cache="--enc-cache" in argv)
    np.savez_compressed(OUT_PATH, **data)
    about = json.loads(str(data["about"]))
    print(OUT_PATH, f"{len(data['classes'])} sections, {about['units']} units from {len(about['trained_on'])} KTs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
