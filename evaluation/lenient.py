"""Lenient fact matching, for LLM-mode runs where the polish rewords
sentences: a fact is present when one rendered sentence restates at least
RESTATE of its content words; correct when such a sentence is in an allowed
section. Strict matching (golden_eval) first; lenient only for what strict
did not find.

    from lenient import lenient_tally
    lenient_tally(golden, result)  -> {"correct": .., "wrong": .., "lost": ..}
"""
import re

import golden_eval

RESTATE = 0.75
MIN_WORDS = 3
_STOP = frozenset("""a an the and or but if then so of to in on at by for with from into is are was were be been
it its this that these those there we our you your they their i me my he she his her do does did have has had
can could will would should may might must not no yes than too very just also only all any some each every
more most other such what which who when where why how up out off over under about as one um uh okay like right
""".split())


def _stems(text):
    return {w[:5] for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _STOP}


def lenient_tally(golden, result):
    ko = result.get("knowledge_object") or {}
    texts = golden_eval.section_texts(ko)
    sentences = [(sid, _stems(s)) for sid, strings in texts.items() for t in strings
                 for s in re.split(r"(?<=[.!?])\s+|\n+", str(t)) if s.strip()]
    strict = golden_eval.score(golden, result)
    out = {"correct": 0, "wrong": 0, "lost": 0, "strict_correct": 0}
    for spec, fact in zip(golden["facts"], strict["facts"]):
        if fact["status"] == "correct":
            out["correct"] += 1
            out["strict_correct"] += 1
            continue
        found = set(fact["found_in"])
        for needle in [spec["quote"]] + list(spec.get("alt", [])):
            want = _stems(needle)
            if len(want) < MIN_WORDS:
                continue
            found |= {sid for sid, have in sentences if len(want & have) / len(want) >= RESTATE}
        if set(spec["in"]) & found:
            out["correct"] += 1
        elif found:
            out["wrong"] += 1
        else:
            out["lost"] += 1
    return out
