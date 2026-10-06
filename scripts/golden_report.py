#!/usr/bin/env python3
"""Score the golden KT suite (tests/goldens) and print where each fact landed.

    python scripts/golden_report.py                    # scores and misplaced facts
    python scripts/golden_report.py --update-baseline  # make the current scores the floor CI enforces

Runs the real pipeline without an LLM: deterministic, no quota used.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import golden_eval  # noqa: E402


def main(argv) -> int:
    import pipeline

    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    scores = golden_eval.run_all()

    def tally(group):
        total = sum(len(s["facts"]) for s in group)
        correct = sum(sum(f["status"] == "correct" for f in s["facts"]) for s in group)
        return correct, total

    correct, total = tally(scores)
    print(f"{'golden':32} {'accuracy':>8} {'wrong':>6} {'lost':>5} {'viol':>5} {'f.miss':>6} {'f.cov':>6} {'rep':>4}")
    for s in scores:
        print(f"{s['id']:32} {s['accuracy']:>8.0%} {len(s['wrong']):>6} {len(s['lost']):>5} {len(s['violations']):>5} "
              f"{len(s['false_missing']):>6} {len(s['false_covered']):>6} {len(s['repeats']):>4}")
    print(f"{'ALL':32} {correct / total:>8.0%}   ({correct} of {total} facts; target {golden_eval.TARGET_ACCURACY:.0%})")
    # The rules were tuned on the non-holdout goldens, so only the holdouts
    # say how well mapping does on a KT it has not seen.
    for label, group in (("tuning set", [s for s in scores if not s["holdout"]]),
                         ("holdout (unseen)", [s for s in scores if s["holdout"]])):
        if group:
            c, t = tally(group)
            print(f"  {label:30} {c / t:>8.0%}   ({c} of {t} facts)")
    for s in scores:
        problems = [f for f in s["facts"] if f["status"] != "correct" or f["violations"]]
        if problems or s["false_missing"] or s["false_covered"] or s["missing_conflicts"] or s["repeats"]:
            print(f"\n{s['id']}")
            for f in problems:
                print(f"  {f['id']} {f['status']:7} expected {f['expected']} found {f['found_in']}  « {f['quote']}")
            for key in ("false_missing", "false_covered", "missing_conflicts", "repeats"):
                if s[key]:
                    print(f"  {key}: {s[key]}")
    if "--update-baseline" in argv:
        with open(golden_eval.BASELINE_PATH, "w", encoding="utf-8") as fh:
            json.dump(golden_eval.baseline_from(scores), fh, indent=1)
        print(f"\nbaseline written: {golden_eval.BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
