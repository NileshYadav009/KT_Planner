#!/usr/bin/env python3
"""
Review DevOps vocabulary candidates the self-learning system has spotted in real
transcripts (see vocabulary_learning.py), and approve or reject them.

Operates entirely on local files (glossary_candidates.json, glossary.json) — no
server needs to be running.

Usage:
    python scripts/review_vocabulary.py
        List pending candidates, sorted by how often they've been seen.

    python scripts/review_vocabulary.py --interactive
        Walk through pending candidates one at a time (y = approve as term,
        n = reject, s = skip for now).

    python scripts/review_vocabulary.py --approve "argos cd" --canonical "ArgoCD"
        Approve a candidate as a recognized term (default --as terms).

    python scripts/review_vocabulary.py --approve "cache clear" --as phrase_corrections --canonical "cache layers"
        Approve a candidate as a phrase-level mishearing correction.

    python scripts/review_vocabulary.py --reject "some false positive"
        Reject a candidate so it never resurfaces.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocabulary_learning import approve_candidate, load_candidates, reject_candidate


def list_pending():
    store = load_candidates()
    pending = [v for v in store.values() if v.get("status") == "pending"]
    pending.sort(key=lambda v: v.get("occurrence_count", 0), reverse=True)

    if not pending:
        print("No pending vocabulary candidates.")
        return pending

    print(f"{len(pending)} pending candidate(s), most-seen first:\n")
    for entry in pending:
        print(f"- \"{entry['phrase']}\"  (seen {entry.get('occurrence_count', 1)}x, signal={entry.get('signal')})")
        if entry.get("near_miss_of"):
            print(f"    near miss of known term: \"{entry['near_miss_of']}\" (score={entry.get('score')})")
        if entry.get("trigger"):
            print(f"    followed trigger phrase: \"{entry['trigger']}\"")
        for ctx in entry.get("example_contexts", [])[:2]:
            print(f"    e.g. \"{ctx}\"")
        print()
    return pending


def interactive_review():
    pending = list_pending()
    if not pending:
        return
    print("Interactive review — for each candidate: [a]pprove / [r]eject / [s]kip / [q]uit\n")
    for entry in pending:
        phrase = entry["phrase"]
        answer = input(f"\"{phrase}\" (seen {entry.get('occurrence_count', 1)}x) — a/r/s/q? ").strip().lower()
        if answer == "q":
            break
        if answer == "a":
            approve_candidate(phrase)
            print(f"  Approved \"{phrase}\" as a recognized term.")
        elif answer == "r":
            reject_candidate(phrase)
            print(f"  Rejected \"{phrase}\".")
        else:
            print("  Skipped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactive", action="store_true", help="Walk through pending candidates one at a time")
    parser.add_argument("--approve", metavar="PHRASE", help="Approve a pending candidate")
    parser.add_argument("--reject", metavar="PHRASE", help="Reject a pending candidate")
    parser.add_argument("--as", dest="target", default="terms", choices=["terms", "acronyms", "phrase_corrections"],
                         help="Where to add an approved candidate (default: terms)")
    parser.add_argument("--canonical", help="Canonical form (required for --as phrase_corrections; optional otherwise)")

    args = parser.parse_args()

    if args.approve:
        if args.target == "phrase_corrections" and not args.canonical:
            print("--as phrase_corrections requires --canonical \"corrected text\"")
            sys.exit(1)
        ok = approve_candidate(args.approve, target=args.target, canonical=args.canonical)
        if ok:
            print(f"Approved \"{args.approve}\" as {args.target}"
                  + (f" -> \"{args.canonical}\"" if args.canonical else "."))
        else:
            sys.exit(1)
        return

    if args.reject:
        ok = reject_candidate(args.reject)
        if ok:
            print(f"Rejected \"{args.reject}\".")
        else:
            sys.exit(1)
        return

    if args.interactive:
        interactive_review()
        return

    list_pending()


if __name__ == "__main__":
    main()
