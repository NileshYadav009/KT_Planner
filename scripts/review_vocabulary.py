#!/usr/bin/env python3
"""
Review DevOps vocabulary candidates the self-learning system has spotted in real
transcripts (see vocabulary_learning.py), and approve or reject them.

Candidates are kept per workspace in the job database (CONTINUUM_DB_PATH); an
approved one goes into glossary.json, which corrects every workspace's
transcripts. No server needs to be running.

Usage:
    python scripts/review_vocabulary.py [--workspace TENANT_ID]
        List pending candidates, sorted by how often they've been seen.

    python scripts/review_vocabulary.py --interactive
        Walk through pending candidates one at a time (y = approve as term,
        n = reject, s = skip for now).

    python scripts/review_vocabulary.py --approve "argos cd" --canonical "ArgoCD"
        Approve a candidate as a recognized term (default --as terms).

    python scripts/review_vocabulary.py --approve "cache clear" --as phrase_corrections --canonical "cache layers"
        Approve a candidate as a phrase-level mishearing correction.

    python scripts/review_vocabulary.py --reject "some false positive"
        Reject a candidate so it never resurfaces in its workspace.

A phrase pending in more than one workspace needs --workspace with --approve
or --reject.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocabulary_learning import LEGACY_TENANT, approve_candidate, load_candidates, reject_candidate


def _workspace(tenant_id: str) -> str:
    if tenant_id == LEGACY_TENANT:
        return "before workspaces were separated"
    from auth import tenant_name
    return f"workspace {tenant_name(tenant_id)} ({tenant_id})"


def list_pending(workspace=None):
    pending = [v for v in load_candidates(workspace) if v.get("status") == "pending"]
    pending.sort(key=lambda v: v.get("occurrence_count", 0), reverse=True)

    if not pending:
        print("No pending vocabulary candidates.")
        return pending

    print(f"{len(pending)} pending candidate(s), most-seen first:\n")
    for entry in pending:
        print(f"- \"{entry['phrase']}\"  (seen {entry.get('occurrence_count', 1)}x, signal={entry.get('signal')}, "
              f"{_workspace(entry['tenant_id'])})")
        if entry.get("near_miss_of"):
            print(f"    near miss of known term: \"{entry['near_miss_of']}\" (score={entry.get('score')})")
        if entry.get("trigger"):
            print(f"    followed trigger phrase: \"{entry['trigger']}\"")
        print()
    return pending


def interactive_review(workspace=None):
    pending = list_pending(workspace)
    if not pending:
        return
    print("Interactive review — for each candidate: [a]pprove / [r]eject / [s]kip / [q]uit\n")
    for entry in pending:
        phrase = entry["phrase"]
        answer = input(f"\"{phrase}\" (seen {entry.get('occurrence_count', 1)}x) — a/r/s/q? ").strip().lower()
        if answer == "q":
            break
        if answer == "a":
            approve_candidate(phrase, entry["tenant_id"])
            print(f"  Approved \"{phrase}\" as a recognized term.")
        elif answer == "r":
            reject_candidate(phrase, entry["tenant_id"])
            print(f"  Rejected \"{phrase}\".")
        else:
            print("  Skipped.")


def _only_workspace_of(phrase: str, workspace):
    """The workspace whose pending candidate `phrase` is; None (with the
    reason printed) when there is none or more than one."""
    if workspace is not None:
        return workspace
    found = sorted({c["tenant_id"] for c in load_candidates()
                    if c["phrase"].lower() == phrase.lower() and c.get("status") == "pending"})
    if len(found) == 1:
        return found[0]
    if not found:
        print(f"No pending candidate \"{phrase}\".")
    else:
        print(f"\"{phrase}\" is pending in {len(found)} workspaces; choose one with --workspace: "
              + ", ".join(repr(t) for t in found))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interactive", action="store_true", help="Walk through pending candidates one at a time")
    parser.add_argument("--workspace", metavar="TENANT_ID", help="Only this workspace's candidates")
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
        tenant_id = _only_workspace_of(args.approve, args.workspace)
        if tenant_id is None or not approve_candidate(args.approve, tenant_id, target=args.target,
                                                      canonical=args.canonical):
            sys.exit(1)
        print(f"Approved \"{args.approve}\" as {args.target}"
              + (f" -> \"{args.canonical}\"" if args.canonical else "."))
        return

    if args.reject:
        tenant_id = _only_workspace_of(args.reject, args.workspace)
        if tenant_id is None or not reject_candidate(args.reject, tenant_id):
            sys.exit(1)
        print(f"Rejected \"{args.reject}\".")
        return

    if args.interactive:
        interactive_review(args.workspace)
        return

    list_pending(args.workspace)


if __name__ == "__main__":
    main()
