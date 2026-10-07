#!/usr/bin/env python3
"""Stage latency, failures and queue state from the job database (P1-6).

    python scripts/ops_report.py              # every recorded KT
    python scripts/ops_report.py --hours 1    # the last hour

The same figures /metrics exports to Prometheus (deploy/grafana-dashboard.json),
for a deployment without one, or to check a test run. Set CONTINUUM_DB_PATH
to read another deployment's database.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_queue import TaskQueue  # noqa: E402
from job_store import JobStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, help="only KTs finished in the last N hours")
    args = parser.parse_args()

    store = JobStore()
    runs = store.runs(since=time.time() - args.hours * 3600 if args.hours else 0.0)
    queue = TaskQueue(store).stats()
    print(f"Database: {store.path}")
    print(f"Queue: {queue['queued']} queued, {queue['running']} running, oldest waiting "
          f"{queue['oldest_queued_seconds']:.0f} s; {queue['workers']} worker(s) alive, {queue['worker_slots']} slot(s)")
    if not runs:
        print("No finished KTs recorded.")
        return 0

    by_status = {}
    for run in runs:
        key = run["status"] + (f" ({run['failure']})" if run.get("failure") else "")
        by_status[key] = by_status.get(key, 0) + 1
    durations = [r["finished_at"] - r["started_at"] for r in runs]
    waits = [r["started_at"] - r["enqueued_at"] for r in runs if r.get("enqueued_at")]
    print(f"\nKTs: {len(runs)} — " + ", ".join(f"{n} {s}" for s, n in sorted(by_status.items())))
    print(f"Duration: median {statistics.median(durations):.0f} s, max {max(durations):.0f} s"
          + (f"; queue wait median {statistics.median(waits):.0f} s, max {max(waits):.0f} s" if waits else ""))

    stages, failures = {}, {}
    for run in runs:
        for stage, seconds in (run.get("stage_seconds") or {}).items():
            stages.setdefault(stage, []).append(seconds)
        for stage in run.get("failed_stages") or []:
            failures[stage] = failures.get(stage, 0) + 1
    print(f"\n{'Stage':<26}{'runs':>6}{'mean s':>9}{'max s':>9}{'failed':>8}")
    order = sorted(set(stages) | set(failures), key=lambda s: -statistics.mean(stages.get(s, [0])))
    for stage in order:
        values = stages.get(stage, [])
        print(f"{stage:<26}{len(values):>6}{statistics.mean(values) if values else 0:>9.1f}"
              f"{max(values, default=0):>9.1f}{failures.get(stage, 0):>8}")

    total = lambda key: sum(int(r.get(key) or 0) for r in runs)
    calls = total("llm_calls")
    fallbacks = total("llm_failed") + total("llm_rejected")
    print(f"\nLLM: {calls} calls, {total('input_tokens')} input / {total('output_tokens')} output tokens, "
          f"{fallbacks} fallback(s)" + (f" ({fallbacks / calls:.0%})" if calls else ""))
    missing = total("sections_missing")
    print(f"Sections reported missing: {missing}; of those mentioned under another section: "
          f"{total('missing_but_mentioned')}" + (f" ({total('missing_but_mentioned') / missing:.0%})" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
