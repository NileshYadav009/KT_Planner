#!/usr/bin/env python3
"""Load test for the job queue (P1-4): N concurrent submissions against a
running server, timing every API request until every KT reaches a final state.

    python scripts/load_test.py --url http://127.0.0.1:8000 --file clip.wav -n 10
    python scripts/load_test.py --url http://127.0.0.1:8000 --transcript kt.txt -n 10

Pass --api-key for a server with authentication on. All N submissions are
sent at once; then every job's /status and the /jobs list are polled once a
second. Prints p50/p95/max latency of the submissions and of the polling
requests, and each job's final state. Exits 1 if the polling p95 is over
--p95-ms, or a KT does not reach a final state within --timeout seconds.
"""
import argparse
import os
import statistics
import sys
import threading
import time

import httpx

FINAL = {"completed", "completed_with_warnings", "failed"}


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def summary(name, values):
    return (f"{name:<12} n={len(values):<5} p50={percentile(values, 50):7.1f} ms  "
            f"p95={percentile(values, 95):7.1f} ms  max={max(values, default=0):7.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="a recording to upload N times")
    source.add_argument("--transcript", help="a transcript file to submit N times")
    parser.add_argument("-n", type=int, default=10, help="concurrent submissions")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds for every KT to finish")
    parser.add_argument("--p95-ms", type=float, default=300, help="polling p95 limit")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    client = httpx.Client(base_url=args.url, headers=headers, timeout=600)
    submit_ms, poll_ms, job_ids, errors = [], [], [None] * args.n, []
    lock = threading.Lock()
    payload = open(args.file, "rb").read() if args.file else None
    text = open(args.transcript, encoding="utf-8").read() if args.transcript else None

    def submit(i):
        started = time.perf_counter()
        try:
            if payload is not None:
                resp = client.post("/upload", files={"file": (os.path.basename(args.file), payload)})
            else:
                resp = client.post("/kt-from-transcript", json={"transcript": text})
            elapsed = (time.perf_counter() - started) * 1000
            with lock:
                submit_ms.append(elapsed)
            if resp.status_code != 200:
                errors.append(f"submission {i}: HTTP {resp.status_code} {resp.text[:200]}")
            else:
                job_ids[i] = resp.json()["job_id"]
        except Exception as exc:
            errors.append(f"submission {i}: {exc}")

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(args.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(summary("submit", submit_ms))
    for err in errors:
        print("  " + err)

    def timed_get(path):
        started = time.perf_counter()
        resp = client.get(path)
        poll_ms.append((time.perf_counter() - started) * 1000)
        return resp

    pending = {j for j in job_ids if j}
    final, peak_queued, started = {}, 0, time.time()
    while pending and time.time() - started < args.timeout:
        queued = 0
        for job_id in list(pending):
            data = timed_get(f"/status/{job_id}").json()
            queued += data.get("status") == "queued"
            if data.get("status") in FINAL:
                final[job_id] = (data["status"], round(time.time() - started), (data.get("error") or "")[:90])
                pending.discard(job_id)
        timed_get("/jobs?limit=20")
        peak_queued = max(peak_queued, queued)
        time.sleep(1)

    print(summary("poll", poll_ms))
    print(f"peak queued: {peak_queued}; finished: {len(final)} of {len([j for j in job_ids if j])} "
          f"in {round(time.time() - started)} s")
    for job_id, (status, seconds, error) in sorted(final.items(), key=lambda kv: kv[1][1]):
        print(f"  {job_id[:8]}  {status:<24} after {seconds:>5} s  {error}")
    for job_id in pending:
        print(f"  {job_id[:8]}  NOT FINISHED")

    ok = not pending and not errors and percentile(poll_ms, 95) <= args.p95_ms
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
