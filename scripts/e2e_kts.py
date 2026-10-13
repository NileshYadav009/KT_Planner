#!/usr/bin/env python3
"""Run the end-to-end KT suite (tests/e2e_kts) and write a report.

Four short synthetic handovers, each taken to a PDF and checked against its
checklist (tests/e2e_eval.py). The mode is printed on every line of the
report and never mixed:

  --llm off       in-process, no LLM: the pytest suite's mode, zero requests.
  --llm replay    in-process, LLM answers only from a recording (LLM_CACHE=
                  replay, --recording JSON). An unrecorded request is skipped
                  and counted, never sent. Not a live model test.
  --api URL       a running server, exactly as a client uses it: POST
                  /kt-from-transcript, GET /status/{id}, GET /export/pdf/{id},
                  with whatever LLM the server is configured for (its own
                  usage report says which model and how many requests). The
                  API key is read from --key-file and never printed.
  --export-recording CACHE.sqlite JSON [--since EPOCH]
                  write a server's LLM cache (responses only, no prompts or
                  keys of any account) to a JSON recording for --llm replay;
                  --since keeps only responses created or reused after then.

Examples:
  python scripts/e2e_kts.py --llm off --out e2e_out
  python scripts/e2e_kts.py --api http://127.0.0.1:8011 --key-file key.txt --out e2e_live
  python scripts/e2e_kts.py --llm replay --recording tests/fixtures/e2e_kts_llm_recording.json --out e2e_replay
"""
import argparse
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def _print_report(reports, out_dir):
    from e2e_eval import problems

    lines = ["# End-to-end KT suite", ""]
    for r in reports:
        llm = r["llm"]
        lines.append(f"## {r['id']} ({r['mode']})")
        lines.append(f"- title: {r['system_name']}; PDF opens: {r['pdf']['opens']}; pages: {r['pdf']['pages']} "
                     f"({r['pdf']['placeholder_sections']} 'not covered' sections) for {r['pdf']['words_in']} words")
        lines.append(f"- facts in the PDF: {r['retained_in_pdf']}/{r['facts']}; placed correctly: "
                     f"{r['placement_accuracy']:.0%}; stated connections drawn: {r['relations']['found']}/"
                     f"{r['relations']['stated']}")
        lines.append(f"- LLM: {llm.get('provider') or 'none'} {llm.get('model') or ''} cache={llm.get('cache_mode')} "
                     f"requests={llm.get('llm_calls') or 0} cache_hits={llm.get('cache_hits') or 0} "
                     f"skipped={llm.get('skipped') or 0} failed={llm.get('failed') or 0} "
                     f"tokens in/out={llm.get('input_tokens') or 0}/{llm.get('output_tokens') or 0}"
                     f"{' (estimated)' if llm.get('estimated_tokens') else ''}")
        for group, title in (("false_negatives", "Lost or missing"), ("false_positives", "Added or misplaced")):
            found = {k: v for k, v in r[group].items() if v}
            lines.append(f"- {title}: " + ("none" if not found else "; ".join(f"{k}: {v}" for k, v in found.items())))
        lines.append(f"- problems: {problems(r)}")
        lines.append("")
        print(f"{r['id']} [{r['mode']}] facts {r['retained_in_pdf']}/{r['facts']} placement "
              f"{r['placement_accuracy']:.0%} relations {r['relations']['found']}/{r['relations']['stated']} "
              f"problems {problems(r)} llm requests {llm.get('llm_calls') or 0}")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=1, ensure_ascii=False, default=str)


def run_off(out_dir):
    import pipeline
    from e2e_eval import run_all

    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    return run_all(out_dir=out_dir)


def run_replay(out_dir, recording):
    """In-process, LLM answers only from the recording; nothing is sent."""
    import pipeline
    from e2e_eval import run_replay as replay

    if pipeline.MAPPER_PIPELINE is None:
        pipeline.load_models()
    return replay(recording, out_dir=out_dir)


def run_api(out_dir, base, key_file):
    import requests

    from e2e_eval import evaluate, load_datasets

    with open(key_file, encoding="utf-8") as fh:
        headers = {"Authorization": f"Bearer {fh.read().strip()}"}
    for _ in range(240):
        try:
            if requests.get(f"{base}/readyz", timeout=5).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    reports = []
    for ds in load_datasets():
        r = requests.post(f"{base}/kt-from-transcript", json={"transcript": ds["transcript"]}, headers=headers,
                          timeout=60)
        r.raise_for_status()
        job_id = r.json()["job_id"]
        started, status = time.time(), {}
        while time.time() - started < 1800:
            status = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30).json()
            if status.get("status") in ("completed", "completed_with_warnings", "failed"):
                break
            time.sleep(5)
        pdf = requests.get(f"{base}/export/pdf/{job_id}", headers=headers, timeout=300)
        pdf.raise_for_status()
        with open(os.path.join(out_dir, ds["id"] + ".pdf"), "wb") as fh:
            fh.write(pdf.content)
        with open(os.path.join(out_dir, ds["id"] + ".status.json"), "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=1, ensure_ascii=False, default=str)
        usage = status.get("llm_usage") or {}
        mode = f"api ({usage.get('provider') or 'no'} LLM{' ' + usage['model'] if usage.get('model') else ''}, " \
               f"cache={usage.get('cache_mode')}, {round(time.time() - started)}s)"
        reports.append(evaluate(ds, status, pdf.content, mode=mode))
    return reports


def export_recording(cache_path, out_json, since=None, batch_calls="1"):
    """Responses only (no prompts). `since` (epoch seconds) keeps the rows
    created or reused from then on, i.e. those the current code asks for."""
    conn = sqlite3.connect(cache_path)
    rows = conn.execute("SELECT key, call_site, provider, model, response, input_tokens, output_tokens, tenant "
                        "FROM llm_responses WHERE MAX(created_at, COALESCE(last_hit_at, 0)) >= ? ORDER BY created_at",
                        (float(since or 0),)).fetchall()
    conn.close()
    responses = [dict(zip(("key", "call_site", "provider", "model", "response", "input_tokens", "output_tokens",
                           "tenant"), r)) for r in rows]
    models = sorted({r["model"] for r in responses})
    providers = sorted({r["provider"] for r in responses})
    with open(out_json, "w", encoding="utf-8") as fh:
        tenants = sorted({r["tenant"] or "" for r in responses})
        json.dump({"recorded_with": {"provider_setting": providers[0] if providers else "groq",
                                     "model": models[0] if models else "", "requests": len(responses),
                                     "tenant": tenants[0] if len(tenants) == 1 else None,
                                     "settings": {"LLM_BATCH_CALLS": batch_calls}},
                   "responses": responses}, fh, indent=1, ensure_ascii=False)
    print(f"{len(responses)} recorded responses -> {out_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=("off", "replay"), default="off")
    ap.add_argument("--api")
    ap.add_argument("--key-file")
    ap.add_argument("--recording")
    ap.add_argument("--batch-calls", default="1",
                    help="with --export-recording: the server's LLM_BATCH_CALLS (it changes the prompts)")
    ap.add_argument("--out", default="e2e_out")
    ap.add_argument("--export-recording", nargs=2, metavar=("CACHE_SQLITE", "JSON"))
    ap.add_argument("--since", type=float, help="with --export-recording: only responses used since this epoch time")
    args = ap.parse_args()
    if args.export_recording:
        export_recording(*args.export_recording, since=args.since, batch_calls=args.batch_calls)
        return
    os.makedirs(args.out, exist_ok=True)
    if args.api:
        reports = run_api(args.out, args.api.rstrip("/"), args.key_file)
    elif args.llm == "replay":
        reports = run_replay(args.out, args.recording)
    else:
        reports = run_off(args.out)
    _print_report(reports, args.out)


if __name__ == "__main__":
    main()
