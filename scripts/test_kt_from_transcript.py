#!/usr/bin/env python3
"""
Test a transcript straight through the KT pipeline — no audio, no UI.

Posts a transcript to POST /kt-from-transcript (skips Whisper entirely), polls
/status/{job_id} the same way scripts/upload_and_get_kt.py does for real audio
uploads, then prints a coverage summary. Useful for iterating on classification,
vocabulary corrections, and rendering without recording audio each time.

Usage:
    python scripts/test_kt_from_transcript.py --text "We use Kubernetes and..."
    python scripts/test_kt_from_transcript.py --file my_transcript.txt
    python scripts/test_kt_from_transcript.py --file my_transcript.txt --pdf out.pdf
    python scripts/test_kt_from_transcript.py --file t.txt --url http://localhost:8000
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

DEFAULT_API_URL = "http://localhost:8000"


def submit_transcript(transcript: str, api_url: str = DEFAULT_API_URL) -> Optional[str]:
    """POST the transcript and return job_id."""
    try:
        print("Submitting transcript...")
        resp = requests.post(f"{api_url}/kt-from-transcript", json={"transcript": transcript}, timeout=30)
        resp.raise_for_status()
        job_id = resp.json().get("job_id")
        print(f"Submitted. Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Submit failed: {e}")
        return None


def poll_status(job_id: str, api_url: str = DEFAULT_API_URL, max_wait_seconds: int = 120) -> Optional[Dict[str, Any]]:
    """Poll job status until completion, failure, or timeout. Returns the final job dict."""
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            resp = requests.get(f"{api_url}/status/{job_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            progress = data.get("progress", 0)
            print(f"Status: {status} ({progress}%)")

            if status == "completed":
                print("Processing complete.")
                return data
            if status == "failed":
                print(f"Processing failed: {data.get('error', 'Unknown error')}")
                return data

            time.sleep(1.5)
        except Exception as e:
            print(f"Status check failed: {e}")
            return None

    print(f"Timeout after {max_wait_seconds} seconds")
    return None


def print_coverage_summary(job: Dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    coverage = job.get("coverage", {})
    for section_id, info in coverage.items():
        status = info.get("status", "unknown")
        conf = info.get("confidence", 0) or 0
        risk = info.get("risk", 0) or 0
        print(f"{section_id:28} {status:9} confidence={conf:.2f} risk={risk:.2f}")

    missing = job.get("missing_required", [])
    if missing:
        print()
        print(f"Missing required sections: {missing}")

    populated = job.get("populated_fields", {})
    field_count = sum(len(v) for v in populated.values())
    auto_filled = sum(
        1 for sec in populated.values() for f in sec.values()
        if isinstance(f, dict) and f.get("source") not in ("unfilled", "")
    )
    print()
    print(f"Fields: {auto_filled}/{field_count} auto-filled")
    print("=" * 70)


def fetch_pdf(job_id: str, output_path: str, api_url: str = DEFAULT_API_URL) -> None:
    try:
        print(f"Fetching PDF export...")
        resp = requests.get(f"{api_url}/export/pdf/{job_id}", timeout=60)
        resp.raise_for_status()
        Path(output_path).write_bytes(resp.content)
        print(f"PDF saved to: {output_path}")
    except Exception as e:
        print(f"PDF export failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a transcript through the KT pipeline without recording audio or using the UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_kt_from_transcript.py --text "We use Kubernetes for..."
  python scripts/test_kt_from_transcript.py --file transcript.txt --pdf out.pdf
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Transcript text inline")
    group.add_argument("--file", help="Path to a text file containing the transcript")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"KT Planner API URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--wait", type=int, default=120, help="Max wait time in seconds (default: 120)")
    parser.add_argument("--pdf", help="Also fetch the rendered PDF and save it to this path")

    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {args.file}")
            sys.exit(1)
        transcript = path.read_text(encoding="utf-8")
    else:
        transcript = args.text

    if not transcript or not transcript.strip():
        print("Transcript is empty.")
        sys.exit(1)

    job_id = submit_transcript(transcript, args.url)
    if not job_id:
        sys.exit(1)

    job = poll_status(job_id, args.url, args.wait)
    if not job or job.get("status") != "completed":
        sys.exit(1)

    print_coverage_summary(job)

    if args.pdf:
        fetch_pdf(job_id, args.pdf, args.url)

    print()
    print("Done. Job ID:", job_id)


if __name__ == "__main__":
    main()
