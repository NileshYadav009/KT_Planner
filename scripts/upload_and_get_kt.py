#!/usr/bin/env python3
"""
KT Planner Upload & Retrieve Script

Automates the workflow:
1. Upload an audio file (MP3, WAV, etc.)
2. Wait for processing to complete
3. Retrieve structured KT output
4. Display formatted KT with:
   - Section assignments
   - Coverage analysis
   - Cross-references
   - Ordered flow
   - Review items (if any)

Usage:
    python upload_and_get_kt.py path/to/your/audio.mp3
    python upload_and_get_kt.py --url http://localhost:8000 path/to/audio.mp3
"""

import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


DEFAULT_API_URL = "http://localhost:8000"


def upload_audio(file_path: str, api_url: str = DEFAULT_API_URL) -> Optional[str]:
    """Upload audio file and return job_id."""
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            print(f"📤 Uploading {file_path}...")
            resp = requests.post(f"{api_url}/upload", files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            job_id = data.get('job_id')
            print(f"✅ Upload successful. Job ID: {job_id}")
            return job_id
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def poll_status(job_id: str, api_url: str = DEFAULT_API_URL, max_wait_seconds: int = 300) -> bool:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        try:
            resp = requests.get(f"{api_url}/status/{job_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            status = data.get('status')
            progress = data.get('progress', 0)
            
            print(f"⏳ Status: {status} ({progress}%)")
            
            if status == "completed":
                print("✅ Processing complete!")
                return True
            elif status == "failed":
                error = data.get('error', 'Unknown error')
                print(f"❌ Processing failed: {error}")
                return False
            
            time.sleep(3)  # Wait 3 seconds before polling again
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return False
    
    print(f"❌ Timeout after {max_wait_seconds} seconds")
    return False


def get_kt_output(job_id: str, api_url: str = DEFAULT_API_URL) -> Optional[Dict[str, Any]]:
    """Retrieve structured KT output."""
    try:
        print(f"📥 Retrieving structured KT...")
        resp = requests.get(f"{api_url}/kt/{job_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to retrieve KT: {e}")
        return None


def get_coverage(job_id: str, api_url: str = DEFAULT_API_URL) -> Optional[Dict[str, Any]]:
    """Retrieve coverage analysis."""
    try:
        resp = requests.get(f"{api_url}/coverage/{job_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠️  Could not retrieve coverage: {e}")
        return None


def format_kt_output(kt: Dict[str, Any], coverage: Optional[Dict[str, Any]] = None) -> str:
    """Format KT output for display."""
    lines = []
    lines.append("=" * 80)
    lines.append("📋 STRUCTURED KT OUTPUT")
    lines.append("=" * 80)
    lines.append("")
    
    # Job metadata
    lines.append("📌 JOB METADATA")
    lines.append("-" * 80)
    lines.append(f"Job ID: {kt.get('job_id')}")
    lines.append(f"Timestamp: {kt.get('timestamp')}")
    lines.append(f"Overall Coverage: {kt.get('overall_coverage_percent'):.1f}%")
    lines.append(f"Overall Risk Score: {kt.get('overall_risk_score'):.2f}")
    lines.append(f"Total Sentences: {kt.get('sentence_count')}")
    lines.append(f"Sections: {kt.get('section_count')}")
    lines.append("")
    
    # Coverage summary
    if coverage:
        lines.append("📊 SECTION COVERAGE")
        lines.append("-" * 80)
        cov_by_sec = coverage.get('coverage_by_section', {})
        for sec_id, sec_cov in cov_by_sec.items():
            status = sec_cov.get('status')
            count = sec_cov.get('sentence_count')
            conf = sec_cov.get('confidence', 0)
            icon = "✅" if status == "covered" else "⚠️ " if status == "weak" else "❌"
            lines.append(f"{icon} {sec_id}: {status} ({count} sentences, confidence={conf:.2f})")
        lines.append("")
    
    # Ordered sections (logical flow)
    if kt.get('ordered_section_content'):
        lines.append("📑 ORDERED SECTION CONTENT (Logical Flow)")
        lines.append("-" * 80)
        for section in kt.get('ordered_section_content', []):
            sec_id = section.get('section_id')
            title = section.get('section_title')
            count = section.get('sentence_count')
            lines.append(f"\n### {title.upper() if title else sec_id}")
            sentences = section.get('sentences', [])
            for i, sent in enumerate(sentences[:5], 1):  # Show first 5
                text = sent.get('text', '')[:100]
                lines.append(f"  {i}. {text}...")
            if count > 5:
                lines.append(f"  ... ({count - 5} more sentences)")
        lines.append("")
    
    # Flow coherence
    if kt.get('flow_coherence_score') is not None:
        lines.append("🔄 FLOW COHERENCE")
        lines.append("-" * 80)
        score = kt.get('flow_coherence_score', 0)
        lines.append(f"Flow Score: {score:.2f} (0.0=scrambled, 1.0=perfect order)")
        flow_issues = kt.get('flow_issues', [])
        if flow_issues:
            lines.append("Issues detected:")
            for issue in flow_issues[:5]:
                lines.append(f"  - {issue}")
        lines.append("")
    
    # Cross-references
    if kt.get('cross_references'):
        lines.append("🔗 CROSS-REFERENCES (Referential links)")
        lines.append("-" * 80)
        for ref in kt.get('cross_references', [])[:5]:
            from_sent = ref.get('from_sentence', '')[:60]
            to_sect = ref.get('to_section')
            note = ref.get('note')
            lines.append(f"  '{from_sent}...'")
            lines.append(f"    → {to_sect} ({note})")
        lines.append("")
    
    # Review required sentences
    if kt.get('review_required_sentences'):
        lines.append("🔍 REVIEW REQUIRED (Low confidence / ambiguous)")
        lines.append("-" * 80)
        for rev in kt.get('review_required_sentences', [])[:10]:
            text = rev.get('text', '')[:80]
            lines.append(f"  - {text}...")
        lines.append("")
    
    # Evidence & marked inferred
    if kt.get('evidence'):
        lines.append("🔎 EVIDENCE SUMMARY")
        lines.append("-" * 80)
        for ev in kt.get('evidence', [])[:5]:
            text = ev.get('text', '')[:60]
            evid = ev.get('evidence', [])
            if evid:
                lines.append(f"  Sentence: {text}...")
                lines.append(f"  Evidence: {', '.join(evid)}")
        lines.append("")
    
    # Missing required
    if kt.get('missing_required_sections'):
        lines.append("⛔ MISSING REQUIRED SECTIONS")
        lines.append("-" * 80)
        for miss in kt.get('missing_required_sections', []):
            lines.append(f"  - {miss}")
        lines.append("")
    
    # Unassigned sentences
    unassigned_count = kt.get('unassigned_count', 0)
    if unassigned_count > 0:
        lines.append(f"📌 UNASSIGNED SENTENCES: {unassigned_count}")
        lines.append("-" * 80)
        for sent in kt.get('unassigned_sentences', [])[:3]:
            text = sent.get('text', '')[:80]
            lines.append(f"  {text}...")
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def export_ordered_markdown(kt: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Export ordered KT as Markdown."""
    markdown = kt.get('ordered_markdown', '')
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"✅ Markdown exported to: {output_file}")
        except Exception as e:
            print(f"⚠️  Could not save Markdown: {e}")
    
    return markdown


def main():
    parser = argparse.ArgumentParser(
        description="Upload audio to KT Planner and retrieve structured output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_and_get_kt.py my_audio.mp3
  python upload_and_get_kt.py --url http://api.example.com:8000 audio.wav
  python upload_and_get_kt.py --markdown output.md audio.mp3
        """
    )
    parser.add_argument('audio_file', help='Path to audio file (MP3, WAV, etc.)')
    parser.add_argument('--url', default=DEFAULT_API_URL, help=f'KT Planner API URL (default: {DEFAULT_API_URL})')
    parser.add_argument('--markdown', help='Export ordered Markdown to file')
    parser.add_argument('--json', help='Export structured KT as JSON to file')
    parser.add_argument('--wait', type=int, default=300, help='Max wait time in seconds (default: 300)')
    
    args = parser.parse_args()
    
    print("🚀 KT Planner - Audio Upload & Retrieve Script")
    print(f"API URL: {args.url}")
    print("")
    
    # Step 1: Upload
    job_id = upload_audio(args.audio_file, args.url)
    if not job_id:
        sys.exit(1)
    
    print("")
    
    # Step 2: Wait for completion
    if not poll_status(job_id, args.url, args.wait):
        sys.exit(1)
    
    print("")
    
    # Step 3: Get KT output
    kt = get_kt_output(job_id, args.url)
    if not kt:
        sys.exit(1)
    
    coverage = get_coverage(job_id, args.url)
    
    print("")
    
    # Step 4: Display formatted output
    formatted = format_kt_output(kt, coverage)
    print(formatted)
    
    # Step 5: Optional exports
    if args.markdown:
        export_ordered_markdown(kt, args.markdown)
    
    if args.json:
        try:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(kt, f, indent=2)
            print(f"✅ JSON exported to: {args.json}")
        except Exception as e:
            print(f"⚠️  Could not save JSON: {e}")
    
    print("")
    print("✅ Done!")


if __name__ == '__main__':
    main()
