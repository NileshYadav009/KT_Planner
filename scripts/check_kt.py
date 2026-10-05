#!/usr/bin/env python3
"""Print a coverage/progress summary for a job from the live API.

Usage: python check_kt.py <job_id> [base_url]
"""
import sys
import requests

job_id = sys.argv[1] if len(sys.argv) > 1 else '6dbfd569-0fe2-4027-bf81-bae917bc7d6a'
base_url = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:8000'

r = requests.get(f'{base_url}/status/{job_id}')
r.raise_for_status()
data = r.json()

print('=== JOB STATUS ===')
print(f'Status: {data.get("status")}')
print(f'Progress: {data.get("progress", 0)}%')
if data.get('error'):
    print(f'Error: {data["error"]}')
print(f'Missing Required Sections: {data.get("missing_required", [])}')
print()

print('=== SECTION COVERAGE ===')
coverage = data.get('coverage', {})
for section, info in coverage.items():
    status = info.get('status', 'unknown')
    conf = info.get('confidence', 0) or 0
    risk = info.get('risk', 0) or 0
    print(f'{section:28} {status:9} confidence={conf:.2f} risk={risk:.2f}')

populated = data.get('populated_fields', {})
field_count = sum(len(v) for v in populated.values())
auto_filled = sum(
    1 for sec in populated.values() for f in sec.values()
    if isinstance(f, dict) and f.get('source') not in ('unfilled', '')
)
print()
print(f'=== FIELDS: {auto_filled}/{field_count} auto-filled ===')
