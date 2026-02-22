#!/usr/bin/env python3
import requests
import json

job_id = '6dbfd569-0fe2-4027-bf81-bae917bc7d6a'
r = requests.get(f'http://localhost:8000/kt/{job_id}')
data = r.json()

print('=== KT COVERAGE SUMMARY ===')
print(f'Overall Coverage: {data.get("overall_coverage_percent", 0)}%')
print(f'Overall Risk Score: {data.get("overall_risk_score", 0):.2f}')
print(f'Total Sentences: {data.get("sentence_count", 0)}')
print(f'Unassigned: {data.get("unassigned_count", 0)}')
print(f'Missing Required Sections: {data.get("missing_required_sections", [])}')
print()

print('=== SECTION COVERAGE ===')
coverage = data.get('coverage', {})
for section, info in coverage.items():
    count = info.get('count', 0)
    conf = info.get('avg_confidence', 0)
    print(f'{section:25} {count:3} sentences (avg confidence: {conf:.2f})')

print()
print('=== FLOW METRICS ===')
print(f'Flow Coherence Score: {data.get("flow_coherence_score", 0):.2f}')
print(f'Flow Issues: {data.get("flow_issues", [])}')

print()
print('=== SAMPLE ORDERED CONTENT ===')
ordered = data.get('ordered_section_content', {})
for i, (section, sentences) in enumerate(ordered.items()):
    if i < 3:  # Show first 3 sections
        print(f'\n{section}:')
        for j, sent in enumerate(sentences[:2], 1):
            preview = (sent[:80] + '...') if len(sent) > 80 else sent
            print(f'  {j}. {preview}')

print()
print(f'✅ Full KT generated with {len(coverage)} sections and {data.get("sentence_count", 0)} sentences')
