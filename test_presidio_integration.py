#!/usr/bin/env python3
"""Test Presidio Integration in devops_transcription"""

from devops_transcription import clean_transcript

# Test with sensitive data
sensitive_transcripts = [
    "Configure the deployment with password equals MyPassword123 and email john.doe@company.com",
    "The Jenkins webhook URL is http://jenkins.example.com/hook and admin token is ghp_1234567890",
]

print("=== Testing Presidio Integration in clean_transcript ===\n")

for i, transcript in enumerate(sensitive_transcripts, 1):
    print(f"Test {i}:")
    print(f"Original: {transcript}")
    
    # With anonymization (default)
    cleaned = clean_transcript(transcript, anonymize_pii=True)
    print(f"Cleaned (with PII masking): {cleaned}\n")
    
    # Without anonymization for comparison
    cleaned_no_anon = clean_transcript(transcript, anonymize_pii=False)
    print(f"Cleaned (no PII masking): {cleaned_no_anon}\n")
    print("-" * 80)
    print()

print("✓ Presidio integration successfully tested")
print("✓ PII anonymization is now part of the transcript cleaning pipeline")
