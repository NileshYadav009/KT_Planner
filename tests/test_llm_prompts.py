"""Regression guard for llm/prompts.py's anti-hallucination wording.

Found via a real-world critique of 3 generated KT PDFs (AWS/Azure/GCP): the
common_failures structured-extraction prompt told the LLM to leave
"resolution" and "preventive_action" null unless the transcript explicitly
stated them, but said nothing of the kind about "cause" or "fix" (and
declared "fix" non-nullable in the JSON schema it hands the LLM). Across all
three unrelated transcripts, "fix" came back filled with suspiciously
uniform, generic troubleshooting phrasing ("Check X when Y fails") that
doesn't read like transcribed speech -- classic LLM-invented content filling
a field nothing told it to leave empty. A KT document's whole value
proposition is that every fact is traceable to something someone actually
said; a fabricated remediation step presented as if the outgoing owner said
it is worse than an honest "Not covered during KT".

This test doesn't call a real LLM (that would be nondeterministic and
require an API key) -- it just guards that the grounding instruction stays
in the prompt text, so a future edit can't silently drop it the way the
original prompt silently lacked it for "cause"/"fix".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.prompts import SECTION_STRUCTURED_PROMPTS


def test_common_failures_prompt_declares_cause_and_fix_nullable():
    prompt = SECTION_STRUCTURED_PROMPTS["common_failures"]
    assert '"cause": string|null' in prompt
    assert '"fix": string|null' in prompt


def test_common_failures_prompt_forbids_inventing_cause_or_fix():
    prompt = SECTION_STRUCTURED_PROMPTS["common_failures"].lower()
    # The grounding instruction must explicitly name all four of these --
    # not just resolution/preventive_action, which already had it before
    # this bug was found.
    for field_name in ("cause", "fix", "resolution", "preventive_action"):
        assert field_name in prompt
    assert "own general troubleshooting knowledge" in prompt or "leave the field null" in prompt
