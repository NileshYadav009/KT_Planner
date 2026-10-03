"""rapidfuzz in place of textdistance, and json-repair for LLM JSON. Both must
leave valid input handled exactly as before."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai import _extract_json_response
from devops_transcription import jaro_winkler_similarity, levenshtein_similarity, clean_transcript


@pytest.mark.parametrize("a,b", [
    ("rabbit mq", "rabbitmq"), ("pager duty", "pagerduty"), ("terraform", "terafrom"),
    ("cold storage", "cloud storage"), ("code is", "code build"), ("", "x"), ("argo", "argo"),
])
def test_similarity_scores_match_textdistance(a, b):
    textdistance = pytest.importorskip("textdistance")
    assert jaro_winkler_similarity(a, b) == pytest.approx(textdistance.jaro_winkler.normalized_similarity(a, b), abs=1e-12)
    assert levenshtein_similarity(a, b) == pytest.approx(textdistance.levenshtein.normalized_similarity(a, b), abs=1e-12)


def test_fuzzy_corrections_still_apply():
    assert "pagerduty" in clean_transcript("We get paged through paegr duty at night.").lower()


EXPECTED = {"application_ownership": "Integration team", "escalation_chain": ["On-call engineer", "Priya Nair"]}


@pytest.mark.parametrize("text", [
    json.dumps(EXPECTED),
    "```json\n" + json.dumps(EXPECTED) + "\n```",
    "Here is the JSON:\n" + json.dumps(EXPECTED),
    # The model adds a remark after the JSON: the old parser returned the
    # nested escalation_chain list and the section's data was dropped.
    json.dumps(EXPECTED) + "\nNote: the chain was stated explicitly.",
    json.dumps(EXPECTED) + "\n" + json.dumps(EXPECTED),
    # Syntax slips repaired by json-repair.
    '{"application_ownership": "Integration team", "escalation_chain": ["On-call engineer", "Priya Nair",],}',
    "{'application_ownership': 'Integration team', 'escalation_chain': ['On-call engineer', 'Priya Nair']}",
    '{application_ownership: "Integration team", escalation_chain: ["On-call engineer", "Priya Nair"]}',
])
def test_complete_json_answers_are_recovered(text):
    pytest.importorskip("json_repair")
    assert _extract_json_response(text) == EXPECTED


@pytest.mark.parametrize("text", [
    '{"application_ownership": "Integration team", "escalation_chain": ["On-call eng',
    '{"application_ownership": "Integration team", "escalation_chain": ["On-call engineer"',
    "I could not find ownership details in the fragments.",
    "",
])
def test_cut_off_or_non_json_answers_are_not_accepted(text):
    assert not isinstance(_extract_json_response(text), dict)
