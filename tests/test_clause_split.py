"""One sentence, two facts for two sections (context_mapper.split_two_topic_sentence).

"Infrastructure is written in Bicep, and all secrets are stored in Key Vault"
went to Security whole: the e2e KT B's one remaining checklist failure (B07).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_mapper import AudioSegment, segment_sentences, split_two_topic_sentence  # noqa: E402

BICEP = "Infrastructure is written in Bicep, and all secrets are stored in Key Vault."


def test_two_halves_claimed_by_different_sections_are_split_verbatim():
    assert split_two_topic_sentence(BICEP) == ["Infrastructure is written in Bicep", "all secrets are stored in Key Vault."]


def test_halves_that_may_depend_on_each_other_stay_whole():
    for text in [
        # Only one half matches a routing rule: the symptom belongs with its fix.
        "The symptom is slow checkout, and the fix is to switch the table to on-demand capacity.",
        # Both halves are deployment.
        "Azure DevOps pipelines build the images and push them to ACR, and Flux syncs the releases to AKS.",
        # No rule on either half.
        "The points API is a Go service, and it is owned by the loyalty team.",
    ]:
        assert split_two_topic_sentence(text) == [text], text


def test_more_than_one_break_is_left_alone():
    text = "Infrastructure is written in Bicep, and all secrets are stored in Key Vault, and PagerDuty pages us."
    assert split_two_topic_sentence(text) == [text]


def test_each_half_becomes_its_own_statement():
    sentences = segment_sentences([AudioSegment(text=BICEP, start=0, end=4, avg_logprob=-0.3)])
    assert [s.text for s in sentences] == ["Infrastructure is written in Bicep.", "All secrets are stored in Key Vault."]
    assert all(s.start == 0 and s.end == 4 for s in sentences)


def test_the_split_half_keeps_its_own_section_after_the_continuation_rules():
    # Its raw text starts lowercase ("all secrets ..."), which the fragment
    # rule read as a cut-off clause and sent back to Deployment.
    from context_mapper import Classification, ClassifiedSentence
    from section_rules import apply_rule_overrides

    sentences = segment_sentences([AudioSegment(text=BICEP, start=0, end=4, avg_logprob=-0.3)])
    classified = [ClassifiedSentence(sentence=s, primary_classification=Classification(
        section_id="system_overview", section_title="System Overview", confidence=0.5, reason="test",
        similarity_score=0.5), secondary_classifications=[]) for s in sentences]
    apply_rule_overrides(classified, {})
    assert [c.primary_classification.section_id for c in classified] == ["architecture_reference", "security_controls"]
