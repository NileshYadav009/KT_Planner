"""Topic segments and the rules added with them (audit 9aaj).

A sentence that says little by itself ("Only six clinics have it.") is about
the topic the speaker last named: a question, a heading read aloud, or a
fronted "Costs, ...". section_rules.segment_topics() finds that topic; the
classifier adds context_mapper.TOPIC_SEGMENT_BOOST for it. No models load
here: the scoring test uses two-dimensional stand-in embeddings.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import context_mapper as cm  # noqa: E402
from section_rules import RECOVERY_EVIDENCE_RE, match_section_rules, segment_topics, topic_anchor  # noqa: E402


def test_a_question_names_the_topic_of_the_answer_that_follows():
    texts = [
        "Nadia: What is Fernhill exactly? Owen: It's the appointment booking system for our forty clinics.",
        "Nadia: What breaks most often? Owen: The VPN tunnel drops.",
        "Reset the tunnel from the Azure portal.",
        "Nadia: Anything else? Owen: Twilio sometimes rejects messages.",
        "Spread the sends over an hour.",
        "Nadia: Anything unfinished? Owen: The move to app notifications is half done.",
        "Only six clinics have it.",
    ]
    assert segment_topics(texts) == [
        "system_overview", "common_failures", "common_failures",
        # A question that names nothing keeps the topic.
        "common_failures", "common_failures",
        "open_responsibilities", "open_responsibilities",
    ]


def test_a_heading_read_aloud_names_the_topic_and_a_short_statement_does_not():
    assert topic_anchor("Known problems.") == ("heading", "common_failures")
    assert topic_anchor("Backups and DR.") == ("heading", "disaster_recovery")
    assert topic_anchor("Calendar.") == ("heading", "known_bad_days")
    # "Do not touch." has a verb; its prohibition rule names the topic.
    assert segment_topics(["Do not touch.", "The MQ queue manager names."])[1] == "danger_zones"
    assert topic_anchor("Open items.") == ("heading", "open_responsibilities")
    # A sentence with a subject or a verb is a statement, not a heading.
    assert topic_anchor("Deployments are pretty simple.") is None
    assert topic_anchor("The RTO is four hours.") is None


def test_a_fronted_topic_names_it_and_a_speaker_label_does_not():
    assert topic_anchor("Costs, Snowflake is most of it.") == ("fronted", "cost_optimization")
    assert topic_anchor("Dan: Security wise, the service account only has read access.") == ("fronted", "security_controls")
    assert topic_anchor("Going back to failures, the other common one is the warehouse queueing.")[1] == "common_failures"
    # Looks like a speaker label, but names a topic.
    assert topic_anchor("Unfinished: the migration of the last ten legacy SQL scripts into dbt.") == \
        ("fronted", "open_responsibilities")
    assert topic_anchor("Priya: So Sparrow is the product recommendations API.") is None


def test_the_opening_is_the_overview_and_a_decided_rule_moves_the_topic():
    texts = [
        "Brightline is what the finance and merchandising teams report from.",
        "If the morning load is late, the daily sales meeting has no numbers.",
        "Never change the DAG start dates.",
        "Airflow will try to backfill years of data.",
    ]
    assert segment_topics(texts) == ["system_overview", "system_overview", "danger_zones", "danger_zones"]


def test_unfinished_work_cost_and_onboarding_phrasings_have_rules():
    cases = {
        "There's also an unfinished A/B test framework migration.": "open_responsibilities",
        "Oh and one thing, there's a half finished migration from the old Node 14 runtime to Node 20.": "open_responsibilities",
        "One item remains open: the onboarding of a fourth bank, which has been scoped but not started.": "open_responsibilities",
        "TLS upgrade on the MQ channels is waiting on the mainframe team.": "open_responsibilities",
        "Kafka upgrade not started.": "open_responsibilities",
        "The Premium plan is the largest cost item.": "cost_optimization",
        "Snowflake warehouses auto-suspend after sixty seconds idle, which keeps the bill down.": "cost_optimization",
        "We pay for a three-year savings plan on the EKS nodes.": "cost_optimization",
        "The App Service plan is oversized.": "cost_optimization",
        "The classic failure is scanners losing their session after the Wi-Fi drops.": "common_failures",
        "Builds run in Cloud Build on every push to the release branch.": "deployment_and_rollback",
        "By week four you should be on the Airflow on-call.": "first_30_day_ownership",
        "By the end of the month you should lead a production release.": "first_30_day_ownership",
    }
    for text, section in cases.items():
        rule = match_section_rules(text)
        assert rule and rule.section_id == section, (text, rule)
    # Said of a port or a restart, not of a piece of work.
    for text in ["The security group port remains open to the VPN range.",
                 "If the service has not started after a deploy, restart it."]:
        rule = match_section_rules(text)
        assert rule is None or rule.section_id != "open_responsibilities", (text, rule)


def test_recovery_words_cover_the_ways_recovery_is_stated():
    for text in ["Veeam backs up the VMs every night", "The whole stack can be rebuilt in europe-west1 from Terraform",
                 "no more than fifteen minutes of lost preference changes", "the raw files stay in S3 for a year",
                 "copied to Azure Blob storage", "Failover drill twice a year.", "RTO thirty minutes, RPO five minutes."]:
        assert RECOVERY_EVIDENCE_RE.search(text), text
    for text in ["Spread the sends over an hour.", "Run the purge job, it is safe in business hours.",
                 "Bump the RU limit for an hour and it catches up."]:
        assert not RECOVERY_EVIDENCE_RE.search(text), text


class _Util:
    @staticmethod
    def cos_sim(a, b):
        a, b = np.atleast_2d(np.asarray(a, dtype=float)), np.atleast_2d(np.asarray(b, dtype=float))
        return (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :])


class _NoEntities:
    def get_context_entities(self, text):
        return {}


def _classifier(monkeypatch):
    monkeypatch.setattr(cm, "util", _Util)
    clf = cm.ContextClassifier.__new__(cm.ContextClassifier)
    clf.section_embeddings = {"common_failures": np.array([1.0, 0.0]), "disaster_recovery": np.array([0.0, 1.0])}
    clf.section_metadata = {sid: {"title": sid} for sid in clf.section_embeddings}
    clf._section_hint_patterns = {}
    clf._example_embeddings = None
    clf._example_sections = []
    clf.entity_extractor = _NoEntities()
    return clf


def _top(clf, text, embedding, topic=None):
    candidates, _, _ = clf._score_sentence_candidates(
        cm.Sentence(text=text, start=0.0, end=0.0), sent_embedding=np.array(embedding),
        context_embedding=np.array(embedding), topic_section=topic)
    return candidates[0].section_id


def test_the_named_topic_decides_a_sentence_that_says_little_and_not_one_that_says_more(monkeypatch):
    clf = _classifier(monkeypatch)
    monkeypatch.setattr(cm, "RECOVERY_WITHOUT_EVIDENCE_PENALTY", 0.0)
    near_tie = [0.70, 0.72]
    assert _top(clf, "Only six clinics have it.", near_tie) == "disaster_recovery"
    assert _top(clf, "Only six clinics have it.", near_tie, topic="common_failures") == "common_failures"
    clear = [0.30, 0.95]
    assert _top(clf, "Only six clinics have it.", clear, topic="common_failures") == "disaster_recovery"


def _placed(text, section, action="classify"):
    pick = cm.Classification(section_id=section, section_title=section, confidence=0.3, reason="", similarity_score=0.3)
    log = cm.ExplainabilityLog(action=action, timestamp="", sentence_id=0, section_id=section, reasoning="",
                               confidence=0.3)
    return cm.ClassifiedSentence(sentence=cm.Sentence(text=text, start=0.0, end=0.0), primary_classification=pick,
                                 secondary_classifications=[], explainability_log=log)


def _without_boost(section, score=0.3):
    return (cm.Classification(section_id=section, section_title=section, confidence=score, reason="",
                              similarity_score=score), score)


def test_the_topic_boost_never_leaves_a_discussed_section_empty():
    # Both onboarding sentences went to the topic (Day-1); without the boost
    # the second was First 30 Days, which would otherwise show as Missing.
    units = [_placed("Get the viewer roles.", "day1_survival_checklist"),
             _placed("Then sit in on one invoice run with me.", "day1_survival_checklist")]
    cm.restore_sections_emptied_by_topic(units, [None, _without_boost("first_30_day_ownership")])
    assert [u.primary_classification.section_id for u in units] == ["day1_survival_checklist", "first_30_day_ownership"]
    assert "topic boost undone" in units[1].explainability_log.reasoning


def test_the_guard_leaves_rules_and_sections_that_still_have_a_sentence_alone():
    # The section has a sentence already: the boost decided nothing away.
    units = [_placed("The tunnel drops.", "common_failures"), _placed("Reset the tunnel.", "common_failures"),
             _placed("Backups run nightly.", "disaster_recovery")]
    cm.restore_sections_emptied_by_topic(units, [None, _without_boost("disaster_recovery"), None])
    assert units[1].primary_classification.section_id == "common_failures"
    # A routing rule decided it: not the boost's doing.
    units = [_placed("Never purge the queue.", "danger_zones", action="rule_override"),
             _placed("Never touch the state file.", "danger_zones")]
    cm.restore_sections_emptied_by_topic(units, [_without_boost("common_failures"), None])
    assert units[0].primary_classification.section_id == "danger_zones"
    # Moving the only sentence of its section would just empty that one.
    units = [_placed("Only six clinics have it.", "open_responsibilities")]
    cm.restore_sections_emptied_by_topic(units, [_without_boost("disaster_recovery")])
    assert units[0].primary_classification.section_id == "open_responsibilities"


def test_the_five_best_are_kept_on_the_threshold_scale_without_a_reranker():
    """The ms-marco cross-encoder was removed (audit 9aam); scores keep the
    0.4 scale its blend gave them, which the thresholds were set on."""
    clf = cm.ContextClassifier.__new__(cm.ContextClassifier)
    clf.section_metadata = {}
    cands = [cm.Classification(section_id=f"s{i}", section_title=f"s{i}", confidence=1.0 - i / 10, reason="r",
                               similarity_score=0.5, topic_boost=0.04 if i == 2 else 0.0) for i in range(7)]
    kept = clf._rerank_candidates("text", cands)
    assert [c.section_id for c in kept] == ["s0", "s1", "s2", "s3", "s4"]
    assert kept[0].confidence == pytest.approx(cm.CLASSIFIER_SCORE_SCALE * 1.0)
    assert kept[2].topic_boost == pytest.approx(cm.CLASSIFIER_SCORE_SCALE * 0.04)
    assert not hasattr(clf, "cross_encoder")


def test_disaster_recovery_needs_a_recovery_word_to_win_a_near_tie(monkeypatch):
    clf = _classifier(monkeypatch)
    near_tie = [0.70, 0.72]
    assert _top(clf, "Spread the sends over an hour.", near_tie) == "common_failures"
    assert _top(clf, "Backups are copied to the second site every hour.", near_tie) == "disaster_recovery"


def test_a_numbered_heading_read_aloud_names_its_topic():
    """Dev d09, a handover document read aloud: "Section eleven, costs." named
    nothing, and the cost figure after it went to System Overview."""
    assert topic_anchor("Section eleven, costs.") == ("heading", "cost_optimization")
    assert topic_anchor("Section six, known issues.") == ("heading", "common_failures")
    assert topic_anchor("Section nine, restrictions.") == ("heading", "danger_zones")
    assert topic_anchor("Part 3: deployment.") == ("heading", "deployment_and_rollback")
    # A statement after the number is not a heading.
    assert topic_anchor("Section two, the web front end is an Azure App Service.") is None
