"""Tests for section_rules.py's deterministic classification patterns."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from section_rules import match_section_rules


def test_explicit_danger_zones_phrase_is_classified_as_danger_zones():
    # Regression test for a real bug found on a live Azure Banking
    # transcript: a sentence that explicitly announces itself as naming
    # the danger zones ("The main danger zones are SQL service tier
    # changes, Service bus retention and Dead-letter configuration, And
    # manual Kubernetes changes outside GitOps.") matched none of the
    # section's existing patterns (which only covered "dangerous area",
    # "do not touch", "terraform state files manually", "autoscaler
    # configuration") because none of them recognize the literal phrase
    # "danger zone(s)" -- the section's own name. The sentence fell
    # through to a different section, silently dropping 2 of its 3 named
    # items from the rendered Danger Zones section.
    text = (
        "The main danger zones are SQL service tier changes, Service bus "
        "retention and Dead-letter configuration, And manual Kubernetes "
        "changes outside GitOps."
    )
    match = match_section_rules(text)
    assert match is not None
    assert match.section_id == "danger_zones"


def test_sensitive_area_phrase_is_classified_as_danger_zones():
    text = (
        "The production Kubernetes cluster autoscaler is another sensitive "
        "area, because incorrect changes can affect the platform."
    )
    match = match_section_rules(text)
    assert match is not None
    assert match.section_id == "danger_zones"


def test_a_disk_filling_up_is_a_failure_and_restoring_a_previous_build_is_a_rollback():
    from section_rules import match_section_rules

    failure = match_section_rules("The other one we see is Oracle archive logs filling the disk on Monday mornings.")
    assert failure and failure.section_id == "common_failures"
    rollback = match_section_rules("If a new job version misbehaves, we restore the previous image from its last savepoint.")
    assert rollback and rollback.section_id == "deployment_and_rollback"
    backup = match_section_rules("We restore the database from the nightly backup in the other region.")
    assert not backup or backup.section_id != "deployment_and_rollback"


def test_a_short_question_routes_the_answer_merged_with_it():
    from section_rules import match_section_rules

    cases = {
        "Security? All traffic is TLS, partners authenticate with API keys, and the services run with task roles.":
            "security_controls",
        "Chloe: What goes wrong? Daniel: The integration times out now and then and bookings fail.": "common_failures",
        "Sam: Anything unfinished? Jordan: The upgrade has not been started and has no owner.": "open_responsibilities",
        "Backups? Cosmos has continuous backup with point in time restore.": "disaster_recovery",
    }
    for text, section in cases.items():
        rule = match_section_rules(text)
        assert rule and rule.section_id == section and rule.reason.startswith("question:"), text
    assert match_section_rules("Security?") is None                    # a bare question is not coverage
    assert match_section_rules("Why? Because finance closes the books.") is None


def test_general_phrasings_from_the_fourth_blind_round():
    from section_rules import match_section_rules

    cases = {
        "After the safety officer signs off, the same pipeline promotes the build to production.": "deployment_and_rollback",
        "And don't change the ALB idle timeout, the partners' clients depend on it.": "danger_zones",
        "Patient data never leaves the UK regions.": "security_controls",
        "The SMS reminders go through an old function that nobody really owns.": "open_responsibilities",
        "It needs moving onto the notification service.": "open_responsibilities",
        "We moved low-priority messages to email, which cut the SMS bill by a third.": "cost_optimization",
        "To get started, request the operators group in Google Cloud.": "day1_survival_checklist",
        "The incident you will meet most often is the provider throttling us.": "common_failures",
        "Usually it recovers by itself, but if it lasts more than ten minutes raise a ticket.": "common_failures",
        "The mobile team and two partners depend on it.": "system_overview",
    }
    for text, section in cases.items():
        rule = match_section_rules(text)
        assert rule and rule.section_id == section, (text, rule)
