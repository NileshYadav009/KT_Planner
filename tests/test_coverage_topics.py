"""P0-8: a section counts as mentioned elsewhere only when another section
states its topic, not when a word of it appears in passing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from coverage_topics import assess_topics


def _elsewhere(section_id, sentence):
    return [label for label, state, _ in assess_topics(section_id, {}, [], [("other", sentence)])
            if state == "elsewhere"]


def test_a_word_in_passing_is_not_the_topic():
    assert _elsewhere("known_bad_days", "We get about three hundred requests a second at peak.") == []
    assert _elsewhere("day1_survival_checklist", "You'll be in the patient access team, which owns Tern.") == []
    assert _elsewhere("day1_survival_checklist", "Monday is the peak, and the first week of January.") == []
    assert _elsewhere("first_30_day_ownership", "The first week of January is busy.") == []


def test_a_deadline_at_the_end_of_the_month_is_not_a_month_end_window():
    # An onboarding milestone marked an undiscussed Operational Calendar as
    # Partial ("1 of 3 topics covered (1 under First 30-Day Ownership Plan)").
    for deadline in ["By the end of the month you should be doing the model releases on your own.",
                     "We plan to finish the migration before the end of the month.",
                     "By month-end you should run the releases yourself.",
                     "The ticket stays open until the end of the month."]:
        assert "Month-end windows" not in _elsewhere("known_bad_days", deadline), deadline
    for window in ["At the end of the month the billing run doubles the load.",
                   "The end of every month is when finance closes the books.",
                   "Month-end close is when finance needs the reports.",
                   "No releases in the last three days of the month."]:
        assert "Month-end windows" in _elsewhere("known_bad_days", window), window


def _discussed(section_id, sentence):
    return [label for label, state, _ in assess_topics(section_id, {}, [sentence], []) if state == "discussed"]


def test_an_escalation_path_is_discussed_without_the_word_escalate():
    # Shown under Ownership & Escalation with "Not covered in the KT session:
    # Escalation path" right below it (audit 9aao probe).
    for said in ["If something breaks at night, page the on-call through PagerDuty, and if they do not answer "
                 "in fifteen minutes, call Maria, the engineering manager.",
                 "If nobody acknowledges within ten minutes, the page goes to the second line.",
                 "Out of hours, call Raj Patel, the application manager.",
                 "Escalate to Dan first, then to Grace Lim."]:
        assert "Escalation path" in _discussed("ownership_escalation", said), said
    for said in ["The platform team owns the ECS cluster.", "Alerts page the on-call through PagerDuty."]:
        assert "Escalation path" not in _discussed("ownership_escalation", said), said


def test_the_topic_stated_under_another_section_still_counts():
    assert _elsewhere("known_bad_days", "Payday and the week before Christmas are the peak periods.") == [
        "High-traffic periods"]
    assert "Access and tools to request" in _elsewhere("day1_survival_checklist",
                                                      "On your first day get access to the AWS accounts.")
    assert _elsewhere("first_30_day_ownership", "In your first week, shadow the on-call.") == ["Week 1"]


def _discussed(section_id, sentence):
    return {label for label, state, _ in assess_topics(section_id, {}, [sentence], []) if state == "discussed"}


def test_topics_said_in_plain_words_are_discussed():
    """KT-taker audit (dev d05, matrix T01): "If it goes wrong, swap back.",
    "About twelve thousand a day.", "The printer queue backs up when a
    warehouse goes offline." and "Clear the queue from the admin page" were
    reported "not discussed"."""
    assert "Rollback procedure" in _discussed("deployment_and_rollback", "If it goes wrong, swap back.")
    assert "Volume" in _discussed("system_overview", "How many labels a day? About twelve thousand a day.")
    assert "Volume" in _discussed("system_overview", "Around 8000 bookings a day.")
    assert "Known failures" in _discussed("common_failures", "The printer queue backs up when a warehouse goes offline.")
    assert "Fixes or workarounds" in _discussed("common_failures", "Clear the queue from the admin page.")
    assert "Fixes or workarounds" in _discussed("common_failures", "Spread the sends over an hour.")
    # A frequency is not a volume.
    assert "Volume" not in _discussed("system_overview", "We restart the pods 3 times a day.")


def test_users_are_recognised_by_who_they_are():
    """KT-taker audit (dev d05): "the appointment booking system for our forty
    clinics" and "Patients book online" were "Users and reach: not
    discussed"."""
    assert "Users and reach" in _discussed("system_overview", "It's the appointment booking system for our forty clinics.")
    assert "Users and reach" in _discussed("system_overview", "Patients book online or through the call centre.")
    assert "Users and reach" in _discussed("system_overview", "Drivers and merchants use the app every day.")
