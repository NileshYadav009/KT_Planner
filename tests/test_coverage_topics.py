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


def test_the_topic_stated_under_another_section_still_counts():
    assert _elsewhere("known_bad_days", "Payday and the week before Christmas are the peak periods.") == [
        "High-traffic periods"]
    assert "Access and tools to request" in _elsewhere("day1_survival_checklist",
                                                      "On your first day get access to the AWS accounts.")
    assert _elsewhere("first_30_day_ownership", "In your first week, shadow the on-call.") == ["Week 1"]
