"""The KT session's own framing is not knowledge: "Hi everyone, today I will
be handing over the AWS e-commerce platform." was printed as System Overview
content. A greeting in front of a sentence is dropped; a sentence that only
announces what is handed over (the title already says it) is not printed; an
introduction that also says what the system is stays."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dialogue import is_handover_announcement, strip_greeting
from renderers.blocks.common import coverage_paragraphs


@pytest.mark.parametrize("text", [
    "Hi everyone, today I will be handing over the AWS e-commerce platform.",
    "Hi everyone, today I will be handing over the GCP data and machine learning platform.",
    "Okay, so this KT is for MedRelay.",
    "Welcome to the Ember KT.",
    "We will be covering the deployment process today.",
])
def test_an_announcement_of_what_is_handed_over_is_not_content(text):
    assert is_handover_announcement(text)


@pytest.mark.parametrize("text", [
    "Today I'm handing over TripWise, the booking backend behind our travel mobile app.",
    "This handover covers Ledgerline, the invoicing platform our finance customers use.",
    "This KT is for Orbitpay, our payout service on AWS Lambda with DynamoDB.",
    "Thanks for taking over Pinecart, the checkout service on Google Kubernetes Engine.",
    "I'll walk you through the payments platform that runs on EKS.",
    "Hi everyone, the platform processes 120,000 orders per day and runs on AKS.",
    "Datadog is our monitoring tool.",
])
def test_an_introduction_that_says_what_the_system_is_stays(text):
    assert not is_handover_announcement(text)


def test_a_greeting_in_front_of_a_fact_is_dropped():
    assert strip_greeting("Hi everyone, the platform runs on AKS.") == "The platform runs on AKS."
    assert strip_greeting("Hey, thanks for jumping on, so Harbor is our fleet telemetry thing.") == \
        "So Harbor is our fleet telemetry thing."
    assert strip_greeting("Hi everyone.") == "Hi everyone."               # only a greeting: dropped elsewhere
    assert strip_greeting("Highly available, multi-AZ.") == "Highly available, multi-AZ."


def test_section_paragraphs_carry_no_session_framing():
    section = {"coverage_content": [
        "Hi everyone, today I will be handing over the AWS e-commerce platform.",
        "Hi everyone, today I will be handing over the AWS e-commerce platform. It is business critical.",
        "This platform processes customer orders across web and mobile channels.",
    ]}
    assert coverage_paragraphs(section) == [
        "It is business critical.", "This platform processes customer orders across web and mobile channels."]


def test_the_announcement_is_not_reported_as_an_unmapped_finding():
    from knowledge.knowledge_builder import _is_session_pleasantry

    assert _is_session_pleasantry("Hi everyone, today I will be handing over the AWS e-commerce platform.")
    assert not _is_session_pleasantry("Hi everyone, the platform processes 120,000 orders per day.")


def test_an_introduction_that_states_a_fact_keeps_the_fact_only():
    from dialogue import without_framing

    assert without_framing("Hi team, today I'm handing over our AWS e-commerce platform, which processes around "
                           "40,000 orders daily.") == "Our AWS e-commerce platform processes around 40,000 orders daily."
    assert without_framing("Today I'm handing over TripWise, the booking backend behind our app.") == \
        "TripWise is the booking backend behind our app."
    assert without_framing("Um, so the database is PostgreSQL on RDS.") == "The database is PostgreSQL on RDS."
    assert without_framing("Right now the cluster runs three nodes.") == "Right now the cluster runs three nodes."
    assert without_framing("Well-known issue: the cache fills up.") == "Well-known issue: the cache fills up."


def test_a_field_filled_with_the_opening_sentence_keeps_only_what_it_says():
    from knowledge.knowledge_builder import _without_session_framing

    field = {"value": "Hi team, today I'm handing over our AWS e-commerce platform, which processes around "
                      "40,000 orders daily.", "source": "semantic", "confidence": 0.7}
    assert _without_session_framing(field)["value"] == "Our AWS e-commerce platform processes around 40,000 orders daily."
    announcement = {"value": "Hi everyone, today I will be handing over the AWS e-commerce platform.",
                    "source": "semantic", "confidence": 0.7}
    assert _without_session_framing(announcement)["source"] == "unfilled"
    assert _without_session_framing({"value": "Amazon EKS", "source": "pattern"})["value"] == "Amazon EKS"


def test_the_kt_is_named_without_the_speakers_possessive():
    from field_populator import _extract_by_pattern

    name = _extract_by_pattern({"id": "system_name", "type": "text"},
                               "Hi team, today I'm handing over our AWS e-commerce platform, which processes orders.")
    assert name and not name.lower().startswith("our ")


def test_typographic_apostrophes_from_a_pasted_transcript_do_not_hide_the_intro():
    from devops_transcription import clean_transcript
    from dialogue import without_framing

    cleaned = clean_transcript("Hi team, today I\u2019m handing over our AWS platform, which processes 40,000 orders "
                               "daily. Don\u2019t touch the Terraform state.")
    assert "\u2019" not in cleaned
    assert without_framing(cleaned.split(". ")[0] + ".") == "Our AWS platform processes 40,000 orders daily."


@pytest.mark.parametrize("said, kept", [
    # The system introduced as "a quick handover for X", or by name with an appositive.
    ("Right, so this is a quick handover for the orders API, which serves order history to the mobile app.",
     "The orders API serves order history to the mobile app."),
    ("Okay, so this is Streamline, our GCP data platform for clickstream analytics.",
     "Streamline is our GCP data platform for clickstream analytics."),
    # The LLM polish pass's own framing.
    ("This document serves as the handover for Paygate, the Azure payment platform that authorises card payments.",
     "Paygate is the Azure payment platform that authorises card payments."),
    # Not an introduction: left as said.
    ("This is important, the cache must be warm before a release.",
     "This is important, the cache must be warm before a release."),
])
def test_named_and_quick_handover_introductions_keep_only_the_fact(said, kept):
    from dialogue import without_framing

    assert without_framing(said) == kept


def test_a_closing_remark_is_not_content_but_a_closing_fact_is():
    from section_rules import _GREETING_ONLY_RE

    assert _GREETING_ONLY_RE.match("I think that's most of it.")
    assert _GREETING_ONLY_RE.match("Okay, that's all from me.")
    assert not _GREETING_ONLY_RE.match("That is it for the rollback steps.")
    assert not _GREETING_ONLY_RE.match("I think that's the main risk.")


@pytest.mark.parametrize("text, name", [
    ("Right, so this is a quick handover for the orders API, which serves order history to the mobile app.",
     "Orders API"),
    ("Okay, so this is Streamline, our GCP data platform for clickstream analytics.", "Streamline"),
])
def test_the_kt_title_comes_from_a_quick_or_named_introduction(text, name):
    from field_populator import _extract_by_pattern

    assert _extract_by_pattern({"id": "system_name", "label": "System name"}, text) == name


def test_a_value_with_no_framing_is_left_exactly_as_it_was():
    """A field value was capitalised by the framing cleanup although nothing
    was removed: 'thirty minutes (replaces four hours ...)' printed 'Thirty'."""
    from dialogue import without_framing, without_session_framing

    value = "thirty minutes (replaces four hours, stated in the session as the old value)"
    assert without_framing(value) == value
    assert without_session_framing(value) == value
    assert without_framing("so the cache is warm") == "The cache is warm"


def test_screen_tour_narration_is_not_a_fact():
    from context_mapper import is_kt_session_meta_commentary

    for text in ["Now let me show you the UI how it looks.", "I will just give you the tour like how this looks like.",
                 "This is just a small overview."]:
        assert is_kt_session_meta_commentary(text), text
    for text in ["Let me show you how the rollback works: run helm rollback payments 3.",
                 "We have an operation console UI as you can see in front of you."]:
        assert not is_kt_session_meta_commentary(text), text


@pytest.mark.parametrize("text", [
    # A user's KT (23 Oct) printed "Today I'll explain our order service." under System Overview.
    "Hi team, today I'll explain our order service.",
    "Today I'm going to introduce the claims portal.",
    "In this session I'll go over our payroll platform.",
    "Today I will give the KT for the order streaming platform.",
])
def test_more_ways_of_announcing_the_handover_are_not_content(text):
    assert is_handover_announcement(text)


def test_an_explanation_that_says_what_the_system_does_stays():
    assert not is_handover_announcement("Today I'll explain our order service, which takes orders from the web shop.")


@pytest.mark.parametrize("opening, name", [
    # Titled "Today I'll Explain Our Order" and "Today I'll Walk You Through Our Billing" (matrix T03, B10).
    ("Hi team, today I'll explain our order service. It runs on AWS EKS.", "Order Service"),
    ("Hi everyone, today I'll walk you through our billing platform.", "Billing Platform"),
    # Two words or a name the speaker capitalised: the descriptor word is not added.
    ("Today I'm handing over the appointment booking system for our clinics.", "Appointment Booking"),
    ("Today I'll explain the CorePay platform.", "CorePay"),
    ("The data comes from the Bank of the West service.", "Bank of the West"),
])
def test_the_kt_is_named_after_the_system_not_the_words_that_introduce_it(opening, name):
    from knowledge.knowledge_builder import _infer_system_name

    assert _infer_system_name({"system_overview": {"content": [opening]}}) == name
