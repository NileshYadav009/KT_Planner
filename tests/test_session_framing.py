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
