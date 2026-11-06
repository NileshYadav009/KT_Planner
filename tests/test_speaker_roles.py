"""Who is handing over, and what of a conversation counts as the KT
(speaker_roles.py). A three-person test (25 Oct): the receiver's guess
"So it stores invoices in MySQL, right?" was printed as a fact after the
giver said "No, it's PostgreSQL"; the receiver's "30 days" stood beside the
giver's "Actually, 14 days"; the giver's "three nodes" stood after he agreed
it was five; an unanswered "Isn't it PagerDuty now?" made PagerDuty the
alerting tool; the offsite small talk described the Production environment."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speaker_roles import (apply_roles, detect_roles, disputes, is_small_talk, roles_notice, settles,  # noqa: E402
                           turn_segments)


def _units(*turns):
    return [SimpleNamespace(text=f"{who}: {what}", speaker=who) for who, what in turns]


def _merge(a, b):
    return SimpleNamespace(text=f"{a.text} {b.text}", speaker=a.speaker)


ROLES = {"giver": "Arjun", "receivers": ["Meera"], "others": ["Tom"], "detected": True, "speakers": []}


def _apply(units, roles=ROLES):
    excluded, findings = [], []
    kept = apply_roles(units, roles, _merge, excluded, findings)
    return [u.text for u in kept], {e["text"]: e["reason"] for e in excluded}, findings


def test_a_pasted_conversation_keeps_who_said_each_turn():
    text = "Arjun: This is Ledgerline. It runs on EKS. Meera: On which cluster? Arjun: The main one."
    segments = turn_segments(text)
    assert [(s["speaker"], s["text"]) for s in segments] == [
        ("Arjun", "Arjun: This is Ledgerline. It runs on EKS."), ("Meera", "Meera: On which cluster?"),
        ("Arjun", "Arjun: The main one.")]
    assert all(s["untimed"] for s in segments)
    # One person talking, or a heading said once, is not a conversation.
    assert turn_segments("Backups: nightly to S3. The cluster runs on EKS.") is None


def test_the_giver_states_the_most_and_a_receiver_mostly_asks():
    units = _units(("Arjun", "This is Ledgerline, our invoicing service on Amazon EKS."),
                   ("Meera", "So it stores invoices in MySQL?"),
                   ("Arjun", "No, it's PostgreSQL on Amazon RDS, and the reconciliation job runs nightly."),
                   ("Meera", "Isn't it PagerDuty now?"),
                   ("Tom", "Also, Meera, you'll own the on-call rotation from next month."))
    roles = detect_roles(units)
    assert (roles["giver"], roles["receivers"], roles["others"], roles["detected"]) == ("Arjun", ["Meera"], ["Tom"],
                                                                                        True)
    # The user's choice wins.
    assert detect_roles(units, giver="Tom")["giver"] == "Tom" and not detect_roles(units, giver="Tom")["detected"]
    assert detect_roles(_units(("Arjun", "It runs on EKS."))) is None


def test_a_receivers_guess_the_giver_turns_down_is_the_giver_s_answer():
    kept, left_out, _ = _apply(_units(("Meera", "So it stores invoices in MySQL?"),
                                      ("Arjun", "No, it's PostgreSQL on Amazon RDS."),
                                      ("Meera", "And I heard backups are kept for 30 days."),
                                      ("Arjun", "Actually, backups are kept for 14 days, not 30.")))
    # The question stays with its answer, as said; the corrected statement goes.
    assert kept == ["Meera: So it stores invoices in MySQL? Arjun: No, it's PostgreSQL on Amazon RDS.",
                    "Arjun: Actually, backups are kept for 14 days, not 30."]
    assert left_out["Meera: And I heard backups are kept for 30 days."] == "said by Meera; corrected by Arjun"


def test_a_receivers_correction_the_giver_agrees_with_takes_back_what_he_said():
    kept, left_out, _ = _apply(_units(("Arjun", "The cluster runs three nodes."),
                                      ("Meera", "I think it's five now, they scaled it last month."),
                                      ("Arjun", "Oh right, yes, it's five nodes since the scaling.")))
    assert kept == ["Arjun: Oh right, yes, it's five nodes since the scaling."]
    assert left_out["Arjun: The cluster runs three nodes."] == "corrected by Meera, and Arjun agreed"
    # A bare agreement keeps the receiver's words with it.
    kept, left_out, _ = _apply(_units(("Arjun", "Backups are kept for seven days."),
                                      ("Meera", "Backups are kept for fourteen days since the audit."),
                                      ("Arjun", "Yes, that's right.")))
    assert kept == ["Meera: Backups are kept for fourteen days since the audit. Arjun: Yes, that's right."]
    assert "Arjun: Backups are kept for seven days." in left_out


def test_a_question_nobody_answers_flags_what_it_doubts():
    kept, left_out, findings = _apply(_units(("Arjun", "Alerts go to Slack."),
                                             ("Meera", "Isn't it PagerDuty now?"),
                                             ("Tom", "Let's not go down that path now, we're short on time."),
                                             ("Meera", "Do we have a staging cluster?")))
    assert kept == ["Arjun: Alerts go to Slack."]
    assert findings[0] == {"kind": "challenged", "text": "Isn't it PagerDuty now?", "by": "Meera",
                           "about": "Alerts go to Slack.", "giver": "Arjun"}
    # Only asking, with no answer: listed to confirm, nothing flagged.
    assert findings[1]["kind"] == "unanswered" and findings[1]["text"] == "Do we have a staging cluster?"
    assert left_out["Tom: Let's not go down that path now, we're short on time."].startswith("moving the conversation")


def test_what_the_receiver_says_alone_is_listed_to_confirm_not_shown():
    kept, left_out, findings = _apply(_units(("Meera", "The nightly job also exports to the warehouse."),
                                             ("Arjun", "Deploys go through ArgoCD.")))
    assert kept == ["Arjun: Deploys go through ArgoCD."]
    assert findings == [{"kind": "unconfirmed", "text": "The nightly job also exports to the warehouse.",
                         "by": "Meera", "giver": "Arjun"}]


def test_a_receivers_thanks_is_not_listed_to_confirm():
    """"Thanks for doing this." was on the follow-up list as something Marcus
    had not confirmed (dev d11, audit §9abj). It is left out, and that is all."""
    kept, left_out, findings = _apply(_units(("Meera", "Thanks for doing this."),
                                             ("Arjun", "Deploys go through ArgoCD.")))
    assert kept == ["Arjun: Deploys go through ArgoCD."]
    assert "Meera: Thanks for doing this." in left_out and findings == []


def test_small_talk_and_asking_to_repeat_are_left_out_but_business_words_are_not():
    kept, left_out, _ = _apply(_units(("Arjun", "Before we start, how was the offsite?"),
                                      ("Meera", "Good, the food was great."),
                                      ("Tom", "Ha, too much coffee though."),
                                      ("Meera", "Sorry, my connection dropped, can you repeat that?"),
                                      ("Arjun", "Never run the reconciliation job twice in one day.")))
    assert kept == ["Arjun: Never run the reconciliation job twice in one day."]
    assert len(left_out) == 4
    for text in ["Hi team, today I'm handing over QuickBite, our food delivery platform.",
                 "demand increasing significantly during weekends and dinner hours",
                 "we're talking like two million events a day, more on bank holidays obviously."]:
        assert not is_small_talk(text), text


def test_the_document_says_who_is_handing_over():
    assert roles_notice(ROLES) == ("Handed over by Arjun (detected: states the most). Facts come from Arjun and Tom; "
                                   "what Meera said counts only where Arjun confirmed it, and the rest is listed to "
                                   "confirm.")
    assert roles_notice(None) is None


# Two trusted speakers who disagree (the user's recording scenarios of 5 Nov,
# TC-01, TC-02, TC-04 and TC-14; matrix W01, W02, W04 and W14). A colleague's
# word counted like the giver's, so both statements stood as facts and a
# destructive instruction the colleague objected to was carded as the fix.

def _roles(giver, other):
    return {"giver": giver, "receivers": [], "others": [other], "detected": True, "speakers": []}


def test_a_rejection_the_first_speaker_answers_is_left_out():
    kept, left_out, findings = _apply(_units(
        ("Priya", "Our production database is Amazon RDS PostgreSQL."),
        ("Dev", "No, it is running on DynamoDB."),
        ("Priya", "We migrated from DynamoDB last year. PostgreSQL is the current database.")), _roles("Priya", "Dev"))
    assert kept == ["Priya: Our production database is Amazon RDS PostgreSQL.",
                    "Priya: We migrated from DynamoDB last year. PostgreSQL is the current database."]
    assert "answered it" in left_out["Dev: No, it is running on DynamoDB."]
    assert findings == []


def test_a_guess_naming_another_tool_is_left_out_when_the_giver_explains():
    kept, left_out, findings = _apply(_units(
        ("Arjun", "We use ArgoCD for deployments."),
        ("Meera", "I think Jenkins directly deploys the containers to production."),
        ("Arjun", "Jenkins builds and pushes images to ECR. ArgoCD deploys them to EKS.")), _roles("Arjun", "Meera"))
    assert not any("directly deploys" in text for text in kept)
    assert "Meera: I think Jenkins directly deploys the containers to production." in left_out
    assert findings == []


def test_an_objection_nobody_answers_keeps_the_two_together_and_flags_them():
    units = _units(("Ali", "If Kafka gets stuck, delete all production messages."),
                   ("Zoe", "No, that can cause data loss."))
    units.append(SimpleNamespace(text="Do not purge messages without approval.", speaker="Zoe"))
    kept, left_out, findings = _apply(units, _roles("Ali", "Zoe"))
    assert kept == ["Ali: If Kafka gets stuck, delete all production messages. Zoe: No, that can cause data loss.",
                    "Do not purge messages without approval."]
    assert findings == [{"kind": "disputed", "text": "No, that can cause data loss.", "by": "Zoe",
                         "about": "If Kafka gets stuck, delete all production messages.", "about_by": "Ali",
                         "giver": "Ali"}]
    assert left_out == {}


def test_two_values_and_a_doubt_stay_one_statement_to_confirm():
    kept, _, findings = _apply(_units(("Omar", "Our RTO is 2 hours."), ("Kim", "I believe it is 4 hours."),
                                      ("Omar", "We should confirm that with the infrastructure team.")),
                               _roles("Omar", "Kim"))
    assert kept == ["Omar: Our RTO is 2 hours. Kim: I believe it is 4 hours. "
                    "Omar: We should confirm that with the infrastructure team."]
    assert [f["kind"] for f in findings] == ["disputed"]


def test_agreeing_with_a_correction_takes_back_the_earlier_value():
    kept, left_out, findings = _apply(_units(("Priya", "The cluster has three nodes."),
                                             ("Dev", "No, it has five nodes since the scaling."),
                                             ("Priya", "Right, five.")), _roles("Priya", "Dev"))
    assert kept[0] == "Dev: No, it has five nodes since the scaling."
    assert "agreed" in left_out["Priya: The cluster has three nodes."]
    assert findings == []


@pytest.mark.parametrize("before, body", [
    ("We don't page for warnings.", "No, we never do."),                                   # agrees with a negative
    ("We use ArgoCD for deployments.", "I think the ArgoCD sync runs every three minutes."),  # adds to it
    ("Do we use Datadog?", "No, we use Grafana."),                                         # answers a question
    ("Backups run nightly.", "No problem, I'll note that."),                               # not a rejection
    ("The API runs on ECS.", "I think it's a good setup."),                                # no other value
])
def test_what_is_not_a_dispute(before, body):
    assert not disputes(body, before)


@pytest.mark.parametrize("reply, how", [
    ("Yes, that's right.", "agreed"),
    ("We should confirm that with the infrastructure team.", "doubted"),
    ("I'm not sure, let's check that with Priya.", "doubted"),
    ("No, it's two hours, I checked last week.", "answered"),
    ("Next, monitoring is in Grafana.", ""),
])
def test_how_the_first_speaker_settles_a_dispute(reply, how):
    assert settles(reply, "I believe it is 4 hours.", "Our RTO is 2 hours.") == how


def test_a_question_is_not_joined_to_a_sentence_that_goes_back_to_the_walkthrough():
    roles = {"giver": "Sam", "receivers": ["Rui"], "others": [], "detected": True, "speakers": []}
    kept, left_out, findings = _apply(_units(
        ("Rui", "Did you complete the deployment?"),
        ("Sam", "Coming back to the production architecture, requests enter through CloudFront.")), roles)
    assert kept == ["Sam: Coming back to the production architecture, requests enter through CloudFront."]
    assert [f["kind"] for f in findings] == ["unanswered"]


def test_sports_small_talk_is_left_out_but_a_game_that_was_down_is_not():
    assert is_small_talk("By the way, yesterday's match was amazing.")
    assert not is_small_talk("The game was down for two hours on Saturday.")
