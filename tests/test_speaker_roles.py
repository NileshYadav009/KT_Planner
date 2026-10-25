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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speaker_roles import apply_roles, detect_roles, is_small_talk, roles_notice, turn_segments  # noqa: E402


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
