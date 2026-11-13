"""P0-11: questions are not facts, and "we don't have that" is a gap, not
coverage. Before: questions became table values ("Staging -> Is there a
staging environment?"), answers lost their topic, and "there is no DR plan"
counted as disaster-recovery coverage."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dialogue import is_gap_statement, is_question, merge_question_answers


@pytest.mark.parametrize("text,gap", [
    ("Do we have a disaster recovery plan? No, honestly there is no DR plan and nobody has tested restores.", True),
    ("Is there a staging environment? Not really, we test directly in production behind a feature flag.", True),
    ("What is the RTO? We never agreed one with the business.", True),
    ("Who do I escalate to? I am not sure anymore, the old manager left.", True),
    ("Is the database backed up? I think so, but I have never checked.", True),
    ("There's no runbook for this.", True),
    # Operational rules and facts that merely contain negative words.
    ("Never delete items from the bookings table by hand.", False),
    ("We never deploy on Fridays.", False),
    ("There is no downtime during deploys because of blue-green.", False),
    ("Still open: the cost review for BigQuery storage has no owner yet.", False),
    ("Is the database backed up? Yes, nightly snapshots to S3.", False),
    ("If you are not sure about a change, involve the platform owner.", False),
    # A user's KT (23 Oct, matrix T03, C05): left off the follow-up list.
    ("We haven't documented the exact rollback revision yet, so that needs confirmation.", True),
    ("I think Redis is using 4 GB of memory, but I'm not completely sure.", True),
    ("The team hasn't tested the restore.", True),
    ("The CDN rules are still to be confirmed.", True),
    ("Production changes need confirmation from the CAB.", False),
    ("We haven't had an outage in a year.", False),
])
def test_stated_gaps_are_told_apart_from_rules_and_facts(text, gap):
    assert is_gap_statement(text) is gap


@pytest.mark.parametrize("text,unconfirmed", [
    # A claim with doubt on it: to confirm, not absent (a user's KT, 23 Oct).
    ("I think Redis is using 4 GB of memory, but I'm not completely sure.", True),
    ("The bucket holds about two million images, but I'm not sure.", True),
    # No claim: the thing is not known or not in place (tuning g08: Missing).
    ("Who do I escalate to? I am not sure anymore, the old manager left.", False),
    ("We haven't documented the exact rollback revision yet, so that needs confirmation.", False),
    ("There is no DR plan.", False),
])
def test_a_doubted_claim_is_told_apart_from_an_absence(text, unconfirmed):
    from dialogue import is_unconfirmed

    assert is_unconfirmed(text) is unconfirmed


def test_a_statement_before_a_hedged_change_is_not_a_stated_gap():
    """Adversarial matrix A18: "The backup retention is seven days. I think we
    changed it to thirty days, but I'm not sure." (one unit, kept whole)
    made Disaster Recovery Missing, "stated as not in place": only the
    hedged clause before "but" was read."""
    from dialogue import states_something

    assert states_something("The backup retention is seven days. I think we changed it to thirty days, "
                            "but I'm not sure.")
    assert states_something("Cosmos has continuous backup turned on, but we have never tried a restore.")
    for gap in ["Is it backed up? I think so, but I have never checked.",
                "The RTO and RPO have not been documented.",
                "What is the RTO? We never agreed one with the business."]:
        assert not states_something(gap), gap


def test_a_statement_before_a_gap_with_no_contrast_word_still_states_something():
    """Speech to text with no full stops (the noisy copies of the visible
    KTs): "firestore has a daily export ... bigquery keeps seven days of time
    travel honestly we haven't tested restoring" made Disaster Recovery
    Missing, "stated as not in place". So did a plain "and"."""
    from dialogue import states_something

    for said in ["Backups firestore has a daily export to Cloud Storage and BigQuery keeps seven days of time travel "
                 "honestly we haven't tested restoring the templates.",
                 "We take nightly backups to S3 and we have never tested a restore.",
                 "sql server full backup every night and log backups every fifteen minutes copied to azure blob storage "
                 "the plan is to rebuild the servers from the vm templates and that's never been tested.",
                 "what about backups aurora has automated backups with point in time recovery for thirty five days and "
                 "we take a snapshot before every schema migration we have never done a cross-region restore though."]:
        assert states_something(said), said
    for gap in ["So for the disaster recovery part I have to say we haven't tested anything.",
                "Honestly the whole recovery thing is something we have never documented.",
                "There is no DR plan and we have never tested a restore.",
                "Our recovery point objective is four hours, or at least that is what the previous document says.",
                "That is all we had time for today we will cover deployment and the rest next week.",
                "Do we have a disaster recovery plan no honestly there is no dr plan and nobody has tested restores."]:
        assert not states_something(gap), gap


def test_questions_are_merged_with_their_answers():
    texts = ["Is there a staging environment?", "Not really, we test in production.",
             "Who owns billing?", "What about alerts?", "Datadog pages us."]
    merged = merge_question_answers(texts, lambda t: t, lambda a, b: f"{a} {b}")
    assert merged == ["Is there a staging environment? Not really, we test in production.",
                      "Who owns billing?",                     # followed by another question: unanswered
                      "What about alerts? Datadog pages us."]
    assert is_question("Who owns billing?")


DIALOGUE = """Thanks for taking over Pinecart, the checkout service on Google Kubernetes Engine.
Do we have a disaster recovery plan? No, honestly there is no DR plan and nobody has tested restores.
Is there a staging environment? Not really, we test directly in production behind a feature flag.
What is the RTO? We never agreed one with the business.
Who do I escalate to? I am not sure anymore, the old manager left.
Is the database backed up? I think so, but I have never checked.
"""


@pytest.fixture(scope="module")
def dialogue_result():
    import ai
    import pipeline
    from context_mapper import ContextMappingPipeline
    from devops_transcription import clean_transcript
    from kt_schema_loader import SCHEMA

    saved = (pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE)
    pipeline.get_llm_provider = ai.get_llm_provider = lambda: None
    pipeline.record_candidates = lambda *a, **k: None
    pipeline.MAPPER_PIPELINE = ContextMappingPipeline(SCHEMA, llm_fallback_fn=None)
    try:
        yield pipeline.run_kt_pipeline("dialogue-0001", clean_transcript(DIALOGUE))
    finally:
        pipeline.get_llm_provider, ai.get_llm_provider, pipeline.record_candidates, pipeline.MAPPER_PIPELINE = saved


def _table_cells(result):
    for sec in result["knowledge_object"]["rendered_sections"]:
        for block in sec["blocks"]:
            if block["type"] in ("DecisionTable", "OwnershipTable", "TechnologyGrid") and sec["section_id"] != "kt_coverage":
                for row in block.get("rows") or []:
                    yield sec["section_id"], " | ".join(str(v) for v in row.values())


def test_no_question_or_stated_gap_appears_as_a_table_value(dialogue_result):
    bad = [(sid, cells) for sid, cells in _table_cells(dialogue_result) if "?" in cells or "not sure" in cells.lower()]
    assert bad == []


def test_negated_topics_are_reported_as_stated_gaps(dialogue_result):
    rows = {r["Domain"]: r for sec in dialogue_result["knowledge_object"]["sections"] if sec["id"] == "kt_coverage"
            for r in sec["_coverage_rows"]}
    dr = rows["Disaster Recovery"]
    assert dr["Coverage"] == "Missing" and "stated as not in place" in dr["Assessment"]
    assert rows["Environments"]["Coverage"] == "Missing"


def test_answers_keep_their_topic_and_the_intro_is_the_overview(dialogue_result):
    cov = dialogue_result["coverage"]
    texts = lambda sid: " ".join(s["text"] for s in cov[sid]["sentences"])
    assert "We never agreed one with the business" in texts("disaster_recovery")
    assert "Thanks for taking over Pinecart" in texts("system_overview")


def test_quick_reference_does_not_quote_a_gap_as_guidance(dialogue_result):
    rendered = json.dumps([s for s in dialogue_result["knowledge_object"]["rendered_sections"]
                           if s["section_id"] == "quick_reference"])
    assert "not sure anymore" not in rendered

def test_speaker_labels_are_names_that_open_turns_not_headings():
    """KT-taker audit (dev d05, matrix T01/T02): every turn of a pasted
    transcript carries its speaker ("Owen: Merge to main ..."), and the
    document printed them, with the questions, as facts."""
    from dialogue import speaker_labels, without_speaker_labels

    said = ("Nadia: How do I deploy? Owen: Merge to main. Nadia: Costs? Owen: The plan is oversized. "
            "Backups: nightly to S3. Fix: restart the pod. Sam Lee: Hello. Sam Lee: Quill is a Go service.")
    labels = speaker_labels(said)
    assert labels == {"Nadia", "Owen", "Sam Lee"}
    assert without_speaker_labels("Nadia: How do I deploy? Owen: Merge to main.", labels) == \
        "How do I deploy? Merge to main."
    # A heading said once keeps its word: it says what the value is.
    assert without_speaker_labels("Backups: nightly to S3.", labels) == "Backups: nightly to S3."
    # A name opening one turn of a conversation is a speaker too; in a
    # monologue a single "Name:" is not enough.
    assert "Ana Ruiz" in speaker_labels("Sam Lee: Hi. Sam Lee: Quill runs on Cloud Run. Ana Ruiz: What happens? "
                                        "Sam Lee: It scales.")
    assert speaker_labels("Backups: nightly to S3. The API runs on ECS.") == set()


def test_a_question_is_dropped_only_when_its_answer_says_it_all():
    from dialogue import without_redundant_question as drop

    assert drop("How do I deploy? Merge to main and GitHub Actions deploys to a staging slot.") == \
        "Merge to main and GitHub Actions deploys to a staging slot."
    assert drop("What should I never do? Never run a schema change during clinic hours.") == \
        "Never run a schema change during clinic hours."
    assert drop("What breaks most often? The printer queue backs up when a warehouse goes offline.") == \
        "The printer queue backs up when a warehouse goes offline."
    assert drop("What happens at month end? Volume triples at month end, so scale Cloud Run to twenty instances.") == \
        "Volume triples at month end, so scale Cloud Run to twenty instances."
    assert drop("Anything else? Twilio rejects messages when we hit the rate limit.") == \
        "Twilio rejects messages when we hit the rate limit."
    # The question names what the answer does not, the answer leans on it,
    # or the answer is not a statement: kept whole.
    for kept in ["Costs? The App Service plan is oversized.",
                 "What is the RTO? We never agreed one with the business.",
                 "Where does it run? It's hybrid.",
                 "Which environments do we have? Dev, staging and production.",
                 "How many bookings is that? Around eight thousand a day, and Mondays are the busiest.",
                 "Is it encrypted at rest? Yes, with KMS keys we manage."]:
        assert drop(kept) == kept, kept


@pytest.mark.parametrize("text,gap,unconfirmed", [
    # The user's six KTs (24 Oct): every stated unknown and doubt was missed,
    # so none reached the follow-up list.
    ("I think Redis has a 4 GB memory limit, but this needs verification.", True, True),
    ("We also use Redis for caching, but I am unsure whether its memory limit is 4 GB or 8 GB.", True, True),
    ("Our recovery point objective is four hours, or at least that is what the previous document says.", True, True),
    ("Do not describe the one-hour recovery estimate as a verified guarantee.", True, True),
    ("The exact rollback revision and weekend escalation contact are still missing from our runbook.", True, False),
    ("We do not have a confirmed rollback revision in the runbook, and the readiness probe timeout also needs "
     "verification.", True, False),
    ("Repository links and the emergency release contact still need to be added to the handover document.", True, False),
    ("Regional disaster recovery testing has not been completed this year.", True, False),
    ("The latest recovery test date is unavailable.", True, False),
    ("The exact dashboard URL and incident communication channel have not been included in the handover "
     "documentation.", True, False),
    # Rules, approvals, outages and facts.
    ("Every schema change needs verification by the DBA team.", False, False),
    ("According to the runbook, restart the consumer first.", False, False),
    ("The payments service is unavailable when the VPN drops.", False, False),
    ("The current backup schedule and retention policy are confirmed.", False, False),
])
def test_more_ways_of_stating_an_unknown_or_a_doubt(text, gap, unconfirmed):
    from dialogue import is_unconfirmed

    assert is_gap_statement(text) is gap
    assert is_unconfirmed(text) is unconfirmed


def test_a_sentence_that_only_doubts_the_one_before_is_kept_with_it():
    """The user's KTs (24 Oct): "That value needs confirmation ..." was filed
    as a deployment step, away from the Redis limit it doubts."""
    from context_mapper import AudioSegment, segment_sentences
    from dialogue import doubts_statement_before

    said = ("Deployments happen through GitHub Actions. We also use Redis for caching, but I am unsure whether its "
            "memory limit is 4 GB or 8 GB. That value needs confirmation before updating the operational "
            "documentation. Weekend escalation contacts are also missing from the runbook.")
    units = [s.text for s in segment_sentences([AudioSegment(text=said, start=0.0, end=9.0, avg_logprob=-0.2)])]
    assert ("We also use Redis for caching, but I am unsure whether its memory limit is 4 GB or 8 GB. "
            "That value needs confirmation before updating the operational documentation.") in units
    assert "Weekend escalation contacts are also missing from the runbook." in units
    # A doubt about its own claim, or a sentence with no doubt, stays apart.
    assert not doubts_statement_before("The Platform team manages Kubernetes infrastructure.",
                                       "I think Redis has a 4 GB memory limit, but this needs verification.")
    assert not doubts_statement_before("The API runs on EKS.", "Deployments happen through GitHub Actions.")
    # A sentence with a gap of its own (matrix C05): its "that" is itself.
    assert not doubts_statement_before("I think the bucket holds about two million images, but I'm not completely "
                                       "sure.", "We haven't documented the CDN cache rules yet, so that needs "
                                       "confirmation.")


def test_a_doubted_objective_does_not_fill_a_field():
    # "RPO: four hours" was printed as settled, in Disaster Recovery and on the
    # Quick Reference (the user's KTs, 24 Oct).
    from dialogue import is_field_candidate

    assert not is_field_candidate("Our recovery point objective is four hours, or at least that is what the "
                                  "previous document says. I need to confirm whether this target is still valid.")
    assert is_field_candidate("Our recovery point objective is four hours.")


def test_an_instruction_about_the_document_is_not_content():
    from dialogue import is_document_instruction

    assert is_document_instruction("Please ensure these uncertainties are clearly identified instead of assuming "
                                   "values.")
    assert not is_document_instruction("Make sure the backups are tested every month.")
    assert not is_document_instruction("Ensure the pods are healthy before switching traffic.")


def test_something_still_undocumented_is_a_stated_gap():
    # A user's KT (25 Oct): left off the follow-up list, and its "escalation
    # contacts" counted Ownership & Escalation as covered.
    assert is_gap_statement("The video-processing recovery procedure and formal escalation contacts are still "
                            "undocumented.")
    assert is_gap_statement("The failover runbook remains untested.")
    assert not is_gap_statement("Recovery testing is performed twice a year.")


def test_a_no_that_gives_the_right_value_is_a_correction_not_a_gap():
    # A three-person test (25 Oct): "So it stores invoices in MySQL? No, it's
    # PostgreSQL on Amazon RDS." made Architecture "stated as not in place".
    from dialogue import is_gap_statement

    assert not is_gap_statement("So it stores invoices in MySQL? No, it's PostgreSQL on Amazon RDS.")
    assert not is_gap_statement("So it runs on EKS? No, it's on ECS now.")
    assert is_gap_statement("Is there a DR plan? No.")
    assert is_gap_statement("Do we have a runbook? No, there isn't one.")
    assert is_gap_statement("Is it encrypted? No, not yet.")


@pytest.mark.parametrize("said, heading", [
    ("Section two, architecture.", True),
    ("Part four: release process.", True),
    ("Ines: Section three, ownership.", True),
    ("Part six: known issues and fixes.", True),          # "issues" is the heading's noun
    ("Section two, Terraform manages infrastructure.", False),
    # Not numbered: in speech to text without punctuation a question reads
    # the same ("Costs?" as "costs."), so these stay.
    ("Known problems.", False),
    ("costs.", False),
    # Statements, not headings.
    ("Section two, the API runs on EKS.", False),
    ("Secrets in CyberArk.", False),
    ("Terraform manages infrastructure.", False),
])
def test_a_heading_read_aloud_is_not_a_fact(said, heading):
    """X82: "Section two, architecture." was printed as a bullet of
    Architecture Knowledge. It still names the topic of what follows
    (section_rules.topic_anchor)."""
    from dialogue import is_heading_only

    assert is_heading_only(said) is heading


@pytest.mark.parametrize("said, remark", [
    # Recorded talks (27 Oct): filed under System Overview, a 21-minute KT
    # ran to 26 pages.
    ("Thoughts from your side?", True),
    ("It does the job.", True),
    ("This tool is very.", True),
    ("And for sure, we will do that.", True),
    ("Yes.", True),
    # A question kept with the sentence after it (matrix L05).
    ("Thoughts from your side? It does the job.", True),
    ("How many a day? About twelve thousand a day.", False),
    # Short facts in the answer keys stay: a figure, a known term, a place,
    # a negation, a name, an instruction.
    ("RTO is four hours.", False),
    ("It's all on premises.", False),
    ("We don't have that.", False),
    ("Owen: Dev, staging and production.", False),
    ("Terraform manages infrastructure.", False),
    ("Swap back.", False),
    ("Only six clinics have it.", False),
])
def test_a_remark_that_says_nothing_specific_is_not_a_fact(said, remark):
    from dialogue import says_nothing_specific

    assert says_nothing_specific(said) is remark


@pytest.mark.parametrize("unit,shown", [
    # The statement after a yes or no says what the question asked.
    ("Is there a cache? Yes, ElastiCache Redis holds the slot availability so the front end doesn't hammer the "
     "database.", "ElastiCache Redis holds the slot availability so the front end doesn't hammer the database."),
    ("Backups are nightly, I assume? No, Aurora takes continuous backups with point-in-time restore for fourteen days.",
     "Aurora takes continuous backups with point-in-time restore for fourteen days."),
    ("Is there anything I must never do? Never purge the SQS queue to fix a backlog.",
     "Never purge the SQS queue to fix a backlog."),
    ("Who owns it? The student systems team in IT owns Millbrook.", "The student systems team in IT owns Millbrook."),
    # X117 (matrix T04): the answer names a product of the kind asked about.
    ("Is there a queue? Yes, RabbitMQ holds the statement jobs until the renderer picks them up.",
     "RabbitMQ holds the statement jobs until the renderer picks them up."),
    ("Do we have a database for that? Yes, Aurora PostgreSQL keeps the bookings.",
     "Aurora PostgreSQL keeps the bookings."),
    # The question stays when the answer leans on it or says less than it asked.
    ("Is there a queue? Yes, it is on the roadmap.", None),
    ("Is there a queue? No, the API writes straight to the database.", None),
    ("Is there a staging environment? Yes, there's one for each team.", None),
    ("Is there a runbook for that? Yes, Priya wrote it last year.", None),
    ("Has anyone tested a restore? Not this year.", None),
    ("Bad days to deploy? Avoid Monday mornings and the first week of January, when everyone rebooks.", None),
    ("And my first week? Get access to the Azure subscription and the on-prem jump host.", None),
])
def test_an_answer_after_yes_or_no_stands_without_its_question(unit, shown):
    """A receiver's reading of a Q&A KT (dev d11, audit §9abj): the question
    was printed before about a dozen answers that say it all themselves."""
    from dialogue import display_text

    assert display_text(unit, set()) == (shown or unit)


def test_a_corrected_assumption_is_not_a_stated_gap():
    """"Backups are nightly, I assume? No, Aurora takes continuous backups ..."
    was Disaster Recovery's stated gap and on the follow-up list (dev d11)."""
    from dialogue import is_gap_statement

    assert not is_gap_statement("Backups are nightly, I assume? No, Aurora takes continuous backups with "
                                "point-in-time restore for fourteen days.")
    assert not is_gap_statement("Is the API public? No, only the VPN can reach the gateway.")
    # Asked whether something exists or was done, a no is a gap.
    assert is_gap_statement("Do we have alerts on the queue? No, we rely on the daily report.")
    assert is_gap_statement("Is there a DR plan? No.")
    assert is_gap_statement("Has anyone tested a restore? Not this year.")
    # A correction that is itself a gap stays one.
    assert is_gap_statement("Backups are nightly, I assume? No, honestly there is no backup at all.")


def test_an_action_on_the_sentence_before_stays_with_it():
    """"Not this year. That's something you should schedule.": alone, the
    second sentence said nothing specific and was left out with its action."""
    from context_mapper import AudioSegment, segment_sentences
    from dialogue import asks_to_act_on_before

    assert asks_to_act_on_before("Not this year.", "That's something you should schedule.")
    assert asks_to_act_on_before("The alert is noisy.", "You should really tune that.")
    assert not asks_to_act_on_before("The alert is noisy.", "You should see my screen now.")       # no refer-back
    assert not asks_to_act_on_before("Has anyone tested it?", "That's something you should schedule.")
    said = "Restores are run by the platform team. Not this year. That's something you should schedule."
    units = [s.text for s in segment_sentences([AudioSegment(text=said, start=0.0, end=9.0, avg_logprob=-0.2)])]
    assert "Not this year. That's something you should schedule." in units


# The user's recording scenarios of 5 Nov (matrix W04, W05, W17, W35).

@pytest.mark.parametrize("raw, cut", [
    ("The most important thing to remember before failing over is that you must never", True),
    ("And the last thing about the backups is that they", True),
    ("If the queue backs up, restart it", False),        # a whole sentence without its full stop
    ("That's what it is", False),
    ("Okay, thank you", False),
    ("You must never fail over without the DBA.", False),
    ("We never", False),                                 # too short to tell
])
def test_a_transcript_that_stops_mid_sentence(raw, cut):
    from dialogue import CUT_OFF_MARK, is_cut_off, is_unconfirmed, mark_cut_off

    assert is_cut_off(raw) is cut
    if cut:
        marked = mark_cut_off(raw + ".")
        assert marked.endswith(" " + CUT_OFF_MARK) and ". [" not in marked
        assert is_unconfirmed(marked) and is_gap_statement(marked)


def test_speech_marked_unclear_is_to_confirm_not_a_value():
    from dialogue import is_field_candidate, is_unconfirmed

    for text in ("The primary database is [unclear].", "The backup retention is [inaudible] days.",
                 "Failover takes (inaudible 00:12:31) minutes."):
        assert is_gap_statement(text) and is_unconfirmed(text), text
        assert not is_field_candidate(text), text
    assert not is_gap_statement("We use Redis for caching.")


def test_who_to_ask_is_a_doubt_but_an_instruction_to_check_is_not():
    from dialogue import is_hedged, voices_doubt

    assert voices_doubt("We should confirm that with the infrastructure team.")
    assert voices_doubt("Let's check that with Priya.")
    assert not voices_doubt("You should check that the pods are healthy before you scale.")
    assert is_hedged("I think the backup happens every six hours.")
    assert not is_hedged("The RTO should be under four hours.")       # a target, not a guess


def test_a_question_is_not_joined_to_a_return_to_the_walkthrough():
    from dialogue import returns_to_topic

    assert returns_to_topic("Coming back to the production architecture, requests enter through CloudFront.")
    assert returns_to_topic("Sam: Anyway, the ALB forwards requests to ECS.")
    assert not returns_to_topic("Back up the database before the migration.")
    merged = merge_question_answers(["Did you complete the deployment?",
                                     "Coming back to the architecture, requests reach the ALB."],
                                    lambda s: s, lambda a, b: f"{a} {b}")
    assert merged == ["Did you complete the deployment?", "Coming back to the architecture, requests reach the ALB."]


def test_only_a_speaker_s_own_sentences_are_joined_before_roles_are_read():
    """"Kim: I believe it is 4 hours. Omar: We should confirm that ..." was
    one sentence with no speaker, so the conversation was not seen."""
    from context_mapper import AudioSegment, segment_sentences

    said = ("Omar: This is the ledger service. Omar: Our RTO is 2 hours. Kim: I believe it is 4 hours. "
            "Omar: We should confirm that with the infrastructure team.")
    units = [s.text for s in segment_sentences([AudioSegment(text=said, start=0.0, end=9.0, avg_logprob=-0.2)])]
    assert not any("4 hours" in u and "confirm" in u and u.startswith("Kim") for u in units)
    alone = "The RTO is 2 hours. That value needs confirmation."
    units = [s.text for s in segment_sentences([AudioSegment(text=alone, start=0.0, end=9.0, avg_logprob=-0.2)])]
    assert units == [alone]


def test_a_value_said_to_need_confirmation_is_a_stated_gap():
    from dialogue import is_gap_statement, stated_before_gap
    # The client's e-commerce KT (13 Nov): no follow-up line asked for RTO and RPO.
    said = "Disaster recovery testing happens twice annually, but the current RTO and RPO need confirmation."
    assert is_gap_statement(said)
    assert stated_before_gap(said) == "Disaster recovery testing happens twice annually"
    # An approval rule is not a gap.
    assert not is_gap_statement("Production changes need confirmation from the CAB.")
    assert not is_gap_statement("Every schema change needs verification by the DBA team.")
