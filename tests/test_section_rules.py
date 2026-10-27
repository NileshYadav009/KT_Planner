"""Tests for section_rules.py's deterministic classification patterns."""
import sys
import os

import pytest

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


def test_the_infrastructure_as_code_tool_is_an_architecture_fact():
    # "Infrastructure is written in Bicep" matched no rule, so it went
    # wherever the sentence it shared went (Security, with Key Vault).
    for text in ["Infrastructure is written in Bicep", "Everything is defined in Bicep.",
                 "All infrastructure is defined in Terraform.", "Terraform manages the infrastructure",
                 "We manage the infrastructure with Pulumi", "We use Terraform for all our infrastructure"]:
        match = match_section_rules(text)
        assert match and match.section_id == "architecture_reference", text
    # Only a sentence that is nothing else: inside a longer one the rule
    # decided the section for the other facts too.
    chain = "Terraform manages the infrastructure, GitHub Actions builds the images, and ArgoCD deploys them to GKE."
    assert match_section_rules(chain).section_id == "deployment_and_rollback"
    for text in ["The infrastructure is in Terraform, applied from Cloud Build. Application releases go through Cloud Deploy.",
                 "GKE is used for Kubernetes workloads. Terraform manages infrastructure."]:
        assert match_section_rules(text) is None, text
    # Its release trigger ("on every merge", audit 9aaj) is a deployment fact;
    # the infrastructure-as-code rule still does not decide it.
    chained = match_section_rules("Everything is in Terraform, and Cloud Build deploys to Cloud Run on every merge to main.")
    assert chained is None or chained.section_id != "architecture_reference"
    # Rebuilding from it is DR, and its state is a Danger Zone.
    assert match_section_rules("For disaster recovery, Terraform is used to recreate infrastructure.").section_id \
        == "disaster_recovery"
    assert match_section_rules("Never modify the Terraform state manually.").section_id == "danger_zones"
    rebuild = match_section_rules("The whole stack can be rebuilt in europe-west1 from Terraform.")
    assert rebuild is None or rebuild.section_id != "architecture_reference"


def test_an_unanswered_page_and_a_named_contact_are_escalation_and_a_region_count_is_architecture():
    """Audit 9aao adversarial probe: the escalation path was filed under
    Monitoring and Ownership & Escalation reported it as not discussed."""
    for text in ["If something breaks at night, page the on-call through PagerDuty, and if they do not answer "
                 "in fifteen minutes, call Maria, the engineering manager.",
                 "If nobody acknowledges within ten minutes, the page goes to the secondary.",
                 "Out of hours, call Raj Patel, the application manager.",
                 "If that fails, call Sam Achebe, he's the head of engineering."]:
        rule = match_section_rules(text)
        assert rule and rule.section_id == "ownership_escalation", (text, rule)
    # Alert routing stays Monitoring; a call centre is not a contact.
    for text in ["The burn-rate alert pages the on-call through PagerDuty.",
                 "Patients book online or through the call centre team."]:
        rule = match_section_rules(text)
        assert rule is None or rule.section_id != "ownership_escalation", (text, rule)
    for text in ["Right now we run in one region, us-east-1.", "The services are deployed in two AWS regions."]:
        rule = match_section_rules(text)
        assert rule and rule.section_id == "architecture_reference", (text, rule)
    for text in ["We keep a warm standby in a second AWS region.", "There's no replica in another region."]:
        rule = match_section_rules(text)
        assert rule is None or rule.section_id != "architecture_reference", (text, rule)


def test_planned_and_unstarted_work_is_an_open_responsibility():
    """Every annotated plan or unstarted piece of work is Open
    Responsibilities ("the scanner firmware upgrade is planned for next
    quarter"); the probe's region plan went to Disaster Recovery."""
    for text in ["Next quarter we plan to add a second region in eu-west-1, but nobody has started on it.",
                 "The scanner firmware upgrade is planned for next quarter.",
                 "Next year we will migrate the reports to Looker.",
                 "We want to move off Jenkins, nobody has started on it yet."]:
        rule = match_section_rules(text)
        assert rule and rule.section_id == "open_responsibilities", (text, rule)
    # An onboarding plan, and a recurring practice, are not open work.
    for text in ["By the end of next month you will own the releases.",
                 "Before Black Friday we plan to add more nodes."]:
        rule = match_section_rules(text)
        assert rule is None or rule.section_id != "open_responsibilities", (text, rule)

def test_the_steps_of_a_procedure_stay_with_its_first_step():
    """Adversarial matrix I02: "To roll back, first pause the sync." "Then
    revert the image tag ..." "After that, resume the sync ..." The last step
    went to Disaster Recovery, cutting the procedure in two."""
    from types import SimpleNamespace

    from context_mapper import Classification, Sentence
    from section_rules import apply_continuation_rules

    def classified(text, section_id):
        primary = Classification(section_id=section_id, section_title=section_id, confidence=0.6, reason="",
                                 similarity_score=0.6)
        return SimpleNamespace(sentence=Sentence(text=text, start=0.0, end=0.0), primary_classification=primary,
                               is_unassigned=False, explainability_log=None)

    steps = [classified("To roll back, first pause the ArgoCD sync.", "deployment_and_rollback"),
             classified("Then revert the image tag in the deploy repo.", "common_failures"),
             classified("After that, resume the sync and watch the error rate for ten minutes.", "disaster_recovery"),
             classified("Finally, tell the release channel.", "ownership_escalation")]
    apply_continuation_rules(steps, {})
    assert [s.primary_classification.section_id for s in steps] == ["deployment_and_rollback"] * 4
    # Not a step: after a statement, or a sequence word starting a new topic.
    others = [classified("Grafana shows queue depth.", "monitoring_observability"),
              classified("Then the backups run every night.", "disaster_recovery"),
              classified("Restart the consumer when lag grows.", "common_failures"),
              classified("Next, the on-call rota is in PagerDuty.", "ownership_escalation")]
    apply_continuation_rules(others, {})
    assert [s.primary_classification.section_id for s in others] == [
        "monitoring_observability", "disaster_recovery", "common_failures", "ownership_escalation"]

def test_a_sentence_about_when_that_happens_stays_with_what_happens():
    """Adversarial matrix B05: "Redis evicts sessions under memory pressure.
    That happens during peak traffic." The second went to another section."""
    from types import SimpleNamespace

    from context_mapper import Classification, Sentence
    from section_rules import apply_continuation_rules

    def classified(text, section_id):
        primary = Classification(section_id=section_id, section_title=section_id, confidence=0.6, reason="",
                                 similarity_score=0.6)
        return SimpleNamespace(sentence=Sentence(text=text, start=0.0, end=0.0), primary_classification=primary,
                               is_unassigned=False, explainability_log=None)

    pair = [classified("Redis evicts sessions under memory pressure.", "common_failures"),
            classified("That happens during peak traffic.", "known_bad_days"),
            classified("It usually occurs after a deployment.", "deployment_and_rollback")]
    apply_continuation_rules(pair, {})
    assert [s.primary_classification.section_id for s in pair] == ["common_failures"] * 3
    # A sentence with its own subject keeps its place.
    other = [classified("Redis evicts sessions under memory pressure.", "common_failures"),
             classified("Peak traffic happens on Fridays.", "known_bad_days")]
    apply_continuation_rules(other, {})
    assert other[1].primary_classification.section_id == "known_bad_days"

def test_a_step_already_with_its_procedure_is_not_taken_back_by_the_topic_restore():
    """Adversarial matrix I02, in the pipeline: the last step was already in
    Deployment (topic boost), so the step rule changed nothing, and
    restore_sections_emptied_by_topic then gave it to Disaster Recovery, its
    pick without the boost, because that section was empty."""
    from types import SimpleNamespace

    from context_mapper import Classification, ExplainabilityLog, Sentence, restore_sections_emptied_by_topic
    from section_rules import apply_continuation_rules

    def classified(text, section_id, confidence=0.6):
        primary = Classification(section_id=section_id, section_title=section_id, confidence=confidence, reason="",
                                 similarity_score=confidence)
        log = ExplainabilityLog(action="classify", timestamp="", sentence_id=0, section_id=section_id,
                                reasoning="", confidence=confidence)
        return SimpleNamespace(sentence=Sentence(text=text, start=0.0, end=0.0), primary_classification=primary,
                               secondary_classifications=[], is_unassigned=False, explainability_log=log)

    steps = [classified("To roll back, first pause the ArgoCD sync.", "deployment_and_rollback"),
             classified("Then revert the image tag in the deploy repo.", "deployment_and_rollback"),
             classified("After that, resume the sync and watch the error rate.", "deployment_and_rollback", 0.33)]
    unboosted = [None, None, (Classification(section_id="disaster_recovery", section_title="disaster_recovery",
                                             confidence=0.24, reason="", similarity_score=0.24), 0.24)]
    apply_continuation_rules(steps, {})
    restore_sections_emptied_by_topic(steps, unboosted)
    assert [s.primary_classification.section_id for s in steps] == ["deployment_and_rollback"] * 3

def test_a_fix_said_right_after_its_failure_stays_with_it():
    """KT-taker audit (dev d05, matrix T01): "The printer queue backs up when a
    warehouse goes offline." "Clear the queue from the admin page, and if that
    doesn't help, call the warehouse IT team." The fix went to Ownership for
    its "call ... team", and Common Failures reported no fix."""
    from types import SimpleNamespace

    from context_mapper import Classification, Sentence
    from section_rules import apply_continuation_rules

    def classified(text, section_id):
        primary = Classification(section_id=section_id, section_title=section_id, confidence=0.6, reason="",
                                 similarity_score=0.6)
        return SimpleNamespace(sentence=Sentence(text=text, start=0.0, end=0.0), primary_classification=primary,
                               is_unassigned=False, explainability_log=None)

    for failure, fix in [
            ("The printer queue backs up when a warehouse goes offline.",
             "Clear the queue from the admin page, and if that doesn't help, call the warehouse IT team."),
            ("The VPN tunnel drops, and then every booking fails with a database timeout.",
             "Reset the tunnel from the Azure portal; if it doesn't come back, call the network team.")]:
        pair = [classified(failure, "common_failures"), classified(fix, "ownership_escalation")]
        apply_continuation_rules(pair, {})
        assert pair[1].primary_classification.section_id == "common_failures", fix
    # Not a fix (an escalation), not after a failure, or a safety rule.
    for previous, nxt, section in [
            ("The printer queue backs up when a warehouse goes offline.",
             "Escalate to Laura Pike, the clinical systems manager.", "ownership_escalation"),
            ("Grafana shows queue depth.", "Restart the consumer when lag grows.", "ownership_escalation"),
            ("The VPN tunnel drops during deploys.", "Never restart the primary database during trading hours.",
             "danger_zones")]:
        pair = [classified(previous, "common_failures" if "drops" in previous or "backs" in previous
                           else "monitoring_observability"), classified(nxt, section)]
        apply_continuation_rules(pair, {})
        assert pair[1].primary_classification.section_id == section, nxt


@pytest.mark.parametrize("text,section", [
    # A user's KT (23 Oct, matrix T03): a scale-up on a threshold went to
    # Cost Optimization, and the size of a cache to Common Failures.
    ("If CPU stays above 75% for five minutes, we should scale up.", "common_failures"),
    ("If the queue backs up, you need to restart the consumer.", "common_failures"),
    ("I think Redis is using 4 GB of memory, but I'm not completely sure.", "architecture_reference"),
    ("The database runs with 32 GB of memory and 8 vCPUs.", "architecture_reference"),
    # Running short of memory is a failure, not a size; not a runbook step.
    ("The pod hits 4 GB of memory and gets OOM-killed.", None),
    ("When Redis fills up its 4 GB, sessions get evicted.", None),
    ("If you need access, you should ask the platform team.", None),
    ("The team has 4 engineers.", None),
])
def test_a_threshold_response_is_a_runbook_step_and_a_size_is_architecture(text, section):
    match = match_section_rules(text)
    assert (match.section_id if match else None) == section


@pytest.mark.parametrize("text,section", [
    # The user's KTs (24 Oct).
    ("I think Redis has a 4 GB memory limit, but this needs verification.", "architecture_reference"),
    ("Redis is used for caching, and I think memory limit is 4 GB, but need to verify this.",
     "architecture_reference"),
    ("The pod hits its 4 GB memory limit and restarts.", None),
    # A caveat about the document is not a "never" rule; a real one is.
    ("Do not describe the one-hour recovery estimate as a verified guarantee.", None),
    ("Never restart the production database without approval from the incident commander.", "danger_zones"),
])
def test_a_stated_limit_is_architecture_and_a_document_caveat_is_no_danger_zone(text, section):
    match = match_section_rules(text)
    assert (match.section_id if match else None) == section


def test_a_permission_is_not_data_moving_to_a_store():
    # Dev d02: "Security wise, the service account only has read access to
    # BigQuery" went to Architecture as if data were read from BigQuery.
    for text in ["the service account only has read access to BigQuery",
                 "The CI role has write access to the artifacts bucket."]:
        assert match_section_rules(text) is None, text
    for text in ["The workers read events from Kafka.", "Exports write files to Cloud Storage."]:
        assert match_section_rules(text).section_id == "architecture_reference", text


def test_what_happens_when_that_happens_stays_with_the_failure():
    # Dev d02: "The thing that goes wrong most is the training job failing ...
    # When that happens the API keeps serving yesterday's recommendations ..."
    # went to Monitoring.
    from types import SimpleNamespace

    from context_mapper import Classification, Sentence
    from section_rules import apply_continuation_rules

    def classified(text, section_id):
        primary = Classification(section_id=section_id, section_title=section_id, confidence=0.6, reason="",
                                 similarity_score=0.6)
        return SimpleNamespace(sentence=Sentence(text=text, start=0.0, end=0.0), primary_classification=primary,
                               is_unassigned=False, explainability_log=None)

    pair = [classified("The thing that goes wrong most is the training job failing because a table schema "
                       "changed upstream.", "common_failures"),
            classified("When that happens the API keeps serving yesterday's recommendations.",
                       "monitoring_observability")]
    apply_continuation_rules(pair, {})
    assert pair[1].primary_classification.section_id == "common_failures"


def test_escalation_contacts_are_ownership_and_escalation():
    # A user's KT (25 Oct): went to Handover Completion Check.
    match = match_section_rules("The video-processing recovery procedure and formal escalation contacts are still "
                                "undocumented.")
    assert match.section_id == "ownership_escalation"
    assert match_section_rules("Escalation paths are clear.") is None


def _classified(text, section_id, source=None):
    from types import SimpleNamespace

    from context_mapper import Classification, Sentence

    primary = Classification(section_id=section_id, section_title=section_id, confidence=0.6, reason="",
                             similarity_score=0.6)
    return SimpleNamespace(sentence=Sentence(text=text, start=0.0, end=0.0, source_text=source),
                           primary_classification=primary, is_unassigned=False, explainability_log=None)

def test_a_topic_named_at_the_head_of_a_sentence_holds_for_every_fact_cut_from_it():
    """Dev d02 and d03: "Security wise, the service account only has read
    access to BigQuery, and the API sits behind Identity-Aware Proxy ..." and
    "uh security, ..., and the certs are rotated every year by the roadside
    team": the second fact went to Architecture and to Ownership."""
    from section_rules import apply_continuation_rules

    meta = {sid: {"title": sid} for sid in ("security_controls", "architecture_reference", "danger_zones",
                                            "ownership_escalation")}
    said = "Security wise, the service account only has read access to BigQuery, and the API sits behind " \
           "Identity-Aware Proxy for the admin endpoints."
    facts = [_classified("Security wise, the service account only has read access to BigQuery.", "security_controls",
                         said),
             _classified("The API sits behind Identity-Aware Proxy for the admin endpoints.", "architecture_reference",
                         said)]
    apply_continuation_rules(facts, meta)
    assert [f.primary_classification.section_id for f in facts] == ["security_controls"] * 2
    # A rule of the fact's own still wins: a never-do rule is a danger zone.
    said = "Security wise, the keys live in Vault, and never share the root token."
    facts = [_classified("Never share the root token.", "danger_zones", said)]
    apply_continuation_rules(facts, meta)
    assert facts[0].primary_classification.section_id == "danger_zones"

def test_a_sentence_about_one_environment_is_environments_and_keeps_the_topic():
    """Dev d03: "staging uses recorded gantry traffic from last year" went to
    Deployment. Dev d01: once that rule moved the topic, "Releases are
    performed with Azure DevOps" after it followed into Environments."""
    from section_rules import segment_topics

    for text in ["staging uses recorded gantry traffic from last year so you can replay a busy day there.",
                 "The test environment receives anonymised statement files from two of our banks.",
                 "Prod has five replicas behind the load balancer."]:
        rule = match_section_rules(text)
        assert rule and rule.section_id == "environments", (text, rule)
    # What a deploy does there, or a stronger rule, is not an environment fact.
    assert match_section_rules("Staging deploys automatically on every merge.").section_id != "environments"
    rule = match_section_rules("Production has point in time recovery switched on, seven days.")
    assert rule and rule.section_id == "disaster_recovery"
    assert segment_topics(["Releases are done in Azure DevOps.", "The test environment receives masked files.",
                           "Releases are performed with Azure DevOps."])[-1] != "environments"

def test_the_kts_opening_definition_of_the_system_is_its_overview():
    """Dev d06: "Brightline is our data pipeline that loads sales and
    inventory data from the stores into Snowflake, and the finance and
    merchandising teams build all their reports on top of it" went to
    Architecture (a data-flow rule)."""
    from section_rules import apply_rule_overrides

    meta = {sid: {"title": sid} for sid in ("system_overview", "architecture_reference", "disaster_recovery")}
    said = "Brightline is our data pipeline that loads sales and inventory data from the stores into Snowflake."
    units = [_classified(said, "architecture_reference", said),
             _classified("Backups, Snowflake keeps time travel for a day.", "disaster_recovery"),
             _classified("Kafka is our event pipeline that writes orders into Snowflake.", "architecture_reference")]
    apply_rule_overrides(units, meta)
    assert units[0].primary_classification.section_id == "system_overview"
    # After a topic is named, the same form describes a component.
    assert units[2].primary_classification.section_id == "architecture_reference"
    rule = match_section_rules("the finance and merchandising teams build all their reports on top of it")
    assert rule and rule.section_id == "system_overview"


@pytest.mark.parametrize("said, section", [
    # Dev d08 (speech to text: no comma, no "on") and d09; the First 30-Day
    # plan was reported Missing although it was discussed (X67).
    ("First day get access to the prod project and the Twilio console.", "day1_survival_checklist"),
    ("In the first month take over the black friday preparation.", "first_30_day_ownership"),
    ("Within the first month, the new owner should complete one release and one dead-letter queue recovery.",
     "first_30_day_ownership"),
    ("In your first month, run the Sunday training on your own.", "first_30_day_ownership"),
    ("After a month I'd like you to be the first responder on your own.", "first_30_day_ownership"),
    ("For your first week, get Vault access and get added to the PagerDuty schedule.", "day1_survival_checklist"),
    # Cut out of a run of speech to text (fact_units._time_frame_pieces).
    ("by the end of the month run a thursday deployment yourself.", "first_30_day_ownership"),
    ("by week three you should own releases.", "first_30_day_ownership"),
    ("Within thirty days: lead one production release and one deduplication run.", "first_30_day_ownership"),
])
def test_the_new_owners_time_frame_opening_a_sentence_is_the_onboarding_plan(said, section):
    rule = match_section_rules(said)
    assert rule and rule.section_id == section, (said, rule)


@pytest.mark.parametrize("said", [
    "The first week of January is the worst week.",
    "In the first month after launch, traffic doubled.",
    "After a month, old logs are deleted.",
    "On day one of an incident, page the lead.",
])
def test_a_calendar_or_retention_time_frame_is_not_onboarding(said):
    rule = match_section_rules(said)
    assert rule is None or rule.section_id not in ("day1_survival_checklist", "first_30_day_ownership"), (said, rule)


def test_a_provider_limiting_us_is_a_failure_whatever_day_it_names():
    """Dev d08: "Twilio rate limits us when there's a big batch like on black
    friday" went to the Operational Calendar."""
    rule = match_section_rules("Twilio rate limits us when there's a big batch like on black friday and messages "
                               "get delayed.")
    assert rule and rule.section_id == "common_failures"
    rule = match_section_rules("Black Friday is our busiest day.")
    assert rule and rule.section_id == "known_bad_days"


@pytest.mark.parametrize("said, section", [
    # Dev d14, d08 and d07.
    ("The Stripe and Postmark keys are in Heroku config vars.", "security_controls"),
    ("The database password is stored in Parameter Store.", "security_controls"),
    ("Card details never touch our servers, Stripe handles all of that.", "security_controls"),
    ("Phone numbers are personal data so we delete message logs after ninety days.", "security_controls"),
    ("It advances every student a year, and undoing it means a database restore.", "danger_zones"),
    # Still prohibitions.
    ("Never touch the production database by hand.", "danger_zones"),
    ("We never edit the Terraform state, it breaks every plan.", "danger_zones"),
    ("A sent payout can't be undone, so check the batch twice.", "danger_zones"),
    ("The thing you really must not do is run the Gift Aid claim job twice for the same month.", "danger_zones"),
    ("You shouldn't ever restart the primary during business hours.", "danger_zones"),
])
def test_where_credentials_live_and_what_data_never_reaches_are_security(said, section):
    rule = match_section_rules(said)
    assert rule and rule.section_id == section, (said, rule)


@pytest.mark.parametrize("said, section", [
    # Dev d13 and d14 (X91): one went to Ownership for the team it names, the
    # other to Cost Optimization for its scaling.
    ("Changes to the MQ bridge configuration require approval from the mainframe team.", "deployment_and_rollback"),
    ("Any change to production requires a sign-off from the CAB.", "deployment_and_rollback"),
    ("Scale the web dynos to twenty before the appeal starts.", "known_bad_days"),
    ("Always pre-warm the cache ahead of the spring sale launch.", "known_bad_days"),
])
def test_change_control_and_getting_ready_for_a_busy_event(said, section):
    rule = match_section_rules(said)
    assert rule and rule.section_id == section, (said, rule)


@pytest.mark.parametrize("said", [
    "Changes to the on-call rota need approval from Maria.",       # who is on call, not change control
    "If the queue backs up, scale the consumers.",                  # a fix
    "We scale down before the night shift starts.",
])
def test_neither_rule_takes_what_is_not_theirs(said):
    rule = match_section_rules(said)
    assert rule is None or rule.section_id not in ("deployment_and_rollback", "known_bad_days"), (said, rule)


@pytest.mark.parametrize("said, aside", [
    # A presenter's aside on a recorded screen share (public SRE recordings,
    # 27 Oct: printed under Danger Zones and System Overview).
    ("I will stop sharing.", True),
    ("And that is all I have.", True),
    ("And I'll use this as an example.", True),
    ("Let's go take a look at,", True),
    ("Can you see my screen?", True),
    # Something named or stated after it: a fact.
    ("Let's take a look at the metrics catalog.", False),
    ("Let me show you how the rollback works.", False),
    ("I will stop the deployment first.", False),
    ("That is all the configuration we need.", False),
])
def test_a_presenter_aside_is_not_a_fact(said, aside):
    from section_rules import _GREETING_ONLY_RE

    assert bool(_GREETING_ONLY_RE.match(said)) is aside
