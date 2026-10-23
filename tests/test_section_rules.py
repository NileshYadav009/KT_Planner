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
