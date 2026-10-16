"""Atomic facts before mapping (fact_units.py, context_mapper.segment_sentences).

A sentence that states several facts was classified as one unit and went to
one section whole: on the QuickBite KT, "Developers own business services,
Platform Engineering owns EKS and deployment tooling, and the Data Platform
team manages MSK" was one Ownership value, and "Recovery testing is performed
twice a year, but the RTO and RPO have not been formally documented" hid a
knowledge gap inside a Disaster Recovery fact.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_mapper import AudioSegment, Sentence, rejoin_facts, segment_sentences  # noqa: E402
from fact_units import fact_texts, split_into_facts  # noqa: E402

OWNERS = ("Developers own business services, Platform Engineering owns EKS and deployment tooling, "
          "and the Data Platform team manages MSK.")


def test_a_sentence_of_independent_clauses_becomes_one_fact_per_clause():
    assert fact_texts(OWNERS) == ["Developers own business services",
                                  "Platform Engineering owns EKS and deployment tooling",
                                  "the Data Platform team manages MSK."]
    assert fact_texts("GitHub Actions runs CI, Amazon ECR stores container images, and ArgoCD deploys "
                      "workloads to EKS.") == ["GitHub Actions runs CI", "Amazon ECR stores container images",
                                               "ArgoCD deploys workloads to EKS."]
    assert fact_texts("Recovery testing is performed twice a year, but the platform's RTO and RPO have not "
                      "been formally documented.") == [
        "Recovery testing is performed twice a year",
        "the platform's RTO and RPO have not been formally documented."]
    assert fact_texts("The Restaurant Service handles menu updates, while the Dispatch Service assigns "
                      "delivery partners.") == ["The Restaurant Service handles menu updates",
                                                "the Dispatch Service assigns delivery partners."]


def test_a_participle_phrase_with_its_own_subject_is_a_fact_of_its_own():
    # The peak period was a phrase inside the volume sentence and never
    # reached the Operational Calendar.
    assert fact_texts("It processes approximately 90,000 orders per day across multiple cities, with demand "
                      "increasing significantly during weekends and dinner hours.") == [
        "It processes approximately 90,000 orders per day across multiple cities",
        "demand increasing significantly during weekends and dinner hours."]
    # A detail of the clause, not a fact of its own.
    for text in ["We run on EKS, with three node groups.", "Backups run nightly, with snapshots kept for 30 days.",
                 "Never deploy on Fridays, with traffic peaking then."]:
        assert fact_texts(text) == [text.rstrip()], text


def test_the_pieces_are_verbatim_spans_of_the_sentence():
    for a, b in split_into_facts(OWNERS):
        assert OWNERS[a:b] in OWNERS
    assert split_into_facts("Terraform manages infrastructure.") == [(0, len("Terraform manages infrastructure."))]


def test_clauses_that_lean_on_each_other_stay_one_fact():
    for text in [
        # a pronoun pointing back
        "The points API is a Go service, and it is owned by the loyalty team.",
        # the fix belongs with its symptom
        "The symptom is slow checkout, and the fix is to switch the table to on-demand capacity.",
        # a rule and its reason
        "Never deploy on Fridays, we had an outage last time.",
        "And don't change the ALB idle timeout, the partners' clients depend on it.",
        # a condition governs every clause after it
        "When traffic spikes, the autoscaler adds nodes, and the cluster rebalances.",
        # a scene-setting phrase, not a statement
        "During dinner peaks, the most common issue is delayed order processing caused by Kafka consumer lag.",
        # lists of nouns, not of clauses
        "The .NET services on AKS depend on Azure SQL, Azure Cache for Redis and Service Bus.",
        "Danger zones include raw event deletion, BigQuery retention, and manual GKE changes outside ArgoCD.",
        "It sends push notifications, SMS and emails for card transactions, login alerts and statements.",
        "The common problems are a Service Bus backlog, and SQL connection exhaustion during sales peaks.",
        # an apposition and a participle phrase
        "This handover covers Ledgerline, the invoicing platform our finance customers use to bill businesses.",
        "Coming back to architecture, all services run on Amazon EKS.",
        # a reply
        "I think so, but I have never checked.",
    ]:
        assert fact_texts(text) == [text.rstrip()], text


def test_a_predicate_without_its_own_subject_stays_with_its_clause():
    assert fact_texts("The application runs on RHEL virtual machines behind HAProxy, the database is PostgreSQL, "
                      "three nodes managed by Patroni.") == [
        "The application runs on RHEL virtual machines behind HAProxy",
        "the database is PostgreSQL, three nodes managed by Patroni."]


def test_a_clause_that_refers_back_is_the_next_step_of_the_same_flow():
    # Cut off, "a second function rolls them up ... for the dashboards the
    # fleet managers use" was filed under Monitoring.
    for text in ["The vans send readings to IoT Hub, an Azure Function picks them up and writes them into Cosmos DB, "
                 "and a second function rolls them up into Azure Data Explorer for the dashboards the fleet "
                 "managers use.",
                 "GitHub Actions builds the images, pushes them to ECR, and ArgoCD deploys them to EKS."]:
        assert fact_texts(text) == [text.rstrip()], text


def test_the_results_of_a_step_stay_with_the_step_that_produced_them():
    # Cut off, "the results land in BigQuery" was filed beside the system's
    # purpose by a one-call LLM check (16 Oct, live Groq), and a flow was
    # drawn from the system itself instead of from Dataflow.
    for text in ["Events are published to Pub/Sub, Dataflow jobs process them, and the results land in BigQuery.",
                 "Jobs run nightly, and the output goes to S3.",
                 "The service writes events to Kafka, and the results are aggregated in Druid."]:
        assert fact_texts(text) == [text.rstrip()], text


def test_an_answer_is_kept_whole_so_the_question_reaches_all_of_it():
    # Cut into facts, only "All traffic is TLS" went with "Security?" and the
    # task-role fact was filed under Ownership.
    facts = _segment("Security? All traffic is TLS, partners authenticate with API keys stored in Secrets Manager, "
                     "and the services run with task roles that only allow what they need.")
    assert len(facts) == 1 and facts[0].text.startswith("Security? All traffic is TLS")


def _segment(text):
    return segment_sentences([AudioSegment(text=text, start=0.0, end=9.0, avg_logprob=-0.3)])


def test_facts_keep_their_source_sentence_and_are_not_merged_back():
    # Three ownership statements are as alike as sentences get; the semantic
    # chunker merged near-identical neighbours into one unit before.
    facts = _segment(OWNERS)
    assert [f.text for f in facts] == ["Developers own business services.",
                                       "Platform Engineering owns EKS and deployment tooling.",
                                       "The Data Platform team manages MSK."]
    assert {f.source_id for f in facts} == {0} and all(f.source_text == OWNERS for f in facts)
    assert [OWNERS[a:b] for a, b in (f.source_span for f in facts)] == [
        "Developers own business services", "Platform Engineering owns EKS and deployment tooling",
        "the Data Platform team manages MSK."]


def test_facts_placed_side_by_side_in_one_section_read_as_the_spoken_sentence():
    facts = _segment(OWNERS)
    rejoined = rejoin_facts(facts)
    assert [s.text for s in rejoined] == [OWNERS]
    assert rejoined[0].source_span == (0, len(OWNERS))
    # Not neighbours (the middle fact went to another section): kept apart.
    assert [s.text for s in rejoin_facts([facts[0], facts[2]])] == [facts[0].text, facts[2].text]
    # Reworded since (copy edit): kept as written.
    edited = [Sentence(**{**f.__dict__, "text": f.text.replace("own", "are responsible for", 1)}) if i == 0 else f
              for i, f in enumerate(facts[:2])]
    assert [s.text for s in rejoin_facts(edited)] == [
        "Developers are responsible for business services. Platform Engineering owns EKS and deployment tooling."]


def test_each_fact_is_classified_into_its_own_section():
    # The facts of one sentence can belong to different sections; the topic
    # blocks (what a section shows) keep them apart.
    from context_mapper import Classification, ClassifiedSentence, detect_gaps
    from kt_schema_loader import SCHEMA

    text = ("Recovery testing is performed twice a year, and Platform Engineering owns EKS and deployment "
            "tooling.")
    facts = _segment(text)
    assert len(facts) == 2
    classified = [ClassifiedSentence(sentence=f, primary_classification=Classification(
        section_id=sid, section_title=sid, confidence=0.9, reason="test", similarity_score=0.9),
        secondary_classifications=[]) for f, sid in zip(facts, ["disaster_recovery", "ownership_escalation"])]
    coverage = detect_gaps(classified, SCHEMA)
    assert [s.text for b in coverage["disaster_recovery"].blocks for s in b.sentences] == [
        "Recovery testing is performed twice a year."]
    assert [s.text for b in coverage["ownership_escalation"].blocks for s in b.sentences] == [
        "Platform Engineering owns EKS and deployment tooling."]


def test_one_ownership_row_per_owner_in_a_list_of_clauses():
    # The plain comma between the first two owners was not a split point:
    # the row read "Developers -> Business services, Platform Engineering
    # owns EKS and deployment tooling".
    from renderers.sections.ownership_escalation import render

    section = {"id": "ownership_escalation", "title": "Ownership", "fields": {}, "coverage_content": [OWNERS]}
    table = next(b for b in render(section)["blocks"] if b["type"] == "OwnershipTable")
    assert [(r["team"], r["role"]) for r in table["rows"]] == [
        ("Developers", "Business services"),
        ("Platform Engineering", "EKS and deployment tooling"),
        ("Data Platform team", "MSK"),
    ]


def _ko(paragraphs_by_section):
    return {
        "sections": [{"id": "kt_coverage", "title": "KT Coverage",
                      "_coverage_rows": [{"Domain": "X", "Coverage": "Strong", "Assessment": "."}]}],
        "rendered_sections": [
            {"section_id": sid, "section_title": sid,
             "blocks": [{"type": "NarrativeBlock", "title": sid, "paragraphs": paras}]}
            for sid, paras in paragraphs_by_section.items()
        ] + [{"section_id": "kt_coverage", "section_title": "KT Coverage", "blocks": []}],
    }


def test_every_fact_ends_in_the_ledger_with_its_source_section_and_status():
    from knowledge.knowledge_builder import verify_document_coverage

    sentence = ("Recovery testing is performed twice a year, but the platform's RTO and RPO have not been "
                "formally documented.")
    ko = _ko({"disaster_recovery": [sentence]})
    facts = [
        {"text": "Recovery testing is performed twice a year.", "section_id": "disaster_recovery",
         "source": sentence, "start": 12.0, "end": 16.0},
        {"text": "The platform's RTO and RPO have not been formally documented.", "section_id": "disaster_recovery",
         "source": sentence, "start": 12.0, "end": 16.0},
        {"text": "Database backups run daily and are retained for 14 days.", "section_id": "disaster_recovery",
         "source": "Database backups run daily and are retained for 14 days.", "start": 9.0, "end": 12.0},
        {"text": "Hi team, thanks for joining.", "section_id": None, "source": "Hi team, thanks for joining."},
    ]
    excluded = [{"text": "This KT is about the platform.", "reason": "about the KT session itself, not the system"}]
    result = verify_document_coverage(ko, facts, excluded=excluded)
    ledger = result["_facts_ledger"]
    by_text = {e["text"]: e for e in ledger}

    # Shown in its section: mapped, with where it was found and its source.
    entry = by_text["Recovery testing is performed twice a year."]
    assert entry["status"] == "mapped" and entry["found_in"] == ["disaster_recovery"]
    assert entry["source"] == sentence and entry["start"] == 12.0
    # The gap half is a fact of its own, marked as a stated gap.
    assert by_text["The platform's RTO and RPO have not been formally documented."]["gap"] is True
    # Shown nowhere: recovered into Additional Notes, never silently lost.
    backups = by_text["Database backups run daily and are retained for 14 days."]
    assert backups["status"] == "recovered"
    notes = next(s for s in result["rendered_sections"] if s["section_id"] == "unmapped_findings")
    assert "Database backups run daily" in notes["blocks"][0]["paragraphs"][0]
    # Left out on purpose, with the reason.
    assert by_text["Hi team, thanks for joining."]["status"] == "excluded"
    assert by_text["This KT is about the platform."]["reason"] == "about the KT session itself, not the system"
    assert all(e["status"] in ("mapped", "unmapped", "recovered", "excluded") for e in ledger)
    assert len({e["id"] for e in ledger}) == len(ledger) == 5

    summary = result["_knowledge_coverage_summary"]
    assert summary == {"facts_identified": 3, "mapped": 2, "unmapped": 0, "recovered": 1, "lost": 0,
                       "verified_against_document": True, "stated_gaps": 1, "excluded": 2}


def test_each_stated_gap_is_listed_as_a_knowledge_gap():
    from knowledge.knowledge_builder import append_coverage_matrix_section
    from kt_schema_loader import SCHEMA

    coverage = {"disaster_recovery": {"status": "covered", "sentence_count": 3, "sentences": [
        {"text": "Recovery testing is performed twice a year, but the platform's RTO and RPO have not been "
                 "formally documented."},
        {"text": "The payment-provider failover procedure and courier-tracking recovery runbook are still missing."},
    ]}}
    ko = append_coverage_matrix_section({"sections": []}, coverage, SCHEMA)
    gaps = next(s for s in ko["sections"] if s["id"] == "kt_coverage")["_knowledge_gaps"]
    dr_gaps = [g for g in gaps if "RTO and RPO have not been" in g or "still missing" in g]
    assert len(dr_gaps) == 2
    # The gap half is quoted, not the recovery-testing fact before it.
    assert not any("Recovery testing is performed" in g for g in dr_gaps)


def test_stated_gaps_with_an_adverb_or_missing_documents_are_gaps_and_operational_absences_are_not():
    from dialogue import is_gap_statement

    assert is_gap_statement("The platform's RTO and RPO have not been formally documented.")
    assert is_gap_statement("The payment-provider failover procedure and the recovery runbook are still missing.")
    assert not is_gap_statement("The alert fires when a heartbeat is missing for five minutes.")
    assert not is_gap_statement("Recovery testing is performed twice a year.")


def test_a_rise_in_load_over_a_period_belongs_in_the_operational_calendar():
    from section_rules import match_section_rules

    for text in ["Demand increasing significantly during weekends and dinner hours.",
                 "Order volume spikes every Monday morning.", "Usage is highest in the first week of January."]:
        assert match_section_rules(text).section_id == "known_bad_days", text
    for text in ["Traffic increased by 20 percent after the migration.",
                 "We handle three hundred requests a second at peak.", "Requests are retried on failure."]:
        match = match_section_rules(text)
        assert not match or match.section_id != "known_bad_days", text


def test_deployment_steps_are_one_statement_per_item():
    # The field joins the section's matching sentences into one line; shown
    # as one checklist item, a recorded KT's steps were a ten-sentence bullet.
    from renderers.sections.deployment import render

    section = {"id": "deployment_and_rollback", "title": "Deployment", "coverage_content": [], "fields": {
        "deployment_steps": {"value": "GitHub Actions handles CI, ArgoCD handles GitOps deployment. Review the "
                                      "workflow logs before making changes. ArgoCD deploys workloads into GKE."}}}
    steps = next(b for b in render(section)["blocks"] if b.get("title") == "Deployment steps")
    assert steps["items"] == ["GitHub Actions handles CI, ArgoCD handles GitOps deployment.",
                              "Review the workflow logs before making changes.", "ArgoCD deploys workloads into GKE."]


def test_a_three_word_fact_naming_a_tool_is_counted_and_a_bare_remark_is_not():
    from knowledge.knowledge_builder import verify_document_coverage

    ko = _ko({"monitoring_observability": ["Prometheus and Grafana monitor service health, and PagerDuty sends alerts."]})
    result = verify_document_coverage(ko, [
        {"text": "PagerDuty sends alerts.", "section_id": "monitoring_observability"},
        {"text": "Okay, makes sense.", "section_id": None},
    ])
    by_text = {e["text"]: e for e in result["_facts_ledger"]}
    assert by_text["PagerDuty sends alerts."]["status"] == "mapped"
    assert by_text["Okay, makes sense."]["status"] == "excluded"


def test_coverage_counts_what_the_facts_say_and_not_what_they_say_is_missing():
    # The Operational Calendar listed "High-traffic periods" and "Change
    # freezes" as not discussed beside its own statements of both, and the
    # sentence saying RTO and RPO were not documented counted them as covered.
    from knowledge.knowledge_builder import append_coverage_matrix_section
    from kt_schema_loader import SCHEMA

    coverage = {
        "known_bad_days": {"status": "covered", "sentence_count": 2, "sentences": [
            {"text": "Demand increasing significantly during weekends and dinner hours."},
            {"text": "Avoid production deployments during major promotional campaigns."}]},
        "disaster_recovery": {"status": "covered", "sentence_count": 2, "sentences": [
            {"text": "Database backups run daily and are retained for 14 days."},
            {"text": "Recovery testing is performed twice a year, but the platform's RTO and RPO have not been "
                     "formally documented."}]},
    }
    ko = append_coverage_matrix_section({"sections": []}, coverage, SCHEMA)
    section = next(s for s in ko["sections"] if s["id"] == "kt_coverage")
    rows = {r["Domain"]: r["Assessment"] for r in section["_coverage_rows"]}
    calendar = next(v for k, v in rows.items() if "Calendar" in k or "Bad Days" in k)
    assert "High-traffic periods" not in calendar.split("not discussed")[-1]
    assert "Change freezes" not in calendar.split("not discussed")[-1]
    dr = next(v for k, v in rows.items() if "Disaster" in k)
    assert "stated as not documented: RTO, RPO" in dr


def _segment_text(text):
    return [s.text for s in _segment(text)]


def test_no_two_sentences_are_merged_before_classification_except_a_question_and_its_answer():
    # Sentences that read alike were merged into one unit with one section:
    # a system's purpose went to System Overview with its request flow.
    text = ("Today I'm handing over Shopfront, our AWS e-commerce platform, which takes every customer order. "
            "Customers use the React storefront, which is served through CloudFront to an Application Load "
            "Balancer in front of our EKS cluster. Cloud Scheduler kicks off the invoice run. Builds run in Cloud "
            "Build on every push to the release branch.")
    assert len(_segment_text(text)) == 4
    # A question and its answer are one statement, and two exchanges are two.
    assert _segment_text("What is the RTO? We never agreed one with the business.") == [
        "What is the RTO? We never agreed one with the business."]
    assert len(_segment_text("Anything unfinished? The runtime upgrade has not been started and has no owner. "
                             "And for me, starting out? Get added to the admins group first.")) == 2
