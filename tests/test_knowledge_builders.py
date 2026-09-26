"""Tests for knowledge/facts.py, entities.py, relationships.py — the builders
that turn populated fields into the knowledge object's facts/entities/
relationships arrays. Covers two real defects fixed this session
(REPOSITORY_AUDIT.md §9j): unfilled fields leaking into `facts` with phantom
evidence, and an escalation-chain separator mismatch between the
pattern-fallback extractor and relationships.py's parser.
"""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.facts import build_facts
from knowledge.entities import build_entities
from knowledge.relationships import build_relationships
from knowledge.knowledge_builder import (
    build_knowledge_object,
    append_unmapped_findings_section,
    enrich_operational_calendar,
    enrich_technology_summary,
    enrich_architecture_knowledge,
    append_tribal_knowledge_section,
    append_coverage_matrix_section,
    append_quick_reference_section,
)
from field_populator import _extract_escalation_chain
from validation import validate_knowledge_object


def _sentence(text, start=0.0, end=1.0, speaker=None, audio_confidence=0.9):
    return SimpleNamespace(text=text, start=start, end=end, speaker=speaker, audio_confidence=audio_confidence)


def test_build_facts_excludes_unfilled_fields():
    fields = {
        "system_name": {"value": "Order Service", "confidence": 0.9, "source": "pattern"},
        "kt_status": {"value": "", "confidence": 0.0, "source": "unfilled"},
        "can_deploy": {"value": None, "confidence": 0.0, "source": "unfilled"},
    }
    facts = build_facts("system_overview", fields)
    assert len(facts) == 1
    assert facts[0]["id"] == "system_name"


def test_build_facts_includes_falsy_but_meaningful_values():
    # A boolean False is a real, meaningful answer (e.g. "can_deploy: No") —
    # must not be treated the same as an unfilled placeholder.
    fields = {"can_deploy": {"value": False, "confidence": 0.9, "source": "pattern"}}
    facts = build_facts("handover_completion", fields)
    assert len(facts) == 1


def test_build_entities_excludes_empty_values():
    fields = {
        "oncall_tool": {"value": "PagerDuty", "confidence": 0.9},
        "unfilled_field": {"value": "", "confidence": 0.0},
    }
    entities = build_entities(fields)
    assert len(entities) == 1
    assert entities[0]["attributes"]["field"] == "oncall_tool"


def test_build_relationships_from_ownership_fields():
    fields = {
        "application_ownership": {"value": "Developers"},
        "infrastructure_ownership": {"value": "Platform Engineers"},
    }
    relationships = build_relationships(fields)
    assert {"subject": "Application code", "relation": "owned by", "target": "Developers"} in relationships
    assert {"subject": "Infrastructure", "relation": "owned by", "target": "Platform Engineers"} in relationships


def test_build_relationships_from_escalation_chain():
    fields = {"escalation_chain": {"value": "on-call engineer -> platform manager -> head of engineering"}}
    relationships = build_relationships(fields)
    assert len(relationships) == 2
    assert relationships[0] == {"subject": "on-call engineer", "relation": "escalates to", "target": "platform manager"}
    assert relationships[1] == {"subject": "platform manager", "relation": "escalates to", "target": "head of engineering"}


def test_escalation_chain_extractor_and_relationship_parser_agree_on_separator():
    # Regression test for the separator mismatch (audit §9j / §9n docstring):
    # _extract_escalation_chain() must produce a string that
    # build_relationships() can actually split back into steps end-to-end.
    text = (
        "Escalation path starts with an on-call engineer followed by the "
        "platform engineering manager and then the head of engineering."
    )
    chain = _extract_escalation_chain(text)
    assert chain is not None
    relationships = build_relationships({"escalation_chain": {"value": chain}})
    assert len(relationships) == 2
    assert relationships[0]["relation"] == "escalates to"


def _base_knowledge_object():
    return {
        "job_id": "job1",
        "system_name": "Test System",
        "sections": [{
            "id": "system_overview", "title": "System Overview", "status": "covered",
            "confidence": 0.9, "risk": 0.1, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
        }],
        "summary": {"section_count": 1, "covered_sections": 1},
    }


def test_append_unmapped_findings_section_is_noop_with_no_sentences():
    ko = _base_knowledge_object()
    result = append_unmapped_findings_section(ko, [])
    assert len(result["sections"]) == 1
    assert result["summary"]["section_count"] == 1


def test_append_unmapped_findings_section_filters_short_filler_sentences():
    ko = _base_knowledge_object()
    # "Thank you." is too short (< 4 words) to be a standalone fact — a
    # generic length filter, not specific to this wording.
    result = append_unmapped_findings_section(ko, [_sentence("Thank you.")])
    assert len(result["sections"]) == 1


def test_append_unmapped_findings_section_adds_well_shaped_section():
    ko = _base_knowledge_object()
    sentences = [
        _sentence("We use Terraform, ArgoCD and Vault for provisioning and secrets."),
        _sentence("Container images are stored in Amazon ECR."),
    ]
    result = append_unmapped_findings_section(ko, sentences)

    assert len(result["sections"]) == 2
    appended = result["sections"][-1]
    assert appended["id"] == "unmapped_findings"
    assert appended["status"] == "covered"
    assert "Terraform" in appended["coverage_content"][0]
    assert "ECR" in appended["coverage_content"][1]
    assert len(appended["evidence"]) == 2

    assert result["summary"]["section_count"] == 2
    assert result["summary"]["covered_sections"] == 2

    # Must not introduce any structural validation warnings.
    assert validate_knowledge_object(result) == []


def test_append_unmapped_findings_section_mutates_and_returns_same_object():
    ko = _base_knowledge_object()
    sentences = [_sentence("We use Terraform, ArgoCD and Vault for provisioning.")]
    result = append_unmapped_findings_section(ko, sentences)
    assert result is ko


def _section(section_id, title, **extra):
    base = {
        "id": section_id, "title": title, "status": "covered", "confidence": 0.8,
        "risk": 0.0, "facts": [], "entities": [], "evidence": [], "relationships": [],
        "fields": {}, "coverage_content": [],
    }
    base.update(extra)
    return base


def test_append_unmapped_findings_section_drops_sentences_already_mapped_elsewhere():
    # context_mapper.py's classifier can independently decide the same
    # sentence both belongs in a real section AND is "unassigned" — without
    # a cross-check that sentence renders twice (once correctly, once again
    # as a fabricated appendix "gap"). A sentence whose text is already part
    # of another section's coverage_content (even folded into a larger
    # narrative paragraph) must not survive into Unmapped Findings.
    ko = {
        "sections": [{
            "id": "system_overview", "title": "System Overview", "status": "covered",
            "confidence": 0.9, "risk": 0.1, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
            "coverage_content": [
                "The platform consists of React frontend applications, Python FastAPI "
                "services, PostgreSQL databases, and Redis cache layers. It processes "
                "50,000 orders per day."
            ],
        }],
        "summary": {"section_count": 1, "covered_sections": 1},
    }
    sentences = [
        # Same fact as system_overview's narrative above, just with
        # different comma placement (LLM-polish vs. raw wording) — must be
        # dropped.
        _sentence("The platform consists of React frontend applications, Python FastAPI services, PostgreSQL, databases and Redis cache layers."),
        # Genuinely never mapped anywhere — must survive.
        _sentence("We also maintain a separate internal admin tool nobody has documented."),
    ]
    result = append_unmapped_findings_section(ko, sentences)
    appended = next(s for s in result["sections"] if s["id"] == "unmapped_findings")
    assert len(appended["coverage_content"]) == 1
    assert "admin tool" in appended["coverage_content"][0]


def test_append_unmapped_findings_section_drops_sentence_with_leading_discourse_marker():
    # A raw sentence can carry a discourse marker ("Coming back to
    # architecture, ...") that a real section's classifier trims off before
    # storing the fact — the unassigned copy is then a superset (not an
    # exact/substring-after-punctuation-strip match) of the mapped one. Must
    # still be recognized as the same fact.
    ko = {
        "sections": [{
            "id": "architecture_reference", "title": "Architecture Reference", "status": "covered",
            "confidence": 0.9, "risk": 0.1, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
            "coverage_content": ["All services run on Amazon EKS."],
        }],
        "summary": {"section_count": 1, "covered_sections": 1},
    }
    sentences = [
        _sentence("Coming back to architecture, all services run on Amazon EKS."),
        _sentence("We also maintain a separate internal admin tool nobody has documented."),
    ]
    result = append_unmapped_findings_section(ko, sentences)
    appended = next(s for s in result["sections"] if s["id"] == "unmapped_findings")
    assert len(appended["coverage_content"]) == 1
    assert "admin tool" in appended["coverage_content"][0]


def test_enrich_operational_calendar_pulls_cost_optimization_levers():
    ko = {
        "sections": [
            _section("known_bad_days", "OPERATIONAL CALENDAR"),
            _section("cost_optimization", "COST OPTIMIZATION", _structured={
                "levers": [{"lever": "Spot instances", "detail": "Used in non-production environment"}]
            }),
        ],
        "summary": {},
    }
    result = enrich_operational_calendar(ko)
    calendar = next(s for s in result["sections"] if s["id"] == "known_bad_days")
    assert calendar["_cost_patterns"] == [{"label": "Spot instances", "value": "Used in non-production environment"}]


def test_enrich_operational_calendar_noop_when_sections_missing():
    ko = {"sections": [_section("known_bad_days", "OPERATIONAL CALENDAR")], "summary": {}}
    result = enrich_operational_calendar(ko)
    calendar = result["sections"][0]
    assert "_cost_patterns" not in calendar


def test_enrich_technology_summary_pulls_monitoring_and_security_tools():
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW", fields={
                "key_technologies": {"value": "Terraform, ArgoCD, Vault", "confidence": 0.9, "source": "pattern"}
            }),
            _section("monitoring_observability", "MONITORING & OBSERVABILITY", fields={
                "tools": {"value": ["Prometheus", "Grafana", "CloudWatch", "PagerDuty"], "confidence": 0.75, "source": "llm_structured"}
            }),
            _section("security_controls", "SECURITY", fields={
                "security_scan_config": {"value": "Trivy for container image scanning", "confidence": 0.75, "source": "llm_structured"}
            }),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    tools = [t.strip() for t in overview["fields"]["key_technologies"]["value"].split(",")]
    assert set(tools) == {"Terraform", "ArgoCD", "Vault", "Prometheus", "Grafana", "CloudWatch", "PagerDuty", "Trivy"}


def test_enrich_technology_summary_dedupes_case_insensitively():
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW", fields={
                "key_technologies": {"value": "Trivy", "confidence": 0.9, "source": "pattern"}
            }),
            _section("security_controls", "SECURITY", fields={
                "security_scan_config": {"value": "Trivy for container image scanning", "confidence": 0.75, "source": "llm_structured"}
            }),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    tools = [t.strip() for t in overview["fields"]["key_technologies"]["value"].split(",")]
    assert tools.count("Trivy") == 1


def test_enrich_technology_summary_falls_back_to_raw_coverage_content():
    # When structured extraction fails/is unavailable (fields stays empty),
    # the raw sentences are still right there in coverage_content — must not
    # lose the tool mentions just because the LLM path didn't fire.
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW", fields={
                "key_technologies": {"value": "Terraform", "confidence": 0.9, "source": "pattern"}
            }),
            _section("monitoring_observability", "MONITORING & OBSERVABILITY", fields={},
                      coverage_content=["We use Prometheus, Grafana and CloudWatch for monitoring."]),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    tools = {t.strip() for t in overview["fields"]["key_technologies"]["value"].split(",")}
    assert tools == {"Terraform", "Prometheus", "Grafana", "CloudWatch"}


def test_enrich_technology_summary_noop_without_system_overview():
    ko = {"sections": [_section("monitoring_observability", "MONITORING")], "summary": {}}
    result = enrich_technology_summary(ko)
    assert result is ko


def test_enrich_architecture_knowledge_pulls_components_from_any_section():
    # architecture_reference's own field schema is 3 admin facts only — the
    # actual component stack gets classified wherever its sentences
    # naturally landed (system_overview here), and must still surface as
    # this section's own knowledge digest rather than being lost.
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE"),
            _section(
                "system_overview", "SYSTEM OVERVIEW",
                coverage_content=[
                    "The platform runs a React frontend backed by FastAPI services on Amazon EKS.",
                    "The primary database is Amazon RDS PostgreSQL, with Redis for caching and SQS for async jobs.",
                    "Traffic is served through CloudFront and an Application Load Balancer, with images stored in Amazon ECR.",
                ],
            ),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    components = {c.lower() for c in arch["_architecture_components"]}
    # Bare "SQS" canonicalizes to "Amazon SQS" (see
    # _canonicalize_component_term) — the same service, one display form
    # regardless of which form this particular mention used.
    assert components == {
        "react", "fastapi", "amazon eks", "amazon rds", "postgresql",
        "redis", "amazon sqs", "cloudfront", "application load balancer", "amazon ecr",
    }


def test_enrich_architecture_knowledge_dedupes_case_insensitively():
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=["All services run on Amazon EKS."]),
            _section("system_overview", "SYSTEM OVERVIEW",
                     coverage_content=["We standardized on amazon eks for orchestration."]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    lowered = [c.lower() for c in arch["_architecture_components"]]
    assert lowered.count("amazon eks") == 1


def test_enrich_architecture_knowledge_does_not_let_one_sections_raw_sentences_starve_another_sections_coverage_content():
    # Regression test for a real, live bug: a section's raw section_content
    # ['sentences'] and its coverage_content are populated by two
    # independent mechanisms that can disagree (see field_populator.py's
    # populate_fields() docstring) — a section can have real
    # coverage_content while its own raw ['sentences'] stays empty. The
    # coverage_content scan used to be gated on "found nothing anywhere
    # yet" (an all-or-nothing fallback), so the instant ANY section's raw
    # sentences produced a match, every OTHER section's coverage_content
    # was silently skipped for the rest of the run. Live symptom: a real
    # AWS transcript's "Amazon RDS PostgreSQL is the primary database...
    # Amazon SQS handles asynchronous order processing" and "The back end
    # consists of Python FastAPI microservices" sentences were visibly
    # rendered elsewhere in the KT (system_overview's own coverage_content)
    # but never reached the component list or diagram, because a
    # DIFFERENT section's raw sentences (e.g. one mentioning Redis) had
    # already produced a match first.
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE"),
            _section(
                "system_overview", "SYSTEM OVERVIEW",
                # coverage_content has the real facts, but raw
                # section_content['sentences'] for this section (below)
                # is EMPTY — simulating the real divergence.
                coverage_content=[
                    "Amazon RDS PostgreSQL is the primary database, Redis is used for caching and short-lived session data, and Amazon SQS handles asynchronous order processing.",
                    "The back end consists of Python FastAPI microservices.",
                ],
            ),
            _section("environments", "ENVIRONMENTS", coverage_content=[]),
        ],
        "summary": {},
    }
    # A DIFFERENT section's raw section_content sentences produce a match
    # FIRST (in iteration order) — this must not starve system_overview's
    # coverage_content scan below.
    section_content = {
        "environments": {"sentences": [{"text": "A previous major incident was caused by Redis memory saturation."}]},
        "system_overview": {"sentences": []},
    }
    result = enrich_architecture_knowledge(ko, section_content)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    components = {c.lower() for c in arch["_architecture_components"]}
    assert "amazon rds" in components
    assert "amazon sqs" in components
    assert "fastapi" in components
    assert "redis" in components


def test_enrich_architecture_knowledge_recognizes_azure_vocabulary():
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=[
                          "The platform runs on Azure Kubernetes Service, or AKS.",
                          "Azure SQL is the primary database.",
                          "Azure Service Bus provides asynchronous messaging.",
                          "Container images are stored in Azure Container Registry, or ACR.",
                          "Azure Key Vault is used for secret management.",
                          "External traffic is handled through Azure Front Door and Application Gateway.",
                      ]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    components = {c.lower() for c in arch["_architecture_components"]}
    assert "azure kubernetes service" in components
    assert "azure sql" in components
    assert "azure service bus" in components
    assert "azure container registry" in components
    assert "azure key vault" in components
    assert "azure front door" in components
    assert "application gateway" in components


def test_enrich_architecture_knowledge_canonicalizes_bare_acronyms_to_branded_names():
    # A transcript commonly states the full/branded name once and uses the
    # bare acronym on every later mention — both must collapse to ONE
    # display entry, not show up as two different-looking components.
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=[
                          "Amazon RDS PostgreSQL is the primary database.",
                          "For disaster recovery, RDS snapshots are restored.",
                          "Images are stored in Amazon ECR.",
                          "GitHub Actions builds images and pushes them to ECR.",
                      ]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    lowered = [c.lower() for c in arch["_architecture_components"]]
    assert lowered.count("amazon rds") == 1
    assert lowered.count("amazon ecr") == 1
    assert "rds" not in lowered
    assert "ecr" not in lowered


def test_enrich_architecture_knowledge_recognizes_gcp_vocabulary():
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=[
                          "GKE is used for Kubernetes workloads.",
                          "Pub/Sub is used for event ingestion and messaging.",
                          "Dataflow is used for data processing.",
                          "BigQuery is used for analytical workloads.",
                          "Vertex AI supports machine learning.",
                          "Airflow handles workflow orchestration.",
                          "Container images are stored in Artifact Registry.",
                          "Secret Manager handles secrets.",
                      ]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    components = {c.lower() for c in arch["_architecture_components"]}
    assert "google kubernetes engine" in components  # "gke" canonicalizes
    assert "pub/sub" in components
    assert "dataflow" in components
    assert "bigquery" in components
    assert "vertex ai" in components
    assert "airflow" in components
    assert "artifact registry" in components
    assert "secret manager" in components


def test_enrich_architecture_knowledge_noop_without_architecture_reference():
    ko = {"sections": [_section("system_overview", "SYSTEM OVERVIEW")], "summary": {}}
    result = enrich_architecture_knowledge(ko)
    assert result is ko


def test_enrich_architecture_knowledge_noop_when_nothing_detected():
    ko = {"sections": [_section("architecture_reference", "ARCHITECTURE REFERENCE",
                                 coverage_content=["Nothing technical was discussed here."])], "summary": {}}
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    assert "_architecture_components" not in arch


def test_enrich_architecture_knowledge_captures_verbatim_descriptive_sentences():
    # A bare component name ("Redis") says nothing about its role — the
    # sentence that named it usually does. Must come from real per-sentence
    # section_content (clean, atomic sentences) when available, verbatim.
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE"),
        ],
        "summary": {},
    }
    section_content = {
        "architecture_reference": {
            "sentences": [
                {"text": "Amazon RDS PostgreSQL is the primary database."},
                {"text": "Redis is used for caching and short-lived session data."},
                {"text": "This sentence names nothing technical at all."},
            ]
        }
    }
    result = enrich_architecture_knowledge(ko, section_content)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    sentences = arch["_architecture_sentences"]
    assert "Amazon RDS PostgreSQL is the primary database." in sentences
    assert "Redis is used for caching and short-lived session data." in sentences
    assert "This sentence names nothing technical at all." not in sentences


def test_enrich_architecture_knowledge_falls_back_to_coverage_content_without_section_content():
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=["All services run on Amazon EKS."]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    assert arch["_architecture_sentences"] == ["All services run on Amazon EKS."]


def test_enrich_architecture_knowledge_attaches_a_flow_diagram_when_a_request_flow_is_present():
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=["React talks to CloudFront and an Application Load Balancer in front of Amazon EKS, which uses Amazon RDS."]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    assert "_architecture_diagram" in arch
    assert "Amazon EKS" in arch["_architecture_diagram"]


def test_enrich_architecture_knowledge_diagram_has_no_main_chain_when_only_supporting_infrastructure_is_named():
    # Terraform (IaC) and Jenkins (CI/CD) are recognized supporting-
    # infrastructure layers with their own diagram sections (see
    # architecture_diagram.py) — a diagram DOES get generated, but it must
    # have no request-flow chain/Customer node, since nothing resembling
    # a frontend/compute/data-layer position was ever named.
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=["We use Terraform and Jenkins for infrastructure automation."]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    diagram = arch.get("_architecture_diagram")
    assert diagram is not None
    assert "Customer" not in diagram
    assert "Terraform" in diagram
    assert "Jenkins" in diagram


def test_enrich_architecture_knowledge_omits_diagram_entirely_when_nothing_recognized():
    ko = {
        "sections": [
            _section("architecture_reference", "ARCHITECTURE REFERENCE",
                      coverage_content=["We reviewed the team's on-call rotation schedule."]),
        ],
        "summary": {},
    }
    result = enrich_architecture_knowledge(ko)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")
    assert "_architecture_diagram" not in arch


def test_append_tribal_knowledge_section_tags_marker_phrases_by_source_section():
    ko = {"sections": [], "summary": {}}
    section_content = {
        "danger_zones": {"sentences": [{"text": "Never modify Terraform state files manually."}]},
        "monitoring_observability": {"sentences": [
            {"text": "By the way, the first thing you should check is Grafana."},
            {"text": "This sentence has no marker phrase at all."},
        ]},
    }
    result = append_tribal_knowledge_section(ko, section_content)
    tribal = next(s for s in result["sections"] if s["id"] == "tribal_knowledge")
    rows = tribal["_tribal_rows"]
    assert len(rows) == 2
    classifications = {row["Classification"] for row in rows}
    assert classifications == {"Safety-critical", "Operational shortcut"}


def test_append_tribal_knowledge_section_noop_with_no_matches():
    ko = {"sections": [], "summary": {}}
    section_content = {"system_overview": {"sentences": [{"text": "Business criticality is High."}]}}
    result = append_tribal_knowledge_section(ko, section_content)
    assert not any(s["id"] == "tribal_knowledge" for s in result["sections"])


def test_append_coverage_matrix_section_buckets_directly_from_status():
    # Regression test: bucketing used to also require confidence >= 0.6 on
    # top of status == "covered", but the pipeline's confidence signal
    # (mean block-confidence) and its status signal (semantic_coverage_score,
    # already required/optional- and density-aware) are only loosely
    # correlated — real runs showed "covered" status with confidence well
    # under 0.6, so "Strong" never fired. Bucketing must now come straight
    # from status alone.
    ko = {"sections": [], "summary": {}}
    dynamic_schema = [
        {"id": "system_overview", "title": "SYSTEM OVERVIEW"},
        {"id": "architecture_reference", "title": "ARCHITECTURE REFERENCE"},
        {"id": "cost_optimization", "title": "COST OPTIMIZATION"},
    ]
    coverage = {
        "system_overview": {"status": "covered", "confidence": 0.2, "sentence_count": 5},
        "architecture_reference": {"status": "weak", "confidence": 0.9, "sentence_count": 2},
        "cost_optimization": {"status": "missing", "confidence": 0.0, "sentence_count": 0},
    }
    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    rows = {row["Domain"]: row["Coverage"] for row in matrix["_coverage_rows"]}
    assert rows == {
        "SYSTEM OVERVIEW": "Strong",
        "ARCHITECTURE REFERENCE": "Partial",
        "COST OPTIMIZATION": "Missing",
    }


def test_append_coverage_matrix_section_reports_field_level_coverage():
    # A raw sentence count says nothing about completeness (five sentences
    # might back one field or ten) — when the section has a real fields
    # schema, the assessment should say how many of those known fields
    # actually got captured instead.
    ko = {
        "sections": [{
            "id": "system_overview", "title": "SYSTEM OVERVIEW", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "coverage_content": [],
            "fields": {
                "business_criticality": {"value": "High", "source": "llm_explicit"},
                "customer_reach": {"value": "global", "source": "llm_explicit"},
                "impact_if_down": {"value": "", "source": "unfilled"},
            },
        }],
        "summary": {},
    }
    dynamic_schema = [{
        "id": "system_overview", "title": "SYSTEM OVERVIEW",
        "fields": [
            {"id": "business_criticality", "label": "Business Criticality", "type": "single_select"},
            {"id": "customer_reach", "label": "Customer Reach", "type": "text"},
            {"id": "impact_if_down", "label": "Impact if Down", "type": "text"},
        ],
    }]
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 5}}

    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    row = matrix["_coverage_rows"][0]
    assert row["Coverage"] == "Strong"
    assert "2 of 3 known field(s) captured" in row["Assessment"]
    assert "Impact if Down" in row["Assessment"]
    assert "supporting sentence" not in row["Assessment"]


def test_append_coverage_matrix_section_splits_knowledge_from_metadata_for_metadata_role_sections():
    # architecture_reference's 3 leaf fields (doc link, last updated,
    # verified-by) are purely administrative metadata — real architecture
    # knowledge lives in _architecture_components instead. Reporting "0 of
    # 3 known field(s) captured" there reads as "nothing was captured" even
    # when the transcript described a rich, real component stack. A
    # fields_role: "metadata" section with knowledge items must report both
    # halves instead of collapsing to the raw field count.
    ko = {
        "sections": [{
            "id": "architecture_reference", "title": "ARCHITECTURE REFERENCE", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "coverage_content": [],
            "_architecture_components": ["EKS", "React", "CloudFront", "ALB", "FastAPI", "RDS", "Redis", "SQS", "ECR"],
            "fields": {
                "architecture_link": {"value": "", "source": "unfilled"},
                "last_updated": {"value": "", "source": "unfilled"},
                "verified_by_incoming_owner": {"value": "", "source": "unfilled"},
            },
        }],
        "summary": {},
    }
    dynamic_schema = [{
        "id": "architecture_reference", "title": "ARCHITECTURE REFERENCE", "fields_role": "metadata",
        "fields": [
            {"id": "architecture_link", "label": "Documentation link", "type": "url"},
            {"id": "last_updated", "label": "Last updated", "type": "date"},
            {"id": "verified_by_incoming_owner", "label": "Verified by incoming owner", "type": "boolean"},
        ],
    }]
    coverage = {"architecture_reference": {"status": "covered", "confidence": 0.8, "sentence_count": 6}}

    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    row = matrix["_coverage_rows"][0]
    assert row["Coverage"] == "Strong"
    assert "9 component(s) identified" in row["Assessment"]
    assert "EKS" in row["Assessment"]
    assert "0 of 3 metadata field(s) discussed" in row["Assessment"]


def test_append_coverage_matrix_section_collects_knowledge_gaps_separately():
    ko = {"sections": [], "summary": {}}
    dynamic_schema = [
        {"id": "system_overview", "title": "SYSTEM OVERVIEW"},
        {"id": "first_30_day_ownership", "title": "FIRST 30-DAY OWNERSHIP PLAN"},
        {"id": "handover_completion", "title": "HANDOVER COMPLETION CHECK"},
    ]
    coverage = {
        "system_overview": {"status": "covered", "confidence": 0.9, "sentence_count": 5},
        "first_30_day_ownership": {"status": "missing", "confidence": 0.0, "sentence_count": 0},
        "handover_completion": {"status": "missing", "confidence": 0.0, "sentence_count": 0},
    }
    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    assert matrix["_knowledge_gaps"] == ["FIRST 30-DAY OWNERSHIP PLAN", "HANDOVER COMPLETION CHECK"]


def test_append_coverage_matrix_section_builds_knowledge_coverage_summary_with_zero_lost():
    # Knowledge coverage (how many transcript FACTS were captured) must be a
    # genuinely separate number from the Domain/Coverage/Assessment matrix
    # (template-field population) — spec's "must never be confused" rule.
    # append_unmapped_findings_section stashes its substantive-unassigned
    # split as knowledge_object["_dedup_stats"] before this function runs;
    # this reads it and must never fabricate a nonzero "lost" count when the
    # accounting is genuinely balanced.
    ko = {"sections": [], "summary": {}, "_dedup_stats": {"substantive_unassigned": 5, "deduplicated": 2, "unmapped": 3}}
    dynamic_schema = [
        {"id": "system_overview", "title": "SYSTEM OVERVIEW"},
        {"id": "architecture_reference", "title": "ARCHITECTURE REFERENCE"},
    ]
    coverage = {
        "system_overview": {"status": "covered", "confidence": 0.9, "sentence_count": 5},
        "architecture_reference": {"status": "weak", "confidence": 0.5, "sentence_count": 2},
    }
    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    summary = matrix["_knowledge_coverage_summary"]
    assert summary["mapped"] == 7  # 5 + 2 sentence_count across the two real sections
    assert summary["deduplicated"] == 2
    assert summary["unmapped"] == 3
    assert summary["facts_identified"] == 12  # 7 + 2 + 3
    assert summary["lost"] == 0
    # The transport key must be consumed, not left dangling on the object.
    assert "_dedup_stats" not in result


def test_append_coverage_matrix_section_knowledge_coverage_summary_defaults_to_zero_without_dedup_stats():
    # append_unmapped_findings_section may have failed/been skipped upstream
    # (pipeline.py wraps it in a try/except) — must degrade to zero counts,
    # not crash.
    ko = {"sections": [], "summary": {}}
    dynamic_schema = [{"id": "system_overview", "title": "SYSTEM OVERVIEW"}]
    coverage = {"system_overview": {"status": "covered", "confidence": 0.9, "sentence_count": 4}}
    result = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in result["sections"] if s["id"] == "kt_coverage")
    summary = matrix["_knowledge_coverage_summary"]
    assert summary == {"facts_identified": 4, "mapped": 4, "deduplicated": 0, "unmapped": 0, "lost": 0}


def test_knowledge_coverage_summary_stays_consistent_end_to_end_with_unmapped_findings():
    # Integration test: chain the two real functions in their actual
    # pipeline.py order (append_unmapped_findings_section, then
    # append_coverage_matrix_section) rather than hand-constructing
    # _dedup_stats, so a future refactor that breaks the handoff between
    # them fails a test instead of silently drifting.
    ko = {
        "sections": [{
            "id": "system_overview", "title": "SYSTEM OVERVIEW", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "coverage_content": ["The system is business critical and handles global traffic."],
            "fields": {},
        }],
        "summary": {},
    }
    sentences = [
        _sentence("The system is business critical and handles global traffic."),  # duplicate of mapped content
        _sentence("CloudFront cache invalidation can take longer than expected during releases."),  # genuinely unmapped
    ]
    ko = append_unmapped_findings_section(ko, sentences)
    dynamic_schema = [{"id": "system_overview", "title": "SYSTEM OVERVIEW"}]
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 3}}
    ko = append_coverage_matrix_section(ko, coverage, dynamic_schema)
    matrix = next(s for s in ko["sections"] if s["id"] == "kt_coverage")
    summary = matrix["_knowledge_coverage_summary"]
    assert summary["deduplicated"] == 1
    assert summary["unmapped"] == 1
    assert summary["mapped"] == 3
    assert summary["facts_identified"] == summary["mapped"] + summary["deduplicated"] + summary["unmapped"]
    assert summary["lost"] == 0


def test_append_quick_reference_section_skips_rows_with_no_source_data():
    ko = {
        "sections": [
            _section("ownership_escalation", "OWNERSHIP", fields={
                "escalation_chain": {"value": "on-call -> manager", "confidence": 0.7, "source": "llm_structured"}
            }),
        ],
        "summary": {},
    }
    result = append_quick_reference_section(ko)
    qr = next(s for s in result["sections"] if s["id"] == "quick_reference")
    rows = qr["_quick_reference_rows"]
    assert rows == [{"Situation": "Escalation", "Immediate reference": "on-call -> manager"}]


def test_append_quick_reference_section_noop_when_nothing_available():
    ko = {"sections": [], "summary": {}}
    result = append_quick_reference_section(ko)
    assert not any(s.get("id") == "quick_reference" for s in result["sections"])


def _minimal_schema(section_id, title, fields, tier=None):
    section = {"id": section_id, "title": title, "fields": fields}
    if tier is not None:
        section["tier"] = tier
    return section


def test_build_knowledge_object_marks_llm_inferred_values_only():
    # Regression guard for the evidence-state marker: only source == "llm"
    # (field_populator.py's free-form gap-fill, no direct grounding
    # sentence) should be flagged — pattern/semantic/llm_structured are all
    # anchored to real transcript text and must render unmodified.
    dynamic_schema = [_minimal_schema("system_overview", "SYSTEM OVERVIEW", [
        {"id": "business_criticality", "type": "text"},
        {"id": "system_name", "type": "text"},
        {"id": "key_technologies", "type": "text"},
    ])]
    populated_fields = {
        "system_overview": {
            "business_criticality": {"value": "High", "confidence": 0.75, "source": "llm"},
            "system_name": {"value": "Order Service", "confidence": 0.9, "source": "pattern"},
            "key_technologies": {"value": "Terraform", "confidence": 0.65, "source": "semantic"},
        }
    }
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 3}}

    ko = build_knowledge_object("job1", coverage, dynamic_schema, populated_fields)
    fields = ko["sections"][0]["fields"]

    assert fields["business_criticality"]["value"].startswith("High")
    assert "inferred" in fields["business_criticality"]["value"]
    assert fields["system_name"]["value"] == "Order Service"
    assert fields["key_technologies"]["value"] == "Terraform"


def test_build_knowledge_object_does_not_mark_llm_explicit_values():
    # field_populator.py tags an LLM gap-fill "llm_explicit" (not "llm")
    # when the model itself judges the fact was directly stated, just
    # paraphrased (e.g. "most business critical system" -> "High"). That
    # must render unmodified, same as pattern/semantic — only the model's
    # own "I had to guess" case ("llm") gets the inferred marker.
    dynamic_schema = [_minimal_schema("system_overview", "SYSTEM OVERVIEW", [
        {"id": "business_criticality", "type": "single_select"},
        {"id": "customer_reach", "type": "text"},
    ])]
    populated_fields = {
        "system_overview": {
            "business_criticality": {"value": "High", "confidence": 0.75, "source": "llm_explicit"},
            "customer_reach": {"value": "global", "confidence": 0.75, "source": "llm_explicit"},
        }
    }
    coverage = {"system_overview": {"status": "covered", "confidence": 0.8, "sentence_count": 3}}

    ko = build_knowledge_object("job1", coverage, dynamic_schema, populated_fields)
    fields = ko["sections"][0]["fields"]

    assert fields["business_criticality"]["value"] == "High"
    assert fields["customer_reach"]["value"] == "global"


def test_build_knowledge_object_leaves_non_string_values_unmarked():
    # A boolean/list value with source "llm" must not get a string suffix
    # appended (which would silently coerce it to a string everywhere).
    dynamic_schema = [_minimal_schema("handover_completion", "HANDOVER COMPLETION CHECK", [
        {"id": "can_deploy", "type": "boolean"},
    ])]
    populated_fields = {"handover_completion": {"can_deploy": {"value": True, "confidence": 0.7, "source": "llm"}}}
    coverage = {"handover_completion": {"status": "weak", "confidence": 0.5, "sentence_count": 1}}

    ko = build_knowledge_object("job1", coverage, dynamic_schema, populated_fields)
    assert ko["sections"][0]["fields"]["can_deploy"]["value"] is True


def test_build_knowledge_object_system_name_fallback_rejects_bare_stopwords():
    # Regression test: the fallback regex used to take the leftmost "<X>
    # platform/system/..." match unconditionally via re.search, and since
    # "the system"/"the platform" is an extremely common phrase, it
    # frequently landed on a bare stopword ("The") — worse than the
    # explicit "KT Document" default. No populated system_name field
    # here — this exercises the coverage_content fallback path directly.
    coverage = {"system_overview": {"content": ["Before we start, the system needs a restart sometimes."]}}
    ko = build_knowledge_object("job1", coverage, [], {})
    assert ko["system_name"] == "KT Document"


def test_build_knowledge_object_system_name_fallback_keeps_contraction_intact():
    # A second, related bug found alongside the stopword one: the capture
    # character class didn't include apostrophes, so "It's the old claims
    # system" matched starting mid-word ("s the old claims") instead of
    # from "It's" — same root-cause class as the "s three" fuzzy-match
    # corruption bug in devops_transcription.py (contractions splitting on
    # the apostrophe), just surfacing in a different regex.
    coverage = {"system_overview": {"content": ["It's the old claims system, been around forever."]}}
    ko = build_knowledge_object("job1", coverage, [], {})
    assert not ko["system_name"].lower().startswith("s the")
    assert ko["system_name"].lower().startswith("it")


def test_build_knowledge_object_system_name_fallback_trims_leading_verb_filler():
    # Live-app regression: "The system is named the Cloud NAT Order
    # Processing Platform." (an LLM-polished rewrite of "The system name
    # is X") has no field-level trigger match at all (no "handing over"/
    # "this is"/etc immediately before the name), so it fell to this
    # fallback — which correctly rejected "The" (bare stopword) via
    # finditer, but then accepted "is named the Cloud NAT Order
    # Processing" wholesale as the next candidate, since it contains real
    # substantive words and so passed the stopword-only guard. The
    # document title ended up "Is Named The Cloud Nat Order Processing".
    coverage = {
        "system_overview": {
            "content": ["The system is named the Cloud NAT Order Processing Platform."]
        }
    }
    ko = build_knowledge_object("job1", coverage, [], {})
    assert ko["system_name"] == "Cloud Nat Order Processing"


def test_build_knowledge_object_recovers_real_label_and_type_from_schema():
    # Regression test: populated field entries (field_populator.py's
    # {"value", "confidence", "source"} shape) never carry the schema's own
    # "label"/"type" — field.get("label", fid) and field.get("type", "text")
    # always fell through to the bare field id and a hardcoded "text"
    # respectively, for every field in every section. Silently wrong
    # everywhere, just rarely visible because most renderers hardcode their
    # own display labels instead of trusting fields[id]["label"].
    dynamic_schema = [_minimal_schema("first_30_day_ownership", "FIRST 30-DAY OWNERSHIP PLAN", [
        {"id": "week1", "label": "Week 1", "type": "text"},
        {"id": "verified", "label": "Verified", "type": "boolean"},
    ])]
    populated_fields = {
        "first_30_day_ownership": {
            "week1": {"value": "Observe, shadow, read-only", "confidence": 0.65, "source": "semantic"},
            "verified": {"value": True, "confidence": 0.75, "source": "pattern"},
        }
    }
    ko = build_knowledge_object("job1", {}, dynamic_schema, populated_fields)
    fields = ko["sections"][0]["fields"]
    assert fields["week1"]["label"] == "Week 1"
    assert fields["week1"]["type"] == "text"
    assert fields["verified"]["label"] == "Verified"
    assert fields["verified"]["type"] == "boolean"


def test_build_knowledge_object_carries_section_tier():
    dynamic_schema = [
        _minimal_schema("system_overview", "SYSTEM OVERVIEW", [], tier=None),
        _minimal_schema("cost_optimization", "COST OPTIMIZATION", [], tier="conditional"),
    ]
    ko = build_knowledge_object("job1", {}, dynamic_schema, {})
    tiers = {s["id"]: s["tier"] for s in ko["sections"]}
    assert tiers == {"system_overview": "core", "cost_optimization": "conditional"}


def test_unmapped_findings_drops_a_reworded_near_duplicate():
    # Real case: the polish pass turned "Tribal knowledge, service bus
    # backlog can temporarily increase during large deployments. Check
    # whether the deployment has completed..." into "...indicates that the
    # service bus backlog ... Verify whether the deployment has
    # completed...". Neither string contains the other, so the substring
    # test missed it and the same fact was published twice -- once in its
    # real section, again as an "unmapped finding".
    ko = {
        "sections": [{
            "id": "known_bad_days", "title": "OPERATIONAL CALENDAR", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
            "coverage_content": [
                "Tribal knowledge indicates that the service bus backlog can temporarily "
                "increase during large deployments. Verify whether the deployment has "
                "completed before treating a short-lived backlog as an incident."
            ],
        }],
        "summary": {},
    }
    sentences = [_sentence(
        "Tribal knowledge, service bus backlog can temporarily increase during large "
        "deployments. Check whether the Deployment has completed before treating a "
        "short-lived backlog as an incident."
    )]
    result = append_unmapped_findings_section(ko, sentences)
    assert not any(s["id"] == "unmapped_findings" for s in result["sections"])


def test_unmapped_findings_keeps_distinct_facts_about_the_same_technology():
    # The overlap rule must not collapse two genuinely different facts that
    # merely share a technology name.
    ko = {
        "sections": [{
            "id": "system_overview", "title": "SYSTEM OVERVIEW", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {},
            "coverage_content": ["Azure Cache for Redis provides caching for the platform."],
        }],
        "summary": {},
    }
    sentences = [_sentence("A previous major incident was caused by Redis memory saturation.")]
    result = append_unmapped_findings_section(ko, sentences)
    unmapped = next(s for s in result["sections"] if s["id"] == "unmapped_findings")
    assert any("memory saturation" in c for c in unmapped["coverage_content"])


def test_architecture_details_excludes_procedural_sentences_from_other_sections():
    # Nearly every sentence in a DevOps KT names some tool, so collecting
    # them all turned Architecture Details into a near-copy of the whole
    # transcript, restating the deployment/monitoring/DR sections verbatim.
    # The COMPONENT list must still see components named anywhere.
    ko = {
        "sections": [{
            "id": "architecture_reference", "title": "ARCHITECTURE REFERENCE", "status": "covered",
            "confidence": 0.8, "risk": 0.0, "facts": [], "entities": [], "evidence": [],
            "relationships": [], "fields": {}, "coverage_content": [],
        }],
        "summary": {},
    }
    section_content = {
        "architecture_reference": {"sentences": [
            {"text": "Azure SQL is the primary database and Azure Service Bus handles async processing."},
        ]},
        "deployment_and_rollback": {"sentences": [
            {"text": "Azure DevOps runs CI, builds the image, and pushes it to ACR."},
        ]},
        "monitoring_observability": {"sentences": [
            {"text": "Monitoring uses Prometheus and Grafana."},
        ]},
    }
    result = enrich_architecture_knowledge(ko, section_content)
    arch = next(s for s in result["sections"] if s["id"] == "architecture_reference")

    details = " ".join(arch.get("_architecture_sentences") or [])
    assert "Azure SQL is the primary database" in details
    assert "pushes it to ACR" not in details, "deployment procedure leaked into Architecture Details"
    assert "Monitoring uses Prometheus" not in details, "monitoring procedure leaked into Architecture Details"

    # ...but every component named anywhere is still inventoried.
    components = [c.lower() for c in arch.get("_architecture_components") or []]
    for expected in ("azure sql", "azure devops", "prometheus", "grafana"):
        assert any(expected in c for c in components), f"component lost: {expected}"


def test_tribal_knowledge_digest_reads_sentences_from_blocks_fallback():
    # section_content[id]['sentences'] can be empty while ['blocks'] holds
    # the real sentences -- the digest used to read only the former and
    # silently lost genuinely tagged tribal knowledge on a live run.
    ko = {"sections": [], "summary": {}}
    section_content = {
        "known_bad_days": {
            "sentences": [],
            "blocks": [{"sentences": [
                {"text": "Tribal knowledge, service bus backlog can temporarily increase during large deployments."},
            ]}],
        }
    }
    result = append_tribal_knowledge_section(ko, section_content)
    tribal = next(s for s in result["sections"] if s["id"] == "tribal_knowledge")
    assert any("service bus backlog" in r["Knowledge"] for r in tribal["_tribal_rows"])


def test_tribal_digest_finds_knowledge_that_exists_only_in_polished_content():
    # A section's raw ['sentences'] and its coverage_content are populated
    # by two independent mechanisms that can disagree, so a sentence
    # explicitly framed as tribal knowledge can exist ONLY in the polished
    # text. Confirmed live: "Tribal knowledge, service bus backlog can
    # temporarily increase during large deployments." reached the document
    # but never the digest that exists to collect exactly that.
    ko = {
        "sections": [_section("known_bad_days", "OPERATIONAL CALENDAR", coverage_content=[
            "- The platform experiences peak activity during month-end periods and major "
            "promotional campaigns; production deployments should be avoided during these times. "
            "- Tribal knowledge indicates that the service bus backlog can temporarily increase "
            "during large deployments. Verify whether the deployment has completed before "
            "treating a short-lived backlog as an incident."
        ])],
        "summary": {},
    }
    result = append_tribal_knowledge_section(ko, {})
    tribal = next(s for s in result["sections"] if s["id"] == "tribal_knowledge")
    knowledge = [r["Knowledge"] for r in tribal["_tribal_rows"]]
    assert any("service bus backlog" in k for k in knowledge)
    # The ordinary calendar sentence in the same blob is not tribal knowledge
    # and must not be swept in alongside it.
    assert not any("month-end periods" in k for k in knowledge)


def test_tribal_digest_does_not_list_a_polished_restatement_twice():
    ko = {
        "sections": [_section("known_bad_days", "OPERATIONAL CALENDAR", coverage_content=[
            "Tribal knowledge indicates that the service bus backlog can temporarily increase "
            "during large deployments. Verify whether the deployment has completed."
        ])],
        "summary": {},
    }
    section_content = {"known_bad_days": {"sentences": [
        {"text": "Tribal knowledge, service bus backlog can temporarily increase during large "
                 "deployments. Check whether the deployment has completed."},
    ]}}
    result = append_tribal_knowledge_section(ko, section_content)
    tribal = next(s for s in result["sections"] if s["id"] == "tribal_knowledge")
    assert len(tribal["_tribal_rows"]) == 1


def test_transcript_sentence_lost_before_rendering_is_recovered_not_silently_dropped():
    # The classifier can drop a sentence without ever reporting it as
    # unassigned, and the polish pass can drop one while rewriting a
    # section. In both cases the sentence is simply gone, and the
    # knowledge-coverage summary cannot see it because that summary counts
    # what the pipeline KNOWS about, not what the transcript said.
    # Confirmed live: a stated danger zone vanished from the document while
    # the summary still reported zero loss.
    transcript = (
        "Production runs on Azure Kubernetes Service. "
        "Production Kubernetes configuration must not be changed manually."
    )
    ko = {
        "sections": [_section("system_overview", "SYSTEM OVERVIEW", coverage_content=[
            "Production runs on Azure Kubernetes Service."
        ])],
        "summary": {},
    }
    result = append_unmapped_findings_section(ko, [], transcript)
    unmapped = next(s for s in result["sections"] if s["id"] == "unmapped_findings")
    assert any("must not be changed manually" in c for c in unmapped["coverage_content"])
    # It is recorded as real evidence, not an invented note.
    assert any("must not be changed manually" in e["text"] for e in unmapped["evidence"])


def test_retention_net_does_not_flag_content_the_document_already_covers():
    # A sentence the polish pass legitimately condensed must not be
    # reported as lost, or every run would end in a wall of false alarms.
    transcript = "Azure SQL is the primary database and Azure Service Bus handles async processing."
    ko = {
        "sections": [_section("architecture_reference", "ARCHITECTURE", coverage_content=[
            "Azure SQL is the primary database. Azure Service Bus handles asynchronous processing."
        ])],
        "summary": {},
    }
    result = append_unmapped_findings_section(ko, [], transcript)
    assert not any(s["id"] == "unmapped_findings" for s in result["sections"])


def test_technology_summary_folds_in_the_architecture_component_inventory():
    # Without this the summary only saw tools named in system_overview's own
    # sentences, so a real Azure KT rendered a "Technology summary" missing
    # its compute, database, cache, messaging and ingress tiers even though
    # all of them were listed in the component inventory right above it.
    ko = {
        "sections": [
            _section("system_overview", "SYSTEM OVERVIEW"),
            _section("architecture_reference", "ARCHITECTURE", **{
                "_architecture_components": [
                    "Angular", "Azure Kubernetes Service", "Azure SQL", "Redis",
                    "Azure Service Bus", "Azure Front Door", ".NET",
                ],
            }),
        ],
        "summary": {},
    }
    result = enrich_technology_summary(ko)
    overview = next(s for s in result["sections"] if s["id"] == "system_overview")
    value = overview["fields"]["key_technologies"]["value"].lower()
    for expected in ("azure kubernetes service", "azure sql", "redis",
                     "azure service bus", "azure front door", ".net"):
        assert expected in value, f"missing from technology summary: {expected}"


# ---------------------------------------------------------------------------
# Fixes from the reviewed Azure Order Processing PDF (REPOSITORY_AUDIT.md §9rr)
# ---------------------------------------------------------------------------

def test_architecture_details_do_not_repeat_the_same_sentence():
    # The list mixes RAW sentences with the polish pass's BULLET BLOBS, which
    # restate the same facts. Exact-match dedup kept both, so a real PDF
    # printed "The frontend uses Angular." twice and showed
    # "...Azure Kubernetes Service, AKS." beside "...Service (AKS)."
    from knowledge.knowledge_builder import enrich_architecture_knowledge

    knowledge_object = {
        "sections": [
            {"id": "architecture_reference", "coverage_content": []},
            {
                "id": "system_overview",
                "coverage_content": [
                    "The frontend uses Angular.",
                    "- The frontend uses Angular. - Infrastructure is provisioned with Bicep.",
                    "Production runs on Azure Kubernetes Service, AKS.",
                    "Production runs on Azure Kubernetes Service (AKS).",
                    "Infrastructure is provisioned with Bicep.",
                ],
            },
        ]
    }
    arch = enrich_architecture_knowledge(knowledge_object, {})["sections"][0]
    sentences = arch.get("_architecture_sentences") or []

    assert sum("frontend uses Angular" in s for s in sentences) == 1
    assert sum("provisioned with Bicep" in s for s in sentences) == 1
    assert sum("Azure Kubernetes Service" in s for s in sentences) == 1


def test_dynamic_fields_are_not_reported_as_knowledge_gaps():
    # A dynamic field exists BECAUSE the transcript named the technology, so
    # counting its absence as a gap is circular. A real PDF reported
    # "missing: Cache Layer" on a KT whose Technology Summary said
    # "Cache | Redis" on the same page.
    from knowledge.knowledge_builder import _leaf_field_specs

    specs = _leaf_field_specs([
        {"id": "business_criticality", "label": "Business Criticality", "type": "single_select"},
        {"id": "orders_per_day", "label": "Business Volume", "type": "text"},
        {"id": "cache_layer", "label": "Cache Layer", "type": "text", "dynamic": True},
    ])
    ids = [fid for fid, _ in specs]

    assert "cache_layer" not in ids
    # Template fields still count -- their absence is genuine news.
    assert "business_criticality" in ids
    assert "orders_per_day" in ids


def test_session_pleasantries_are_not_surfaced_as_unmapped_findings():
    from knowledge.knowledge_builder import _is_session_pleasantry

    assert _is_session_pleasantry("Hi everyone, today I will be handing over the Azure Order Processing Platform.")
    assert _is_session_pleasantry("That concludes the Atlas trading platform handover.")
    assert _is_session_pleasantry("Good morning, this is the handover for the Helios payments platform.")


def test_pleasantry_filter_never_drops_a_real_fact():
    # The filter removes noise; it must never cost a fact. Two false
    # positives were found by probing it against phrasings it was not
    # written for -- a length heuristic dropped a greeting that also stated
    # the business volume, and a bare "That concludes..." threw away a real
    # canary-rollback fact. Substance (a figure or a named technology) is
    # the test, and a closing verb now needs a SESSION noun. See §9rr.
    from knowledge.knowledge_builder import _is_session_pleasantry

    keep = [
        "Hi everyone, the platform processes 120,000 orders per day and runs on AKS.",
        "That concludes the rollback if the canary thresholds are breached.",
        "That concludes the deployment once Flux has synchronized.",
        "Thanks to the platform engineering team who own the AKS infrastructure.",
        "Production Kubernetes configuration must not be changed manually.",
        "Good morning traffic peaks at 30 million events.",
        "This is the end of the retention window for backups.",
    ]
    for sentence in keep:
        assert not _is_session_pleasantry(sentence), sentence


def test_closing_remarks_need_a_session_noun_not_just_the_verb():
    from knowledge.knowledge_builder import _is_session_pleasantry

    assert _is_session_pleasantry("That concludes the handover.")
    assert _is_session_pleasantry("This wraps up the knowledge transfer session.")
    # Same verb, applied to a procedure rather than the session.
    assert not _is_session_pleasantry("That concludes the rollback procedure.")


def test_pleasantry_filter_is_content_driven_not_transcript_specific():
    # Works on wording from unrelated KTs (AWS / GCP / Azure), because it
    # keys on greeting phrases plus substance, never on any one stack.
    from knowledge.knowledge_builder import _is_session_pleasantry

    assert _is_session_pleasantry("Good morning, this is the handover session for today.")
    assert not _is_session_pleasantry("Hello, Memorystore caches session tokens.")
    assert not _is_session_pleasantry("Hi team, Amazon MSK carries settlement events.")
