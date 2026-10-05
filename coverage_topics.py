"""What a KT section is expected to cover, and how to tell it was (P0-8).

The coverage matrix used to count filled template fields, so it reported
topics as missing whenever a field stayed empty, even when the section right
above it stated the fact ("Operational Calendar: 0 of 3 captured; missing:
deployment_blackout_times" under "Avoid deploying on Fridays"). Coverage is
now judged per topic: captured as a field, discussed in the section's text,
discussed under another section, or genuinely not discussed. Only the last
is reported as a gap.

Each topic: (label, field ids that capture it, evidence pattern). A topic
with no pattern can only be satisfied by its fields.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_I = re.IGNORECASE

SECTION_TOPICS: Dict[str, List[Tuple[str, Tuple[str, ...], Optional[str]]]] = {
    "system_overview": [
        ("Purpose", ("what_it_does", "problem_solved", "system_name"),
         r"\b(?:this\s+(?:kt|handover)\s+(?:is\s+for|covers)|handing\s+over|taking\s+over|responsible\s+for|"
         r"handles|processes|sends|lets|allows|is\s+(?:the|our)\s+\w+)\b"),
        ("Users and reach", ("users", "customer_reach", "who_affected"),
         r"\b(?:used\s+by|customers?|users|clients|travell?ers|hospitals|businesses|teams?\s+use)\b"),
        ("Business criticality and impact",
         ("business_criticality", "business_impact", "worst_case", "what_breaks", "business_dependency", "critical_time"),
         r"\b(?:critical|if\s+(?:it|the\s+\w+)\s+(?:is|goes)\s+down|unavailable|outage|depend\w*|revenue|cannot|can't)\b"),
        ("Volume", ("orders_per_day",),
         r"\b(?:\d[\d,.]*|thousand|million|billion|hundred)\s+(?:\w+\s+)?(?:orders|events|requests|transactions|"
         r"messages|alerts|users|travell?ers|bookings|invoices|calls|businesses)\b"),
        ("Technology stack", ("key_technologies",), None),
    ],
    "architecture_reference": [
        ("Architecture documentation link", ("architecture_link",), r"https?://|\bwww\."),
        ("Last reviewed date", ("last_updated",), r"\b(?:last\s+updated|updated\s+(?:in|on)|reviewed\s+(?:in|on))\b"),
    ],
    "environments": [
        ("Production", ("production_notes",), r"\bprod(?:uction)?\b"),
        ("Staging / pre-production", ("staging_notes",), r"\b(?:staging|pre-?prod\w*|uat)\b"),
        ("Development / test", ("non_production_notes",), r"\b(?:dev|development|test|qa|sandbox|non-?prod\w*)\b"),
        ("Differences between environments", ("known_differences",),
         r"\b(?:differ\w*|mirrors?|unlike|whereas|only\s+in|test\s+keys|mocked|smaller|scaled?\s+down)\b"),
    ],
    "monitoring_observability": [
        ("Monitoring tools", ("tools",),
         r"\b(?:datadog|grafana|prometheus|cloudwatch|new\s+relic|splunk|azure\s+monitor|application\s+insights|"
         r"cloud\s+monitoring|dynatrace|kibana|opensearch|elk|monitoring\s+tool)\b"),
        ("Key alerts and dashboards", ("first_response_steps",), r"\b(?:alert\w*|dashboards?|thresholds?|slos?|slis?)\b"),
        ("Paging and alert routing", ("alert_routing",), r"\b(?:pages?|paging|pagerduty|opsgenie|on-?call|notif\w*)\b"),
    ],
    "disaster_recovery": [
        ("RTO", ("rto_metric",), r"\brto\b|\brecovery\s+time\b"),
        ("RPO", ("rpo_metric",), r"\brpo\b|\brecovery\s+point\b"),
        ("Backups", ("rpo_steps",), r"\bbackups?\b|\bsnapshots?\b|\bpoint[\s-]in[\s-]time\b"),
        ("Failover / standby", ("rto_steps",), r"\b(?:failover|standby|replica\w*|geo-?replicat\w*)\b"),
        ("DR testing", ("dr_testing_frequency",),
         r"\b(?:test\w*\s+(?:the\s+)?(?:failover|restores?|recovery|dr)|(?:failover|restore|recovery|dr)\s+(?:test|drill)\w*)\b"),
    ],
    "security_controls": [
        ("Secrets management", ("vault_configuration",), r"\b(?:secrets?|key\s+vault|vault|kms|credentials?)\b"),
        ("Access control", (),
         r"\b(?:access|iam|rbac|least\s+privilege|permissions?|service\s+accounts?|workload\s+identity|sso)\b"),
        ("Security scanning", ("security_scan_config",),
         r"\b(?:scan\w*|trivy|snyk|sonarqube|dependabot|checkov|vulnerab\w*)\b"),
    ],
    "day1_survival_checklist": [
        ("Access and tools to request", ("required_access",), r"\b(?:access|permissions?|accounts?|vpn|sso|onboard\w*)\b"),
        ("Safe first actions", ("first_safe_actions",),
         r"\b(?:shadow\w*|read\s+the\s+runbooks?|read-only|observe|first\s+(?:week|day)|start\s+by)\b"),
        ("Actions to avoid at first", ("actions_not_to_perform",),
         r"\b(?:don't|do\s+not|never|avoid)\b[^.]*\b(?:first|initially|until|yet)\b"),
    ],
    "deployment_and_rollback": [
        ("Deployment process", ("deployment_steps",), r"\b(?:deploy\w*|release\w*|pipelines?|builds?)\b"),
        ("Trigger and approvals", ("trigger_type", "pre_deployment_checks"),
         r"\b(?:on\s+merge|merge\w*|automatic\w*|manual\w*|approv\w*|scheduled|nightly)\b"),
        ("Deployment window", ("deployment_window",),
         r"\b(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|weekends?|window|freeze|blackout|mornings?|"
         r"evenings?|business\s+hours)\b"),
        ("Rollback procedure", ("rollback_scenarios", "rollback_trigger"),
         r"\b(?:roll\s?back|revert\w*|redeploy\s+the\s+previous|previous\s+(?:version|revision|task\s+definition))\b"),
        ("Rollback time and approval", ("rollback_time", "rollback_approval"),
         r"\b(?:roll\s?back|redeploy|revert)\w*\b[^.]*\b(?:minutes?|hours?|takes|approv\w*)\b"),
        ("Repository and pipeline links", ("repo_link", "pipeline_link"), r"https?://|\bwww\."),
    ],
    "common_failures": [
        ("Known failures with fixes", (),
         r"\b(?:fix\w*|resolv\w*|workaround|restart\w*|renew\w*|rotat\w*|switch\w*|drain\w*|redeploy\w*|scal\w+)\b"),
    ],
    "known_bad_days": [
        ("High-traffic periods", ("high_traffic_periods",),
         r"\b(?:peak|busiest|black\s+friday|sales?|holiday\w*|season\w*|traffic|market\s+open|volatility)\b"),
        ("Month-end windows", ("month_end_windows",), r"\bmonth[\s-]end\b|\bend\s+of\s+(?:the\s+)?month\b|\bdays\s+of\s+the\s+month\b"),
        ("Change freezes", ("deployment_blackout_times",),
         r"\b(?:avoid\s+deploying|no\s+(?:changes|deploys?|deployments)|freeze|blackout|never\s+deploy|don't\s+deploy|do\s+not\s+deploy)\b"),
    ],
    "danger_zones": [
        ("Never-do rules", (), r"\b(?:never|do\s+not|don't|must\s+not|avoid)\b"),
    ],
    "ownership_escalation": [
        ("Owning teams", ("application_ownership", "infrastructure_ownership"), r"\bowns?\b|\bresponsible\s+for\b"),
        ("Escalation path", ("escalation_chain",), r"\bescalat\w*"),
        ("On-call tool", ("oncall_tool",), r"\b(?:pagerduty|opsgenie|on-?call)\b"),
    ],
    "first_30_day_ownership": [
        ("Week 1", ("week1",), r"\b(?:first\s+week|week\s+(?:one|1))\b"),
        ("Week 2", ("week2",), r"\b(?:second\s+week|week\s+(?:two|2))\b"),
        ("Week 3", ("week3",), r"\b(?:third\s+week|week\s+(?:three|3))\b"),
        ("Week 4", ("week4",), r"\b(?:fourth\s+week|last\s+week|week\s+(?:four|4)|end\s+of\s+the\s+(?:first\s+)?month|day\s+30)\b"),
    ],
    "open_responsibilities": [
        ("Open work", ("open_tasks",),
         r"\b(?:not\s+yet|still\s+open|pending|in\s+progress|no\s+owner|nobody\s+has|unfinished|migration)\b"),
        ("Recurring duties", ("recurring_responsibilities",),
         r"\b(?:every\s+(?:week|month|day|quarter)|weekly|monthly|recurring|renew\w*|rotat\w*)\b"),
    ],
    # Readiness is confirmed, not "discussed": only captured fields count.
    "handover_completion": [
        ("Readiness checks confirmed",
         ("can_deploy", "understands_rollback", "knows_danger_zones", "escalation_clear", "architecture_verified"), None),
    ],
    "cost_optimization": [
        ("Cost levers", ("levers",), r"\b(?:cost\w*|sav(?:e|ing)\w*|spend|budget|scale\w*\s+(?:down|to\s+zero)|spot|reserved)\b"),
    ],
}

# Sections recorded after the session, never a gap in the session itself.
AFTER_REVIEW_SECTIONS = {"signoff"}

_COMPILED = {sid: [(label, fields, re.compile(rx, _I) if rx else None) for label, fields, rx in topics]
             for sid, topics in SECTION_TOPICS.items()}


def _filled(field_objects: Dict[str, Any], fid: str) -> bool:
    entry = field_objects.get(fid)
    if isinstance(entry, dict) and "value" in entry:
        return entry.get("source", "unfilled") != "unfilled" and entry.get("value") not in (None, "", [], {})
    if isinstance(entry, dict):  # a group: any filled child counts
        return any(_filled(entry, k) for k in entry)
    return False


def assess_topics(section_id: str, field_objects: Dict[str, Any], own_texts: Sequence[str],
                  other_texts: Sequence[Tuple[str, str]], structured: Optional[Dict[str, Any]] = None
                  ) -> Optional[List[Tuple[str, str, Optional[str]]]]:
    """[(topic label, state, where)] with state one of captured / discussed /
    elsewhere / missing; `where` is the other section's id for "elsewhere".
    None when the section has no topic checklist."""
    topics = _COMPILED.get(section_id)
    if topics is None:
        return None
    structured = structured or {}
    out: List[Tuple[str, str, Optional[str]]] = []
    for label, fields, rx in topics:
        if any(_filled(field_objects, f) or structured.get(f) for f in fields):
            out.append((label, "captured", None))
        elif rx is not None and any(rx.search(t) for t in own_texts):
            out.append((label, "discussed", None))
        elif rx is not None and section_id != "handover_completion":
            where = next((sid for sid, t in other_texts if rx.search(t)), None)
            out.append((label, "elsewhere", where) if where else (label, "missing", None))
        else:
            out.append((label, "missing", None))
    return out
