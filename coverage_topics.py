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

from section_rules import RISING_LOAD_PERIOD

_I = re.IGNORECASE

SECTION_TOPICS: Dict[str, List[Tuple[str, Tuple[str, ...], Optional[str]]]] = {
    "system_overview": [
        ("Purpose", ("what_it_does", "problem_solved", "system_name"),
         r"\b(?:this\s+(?:kt|handover)\s+(?:is\s+for|covers)|handing\s+over|taking\s+over|responsible\s+for|"
         r"handles|processes|sends|lets|allows|serves|takes|authori[sz]es|powers|is\s+(?:the|our)\s+\w+)\b"),
        ("Users and reach", ("users", "customer_reach", "who_affected"),
         r"\b(?:used\s+by|customers?|users|clients|travell?ers|hospitals|businesses|teams?\s+use"
         r"|patients|clinics|stores|shops|branches|members|staff|employees|students|drivers|riders|merchants"
         r"|sellers|buyers|tenants|subscribers|agents)\b"),
        ("Business criticality and impact",
         ("business_criticality", "business_impact", "worst_case", "what_breaks", "business_dependency", "critical_time"),
         r"\b(?:critical|if\s+(?:it|the\s+\w+)\s+(?:is|goes)\s+down|unavailable|outage|depend\w*|revenue|cannot|can't)\b"),
        # Counts ("50,000 orders"), data sizes ("3TB of data") and rates
        # ("... per day").
        ("Volume", ("orders_per_day",),
         r"\b(?:\d[\d,.]*|thousand|million|billion|hundred)\s+(?:\w+\s+)?(?:orders|events|requests|transactions|"
         r"messages|alerts|users|travell?ers|bookings|invoices|calls|businesses|readings|records|jobs|files)\b"
         r"|\b\d[\d,.]*\s*(?:tb|gb|pb|terabytes?|gigabytes?|petabytes?)\b"
         r"|\b(?:processes|handles|serves|ingests|receives)\b[^.]*\b(?:per|a|each|every)\s+(?:second|minute|hour|day|month)\b"
         # A large number per period, said as an answer: "About twelve
         # thousand a day", "8000 bookings a day". Not "3 times a day".
         r"|\b(?:\d{3,}[\d,.]*|\d+(?:k|m)|thousand|million|billion|hundred)\s+(?:\w+\s+){0,2}?"
         r"(?:a|per|each|every)\s+(?:second|minute|hour|day|week|month)\b"),
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
        # Two environments set against each other state a difference without
        # the word: "We have three replicas in staging, but production runs
        # five replicas" was followed by "not discussed: Differences between
        # environments" (a user's KT, 23 Oct).
        ("Differences between environments", ("known_differences",),
         r"\b(?:differ\w*|mirrors?|unlike|whereas|only\s+in|test\s+keys|mocked|smaller|scaled?\s+down)\b"
         r"|\b" + r"(?:prod(?:uction)?|staging|dev(?:elopment)?|test|qa|uat|pre-?prod\w*|sandbox|non-?prod\w*)"
         r"\b[^.;]*?(?:\b(?:but|while|whereas|versus|vs)\b|\bcompared\s+(?:to|with)\b|;)[^.]*?\b"
         r"(?:prod(?:uction)?|staging|dev(?:elopment)?|test|qa|uat|pre-?prod\w*|sandbox|non-?prod\w*)\b"),
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
        ("Backups", ("rpo_steps",),
         r"\bbackups?\b|\bback(?:ed|s)?\s+up\b|\bsnapshots?\b|\bpoint[\s-]in[\s-]time\b|\bretain\w*\b|\bretention\b"),
        # A standby copy, not any replica: "production runs five replicas"
        # (pods) covered failover (a user's KT, 23 Oct).
        ("Failover / standby", ("rto_steps",),
         r"\b(?:fail(?:s|ed|ing)?\s*over|failover|standby|hot\s+spare|geo-?replicat\w*"
         r"|(?:read|standby|database|db|cross-?region|secondary)\s+replicas?"
         r"|replicas?\s+in\s+(?:another|a\s+(?:second|different)|the\s+other)\s+region"
         r"|secondary\s+(?:site|region|data\s*cent(?:er|re)))\b"),
        ("DR testing", ("dr_testing_frequency",),
         r"\b(?:test\w*\s+(?:the\s+)?(?:failover|restores?|recovery|dr)|(?:failover|restore|recovery|dr)\s+(?:test|drill)\w*)\b"),
    ],
    "security_controls": [
        # Where secrets are kept and how they reach the system, not any
        # mention of a secret: "pods fail to start because a Kubernetes
        # secret reference is wrong" (a failure) counted as Security coverage.
        ("Secrets management", ("vault_configuration",),
         r"\b(?:key\s+vault|vault|kms|secrets?\s+manager|secret\s+manager)\b"
         r"|\b(?:secrets?|credentials?|passwords?|api\s+keys?)\s+(?:are\s+|is\s+)?(?:all\s+)?(?:stored|kept|managed|"
         r"rotated|held|injected|loaded|pulled|read|fetched|live|lives|come|comes)\b"
         r"|\b(?:stor\w*|keep\w*|manag\w*|rotat\w*)\s+(?:the\s+|all\s+|our\s+)?(?:secrets?|credentials?)\b"),
        ("Access control", (),
         r"\b(?:access|iam|rbac|least\s+privilege|permissions?|service\s+accounts?|workload\s+identity|sso)\b"),
        ("Security scanning", ("security_scan_config",),
         r"\b(?:scan\w*|trivy|snyk|sonarqube|dependabot|checkov|vulnerab\w*)\b"),
    ],
    "day1_survival_checklist": [
        # Access as something to get, not a name: "the patient access team"
        # marked an undiscussed Day-1 checklist as mentioned elsewhere.
        ("Access and tools to request", ("required_access",),
         r"\b(?:get|request|ask\s+for|need|grant\w*|given|set\s+up)\s+(?:\w+\s+){0,3}(?:access|permissions?|accounts?|"
         r"logins?)\b|\baccess\s+to\b|\b(?:vpn|sso|onboard\w*)\b"),
        # "the first week of January" is a date, not the newcomer's first week.
        ("Safe first actions", ("first_safe_actions",),
         r"\b(?:shadow\w*|read\s+the\s+runbooks?|read-only|observe|first\s+(?:week|day)(?!\s+of\b)|start\s+by|"
         r"review\w*)\b"),
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
         r"\b(?:roll\s?back|revert\w*|redeploy\s+the\s+previous|previous\s+(?:version|revision|task\s+definition)"
         r"|(?:swap|switch|go)\s+back)\b"),
        ("Rollback time and approval", ("rollback_time", "rollback_approval"),
         r"\b(?:roll\s?back|redeploy|revert)\w*\b[^.]*\b(?:minutes?|hours?|takes|approv\w*)\b"),
        ("Repository and pipeline links", ("repo_link", "pipeline_link"), r"https?://|\bwww\."),
    ],
    # Failures named without a fix are half the knowledge, not none of it:
    # a KT listing three failures with no fixes was reported Missing.
    "common_failures": [
        ("Known failures", (),
         r"\b(?:problem|issue|failure|fail(?:s|ed|ing)?|error|outage|incident|backlog|lag|throttl\w*|timeout|crash\w*|"
         r"exhaust\w*|leak\w*|stuck|down|offline|unavailable|reject\w*|hang\w*|stall\w*|drops|dropped"
         r"|backs\s+up\s+when|backed\s+up)\b"),
        ("Fixes or workarounds", (),
         r"\b(?:fix\w*|resolv\w*|workaround|restart\w*|renew\w*|rotat\w*|switch\w*|drain\w*|redeploy\w*|scal\w+"
         r"|clear\w*|reset\w*|purge\w*|flush\w*|spread\w*|retr(?:y|ies|ied)|re-?run\w*|reboot\w*|fail\s*over\w*)\b"),
    ],
    "known_bad_days": [
        # Sales EVENTS, not a sales team: "Finance and sales operations use
        # it" marked an incomplete KT's calendar as discussed.
        # A period, not a rate: "three hundred requests a second at peak"
        # marked an undiscussed calendar as mentioned elsewhere.
        ("High-traffic periods", ("high_traffic_periods",),
         r"\b(?:peak\s+(?:periods?|times?|seasons?|hours?|days?|weeks?|months?|traffic|load|usage)|"
         r"(?:is|are|was|were)\s+(?:the\s+|our\s+)?peak|the\s+peak\s+(?:is|was)|busiest|black\s+friday|(?:flash|summer|winter|holiday|seasonal|big)\s+sales?|"
         r"sales?\s+(?:events?|periods?|seasons?|days?|campaigns?)|holiday\w*|season\w*|"
         r"(?:high|peak)\s+traffic|traffic\s+(?:peaks?|spikes?)|market\s+open|volatility)\b|" + RISING_LOAD_PERIOD),
        # A window that recurs, not a deadline: "By the end of the month you
        # should be doing the releases" (an onboarding milestone) marked an
        # undiscussed calendar as covered under First 30 Days.
        ("Month-end windows", ("month_end_windows",),
         r"(?<!\bby\s)(?<!\bbefore\s)(?<!\buntil\s)(?<!\btill\s)(?<!\bwithin\s)\bmonth[\s-]end\b|"
         r"(?<!\bby\sthe\s)(?<!\bbefore\sthe\s)(?<!\buntil\sthe\s)(?<!\btill\sthe\s)(?<!\bwithin\sthe\s)"
         r"\bend\s+of\s+(?:the\s+|every\s+|each\s+)?month\b|\bdays\s+of\s+the\s+month\b"),
        # "Avoid production deployments during ...": a word may sit between.
        ("Change freezes", ("deployment_blackout_times",),
         r"\b(?:avoid\s+(?:\w+\s+)?(?:deploying|deployments?|releasing|releases?|changes)|no\s+(?:changes|deploys?|deployments)|"
         r"freeze|blackout|never\s+deploy|don't\s+deploy|do\s+not\s+deploy)\b"),
    ],
    "danger_zones": [
        ("Never-do rules", (),
         r"\b(?:never|do\s+not|don't|must\s+not|avoid|danger\w*|careful|risky|do\s+not\s+touch|sensitive)\b"),
    ],
    "ownership_escalation": [
        ("Owning teams", ("application_ownership", "infrastructure_ownership"), r"\bowns?\b|\bresponsible\s+for\b"),
        # Said without the word too: "if they do not answer in fifteen
        # minutes, call Maria, the engineering manager" was reported as an
        # escalation path not discussed, below that very sentence.
        ("Escalation path", ("escalation_chain",),
         r"\bescalat\w*|\b(?:second|third)\s+(?:line|tier)\b|"
         r"\bif\s+(?:(?:they|he|she|the\s+(?:on-?call|primary)(?:\s+(?:engineer|person))?)\s+(?:does\s+not|do\s+not|"
         r"doesn'?t|don'?t|did\s+not|didn'?t|fails?\s+to|can'?t|cannot)|nobody|no\s+one|no-one)\s+"
         r"(?:answer|respond|reply|pick\s+up|acknowledge|ack)s?\b|"
         r"\b(?:call|contact|ring)(?!\s+cent(?:re|er)\b)\s+(?:the\s+)?(?:[\w'-]+,?\s+){0,4}?"
         r"(?:manager|lead|head\s+of\s+[\w-]+|director|duty\s+manager)\b"),
        ("On-call tool", ("oncall_tool",), r"\b(?:pagerduty|opsgenie|on-?call)\b"),
    ],
    "first_30_day_ownership": [
        # "In the first two weeks read the last three incident reviews" and
        # "for the first few days" are the plan's first weeks.
        ("Week 1", ("week1",),
         r"\b(?:first\s+(?:week(?!\s+of\b)|(?:two|2|couple\s+of|few)\s+weeks|(?:two|2|three|3|few|couple\s+of)\s+days)"
         r"|week\s+(?:one|1))\b"),
        ("Week 2", ("week2",), r"\b(?:second\s+week|week\s+(?:two|2)|first\s+(?:two|2|couple\s+of)\s+weeks)\b"),
        ("Week 3", ("week3",), r"\b(?:third\s+week|week\s+(?:three|3))\b"),
        # The plan's last week, not a past one: "we upgraded it last week"
        # covered Week 4 (a user's KT, 23 Oct).
        ("Week 4", ("week4",),
         r"\b(?:fourth\s+week|(?:in|during|by)\s+(?:the\s+|your\s+)?last\s+week(?!\s+of\b)|week\s+(?:four|4)|"
         r"end\s+of\s+the\s+(?:first\s+)?month|day\s+30)\b"),
    ],
    "open_responsibilities": [
        ("Open work", ("open_tasks",),
         # A migration under way, not any mention of one: "the last database
         # migration might be hard to reverse" (a warning) was listed as open work.
         r"\b(?:not\s+yet|still\s+open|pending|in\s+progress|no\s+owner|nobody\s+has|unfinished)\b"
         r"|\b(?:ongoing|planned|pending|unfinished|half[\s-]done|in[\s-]flight)\s+(?:\w+\s+)?migration\b"
         r"|\bmigrat(?:e|es|ing|ion)\s+(?:off|away\s+from)\b|\bmigrating\s+(?:to|from)\b"
         r"|\bmigration\s+(?:is|was)\s+(?:not\s+(?:yet\s+)?(?:done|finished|complete)|still|half|planned|scheduled)\b"),
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

# A topic stated under another section counts only when that sentence names
# the topic's subject. Inside its own section the subject is understood ("Only
# on Tuesday mornings."); elsewhere "never restart the production database
# during business hours unless the incident commander approves it" (a
# never-do rule) covered the deployment window and its approvals.
_DEPLOY_CONTEXT = re.compile(
    r"\b(?:deploy\w*|releas\w*|rollouts?|roll\s+out|ship\w*|go-?live|merg\w*|pipelines?|production\s+changes?|"
    r"changes?\s+(?:to|in)\s+production|change\s+(?:freeze|window))\b", _I)
_ELSEWHERE_CONTEXT = {
    ("deployment_and_rollback", "Trigger and approvals"): _DEPLOY_CONTEXT,
    ("deployment_and_rollback", "Deployment window"): _DEPLOY_CONTEXT,
}


def states_topic_elsewhere(section_id: str, label: str, rx: Optional["re.Pattern[str]"], text: str) -> bool:
    """`text`, said under another section, states this section's topic."""
    if rx is None or not rx.search(text):
        return False
    context = _ELSEWHERE_CONTEXT.get((section_id, label))
    return context is None or bool(context.search(text))


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
            where = next((sid for sid, t in other_texts if states_topic_elsewhere(section_id, label, rx, t)), None)
            out.append((label, "elsewhere", where) if where else (label, "missing", None))
        else:
            out.append((label, "missing", None))
    return out
