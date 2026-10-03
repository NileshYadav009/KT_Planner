"""
Deterministic section routing rules for KT classification.

High-priority phrase and pattern rules override weak semantic matches,
especially preventing the system_overview section from absorbing specialized content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SectionRuleMatch:
    section_id: str
    confidence: float
    reason: str
    matched_pattern: str


# (section_id, compiled_patterns, base_confidence)
# Order matters: earlier specialized rules win ties.
SECTION_RULES: List[Tuple[str, List[str], float]] = [
    (
        "danger_zones",
        [
            r"\bnever\s+modify\b",
            r"\bdangerous\s+area\b",
            r"\bdo\s+not\s+touch\b",
            r"\bterraform\s+state\s+files?\s+manually\b",
            r"\bautoscaler\s+configuration\b",
            r"\brecovering\s+from\s+state\s+corruption\b",
            # The section's own name spoken explicitly ("The main danger
            # zones are X, Y, Z") is the single strongest possible signal,
            # yet was missing from this list entirely — confirmed on a real
            # Azure Banking transcript where that exact sentence fell
            # through to a different section, silently dropping 2 of its 3
            # named danger zones (Service Bus retention/dead-letter config,
            # manual Kubernetes changes outside GitOps) from the rendered
            # Danger Zones section even though they were said in the same
            # breath as the one item that did get classified correctly.
            # ...but only when the sentence NAMES danger zones. A sentence
            # that merely refers to them ("the handover is complete once the
            # engineer has reviewed the danger zones") is not one, and the
            # bare phrase at 0.97 pulled such handover criteria in here.
            r"\bdanger\s+zones?\s+(?:are|is|include[sd]?|would\s+be)\b",
            r"\b(?:main|key|biggest|major|other|another|first|second)\s+danger\s+zones?\b",
            r"\bsensitive\s+area\b",
            r"\brequir(?:es|ing)\s+caution\b",
            # Generic prohibition phrasings. Without these, a plainly-stated
            # prohibition like "Production Kubernetes configuration must not
            # be changed manually." matched no rule at all and depended
            # entirely on embedding similarity to reach Danger Zones — on a
            # live run it reached nothing and vanished from the document
            # altogether. A silently dropped prohibition is the worst
            # failure this document can have, so the phrasings that
            # introduce one are matched deterministically.
            r"\bmust\s+not\s+be\s+(?:changed|modified|edited|deleted|touched|altered)\b",
            r"\bshould\s+not\s+be\s+(?:changed|modified|edited|deleted|touched|altered)\b",
            r"\bdo\s+not\s+(?:ever\s+|manually\s+)?(?:modify|change|edit|delete|touch|alter|run)\b",
            # "Do not assume a healthy health check means the path is healthy"
            # is a trap warning of the same kind.
            r"\bdo\s+not\s+assume\b",
            r"\bnever\s+(?:manually\s+)?(?:change|edit|delete|touch|alter|run)\b",
        ],
        0.97,
    ),
    (
        "ownership_escalation",
        [
            r"\bdevelopers?\s+own\b",
            r"\bplatform\s+engineering(?:\s+team)?\s+owns?\b",
            # Any "<the X team> owns ..." statement, not only the two team
            # names a past transcript happened to use.
            r"\b(?:team|group|squad|engineers?)\s+(?:is\s+responsible\s+for|owns?)\b",
            r"\bescalation\s+(?:path|chain|point|goes|matrix)\b",
            r"\bescalated?\s+to\b",
            r"\bdesklation\s+path\b",
            # A role NAMED in a sentence is not evidence the sentence is about
            # ownership: "the on-call engineer still needs to understand which
            # services have automated rollback" is deployment knowledge, and
            # "deployments are frozen unless the platform engineering manager
            # approves" is a calendar rule. Bare role mentions forced both
            # into Ownership at 0.97, above every other signal. A role only
            # routes here together with ownership/escalation language.
            # "Alerts page the on-call engineer" is alert routing (Monitoring),
            # so paging is not ownership language.
            r"\b(?:on-?call\s+engineer|head\s+of\s+engineering|platform\s+engineering\s+manager|engineering\s+director)\b"
            r".{0,80}\b(?:escalat\w*|owns?|responsible|contact|involve)\b",
            r"\b(?:escalat\w*|owns?|responsible|contact)\b.{0,80}"
            r"\b(?:on-?call\s+engineer|head\s+of\s+engineering|platform\s+engineering\s+manager|engineering\s+director)\b",
        ],
        0.97,
    ),
    (
        "common_failures",
        [
            r"\bmost\s+common\s+production\s+issue\b",
            r"\banother\s+common\s+issue\b",
            r"\bconnection\s+pool\s+exhaustion\b",
            r"\bmissing\s+kubernetes\s+secrets?\b",
            r"\bredis\s+memory\s+saturation\b",
            r"\bmajor\s+outage\b",
            r"\bpod(?:s)?\s+fail(?:s|ed)?\s+immediately\s+after\s+deployment\b",
        ],
        0.96,
    ),
    (
        "known_bad_days",
        [
            r"\bblack\s+friday\b",
            r"\bmonth-?end\s+sales\b",
            r"\bpromotional\s+campaigns?\b",
            r"\bdeployments?\s+must\s+be\s+avoided\b",
            r"\bavoid\s+deployments?\s+during\b",
            r"\bhigh[\s-]?traffic\s+periods?\b",
            # Change freezes are calendar rules regardless of who can grant
            # an exception to them.
            r"\bdeployments?\s+(?:are|is)\s+(?:also\s+)?frozen\b",
            r"\b(?:deployment|change|code|release)\s+freezes?\b",
        ],
        0.96,
    ),
    (
        # Days and periods to stay away from, in general form: "avoid
        # deploying on Fridays or during the summer sale in July" went to
        # Danger Zones (a prohibition, but a calendar one).
        "known_bad_days",
        [
            r"\bavoid\s+(?:deploying|releasing|deployments?|releases?|changes)\b",
            r"\b(?:no|never|don'?t|do\s+not)\s+(?:deploy|release|ship)\w*\s+(?:on|during|over)\s+"
            r"(?:fridays?|weekends?|holidays?|the\s+\w+\s+(?:sale|season|peak|close))\b",
        ],
        0.975,
    ),
    (
        "deployment_and_rollback",
        [
            r"\bmerged\s+into\s+the\s+main\s+branch\b",
            r"\bgithub\s+actions?\b",
            r"\bci/?cd\b",
            r"\bcsed\s+pipelines?\b",
            r"\bhelm\s+roll[\s-]?back\b",
            r"\broll[\s-]?back\b",
            r"\bcompleted\s+within\s+\d+\s+minutes\b",
            r"\bargocd\b",
            r"\bargos\s+cd\b",
            r"\bgitops\b",
        ],
        0.94,
    ),
    (
        # A stated deployment window is the Deployment section's own field,
        # even when the same statement goes on to list freeze periods (which
        # the known_bad_days freeze rule would otherwise claim at 0.96).
        "deployment_and_rollback",
        [
            r"\b(?:normal\s+)?(?:production\s+)?deployment\s+window\s+is\b",
            r"\bcanary\s+(?:rollouts?|releases?|deployments?)\b",
            r"\bprogressive\s+(?:rollout|delivery)\b",
            r"\bautomated\s+rollback\b",
            # Rollback procedure language: restoring a previous good release
            # or halting a rollout. These were scored into Handover Completion
            # (whose checklist mentions "understands rollback").
            r"\b(?:previous|last)\s+(?:known[\s-]good\s+)?(?:\w+\s+)?(?:release|version|revision)\s+can\s+be\s+(?:restored|redeployed|rolled\s+back)\b",
            r"\brollout\s+can\s+be\s+(?:stopped|paused|aborted|halted)\b",
            r"\breverted\s+in\s+git\b",
            r"\bredeploy\s+the\s+(?:previous|last)\b",
            r"\bif\s+a\s+(?:release|deploy(?:ment)?|rollout)\s+(?:misbehaves|fails|breaks|goes\s+wrong|goes\s+bad)\b",
        ],
        0.965,
    ),
    (
        "disaster_recovery",
        [
            r"\bdisaster\s+recovery\b",
            r"\brds\s+snapshots?\b",
            r"\brestoring\s+rds\b",
            r"\bdaily\s+backups?\b",
            r"\bdr\s+testing\b",
            r"\bdisaster\s+recovery\s+testing\b",
            r"\bquarterly\b",
            r"\b30[\s-]?day\s+retention\b",
            r"\bretained\s+for\s+30\s+days\b",
            r"\binfrastructure\s+as\s+code\b",
            r"\binfrastructure\s+as\s+cold\b",
            # Recovery objectives and mechanisms in general form.
            r"\b(?:rto|rpo)\s+(?:is|of|target|was)\b",
            r"\brecovery\s+(?:time|point)\s+objectives?\b",
            r"\bpoint[\s-]in[\s-]time\s+recovery\b",
            r"\bbackups?\s+(?:use|are|run|is|get|taken|stored)\b",
            r"\b(?:warm|hot|cold)\s+standby\b",
        ],
        0.95,
    ),
    (
        "monitoring_observability",
        [
            r"\bprometheus\b",
            r"\bgrafana\s+and\s+cloudwatch\b",
            r"\bmonitoring\s+stack\b",
            r"\bpagerduty\s+alert\b",
            r"\bfirst\s+thing\s+you\s+should\s+check\s+is\s+grafana\b",
            r"\bwe\s+use\s+prometheus\b",
            r"\bwe\s+monitor\b",
            r"\bmonitor\s+everything\b",
            r"\bgrafana\s+dashboards?\s+with\b",
            r"\breal[\s-]?time\s+metrics\b",
        ],
        0.95,
    ),
    (
        # Deliberately does NOT include "amazon ecr" / "container images" --
        # those used to be here to match one synthetic test transcript where
        # a container-registry mention happened to share a sentence with a
        # security-scanning statement ("...using Trevi and container images
        # are stored in Amazon ECR."), which the "trevi" pattern above
        # already catches on its own. As a general rule they were far too
        # broad: any real transcript's plain architecture fact ("Amazon ECR
        # stores container images.") force-matched into security_controls
        # with 0.95 confidence, overriding the classifier's own (correct)
        # judgment that it belongs in architecture_reference -- confirmed as
        # the root cause of that exact misclassification in 2 of 3 real
        # audited transcripts (AWS's ECR mention, Azure's equivalent "Container
        # images are stored in Azure Container Registry").
        "security_controls",
        [
            r"\bsecurity\s+scanning\b",
            r"\btrivy\b",
            r"\btrevi\b",
            r"\bvault\s+for\s+secret\b",
            r"\bsecret\s+management\b",
        ],
        0.95,
    ),
    (
        "cost_optimization",
        [
            r"\bcost\s+optimization\b",
            r"\bspot\s+instances?\b",
            r"\bscheduled\s+scaling\b",
            r"\blow[\s-]?traffic\s+periods?\b",
            r"\bload\s+traffic\s+periods?\b",
            r"\bnon-?production\s+environments?\b",
        ],
        0.94,
    ),
    (
        "day1_survival_checklist",
        [
            r"\bfor\s+new\s+team\s+members\b",
            # Singular/article forms and explicit first-day framing. Only the
            # exact plural "for new team members" used to match, so "For a new
            # team member, the first day should include access to Grafana,
            # PagerDuty, GitHub, ArgoCD..." fell to the ArgoCD deployment rule
            # and Day-1 was reported as not covered.
            r"\bfor\s+(?:a|any|every|each)\s+new\s+(?:team\s+member|engineer|joiner|hire)\b",
            r"\b(?:the\s+)?first\s+day\s+(?:should|must|will|needs?\s+to)\b",
            r"\bon\s+(?:their|your|the)\s+first\s+day\b",
            r"\bstart\s+by\s+reviewing\b",
            r"\bfirst\s+safe\s+actions?\b",
            r"\brequired\s+access\b",
            r"\bday[\s-]?1\b",
            r"\bfor\s+new\s+team\s+members\b.*\bgrafana\s+dashboards?\b",
            r"\bkubernetes\s+namespaces?\b.*\bdeployment\s+pipelines?\b",
        ],
        # Above the 0.94 tool-mention rules (ArgoCD, GitHub Actions): a
        # first-day checklist names those tools as things to get access to.
        0.95,
    ),
    (
        "first_30_day_ownership",
        [
            r"\bweek\s+[1one]\b",
            r"\bfirst\s+(?:30|thirty)\s+days?\b",
            r"\b30[\s-]?day\s+plan\b",
            r"\bobserve\s+(?:and\s+)?shadow\b",
            r"\bindependent\s+ownership\b",
        ],
        0.94,
    ),
    (
        "handover_completion",
        [
            r"\bhandover\s+(?:is\s+)?complete\b",
            r"\breplacement\s+(?:can|should|will)\b",
            r"\bsign[\s-]?off\b",
            r"\bKT\s+(?:is\s+)?(?:done|complete|finished)\b",
            r"\bthis\s+(?:concludes|completes)\b",
            # Readiness evidence: what the incoming owner has actually done
            # or confirmed. Outranks a passing tool mention ("...access to
            # Grafana, ArgoCD... has been granted") that would otherwise pull
            # the statement into Deployment via the ArgoCD rule.
            r"\bincoming\s+(?:owner|engineer)\s+has\s+(?:also\s+)?(?:successfully\s+)?"
            r"(?:confirmed|completed|reviewed|accepted|demonstrated|walked|verified)\b",
            r"\b(?:incoming\s+and\s+outgoing|outgoing\s+and\s+incoming)\s+owners?\b",
            r"\bexercise\s+has\s+(?:also\s+)?been\s+completed\b",
            r"\bhandover\s+checklist\b",
        ],
        0.95,
    ),
    (
        # The completion criteria themselves. They name the escalation path
        # and danger zones as things to have reviewed, so without a higher
        # priority the 0.97 ownership/danger rules claimed them.
        "handover_completion",
        [
            r"\bhandover\s+is\s+(?:only\s+)?considered\s+complete\b",
            r"\bconsidered\s+complete\s+only\s+when\b",
        ],
        0.975,
    ),
    (
        "architecture_reference",
        [
            r"\barchitecture\s+diagram\b",
            r"\bmaintained\s+in\s+confluence\b",
            r"\btraffic\s+enters\s+through\b",
            r"\bapplication\s+load\s+balancer\b",
            r"\bamazon\s+eks\b",
            r"\bcloud\s*front\b",
            r"\bcoming\s+back\s+to\s+architecture\b",
            r"\bprimary\s+database\s+runs\s+on\b",
            r"\bpostgres(?:ql)?\b.*\bmulti[\s-]?az\b",
        ],
        0.90,
    ),
    (
        "open_responsibilities",
        [
            r"\bcontact\s+platform\s+engineering\s+before\b",
            r"\bunsure\s+about\s+.*production\b",
            r"\bunaware\s+about\s+.*production\b",
        ],
        0.93,
    ),
    (
        "system_overview",
        [
            r"\bhanding\s+over\s+the\b",
            r"\b50[,\s]?000\s+orders\b",
            r"\bbusiness[\s-]?critical\b",
            # Stated criticality and outage impact, in general form rather
            # than the one transcript's wording above.
            r"\bmost\s+(?:business[\s-]+)?critical\s+(?:platforms?|systems?|services?|applications?)\b",
            r"\bmission[\s-]critical\b",
            r"\bif\s+the\s+(?:platform|system|service|application)\s+(?:is|goes|was|were|becomes)\s+"
            r"(?:unavailable|down|offline)\b",
            r"\breact\s+front[\s-]?end\b",
            r"\bfast\s*api\b",
            r"\bplatform\s+consists\s+of\b",
            r"\bcustomers?\s+cannot\s+place\s+orders\b",
            r"\bwarehouse\s+fulfillment\b",
            r"\bserves\s+customers?\s+globally\b",
            r"\bpayments?,?\s+inventory\s+updates?\b",
            r"\bshipment\s+orchestration\b",
            r"\bwe\s+use\s+terraform\b.*\b(?:argocd|argos\s+cd|gitops)\b",
            r"\binfrastructure\s+provisioning\b",
            r"\bterraform\b.*\bargocd\b.*\bvault\b",
        ],
        0.97,
    ),
    (
        # Generic phrasings. Every earlier monitoring pattern names one tool
        # or one transcript's wording ("grafana and cloudwatch"), so
        # "Monitoring is Prometheus with Grafana dashboards, and alerts page
        # the on-call engineer" went to Ownership on a role mention.
        "monitoring_observability",
        [
            r"^\W*(?:our\s+|the\s+)?(?:monitoring|observability|alerting)\s+(?:is|are|uses|runs|stack|setup|covers)\b",
            r"\balerts?\s+(?:page|pages|paging|are\s+routed|route|go\s+to|fire|trigger)\b",
            r"\b(?:key|main|critical|most\s+important|primary)\s+(?:alert|metric|dashboard|slo|sli)s?\b",
        ],
        0.97,
    ),
    (
        # Unfinished work with no owner. "Still open is the TLS certificate
        # renewal ... has no owner yet" was scored into Ownership by
        # similarity to "owner", and Open Responsibilities reported "not
        # covered".
        "open_responsibilities",
        [
            r"\bstill\s+(?:open|outstanding|pending|unresolved|to\s+be\s+done)\b",
            r"\b(?:has|have)\s+no\s+(?:clear\s+|named\s+)?owner\b",
            r"\bno\s+owner\s+(?:yet|assigned)\b",
            r"\bunowned\b",
            r"\b(?:open|outstanding|pending|unresolved)\s+(?:items?|tasks?|actions?|work|issues?|questions?)\b",
            r"\bnot\s+yet\s+(?:been\s+)?(?:assigned|resolved|completed|done|migrated|fixed)\b",
        ],
        0.975,
    ),
    (
        # How a KT introduces its subject, in general form.
        "system_overview",
        [
            r"\b(?:this|the|today's)\s+(?:kt|handover|knowledge\s+transfer|session|walkthrough)\s+(?:is\s+)?(?:for|about|on|covers)\b",
            r"\b(?:i'?m|i\s+am|we'?re|we\s+are|i'?ll\s+be|i\s+will\s+be)\s+(?:going\s+to\s+be\s+)?handing\s+over\b",
            r"\bwelcome\s+to\s+(?:the\s+)?[\w-]+\s+(?:kt|handover|knowledge\s+transfer|session|walkthrough)\b",
            r"\b(?:service|platform|system|application)\s+(?:used|relied\s+on|depended\s+on)\s+by\b",
            r"\b(?:\w+\s+){0,3}(?:depend|rely|relies)\s+on\s+(?:it|this\s+(?:service|platform|system))\b",
        ],
        0.97,
    ),
    (
        "environments",
        [
            r"\bstaging\s+environment\b.*\bmirrors?\s+production\b",
            r"\bpayment\s+integrations?\s+are\s+mocked\b",
        ],
        0.97,
    ),
]

# Content that must NOT remain in system_overview when matched.
OVERVIEW_MISPLACEMENT_PATTERNS: List[Tuple[str, str]] = []
for section_id, patterns, confidence in SECTION_RULES:
    if section_id == "system_overview":
        continue
    for pattern in patterns:
        OVERVIEW_MISPLACEMENT_PATTERNS.append((section_id, pattern))

_COMPILED_RULES: List[Tuple[str, List[re.Pattern], float]] = [
    (section_id, [re.compile(p, re.IGNORECASE) for p in patterns], confidence)
    for section_id, patterns, confidence in SECTION_RULES
]

_COMPILED_OVERVIEW_MISPLACEMENT: List[Tuple[str, re.Pattern]] = [
    (section_id, re.compile(pattern, re.IGNORECASE))
    for section_id, pattern in OVERVIEW_MISPLACEMENT_PATTERNS
]


def match_section_rules(text: str) -> Optional[SectionRuleMatch]:
    """Return the best rule-based section match for a sentence, if any."""
    if not text or not text.strip():
        return None

    # Ranked by (confidence, number of this section's patterns that fired,
    # earliest match position). Ties used to go to whichever rule happened
    # to be listed first, so "During the first thirty days, the engineer
    # should ... perform a supervised rollback" went to Deployment (a later
    # passing mention) instead of the 30-day plan the sentence opens with.
    best: Optional[SectionRuleMatch] = None
    best_key = None
    for section_id, patterns, base_confidence in _COMPILED_RULES:
        hits = [(m.start(), pattern) for pattern in patterns for m in [pattern.search(text)] if m]
        if not hits:
            continue
        first_pos, first_pattern = min(hits, key=lambda h: h[0])
        key = (base_confidence, len(hits), -first_pos)
        if best_key is None or key > best_key:
            best_key = key
            best = SectionRuleMatch(
                section_id=section_id,
                confidence=base_confidence,
                reason=f"rule:{section_id}",
                matched_pattern=first_pattern.pattern,
            )
    return best


def find_overview_reassignment(text: str) -> Optional[SectionRuleMatch]:
    """If text matches a specialized rule, it should not stay in system_overview."""
    if not text or not text.strip():
        return None

    best: Optional[SectionRuleMatch] = None

    exclusion_patterns = [
        ("disaster_recovery", r"\b(daily\s+backups?|rds\s+snapshots?|restore|recovery\s+procedure|disaster\s+recovery\s+testing|retained\s+for\s+30\s+days)\b"),
        # "amazon ecr" / "container images" deliberately excluded here too --
        # see the matching comment on SECTION_RULES' security_controls entry
        # above for why (they're plain architecture facts, not a security
        # signal, and wrongly pulling system_overview content toward
        # security_controls was the same root cause either way).
        ("security_controls", r"\b(trivy|security\s+scanning|vault|secret\s+management)\b"),
        ("cost_optimization", r"\b(spot\s+instances?|scheduled\s+scaling|cost\s+optimization|non-production|low\s+traffic)\b"),
        ("danger_zones", r"\b(never\s+modify|do\s+not\s+touch|dangerous\s+area|terraform\s+state|autoscaler\s+configuration)\b"),
        ("ownership_escalation", r"\b(on-call\s+engineer|escalation\s+path|platform\s+engineering\s+manager|head\s+of\s+engineering|contact\s+platform\s+engineering)\b"),
        ("day1_survival_checklist", r"\b(new\s+team\s+members|first\s+safe\s+actions|grafana\s+dashboards?|pipeline\s+view|read-only\s+checks)\b"),
        ("deployment_and_rollback", r"\b(merged\s+into\s+the\s+main\s+branch|helm\s+roll[ -]?back|rollback\s+procedure|deployment\s+window|pre-deployment\s+checks)\b"),
    ]

    for section_id, pattern in exclusion_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            match = SectionRuleMatch(
                section_id=section_id,
                confidence=0.96,
                reason=f"overview_reassign:{section_id}",
                matched_pattern=pattern,
            )
            if best is None or match.confidence > best.confidence:
                best = match

    for section_id, pattern in _COMPILED_OVERVIEW_MISPLACEMENT:
        if section_id == "system_overview":
            continue
        if pattern.search(text):
            match = SectionRuleMatch(
                section_id=section_id,
                confidence=0.96,
                reason=f"overview_reassign:{section_id}",
                matched_pattern=pattern.pattern,
            )
            if best is None or match.confidence > best.confidence:
                best = match
    return best


# A rule at or above this confidence replaces the classifier's section
# unconditionally (apply_rule_overrides). context_mapper uses the same value
# to know which LLM classification checks would be discarded.
RULE_OVERRIDE_CONFIDENCE = 0.94

_GREETING_WORD = (
    r"(?:ok(?:ay)?|alright|all\s+right|hi|hello|hey|welcome|thanks?|thank\s+you|"
    r"good\s+(?:morning|afternoon|evening)|so|right|everyone|all|team|folks|guys|then|again|back|"
    r"for|joining|coming|being|here|today|and)"
)
_GREETING_ONLY_RE = re.compile(
    r"^\W*" + _GREETING_WORD + r"(?:[\s,!.]+" + _GREETING_WORD + r")*\W*$",
    re.IGNORECASE,
)


def apply_rule_overrides(classified_sentences, schema_metadata: Dict[str, Dict]) -> None:
    """
    Apply deterministic routing and pull misplaced sentences out of system_overview.
    Mutates classified_sentences in place.
    """
    from context_mapper import Classification, ClassifiedSentence, ExplainabilityLog
    from datetime import datetime

    for idx, cs in enumerate(classified_sentences):
        text = cs.sentence.text or ""
        if _GREETING_ONLY_RE.match(text):
            # "Alright, welcome." carries no knowledge; similarity scoring
            # still gave it a section, and Sign-off rendered it.
            cs.primary_classification = None
            cs.secondary_classifications = []
            cs.multi_section_assignments = []
            cs.is_unassigned = True
            continue
        rule_match = match_section_rules(text)

        current_section = (
            cs.primary_classification.section_id
            if cs.primary_classification
            else None
        )

        # Specialized rule beats semantic classification.
        if rule_match and (
            current_section is None
            or current_section == "system_overview"
            or (
                current_section in {"security_controls", "deployment_and_rollback"}
                and rule_match.section_id == "system_overview"
            )
            or rule_match.confidence >= RULE_OVERRIDE_CONFIDENCE
        ):
            meta = schema_metadata.get(rule_match.section_id, {})
            cs.primary_classification = Classification(
                section_id=rule_match.section_id,
                section_title=meta.get("title", rule_match.section_id),
                confidence=rule_match.confidence,
                similarity_score=rule_match.confidence,
                reason=f"Rule override: {rule_match.matched_pattern}",
            )
            cs.is_unassigned = False
            cs.explainability_log = ExplainabilityLog(
                action="rule_override",
                timestamp=datetime.utcnow().isoformat() + "Z",
                sentence_id=idx,
                section_id=rule_match.section_id,
                reasoning=f"Matched routing rule ({rule_match.matched_pattern})",
                confidence=rule_match.confidence,
            )
            continue

        # Secondary pass: pull specialized content out of overview.
        if current_section == "system_overview":
            _reassign_out_of_overview(cs, idx, text, schema_metadata, Classification, ExplainabilityLog, datetime)

    apply_continuation_rules(classified_sentences, schema_metadata)


def _reassign_out_of_overview(cs, idx, text, schema_metadata, Classification, ExplainabilityLog, datetime):
    """Pull specialized content out of system_overview (see find_overview_reassignment)."""
    reassignment = find_overview_reassignment(text)
    if not reassignment:
        return
    meta = schema_metadata.get(reassignment.section_id, {})
    cs.primary_classification = Classification(
        section_id=reassignment.section_id,
        section_title=meta.get("title", reassignment.section_id),
        confidence=reassignment.confidence,
        similarity_score=reassignment.confidence,
        reason=f"Overview reassignment: {reassignment.matched_pattern}",
    )
    cs.is_unassigned = False
    cs.explainability_log = ExplainabilityLog(
        action="overview_reassign",
        timestamp=datetime.utcnow().isoformat() + "Z",
        sentence_id=idx,
        section_id=reassignment.section_id,
        reasoning=f"Reassigned from system_overview via {reassignment.matched_pattern}",
        confidence=reassignment.confidence,
    )


# ============================================================================
# Discourse continuation
# ============================================================================
#
# Some sentences are grammatically dependent on an earlier one and carry no
# topic signal of their own, so scoring them in isolation places them almost
# at random. Observed on a real KT:
#   "There are currently several open responsibilities." -> Open Responsibilities
#   "The first is to review the certificate renewal configuration..." -> Handover
#   "The second is to document the Kafka lag procedure..."           -> Operational Calendar
#   "The third is to review the canary rollback configuration..."    -> Deployment
# and, in a failure narrative, "The usual remediation is to..." / "The
# corrective action was to..." landed in Disaster Recovery instead of with
# the failure they resolve. These rules make such a sentence inherit the
# section of the sentence it continues. They are linguistic, not tied to any
# transcript's subject matter.

_ORDINAL_ITEM_RE = re.compile(
    r"^\s*(?:and\s+)?(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|next|final|last)"
    r"(?:\s+one)?\s+(?:is|was|would\s+be|will\s+be)\b",
    re.IGNORECASE,
)
_ENUMERATION_HEAD_RE = re.compile(
    r"\bthere\s+(?:are|were|is)\s+(?:currently\s+|still\s+)?"
    r"(?:several|a\s+few|a\s+number\s+of|some|many|two|three|four|five|six|\d+)\b",
    re.IGNORECASE,
)
_DEPENDENT_CONTINUATION_RE = re.compile(
    r"^\s*(?:the\s+(?:\w+\s+)?(?:reason|root\s+cause|cause|symptom|fix|remediation|"
    r"corrective\s+action|workaround|mitigation)\b|(?:this|that)\s+(?:was|is)\s+because\b)",
    re.IGNORECASE,
)
# How far back an ordinal item may look for its enumeration head.
_ENUMERATION_WINDOW = 6
# An item keeps its own placement only when it independently matches a rule
# for one of these sections (an explicit "never modify ..." prohibition).
_CONTINUATION_YIELDS_TO_SECTIONS = frozenset({"danger_zones"})
# Openers that mark a clause cut from a longer sentence.
_FRAGMENT_START_RE = re.compile(r"^(?:and|or|but|then|so|while|which|whereas|nor|plus)\b", re.IGNORECASE)


def _inherit_section(cs, source, idx, reason, Classification, ExplainabilityLog, datetime) -> None:
    src = source.primary_classification
    cs.primary_classification = Classification(
        section_id=src.section_id,
        section_title=src.section_title,
        confidence=src.confidence,
        similarity_score=src.confidence,
        reason=reason,
    )
    cs.is_unassigned = source.is_unassigned
    cs.explainability_log = ExplainabilityLog(
        action="continuation",
        timestamp=datetime.utcnow().isoformat() + "Z",
        sentence_id=idx,
        section_id=src.section_id,
        reasoning=reason,
        confidence=src.confidence,
    )


def apply_continuation_rules(classified_sentences, schema_metadata: Dict[str, Dict]) -> None:
    """Place grammatically dependent sentences with the sentence they
    continue. Mutates classified_sentences in place; see module comment."""
    from context_mapper import Classification, ExplainabilityLog
    from datetime import datetime

    for idx, cs in enumerate(classified_sentences):
        raw = (cs.sentence.raw_text or cs.sentence.text or "").strip()
        text = (cs.sentence.text or "").strip()
        if not text:
            continue
        own_rule = match_section_rules(text)
        # Only an explicit safety prohibition outranks the sentence's
        # grammatical dependence on its antecedent. A passing keyword (e.g.
        # "automated rollback" in "The third is to review which services
        # have automated rollback") must not pull an enumerated task away
        # from the list it belongs to.
        if own_rule and own_rule.section_id in _CONTINUATION_YIELDS_TO_SECTIONS:
            continue

        if _ORDINAL_ITEM_RE.search(text):
            # Anchor to the enumeration HEAD ("There are several open
            # responsibilities."), not merely the previous item: chaining
            # item-to-item let one misplaced item drag every later item with
            # it. Falls back to the nearest earlier item when no head is in
            # range.
            anchor = None
            for back in range(idx - 1, max(-1, idx - 1 - _ENUMERATION_WINDOW), -1):
                prev = classified_sentences[back]
                prev_text = (prev.sentence.text or "").strip()
                if not prev.primary_classification:
                    continue
                if _ENUMERATION_HEAD_RE.search(prev_text):
                    anchor = prev
                    break
                if anchor is None and _ORDINAL_ITEM_RE.search(prev_text):
                    anchor = prev
            if anchor is not None and anchor.primary_classification.section_id != getattr(
                cs.primary_classification, "section_id", None
            ):
                _inherit_section(
                    cs, anchor, idx,
                    f"Continuation: enumerated item of \"{(anchor.sentence.text or '')[:60]}\"",
                    Classification, ExplainabilityLog, datetime,
                )
            continue

        # A fragment produced by splitting one long spoken sentence at a
        # comma ("then review the major Terraform modules...", "or bypass the
        # production approval process.") starts lowercase or with a
        # coordinating conjunction and has no subject of its own; it belongs
        # wherever the sentence it was cut from went.
        # A conjunction opener is only a cut-off clause when the previous
        # chunk really ended mid-sentence. After a full stop, "And avoid
        # deploying on Fridays..." is a new statement with its own topic,
        # and inheriting dragged it from the calendar into Danger Zones.
        prev_text = (classified_sentences[idx - 1].sentence.text or "").rstrip() if idx > 0 else ""
        cut_off = prev_text.endswith((",", ";", ":")) or not prev_text.endswith((".", "!", "?"))
        is_fragment = (bool(_FRAGMENT_START_RE.match(raw)) and cut_off) or (raw[:1].islower())
        if (is_fragment or _DEPENDENT_CONTINUATION_RE.search(text)) and idx > 0:
            prev = classified_sentences[idx - 1]
            if prev.primary_classification and (
                prev.primary_classification.section_id != getattr(cs.primary_classification, "section_id", None)
            ):
                _inherit_section(
                    cs, prev, idx,
                    "Continuation: fragment of the preceding sentence" if is_fragment
                    else "Continuation: explains the preceding sentence",
                    Classification, ExplainabilityLog, datetime,
                )


# ============================================================================
# Entity-type -> section affinity (Phase 5 signal: "extracted entities")
# ============================================================================
#
# Deliberately small and conservative. Only entity types with a strong,
# unambiguous section affinity are listed — generic types like "tool",
# "technology", "platform", "service" are left out on purpose: they're not
# selective enough and would nudge nearly every sentence toward the same
# architecture-ish sections regardless of what it's actually about.
#
# entity_type -> (section_id, max_boost)
ENTITY_TYPE_SECTION_AFFINITY: Dict[str, Tuple[str, float]] = {
    "monitoring": ("monitoring_observability", 0.10),
    "escalation": ("ownership_escalation", 0.10),
    "owner": ("ownership_escalation", 0.08),
}


def entity_affinity_boost(section_id: str, entities: Optional[Dict[str, List[str]]]) -> Tuple[float, Optional[str]]:
    """Return (boost, note) if extracted entities support classifying this
    sentence into `section_id`, else (0.0, None).

    `entities` is the dict EntityExtractor.get_context_entities() returns,
    e.g. {"monitoring": ["PagerDuty"], "owner": ["Platform Team"]}.
    """
    if not entities:
        return 0.0, None

    for entity_type, values in entities.items():
        affinity = ENTITY_TYPE_SECTION_AFFINITY.get(entity_type)
        if affinity and affinity[0] == section_id and values:
            return affinity[1], f"{entity_type}:{values[0]}"

    return 0.0, None


# Generic phrase markers for "this is non-obvious, experience-based
# knowledge" framing — not tied to any one transcript's subject matter. Used
# to *tag* a sentence for the Tribal Knowledge digest regardless of which
# section it's already been classified into (see
# knowledge_builder.append_tribal_knowledge_section) — this is additive
# tagging, not a reclassification, so it doesn't affect SECTION_RULES above.
TRIBAL_KNOWLEDGE_MARKERS: List[str] = [
    r"\btribal\s+knowledge\b",
    r"\bone\s+(?:important\s+)?(?:piece\s+of|item\s+of|thing\s+to\s+remember)\b",
    r"\bgotcha\b",
    r"\bheads[\s-]?up\b",
    r"\bkeep\s+in\s+mind\b",
    r"\bby\s+the\s+way\b",
    # Procedural-ordering language ("do X before Y") — a generic linguistic
    # signal for experience-based operational sequencing, not tied to any
    # one transcript's subject.
    r"\bbefore\s+investigating\b",
    r"\bfirst\s+(?:thing\s+you\s+should\s+)?check\b",
    r"\bmust\s+(?:not\s+)?\s*avoid\b",
]

_COMPILED_TRIBAL_MARKERS = [re.compile(p, re.IGNORECASE) for p in TRIBAL_KNOWLEDGE_MARKERS]


def is_tribal_knowledge(text: str) -> bool:
    """True if `text` reads as non-obvious, experience-based operational
    knowledge worth surfacing in the Tribal Knowledge digest."""
    if not text or not text.strip():
        return False
    return any(pattern.search(text) for pattern in _COMPILED_TRIBAL_MARKERS)
