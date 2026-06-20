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
        ],
        0.97,
    ),
    (
        "ownership_escalation",
        [
            r"\bdevelopers?\s+own\b",
            r"\bplatform\s+engineering(?:\s+team)?\s+owns?\b",
            r"\bescalation\s+path\b",
            r"\bdesklation\s+path\b",
            r"\bon-?call\s+engineer\b",
            r"\bhead\s+of\s+engineering\b",
            r"\bplatform\s+engineering\s+manager\b",
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
        ],
        0.96,
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
        "security_controls",
        [
            r"\bsecurity\s+scanning\b",
            r"\btrivy\b",
            r"\btrevi\b",
            r"\bamazon\s+ecr\b",
            r"\bcontainer\s+images?\s+are\s+stored\b",
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
        "plain_english_notes",
        [
            r"\btribal\s+knowledge\b",
            r"\bcache\s+invalidation\s+can\s+take\s+longer\b",
            r"\bsharp\s+edges?\b",
            r"\bgotcha\b",
            r"\bworkaround\b",
        ],
        0.96,
    ),
    (
        "day1_survival_checklist",
        [
            r"\bfor\s+new\s+team\s+members\b",
            r"\bstart\s+by\s+reviewing\b",
            r"\bfirst\s+safe\s+actions?\b",
            r"\brequired\s+access\b",
            r"\bday[\s-]?1\b",
            r"\bfor\s+new\s+team\s+members\b.*\bgrafana\s+dashboards?\b",
            r"\bkubernetes\s+namespaces?\b.*\bdeployment\s+pipelines?\b",
        ],
        0.92,
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
            r"\breact\s+front[\s-]?end\b",
            r"\bfast\s*api\b",
            r"\bplatform\s+consists\s+of\b",
            r"\bcustomers?\s+cannot\s+place\s+orders\b",
            r"\bwarehouse\s+fulfillment\b",
            r"\bserves\s+customers?\s+globally\b",
            r"\bpayments?,?\s+inventory\s+updates?\b",
            r"\bshipment\s+orchestration\b",
            r"\bstaging\s+environment\b.*\bmirrors?\s+production\b",
            r"\bpayment\s+integrations?\s+are\s+mocked\b",
            r"\bwe\s+use\s+terraform\b.*\b(?:argocd|argos\s+cd|gitops)\b",
            r"\binfrastructure\s+provisioning\b",
            r"\bterraform\b.*\bargocd\b.*\bvault\b",
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

    best: Optional[SectionRuleMatch] = None
    for section_id, patterns, base_confidence in _COMPILED_RULES:
        for pattern in patterns:
            if pattern.search(text):
                match = SectionRuleMatch(
                    section_id=section_id,
                    confidence=base_confidence,
                    reason=f"rule:{section_id}",
                    matched_pattern=pattern.pattern,
                )
                if best is None or match.confidence > best.confidence:
                    best = match
    return best


def find_overview_reassignment(text: str) -> Optional[SectionRuleMatch]:
    """If text matches a specialized rule, it should not stay in system_overview."""
    if not text or not text.strip():
        return None

    best: Optional[SectionRuleMatch] = None

    exclusion_patterns = [
        ("disaster_recovery", r"\b(daily\s+backups?|rds\s+snapshots?|restore|recovery\s+procedure|disaster\s+recovery\s+testing|retained\s+for\s+30\s+days)\b"),
        ("security_controls", r"\b(trivy|security\s+scanning|vault|secret\s+management|amazon\s+ecr|container\s+images?)\b"),
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


def apply_rule_overrides(classified_sentences, schema_metadata: Dict[str, Dict]) -> None:
    """
    Apply deterministic routing and pull misplaced sentences out of system_overview.
    Mutates classified_sentences in place.
    """
    from context_mapper import Classification, ClassifiedSentence, ExplainabilityLog
    from datetime import datetime

    for idx, cs in enumerate(classified_sentences):
        text = cs.sentence.text or ""
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
            or rule_match.confidence >= 0.94
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
            reassignment = find_overview_reassignment(text)
            if reassignment:
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
