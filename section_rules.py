"""
Deterministic section routing rules for KT classification.

High-priority phrase and pattern rules override weak semantic matches,
especially preventing the system_overview section from absorbing specialized content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dialogue import says_nothing_specific, without_label
from fact_units import ONBOARDING_VERBS


@dataclass
class SectionRuleMatch:
    section_id: str
    confidence: float
    reason: str
    matched_pattern: str


# (section_id, compiled_patterns, base_confidence)
# Order matters: earlier specialized rules win ties.
def _data_components() -> str:
    """Names of data stores, queues, caches, storage and pipelines (the
    architecture diagram's catalog), as one regex alternation, longest first."""
    from architecture_diagram import _LAYER_TERMS

    names = {n for layer in ("database", "cache", "queue", "object_storage", "data_pipeline")
             for n in _LAYER_TERMS.get(layer, [])}
    return "|".join(r"\s+".join(re.escape(word) for word in n.split())
                    for n in sorted(names, key=len, reverse=True))


_DATA_COMPONENTS = _data_components()

# Infrastructure-as-code tools, for "Infrastructure is written in Bicep".
_IAC = r"(?:terraform|opentofu|bicep|pulumi|cloudformation|arm\s+templates?|(?:aws\s+)?cdk|crossplane|ansible)"

# When the load rises: "demand increasing significantly during weekends and
# dinner hours", "traffic peaks on Monday mornings". A load word, a rise and
# a period of time; a bare rate ("300 requests a second at peak") is not a
# period. Shared with coverage_topics.py, so the coverage report counts as
# discussed what these rules file under the Operational Calendar.
RISING_LOAD_PERIOD = (
    r"\b(?:traffic|demand|load|volumes?|orders?|usage|requests?|activity|transactions?)\s+(?:\w+\s+){0,2}?"
    r"(?:peak(?:s|ing)?|spik(?:es?|ing)|surg(?:es?|ing)|increas(?:es?|ing)|ris(?:es?|ing)|grow(?:s|ing)?|"
    r"doubl(?:es?|ing)|(?:is|are)\s+(?:highest|heaviest|busiest))\b[^.]{0,40}?\b"
    r"(?:during|on|at|in|over|around|every|before)\s+(?:the\s+)?(?:\w+\s+){0,3}?"
    r"(?:weekends?|weekdays?|evenings?|mornings?|nights?|holidays?|dinner|lunch|breakfast|month-?ends?|"
    r"quarter-?ends?|year-?ends?|christmas|festive|seasons?|mondays?|tuesdays?|wednesdays?|thursdays?|"
    r"fridays?|saturdays?|sundays?|sales?|campaigns?|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b"
)

# A sentence about one environment, its subject: "staging uses recorded
# gantry traffic from last year" (in SECTION_RULES below). Such a detail does
# not change the topic being discussed (segment_topics): "Releases are
# performed with Azure DevOps" after "The test environment receives ..." went
# to Environments on the topic boost (dev d01).
_ENVIRONMENT_SUBJECT_RX = (
    r"^\W*(?:(?:and|so|also|but|then)\s+)?(?:the\s+|our\s+)?(?:staging|production|prod|dev|development|test|"
    r"qa|uat|sandbox|pre-?prod(?:uction)?)(?:\s+(?:environment|cluster|account|subscription|project))?\s+"
    r"(?:uses?|has|have|gets?|receives?|mirrors?|holds?|contains?|shares?|talks?\s+to|points?\s+(?:at|to)|"
    r"is\s+(?:a\s+)?(?:copy|clone|replica|smaller|identical|the\s+same|separate)|runs?\s+(?:a\s+)?(?:smaller|"
    r"single|cheaper|scaled))\b")
_NOT_TOPIC_ANCHORS = {_ENVIRONMENT_SUBJECT_RX}

# The new owner's own time frame, opening the sentence: "In the first month,
# ...", "Within the first month, the new owner should ...", "First day get
# access to ..." (speech to text, no comma), "After a month I'd like you to
# ...". In SECTION_RULES below, above the calendar's peak-day rules: "In the
# first month take over the black friday preparation" went to the
# Operational Calendar, "Within the first month, the new owner should
# complete one release" to Ownership, and the First 30-Day plan was reported
# Missing although it was discussed (X67, dev d08 and d09). Not "the first
# week of January" or "the first month after launch", and a bare "after a
# month" only with a person ("After a month, old logs are deleted" is
# retention).
_ONBOARDING_LEAD = r"^\W*(?:(?:so|and|then|okay|ok|right|now|um+|uh+)[\s,]+)*"
_ONBOARDING_FIRST = r"(?:(?:on|in|for|during|within|over|by\s+the\s+end\s+of)\s+(?:the\s+|your\s+)?|your\s+)?(?:very\s+)?first\s+"
_NOT_ONBOARDING_TAIL = r"(?!\s+(?:of|after|back)\b)"
ONBOARDING_MONTH_RX = (
    _ONBOARDING_LEAD + r"(?:" + _ONBOARDING_FIRST + r"(?:month|(?:four|few|couple\s+of)\s+weeks|(?:30|thirty)\s+days)\b"
    + _NOT_ONBOARDING_TAIL + r"|(?:(?:after|within)\s+(?:a|one)\s+month|within\s+(?:30|thirty)\s+days"
    r"|by\s+the\s+end\s+of\s+(?:the|your)\s+month"
    r"|by\s+week\s+(?:three|four|five|3|4|5))\b[\s,:]+(?:i'?d\s+like\s+you|you|the\s+new\s+(?:owner|engineer|lead)|"
    + ONBOARDING_VERBS + r")\b)")
ONBOARDING_DAY_RX = (
    _ONBOARDING_LEAD + r"(?:" + _ONBOARDING_FIRST + r"(?:day|week|couple\s+of\s+days)\b|(?:on\s+)?day\s+(?:one|1)\b)"
    + _NOT_ONBOARDING_TAIL)

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
            # Not data that never reaches a system ("Card details never touch
            # our servers"): that is a control (security_controls).
            r"(?<!\bdetails\s)(?<!\bdata\s)(?<!\bnumbers\s)(?<!\binformation\s)"
            r"\bnever\s+(?:manually\s+)?(?:change|edit|delete|touch|alter|run)\b",
            # "The thing you really must not do is run the claim job twice"
            # (dev d14) matched nothing and went to Open Responsibilities.
            r"\b(?:(?:must|should)\s+(?:not|never)|mustn'?t|shouldn'?t)\s+(?:ever\s+)?(?:do|run(?!\s+into)|change|edit"
            r"|delete|touch|alter|modify|drop|purge|disable|truncate|restart)\b",
            # An action that cannot be taken back: "It advances every student
            # a year, and undoing it means a database restore."
            r"\b(?:undoing|reversing|reverting)\s+(?:it|that|this)\s+(?:means|requires|needs|takes)\b",
            r"\bcan(?:'t|not)\s+be\s+(?:undone|reversed)\b",
            # Spoken warnings: "One warning: the last database migration might
            # be hard to reverse, so be careful with it." went to Disaster
            # Recovery, and Danger Zones reported the topic not discussed.
            r"^\W*(?:(?:one|a|another|final|big|quick)\s+)?(?:warning|caution|word\s+of\s+(?:warning|caution))\s*[:,-]",
            r"\bbe\s+(?:very\s+|really\s+|extra\s+)?careful\s+(?:with|when|about|not\s+to)\b",
            r"\b(?:hard|difficult|impossible|risky|tricky)\s+to\s+(?:reverse|undo|roll\s*back|revert)\b",
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
            # "... and formal escalation contacts are still undocumented" went
            # to Handover Completion Check (a user's KT, 25 Oct).
            r"\bescalation\s+(?:path|chain|point|goes|matrix|contacts?|list|policy|procedure|process|rota)\b",
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
            # Who to go to: "Day-to-day questions go to Omar Haddad", "for
            # anything with invoices, contact the finance systems lead".
            r"\b(?:questions?|issues?|requests?|queries)\s+(?:about\s+[^.;]{1,30}?\s+)?(?:go(?:es)?|should\s+go|are\s+routed)\s+to\b",
            # A named contact too: "call Raj Patel, the application manager",
            # "call Sam Achebe, he's the head of engineering". Not "the call
            # centre".
            r"\b(?:contact|reach\s+out\s+to|call)(?!\s+cent(?:re|er)\b)\s+(?:the\s+)?(?:[\w'-]+,?\s+){0,4}?"
            r"(?:lead|manager|owner|team|squad|service\s+desk|help\s*desk|dba|architect|head\s+of\s+[\w-]+)\b",
            # The next step when a page goes unanswered: "if they do not
            # answer in fifteen minutes, call Maria" went to Monitoring, and
            # the escalation path was reported as not discussed.
            r"\bif\s+(?:(?:they|he|she|the\s+(?:on-?call|primary)(?:\s+(?:engineer|person))?)\s+(?:does\s+not|do\s+not|"
            r"doesn'?t|don'?t|did\s+not|didn'?t|fails?\s+to|can'?t|cannot)|nobody|no\s+one|no-one)\s+"
            r"(?:answer|respond|reply|pick\s+up|acknowledge|ack)s?\b",
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
            # A failure introduced as the one you will meet ("The incident you
            # will meet most often is Twilio throttling us") went to
            # Monitoring; one that clears itself ("Usually it recovers by
            # itself, but if it lasts more than ten minutes ...") to Disaster
            # Recovery.
            r"\b(?:incident|failure|problem|issue)s?\s+(?:you(?:'ll|\s+will)\s+)?(?:meet|see|hit|run\s+into)\s+"
            r"(?:most\s+)?(?:often|most)\b",
            r"\brecovers?\s+(?:by|on)\s+(?:itself|its\s+own)\b",
            r"\bif\s+it\s+(?:lasts|goes\s+on|persists)\s+(?:for\s+)?(?:more|longer)\s+than\b",
            # A symptom: "Oracle archive logs filling the disk on Monday
            # mornings" went to Disaster Recovery (archive logs, backups).
            r"\b(?:fill(?:s|ed|ing)?\s+(?:up\s+)?the|filled\s+up\s+the)\s+(?:disk|disks|volume|partition|drive)\b",
            r"\b(?:disk|disks|volume|partition)\s+(?:fills?|filling|filled|runs?\s+out|running\s+out)\b",
            # A failure introduced by how often or how badly it bites: "The
            # classic failure is scanners losing their session", "the most
            # annoying failure: a store sends an empty file".
            r"\b(?:classic|typical|usual|biggest|most\s+(?:common|frequent|annoying|painful))\s+"
            r"(?:failure|problem|issue|incident|outage)\b",
            # A provider limiting us is a failure even when the sentence names
            # the day it happens: "Twilio rate limits us when there's a big
            # batch like on black friday" went to the calendar (dev d08).
            r"\b(?:rate[\s-]?limit(?:s|ed|ing)?|throttl(?:es|ed|ing))\s+us\b",
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
            r"\b(?:deploy(?:ment)?|change|code|release)\s+freezes?\b",
            RISING_LOAD_PERIOD,
            # Getting ready for a known busy event: "Scale the web dynos to
            # twenty before the appeal starts" went to Cost Optimization
            # (dev d14, X91).
            r"^\W*(?:(?:always|just|then|so|and|also)\s+)?(?:scale|pre-?scale|pre-?warm|warm\s+up)\b[^.;]{0,60}?"
            r"\b(?:before|ahead\s+of)\s+(?:the|each|every|any|our)\s+(?:[\w'-]+\s+){0,3}?"
            r"(?:starts?|begins?|kicks?\s+off|opens?|goes\s+live|launch(?:es)?)\b",
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
            # The build-and-release chain in general form: "Azure DevOps
            # pipelines build the images and push them to ACR, and Flux syncs
            # the releases to AKS" reached System Overview, and Deployment &
            # Rollback showed only a Key Vault sentence.
            r"\bazure\s+devops\s+pipelines?\b",
            r"\bflux\s+(?:syncs?|deploys?|reconciles?|applies)\b",
            r"\b(?:builds?|built|building)\s+the\s+(?:container\s+|docker\s+)?images?\b",
            r"\bpush(?:es|ed|ing)?\s+(?:them|the\s+(?:container\s+)?images?)\s+to\b",
            # What starts a build or a release: "Cloud Build deploys to Cloud
            # Run on every merge to main", "Builds run in Cloud Build on every
            # push to the release branch".
            r"\bon\s+(?:every|each)\s+(?:push|merge|commit|pull\s+request|tag)\b",
        ],
        0.94,
    ),
    (
        # A statement that is only about the infrastructure-as-code tool
        # ("Infrastructure is written in Bicep.", "Terraform manages
        # infrastructure."): what the platform is built with. The whole
        # sentence must be that statement. Matched inside a longer one, the
        # rule decided the section for everything else in it, and moved those
        # facts as well: "Cloud Build deploys to Cloud Run" in "Everything is
        # in Terraform, and Cloud Build deploys ...", a canary release chunked
        # after "The infrastructure is in Terraform, applied from Cloud Build",
        # "GKE is used for Kubernetes workloads" chunked before "Terraform
        # manages infrastructure". Those keep the classifier's placement. It
        # needs the infrastructure itself, so rebuilding from the tool (DR)
        # and its state (Danger Zones) keep their own rules.
        "architecture_reference",
        [
            r"^\W*(?:"
            r"(?:all\s+(?:of\s+)?(?:the\s+|our\s+)?|the\s+|our\s+)?(?:infrastructure|infra|everything|"
            r"(?:whole\s+|entire\s+)?stack)\s+(?:is|are)\s+(?:all\s+)?(?:(?:written|defined|managed|provisioned|"
            r"described|declared|codified|kept)\s+)?(?:(?:in|with|using|as|via)\s+)?" + _IAC +
            r"|" + _IAC + r"\s+(?:manages|provisions|defines|describes|creates)\s+(?:all\s+(?:of\s+)?)?"
            r"(?:the\s+|our\s+)?(?:infrastructure|infra|cloud\s+resources)"
            r"|(?:we\s+)?(?:manage|provision|define)s?\s+(?:the\s+|our\s+|all\s+(?:the\s+|our\s+)?)?(?:infrastructure|infra)\s+"
            r"(?:with|in|using|through)\s+" + _IAC +
            r"|(?:we\s+)?(?:use|uses|using)\s+" + _IAC + r"\s+(?:for|to\s+(?:manage|provision|define|build))\s+"
            r"(?:the\s+|our\s+|all\s+(?:the\s+|our\s+)?)?(?:infrastructure|infra)"
            r")[\s.!]*$",
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
            # A release approval gate: "After the clinical safety officer
            # signs off, the same pipeline promotes the build to production"
            # was filed under Handover Completion (sign-off).
            r"\bpromot(?:e|es|ed|ing)\s+(?:the\s+|a\s+)?(?:build|release|image|artifact|version)\s+to\s+"
            r"(?:production|prod|staging|live)\b",
            r"\b(?:safety|change|cab|release|security|go[\s-]live)\s+(?:sign[\s-]?off|approval)\s+is\s+"
            r"(?:mandatory|required|needed)\b",
            # Change control: "Changes to the MQ bridge configuration require
            # approval from the mainframe team" went to Ownership for the
            # team it names (dev d13, X91). Only as the sentence's subject:
            # anywhere in a run with no punctuation, it took a restriction
            # said before it to Deployment (d13's noisy copy).
            r"^\W*(?:(?:and|so|also|then|but)\s+)?(?:any\s+)?changes?\s+to\s+(?:the\s+|our\s+|its\s+|a\s+|an\s+)?+"
            r"(?!(?:on-?call|rota|roster|budget|spend|team|ownership)\b)[\w/.-]+(?:\s+[\w/.-]+){0,4}?\s+"
            r"(?:require|requires|need|needs)\s+(?:an?\s+|prior\s+|written\s+|formal\s+|explicit\s+)?"
            r"(?:approval|sign[\s-]?off)\b",
            # "we restore the previous image from its last savepoint" is a
            # rollback, not disaster recovery: what is restored is a build.
            r"\b(?:restore|reinstate)\s+the\s+(?:previous|last\s+(?:good|known[\s-]good|working))\s+(?:\w+\s+)?"
            r"(?:image|images|version|release|build|revision|tag|artifact)\b",
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
            # Asked rather than stated ("What is the RTO? We never agreed
            # one."): still a disaster-recovery topic, merged with its answer.
            r"\b(?:what(?:'s|\s+is|\s+are)|is\s+there\s+an?|do\s+we\s+have\s+an?)\s+(?:the\s+|our\s+)?(?:rto|rpo)s?\b",
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
            # Data residency is a control, not a recovery region: "Patient
            # data never leaves the UK regions" went to Disaster Recovery.
            r"\bdata\s+(?:never\s+leaves|must\s+(?:stay|remain)\s+in|stays\s+in|resides\s+in|is\s+kept\s+in)\s+"
            r"(?:the\s+)?(?:\w+\s+){0,2}(?:regions?|country|countries|eu|uk|us|data\s+centres?|data\s+centers?)\b",
            r"\bdata\s+residency\b",
            # Sensitive data kept out of a system, or handled as personal
            # data: "Card details never touch our servers", "Phone numbers are
            # personal data so we delete message logs after ninety days".
            r"\b(?:card|cardholder|payment|credit\s+card|customer|personal|patient)\s+(?:data|details|numbers|information)"
            r"\s+(?:never|does\s+not|doesn'?t|do\s+not|don'?t)\s+(?:touch(?:es)?|reach(?:es)?|hits?|leaves?"
            r"|pass(?:es)?\s+through|go(?:es)?\s+through)\b",
            r"\b(?:is|are|counts?\s+as)\s+(?:personal|sensitive|personally\s+identifiable)\s+(?:data|information)\b",
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
            # A saving stated by its effect: "..., which cut the SMS bill by
            # about a third" was filed under System Overview.
            r"\b(?:cut|cuts|reduc(?:e|es|ed)|lower(?:s|ed)?|halv(?:e|es|ed)|sav(?:e|es|ed))\s+(?:\w+\s+){0,3}"
            r"(?:bill|costs?|spend|invoice)\b",
            # Where the money goes and what holds it down: "The Premium plan
            # is the largest cost item", "auto-suspend after sixty seconds
            # idle, which keeps the bill down", "We pay for a three-year
            # savings plan", "The App Service plan is oversized."
            r"\b(?:largest|biggest|main|major)\s+(?:cost|expense|spend|line\s+item)\b",
            r"\bkeeps?\s+(?:the\s+)?(?:bill|costs?|spend)\s+down\b",
            r"\b(?:savings\s+plans?|reserved\s+instances?|committed[\s-]use\s+discounts?)\b",
            r"\bauto[\s-]?suspend\w*\b",
            r"\bwe\s+(?:spend|pay)\s+(?:a\s+lot|the\s+most|too\s+much)\b",
            r"\b(?:oversized|over-?provisioned|under-?utili[sz]ed|right-?siz\w+)\b",
        ],
        0.94,
    ),
    (
        "day1_survival_checklist",
        [
            r"\bfor\s+new\s+team\s+members\b",
            # "To get started, request the beacon-operators group in Google
            # Cloud" went to Monitoring.
            r"^\W*(?:to\s+get\s+started|getting\s+started|when\s+you\s+start|before\s+anything\s+else)\b",
            r"\b(?:request|ask\s+for|get\s+added\s+to)\s+(?:access\s+to\s+)?(?:the\s+)?[\w-]+(?:\s+[\w-]+)?\s+"
            r"(?:group|role)\b",
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
            # A milestone of the first month: "By week four you should be on
            # the Airflow on-call", "By the end of the month you should lead a
            # production release" (which went to the month-end calendar).
            r"\bby\s+(?:the\s+end\s+of\s+(?:the\s+|your\s+)?(?:first\s+)?(?:month|week\s+\w+)|"
            r"week\s+(?:two|three|four|[234]))\b[^.]{0,60}?\byou\s+(?:should|will|can|could|must)\b",
        ],
        0.94,
    ),
    # The new owner's time frame opening the sentence (ONBOARDING_MONTH_RX).
    ("first_30_day_ownership", [ONBOARDING_MONTH_RX], 0.97),
    ("day1_survival_checklist", [ONBOARDING_DAY_RX], 0.97),
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
            # The system introduced by name: "Okay, so this is Streamline, our
            # GCP data platform for clickstream analytics." went to
            # Architecture (it names a cloud) and System Overview had no purpose.
            r"^\W*(?:(?:so|okay|ok|alright|right|well|now|and|um|uh)\W+)*this\s+is\s+[\w.&+-]+(?:\s+[\w.&+-]+){0,3}"
            r"\s*,\s*(?:our|the|a|an)\s+(?:[\w/&.+-]+\s+){0,5}?(?:platform|system|service|application|app|api|"
            r"backend|pipeline|portal|product|website|site|engine)\b",
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
            # "The move to MySQL 8.0 still needs doing, and no one has picked it up."
            r"\bstill\s+needs?\s+(?:doing|to\s+be\s+(?:done|finished|completed))\b",
            r"\b(?:no\s*one|nobody)\s+(?:has\s+)?(?:picked\s+(?:it|this|that)\s+up|owns\s+(?:it|this|that))\b",
            # "...an old Azure Function that nobody really owns. It needs
            # moving onto the notification service."
            r"\b(?:no\s*one|nobody)\s+(?:really|actually|properly|officially)\s+owns\b",
            r"\bneeds?\s+(?:moving|migrating|replacing|upgrading|rewriting|retiring|decommissioning)\b",
            r"\bneeds?\s+to\s+be\s+(?:moved|migrated|replaced|upgraded|rewritten|retired|decommissioned)\b",
            r"\bnot\s+assigned\s+to\s+anyone\b",
            # Work left part-done: "a half finished migration from Node 14",
            # "There's also an unfinished A/B test framework migration",
            # "One item remains open: ...", "scoped but not started",
            # "Kafka upgrade not started", "The TLS upgrade on the MQ
            # channels is waiting on the mainframe team." Said of a piece of
            # work, not of a port or a connection ("the port remains open").
            r"\bhalf[\s-]?(?:finished|done|complete[d]?|migrated|built)\b",
            r"\bunfinished\b",
            r"\b(?:items?|tasks?|tickets?|actions?|questions?|work)\s+(?:still\s+)?(?:remains?|is\s+still|are\s+still)\s+open\b",
            r"\b(?:scoped|planned|agreed|approved|budgeted)\s+but\s+(?:not|never)\s+(?:yet\s+)?(?:started|begun|done|finished|scheduled)\b",
            r"\b(?:upgrade|migration|move|rollout|roll-out|project|onboarding|replacement|clean-?up|decommission\w*)\s+"
            r"(?:is\s+|has\s+)?(?:still\s+)?not\s+(?:yet\s+)?(?:been\s+)?(?:started|begun|scheduled|finished|completed?)\b",
            r"\b(?:upgrade|migration|move|rollout|roll-out|project|onboarding|replacement)\b[^.;]{0,50}?\b(?:is|are)\s+"
            r"(?:still\s+)?(?:waiting|blocked)\s+(?:on|for|by)\b",
            # Work planned for a later quarter or year, or not begun ("Next
            # quarter we plan to add a second region, but nobody has started
            # on it" went to Disaster Recovery). Not "next month you will
            # own the releases" (an onboarding plan).
            r"\b(?:no\s*one|nobody)\s+(?:has\s+)?(?:started|begun|worked)\s+on\s+(?:it|this|that)\b",
            r"\bnext\s+(?:quarter|year|fiscal\s+year|financial\s+year)\b[^.;]{0,40}?\b(?:plan|plans|planned|planning|"
            r"will|going\s+to|intend)\b|\b(?:planned|scheduled)\s+for\s+(?:next\s+(?:quarter|year)|q[1-4]|later\s+"
            r"this\s+year)\b",
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
            # ...but not the reason given for a prohibition: "And don't change
            # the ALB idle timeout, the partners' clients depend on it."
            r"^(?!\W*(?:(?:and|but|also|so|please)\W+)*(?:never|do\s+not|don'?t)\b)"
            r".*?\b(?:\w+\s+){0,3}(?:depend|rely|relies)\s+on\s+(?:it|this\s+(?:service|platform|system))\b",
            # "Thanks for taking over Pinecart, the checkout service ..." introduces
            # the system; it was filed under Handover Completion.
            r"\b(?:thanks?|thank\s+you)\s+for\s+taking\s+over\b",
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
    (
        # A sentence about one environment, which is its subject: "staging
        # uses recorded gantry traffic from last year" went to Deployment
        # (dev d03). What a deploy does there ("staging deploys on merge") is
        # not one of these verbs; a recovery, monitoring or security rule
        # (0.95 and up) still wins over it.
        "environments",
        [_ENVIRONMENT_SUBJECT_RX],
        0.945,
    ),
    # ------------------------------------------------------------------
    # General phrasings found by the golden suite (tests/goldens, P1-11).
    # Each is a way of saying something, not a fact from one transcript.
    # ------------------------------------------------------------------
    (
        # A sentence that opens with a prohibition is a danger-zone rule
        # ("Never rotate the SQL admin password without telling finance",
        # "Also do not restart Vault without the unseal keys"). Below the
        # calendar rule (0.975), so "never deploy on Fridays" stays a
        # calendar rule.
        "danger_zones",
        [
            r"^(?:\W*(?:and|but|also|so|please|be\s+careful)\b[\s,:]*)*(?:never|do\s+not|don'?t)\s+"
            r"(?!worry\b|hesitate\b|forget\b|mind\b)"
            # Not a caveat about the document ("Do not describe the estimate
            # as a verified guarantee": dialogue._DOCUMENT_CAVEAT_RE).
            r"(?!(?:describe|present|report|document|state|treat|label|record)\b[^.]*\bas\s+(?:an?\s+)?"
            r"(?:verified|confirmed|guaranteed|guarantee|final|fact|certain|settled)\b)[a-z]+",
            # The same prohibition after a lead-in: "Watch out for the
            # nightly batch at 2 AM: do not restart Tomcat while it runs."
            r"[:;]\s*(?:please\s+)?(?:never|do\s+not|don'?t)\s+(?!worry\b|hesitate\b|forget\b|mind\b|panic\b)[a-z]+",
        ],
        0.955,
    ),
    (
        "common_failures",
        [
            # A runbook step: "When invoices get stuck, check whether ...", or
            # with who does it: "If CPU stays above 75% for five minutes, we
            # should scale up." (went to Cost Optimization, a user's KT, 23 Oct).
            r"^(?:\W*(?:and|so|also)\s+)?(?:when|whenever|if)\s+[^,;]{4,120}?,\s*(?:then\s+|first\s+)?"
            r"(?:(?:we|you)\s+(?:should|must|need\s+to|have\s+to|can|just)\s+(?:then\s+|first\s+|just\s+)?)?"
            r"(?:check|restart|run|scale|rotate|renew|redeploy|switch|verify|look\s+at|clear|flush|drain|"
            r"increase|bump|recreate)\b",
            # A failure introduced as such, and its fix.
            r"\b(?:one|another|the|a)\s+(?:(?:most|second|third|other|big|main|recurring|common|known|frequent|"
            r"typical)\s+){1,3}(?:production\s+)?(?:problem|issue|failure)s?\s+(?:is|was|we\s+(?:see|hit))\b",
            r"\bthe\s+fix\s+(?:is|was)\b",
            # The fix stated by its effect: "Restarting the scanner app fixes
            # it", "adding a second instance sorts it out".
            r"\b(?:fixes|fixed|sorts|sorted|clears|cleared|solves|solved|resolves|resolved|cures)\s+it(?:\s+out)?\b",
            # The instruction that follows a symptom: "Renew it in Key Vault and restart the worker."
            r"^(?:then\s+)?(?:renew|restart)\s+(?:it|the|them)\b",
            # A triage step: "For Kubernetes issues, check GKE and ArgoCD." It
            # filled Deployment's pre-deployment checks.
            r"^(?:\W*(?:and|so|also)\s+)?for\s+(?:[\w/-]+\s+){0,3}(?:issues?|problems?|failures?|incidents?|errors?|"
            r"outages?),?\s+(?:first\s+)?(?:check|look\s+at|start\s+with|verify|inspect)\b",
        ],
        0.955,
    ),
    (
        "deployment_and_rollback",
        [
            r"\bdeployments?\s+(?:go|goes|run|runs|happen|happens|are\s+done|is\s+done)\s+"
            r"(?:through|via|with|using|every|on|weekly|nightly)\b",
            r"\breleases?\s+(?:go|goes)\s+out\s+(?:every|on|weekly)\b",
        ],
        0.955,
    ),
    (
        # How data moves through the system's components.
        "architecture_reference",
        [
            r"\b(?:events?|messages?|jobs?|payloads?)\s+(?:go|goes|flow|flows|are\s+(?:sent|published|pushed|queued))\s+"
            r"(?:through|via|to|into|on)\s+(?:kafka|rabbitmq|sqs|sns|pub/?sub|kinesis|service\s+bus|event\s*hubs?|"
            r"nats|activemq)\b",
            r"\b(?:cached|caches|caching)\s+(?:in|with|on)\s+(?:redis|memcached|memorystore|elasticache)\b",
            # A data-moving verb and a named data component: "Devices publish
            # to Pub/Sub", "writes them to BigQuery", "archived in Cloud
            # Storage". Scored by similarity alone, these went to Disaster
            # Recovery ("archived" reads like a backup).
            # Where the workloads run: "The ingestion services run on GKE." was
            # merged into the next sentence (a deployment chain) and printed
            # under Deployment & Rollback.
            r"^\W*(?:the\s+|our\s+|all\s+)?(?:[\w-]+\s+){0,3}(?:services?|workloads?|apis?|apps?|applications?|"
            r"microservices|consumers?|workers?)\s+(?:all\s+)?(?:run|runs|are\s+running|is\s+running|are\s+hosted|"
            r"is\s+hosted)\s+on\s+(?:the\s+|our\s+|an?\s+)?(?:[\w-]+\s+){0,2}?(?:gke|eks|aks|kubernetes|k8s|ecs|fargate|"
            r"cloud\s+run|app\s+service|ec2|vms?|openshift)\b",
            # How many regions it runs in ("Right now we run in one region"
            # went to Ownership); a standby or replica region is DR.
            r"\b(?:run|runs|running|deployed|hosted|live|lives)\s+(?:only\s+)?in\s+(?:one|a\s+single|two|three|four|"
            r"\d+|multiple|several)\s+(?:[\w-]+\s+)?regions?\b",
            # Not a permission: "the service account only has read access to
            # BigQuery" said under "Security wise" went to Architecture (dev d02).
            r"\b(?:publish\w*|writ(?:e|es|ten|ing)(?![\s-]+(?:access|permissions?|rights?|roles?|scopes?)\b)|"
            r"stor(?:e|es|ed|ing)|archiv\w*|reads?(?![\s-]+(?:access|only|permissions?|rights?|roles?|scopes?)\b)|"
            r"reading|send\w*|stream\w*|"
            r"push\w*|load\w*|ingest\w*|enqueue\w*)\b[^.;]{0,40}?\b(?:to|into|in|from|on)\s+(?:the\s+|a\s+|an\s+)?"
            r"(?:" + _DATA_COMPONENTS + r")\b",
            # A component's size: "I think Redis is using 4 GB of memory" went
            # to Common Failures for "memory" (a user's KT, 23 Oct). Not when
            # the sentence is about running short ("hits 4 GB and gets killed").
            r"^(?!.*\b(?:fail\w*|error\w*|crash\w*|oom\w*|evict\w*|leak\w*|spik\w*|exhaust\w*|runs?\s+out|fill\w*|"
            r"full|hit|hits|exceed\w*|above|over|throttl\w*|slow\w*)\b)"
            r".*\b(?:(?:is|are)\s+(?:using|allocated|sized\s+(?:at|with)|provisioned\s+with|configured\s+with|given)|"
            r"uses|has|have|runs?\s+with|gets?|(?:memory|ram|disk|storage)\s+limit\s+(?:is|of))\s+"
            r"(?:about\s+|around\s+|roughly\s+|approximately\s+|up\s+to\s+|~|an?\s+)?"
            r"\d[\d.,]*\s*(?:gb|gib|mb|mib|tb|tib|vcpus?|cpus?|cores?)\b",
        ],
        0.95,
    ),
    (
        # Who uses the system: "Finance and sales operations use it to pull
        # monthly reports." Not "we use it to ...", where "it" is a tool.
        "system_overview",
        [
            r"^(?!\s*(?:we|you|they|i)\b)(?:\W*[\w-]+\s+){1,6}?(?:use|uses|rely\s+on|relies\s+on)\s+it\s+(?:to|for)\b",
            # "the finance and merchandising teams build all their reports on
            # top of it" (dev d06): who builds on the system.
            r"^(?!\s*(?:we|you|they|i)\b)(?:\W*[\w-]+\s+){1,6}?(?:build|builds|run|runs)\s+(?:all\s+)?"
            r"(?:their|its|his|her)\s+(?:[\w-]+\s+){0,2}on\s+top\s+of\s+it\b",
        ],
        0.95,
    ),
    (
        # How much the system handles: "This platform processes approximately
        # 3TB of data per day" went to Disaster Recovery ("per day" reads like
        # a backup schedule).
        "system_overview",
        [
            r"\b(?:processes|handles|serves|receives|ingests|settles|sends|takes|supports)\b[^.;]{0,40}?"
            r"\b(?:\d[\d.,]*\s*(?:k|m|tb|gb|pb|million|thousand|billion)?|one|two|three|four|five|six|seven|eight|"
            r"nine|ten|twenty|thirty|forty|fifty|hundred|thousand|million|billion)\b[^.;]{0,40}?"
            r"\b(?:per|a|an|each|every)\s+(?:second|minute|hour|day|week|month|year)\b",
        ],
        0.955,
    ),
    (
        "disaster_recovery",
        [
            r"\bgeo-?replicat\w*",
            r"\bsecondary\s+(?:region|data\s*cent(?:er|re)|site)\b",
            r"\btest\s+failover\b",
        ],
        0.96,
    ),
    (
        "security_controls",
        [
            r"\bleast\s+privilege\b",
            r"\bservice\s+controls\b",
            r"\bsecrets?\s+(?:live|lives|are\s+kept|are\s+stored)\s+in\b",
            # Where credentials are kept: "The Stripe and Postmark keys are in
            # Heroku config vars" went to Environments.
            r"\b(?:api\s+)?(?:keys|credentials|passwords|tokens|connection\s+strings)\s+(?:live|are(?:\s+(?:kept|stored|held))?)\s+in\b",
            r"\b(?:api\s+key|credential|password|token|connection\s+string)\s+(?:lives|is\s+(?:kept|stored|held))\s+in\b",
            r"\b(?:production\s+|console\s+)?access\s+(?:goes|is)\s+(?:through|via)\b",
            r"\bhardware\s+(?:security\s+)?keys?\b",
        ],
        0.95,
    ),
    (
        # Security controls named as such: "everything sits behind the
        # corporate firewall", "the VM uses a managed identity". Below the
        # day-one checklist (0.95), whose access lists name the same things.
        "security_controls",
        [
            r"\bfirewalls?\b",
            r"\bmanaged\s+identit(?:y|ies)\b",
            r"\bservice\s+principals?\b",
            r"\bencrypt(?:ed|ion|s)?\b",
            r"\bprivate\s+(?:endpoints?|link)\b",
            r"\bnetwork\s+polic(?:y|ies)\b",
            r"\biam\s+(?:roles?|polic(?:y|ies))\b",
            r"\brbac\b",
        ],
        0.945,
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


# The receiver's short question names the topic of the answer it is merged
# with (dialogue.merge_question_answers). "Security? All traffic is TLS,
# partners authenticate with API keys ... task roles ..." went to Ownership
# (roles), and "What goes wrong? The NHS app integration times out now and
# then" to Monitoring. Above every phrase rule: the answer is about what was
# asked, whatever else it mentions.
QUESTION_TOPIC_CONFIDENCE = 0.98
_SPEAKER_PREFIX_RE = re.compile(r"^\s*[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)?\s*:\s+")
_QUESTION_TOPICS = [
    ("ownership_escalation", r"\bwho\s+(?:owns|is\s+responsible|looks\s+after|do\s+(?:i|we|you)\s+(?:call|contact|ask|"
                             r"escalate)|to\s+(?:call|contact|escalate))\b|\bown(?:s|er|ers|ership)\b|\bescalat\w*"),
    ("common_failures", r"\bwhat\s+(?:usually\s+|normally\s+|typically\s+)?(?:goes|go|can\s+go)\s+wrong\b|"
                        r"\bwhat\s+(?:usually\s+|normally\s+)?(?:breaks|fails)\b|"
                        r"\b(?:common|usual|typical|frequent|recurring)\s+(?:problems?|issues?|failures?|incidents?)\b"),
    ("deployment_and_rollback", r"\brelease\w*|\bdeploy\w*|\broll\s*back\w*|\bci\s*/?\s*cd\b"),
    ("disaster_recovery", r"\bbackups?\b|\bbacked\s+up\b|\bdisaster\b|\bdr\b|\brecover\w*|\brestor\w*|\b(?:rto|rpo)\b|"
                          r"\bfailover\b"),
    ("monitoring_observability", r"\bmonitor\w*|\balert\w*|\bdashboards?\b|\bobservab\w*|\bmetrics?\b|\blogging\b"),
    ("danger_zones", r"\brisky\b|\brisks?\b|\bdanger\w*|\bshould(?:n'?t|\s+not)\s+(?:i\s+|we\s+)?(?:touch|change)\b"),
    ("known_bad_days", r"\bbusy\s+(?:dates?|days?|times?|periods?)\b|\bpeak\s+(?:times?|periods?|days?)\b|\bfreezes?\b|"
                       r"\bbad\s+days?\b"),
    ("environments", r"\benvironments?\b|\bstaging\b"),
    ("security_controls", r"\bsecur\w*|\bsecrets?\b|\bpermissions?\b|\bcompliance\b|\bencrypt\w*"),
    ("cost_optimization", r"\bcosts?\b|\bspend\w*|\bbill(?:s|ing)?\b|\bbudget\w*"),
    ("open_responsibilities", r"\bunfinished\b|\bopen\s+(?:items?|work|tasks?)\b|\bleft\s+to\s+do\b|\boutstanding\b"),
    ("day1_survival_checklist", r"\bstart(?:ing)?\s+out\b|\bget(?:ting)?\s+started\b|\bfirst\s+day\b|\bday\s+(?:one|1)\b"),
    ("first_30_day_ownership", r"\bfirst\s+(?:month|30\s+days|thirty\s+days)\b"),
]
_QUESTION_TOPIC_RES = [(sid, re.compile(rx, re.IGNORECASE)) for sid, rx in _QUESTION_TOPICS]


def question_topic(text: str) -> Optional[SectionRuleMatch]:
    """The section a short opening question asks about, when it is answered
    in the same unit ("Backups? Cosmos has continuous backup ...")."""
    body = _SPEAKER_PREFIX_RE.sub("", text or "", count=1)
    match = re.match(r"([^?.!]{1,90})\?\s*(.*)", body, re.S)
    if not match:
        return None
    question, answer = match.group(1), _SPEAKER_PREFIX_RE.sub("", match.group(2), count=1)
    if len(question.split()) > 10 or len(re.findall(r"[A-Za-z0-9]+", answer)) < 3:
        return None
    hits = [(m.start(), sid, rx.pattern) for sid, rx in _QUESTION_TOPIC_RES for m in [rx.search(question)] if m]
    if not hits:
        return None
    _, section_id, pattern = min(hits)
    return SectionRuleMatch(section_id=section_id, confidence=QUESTION_TOPIC_CONFIDENCE,
                            reason=f"question:{section_id}", matched_pattern=pattern)


# ============================================================================
# Topic segments
# ============================================================================
#
# A KT moves through topics, and the speaker usually names each one: the
# receiver asks ("What breaks most often?"), notes read aloud have headings
# ("Known problems."), and a talk fronts the topic ("Costs, Snowflake is most
# of it.", "Security wise, ..."). The sentences that follow are about that
# topic even when they carry no signal of their own: "Only six clinics have
# it." (after "Anything unfinished?"), "Spread the sends over an hour.",
# "Run the purge job, it is safe in business hours." Scored alone, such a
# sentence went to whichever section a stray word suggested (Disaster
# Recovery for "an hour"). segment_topics() gives every unit the topic most
# recently named; the classifier adds a small boost for it (context_mapper.
# TOPIC_SEGMENT_BOOST), so a sentence with clear evidence of its own still
# goes elsewhere, and routing rules still decide over it.

# What a short question, heading or fronted phrase names, beyond the
# question words above: "What is X?", "Where does it run?", "Overview.",
# "Known problems.", "Calendar.", "First steps.", "Unfinished: ...".
_SEGMENT_EXTRA_TOPICS = [
    ("system_overview", r"\bwhat\s+is\b|\bwhat\s+does\s+(?:it|\w+)\s+do\b|\bwho\s+uses\b|\boverview\b|"
                        r"\bhow\s+(?:many|much|big)\b|\bpurpose\b"),
    ("architecture_reference", r"\barchitecture\b|\bwhere\s+does\s+(?:it|\w+)\s+run\b|\bbuilt\s+(?:on|with)\b|\bstack\b|"
                               r"\bcomponents?\b|\bhow\s+does\s+(?:it|\w+)\s+work\b|\bmessaging\b|\binfrastructure\b|"
                               r"\bdesign\b|\bdata\s+flow\b"),
    ("common_failures", r"\b(?:known\s+)?(?:problems?|issues?|incidents?|failures?|outages?)\b|\bbreaks?\b|\bgoes\s+wrong\b"),
    ("danger_zones", r"\bdo\s+not\s+touch\b|\bdon'?t\s+touch\b|\bnever\s+do\b|\bshould\s+(?:i\s+)?never\b|\bwatch\s+out\b|"
                     r"\brestrictions?\b"),
    ("known_bad_days", r"\bcalendar\b|\bpeak\s+(?:season|times?)\b|\bbusy\b|\bblackout\b|\bchange\s+freeze\b"),
    ("deployment_and_rollback", r"\brelease\s+process\b|\bpipeline\b"),
    ("day1_survival_checklist", r"\bfirst\s+(?:steps?|week|things?)\b|\bonboarding\b|\baccess\b"),
    ("open_responsibilities", r"\bopen\s+items?\b|\bpending\b|\bhalf\s+(?:done|finished)\b|\bin\s+(?:flight|progress)\b"),
    ("ownership_escalation", r"\bowners?\b|\bownership\b|\bwho\s+do\s+i\s+go\s+to\b|\bcontacts?\b"),
    ("security_controls", r"\bsecurity\b|\baccess\s+control\b|\bauth\w*\b"),
    ("disaster_recovery", r"\bbackups?\b|\bdr\b|\brecovery\b"),
    ("cost_optimization", r"\bcosts?\b"),
]
_SEGMENT_TOPIC_RES = _QUESTION_TOPIC_RES + [(sid, re.compile(rx, re.IGNORECASE)) for sid, rx in _SEGMENT_EXTRA_TOPICS]
_SEGMENT_QUESTION_RE = re.compile(r"^([^?.!]{1,90})\?")
# A heading is a short noun phrase on its own: "Release process.", "Do not
# touch.", "Backups and DR." A short sentence with a verb or a subject
# ("Secrets in CyberArk." passes; "Deployments are pretty simple." does not)
# is a statement, not a heading.
_SEGMENT_HEADING_RE = re.compile(r"^\W*([A-Za-z][\w &/'-]{0,40}?)\s*[.:]\s*$")
_SEGMENT_NOT_HEADING_RE = re.compile(
    r"^(?:the|a|an|it|we|you|i|they|there)\b|\b(?:is|are|was|were|be|been|has|have|had|does|do|did|owns?|runs?|"
    r"goes|uses?|gets?|takes?|needs?|keeps?)\b", re.IGNORECASE)
# A fronted topic: "Costs, ...", "Security wise, ...", "Going back to
# failures, ...", "For recovery, ...", "Unfinished: ...".
_SEGMENT_FRONTED_RE = re.compile(
    r"^(?:(?:so|and|okay|ok|um+|uh+|right|now|alright|then)[\s,]+)*"
    r"(?:(?:going\s+back\s+to|back\s+to|moving\s+on\s+to|on|for|in\s+terms\s+of|as\s+for|regarding)\s+)?"
    r"((?:the\s+)?[A-Za-z][\w/&'-]*(?:\s+(?:and\s+)?[A-Za-z][\w/&'-]*){0,2}?)"
    r"(?:\s+(?:wise|side))?\s*[,:]", re.IGNORECASE)


_NUMBERED_HEADING_RE = re.compile(
    r"^\W*(?:section|part|chapter|item|point|topic)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b\s*[,:.-]\s*", re.IGNORECASE)


def _segment_topic_of(phrase: str) -> Optional[str]:
    hits = [(m.start(), sid) for sid, rx in _SEGMENT_TOPIC_RES for m in [rx.search(phrase)] if m]
    return min(hits)[1] if hits else None


def _heading_body(text: str) -> str:
    body = _SPEAKER_PREFIX_RE.sub("", text or "", count=1).strip()
    # A document read aloud numbers its headings: "Section eleven, costs."
    return _NUMBERED_HEADING_RE.sub("", body, count=1) or body


def heading_phrase(text: str) -> Optional[str]:
    """The phrase of a unit that is a heading naming a topic ("Backups and
    DR.", "Section two, architecture." gives "architecture"), else None."""
    match = _SEGMENT_HEADING_RE.match(_heading_body(text))
    if match and len(match.group(1).split()) <= 4 and not _SEGMENT_NOT_HEADING_RE.search(match.group(1)) \
            and _segment_topic_of(match.group(1)):
        return match.group(1)
    return None


def topic_anchor(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """(kind, section) when the unit names a topic: a short question, a
    heading or a fronted topic. A question naming nothing ("Anything else?")
    gives (\"question\", None): the topic carries on."""
    label = _SPEAKER_PREFIX_RE.match(text or "")
    if label:
        # "Unfinished: the migration ..." reads like a speaker label; a
        # label that names a topic is a fronted topic, not a person.
        topic = _segment_topic_of(label.group(0).strip().rstrip(":"))
        if topic:
            return ("fronted", topic)
    body = _heading_body(text)
    match = _SEGMENT_QUESTION_RE.match(body)
    if match and len(match.group(1).split()) <= 10:
        return ("question", _segment_topic_of(match.group(1)))
    phrase = heading_phrase(text)
    if phrase is not None:
        return ("heading", _segment_topic_of(phrase))
    match = _SEGMENT_FRONTED_RE.match(body)
    if match and len(match.group(1).split()) <= 4:
        topic = _segment_topic_of(match.group(1))
        if topic:
            return ("fronted", topic)
    return None


# What a statement about recovery says: backups, restores, failover,
# replicas, retention, RTO/RPO, rebuilding. Disaster Recovery's examples
# are full of durations ("RTO is four hours", "kept for 35 days"), so a fix
# "for an hour" or a job "safe in business hours" scored closest to it with
# none of these words. Over 90% of the annotated recovery facts use one.
RECOVERY_EVIDENCE_RE = re.compile(
    r"\b(?:backups?|backs?\s+up|backed\s+up|restor\w*|recover\w*|disaster|dr|rto|rpo|failover|fail(?:s|ed)?\s+over|"
    r"replica\w*|replicat\w*|snapshots?|point[\s-]in[\s-]time|pitr|retention|retain\w*|versioning|versioned|"
    r"time\s+travel|second\s+(?:site|region|data\s*cent(?:re|er))|cross[\s-]region|multi[\s-]region|regions?|"
    r"standby|active[\s-](?:passive|active)|geo[\s-]\w+|archiv\w*|rebuil[dt]|recreat\w*|drills?|data\s+loss|"
    r"lost\s+(?:\w+\s+){0,2}(?:data|changes|transactions|writes|events)|cop(?:y|ies|ied)\s+to|rman|"
    r"(?:kept|stays?|stored)\s+(?:\w+\s+){0,3}for\s+(?:a|one|\d+|\w+)\s+(?:years?|months?|days?))\b",
    re.IGNORECASE,
)

# Before any topic is named, a KT is saying what the system is.
SEGMENT_OPENING_TOPIC: Optional[str] = "system_overview"
# A unit a routing rule claims at or above this confidence (the override
# level, RULE_OVERRIDE_CONFIDENCE) also moves the topic there ("The thing
# that goes wrong most is ..."). None: only named topics move it.
SEGMENT_RULE_ANCHOR_MIN: Optional[float] = 0.94


def segment_topics(texts: List[str]) -> List[Optional[str]]:
    """For each unit, the topic most recently named at or before it."""
    topics: List[Optional[str]] = []
    current: Optional[str] = SEGMENT_OPENING_TOPIC
    for text in texts:
        anchor = topic_anchor(text)
        if anchor is not None and anchor[1] is not None:
            current = anchor[1]
        elif anchor is None and SEGMENT_RULE_ANCHOR_MIN is not None:
            rule = match_section_rules(text)
            if rule is not None and rule.confidence >= SEGMENT_RULE_ANCHOR_MIN \
                    and rule.matched_pattern not in _NOT_TOPIC_ANCHORS:
                current = rule.section_id
        topics.append(current)
    return topics


def match_section_rules(text: str) -> Optional[SectionRuleMatch]:
    """Return the best rule-based section match for a sentence, if any."""
    if not text or not text.strip():
        return None
    asked = question_topic(text)
    if asked:
        return asked

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
    r"for|joining|coming|being|here|today|and|"
    # Session openers: "Let's start." / "Okay, let's get started." On a GCP
    # KT, "Let's start." was scored into Disaster Recovery and the
    # architecture sentences after it followed it there.
    r"let'?s|let\s+us|start|started|begin|get|kick|off|go|ahead|dive|in)"
)
# Closers that carry no knowledge: "That is all we had time for today."
_CLOSING_ONLY = (
    # "I think that's most of it." ended a short KT and was printed under
    # Architecture as "Additional details".
    r"(?:(?:so|okay|ok|well|right|alright|all\s+right|um+|uh+|and)[\s,]+)*(?:i\s+(?:think|guess|believe)\s+)?"
    r"(?:that'?s|that\s+is)\s+(?:about\s+|pretty\s+much\s+|basically\s+)?(?:all|it|everything|most\s+of\s+it)"
    r"(?:\s+(?:we\s+had\s+time\s+for|for\s+(?:today|now)|from\s+(?:me|my\s+side)|(?:that\s+)?i\s+(?:have|had|"
    r"(?:'ve|\s+have)\s+got)(?:\s+for\s+(?:you|today))?))?(?:\s+today)?"
    # A presenter's aside on a recorded screen share, with nothing after it:
    # "I will stop sharing.", "And I'll use this as an example.", "Let's go
    # take a look." (public SRE recordings, 27 Oct: printed under Danger
    # Zones and System Overview). "Let's take a look at the metrics
    # catalog" names something and stays.
    r"|(?:(?:so|okay|ok|well|right|alright|all\s+right|now|and|um+|uh+)[\s,]+)*"
    r"(?:(?:i(?:'ll|\s+will|\s+am\s+going\s+to|'m\s+going\s+to|'m\s+gonna)|let\s+me|let'?s)\s+"
    r"(?:go\s+(?:ahead\s+and\s+)?|just\s+)?"
    r"(?:stop\s+(?:sharing|the\s+recording|recording)(?:\s+(?:my\s+|the\s+)?screen)?(?:\s+now)?"
    r"|share\s+(?:my|the)\s+screen(?:\s+now)?"
    r"|take\s+a\s+(?:quick\s+)?look(?:\s+at(?:\s+(?:it|this|that|them|these|those))?)?(?:\s+(?:here|there|now))?"
    r"|show\s+you(?:\s+(?:this|that|it|here|now))?"
    r"|use\s+(?:this|that|it)\s+as\s+(?:an\s+)?example))"
    r"|(?:can|could)\s+(?:you|everyone|everybody)\s+(?:all\s+)?see\s+my\s+screen(?:\s+(?:now|okay))?"
    # "That concludes the GCP data platform." ends the session; it is not a
    # statement that the handover is complete (the LLM inferred "Complete"
    # from it). Only when what concludes is the session or the thing handed
    # over: "That concludes the rollback procedure." is a fact. One that
    # states readiness is kept.
    r"|(?:so\s+)?(?:that|this)\s+(?:concludes|wraps\s+up|ends)\s+"
    r"(?![^.?!]*\b(?:complete|ready|signed|sign-?off|accept\w*|confirm\w*|checklist|criteria)\b)"
    r"(?:(?:the|our|my|this|today'?s)\s+)?(?:[\w/&'\-]+\s+){0,8}?"
    r"(?:session|kt|handover|hand-?off|walk-?through|overview|knowledge\s+transfer|presentation|call|meeting|demo|"
    r"platform|system|application|service|stack)(?:\s+(?:for\s+)?today)?"
)
_GREETING_ONLY_RE = re.compile(
    r"^\W*(?:" + _GREETING_WORD + r"(?:[\s,!.]+" + _GREETING_WORD + r")*|" + _CLOSING_ONLY + r")\W*$",
    re.IGNORECASE,
)


# "<Name> is our data pipeline that ...", "This is the claims platform
# which ...": the system defined by its kind.
_DEFINES_THE_SYSTEM_RE = re.compile(
    r"^\W*(?:(?:so|okay|ok|alright|right|well|now|and|um|uh)\W+)*(?:[\w.&+-]+\s+){1,3}?(?:is|are)\s+"
    r"(?:our|the|a|an)\s+(?:[\w/&.+-]+\s+){0,4}?(?:platform|pipeline|service|system|application|app|backend|tool|"
    r"product|api|portal|engine|site|website)\b", re.IGNORECASE)


def _source_of(cs) -> str:
    return (getattr(cs.sentence, "source_text", None) or cs.sentence.text or "").strip()


def apply_rule_overrides(classified_sentences, schema_metadata: Dict[str, Dict]) -> None:
    """
    Apply deterministic routing and pull misplaced sentences out of system_overview.
    Mutates classified_sentences in place.
    """
    from context_mapper import Classification, ClassifiedSentence, ExplainabilityLog
    from datetime import datetime

    # Before any topic is named, the KT is saying what the system is.
    opening = True
    for idx, cs in enumerate(classified_sentences):
        text = cs.sentence.text or ""
        if _GREETING_ONLY_RE.match(without_label(text)) or says_nothing_specific(text):
            # "Alright, welcome." carries no knowledge; similarity scoring
            # still gave it a section, and Sign-off rendered it. Nor does a
            # remark like "It does the job." (recorded talks, 27 Oct).
            cs.primary_classification = None
            cs.secondary_classifications = []
            cs.multi_section_assignments = []
            cs.is_unassigned = True
            continue
        rule_match = match_section_rules(text)
        if rule_match is not None and rule_match.section_id == "architecture_reference" \
                and _DEFINES_THE_SYSTEM_RE.match(_source_of(cs)) and opening:
            # "Brightline is our data pipeline that loads sales and inventory
            # data from the stores into Snowflake": the KT's opening sentence
            # says what the system is; the data-flow rule took it to
            # Architecture (dev d06). Only before any topic is named:
            # "Kafka is our event pipeline that ..." later on is a component.
            rule_match = SectionRuleMatch(section_id="system_overview", confidence=RULE_OVERRIDE_CONFIDENCE,
                                          reason="rule:system_overview", matched_pattern=_DEFINES_THE_SYSTEM_RE.pattern)
        if topic_anchor(text) is not None or (rule_match is not None
                                              and rule_match.confidence >= SEGMENT_RULE_ANCHOR_MIN
                                              and rule_match.section_id != "system_overview"):
            opening = False

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
    r"^\s*(?:the\s+(?:\w+\s+)?(?:reason|root\s+cause|cause|symptom|fix|remediation|remedy|resolution|solution|"
    r"corrective\s+action|workaround|mitigation)\b|(?:this|that)\s+(?:was|is)\s+because\b"
    # When or how often it happens: "Redis evicts sessions under memory
    # pressure. That happens during peak traffic." (adversarial matrix B05).
    r"|(?:this|that|it)\s+(?:usually\s+|mostly\s+|only\s+|typically\s+|normally\s+|often\s+|always\s+|also\s+)?"
    r"(?:happens|happened|occurs|occurred|tends\s+to\s+happen|shows\s+up|kicks\s+in)\b"
    # What follows it: "When that happens the API keeps serving yesterday's
    # recommendations, ..." (dev d02; it went to Monitoring).
    r"|(?:when|whenever|if|once|after)\s+(?:this|that|it)\s+(?:happens|occurs|fails|breaks|goes\s+down)\b)",
    re.IGNORECASE,
)
# How far back an ordinal item may look for its enumeration head.
_ENUMERATION_WINDOW = 6
# An item keeps its own placement only when it independently matches a rule
# for one of these sections (an explicit "never modify ..." prohibition).
_CONTINUATION_YIELDS_TO_SECTIONS = frozenset({"danger_zones"})
# Openers that mark a clause cut from a longer sentence.
_FRAGMENT_START_RE = re.compile(r"^(?:and|or|but|then|so|while|which|whereas|nor|plus)\b", re.IGNORECASE)
# A procedure said one step per sentence ("To roll back, first pause the
# sync." "Then revert the image tag." "After that, resume the sync and watch
# the error rate."): a step opens with a sequence word and gives an
# instruction, and goes where the step before it went. Classified alone, the
# last step of a rollback went to Disaster Recovery (adversarial matrix I02).
_STEP_OPENER_RE = re.compile(
    r"^\W*(?:and\s+)?(?:then|next|after\s+(?:that|this)|afterwards|finally|lastly|second(?:ly)?|third(?:ly)?"
    r"|(?:once|when)\s+(?:that|this|it)(?:'s|\s+is|\s+has\s+been)?\s+(?:done|finished|complete|completed|back|up))"
    r"\b[\s,]*", re.IGNORECASE)
_FIRST_STEP_RE = re.compile(
    r"(?<!\bthe\s)(?<!\ba\s)\b(?:first(?:ly)?|step\s+(?:one|1))\b[\s,]+(?:you\s+)?(?P<word>[a-z]+)", re.IGNORECASE)
# Words that do not open an instruction: "Then the backups run nightly",
# "Next, we have the monitoring side".
_NOT_AN_INSTRUCTION = frozenset("""the a an this that these those there here it its our their my his her we i they
he she is are was were has have had of for time thing step one""".split())


# A fix said right after the failure it fixes ("The printer queue backs up
# when a warehouse goes offline." "Clear the queue from the admin page, and
# if that doesn't help, call the warehouse IT team."): its "call ... team"
# sent it to Ownership, and Common Failures reported no fix (KT-taker audit,
# dev d05; matrix T01). An instruction that opens with a remedy verb.
_REMEDY_START_RE = re.compile(
    r"^\W*(?:(?:then|so|just|first)\W+)?(?:clear|reset|restart|rerun|re-run|retry|purge|flush|spread|drain|scale"
    r"|fail\s+over|failover|roll\s+back|revert|redeploy|reboot|kill|delete|increase|raise|bump|rotate|renew|swap"
    r"|switch|resubmit|replay|reprocess|requeue|free|clean|recreate|rebuild|restore|check|run|stop|disable|enable)\b",
    re.IGNORECASE)
# "Oracle sessions table grows and slows logins" is a failure too: the fix
# after it ("Run the purge job, ...") went to the calendar (dev d04).
_FAILURE_RE = re.compile(
    r"\b(?:fail\w*|error\w*|outage|time\s?outs?|crash\w*|down|drops|dropped|backs\s+up|backed\s+up|offline|stuck"
    r"|hang\w*|reject\w*|lag\w*|exhaust\w*|leak\w*|slow\w*|break\w*|broken|problem|issue|throttl\w*|stall\w*"
    r"|saturat\w*|fill(?:s|ed)?\s+up|runs?\s+out\s+of|ran\s+out\s+of|stale|degrad\w*)\b", re.IGNORECASE)


def _gives_instruction(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text or "")
    if words and words[0].lower() == "you":
        words = words[1:]
    return bool(words) and words[0].lower() not in _NOT_AN_INSTRUCTION


def _is_step(text: str) -> bool:
    """`text` is a step of a procedure: an instruction, the first step of
    one ("To roll back, first pause ..."), or a step that follows one."""
    from fact_units import _is_imperative

    opener = _STEP_OPENER_RE.match(text or "")
    if opener and _gives_instruction(text[opener.end():]):
        return True
    first = _FIRST_STEP_RE.search(text or "")
    return bool(first and first.group("word").lower() not in _NOT_AN_INSTRUCTION) or _is_imperative(text or "")


def _assign_section(cs, section_id, schema_metadata, idx, reason, Classification, ExplainabilityLog, datetime) -> None:
    """Place a unit in a named section for a continuation reason (kept by
    restore_sections_emptied_by_topic, which reconsiders only the classifier's
    own picks)."""
    cs.primary_classification = Classification(
        section_id=section_id,
        section_title=(schema_metadata.get(section_id) or {}).get("title", section_id),
        confidence=RULE_OVERRIDE_CONFIDENCE,
        similarity_score=RULE_OVERRIDE_CONFIDENCE,
        reason=reason,
    )
    cs.is_unassigned = False
    cs.explainability_log = ExplainabilityLog(
        action="continuation",
        timestamp=datetime.utcnow().isoformat() + "Z",
        sentence_id=idx,
        section_id=section_id,
        reasoning=reason,
        confidence=RULE_OVERRIDE_CONFIDENCE,
    )


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
        # "Security wise, the service account only has read access to
        # BigQuery, and the API sits behind Identity-Aware Proxy for the admin
        # endpoints": the speaker named the topic of the whole sentence, and
        # each fact cut from it belongs there, unless a rule of its own
        # claims it (a never-do rule stays a never-do rule). The second fact
        # went to Architecture, and "... the certs are rotated every year by
        # the roadside team" after "security," to Ownership (dev d02, d03).
        fronted = topic_anchor((getattr(cs.sentence, "source_text", None) or text).strip())
        if fronted and fronted[0] == "fronted" and fronted[1] and fronted[1] in schema_metadata:
            own = match_section_rules(text)
            current = getattr(cs.primary_classification, "section_id", None)
            if current != fronted[1] and not (own and own.section_id != fronted[1]
                                              and own.confidence >= RULE_OVERRIDE_CONFIDENCE):
                _assign_section(cs, fronted[1], schema_metadata, idx,
                                "Continuation: the topic the speaker named for this sentence",
                                Classification, ExplainabilityLog, datetime)
            continue
        # Half of a two-topic sentence: split off because its own rule claims
        # it for another section. Its raw text starts lowercase ("all secrets
        # are stored in Key Vault"), which read as a cut-off fragment and sent
        # it back to the section it had been split from.
        if getattr(cs.sentence, "split_clause", False):
            continue
        own_rule = match_section_rules(text)
        # Only an explicit safety prohibition outranks the sentence's
        # grammatical dependence on its antecedent. A passing keyword (e.g.
        # "automated rollback" in "The third is to review which services
        # have automated rollback") must not pull an enumerated task away
        # from the list it belongs to.
        if own_rule and own_rule.section_id in _CONTINUATION_YIELDS_TO_SECTIONS:
            continue

        opener = _STEP_OPENER_RE.match(text)
        if opener and idx > 0 and _gives_instruction(text[opener.end():]):
            prev = classified_sentences[idx - 1]
            if prev.primary_classification and _is_step((prev.sentence.text or "").strip()):
                # Recorded as a continuation even when it is already there
                # (by the topic boost): restore_sections_emptied_by_topic()
                # reconsiders only the classifier's own picks, and took the
                # last step of a rollback away to an empty section.
                _inherit_section(
                    cs, prev, idx, "Continuation: the next step of the procedure before it",
                    Classification, ExplainabilityLog, datetime,
                )
                continue

        if idx > 0 and _REMEDY_START_RE.match(text):
            prev = classified_sentences[idx - 1]
            if (prev.primary_classification and prev.primary_classification.section_id == "common_failures"
                    and _FAILURE_RE.search(prev.sentence.text or "")):
                if getattr(cs.primary_classification, "section_id", None) != "common_failures":
                    _inherit_section(
                        cs, prev, idx, "Continuation: the fix for the failure before it",
                        Classification, ExplainabilityLog, datetime,
                    )
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


_OBSERVABILITY_NOUN = re.compile(
    r"\b(?:alert\w*|alarms?|dashboards?|metrics?|monitor\w*|observability|tracing|traces?|apm|slos?|slis?|uptime)\b",
    re.IGNORECASE,
)
_MONITORING_TOOLS: Optional[re.Pattern] = None


def _is_monitoring_entity(value: str) -> bool:
    """GLiNER's "monitoring" label is loose: it tagged the vans' "readings"
    and "Oracle archive logs" (database redo logs), and its 0.10 boost then
    outweighed the 0.01-0.07 margins between sections, so a failure and its
    fix were filed under Monitoring. Only a monitoring or alerting tool from
    the component catalog, or an observability noun, counts."""
    global _MONITORING_TOOLS
    if _MONITORING_TOOLS is None:
        from architecture_diagram import _LAYER_TERMS

        names = sorted({n for layer in ("monitoring", "alerting") for n in _LAYER_TERMS.get(layer, [])},
                       key=len, reverse=True)
        _MONITORING_TOOLS = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    return bool(_OBSERVABILITY_NOUN.search(value) or _MONITORING_TOOLS.search(value))


# A person or team named in a sentence says nothing about ownership unless
# the sentence is about owning, contacting or escalating: "Pickers use
# handheld scanners that talk to it all day" was filed under Ownership on
# the "owner" entity "Pickers".
_OWNERSHIP_LANGUAGE = re.compile(
    r"\b(?:own\w*|responsib\w*|escalat\w*|contact\w*|on-?call|page[sd]?|paging|call|ask|reach|"
    r"lead|manager|team|squad|questions?|support)\b",
    re.IGNORECASE,
)


def entity_affinity_boost(section_id: str, entities: Optional[Dict[str, List[str]]],
                          text: Optional[str] = None) -> Tuple[float, Optional[str]]:
    """Return (boost, note) if extracted entities support classifying this
    sentence into `section_id`, else (0.0, None).

    `entities` is the dict EntityExtractor.get_context_entities() returns,
    e.g. {"monitoring": ["PagerDuty"], "owner": ["Platform Team"]}. With the
    sentence `text`, an owner or escalation entity counts only when the
    sentence talks about ownership.
    """
    if not entities:
        return 0.0, None

    for entity_type, values in entities.items():
        affinity = ENTITY_TYPE_SECTION_AFFINITY.get(entity_type)
        if entity_type == "monitoring":
            values = [v for v in values or [] if _is_monitoring_entity(str(v))]
        if entity_type in ("owner", "escalation") and text is not None and not _OWNERSHIP_LANGUAGE.search(text):
            values = []
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
