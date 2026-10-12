"""Runbooks from a KT (P3-5): one per known failure, as Markdown for a
repository or a wiki.

Each runbook is assembled from the KT's incident card (the Quick Reference
section, as the reviewer last saved it): how you notice the failure (the
alert guidance), what to do (the fix the KT stated, or that none was), when
to roll back, who to escalate to, and the never-do rules that name the same
component. Nothing is written that the KT did not say, so a runbook is only
as complete as the session; the gaps show as "No fix was stated".
"""
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from markdown_export import _line

ALERT, ROLLBACK, ESCALATE = "An alert fires", "A release misbehaves (roll back)", "Who to page or escalate to"
CAUTIONS = ("Never", "Be careful")
# Incident-card rows that are guidance for every failure, not a failure.
GENERAL = frozenset({ALERT, ROLLBACK, ESCALATE, *CAUTIONS, "Do not deploy", "Production activity unclear",
                     "System unavailable (impact)", "Disaster recovery", "Planning a production change"})
_CONFLICT_PREFIX = "Conflicting guidance"
# A failure a release may have caused: the rollback applies to it.
_RELEASE_RE = re.compile(r"\b(?:deploy\w*|release\w*|rollout|upgrade\w*|migration\w*|new\s+version|after\s+a\s+change)\b",
                         re.IGNORECASE)
_COMMON = frozenset("""after before during when with from into that this they them their there then than about
should would could cause causes caused issue issues error errors failure failures problem problems happens
occur occurs check checks first again never always every service services system production prod data""".split())


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", (text or "").lower()) if w not in _COMMON}


def card_rows(knowledge_object: Dict[str, Any]) -> List[Dict[str, str]]:
    """The incident card's rows as the document shows them (edits included)."""
    for section in knowledge_object.get("rendered_sections") or []:
        if section.get("section_id") != "quick_reference":
            continue
        rows: List[Dict[str, str]] = []
        for block in section.get("blocks") or []:
            if block.get("type") == "DecisionTable":
                rows += [{k: _line(v) for k, v in r.items()} for r in block.get("rows") or [] if isinstance(r, dict)]
        return rows
    return []


def runbooks(knowledge_object: Dict[str, Any]) -> Dict[str, Any]:
    """{"failures": [{situation, do, notice, rollback, escalate, careful, source}], "always": [(label, text)]}"""
    rows = card_rows(knowledge_object)
    first = {}
    for r in rows:
        first.setdefault(r.get("Situation"), r)
    cautions = [r for r in rows if r.get("Situation") in CAUTIONS]
    failures, used_cautions = [], set()
    for r in rows:
        situation = r.get("Situation") or ""
        if situation in GENERAL or situation.startswith(_CONFLICT_PREFIX) or not r.get("What to do"):
            continue
        about = _words(f"{situation} {r.get('What to do')}")
        careful = [c["What to do"] for c in cautions if about & _words(c["What to do"])]
        used_cautions.update(careful)
        rollback = (first.get(ROLLBACK) or {}).get("What to do")
        failures.append({
            "situation": situation, "do": r["What to do"], "source": r.get("Source"),
            "notice": (first.get(ALERT) or {}).get("What to do"),
            "rollback": rollback if rollback and _RELEASE_RE.search(f"{situation} {r['What to do']}") else None,
            "rollback_general": rollback,
            "escalate": (first.get(ESCALATE) or {}).get("What to do"),
            "careful": careful,
        })
    always = [(c["Situation"], c["What to do"]) for c in cautions if c["What to do"] not in used_cautions]
    always += [(r["Situation"], r["What to do"]) for r in rows
               if r.get("Situation") == "Do not deploy" or (r.get("Situation") or "").startswith(_CONFLICT_PREFIX)]
    return {"failures": failures, "always": always}


def runbook_markdown(job_id: str, job: Dict[str, Any], created: Optional[float] = None) -> str:
    ko = job.get("knowledge_object") or {}
    system = _line(ko.get("system_name") or job.get("title") or "the system")
    books = runbooks(ko)
    date_str = (datetime.fromtimestamp(created) if created else datetime.now()).strftime("%d %B %Y")
    out = [f"# Runbooks: {system}", "",
           f"From the KT of {date_str} (job {job_id[:8].upper()}, version {int(job.get('document_version') or 1)}). "
           "Every step is a line the KT stated, taken from its incident card; nothing was added. A failure the KT "
           "gave no fix for says so. Check them with the outgoing owner before relying on them.", ""]
    if not books["failures"]:
        out += ["The KT named no failures with symptoms, so there is nothing to build a runbook from. Ask the "
                "outgoing owner for the incidents that recur.", ""]
    if books["always"]:
        out += ["## Rules that always apply", ""]
        out += [f"- **{_line(label)}:** {_line(text)}" for label, text in books["always"]] + [""]
    for n, f in enumerate(books["failures"], start=1):
        out += [f"## {n}. {f['situation'][:1].upper()}{f['situation'][1:]}", ""]
        if f["notice"]:
            out.append(f"- **How you notice:** {f['notice']}")
        out.append(f"- **What to do:** {f['do']}")
        if f["rollback"]:
            out.append(f"- **If a release caused it, roll back:** {f['rollback']}")
        if f["escalate"]:
            out.append(f"- **Escalate to:** {f['escalate']}")
        out += [f"- **Careful:** {c}" for c in f["careful"]]
        if f["source"]:
            out.append(f"- *From: {f['source']}*")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
