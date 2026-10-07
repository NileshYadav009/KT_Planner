"""Sign-off workflow (P1-3).

A KT is signed when its named giver, receiver and approver have each
acknowledged the same version of the document, and every knowledge gap in
that version has been closed or accepted as a risk by one of them. The
signed version is locked: no further edits (document_model.py). Every step
is written to the audit log.

    draft      no participants named yet
    in_review  participants named; gaps are resolved and people acknowledge
    signed     all three acknowledged one version with no open gaps

A new version (an edit) clears the acknowledgements given so far: people
acknowledge what the document says, not what it used to say.

State lives on the job as job["signoff"]:
    participants   {"giver"|"receiver"|"approver": {"user_id", "email", "name"}}
    gaps           [{"id", "text", "resolution", "note", "by", "at"}]
    acknowledgements {role: {"user_id", "email", "at", "version"}}
    signed_version, signed_at
"""
import hashlib
import time
from typing import Any, Dict, List, Optional

ROLES = ("giver", "receiver", "approver")
ROLE_LABELS = {"giver": "KT giver (outgoing owner)", "receiver": "KT receiver (incoming owner)",
               "approver": "Approver"}
RESOLUTIONS = {"closed": "Closed", "accepted_risk": "Accepted as a risk"}
KNOWLEDGE_GAPS_TITLE_PREFIX = "Knowledge"


class SignoffError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def state(job: Dict[str, Any]) -> Dict[str, Any]:
    return job.get("signoff") or {"status": "draft", "participants": {}, "gaps": [], "acknowledgements": {}}


def is_locked(job: Dict[str, Any]) -> bool:
    return state(job).get("status") == "signed"


def open_gaps(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [g for g in state(job).get("gaps") or [] if not g.get("resolution")]


def document_gaps(knowledge_object: Dict[str, Any]) -> List[str]:
    """The knowledge gaps the document lists (KT Coverage section), as the
    reviewers left them."""
    from renderers.sections.kt_coverage import KNOWLEDGE_GAPS_TITLE

    for section in knowledge_object.get("rendered_sections") or []:
        for block in section.get("blocks") or []:
            if block.get("type") == "ChecklistBlock" and block.get("title") == KNOWLEDGE_GAPS_TITLE:
                return [str(item) for item in block.get("items") or [] if str(item).strip()]
    return []


def _gap_id(text: str) -> str:
    return "g_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _person(user: Dict[str, Any]) -> Dict[str, Any]:
    return {"user_id": user["user_id"], "email": user["email"], "name": user.get("name")}


def roles_of(job: Dict[str, Any], user_id: Optional[str]) -> List[str]:
    participants = state(job).get("participants") or {}
    return [role for role in ROLES if user_id and (participants.get(role) or {}).get("user_id") == user_id]


def start(job: Dict[str, Any], people: Dict[str, Dict[str, Any]], version: int) -> Dict[str, Any]:
    """Name the giver, receiver and approver (`people`: role -> user record)
    and take the document's knowledge gaps as the gaps to resolve. Naming
    them again replaces the participants and clears acknowledgements; gap
    resolutions already recorded are kept."""
    current = state(job)
    if current.get("status") == "signed":
        raise SignoffError("This KT is already signed", 409)
    missing = [r for r in ROLES if not people.get(r)]
    if missing:
        raise SignoffError("Name a " + ", ".join(ROLE_LABELS[r] for r in missing))
    if len({people[r]["user_id"] for r in ROLES}) < len(ROLES):
        raise SignoffError("The giver, receiver and approver must be three different people")
    resolved = {g["id"]: g for g in current.get("gaps") or [] if g.get("resolution")}
    gaps = []
    for text in document_gaps(job.get("knowledge_object") or {}):
        gap_id = _gap_id(text)
        gaps.append(resolved.get(gap_id) or {"id": gap_id, "text": text, "resolution": None})
    job["signoff"] = {"status": "in_review", "participants": {r: _person(people[r]) for r in ROLES},
                      "gaps": gaps, "acknowledgements": {}, "started_version": version}
    return job["signoff"]


def resolve_gap(job: Dict[str, Any], gap_id: str, resolution: str, note: str, actor: Dict[str, Any]) -> Dict[str, Any]:
    current = state(job)
    if current.get("status") != "in_review":
        raise SignoffError("Sign-off has not started" if current.get("status") == "draft" else "This KT is signed", 409)
    if resolution not in RESOLUTIONS:
        raise SignoffError("Resolution is 'closed' or 'accepted_risk'")
    if not isinstance(note, str) or not note.strip():
        raise SignoffError("Say how the gap was closed, or why the risk is accepted")
    gap = next((g for g in current.get("gaps") or [] if g["id"] == gap_id), None)
    if gap is None:
        raise SignoffError("Unknown gap", 404)
    gap.update(resolution=resolution, note=note.strip()[:1000], by=actor.get("email") or actor.get("user_id"),
               at=time.time())
    # Acknowledgements were given with this gap in a different state.
    current["acknowledgements"] = {}
    return gap


def acknowledge(job: Dict[str, Any], role: str, version: int, actor: Dict[str, Any], current_version: int) -> bool:
    """Record `actor`'s acknowledgement as `role` of `version`. Returns True
    when this completes the sign-off."""
    current = state(job)
    if current.get("status") != "in_review":
        raise SignoffError("Sign-off has not started" if current.get("status") == "draft" else "This KT is signed", 409)
    if role not in ROLES or (current["participants"].get(role) or {}).get("user_id") != actor.get("user_id"):
        raise SignoffError("Only the person named for this role can acknowledge it", 403)
    if int(version) != int(current_version):
        raise SignoffError(f"Acknowledge the current version ({current_version}); the document changed", 409)
    still_open = open_gaps(job)
    if still_open:
        raise SignoffError(f"{len(still_open)} knowledge gap(s) are still open: close each one or accept it as a "
                           "risk before acknowledging", 409)
    current["acknowledgements"][role] = {"user_id": actor["user_id"], "email": actor.get("email"),
                                         "at": time.time(), "version": int(version)}
    acks = current["acknowledgements"]
    if all(r in acks and acks[r]["version"] == int(version) for r in ROLES):
        current.update(status="signed", signed_version=int(version), signed_at=time.time())
        return True
    return False


def on_new_version(job: Dict[str, Any], version: int, actor: str) -> None:
    """An edit makes earlier acknowledgements stale."""
    current = job.get("signoff")
    if current and current.get("status") == "in_review" and current.get("acknowledgements"):
        current["acknowledgements"] = {}
        current["reset_by_version"] = version


def public_state(job: Dict[str, Any], user_id: Optional[str], current_version: int) -> Dict[str, Any]:
    """The sign-off as the UI shows it, with what this person may do."""
    current = state(job)
    mine = roles_of(job, user_id)
    acks = current.get("acknowledgements") or {}
    return dict(current, current_version=current_version, my_roles=mine,
                open_gaps=len(open_gaps(job)),
                can_acknowledge=[r for r in mine if r not in acks] if current.get("status") == "in_review" else [])


# --------------------------------------------------------------------------
# The Sign-off section of the document
# --------------------------------------------------------------------------

def _when(ts: Optional[float]) -> str:
    return time.strftime("%d %b %Y %H:%M UTC", time.gmtime(ts)) if ts else ""


def render_section(job: Dict[str, Any], title: str, version: int) -> Optional[Dict[str, Any]]:
    """The document's Sign-off section from the recorded sign-off, or None
    when sign-off has not started (the transcript-based section stays)."""
    current = state(job)
    if current.get("status") == "draft" or not current.get("participants"):
        return None
    acks = current.get("acknowledgements") or {}
    rows = []
    for role in ROLES:
        person = current["participants"][role]
        ack = acks.get(role)
        who = person.get("name") or person.get("email")
        rows.append({"label": ROLE_LABELS[role],
                     "value": f"{who}: acknowledged version {ack['version']} on {_when(ack['at'])}" if ack
                     else f"{who}: not yet acknowledged"})
    if current.get("status") == "signed":
        status = f"Signed: version {current['signed_version']}, {_when(current.get('signed_at'))}. This version is locked."
    else:
        waiting = [ROLE_LABELS[r] for r in ROLES if r not in acks]
        gaps = len(open_gaps(job))
        status = (f"Not signed. Document version {version}. "
                  + (f"{gaps} knowledge gap(s) still open. " if gaps else "")
                  + ("Waiting for: " + ", ".join(waiting) + "." if waiting else ""))
    blocks = [{"type": "NarrativeBlock", "title": title, "paragraphs": [status]},
              {"type": "TechnologyGrid", "title": "Sign-off record", "rows": rows}]
    if current.get("gaps"):
        blocks.append({"type": "DecisionTable", "title": "Knowledge gaps at sign-off",
                       "columns": ["Gap", "Resolution", "By", "Note"],
                       "rows": [{"Gap": g["text"], "Resolution": RESOLUTIONS.get(g.get("resolution"), "Open"),
                                 "By": g.get("by") or "", "Note": g.get("note") or ""} for g in current["gaps"]]})
    return {"section_id": "signoff", "section_title": title, "blocks": blocks}


def apply_to_document(rendered_sections: List[Dict[str, Any]], job: Dict[str, Any], version: int) -> List[Dict[str, Any]]:
    """Rendered sections with the Sign-off section replaced by the recorded
    sign-off (for the PDF and the UI). The stored document is not changed."""
    out = []
    for section in rendered_sections or []:
        if section.get("section_id") == "signoff":
            recorded = render_section(job, section.get("section_title") or "Sign-off", version)
            out.append(recorded or section)
        else:
            out.append(section)
    return out
