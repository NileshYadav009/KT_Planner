"""Multi-session KTs (P2-1).

A real transition is 5-20 sessions, and one recording used to make one
document. Here a finished KT takes follow-up sessions: a recording or a
pasted transcript, each kept with its own times and speakers. Adding one
rebuilds the document from every session as the next version of the same
KT, so:

  * facts from all sessions sit in one document, and each source says which
    session it came from ("S2 · 03:15");
  * the knowledge gaps left after a session are the agenda for the next one
    (agenda()), and a gap a later session covers disappears from the
    document; the version says which gaps it closed;
  * the KT keeps its workspace, owner, version history and sign-off record
    (acknowledgements reset, as for any new version); a signed KT takes no
    more sessions.

Reviewer edits to the previous version are not replayed onto the rebuilt
document: that version stays in History (and exports) as it was.
"""
from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, List, Optional, Tuple

MAX_SESSIONS = 20
# Job fields that belong to the KT, not to one pipeline run.
_KT_FIELDS = ("tenant_id", "created_by", "created_by_id", "signoff", "document_version", "title")


class SessionError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def sessions_of(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The KT's sessions; a KT from before sessions existed is session 1."""
    if job.get("sessions"):
        return copy.deepcopy(job["sessions"])
    return [{"n": 1, "kind": "audio" if job.get("segments") else "paste", "transcript": job.get("transcript") or "",
             "segments": job.get("segments"), "time_offset": float(job.get("time_offset") or 0.0)}]


def combined_input(sessions: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """One transcript and one segment list for the pipeline. Each session's
    times stay its own (its leading-silence offset folded in); a pasted
    session is one untimed segment, the same shape a single paste gets."""
    multi = len(sessions) > 1
    parts, segments = [], []
    for session in sessions:
        tag = {"session": session["n"]} if multi else {}
        if session.get("transcript", "").strip():
            parts.append(session["transcript"].strip())
        if session.get("segments"):
            offset = float(session.get("time_offset") or 0.0)
            for seg in session["segments"]:
                segments.append(dict(seg, start=float(seg.get("start") or 0.0) + offset,
                                     end=float(seg.get("end") or 0.0) + offset, **tag))
        elif session.get("transcript", "").strip():
            segments.append({"id": len(segments), "seek": 0, "start": 0.0, "end": 0.0, "text": session["transcript"],
                             "avg_logprob": -0.3, "compression_ratio": None, "no_speech_prob": None,
                             "untimed": True, **tag})
    return "\n\n".join(parts), segments


def knowledge_gaps(job: Dict[str, Any]) -> List[str]:
    for section in (job.get("knowledge_object") or {}).get("sections") or []:
        if section.get("id") == "kt_coverage":
            return [str(g) for g in section.get("_knowledge_gaps") or [] if str(g).strip()]
    return []


def agenda(job: Dict[str, Any]) -> Dict[str, Any]:
    """What the next session should cover: the knowledge gaps left now."""
    return {"next_session": len(sessions_of(job)) + 1, "topics": knowledge_gaps(job)}


def _gap_key(gap: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", gap.lower()).strip()


def check_can_add(job: Optional[Dict[str, Any]]) -> None:
    import pipeline
    import signoff

    if job is None or job.get("status") not in pipeline.COMPLETED_STATUSES:
        raise SessionError("Only a finished KT can take another session", 409)
    if signoff.is_locked(job):
        raise SessionError("This KT is signed; start a new KT for further sessions", 409)
    if job.get("session_pending"):
        raise SessionError("A session is already being added to this KT", 409)
    if len(sessions_of(job)) >= MAX_SESSIONS:
        raise SessionError(f"A KT takes at most {MAX_SESSIONS} sessions", 409)


def add_session(job_id: str, payload: Dict[str, Any]) -> None:
    """Worker task: add one session (payload {"transcript"} or an uploaded
    recording {"input_path", "media_format"}) and rebuild the document."""
    import document_model
    import pipeline
    import signoff

    before = pipeline.JOB_QUEUE.get(job_id)
    if before is None:
        return
    before = copy.deepcopy(before)
    before.pop("session_pending", None)
    actor = str(payload.get("added_by") or "session")
    try:
        sessions = sessions_of(before)
        n = len(sessions) + 1
        session: Dict[str, Any] = {"n": n, "added_at": time.time(), "added_by": actor}
        warnings: List[str] = []
        notices: List[str] = []
        if payload.get("input_path"):
            rec = pipeline.transcribe_recording(job_id, payload["input_path"], payload.get("media_format"),
                                                capture_screen=False)
            session.update(kind="audio", transcript=rec["transcript"], segments=rec["segments"],
                           time_offset=rec["time_offset"])
            warnings += [f"Session {n}: {w}" for w in rec["warnings"]]
            notices += [f"Session {n}: {x}" for x in rec["notices"]]
        else:
            session.update(kind="paste", transcript=str(payload.get("transcript") or ""), segments=None,
                           time_offset=0.0)
            warnings += [f"Session {n}: {w}" for w in payload.get("warnings") or []]
        sessions.append(session)

        # The document as it was stays in History as its own version.
        document_model.ensure_first_version(pipeline, job_id, before)
        transcript, segments = combined_input(sessions)
        pipeline.run_kt_pipeline(job_id, transcript, segments, warnings=warnings, notices=notices)
        after = pipeline.JOB_QUEUE.get(job_id) or {}
        if after.get("status") not in pipeline.COMPLETED_STATUSES:
            raise RuntimeError(after.get("error") or "the document could not be rebuilt")

        remaining = {_gap_key(g) for g in knowledge_gaps(after)}
        closed = [g for g in knowledge_gaps(before) if _gap_key(g) not in remaining]
        for field in _KT_FIELDS:
            if field in before:
                after[field] = before[field]
        after["sessions"] = sessions
        after["session_changes"] = {"session": n, "closed_gaps": closed}
        version = document_model.current_version(before) + 1
        summary = f"Session {n} added" + (f"; it closed {len(closed)} knowledge gap(s)" if closed else "")
        pipeline.JOB_STORE.save_version(job_id, version, actor, summary,
                                        document_model.snapshot(after.get("knowledge_object") or {}))
        after["document_version"] = version
        signoff.on_new_version(after, version, actor)
        after.setdefault("notices", []).append(
            f"Built from {n} KT sessions." + (f" Session {n} closed: " + "; ".join(closed) + "." if closed else ""))
        pipeline.JOB_QUEUE[job_id] = after
    except Exception as exc:
        # The KT stays exactly as it was; the failure is shown with it.
        before["session_error"] = f"The new session could not be added: {exc}"
        pipeline.JOB_QUEUE[job_id] = before
    finally:
        if payload.get("input_path"):
            pipeline.remove_upload(payload["input_path"])
