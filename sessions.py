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
_KT_FIELDS = ("tenant_id", "created_by", "created_by_id", "signoff", "document_version", "title", "participants")
# A speaker's name. In a pasted transcript it is a turn label, which
# speaker_roles reads as one to three capitalised words ("Priya Sharma:").
MAX_NAME = 60
_LABEL_NAME_RE = re.compile(r"[A-Z][\w'.-]*(?:\s[A-Z][\w'.-]*){0,2}")


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
    session is one untimed segment, the same shape a single paste gets, or
    one per turn when it names its speakers ("Meera: ..."), so a rebuild
    keeps who said what (speaker_roles)."""
    from speaker_roles import turn_segments

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
            turns = turn_segments(session["transcript"])
            if turns:
                segments.extend(dict(turn, id=len(segments) + i, **tag) for i, turn in enumerate(turns))
                continue
            segments.append({"id": len(segments), "seek": 0, "start": 0.0, "end": 0.0, "text": session["transcript"],
                             "avg_logprob": -0.3, "compression_ratio": None, "no_speech_prob": None,
                             "untimed": True, **tag})
    return "\n\n".join(parts), segments


def speakers_of(job: Dict[str, Any]) -> List[str]:
    """The speakers a finished KT's conversation had, in order (speaker_roles)."""
    roles = job.get("speaker_roles") or {}
    return [s.get("name") for s in roles.get("speakers") or [] if s.get("name")]


def speakers_said(job: Dict[str, Any]) -> List[str]:
    """Everyone who speaks in the KT, in order: a recording's diarised
    speakers ("Speaker 1") and a pasted transcript's labels. Unlike
    speakers_of, also one speaker alone."""
    from speaker_roles import turn_segments

    listed = job.get("sessions") or [{"transcript": job.get("transcript"), "segments": job.get("segments")}]
    names: List[str] = []
    for session in listed:
        for seg in session.get("segments") or turn_segments(session.get("transcript") or "") or []:
            name = seg.get("speaker")
            if name and name not in names:
                names.append(name)
    return names


def _check_can_rebuild(job: Optional[Dict[str, Any]]) -> None:
    if job is None:
        raise SessionError("Job not found", 404)
    if job.get("status") not in ("completed", "completed_with_warnings"):
        raise SessionError("The KT is not finished yet", 409)
    if job.get("session_pending") or job.get("giver_pending") or job.get("speakers_pending"):
        raise SessionError("The KT is already being rebuilt", 409)
    if (job.get("signoff") or {}).get("signed_version"):
        raise SessionError("A signed KT cannot be rebuilt", 409)


def check_can_set_giver(job: Optional[Dict[str, Any]], giver: Optional[str]) -> None:
    _check_can_rebuild(job)
    if giver is not None and giver not in speakers_of(job):
        raise SessionError("That person did not speak in this KT", 400)


def check_can_rename_speakers(job: Optional[Dict[str, Any]], names: Any) -> Dict[str, str]:
    """The renames to make ({"Speaker 1": "Priya Sharma"}), checked: each
    speaker spoke in the KT, each name is a name, and no two people end up
    with the same one."""
    _check_can_rebuild(job)
    if not isinstance(names, dict) or not names:
        raise SessionError("names must map each speaker to a person's name")
    speakers = speakers_said(job)
    # A pasted session has no segments, or (once rebuilt) untimed turns.
    pasted = any(all(seg.get("untimed") for seg in s.get("segments") or [])
                 for s in (job.get("sessions") or [{"segments": job.get("segments")}]))
    renames: Dict[str, str] = {}
    for old, new in names.items():
        if old not in speakers:
            raise SessionError(f"{old} did not speak in this KT")
        if not isinstance(new, str) or not new.strip():
            raise SessionError(f"Give {old} a name")
        new = " ".join(new.split())
        if len(new) > MAX_NAME or ":" in new:
            raise SessionError(f"A name has at most {MAX_NAME} characters and no colon")
        if pasted and not _LABEL_NAME_RE.fullmatch(new):
            raise SessionError("In a pasted transcript a name is one to three words, each starting with a capital letter")
        if new != old:
            renames[old] = new
    final = [renames.get(s, s) for s in speakers]
    if len(set(final)) != len(final):
        raise SessionError("Two speakers cannot have the same name")
    if not renames:
        raise SessionError("Nothing to rename")
    return renames


def set_giver(job_id: str, payload: Dict[str, Any]) -> None:
    """Worker task: rebuild a finished KT with the person the user chose as
    the one handing over (payload {"giver": a speaker, or None to detect it
    again}). Like a new session: the document as it was stays in History,
    and the KT keeps its workspace, owner and sign-off record."""
    import document_model
    import pipeline
    import signoff

    before = pipeline.JOB_QUEUE.get(job_id)
    if before is None:
        return
    before = copy.deepcopy(before)
    before.pop("giver_pending", None)
    actor = str(payload.get("set_by") or "giver")
    giver = payload.get("giver") or None
    try:
        sessions = sessions_of(before)
        document_model.ensure_first_version(pipeline, job_id, before)
        transcript, segments = combined_input(sessions)
        pipeline.run_kt_pipeline(job_id, transcript, segments, giver=giver)
        after = pipeline.JOB_QUEUE.get(job_id) or {}
        if after.get("status") not in pipeline.COMPLETED_STATUSES:
            raise RuntimeError(after.get("error") or "the document could not be rebuilt")
        for field in _KT_FIELDS:
            if field in before:
                after[field] = before[field]
        if before.get("sessions"):
            after["sessions"] = sessions
        version = document_model.current_version(before) + 1
        who = (after.get("speaker_roles") or {}).get("giver") or giver
        summary = f"Handed over by {who}" + (" (chosen)" if giver else " (detected)")
        pipeline.JOB_STORE.save_version(job_id, version, actor, summary,
                                        document_model.snapshot(after.get("knowledge_object") or {}))
        after["document_version"] = version
        signoff.on_new_version(after, version, actor)
        pipeline.JOB_QUEUE[job_id] = after
    except Exception as exc:
        before["giver_error"] = f"The document could not be rebuilt with that person handing over: {exc}"
        pipeline.JOB_QUEUE[job_id] = before


def rename_speakers(job_id: str, payload: Dict[str, Any]) -> None:
    """Worker task: rebuild a finished KT with its speakers named (payload
    {"names": {"Speaker 1": "Priya Sharma"}}). Every session is renamed, so
    later sessions and rebuilds keep the names, and so is a chosen giver.
    Like a new session: the document as it was stays in History, and the KT
    keeps its workspace, owner and sign-off record."""
    import document_model
    import pipeline
    import signoff
    from speaker_roles import rename_turn_labels

    before = pipeline.JOB_QUEUE.get(job_id)
    if before is None:
        return
    before = copy.deepcopy(before)
    before.pop("speakers_pending", None)
    actor = str(payload.get("set_by") or "speakers")
    names = dict(payload.get("names") or {})
    try:
        renamed = []
        for session in sessions_of(before):
            # A pasted turn keeps its label in its text ("Speaker 1: ...").
            session["segments"] = [
                dict(seg, speaker=names.get(seg.get("speaker"), seg.get("speaker")),
                     text=rename_turn_labels(seg.get("text") or "", names) if seg.get("untimed") else seg.get("text"))
                for seg in session.get("segments") or []] or None
            session["transcript"] = rename_turn_labels(session.get("transcript") or "", names)
            renamed.append(session)
        giver = names.get(before.get("giver"), before.get("giver"))
        document_model.ensure_first_version(pipeline, job_id, before)
        transcript, segments = combined_input(renamed)
        pipeline.run_kt_pipeline(job_id, transcript, segments, giver=giver)
        after = pipeline.JOB_QUEUE.get(job_id) or {}
        if after.get("status") not in pipeline.COMPLETED_STATUSES:
            raise RuntimeError(after.get("error") or "the document could not be rebuilt")
        for field in _KT_FIELDS:
            if field in before:
                after[field] = before[field]
        after["sessions"] = renamed
        version = document_model.current_version(before) + 1
        summary = "Speakers named: " + ", ".join(f"{old} as {new}" for old, new in names.items())
        pipeline.JOB_STORE.save_version(job_id, version, actor, summary,
                                        document_model.snapshot(after.get("knowledge_object") or {}))
        after["document_version"] = version
        signoff.on_new_version(after, version, actor)
        pipeline.JOB_QUEUE[job_id] = after
    except Exception as exc:
        before["speakers_error"] = f"The document could not be rebuilt with those names: {exc}"
        pipeline.JOB_QUEUE[job_id] = before


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
        pipeline.run_kt_pipeline(job_id, transcript, segments, warnings=warnings, notices=notices,
                                 giver=before.get("giver"))
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
