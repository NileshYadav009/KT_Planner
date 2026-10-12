"""KTs due for review (P3-3).

A KT describes a system as it was on the day of the session. Systems change,
so a KT nobody has looked at for months is a risk: the next engineer trusts
an alert name, a runbook step or an escalation contact that is no longer
true. A finished KT is due for review when nothing has changed it (an edit,
a follow-up session, a sign-off step) for CONTINUUM_KT_REVIEW_DAYS days
(default 180; 0 turns this off).

A KT that is due shows "Review due" in the app's list, and the workers post
one reminder per quiet period to the team channel
(CONTINUUM_REVIEW_WEBHOOK_URL), naming the incoming owner from the sign-off
(or whoever created the KT). Any change to the KT starts a new period.
"""
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_DAYS = 180
CHECK_INTERVAL_SECONDS = 3600
DAY = 86400.0

_check_lock = threading.Lock()
_last_check = 0.0


def review_days() -> float:
    try:
        return max(0.0, float(os.getenv("CONTINUUM_KT_REVIEW_DAYS", str(DEFAULT_REVIEW_DAYS)) or 0))
    except ValueError:
        return float(DEFAULT_REVIEW_DAYS)


def annotate(job: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """Add review_due / review_due_at to a listed job (finished KTs only)."""
    days = review_days()
    finished = job.get("status") in ("completed", "completed_with_warnings")
    if not days or not finished or not job.get("updated_at"):
        job["review_due"], job["review_due_at"] = False, None
        return job
    due_at = float(job["updated_at"]) + days * DAY
    job["review_due"], job["review_due_at"] = (now or time.time()) >= due_at, due_at
    return job


def owner_of(job: Dict[str, Any]) -> Optional[str]:
    """Who should review it: the incoming owner named in the sign-off, else
    whoever created the KT."""
    receiver = ((job.get("signoff") or {}).get("participants") or {}).get("receiver") or {}
    return receiver.get("email") or job.get("created_by")


def _message(row: Dict[str, Any], job: Dict[str, Any], days: float, now: float) -> str:
    title = row.get("title") or (job.get("knowledge_object") or {}).get("system_name") or "A KT"
    quiet = int((now - float(row["updated_at"])) // DAY)
    owner = owner_of(job)
    who = f"{owner}, please" if owner else "Please"
    return (f"{title} has not been reviewed for {quiet} days (the review interval is {int(days)}). "
            f"{who} check it still matches the system: edit it, add a session or sign it again in Continuum.")


def send_reminders(store, get_job: Callable[[str], Optional[Dict[str, Any]]], now: Optional[float] = None,
                   notify: Optional[Callable[..., bool]] = None, limit: int = 50) -> int:
    """Post one reminder for each KT that became due and has not had one
    since its last change. Returns how many were sent."""
    import observability

    days = review_days()
    if not days or not os.getenv("CONTINUUM_REVIEW_WEBHOOK_URL"):
        return 0
    notify = notify or observability.notify_review
    now = now or time.time()
    sent = 0
    for row in store.due_for_review(now - days * DAY, limit=limit):
        if not store.claim_review_reminder(row["job_id"], row["updated_at"], now):
            continue                        # another worker took it
        job = get_job(row["job_id"]) or {}
        if notify(_message(row, job, days, now), event="kt_review_due", job_id=row["job_id"],
                  tenant_id=row.get("tenant_id"), owner=owner_of(job)):
            sent += 1
        else:
            store.release_review_reminder(row["job_id"])   # try again at the next check
    return sent


def check_due_reviews(now: Optional[float] = None) -> int:
    """At most once an hour per process (called from the worker heartbeat)."""
    global _last_check
    now = now or time.time()
    with _check_lock:
        if now - _last_check < CHECK_INTERVAL_SECONDS:
            return 0
        _last_check = now
    import pipeline

    try:
        return send_reminders(pipeline.JOB_STORE, pipeline.JOB_QUEUE.get, now)
    except Exception:
        logger.warning("Review reminders failed", exc_info=True)
        return 0
