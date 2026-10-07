"""Observability (P1-6): structured logs, stage timings, run records,
metrics, alerts and error tracking.

Logs      CONTINUUM_LOG_FORMAT=json writes one JSON object per line, with the
          job, tenant and request ids of the work that logged it (the image
          sets it; "text" is the readable form for development).
Timings   checkpoint("Classification") after each pipeline phase records how
          long it took, for the KT being processed on this thread.
Runs      One job_runs row per finished KT: queue wait, duration, seconds per
          stage, failed stages, LLM calls, tokens and fallbacks, and an
          estimate of false "missing" sections. Numbers and ids only, never
          transcript content.
Metrics   GET /metrics (Prometheus text) from those rows and the queue. Only
          served when CONTINUUM_METRICS_TOKEN is set, to that bearer token.
Alerts    CONTINUUM_ALERT_WEBHOOK_URL receives {"text": ...} (Slack and Teams
          incoming webhooks accept it) when a KT fails for a reason other
          than its input, and when the oldest queued KT has waited longer
          than CONTINUUM_ALERT_QUEUE_AGE_SECONDS (default 900).
Errors    SENTRY_DSN sends exceptions to Sentry, with no request bodies, no
          local variables and no personal data.

deploy/prometheus-alerts.yml and deploy/grafana-dashboard.json use the
metrics; scripts/ops_report.py prints the same figures from the database.
"""
import contextlib
import contextvars
import json
import logging
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

JOB_ID: contextvars.ContextVar = contextvars.ContextVar("continuum_job_id", default=None)
TENANT_ID: contextvars.ContextVar = contextvars.ContextVar("continuum_tenant_id", default=None)
REQUEST_ID: contextvars.ContextVar = contextvars.ContextVar("continuum_request_id", default=None)
_TIMER: contextvars.ContextVar = contextvars.ContextVar("continuum_stage_timer", default=None)

# Failures caused by what was submitted (no speech, not a KT, not a
# recording): the user is told why; nobody needs to be paged.
FAILURE_INPUT = "input"
FAILURE_ERROR = "error"


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = JOB_ID.get()
        record.tenant_id = TENANT_ID.get()
        record.request_id = REQUEST_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("job_id", "tenant_id", "request_id"):
            value = getattr(record, key, None)
            if value:
                out[key] = value
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        job = getattr(record, "job_id", None)
        return f"[job {job[:8]}] {line}" if job else line


_configured = False


def configure_logging(default_format: Optional[str] = None) -> None:
    """Log through one handler in the chosen format. Does nothing unless
    CONTINUUM_LOG_FORMAT (or `default_format`) is set, so development and
    test logging stay as they are."""
    global _configured
    fmt = (os.getenv("CONTINUUM_LOG_FORMAT") or default_format or "").strip().lower()
    if _configured or fmt not in ("json", "text"):
        return
    handler = logging.StreamHandler()
    handler.addFilter(_ContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json"
                         else _TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(os.getenv("CONTINUUM_LOG_LEVEL", "INFO").upper())
    # uvicorn's own loggers write through the same handler and format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
    # One line per HTTP call to the LLM or model hub is noise at INFO.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    _configured = True


# --------------------------------------------------------------------------
# Stage timings
# --------------------------------------------------------------------------

class StageTimer:
    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}
        self._last = time.perf_counter()

    def checkpoint(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages[stage] = round(self.stages.get(stage, 0.0) + now - self._last, 3)
        self._last = now


def checkpoint(stage: str) -> None:
    """Record the time since the previous checkpoint (or since the KT started)
    as `stage`. A no-op outside a job_context."""
    timer = _TIMER.get()
    if timer is not None:
        timer.checkpoint(stage)


@contextlib.contextmanager
def job_context(job_id: str, tenant_id: Optional[str] = None) -> Iterator[StageTimer]:
    """The KT being processed on this thread: its ids reach every log line
    and Sentry event, and checkpoints are recorded on the yielded timer."""
    timer = StageTimer()
    tokens = (JOB_ID.set(job_id), TENANT_ID.set(tenant_id), _TIMER.set(timer))
    try:
        yield timer
    finally:
        _TIMER.reset(tokens[2])
        TENANT_ID.reset(tokens[1])
        JOB_ID.reset(tokens[0])


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------

def run_record(job_id: str, job: Dict[str, Any], task: Dict[str, Any], timer: StageTimer,
               started_at: float, finished_at: float) -> Dict[str, Any]:
    """The numbers kept for one finished KT. No transcript content."""
    usage = job.get("llm_usage") or {}
    ko = job.get("knowledge_object") or {}
    missing = _missing_sections(ko)
    mentioned = set(ko.get("_mentioned_elsewhere") or {})
    return {
        "job_id": job_id,
        "tenant_id": job.get("tenant_id"),
        "kind": task.get("kind"),
        "status": job.get("status") or "failed",
        "failure": job.get("failure") or (FAILURE_ERROR if job.get("status") == "failed" else None),
        "attempt": task.get("attempt"),
        "enqueued_at": task.get("enqueued_at"),
        "started_at": started_at,
        "finished_at": finished_at,
        "stage_seconds": dict(timer.stages),
        "failed_stages": sorted({e.get("stage") for e in job.get("stage_errors") or [] if e.get("stage")}),
        "llm_calls": int(usage.get("llm_calls") or 0),
        "llm_failed": int(usage.get("failed") or 0),
        "llm_rejected": int(usage.get("rejected") or 0),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "sections_missing": len(missing),
        # Sections reported missing although their topics came up under
        # another section: an estimate of false "missing" (P0-8).
        "missing_but_mentioned": len(missing & mentioned),
    }


def _missing_sections(ko: Dict[str, Any]) -> set:
    try:
        from kt_schema_loader import SCHEMA

        title_to_id = {s["title"]: s["id"] for s in SCHEMA}
    except Exception:
        title_to_id = {}
    for sec in ko.get("sections") or []:
        if sec.get("id") == "kt_coverage":
            return {title_to_id.get(r.get("Domain"), r.get("Domain")) for r in sec.get("_coverage_rows") or []
                    if str(r.get("Coverage", "")).lower().startswith("missing")}
    return set()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

_DURATION_BUCKETS = (30, 60, 120, 300, 600, 1200, 1800, 3600)


def _label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def prometheus_text(runs: List[Dict[str, Any]], queue: Dict[str, Any]) -> str:
    """Prometheus exposition of all recorded runs and the current queue."""
    lines: List[str] = []

    def metric(name, kind, help_text, samples):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            label_text = ",".join(f'{k}="{_label(v)}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_text}}} {value}" if label_text else f"{name} {value}")

    by_status: Dict[tuple, int] = {}
    stage_sum: Dict[str, float] = {}
    stage_count: Dict[str, int] = {}
    stage_failures: Dict[str, int] = {}
    buckets = [0] * len(_DURATION_BUCKETS)
    duration_sum = wait_sum = 0.0
    totals = dict.fromkeys(("llm_calls", "llm_failed", "llm_rejected", "input_tokens", "output_tokens",
                            "sections_missing", "missing_but_mentioned"), 0)
    for run in runs:
        key = (run["status"], run.get("failure") or "")
        by_status[key] = by_status.get(key, 0) + 1
        duration = max(0.0, run["finished_at"] - run["started_at"])
        duration_sum += duration
        for i, bound in enumerate(_DURATION_BUCKETS):
            if duration <= bound:
                buckets[i] += 1
        if run.get("enqueued_at"):
            wait_sum += max(0.0, run["started_at"] - run["enqueued_at"])
        for stage, seconds in (run.get("stage_seconds") or {}).items():
            stage_sum[stage] = stage_sum.get(stage, 0.0) + seconds
            stage_count[stage] = stage_count.get(stage, 0) + 1
        for stage in run.get("failed_stages") or []:
            stage_failures[stage] = stage_failures.get(stage, 0) + 1
        for k in totals:
            totals[k] += int(run.get(k) or 0)

    metric("continuum_jobs_total", "counter", "KTs finished, by final status and failure reason.",
           [({"status": s, "reason": r} if r else {"status": s}, n) for (s, r), n in sorted(by_status.items())])
    lines.append("# HELP continuum_job_duration_seconds Time from a worker starting a KT to its final state.")
    lines.append("# TYPE continuum_job_duration_seconds histogram")
    for bound, count in zip(_DURATION_BUCKETS, buckets):
        lines.append(f'continuum_job_duration_seconds_bucket{{le="{bound}"}} {count}')
    lines.append(f'continuum_job_duration_seconds_bucket{{le="+Inf"}} {len(runs)}')
    lines.append(f"continuum_job_duration_seconds_sum {round(duration_sum, 3)}")
    lines.append(f"continuum_job_duration_seconds_count {len(runs)}")
    metric("continuum_queue_wait_seconds_total", "counter", "Total time finished KTs waited for a worker.",
           [({}, round(wait_sum, 3))])
    metric("continuum_stage_seconds_total", "counter", "Time spent per pipeline stage.",
           [({"stage": s}, round(v, 3)) for s, v in sorted(stage_sum.items())])
    metric("continuum_stage_runs_total", "counter", "Times each pipeline stage ran.",
           [({"stage": s}, n) for s, n in sorted(stage_count.items())])
    metric("continuum_stage_failures_total", "counter", "Pipeline stage failures (the KT may still complete).",
           [({"stage": s}, n) for s, n in sorted(stage_failures.items())])
    metric("continuum_llm_calls_total", "counter", "LLM calls made.", [({}, totals["llm_calls"])])
    metric("continuum_llm_fallbacks_total", "counter",
           "LLM results not used: the call failed, or its value was not supported by the transcript.",
           [({"reason": "failed"}, totals["llm_failed"]), ({"reason": "rejected"}, totals["llm_rejected"])])
    metric("continuum_llm_tokens_total", "counter", "LLM tokens.",
           [({"direction": "input"}, totals["input_tokens"]), ({"direction": "output"}, totals["output_tokens"])])
    metric("continuum_sections_missing_total", "counter", "Template sections reported missing.",
           [({}, totals["sections_missing"])])
    metric("continuum_sections_missing_but_mentioned_total", "counter",
           "Sections reported missing whose topics were mentioned under another section (false-missing estimate).",
           [({}, totals["missing_but_mentioned"])])
    metric("continuum_queue_depth", "gauge", "KTs waiting for or running on a worker.",
           [({"state": "queued"}, queue.get("queued", 0)), ({"state": "running"}, queue.get("running", 0))])
    metric("continuum_queue_oldest_seconds", "gauge", "How long the oldest queued KT has waited.",
           [({}, queue.get("oldest_queued_seconds", 0.0))])
    metric("continuum_workers", "gauge", "Workers that checked in within the last minute.",
           [({}, queue.get("workers", 0))])
    metric("continuum_worker_slots", "gauge", "KTs the live workers can run at once.",
           [({}, queue.get("worker_slots", 0))])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------

def send_alert(text: str, **fields: Any) -> bool:
    """POST {"text": ..., ...} to CONTINUUM_ALERT_WEBHOOK_URL. Never raises."""
    url = os.getenv("CONTINUUM_ALERT_WEBHOOK_URL")
    if not url:
        return False
    body = json.dumps(dict(fields, text=text), default=str).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 (operator-configured URL)
            return 200 <= resp.status < 300
    except Exception as exc:
        logger.warning("Alert webhook failed: %s", exc)
        return False


def alert_on_run(run: Dict[str, Any], error: Optional[str]) -> bool:
    """Page on a KT that failed for a reason other than its input."""
    if run["status"] != "failed" or run.get("failure") == FAILURE_INPUT:
        return False
    return send_alert(
        f"Continuum: KT {run['job_id'][:8]} failed ({(error or 'unknown error')[:200]})",
        alert="kt_failed", job_id=run["job_id"], tenant_id=run.get("tenant_id"),
        failed_stages=run.get("failed_stages"))


_queue_alert_lock = threading.Lock()
_queue_alert_sent = 0.0


def check_queue_age(queue: Dict[str, Any], now: Optional[float] = None) -> bool:
    """Alert when the oldest queued KT has waited too long (at most once an
    hour per process)."""
    global _queue_alert_sent
    limit = float(os.getenv("CONTINUUM_ALERT_QUEUE_AGE_SECONDS", "900"))
    now = now or time.time()
    age = float(queue.get("oldest_queued_seconds") or 0)
    with _queue_alert_lock:
        if age <= limit or now - _queue_alert_sent < 3600:
            return False
        _queue_alert_sent = now
    return send_alert(
        f"Continuum: the oldest queued KT has waited {int(age // 60)} min ({queue.get('queued')} queued, "
        f"{queue.get('workers')} worker(s) alive)", alert="queue_age", **queue)


# --------------------------------------------------------------------------
# Error tracking
# --------------------------------------------------------------------------

def init_error_tracking() -> bool:
    """Send exceptions (and ERROR log lines) to Sentry when SENTRY_DSN is set."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed")
        return False

    def _before_send(event, hint):
        # Transcript text must not leave the deployment: no request bodies,
        # and no local variables (the pipeline's hold the transcript).
        event.pop("request", None)
        for key, var in (("job_id", JOB_ID), ("tenant_id", TENANT_ID), ("request_id", REQUEST_ID)):
            if var.get():
                event.setdefault("tags", {})[key] = var.get()
        return event

    sentry_sdk.init(dsn=dsn, environment=os.getenv("CONTINUUM_ENV", "production"), send_default_pii=False,
                    include_local_variables=False, max_request_body_size="never", traces_sample_rate=0.0,
                    before_send=_before_send)
    return True
