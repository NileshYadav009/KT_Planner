"""KT workers (P1-4): run queued KTs, at most `concurrency` at a time.

    python worker.py                    # one worker process, one KT at a time
    python worker.py --concurrency 2    # two at a time (two KTs' memory)
    python job_queue.py --worker-alive  # health check (light: no models imported)

By default the API process runs one worker thread itself
(CONTINUUM_WORKERS=1), so `uvicorn main:app` alone still processes KTs. In
production, run the API with CONTINUUM_WORKERS=0 and worker processes beside
it on the same data volume (docker-compose.yml): the API then only accepts
uploads and answers requests, and does not load the models, so a heavy KT
cannot slow it down, and a worker that runs out of memory takes no requests
down with it (its KT is retried by the next worker, see job_queue.py).
"""
import argparse
import logging
import os
import signal
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

import observability
import pipeline
import sessions
from job_queue import LEASE_SECONDS

logger = logging.getLogger(__name__)
POLL_SECONDS = 1.0


def configured_threads() -> int:
    """Worker threads the API process runs itself (CONTINUUM_WORKERS)."""
    try:
        return max(0, int(os.getenv("CONTINUUM_WORKERS", "1")))
    except ValueError:
        return 1


def run_task(task: Dict[str, Any], worker_id: str) -> None:
    """Run one claimed task to a final job state, store its PDF, record its
    numbers (P1-6) and close it."""
    job = pipeline.JOB_QUEUE.get(task["job_id"]) or {}
    started = time.time()
    with observability.job_context(task["job_id"], job.get("tenant_id")) as timer:
        error = _run(task, worker_id)
        job = pipeline.JOB_QUEUE.get(task["job_id"])
        if job is None:
            return                                    # deleted while it ran
        run = observability.run_record(task["job_id"], job, task, timer, started, time.time())
        try:
            pipeline.JOB_STORE.record_run(run)
        except Exception:
            logger.warning("Could not record the run of %s", task["job_id"], exc_info=True)
        logger.info("KT %s %s in %.0f s", run["status"], task["kind"], run["finished_at"] - started)
        observability.alert_on_run(run, error or job.get("error"))


def _run(task: Dict[str, Any], worker_id: str) -> Optional[str]:
    job_id, kind, payload = task["job_id"], task["kind"], task["payload"]
    error: Optional[str] = None
    try:
        if kind == "upload":
            input_path = payload["input_path"]
            pipeline.process_upload_task(job_id, input_path, f"{input_path}.mp3", media_format=payload.get("media_format"))
        elif kind == "transcript":
            pipeline.run_kt_pipeline(job_id, payload["transcript"], warnings=payload.get("warnings") or [])
        elif kind == "session":
            sessions.add_session(job_id, payload)         # a follow-up session of a finished KT (P2-1)
        else:
            raise ValueError(f"unknown task kind {kind!r}")
    except Exception as exc:     # the pipeline records its own failures; this is anything else
        logger.exception("KT task %s failed", job_id)
        error = str(exc) or exc.__class__.__name__
        job = pipeline.JOB_QUEUE.get(job_id)
        if job is not None:
            job.update(status="failed", error="Processing failed unexpectedly. Submit the KT again.")
            pipeline.JOB_QUEUE[job_id] = job

    observability.checkpoint("Saving")
    job = pipeline.JOB_QUEUE.get(job_id) or {}
    if job.get("status") in pipeline.COMPLETED_STATUSES:
        try:
            pipeline.render_and_store_document(job_id)
        except Exception:        # the download renders it instead
            logger.warning("Could not pre-render the PDF for %s", job_id, exc_info=True)
        observability.checkpoint("PDF rendering")
    elif job.get("status") == "failed" and not error:
        error = job.get("error") or "failed"

    if not pipeline.TASKS.finish(job_id, worker_id, error):
        # The KT was deleted while it ran: do not leave its result behind.
        pipeline.delete_job_data(job_id)
    return error


class Worker:
    def __init__(self, concurrency: int = 1, worker_id: Optional[str] = None):
        self.concurrency = max(1, int(concurrency))
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._stop = threading.Event()
        self._running: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._threads = []

    # -- one task at a time (also used by tests to drain the queue) ---------

    def run_next(self, job_id: Optional[str] = None) -> bool:
        """Claim and run one task (only `job_id`'s, if given). False when
        there is none."""
        task = pipeline.TASKS.claim(self.worker_id, job_id)
        if task is None:
            return False
        with self._lock:
            self._running[task["job_id"]] = threading.current_thread().name
        # The job shows as processing from the moment a worker has it.
        pipeline.JOB_QUEUE.set_status(task["job_id"], "processing")
        try:
            run_task(task, self.worker_id)
        finally:
            with self._lock:
                self._running.pop(task["job_id"], None)
        return True

    def drain(self, *job_ids: str) -> int:
        """Run tasks until none is left (only these jobs', if given). Returns
        how many ran."""
        ran = 0
        for job_id in job_ids or (None,):
            while self.run_next(job_id):
                ran += 1
        return ran

    # -- background threads ---------------------------------------------------

    def start(self) -> "Worker":
        pipeline.TASKS.heartbeat(self.worker_id, self.concurrency, 0)
        for i in range(self.concurrency):
            thread = threading.Thread(target=self._loop, name=f"kt-worker-{i}", daemon=True)
            thread.start()
            self._threads.append(thread)
        beat = threading.Thread(target=self._heartbeat_loop, name="kt-worker-heartbeat", daemon=True)
        beat.start()
        self._threads.append(beat)
        logger.info("Worker %s started with %d slot(s)", self.worker_id, self.concurrency)
        return self

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop claiming. Running KTs finish first when `timeout` is None."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout)
        pipeline.TASKS.remove_worker(self.worker_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ran = self.run_next()
            except Exception:
                logger.exception("Worker loop error")
                ran = False
            if not ran:
                self._stop.wait(POLL_SECONDS)

    def _heartbeat_loop(self) -> None:
        interval = max(1.0, LEASE_SECONDS / 4)
        while not self._stop.wait(interval):
            with self._lock:
                running = list(self._running)
            try:
                pipeline.TASKS.renew(self.worker_id, running)
                pipeline.TASKS.heartbeat(self.worker_id, self.concurrency, len(running))
                observability.check_queue_age(pipeline.TASKS.stats())
            except Exception:
                logger.warning("Worker heartbeat failed", exc_info=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run queued Continuum KT jobs.")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONTINUUM_WORKER_CONCURRENCY", "1")),
                        help="KTs processed at the same time by this process (each needs its own memory)")
    args = parser.parse_args()
    observability.configure_logging(default_format="text")
    observability.init_error_tracking()

    pipeline.load_models()
    worker = Worker(args.concurrency).start()
    stopped = threading.Event()

    def _shutdown(signum, frame):
        logger.info("Stopping: finishing running KTs, taking no new ones")
        stopped.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    while not stopped.wait(1.0):     # a timed wait, so Ctrl+C is seen on Windows too
        pass
    worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
