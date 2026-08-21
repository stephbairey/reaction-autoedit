"""Threaded job manager for the GUI: long pipeline stages run in worker threads, the frontend
polls ``/api/jobs``. One pipeline job per project at a time; renders queue FIFO."""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str                       # detect | analyze | narrative | select | render | auto | card
    project: str
    state: str = "queued"           # queued | running | done | error
    frac: float = 0.0
    msg: str = ""
    log: list[str] = field(default_factory=list)
    error: str | None = None
    result: Any = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "project": self.project, "state": self.state,
                "frac": round(self.frac, 4), "msg": self.msg, "log": self.log[-12:],
                "error": self.error, "created": self.created,
                "result": self.result if isinstance(self.result, (str, int, float, dict, list, type(None))) else str(self.result)}


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_workers)

    def submit(self, kind: str, project: str, fn: Callable[[Job], Any]) -> Job:
        with self._lock:
            active = [j for j in self._jobs.values() if j.project == project and j.state in ("queued", "running")]
            if active:
                raise RuntimeError(f"project '{project}' already has a {active[0].kind} job {active[0].state}")
            job = Job(id=uuid.uuid4().hex[:10], kind=kind, project=project)
            self._jobs[job.id] = job

        def run():
            with self._sem:
                job.state = "running"
                try:
                    job.result = fn(job)
                    job.frac, job.state, job.msg = 1.0, "done", "done"
                except Exception as e:  # noqa: BLE001
                    job.state = "error"
                    job.error = f"{e.__class__.__name__}: {e}"
                    job.log.append(traceback.format_exc(limit=6))

        threading.Thread(target=run, daemon=True, name=f"job-{kind}-{project}").start()
        return job

    def progress_cb(self, job: Job) -> Callable[[float, str], None]:
        def cb(frac: float, msg: str) -> None:
            job.frac, job.msg = float(frac), str(msg)
        return cb

    def log_cb(self, job: Job) -> Callable[[str], None]:
        def cb(msg: str) -> None:
            job.log.append(str(msg))
            job.msg = str(msg)
        return cb

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, project: str | None = None) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.as_dict() for j in jobs if project is None or j.project == project][:50]
