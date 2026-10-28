"""
Background analysis jobs behind the shared six-verb envelope.

Stitching a large map takes minutes, longer than an MCP request should
block. Jobs therefore follow the same contract as asyncroscopy's instrument
jobs (``schemas/envelope.schema.json``, an exact copy of asyncroscopy's):
reserve -> start -> status -> results -> release, with abort at any time.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

ENVELOPE_SCHEMA_PATH = Path(__file__).with_name("schemas") / "envelope.schema.json"
ENVELOPE_VERSION = json.loads(ENVELOPE_SCHEMA_PATH.read_text(encoding="utf-8"))["schema_version"]


class Aborted(Exception):
    pass


class JobRunner:
    """Runs one task function per job in a thread; tasks call ``progress(done, total)`` and may be aborted there."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.tasks: dict[str, tuple[Callable[[dict], dict], Callable[[dict, Callable], dict]]] = {}

    def register(self, name: str, plan_check: Callable[[dict], dict], run: Callable[[dict, Callable], dict]) -> None:
        """``plan_check(plan) -> estimate`` raises ValueError to refuse; ``run(plan, progress) -> result_keys``."""
        self.tasks[name] = (plan_check, run)

    @staticmethod
    def reply(payload: dict) -> dict:
        return {"schema_version": ENVELOPE_VERSION, **payload}

    @staticmethod
    def error(code: str, message: str) -> dict:
        return {"schema_version": ENVELOPE_VERSION, "error": {"code": code, "message": message}}

    def _find(self, key: str) -> dict | None:
        if key in self._jobs:
            return self._jobs[key]
        return next((j for j in self._jobs.values() if j.get("job_id") == key), None)

    def reserve(self, plan: dict, caller: str | None = None) -> dict:
        task = plan.get("task")
        if task not in self.tasks:
            return self.error("plan_rejected", f"unknown task {task!r}; known: {sorted(self.tasks)}")
        try:
            estimate = self.tasks[task][0](plan)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return self.error("plan_rejected", str(exc))
        reservation_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[reservation_id] = {
                "reservation_id": reservation_id, "plan_id": uuid.uuid4().hex[:12], "plan": plan, "caller": caller,
                "state": "reserved", "done": 0, "total": int(estimate.get("steps", 1)), "result_keys": {}, "summary": {},
                "last_error": None, "abort": threading.Event(),
            }
        job = self._jobs[reservation_id]
        return self.reply({"reservation_id": reservation_id, "plan_id": job["plan_id"], "resolved_plan": plan, "estimate": estimate})

    def start(self, reservation_id: str) -> dict:
        job = self._jobs.get(reservation_id)
        if job is None:
            return self.error("unknown_reservation", f"no reservation {reservation_id!r}")
        if job["state"] not in {"reserved", "failed", "aborted"}:
            return self.error("bad_state", f"reservation is {job['state']}")
        job["job_id"] = uuid.uuid4().hex[:12]
        job["state"] = "running"
        job["abort"].clear()

        def progress(done: int, total: int | None = None) -> None:
            job["done"] = int(done)
            if total is not None:
                job["total"] = int(total)
            if job["abort"].is_set():
                raise Aborted()

        def work() -> None:
            try:
                result = self.tasks[job["plan"]["task"]][1](job["plan"], progress)
                job["result_keys"].update(result.get("result_keys", {}))
                job["summary"].update(result.get("summary", {}))
                job["state"] = "done"
            except Aborted:
                job["state"] = "aborted"
            except Exception as exc:
                job["last_error"] = {"code": "step_failed", "message": f"{type(exc).__name__}: {exc}"}
                job["traceback"] = traceback.format_exc(limit=4)
                job["state"] = "failed"
            job["finished"] = time.time()

        thread = threading.Thread(target=work, name=f"pytemlib-job-{job['job_id']}", daemon=True)
        job["thread"] = thread
        thread.start()
        return self.reply({"job_id": job["job_id"], "reservation_id": reservation_id, "state": "running"})

    def status(self, job_id: str) -> dict:
        job = self._find(job_id)
        if job is None:
            return self.error("unknown_job", f"no job {job_id!r}")
        return self.reply(
            {
                "job_id": job.get("job_id", job["reservation_id"]),
                "reservation_id": job["reservation_id"],
                "state": job["state"],
                "progress": {"done": job["done"], "total": job["total"]},
                "last_error": job["last_error"],
                "pending_request_id": None,
            }
        )

    def abort(self, job_id: str) -> dict:
        job = self._find(job_id)
        if job is None:
            return self.error("unknown_job", f"no job {job_id!r}")
        if job["state"] == "running":
            job["state"] = "aborting"
            job["abort"].set()
            job["thread"].join(timeout=30)
        elif job["state"] == "reserved":
            job["state"] = "aborted"
        return self.status(job_id)

    def results(self, job_id: str) -> dict:
        job = self._find(job_id)
        if job is None:
            return self.error("unknown_job", f"no job {job_id!r}")
        return self.reply({"job_id": job.get("job_id", job["reservation_id"]), "state": job["state"], "result_keys": job["result_keys"], "summary": job["summary"]})

    def release(self, reservation_id: str) -> dict:
        job = self._jobs.get(reservation_id)
        if job is None:
            return self.error("unknown_reservation", f"no reservation {reservation_id!r}")
        if job["state"] in {"running", "aborting"}:
            return self.error("bad_state", f"job is {job['state']}; abort it first")
        job["state"] = "released"
        return self.reply({"reservation_id": reservation_id, "state": "released"})

    def wait(self, job_id: str, timeout: float = 600.0) -> dict:
        """Block until a job ends (for scripts and tests)."""
        job = self._find(job_id)
        if job is not None and "thread" in job:
            job["thread"].join(timeout=timeout)
        return self.status(job_id)
