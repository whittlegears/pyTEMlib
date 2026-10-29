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


RESERVATION_TTL_S = 900.0  # a reservation nobody started within this is released, so it cannot hold a run or map forever
KEEP_FINISHED_S = 6 * 3600.0  # finished jobs stay readable this long, then are forgotten (a long-lived server keeps no history)


class Aborted(Exception):
    pass


class JobRunner:
    """Runs one task function per job in a thread; tasks call ``progress(done, total)`` and may be aborted there.

    Job dicts are shared between the MCP request handlers and the worker
    threads, so every read and change of a job's state happens under
    ``self._lock``. The lock is never held while a task runs or while
    waiting for a thread.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.tasks: dict[str, tuple[Callable[[dict], dict], Callable[[dict, Callable], dict]]] = {}
        self.resources: dict[str, Callable[[dict], str]] = {}

    def register(self, name: str, plan_check: Callable[[dict], dict], run: Callable[[dict, Callable], dict], resource: Callable[[dict], str] | None = None) -> None:
        """``plan_check(plan) -> estimate`` raises ValueError to refuse; ``run(plan, progress) -> result_keys``.

        ``resource(plan)`` names what the job writes (e.g. a run's map): a second job on the same
        resource is refused as busy while one is reserved or running, with that job's id, so a
        client that lost track of its job (a timeout, a restart) adopts it instead of racing it.
        """
        self.tasks[name] = (plan_check, run)
        if resource is not None:
            self.resources[name] = resource

    @staticmethod
    def reply(payload: dict) -> dict:
        return {"schema_version": ENVELOPE_VERSION, **payload}

    @staticmethod
    def error(code: str, message: str) -> dict:
        return {"schema_version": ENVELOPE_VERSION, "error": {"code": code, "message": message}}

    def _find(self, key: str) -> dict | None:
        """Job by reservation or job id; call with ``self._lock`` held."""
        if key in self._jobs:
            return self._jobs[key]
        return next((j for j in self._jobs.values() if j.get("job_id") == key), None)

    def reserve(self, plan: dict, caller: str | None = None) -> dict:
        task = plan.get("task")
        if task not in self.tasks:
            return self.error("plan_rejected", f"unknown task {task!r}; known: {sorted(self.tasks)}")
        try:
            estimate = self.tasks[task][0](plan)  # outside the lock: may touch the store
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return self.error("plan_rejected", str(exc))
        reservation_id = uuid.uuid4().hex[:12]
        plan_id = uuid.uuid4().hex[:12]
        resource = self.resources[task](plan) if task in self.resources else None
        with self._lock:
            self._tidy_locked(time.time())
            holder = self._holder(resource)
            if holder is not None:
                return {
                    **self.error("busy", f"{resource} is already being worked on by job {holder.get('job_id', holder['reservation_id'])}; wait for it (analysis_job_status) instead of starting another"),
                    "job_id": holder.get("job_id"),
                    "reservation_id": holder["reservation_id"],
                }
            self._jobs[reservation_id] = {
                "resource": resource,
                "reservation_id": reservation_id, "plan_id": plan_id, "plan": plan, "caller": caller,
                "state": "reserved", "done": 0, "total": int(estimate.get("steps", 1)), "result_keys": {}, "summary": {},
                "last_error": None, "abort": threading.Event(), "created": time.time(),
            }
        return self.reply({"reservation_id": reservation_id, "plan_id": plan_id, "resolved_plan": plan, "estimate": estimate})

    def start(self, reservation_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(reservation_id)
            if job is None:
                return self.error("unknown_reservation", f"no reservation {reservation_id!r}")
            if job["state"] not in {"reserved", "failed", "aborted"}:
                return self.error("bad_state", f"reservation is {job['state']}")
            job_id = uuid.uuid4().hex[:12]
            job["job_id"] = job_id
            job["state"] = "running"
            job["done"] = 0
            job["last_error"] = None
            job.pop("traceback", None)
            job["abort"].clear()
            plan = job["plan"]
            abort = job["abort"]

        def progress(done: int, total: int | None = None) -> None:
            with self._lock:
                job["done"] = int(done)
                if total is not None:
                    job["total"] = int(total)
            if abort.is_set():
                raise Aborted()

        def work() -> None:
            outcome: dict = {}
            try:
                result = self.tasks[plan["task"]][1](plan, progress)
                outcome = {"state": "done", "result_keys": result.get("result_keys", {}), "summary": result.get("summary", {})}
            except Aborted:
                outcome = {"state": "aborted"}
            except Exception as exc:
                outcome = {
                    "state": "failed",
                    "last_error": {"code": "step_failed", "message": f"{type(exc).__name__}: {exc}"},
                    "traceback": traceback.format_exc(limit=4),
                }
            with self._lock:
                job["result_keys"].update(outcome.pop("result_keys", {}))
                job["summary"].update(outcome.pop("summary", {}))
                job.update(outcome)
                job["finished"] = time.time()

        thread = threading.Thread(target=work, name=f"pytemlib-job-{job_id}", daemon=True)
        with self._lock:
            job["thread"] = thread
        thread.start()
        return self.reply({"job_id": job_id, "reservation_id": reservation_id, "state": "running"})

    def _tidy_locked(self, now: float) -> None:
        """Release reservations left unstarted past RESERVATION_TTL_S and forget jobs that ended KEEP_FINISHED_S ago."""
        for key, job in list(self._jobs.items()):
            if job["state"] == "reserved" and now - job.get("created", now) > RESERVATION_TTL_S:
                job.update(state="released", finished=now, last_error={"code": "expired", "message": f"not started within {RESERVATION_TTL_S:g} s"})
            elif job["state"] in {"done", "failed", "aborted", "released"} and now - job.get("finished", now) > KEEP_FINISHED_S:
                del self._jobs[key]

    def _holder(self, resource: str | None) -> dict | None:
        """The reserved or running job on ``resource``; call with ``self._lock`` held."""
        if resource is None:
            return None
        return next((j for j in self._jobs.values() if j.get("resource") == resource and j["state"] in {"reserved", "running", "aborting"}), None)

    def _status_locked(self, job: dict) -> dict:
        return self.reply(
            {
                "job_id": job.get("job_id", job["reservation_id"]),
                "reservation_id": job["reservation_id"],
                "state": job["state"],
                "progress": {"done": job["done"], "total": job["total"]},
                "last_error": dict(job["last_error"]) if job["last_error"] else None,
                "pending_request_id": None,
            }
        )

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return self.error("unknown_job", f"no analysis job {job_id!r} (a job id from Mapper_start belongs to the instrument: use wait_for_job or Mapper_job_status there)")
            return self._status_locked(job)

    def abort(self, job_id: str) -> dict:
        thread = None
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return self.error("unknown_job", f"no analysis job {job_id!r} (a job id from Mapper_start belongs to the instrument: use wait_for_job or Mapper_job_status there)")
            if job["state"] == "running":
                job["state"] = "aborting"
                job["abort"].set()
                thread = job.get("thread")
            elif job["state"] == "reserved":
                job["state"] = "aborted"
        if thread is not None:
            # Briefly, so a job between checks reports "aborted" at once; a long step (a large solve)
            # stays "aborting" until its next check instead of blocking this call.
            thread.join(timeout=2)
        return self.status(job_id)

    def results(self, job_id: str) -> dict:
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return self.error("unknown_job", f"no analysis job {job_id!r} (a job id from Mapper_start belongs to the instrument: use wait_for_job or Mapper_job_status there)")
            return self.reply(
                {
                    "job_id": job.get("job_id", job["reservation_id"]),
                    "state": job["state"],
                    "result_keys": dict(job["result_keys"]),
                    "summary": dict(job["summary"]),
                }
            )

    def release(self, reservation_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(reservation_id)
            if job is None:
                return self.error("unknown_reservation", f"no reservation {reservation_id!r}")
            if job["state"] in {"running", "aborting"}:
                return self.error("bad_state", f"job is {job['state']}; abort it first")
            job["state"] = "released"
            job["finished"] = time.time()
        return self.reply({"reservation_id": reservation_id, "state": "released"})

    def wait(self, job_id: str, timeout: float = 600.0) -> dict:
        """Block until a job ends (for scripts and tests)."""
        with self._lock:
            job = self._find(job_id)
            thread = job.get("thread") if job is not None else None
        if thread is not None:
            thread.join(timeout=timeout)
        return self.status(job_id)
