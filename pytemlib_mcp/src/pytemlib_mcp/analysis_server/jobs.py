"""
Background analysis jobs behind the shared job envelope.

Stitching a large map takes minutes, longer than an MCP request should
block. Jobs therefore follow the same contract as asyncroscopy's instrument
jobs (``asyncroscopy/contracts/envelope.schema.json``):
reserve -> start -> status -> results -> release, with abort at any time.

Started jobs run on a bounded pool of worker threads (``max_workers``); a job
started while every worker is busy is ``queued`` until one is free.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from typing import Callable

from .names import ACTIVE_JOB_STATES, ENVELOPE_VERSION, FINISHED_JOB_STATES

RESERVATION_TTL_S = 900.0  # a reservation nobody started is released after this, so it cannot hold a run or map forever
KEEP_FINISHED_S = 6 * 3600.0  # finished jobs stay readable this long and are then forgotten
DEFAULT_MAX_WORKERS = 2  # each stitching job holds a tile cache, and more at once only share the same CPUs and disk


class Aborted(Exception):
    pass


class JobRunner:
    """Runs one task function per job on a bounded thread pool; tasks call ``progress(done, total)`` and may be aborted there.

    Job dicts are shared between the MCP request handlers and the worker
    threads, so every read and change of a job's state happens under
    ``self._lock``. The lock is never held while a task runs or while
    waiting for one.
    """

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.max_workers = max(1, int(max_workers))
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="analysis-job")
        self.tasks: dict[str, tuple[Callable[[dict], dict], Callable[[dict, Callable], dict]]] = {}
        self.resources: dict[str, Callable[[dict], str]] = {}

    def register(self, name: str, plan_check: Callable[[dict], dict], run: Callable[[dict, Callable], dict], resource: Callable[[dict], str] | None = None) -> None:
        """``plan_check(plan) -> estimate`` raises ValueError to refuse; ``run(plan, progress) -> result_keys``.

        ``resource(plan)`` names what the job writes (e.g. a run's map): a second job on the same
        resource is refused as busy while one is reserved, queued or running, with that job's id, so a
        client that lost track of its job (a timeout, a restart) adopts it instead of racing it.
        """
        self.tasks[name] = (plan_check, run)
        if resource is not None:
            self.resources[name] = resource

    def shutdown(self) -> None:
        """Stop taking jobs: queued ones are aborted, running ones are asked to stop at their next check."""
        with self._lock:
            for job in self._jobs.values():
                if job["state"] in ("queued", "running"):
                    job["abort"].set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def reply(payload: dict) -> dict:
        return {"schema_version": ENVELOPE_VERSION, **payload}

    @staticmethod
    def error(code: str, message: str, **fields) -> dict:
        return {"schema_version": ENVELOPE_VERSION, "error": {"code": code, "message": message}, **fields}

    def _unknown(self, job_id: str) -> dict:
        # Job ids from an instrument's job device are not kept here.
        # The instrument endpoint's wait_for_job waits on both kinds.
        return self.error("unknown_job", f"no analysis job {job_id!r}",
                          next_actions=[{"server": "instrument", "tool": "wait_for_job", "args": {"job_id": job_id}}])

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
            estimate = self.tasks[task][0](plan)  # runs outside the lock because it may touch the store
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return self.error("plan_rejected", str(exc))
        reservation_id = uuid.uuid4().hex[:12]
        plan_id = uuid.uuid4().hex[:12]
        resource = self.resources[task](plan) if task in self.resources else None
        with self._lock:
            self._tidy_locked(time.time())
            holder = self._holder(resource)
            if holder is not None:
                held_by = holder.get("job_id") or holder["reservation_id"]
                return self.error(
                    "busy", f"{resource} is already being worked on by job {held_by}",
                    job_id=holder.get("job_id"), reservation_id=holder["reservation_id"],
                    next_actions=[{"tool": "analysis_job_status", "args": {"job_id": held_by}}],
                )
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
            job.update(job_id=job_id, state="queued", done=0, last_error=None)
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
            with self._lock:
                if abort.is_set():  # aborted while it waited for a worker
                    job.update(state="aborted", finished=time.time())
                    return
                job["state"] = "running"
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

        try:
            future = self._pool.submit(work)
        except RuntimeError as exc:  # the pool was shut down
            with self._lock:
                job.update(state="failed", finished=time.time(), last_error={"code": "shutting_down", "message": str(exc)})
            return self.error("shutting_down", "this analysis server is shutting down")
        with self._lock:
            job["future"] = future
            state = job["state"]
        return self.reply({"job_id": job_id, "reservation_id": reservation_id, "state": state})

    def _tidy_locked(self, now: float) -> None:
        """Release reservations left unstarted past RESERVATION_TTL_S and forget jobs that ended KEEP_FINISHED_S ago."""
        for key, job in list(self._jobs.items()):
            if job["state"] == "reserved" and now - job.get("created", now) > RESERVATION_TTL_S:
                job.update(state="released", finished=now, last_error={"code": "expired", "message": f"not started within {RESERVATION_TTL_S:g} s"})
            elif job["state"] in FINISHED_JOB_STATES | {"released"} and now - job.get("finished", now) > KEEP_FINISHED_S:
                del self._jobs[key]

    def _holder(self, resource: str | None) -> dict | None:
        """The reserved, queued or running job on ``resource``; call with ``self._lock`` held."""
        if resource is None:
            return None
        return next((j for j in self._jobs.values() if j.get("resource") == resource and j["state"] in ACTIVE_JOB_STATES), None)

    def _status_locked(self, job: dict) -> dict:
        return self.reply(
            {
                "job_id": job.get("job_id", job["reservation_id"]),
                "reservation_id": job["reservation_id"],
                "state": job["state"],
                "progress": {"done": job["done"], "total": job["total"]},
                "last_error": dict(job["last_error"]) if job["last_error"] else None,
            }
        )

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return self._unknown(job_id)
            return self._status_locked(job)

    def abort(self, job_id: str) -> dict:
        future: Future | None = None
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return self._unknown(job_id)
            if job["state"] == "queued":
                job["abort"].set()
                if job.get("future") is not None and job["future"].cancel():
                    job.update(state="aborted", finished=time.time())
            elif job["state"] == "running":
                job["state"] = "aborting"
                job["abort"].set()
                future = job.get("future")
            elif job["state"] == "reserved":
                job.update(state="aborted", finished=time.time())
        if future is not None:
            # A short wait, so a job between checks reports "aborted" at once.
            # A long step stays "aborting" until its next check instead of blocking this call.
            wait_futures([future], timeout=2)
        return self.status(job_id)

    def results(self, job_id: str) -> dict:
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return self._unknown(job_id)
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
            if job["state"] in {"queued", "running", "aborting"}:
                return self.error("bad_state", f"job is {job['state']}; abort it first")
            job["state"] = "released"
            job["finished"] = time.time()
        return self.reply({"reservation_id": reservation_id, "state": "released"})

    def wait(self, job_id: str, timeout: float = 600.0) -> dict:
        """Block until a job ends, at most ``timeout`` seconds (for scripts, tests and analysis_job_status's wait_s)."""
        with self._lock:
            job = self._find(job_id)
            future = job.get("future") if job is not None else None
        if future is not None:
            wait_futures([future], timeout=timeout)
        return self.status(job_id)
