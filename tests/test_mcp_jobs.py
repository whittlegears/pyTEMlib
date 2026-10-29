"""The analysis server's job runner: a reservation nobody starts cannot hold a resource forever, and finished jobs are forgotten."""

from pyTEMlib.mcp_server import jobs
from pyTEMlib.mcp_server.jobs import JobRunner


def runner():
    r = JobRunner()
    r.register("stitch", lambda plan: {"steps": 1}, lambda plan, progress: {"result_keys": {"map": "m"}}, resource=lambda plan: plan["run"])
    return r


def test_an_unstarted_reservation_expires_and_frees_its_resource(monkeypatch):
    r = runner()
    first = r.reserve({"task": "stitch", "run": "run_a.h5"})
    assert r.reserve({"task": "stitch", "run": "run_a.h5"})["error"]["code"] == "busy"
    monkeypatch.setattr(jobs, "RESERVATION_TTL_S", -1.0)
    second = r.reserve({"task": "stitch", "run": "run_a.h5"})
    assert "reservation_id" in second
    assert r.status(first["reservation_id"])["state"] == "released"


def test_finished_jobs_are_forgotten_after_a_while(monkeypatch):
    r = runner()
    done = r.reserve({"task": "stitch", "run": "run_b.h5"})
    started = r.start(done["reservation_id"])
    assert r.wait(started["job_id"])["state"] == "done"
    monkeypatch.setattr(jobs, "KEEP_FINISHED_S", -1.0)
    r.reserve({"task": "stitch", "run": "run_c.h5"})
    assert r.status(started["job_id"])["error"]["code"] == "unknown_job"
