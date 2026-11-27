"""The analysis server's job runner: a reservation nobody starts cannot hold a resource forever, and finished jobs are forgotten."""

from pyTEMlib.analysis_server import jobs
from pyTEMlib.analysis_server.jobs import JobRunner


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


def test_jobs_beyond_the_worker_pool_wait_queued_and_a_queued_job_can_be_aborted():
    import threading

    release = threading.Event()
    r = JobRunner(max_workers=1)
    r.register("hold", lambda plan: {"steps": 1}, lambda plan, progress: (release.wait(10), {"result_keys": {}})[1])
    first = r.start(r.reserve({"task": "hold"})["reservation_id"])
    second = r.start(r.reserve({"task": "hold"})["reservation_id"])
    third = r.start(r.reserve({"task": "hold"})["reservation_id"])
    try:
        assert second["state"] == "queued" and r.status(second["job_id"])["state"] == "queued"
        assert r.abort(third["job_id"])["state"] == "aborted"  # never ran
    finally:
        release.set()
    assert r.wait(first["job_id"], timeout=10)["state"] == "done"
    assert r.wait(second["job_id"], timeout=10)["state"] == "done"
    assert r.status(third["job_id"])["state"] == "aborted"
    assert r.max_workers == 1 and r._pool._max_workers == 1


def test_an_unknown_job_and_a_busy_resource_name_the_next_call_instead_of_prose():
    r = runner()
    unknown = r.status("a1b2c3")
    assert unknown["error"] == {"code": "unknown_job", "message": "no analysis job 'a1b2c3'"}
    assert unknown["next_actions"] == [{"server": "instrument", "tool": "wait_for_job", "args": {"job_id": "a1b2c3"}}]
    held = r.reserve({"task": "stitch", "run": "run_d.h5"})
    busy = r.reserve({"task": "stitch", "run": "run_d.h5"})
    assert busy["error"]["code"] == "busy" and busy["reservation_id"] == held["reservation_id"]
    assert busy["next_actions"] == [{"tool": "analysis_job_status", "args": {"job_id": held["reservation_id"]}}]
