"""The map store's locks are per map, a batch of points of any size is stored, and replacing a scorer's tagged
regions leaves the run's EDS points alone."""

import json
import threading
import time

import h5py

from pyTEMlib.analysis_server.map_store import HIGHLIGHTS

from .test_analysis_map_store_reads import stitched


def test_work_on_one_maps_points_does_not_wait_for_another_map(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    tools_a, _path_a, map_a = stitched(a_dir)
    tools_b, _path_b, map_b = stitched(b_dir)
    held, done = threading.Event(), threading.Event()

    def hold_a():
        with tools_a.store._lock(tools_a.store._points_target(map_a)):
            held.set()
            done.wait(10)

    holder = threading.Thread(target=hold_a)
    holder.start()
    try:
        assert held.wait(10)
        began = time.perf_counter()
        tools_b.store.add_pois(map_b, [{"id": "b1", "x_m": 0.0, "y_m": 0.0, "kind": "note", "links": []}])
        assert time.perf_counter() - began < 5  # map A's lock did not hold up map B
        assert [p["id"] for p in tools_b.store.list_pois(map_b, kind="note")] == ["b1"]
    finally:
        done.set()
        holder.join(10)


def test_a_batch_far_over_hdf5s_64_kb_attribute_limit_is_stored_and_read_back(tmp_path):
    tools, _path, key = stitched(tmp_path)
    store = tools.store
    many = [{"id": f"r{k}", "x_m": 0.0, "y_m": 0.0, "kind": "candidate", "reason": "x" * 200, "links": []} for k in range(2000)]
    assert len(store.add_pois(key, many)) == 2000  # about 0.5 MB of JSON in one batch
    assert len(store.list_pois(key, kind="candidate")) == 2000
    assert store.poi_counts(key)["highlights"] == 2000


def test_retagging_regions_rewrites_only_the_highlights(tmp_path, monkeypatch):
    tools, path, key = stitched(tmp_path)
    store = tools.store
    eds_before = store.list_pois(key, kind="eds_point")
    written: list[str] = []
    real = store._append
    monkeypatch.setattr(store, "_append", lambda map_key, meta, name, points: (written.append(name), real(map_key, meta, name, points))[1])
    for _ in range(2):  # a retried scoring with the same tag replaces its regions
        job = tools.jobs.start(tools.jobs.reserve({"task": "score_map", "map": key, "top_k": 3, "tag": "t1"})["reservation_id"])
        assert tools.jobs.wait(job["job_id"])["state"] == "done"
    assert written and all(name.startswith(HIGHLIGHTS + "_") for name in written)  # no EDS cell was written again
    tagged = store.list_pois(key, kind="candidate")
    assert len(tagged) == 3 and {p["tag"] for p in tagged} == {"t1"}
    assert store.list_pois(key, kind="eds_point") == eds_before
    points_file = store.points_key(path, local=True)
    with h5py.File(points_file, "r") as h5:
        meta = json.loads(h5.attrs["metadata"])
    # The second scoring replaced the first highlights and kept them listed.
    assert meta["highlights"] == written[-1] and [r["group"] for r in meta["replaced_highlights"]] == [written[0]] and meta["layers"] == []


def test_batches_written_as_group_attributes_before_are_still_read(tmp_path):
    tools, path, key = stitched(tmp_path)
    store = tools.store
    points_file = store.points_key(path, local=True)
    with h5py.File(points_file, "a") as h5:
        layer = json.loads(h5.attrs["metadata"])["poi_layer"]
        group = h5.require_group(layer).require_group(HIGHLIGHTS)
        group.create_group("b_00000000000000000001_old").attrs["pois"] = json.dumps([{"id": "old1", "x_m": 0.0, "y_m": 0.0, "kind": "note"}])
        group.attrs["count"] = 1
    store.add_pois(key, [{"id": "new1", "x_m": 0.0, "y_m": 0.0, "kind": "note"}])
    assert [p["id"] for p in store.list_pois(key, kind="note")] == ["old1", "new1"]
