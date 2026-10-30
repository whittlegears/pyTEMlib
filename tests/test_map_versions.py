"""Map versions are never deleted, and a run's points of interest are kept once for all its versions."""

import json
import os
from pathlib import Path

import h5py

from pyTEMlib.mcp_server import create_server
from pyTEMlib.mcp_server.map_store import HIGHLIGHTS, MapStore

from .test_map_store_remote_reads import stitched, write_layout2


def restitch(tools, path, **plan):
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path, **plan})["reservation_id"])
    assert tools.jobs.wait(job["job_id"])["state"] == "done"
    return tools.jobs.results(job["job_id"])


def test_every_stitch_is_a_new_version_and_the_old_ones_stay_readable(tmp_path):
    tools, path, first = stitched(tmp_path)
    second = restitch(tools, path, refine=False)["result_keys"]["map"]
    third = restitch(tools, path)["result_keys"]["map"]
    store = tools.store
    assert len({first, second, third}) == 3
    assert [m["map"] for m in store.list_maps(path)] == [third, second, first]
    assert store.current_map(path) == third
    for key in (first, second, third):
        meta = store.map_metadata(key)
        assert meta["version"] == key and store.read_level(key, 0).size
    assert store.map_metadata(first)["inputs_sha256"] == store.map_metadata(third)["inputs_sha256"]
    assert store.map_metadata(second)["inputs_sha256"] != store.map_metadata(first)["inputs_sha256"]


def test_map_info_names_the_newest_version(tmp_path):
    tools, path, first = stitched(tmp_path)
    second = restitch(tools, path)["result_keys"]["map"]
    old, new = tools.map_info(first), tools.map_info(second)
    assert old["versions"] == {"count": 2, "newest": second, "this_is_newest": False}
    assert new["versions"]["this_is_newest"] is True and new["inputs_sha256"]


def test_points_are_stored_once_per_run(tmp_path):
    tools, path, first = stitched(tmp_path)
    store = tools.store
    points = store.points_key(path, local=True)
    assert Path(points).exists() and not list(tmp_path.glob("*_map*.h5.*"))
    with h5py.File(first, "r") as h5:
        assert "poi" not in h5 and not any(name.startswith("poi_") for name in h5)  # no points inside the map


def legacy_map(tools, path, first):
    """Turn a stitched map into one written before versions and point stores: key <run>_map, points inside it."""
    store = tools.store
    os.remove(store.points_key(path, local=True))
    legacy = str(Path(path).with_name(f"{store.map_name(path)}.h5"))
    os.replace(first, legacy)
    with h5py.File(legacy, "a") as h5:
        meta = json.loads(h5.attrs["metadata"])
        for key in ("version", "inputs_sha256", "poi_layer"):
            meta.pop(key, None)
        h5.attrs["metadata"] = json.dumps(meta)
    old = {"id": "old1", "x_m": 0.0, "y_m": 0.0, "kind": "candidate", "links": []}
    store._append(legacy, store.map_metadata(legacy), HIGHLIGHTS, [old])
    return legacy, old


def test_a_map_from_before_point_stores_keeps_its_points_and_hands_them_on(tmp_path):
    tools, path, first = stitched(tmp_path)
    legacy, old = legacy_map(tools, path, first)
    store = tools.store
    assert store.current_map(path) == legacy
    assert store.list_pois(legacy, kind="candidate") == [old]
    new = restitch(tools, path)["result_keys"]["map"]
    assert new != legacy and Path(legacy).exists()
    for key in (legacy, new):
        assert store.list_pois(key, kind="candidate") == [old]
    assert store._points_at(new)[1]["migrated_from"] == legacy


def test_adding_a_point_to_a_map_from_before_point_stores_copies_its_points_first(tmp_path):
    tools, path, first = stitched(tmp_path)
    legacy, old = legacy_map(tools, path, first)
    note = {"id": "n1", "x_m": 0.0, "y_m": 0.0, "kind": "note", "links": []}
    tools.store.add_pois(legacy, [note])
    assert {p["id"] for p in tools.store.list_pois(legacy)} >= {"old1", "n1"}


def test_a_restitch_aborted_at_the_end_leaves_no_new_version(tmp_path, monkeypatch):
    tools, path, first = stitched(tmp_path)
    monkeypatch.setattr(tools.store, "update_pois", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path, "eds_pois": True})["reservation_id"])
    assert tools.jobs.wait(job["job_id"])["state"] == "failed"
    monkeypatch.undo()
    assert [m["map"] for m in tools.store.list_maps(path)] == [first]


def test_no_analysis_tool_deletes_or_removes():
    import asyncio

    app, _tools = create_server()
    names = [t.name for t in asyncio.run(app.list_tools())]
    assert "list_maps" in names
    assert not [n for n in names if any(word in n for word in ("delete", "remove", "drop", "purge"))]


def test_two_runs_beside_each_other_keep_their_own_maps(tmp_path):
    from pyTEMlib.mcp_server.map_tools import MapTools

    tools = MapTools(MapStore(local_roots=[tmp_path]))
    a, _ = write_layout2(tmp_path / "run.h5")
    b, _ = write_layout2(tmp_path / "run_map_b.h5")
    map_a = restitch(tools, a)["result_keys"]["map"]
    map_b = restitch(tools, b)["result_keys"]["map"]
    assert [m["map"] for m in tools.store.list_maps(a)] == [map_a]
    assert [m["map"] for m in tools.store.list_maps(b)] == [map_b]
