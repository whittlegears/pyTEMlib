"""Run layouts, lazy tile reads with a bounded cache, map replacement and the spatial POI layer."""

import time

import h5py
import numpy as np
import pytest

from pyTEMlib.image_tools import mosaic
from pyTEMlib.mcp_server import map_store
from pyTEMlib.mcp_server.map_store import MapStore

from .test_mosaic import PIXEL, TILE, make_tiles


def write_layout2(path, grid=3, tiles=None):
    tiles = tiles if tiles is not None else make_tiles(stage_error_px=4.0, grid=grid)[0]
    n = len(tiles)
    shape = tiles[0].image.shape
    with h5py.File(path, "w") as h5:
        h5.attrs["run_layout"] = 2
        h5.attrs["plan"] = "{}"
        data = h5.create_dataset("tiles", shape=(n + 1, *shape), dtype=np.uint16, chunks=(1, *shape))
        data.attrs["pixel_size_m"] = PIXEL
        written = np.zeros(n + 1, bool)
        stage = np.zeros((n + 1, 5))
        for k, t in enumerate(tiles):
            data[k] = t.image.astype(np.uint16)
            written[k] = True
            stage[k, :2] = t.stage_xy_m
        h5["written"] = written  # the last slot was never acquired (an aborted run)
        h5["stage_reported_m"] = stage
        h5["acquired_s"] = np.arange(n + 1, dtype=np.float64)
        h5["row_col"] = np.array([[k // grid, k % grid] for k in range(n + 1)], dtype=np.int32)
        keys = h5.create_dataset("spectrum_keys", shape=(n + 1,), dtype=h5py.string_dtype())
        for k in range(n):
            keys[k] = f"spectrum_{k}.h5"
    return str(path), tiles


def stitched(tmp_path, grid=3):
    from pyTEMlib.mcp_server.map_tools import MapTools

    path, _tiles = write_layout2(tmp_path / "run.h5", grid=grid)
    tools = MapTools(MapStore(local_roots=[tmp_path]))
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path, "eds_pois": True})["reservation_id"])
    assert tools.jobs.wait(job["job_id"])["state"] == "done"
    return tools, path, tools.jobs.results(job["job_id"])["result_keys"]["map"]


def test_a_layout_2_run_is_read_as_arrays_and_lazily_and_skips_unwritten_slots(tmp_path):
    path, tiles = write_layout2(tmp_path / "run.h5")
    store = MapStore(local_roots=[tmp_path])
    info = store.scan_run(path)
    assert info.layout == 2 and len(info) == len(tiles)
    assert info.acquired_s[4] == 4.0 and list(info.row_col[4]) == [1, 1]
    run = store.open_run(path)
    np.testing.assert_array_equal(run.tiles[4].image, tiles[4].image.astype(np.uint16))
    assert run.tiles[4].shape == (TILE, TILE) and run.tiles.arrays()["shapes"].shape == (len(tiles), 2)
    run.close()


def test_the_tile_cache_stays_within_its_byte_limit(tmp_path, monkeypatch):
    path, _tiles = write_layout2(tmp_path / "run.h5", grid=4)
    monkeypatch.setattr(map_store, "CHUNK_TILES", 2)
    monkeypatch.setattr(map_store, "CACHE_BYTES", 3 * TILE * TILE * 2)
    run = MapStore(local_roots=[tmp_path]).open_run(path)
    for tile in run.tiles:
        tile.image
    assert run.source._bytes <= 2 * 2 * TILE * TILE * 2  # never more than the limit plus one chunk
    run.close()


def test_registering_a_wide_grid_reads_each_chunk_about_once_with_a_small_cache(tmp_path, monkeypatch):
    # 4 rows x 40 columns, a cache of 3 rows: matching in strips keeps the working set to two rows.
    rng = np.random.default_rng(1)
    scene = rng.poisson(50, (4 * 48 + 64, 40 * 48 + 64)).astype(np.float64)
    tiles = []
    for r in range(4):
        cols = range(40) if r % 2 == 0 else reversed(range(40))
        for c in cols:
            image = scene[r * 48 : r * 48 + 64, c * 48 : c * 48 + 64]
            tiles.append(mosaic.Tile(image, ((c * 48 + 32) * PIXEL, (r * 48 + 32) * PIXEL), PIXEL, len(tiles)))
    path, _ = write_layout2(tmp_path / "wide.h5", grid=40, tiles=tiles)
    monkeypatch.setattr(map_store, "CHUNK_TILES", 8)
    monkeypatch.setattr(map_store, "CACHE_BYTES", 3 * 40 * 64 * 64 * 2)
    monkeypatch.setattr(mosaic, "STRIP_PX", 16 * 48)
    run = MapStore(local_roots=[tmp_path]).open_run(path)
    mosaic.register_tiles(run.tiles, stage_sigma_m=4 * PIXEL)
    chunks = -(-len(tiles) // 8)
    assert run.source.reads <= 2.5 * chunks  # two matching passes, each reading a chunk about once
    run.close()


class ExportingGroup:
    def __init__(self, images, can_export=True):
        self.images, self.can_export, self.reads, self.exports = images, can_export, 0, 0

    def __getitem__(self, name):
        group = self

        class Node:
            def read(self):
                group.reads += 1
                return group.images[name]

        return Node()

    def export(self, path, format):
        if not self.can_export:
            raise RuntimeError("415 Unsupported Media Type")
        self.exports += 1
        with h5py.File(path, "w") as h5:
            for name, image in self.images.items():
                h5.create_dataset(name, data=image)


@pytest.mark.parametrize("can_export", [True, False])
def test_a_layout_1_run_from_tiled_comes_in_one_export_or_tile_by_tile(can_export):
    names = [f"tile_{k:05d}" for k in range(5)]
    group = ExportingGroup({n: np.full((4, 4), k, np.uint16) for k, n in enumerate(names)}, can_export)
    store = MapStore(tiled_uri="http://tiled.invalid")
    store._client = {"run.h5": {"tiles": group}}
    info = map_store._layout1_info("run.h5", {}, names, [(4, 4)] * 5, [{"stage_reported_m": [0.0, 0.0], "pixel_size_m": PIXEL}] * 5)
    source = map_store.TileSource(store, "run.h5", info)
    assert [int(source.image(k)[0, 0]) for k in range(5)] == list(range(5))
    assert (group.exports, group.reads) == ((1, 0) if can_export else (0, 5))
    source.close()


def test_every_eds_spectrum_is_a_point_of_interest_and_restitching_keeps_one_each(tmp_path):
    tools, path, key = stitched(tmp_path)
    store = tools.store
    eds = store.list_pois(key, kind="eds_point")
    assert len(eds) == 9 and {p["links"][0] for p in eds} == {f"spectrum_{k}.h5" for k in range(9)}
    store.add_pois(key, [{"id": "note1", "x_m": eds[0]["x_m"], "y_m": eds[0]["y_m"], "kind": "note"}])
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path, "eds_pois": True, "refine": False})["reservation_id"])
    assert tools.jobs.wait(job["job_id"])["state"] == "done"
    again = store.current_map(path)
    assert again != key
    for version in (key, again):  # the points belong to the run, so every version lists the same ones
        assert len(store.list_pois(version, kind="eds_point")) == 9 and [p["id"] for p in store.list_pois(version, kind="note")] == ["note1"]


def test_a_failed_restitch_leaves_the_old_map_and_its_points(tmp_path, monkeypatch):
    tools, path, key = stitched(tmp_path)
    before = tools.store.poi_count(key)
    monkeypatch.setattr(tools.store, "update_pois", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path, "eds_pois": True})["reservation_id"])
    assert tools.jobs.wait(job["job_id"])["state"] == "failed"
    monkeypatch.undo()
    assert tools.store.poi_count(key) == before and tools.store.map_metadata(key)["kind"] == "stitched_map"
    assert not list(tmp_path.glob("*.tmp"))  # the half-written new map was removed
    assert tools.store.current_map(path) == key


def test_poi_layer_answers_box_queries_at_a_hundred_thousand_points_reading_only_the_cells_it_touches(tmp_path, monkeypatch):
    tools, _path, key = stitched(tmp_path)
    store = tools.store
    meta = store.map_metadata(key)
    grid = store._grid(key, meta)
    side = grid["cell_m"] * grid["n"]
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, side, size=(100_000, 2)) + [grid["x0"], grid["y0"]]
    store.add_pois(key, [{"id": f"p{k}", "x_m": float(x), "y_m": float(y), "kind": "eds_point", "links": []} for k, (x, y) in enumerate(xy)])
    counts = store.poi_counts(key)
    assert counts["total"] == 100_000 + 9 and sum(counts["cells"].values()) == 100_000 + 9
    read = []
    original = store._read_group
    monkeypatch.setattr(store, "_read_group", lambda k, m, name: read.append(name) or original(k, m, name))
    box = [grid["x0"] + 0.1 * side, grid["x0"] + 0.13 * side, grid["y0"] + 0.5 * side, grid["y0"] + 0.53 * side]
    t0 = time.perf_counter()
    found = store.list_pois(key, kind="eds_point", box_m=box)
    elapsed = time.perf_counter() - t0
    inside = (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[1]) & (xy[:, 1] >= box[2]) & (xy[:, 1] <= box[3])
    assert {p["id"] for p in found if p["id"].startswith("p")} == {f"p{k}" for k in np.flatnonzero(inside)}
    assert len(read) <= 9 and elapsed < 5.0  # about 3 x 3 of the 64 x 64 cells, not 100 000 points


def test_adding_points_with_ids_twice_stores_them_once_and_replacing_switches_layers_whole(tmp_path):
    tools, _path, key = stitched(tmp_path)
    store = tools.store
    eds = store.list_pois(key, kind="eds_point")
    note = {"id": "n1", "x_m": eds[0]["x_m"], "y_m": eds[0]["y_m"], "kind": "high_mag", "links": ["img.h5"]}
    assert store.add_pois(key, [note], dedupe=True) == [note]
    assert store.add_pois(key, [note], dedupe=True) == []
    assert len(store.list_pois(key, kind="high_mag")) == 1
    points, meta = store._points_at(key)
    old_layer = meta["poi_layer"]
    store.update_pois(key, lambda current: [p for p in current if p["kind"] != "high_mag"])
    meta = store._points_at(key)[1]
    assert meta["poi_layer"] != old_layer and store.poi_count(key) == 9
    assert [entry["layer"] for entry in meta["layers"]][-1] == old_layer
    with h5py.File(points, "r") as h5:
        assert old_layer in h5  # the replaced layer is kept, not deleted


def test_a_second_assemble_of_the_same_run_is_busy_and_names_the_running_job(tmp_path, monkeypatch):
    from pyTEMlib.mcp_server.map_tools import MapTools

    path, _ = write_layout2(tmp_path / "run.h5")
    tools = MapTools(MapStore(local_roots=[tmp_path]))
    release = __import__("threading").Event()
    real = tools._run_assemble
    monkeypatch.setattr(tools.jobs, "tasks", {**tools.jobs.tasks, "assemble_map": (tools._check_assemble, lambda plan, progress: (release.wait(10), real(plan, progress))[1])})
    first = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path})["reservation_id"])
    second = tools.jobs.reserve({"task": "assemble_map", "run": path})
    assert second["error"]["code"] == "busy" and second["job_id"] == first["job_id"]
    release.set()
    assert tools.jobs.wait(first["job_id"])["state"] == "done"
    assert "error" not in tools.jobs.reserve({"task": "assemble_map", "run": path})


def test_an_abandoned_map_version_is_deleted_with_everything_tiled_wrote():
    calls = []

    class Node:
        def delete(self, recursive=False, external_only=True):
            calls.append((recursive, external_only))

    map_store._delete_node(Node())
    assert calls == [(True, False)]  # tiled 0.2: delete() acts on the node itself; keep nothing behind
