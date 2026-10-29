"""Run layouts, lazy tile reads with a bounded cache, and the batched POI layer."""

import h5py
import numpy as np
import pytest

from pyTEMlib.mcp_server import map_store
from pyTEMlib.mcp_server.map_store import MapStore

from .test_mosaic import PIXEL, TILE, make_tiles


def write_layout2(path, grid=3):
    tiles, _truth = make_tiles(stage_error_px=4.0, grid=grid)
    n = len(tiles)
    with h5py.File(path, "w") as h5:
        h5.attrs["run_layout"] = 2
        h5.attrs["plan"] = "{}"
        data = h5.create_dataset("tiles", shape=(n + 1, TILE, TILE), dtype=np.uint16, chunks=(1, TILE, TILE))
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
    return str(path), tiles


def test_a_layout_2_run_is_read_lazily_and_skips_unwritten_slots(tmp_path):
    path, tiles = write_layout2(tmp_path / "run.h5")
    store = MapStore(local_roots=[tmp_path])
    info = store.scan_run(path)
    assert info.layout == 2 and len(info.tiles) == len(tiles)
    assert info.tiles[4].acquired_s == 4.0 and info.tiles[4].attrs == {"row": 1, "col": 1}
    run = store.open_run(path)
    np.testing.assert_array_equal(run.tiles[4].image, tiles[4].image.astype(np.uint16))
    assert run.tiles[4].shape == (TILE, TILE)
    run.close()


def test_the_tile_cache_stays_within_its_byte_limit(tmp_path, monkeypatch):
    path, tiles = write_layout2(tmp_path / "run.h5", grid=4)
    monkeypatch.setattr(map_store, "CHUNK_TILES", 2)
    monkeypatch.setattr(map_store, "CACHE_BYTES", 3 * TILE * TILE * 2)
    run = MapStore(local_roots=[tmp_path]).open_run(path)
    for tile in run.tiles:
        tile.image
    assert run.source._bytes <= 2 * 2 * TILE * TILE * 2  # never more than the limit plus one chunk
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
    info = map_store.RunInfo("run.h5", [map_store.TileInfo(k, (0.0, 0.0), PIXEL, (4, 4), {}, n) for k, n in enumerate(names)], {}, 1)
    source = map_store.TileSource(store, "run.h5", info)
    assert [int(source.image(k)[0, 0]) for k in range(5)] == list(range(5))
    assert (group.exports, group.reads) == ((1, 0) if can_export else (0, 5))
    source.close()


def test_poi_layer_adds_batches_and_answers_box_queries(tmp_path):
    from pyTEMlib.mcp_server.map_tools import MapTools

    path, _tiles = write_layout2(tmp_path / "run.h5")
    tools = MapTools(MapStore(local_roots=[tmp_path]))
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path})["reservation_id"])
    tools.jobs.wait(job["job_id"])
    key = tools.jobs.results(job["job_id"])["result_keys"]["map"]
    store = tools.store
    store.add_pois(key, [{"id": f"a{k}", "x_m": k * 1e-9, "y_m": 0.0, "kind": "candidate"} for k in range(100)])
    store.add_pois(key, [{"id": "e", "x_m": 5e-9, "y_m": 0.0, "kind": "eds_point", "links": ["spectrum.h5"]}])
    assert store.poi_count(key) == 101
    assert [p["id"] for p in store.list_pois(key, kind="eds_point")] == ["e"]
    assert {p["id"] for p in store.list_pois(key, box_m=[4.5e-9, 6.5e-9, -1e-9, 1e-9])} == {"a5", "a6", "e"}
    # Re-stitching keeps them.
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": path, "refine": False})["reservation_id"])
    tools.jobs.wait(job["job_id"])
    assert store.poi_count(key) == 101


def test_a_replaced_map_node_is_deleted_with_everything_tiled_wrote():
    calls = []

    class Node:
        def delete(self, recursive=False, external_only=True):
            calls.append((recursive, external_only))

    map_store._delete_node(Node())
    assert calls == [(True, False)]  # tiled 0.2: delete() acts on the node itself; keep nothing behind
