"""What a client needs to draw a map, through an in-memory MCP client on local HDF5 maps: the map listing with its
pyramid and transform, the kinds of point with labels, the point query with per-cell counts, and image_info."""

import json
from pathlib import Path

import h5py
import jsonschema
import pytest

from pytemlib_mcp.analysis_server.map_store import pixel_to_stage

from .test_analysis_map_store_reads import write_layout2
from .test_analysis_map_tools import call, open_client

schema_path = Path(__file__).parents[2] / "schemas" / "stitched_map.schema.json"


def validator(name):
    full = json.loads(schema_path.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator({"$schema": full["$schema"], "$defs": full["$defs"], "$ref": f"#/$defs/{name}"})


@pytest.fixture
def served(tmp_path):
    c, tools, loop = open_client(local_roots=[tmp_path])
    path, tiles = write_layout2(tmp_path / "run.h5", grid=3)
    started = call(c, "assemble_map", run=path)
    assert tools.jobs.wait(started["job_id"])["state"] == "done"
    key = call(c, "analysis_job_results", job_id=started["job_id"])["result_keys"]["map"]
    yield c, tools, path, key, len(tiles)
    loop.run_until_complete(c.__aexit__(None, None, None))
    loop.close()


def score(c, tools, key, tag):
    job = call(c, "score_map", map=key, top_k=2, level=0, tag=tag)
    assert tools.jobs.wait(job["job_id"])["state"] == "done"
    return call(c, "analysis_job_results", job_id=job["job_id"])["summary"]["added_pois"]


def test_list_maps_gives_each_map_its_pyramid_and_transform_read_from_the_stored_levels(served):
    c, tools, path, key, _n = served
    listed = call(c, "list_maps")
    assert listed["total"] == 1 and call(c, "list_maps", run=path)["maps"] == listed["maps"]
    entry = listed["maps"][0]
    validator("map_listing_entry").validate(entry)
    meta = tools.store.map_metadata(key)
    assert entry["map"] == key and entry["run"] == path and entry["local"] is True
    assert entry["title"].startswith("run, stitched ") and entry["title"].endswith(" UTC")
    assert entry["canvas_shape"] == meta["canvas_shape"] and entry["origin_xy_m"] == meta["origin_xy_m"]
    assert len(entry["pyramid"]) == meta["levels"] > 1
    with h5py.File(key, "r") as h5:
        for level in entry["pyramid"]:
            data = h5[level["key"].rpartition("#")[2]]
            assert level["key"] == f"{key}#level_{level['level']}"
            assert level["shape"] == list(data.shape) and level["dtype"] == data.dtype.str
            assert [sum(axis) for axis in level["chunks"]] == level["shape"]
            assert level["downsample"] == 2 ** level["level"]
            r, c_ = level["shape"][0] - 1, level["shape"][1] - 1
            x, y = pixel_to_stage(meta, r, c_, level["level"])
            assert level["origin_xy_m"][0] + c_ * level["pixel_size_m"] == pytest.approx(x, abs=1e-15)
            assert level["origin_xy_m"][1] + r * level["pixel_size_m"] == pytest.approx(y, abs=1e-15)


def test_map_info_carries_the_same_view_and_labels_every_kind_of_point_on_the_map(served):
    c, tools, _path, key, _n = served
    info = call(c, "map_info", map=key)
    validator("map_view").validate(info)
    assert info["pyramid"] == call(c, "list_maps")["maps"][0]["pyramid"]
    assert info["levels"] == len(info["pyramid"])
    score(c, tools, key, "first")
    call(c, "add_poi", map=key, x_m=info["origin_xy_m"][0], y_m=info["origin_xy_m"][1])
    call(c, "add_poi", map=key, x_m=info["origin_xy_m"][0], y_m=info["origin_xy_m"][1], kind="revisit")
    kinds = call(c, "map_info", map=key)["poi_kinds"]
    for kind in kinds:
        validator("poi_kind").validate(kind)
    assert kinds == [
        {"kind": "candidate", "label": "Scored region", "in_cells": False},
        {"kind": "eds_point", "label": "EDS point spectrum", "in_cells": True},
        {"kind": "note", "label": "Note", "in_cells": False},
        {"kind": "revisit", "label": None, "in_cells": False},
    ]


def test_a_crowded_query_gives_cell_counts_in_place_of_cell_points_and_lists_the_current_highlights(served):
    c, tools, _path, key, n = served
    score(c, tools, key, "scan")
    replaced = score(c, tools, key, "scan")
    store_meta = tools.store._store_meta(tools.store.points_key(tools.store.map_metadata(key)["run"], True))
    assert store_meta["highlights"].startswith("highlights_")
    current = sorted(p["id"] for p in replaced)

    plain = call(c, "list_pois", map=key, counts_above=n)
    assert "cells" not in plain and plain["total"] == n + 2
    assert sorted(p["id"] for p in plain["pois"] if p["kind"] == "candidate") == current

    crowded = call(c, "list_pois", map=key, counts_above=n - 1)
    assert crowded["total"] == n + 2 and crowded["cell_kinds"] == ["eds_point"]
    assert sorted(p["id"] for p in crowded["pois"]) == current
    for cell in crowded["cells"]:
        validator("poi_cell_count").validate(cell)
    assert sum(cell["count"] for cell in crowded["cells"]) == n
    eds = call(c, "list_pois", map=key, kind="eds_point")["pois"]
    for cell in crowded["cells"]:
        x0, x1, y0, y1 = cell["box_m"]
        inside = [p for p in eds if x0 <= p["x_m"] < x1 and y0 <= p["y_m"] < y1]
        assert len(inside) == cell["count"]

    first = crowded["cells"][0]
    boxed = call(c, "list_pois", map=key, box_m=first["box_m"], counts_above=0)
    assert [cell["cell"] for cell in boxed["cells"]] == [first["cell"]]
    only_cells = call(c, "list_pois", map=key, kind="eds_point", counts_above=0)
    assert only_cells["pois"] == [] and sum(cell["count"] for cell in only_cells["cells"]) == n
    highlights_only = call(c, "list_pois", map=key, kind="candidate", counts_above=0)
    assert "cells" not in highlights_only and sorted(p["id"] for p in highlights_only["pois"]) == current


def test_image_info_on_a_map_names_its_level_0_and_its_frame(served):
    c, tools, _path, key, _n = served
    info = call(c, "image_info", dataset=key)
    validator("image_info").validate(info)
    meta = tools.store.map_metadata(key)
    assert info["kind"] == "map" and info["image"] == f"{key}#level_0" and info["shape"] == meta["canvas_shape"]
    assert info["frame"]["origin_xy_m"] == meta["origin_xy_m"] and info["frame"]["pixel_size_m"] == meta["pixel_size_m"]
    rows, cols = meta["canvas_shape"]
    assert info["frame"]["centre_xy_m"] == pytest.approx(list(pixel_to_stage(meta, (rows - 1) / 2, (cols - 1) / 2)))
    missing = call(c, "image_info", dataset="no_such_image")
    assert missing["error"]["code"] in ("unknown_dataset", "no_image")
