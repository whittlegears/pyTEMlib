"""The map tools against a real local Tiled server: images written to Tiled, gathered into a run, stitched, listed,
a level read through the descriptor the server gives, and points read back, the highlights after a rescore included."""

import asyncio
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from fastmcp import Client
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

from pytemlib_mcp.analysis_server import create_server
from pytemlib_mcp.analysis_server.map_store import pixel_to_stage

from tests.conftest import PIXEL, make_tiles

schema_path = Path(__file__).parents[2] / "schemas" / "stitched_map.schema.json"


def validator(name):
    full = json.loads(schema_path.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator({"$schema": full["$schema"], "$defs": full["$defs"], "$ref": f"#/$defs/{name}"})


@pytest.fixture(scope="module")
def tiled(tmp_path_factory):
    try:
        server = SimpleTiledServer(directory=tmp_path_factory.mktemp("tiled"))
        client = from_uri(server.uri)
    except Exception as exc:
        print(f"skipping the live Tiled map test: a local Tiled server could not start ({exc!r})")
        pytest.skip(f"a local Tiled server could not start: {exc!r}")
    yield server, client
    server.close()


@pytest.fixture(scope="module")
def served(tiled):
    server, client = tiled
    mcp, tools = create_server(tiled_uri=server.uri, include_basic=False)
    loop = asyncio.new_event_loop()
    session = Client(mcp)
    loop.run_until_complete(session.__aenter__())

    def call(tool, **args):
        result = loop.run_until_complete(session.call_tool(tool, args))
        return result.structured_content if result.structured_content is not None else result.data

    yield call, tools, client
    loop.run_until_complete(session.__aexit__(None, None, None))
    loop.close()


def write_images(client, prefix="img"):
    tiles, _truth = make_tiles(stage_error_px=4.0, grid=3)
    keys = []
    for t in tiles:
        key = f"{prefix}_{t.index}"
        client.create_container(key).create_container("image").write_array(t.image.astype(np.uint16), key="HAADF", metadata={
            "pixel_size_m": PIXEL, "stage_reported_m": json.dumps([t.stage_xy_m[0], t.stage_xy_m[1], 0.0, 0.0, 0.0]),
            "axes": "rows=+y, columns=+x", "scan_region": "[0.0, 0.0, 1.0, 1.0]"})
        keys.append(key)
    return keys, tiles


def finish(call, tools, started):
    assert tools.jobs.wait(started["job_id"])["state"] == "done", tools.jobs.status(started["job_id"])
    return call("analysis_job_results", job_id=started["job_id"])


def test_a_run_written_to_tiled_is_stitched_listed_read_and_its_points_come_back(served):
    call, tools, client = served
    keys, tiles = write_images(client)

    image = call("image_info", dataset=keys[0])
    validator("image_info").validate(image)
    assert image["kind"] == "image" and image["image"] == f"{keys[0]}/image/HAADF" and image["shape"] == list(tiles[0].image.shape)
    assert image["frame"]["pixel_size_m"] == pytest.approx(PIXEL)
    assert image["frame"]["centre_xy_m"] == pytest.approx(list(tiles[0].stage_xy_m))
    rows, cols = image["shape"]
    assert image["frame"]["origin_xy_m"][0] + (cols - 1) / 2 * PIXEL == pytest.approx(tiles[0].stage_xy_m[0])

    run = call("collect_map_run", image_keys=keys)["run"]
    key = finish(call, tools, call("assemble_map", run=run))["result_keys"]["map"]

    listed = call("list_maps")
    assert [m["map"] for m in listed["maps"]] == [key] and call("list_maps", run=run)["maps"] == listed["maps"]
    entry = listed["maps"][0]
    validator("map_listing_entry").validate(entry)
    assert entry["run"] == run and entry["local"] is False and entry["title"].startswith(f"{run}, stitched ")

    meta = tools.store.map_metadata(key)
    for level in entry["pyramid"]:
        assert level["key"] == f"{key}/level_{level['level']}"
        node = client[level["key"]]
        assert list(node.structure().shape) == level["shape"] and [list(a) for a in node.structure().chunks] == level["chunks"]
        block = node.read_block((0, 0))
        assert list(block.shape) == [level["chunks"][0][0], level["chunks"][1][0]]
        assert np.dtype(level["dtype"]) == block.dtype
        np.testing.assert_array_equal(block, tools.store.read_level(key, level["level"], slice(0, block.shape[0]), slice(0, block.shape[1])))
        x, y = pixel_to_stage(meta, 1, 2, level["level"])
        assert (level["origin_xy_m"][0] + 2 * level["pixel_size_m"], level["origin_xy_m"][1] + level["pixel_size_m"]) == pytest.approx((x, y), abs=1e-15)

    info = call("map_info", map=key)
    validator("map_view").validate(info)
    assert info["pyramid"] == entry["pyramid"]
    x0, x1, y0, y1 = info["stage_box_m"]
    for k in range(4):
        call("add_poi", map=key, x_m=x0 + (k + 0.5) * (x1 - x0) / 4, y_m=(y0 + y1) / 2, kind="eds_point", id=f"eds_{k}")
    first = finish(call, tools, call("score_map", map=key, top_k=2, level=0, tag="scan"))["summary"]["added_pois"]
    second = finish(call, tools, call("score_map", map=key, top_k=1, level=0, tag="scan"))["summary"]["added_pois"]
    assert len(first) == 2 and len(second) == 1
    store_meta = dict(client[tools.store.points_key(run, False)].metadata)
    assert store_meta["highlights"].startswith("highlights_")

    kinds = call("map_info", map=key)["poi_kinds"]
    assert [(k["kind"], k["label"]) for k in kinds] == [("candidate", "Scored region"), ("eds_point", "EDS point spectrum")]

    every = call("list_pois", map=key)
    assert every["total"] == 5
    assert [p["id"] for p in every["pois"] if p["kind"] == "candidate"] == [second[0]["id"]]

    crowded = call("list_pois", map=key, counts_above=3)
    assert [p["id"] for p in crowded["pois"]] == [second[0]["id"]]
    assert sum(c["count"] for c in crowded["cells"]) == 4 and crowded["total"] == 5
    for cell in crowded["cells"]:
        validator("poi_cell_count").validate(cell)
    left = call("list_pois", map=key, box_m=[x0, (x0 + x1) / 2, y0, y1], counts_above=10)
    assert "cells" not in left and sorted(p["id"] for p in left["pois"] if p["kind"] == "eds_point") == ["eds_0", "eds_1"]

    on_map = call("image_info", dataset=key)
    validator("image_info").validate(on_map)
    assert on_map["kind"] == "map" and on_map["image"] == f"{key}/level_0"


def test_image_info_refuses_a_stack_of_tiles_and_names_list_tiles(served):
    call, _tools, client = served
    tiles, _truth = make_tiles(stage_error_px=4.0, grid=3)
    stack = np.stack([t.image.astype(np.uint16) for t in tiles])
    client.create_container("stack").create_container("image").write_array(stack, key="HAADF", metadata={
        "pixel_size_m": PIXEL, "axes": "rows=+y, columns=+x"})
    refused = call("image_info", dataset="stack")
    assert refused["error"]["code"] == "no_image"
    assert f"holds {len(tiles)} images" in refused["error"]["message"] and "list_tiles" in refused["error"]["message"]


def test_a_run_gathered_in_tiled_with_spectra_gives_eds_points_where_they_were_taken(served):
    call, tools, client = served
    keys, tiles = write_images(client, prefix="eds_img")
    for t in tiles[::4]:
        client.create_container(f"eds_{t.index}")
    spectra = [f"eds_{t.index}" if t.index % 4 == 0 else "" for t in tiles]
    gathered = call("collect_map_run", image_keys=keys, spectrum_keys=spectra, spectra_at=[0.5, 0.25])
    assert gathered["spectra"] == 3
    run = gathered["run"]
    assert [str(k) for k in client[run]["spectrum_keys"].read()] == spectra and dict(client[run].metadata)["spectra_at"] == [0.5, 0.25]

    done = finish(call, tools, call("assemble_map", run=run))
    assert done["summary"]["eds_pois"] == 3
    key = done["result_keys"]["map"]
    assert [k["kind"] for k in call("map_info", map=key)["poi_kinds"]] == ["eds_point"]
    meta = tools.store.map_metadata(key)
    inverse = np.linalg.inv(np.asarray(meta["scan_linear_applied"]))
    points = call("list_pois", map=key, kind="eds_point")["pois"]
    assert sorted(p["links"][0] for p in points) == sorted(k for k in spectra if k)
    for point in points:
        index = spectra.index(point["links"][0])
        rows, cols = tiles[index].image.shape
        row, col = np.asarray(meta["tile_positions_px"][index]) + (np.array([rows, cols]) - 1) / 2 + inverse @ [-0.25 * rows, 0.0]
        assert (point["x_m"], point["y_m"]) == pytest.approx(pixel_to_stage(meta, row, col), abs=1e-11)
