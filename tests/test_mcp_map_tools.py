"""Map tools through an in-memory MCP client, on a run file in asyncroscopy Mapper's layout."""

import asyncio
import json

import h5py
import jsonschema
import numpy as np
import pytest
from fastmcp import Client

from pyTEMlib.mcp_server import ENVELOPE_SCHEMA_PATH, create_server
from test_mosaic import PIXEL, TILE, make_tiles


@pytest.fixture
def run_file(tmp_path):
    tiles, _truth = make_tiles(stage_error_px=6.0, grid=3)
    path = tmp_path / "map_test.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["plan"] = json.dumps({"fov_m": TILE * PIXEL})
        group = h5.create_group("tiles")
        for t in tiles:
            d = group.create_dataset(f"tile_{t.index:05d}", data=t.image.astype(np.uint16))
            d.attrs["stage_reported_m"] = [t.stage_xy_m[0], t.stage_xy_m[1], 0.0, 0.0, 0.0]
            d.attrs["pixel_size_m"] = PIXEL
            d.attrs["index"] = t.index
    return str(path)


def call(client, tool, **args):
    result = asyncio.get_event_loop().run_until_complete(client.call_tool(tool, args))
    return result.structured_content if result.structured_content is not None else result.data


@pytest.fixture
def client():
    mcp, tools = create_server(include_basic=False)
    c = Client(mcp)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(c.__aenter__())
    yield c, tools
    loop.run_until_complete(c.__aexit__(None, None, None))
    loop.close()


def envelope(name):
    schema = json.loads(ENVELOPE_SCHEMA_PATH.read_text())
    return {"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]}


def finish(client, tools, job_id):
    tools.jobs.wait(job_id)
    status = call(client, "analysis_job_status", job_id=job_id)
    jsonschema.validate(status, envelope("status_response"))
    assert status["state"] == "done", status
    results = call(client, "analysis_job_results", job_id=job_id)
    jsonschema.validate(results, envelope("results_response"))
    return results


def test_stitch_view_score_and_annotate(client, run_file):
    c, tools = client
    started = call(c, "assemble_map", run=run_file)
    results = finish(c, tools, started["job_id"])
    map_key = results["result_keys"]["map"]
    assert results["summary"]["tiles"] == 9

    info = call(c, "map_info", map=map_key)
    x0, x1, y0, y1 = info["stage_box_m"]
    assert x1 - x0 == pytest.approx(info["canvas_shape"][1] * PIXEL, rel=1e-6)

    region = asyncio.get_event_loop().run_until_complete(c.call_tool("get_region", {"map": map_key, "box_m": [x0, x0 + 200 * PIXEL, y0, y0 + 100 * PIXEL], "level": 0}))
    meta = region.structured_content
    assert meta["shape"][1] in (200, 201) and meta["shape"][0] in (100, 101)
    assert any(getattr(block, "type", "") == "image" for block in region.content)

    scored = call(c, "score_map", map=map_key, top_k=3, window_m=100 * PIXEL, level=0)
    score_results = finish(c, tools, scored["job_id"])
    added = score_results["summary"]["added_pois"]
    assert len(added) == 3 and added[0]["score"] >= added[-1]["score"]
    assert all(x0 <= p["x_m"] <= x1 and y0 <= p["y_m"] <= y1 for p in added)
    assert added[0]["scorer_version"] == "1.0.0"

    poi = call(c, "add_poi", map=map_key, x_m=(x0 + x1) / 2, y_m=(y0 + y1) / 2, kind="eds_point", reason="spectrum here", links=["spectrum_001.h5"])
    assert poi["links"] == ["spectrum_001.h5"]
    outside = call(c, "add_poi", map=map_key, x_m=x1 + 1e-6, y_m=y0)
    assert outside["error"]["code"] == "outside_map"
    pois = call(c, "list_pois", map=map_key)
    pois = pois["result"] if isinstance(pois, dict) else pois
    assert len(pois) == 4
    assert [p["kind"] for p in pois].count("eds_point") == 1


def test_bad_run_is_refused_through_the_envelope(client, tmp_path):
    c, _tools = client
    reply = call(c, "assemble_map", run=str(tmp_path / "missing.h5"))
    jsonschema.validate(reply, envelope("error_response"))
    assert reply["error"]["code"] == "plan_rejected"


def test_unknown_scorer_refused(client, run_file):
    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    reply = call(c, "score_map", map=map_key, scorer="does-not-exist")
    assert reply["error"]["code"] == "plan_rejected"


def test_scorers_listed(client):
    c, _tools = client
    scorers = call(c, "list_scorers")
    scorers = scorers["result"] if isinstance(scorers, dict) else scorers
    assert any(s["name"] == "baseline" for s in scorers)
