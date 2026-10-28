"""Map tools through an in-memory MCP client, on a run file in asyncroscopy Mapper's layout."""

import asyncio
import json

import h5py
import jsonschema
import numpy as np
import pytest
from fastmcp import Client

from pyTEMlib.mcp_server import ENVELOPE_SCHEMA_PATH, create_server
from .test_mosaic import PIXEL, TILE, make_tiles


def write_run(path, grid=3):
    tiles, _truth = make_tiles(stage_error_px=6.0, grid=grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["plan"] = json.dumps({"fov_m": TILE * PIXEL})
        group = h5.create_group("tiles")
        for t in tiles:
            d = group.create_dataset(f"tile_{t.index:05d}", data=t.image.astype(np.uint16))
            d.attrs["stage_reported_m"] = [t.stage_xy_m[0], t.stage_xy_m[1], 0.0, 0.0, 0.0]
            d.attrs["pixel_size_m"] = PIXEL
            d.attrs["index"] = t.index
    return str(path)


@pytest.fixture
def run_file(tmp_path):
    return write_run(tmp_path / "map_test.h5")


def call(client, tool, **args):
    result = asyncio.get_event_loop().run_until_complete(client.call_tool(tool, args))
    return result.structured_content if result.structured_content is not None else result.data


def open_client(**kwargs):
    mcp, tools = create_server(include_basic=False, **kwargs)
    c = Client(mcp)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(c.__aenter__())
    return c, tools, loop


@pytest.fixture
def client(tmp_path):
    c, tools, loop = open_client(local_roots=[tmp_path])
    yield c, tools
    loop.run_until_complete(c.__aexit__(None, None, None))
    loop.close()


@pytest.fixture
def client_factory():
    opened = []

    def make(**kwargs):
        c, tools, loop = open_client(**kwargs)
        opened.append((c, loop))
        return c, tools

    yield make
    for c, loop in opened:
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


def test_level_shapes_stored_and_pixel_stage_round_trip_above_level_0(client, run_file):
    from pyTEMlib.mcp_server.map_store import level_shape_of, pixel_to_stage, stage_box_to_pixels

    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    meta = tools.store.map_metadata(map_key)
    assert meta["levels"] >= 3
    legacy = {k: v for k, v in meta.items() if k != "level_shapes"}  # a map written before level_shapes existed
    for level in range(meta["levels"]):
        actual = tools.store.level_shape(map_key, level)
        assert tuple(meta["level_shapes"][level]) == actual
        assert level_shape_of(meta, level) == actual
        assert level_shape_of(legacy, level) == actual

    pixel0 = meta["pixel_size_m"]
    for level in range(1, meta["levels"]):
        h, w = level_shape_of(meta, level)
        quarter = 0.25 * pixel0 * 2**level
        for row, col in [(0, 0), (h // 2, w // 3), (h - 1, w - 1)]:
            x, y = pixel_to_stage(meta, row, col, level)
            rows, cols = stage_box_to_pixels(meta, [x - quarter, x + quarter, y - quarter, y + quarter], level)
            assert (rows.start, rows.stop, cols.start, cols.stop) == (row, row + 1, col, col + 1), (level, row, col)
        # the whole map at this level, clipped to the stored shape
        x_min, y_min = pixel_to_stage(meta, -0.5, -0.5)
        x_max, y_max = pixel_to_stage(meta, meta["canvas_shape"][0] - 0.5, meta["canvas_shape"][1] - 0.5)
        rows, cols = stage_box_to_pixels(meta, [x_min - 1e-6, x_max + 1e-6, y_min - 1e-6, y_max + 1e-6], level)
        assert (rows.stop, cols.stop) == (h, w)
        assert tools.store.read_level(map_key, level, rows, cols).shape == (h, w)


def test_concurrent_add_poi_keeps_every_point(client, run_file):
    import threading

    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    info = call(c, "map_info", map=map_key)
    x0, x1, y0, y1 = info["stage_box_m"]
    added = []

    def add(k):
        added.append(tools.store.add_pois(map_key, [{"id": f"p{k}", "x_m": (x0 + x1) / 2, "y_m": (y0 + y1) / 2, "kind": "note"}]))

    threads = [threading.Thread(target=add, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = sorted(p["id"] for p in tools.store.map_metadata(map_key)["pois"])
    assert ids == sorted(f"p{k}" for k in range(16))


def test_assemble_check_reads_no_images(client, run_file, monkeypatch):
    c, tools = client
    monkeypatch.setattr(tools.store, "open_run", lambda *a, **k: pytest.fail("pre-check loaded tile images"))
    reserved = tools.jobs.reserve({"task": "assemble_map", "run": run_file})
    assert reserved["estimate"]["tiles"] == 9
    tiles = call(c, "list_tiles", run=run_file)
    tiles = tiles["result"] if isinstance(tiles, dict) else tiles
    assert len(tiles) == 9 and tiles[0]["shape"] == [TILE, TILE]


def test_server_object_exported_and_create_server_repeatable():
    import pyTEMlib.mcp_server as package
    from pyTEMlib.mcp_server import mcp
    from pyTEMlib.mcp_server import basic_tools

    assert mcp is package.default_server()[0]
    first, first_tools = create_server()
    second, _ = create_server()
    assert first is not second

    async def names(server):
        return sorted(t.name for t in await server.list_tools())

    loop = asyncio.new_event_loop()
    try:
        first_names, second_names, basic_names = (loop.run_until_complete(names(s)) for s in (first, second, basic_tools.mcp))
    finally:
        loop.close()
    assert first_names == second_names
    assert len(first_names) == len(set(first_names))
    assert "assemble_map" in first_names and "assemble_map" not in basic_names
    assert set(basic_names) <= set(first_names)

    first_tools.register(first)  # same tools again: no-op
    with pytest.raises(RuntimeError):
        _, other_tools = create_server(include_basic=False)
        other_tools.register(first)


def test_restitching_keeps_points_of_interest(client, run_file):
    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    info = call(c, "map_info", map=map_key)
    x0, x1, y0, y1 = info["stage_box_m"]
    poi = call(c, "add_poi", map=map_key, x_m=(x0 + x1) / 2, y_m=(y0 + y1) / 2, kind="eds_point", links=["spectrum_001.h5"])

    again = finish(c, tools, call(c, "assemble_map", run=run_file, refine=False)["job_id"])["result_keys"]["map"]
    assert again == map_key
    meta = tools.store.map_metadata(map_key)
    assert meta["refined"] is False  # the map really was rewritten
    assert meta["pois"] == [poi]


def test_local_paths_outside_the_roots_are_refused(client_factory, tmp_path):
    from pyTEMlib.mcp_server.map_store import LocalPathRefused, MapStore

    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    run_outside = write_run(outside / "secret.h5", grid=2)
    c, tools = client_factory(local_roots=[inside])

    reply = call(c, "assemble_map", run=run_outside)
    jsonschema.validate(reply, envelope("error_response"))
    assert reply["error"]["code"] == "plan_rejected"
    assert str(inside.resolve()) in reply["error"]["message"]
    with pytest.raises(LocalPathRefused, match="outside the directories"):
        tools.store.map_metadata(run_outside)
    assert not (outside / "secret_map.h5").exists()

    # the same file is fine once its directory is a root
    assert MapStore(local_roots=[outside]).is_local(run_outside)
    assert not MapStore(local_roots=[inside]).is_local(str(inside / "missing.h5"))  # not a file: treated as a Tiled key


def test_abort_during_registration_stops_promptly(client, run_file, monkeypatch):
    import threading
    import time

    from pyTEMlib.image_tools import mosaic

    real = mosaic.phase_cross_correlation
    calls = []
    started = threading.Event()

    def slow(*args, **kwargs):  # stands in for a large tile pair (~0.3 s each on 512 x 512)
        calls.append(1)
        started.set()
        time.sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(mosaic, "phase_cross_correlation", slow)
    c, tools = client
    job_id = call(c, "assemble_map", run=run_file)["job_id"]
    assert started.wait(30)
    status = call(c, "analysis_job_status", job_id=job_id)
    assert status["state"] == "running"
    pairs = status["progress"]["total"] - 2
    assert pairs >= 12  # 3 x 3 grid: 12 edge pairs, plus diagonals if they overlap enough

    t0 = time.monotonic()
    aborted = call(c, "analysis_job_abort", job_id=job_id)
    assert time.monotonic() - t0 < 2.0
    assert aborted["state"] == "aborted", aborted
    assert len(calls) < pairs
    assert call(c, "analysis_job_results", job_id=job_id)["result_keys"] == {}


def test_oversized_map_refused_before_reading_pixels(client_factory, run_file, tmp_path, monkeypatch):
    c, tools = client_factory(local_roots=[tmp_path], max_canvas_pixels=100_000)
    monkeypatch.setattr(tools.store, "open_run", lambda *a, **k: pytest.fail("limit checked after loading tile images"))
    reply = call(c, "assemble_map", run=run_file)
    jsonschema.validate(reply, envelope("error_response"))
    assert reply["error"]["code"] == "plan_rejected"
    assert "megapixels" in reply["error"]["message"] and "box_m" in reply["error"]["message"]

    # a part of the run that fits is accepted, and the estimate gives the canvas size
    first = tools.store.scan_run(run_file).tiles[0]
    x, y = first.stage_xy_m
    reserved = tools.jobs.reserve({"task": "assemble_map", "run": run_file, "box_m": [x - 1e-12, x + 1e-12, y - 1e-12, y + 1e-12]})
    assert reserved["estimate"]["canvas_shape"] == [TILE, TILE]


def test_cli_defaults_keep_the_asyncroscopy_launch_command(monkeypatch, tmp_path):
    from pyTEMlib.mcp_server import __main__ as cli

    seen = {}

    def fake_create_server(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise SystemExit(0)

    monkeypatch.setattr(cli, "create_server", fake_create_server)
    with pytest.raises(SystemExit):
        cli.main(["--tiled", "http://localhost:9091", "--api-key", "secret", "--port", "8003"])
    assert seen["args"] == ("http://localhost:9091", "secret")
    assert seen["kwargs"] == {"local_roots": None, "max_canvas_pixels": 400_000_000}
    with pytest.raises(SystemExit):
        cli.main(["--local-root", str(tmp_path), "--local-root", "data", "--max-canvas-megapixels", "50"])
    assert seen["kwargs"] == {"local_roots": [str(tmp_path), "data"], "max_canvas_pixels": 50_000_000}
