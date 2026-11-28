"""Map tools through an in-memory MCP client, on a run file in asyncroscopy Mapper's layout."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from fastmcp import Client

jsonschema = pytest.importorskip("jsonschema")  # the job replies are validated against the envelope schema

from pytemlib_mcp.analysis_server import create_server
from pytemlib_mcp.analysis_server.map_tools import MapTools, drift_measured, measured_drift_only
from pyTEMlib.image_tools import BaselineScorer
from tests.conftest import PIXEL, TILE, make_tiles


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
    schema = json.loads((Path(__file__).parent / "envelope.schema.json").read_text(encoding="utf-8"))
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
    assert added[0]["scorer_version"] == BaselineScorer.version
    listed = {p["id"]: p for p in call(c, "list_pois", map=map_key)["pois"]}
    assert all(listed[p["id"]]["scorer_version"] == BaselineScorer.version for p in added)

    poi = call(c, "add_poi", map=map_key, x_m=(x0 + x1) / 2, y_m=(y0 + y1) / 2, kind="eds_point", reason="spectrum here", links=["spectrum_001.h5"])
    assert poi["links"] == ["spectrum_001.h5"]
    outside = call(c, "add_poi", map=map_key, x_m=x1 + 1e-6, y_m=y0)
    assert outside["error"]["code"] == "outside_map"
    pois = call(c, "list_pois", map=map_key)
    assert pois["total"] == 4
    pois = pois["pois"]
    assert len(pois) == 4
    assert [p["kind"] for p in pois].count("eds_point") == 1

    # Small models write the window as width then height.
    as_pair = call(c, "score_map", map=map_key, top_k=1, window_m=[100 * PIXEL, 80 * PIXEL], level=0)
    assert finish(c, tools, as_pair["job_id"])["summary"]["added_pois"]


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


def test_a_run_passed_as_a_map_is_refused_with_a_pointer_to_assemble_map(client, run_file):
    c, _tools = client
    reply = call(c, "score_map", map=run_file)
    assert reply["error"]["code"] == "plan_rejected"
    assert "assemble_map" in reply["error"]["message"]


def test_scorers_listed(client):
    c, _tools = client
    scorers = call(c, "list_scorers")
    scorers = scorers["result"] if isinstance(scorers, dict) else scorers
    assert any(s["name"] == "baseline" for s in scorers)


def test_level_shapes_stored_and_pixel_stage_round_trip_above_level_0(client, run_file):
    from pytemlib_mcp.analysis_server.map_store import level_shape_of, pixel_to_stage, stage_box_to_pixels

    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    meta = tools.store.map_metadata(map_key)
    assert meta["levels"] >= 3
    for level in range(meta["levels"]):
        actual = tools.store.level_shape(map_key, level)
        assert tuple(meta["level_shapes"][level]) == actual
        assert level_shape_of(meta, level) == actual
    # Maps without level_shapes round rows up at each halving.
    assert level_shape_of({"canvas_shape": [5, 7]}, 1) == (3, 3)

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
    ids = sorted(p["id"] for p in tools.store.list_pois(map_key))
    assert ids == sorted(f"p{k}" for k in range(16))


def test_assemble_check_reads_no_images(client, run_file, monkeypatch):
    c, tools = client
    monkeypatch.setattr(tools.store, "open_run", lambda *a, **k: pytest.fail("pre-check loaded tile images"))
    reserved = tools.jobs.reserve({"task": "assemble_map", "run": run_file})
    assert reserved["estimate"]["tiles"] == 9
    tiles = call(c, "list_tiles", run=run_file)
    assert tiles["total"] == 9 and tiles["tiles"][0]["shape"] == [TILE, TILE]


def test_create_server_is_repeatable_and_mounts_pytemlibs_tools():
    from pytemlib_mcp import server as basic_tools

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

    first_tools.register(first)  # registering the same tools again does nothing
    with pytest.raises(RuntimeError):
        _, other_tools = create_server(include_basic=False)
        other_tools.register(first)


def test_every_tool_says_whether_it_only_reads_and_marked_tools_still_run(tmp_path):
    from pytemlib_mcp.analysis_server.server import READ_ONLY, WRITES

    mcp, _ = create_server(local_roots=[tmp_path], annotations_path=str(tmp_path / "n.jsonl"), placements_path=str(tmp_path / "p.jsonl"))

    async def listed_and_called():
        async with Client(mcp) as c:
            return await c.list_tools(), await c.call_tool("list_scorers", {})

    loop = asyncio.new_event_loop()
    try:
        listed, scorers = loop.run_until_complete(listed_and_called())
    finally:
        loop.close()
    names = {t.name for t in listed}
    assert not READ_ONLY & WRITES
    assert names == READ_ONLY | WRITES, f"say whether these only read: {sorted(names ^ (READ_ONLY | WRITES))}"
    for t in listed:
        assert bool(t.annotations and t.annotations.read_only_hint) == (t.name in READ_ONLY), t.name
    assert not scorers.is_error


def test_restitching_keeps_points_of_interest(client, run_file):
    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    info = call(c, "map_info", map=map_key)
    x0, x1, y0, y1 = info["stage_box_m"]
    poi = call(c, "add_poi", map=map_key, x_m=(x0 + x1) / 2, y_m=(y0 + y1) / 2, kind="eds_point", links=["spectrum_001.h5"])

    again = finish(c, tools, call(c, "assemble_map", run=run_file, refine=False)["job_id"])["result_keys"]["map"]
    assert again != map_key  # a new version beside the first, which is kept
    assert tools.store.map_metadata(map_key)["refined"] is True and tools.store.map_metadata(again)["refined"] is False
    assert tools.store.current_map(run_file) == again
    assert tools.store.list_pois(again) == [poi] and tools.store.list_pois(map_key) == [poi]


def test_a_map_of_too_few_tiles_reports_no_uncertainty_instead_of_infinity(client, tmp_path):
    c, tools = client
    run = write_run(tmp_path / "small.h5", grid=2)
    results = finish(c, tools, call(c, "assemble_map", run=run)["job_id"])
    stderr = results["summary"]["scan_linear"]["stderr"]
    assert stderr is None
    assert results["summary"]["scan_linear"]["applied"] is False
    json.dumps(tools.store.map_metadata(results["result_keys"]["map"]), allow_nan=False)


def test_local_paths_outside_the_roots_are_refused(client_factory, tmp_path):
    from pytemlib_mcp.analysis_server.map_store import LocalPathRefused, MapStore

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
    assert not MapStore(local_roots=[inside]).is_local(str(inside / "missing.h5"))  # a missing file is treated as a Tiled key


def test_abort_during_registration_stops_promptly(client, run_file, monkeypatch):
    import threading
    import time

    from pyTEMlib.image_tools import image_mosaic as mosaic

    real = mosaic.bounded_shift
    calls = []
    started = threading.Event()

    def slow(*args, **kwargs):  # stands in for a large tile pair that takes about 0.3 s
        calls.append(1)
        started.set()
        time.sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(mosaic, "bounded_shift", slow)
    c, tools = client
    job_id = call(c, "assemble_map", run=run_file)["job_id"]
    assert started.wait(30)
    status = call(c, "analysis_job_status", job_id=job_id)
    assert status["state"] == "running"
    pairs = (status["progress"]["total"] - 1) // 2  # pairs matched, then as many steps of pyramid blocks, then the write
    assert pairs >= 12  # a 3 x 3 grid has 12 edge pairs

    t0 = time.monotonic()
    aborted = call(c, "analysis_job_abort", job_id=job_id)
    assert time.monotonic() - t0 < 2.0
    assert aborted["state"] == "aborted", aborted
    assert len(calls) < pairs
    assert call(c, "analysis_job_results", job_id=job_id)["result_keys"] == {}


def test_oversized_map_refused_before_reading_pixels(client_factory, run_file, tmp_path, monkeypatch):
    from pytemlib_mcp.analysis_server import map_store

    c, tools = client_factory(local_roots=[tmp_path], max_canvas_pixels=100_000)
    monkeypatch.setattr(map_store.TileSource, "image", lambda *a, **k: pytest.fail("limit checked after loading tile images"))
    started = call(c, "assemble_map", run=run_file)  # the request answers at once and the job checks the canvas
    status = tools.jobs.wait(started["job_id"])
    assert status["state"] == "failed"
    assert "megapixels" in status["last_error"]["message"] and "box_m" in status["last_error"]["message"]

    # a part of the run that fits is stitched
    x, y = tools.store.scan_run(run_file).stage_xy_m[0]
    monkeypatch.undo()
    job = tools.jobs.start(tools.jobs.reserve({"task": "assemble_map", "run": run_file, "box_m": [x - 1e-12, x + 1e-12, y - 1e-12, y + 1e-12]})["reservation_id"])
    assert tools.jobs.wait(job["job_id"])["state"] == "done"


def test_cli_defaults_keep_the_asyncroscopy_launch_command(monkeypatch, tmp_path):
    from pytemlib_mcp.analysis_server import __main__ as cli

    seen = {}

    def fake_create_server(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise SystemExit(0)

    outputs = tmp_path / "out"
    monkeypatch.setenv("ASYNCROSCOPY_OUTPUTS", str(outputs))
    monkeypatch.delenv("ASYNCROSCOPY_TILED_API_KEY", raising=False)
    monkeypatch.delenv("TILED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "create_server", fake_create_server)
    files = {"trace_path": str(outputs / "trace" / "analysis.jsonl"), "annotations_path": str(outputs / "annotations" / "notes.jsonl"),
             "placements_path": str(outputs / "placements" / "placements.jsonl")}
    with pytest.raises(SystemExit):
        cli.main(["--tiled", "http://localhost:9091", "--api-key", "secret", "--port", "8003"])
    assert seen["args"] == ("http://localhost:9091", "secret")
    assert seen["kwargs"] == {"local_roots": [str(tmp_path)], "max_canvas_pixels": 400_000_000_000, "max_jobs": 2, "local_runs": None, **files}
    monkeypatch.setenv("ASYNCROSCOPY_TILED_API_KEY", "from-env")
    with pytest.raises(SystemExit):
        cli.main(["--local-root", str(tmp_path), "--local-root", "data", "--max-canvas-megapixels", "50", "--max-jobs", "3", "--trace", ""])
    assert seen["args"] == (None, "from-env")
    assert seen["kwargs"] == {"local_roots": [str(tmp_path), "data"], "max_canvas_pixels": 50_000_000, "max_jobs": 3, "local_runs": None, **files, "trace_path": None}


def test_status_can_wait_for_the_job(client, run_file):
    c, tools = client
    started = call(c, "assemble_map", run=run_file)
    status = call(c, "analysis_job_status", job_id=started["job_id"], wait_s=25)
    assert status["state"] == "done", status
    jsonschema.validate(status, envelope("status_response"))
    unknown = call(c, "analysis_job_status", job_id="nope", wait_s=1)
    # An unknown job id may be an instrument job. The instrument's wait_for_job waits on both.
    assert unknown["error"]["code"] == "unknown_job"
    assert unknown["next_actions"] == [{"server": "instrument", "tool": "wait_for_job", "args": {"job_id": "nope"}}]


def test_map_info_names_the_runs_stitched_map_and_counts_tiles(client, run_file):
    c, tools = client
    with pytest.raises(Exception, match="not been stitched.*assemble_map"):
        call(c, "map_info", map=run_file)
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    with pytest.raises(Exception, match=r"is a Mapper run, not a stitched map; its newest stitched map is .*_map_\w+\.h5"):
        call(c, "map_info", map=run_file)
    with pytest.raises(ValueError, match="its newest stitched map is"):
        tools.store.map_metadata(run_file)
    info = call(c, "map_info", map=map_key)
    assert info["tiles"] == 9 and info["run"] == run_file


def test_list_pois_always_answers_total_and_pois(client, run_file):
    c, tools = client
    map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
    x0, x1, y0, y1 = call(c, "map_info", map=map_key)["stage_box_m"]
    for k in range(5):
        call(c, "add_poi", map=map_key, x_m=x0 + (k + 1) * (x1 - x0) / 7, y_m=(y0 + y1) / 2, kind="note" if k % 2 else "eds_point")
    for args in ({}, {"kind": "note"}, {"box_m": [x0, (x0 + x1) / 2, y0, y1]}, {"kind": "eds_point", "limit": 1}):
        reply = call(c, "list_pois", map=map_key, **args)
        stored = tools.list_pois(map_key, **args)
        assert set(reply) == {"total", "pois"} and reply["total"] == stored["total"]
        assert [p["id"] for p in reply["pois"]] == [p["id"] for p in stored["pois"]]
        # Replies round positions to 0.01 nm so a model can copy them. The store keeps them exact.
        assert all(p["x_m"] == round(s["x_m"], 11) and abs(p["x_m"] - s["x_m"]) <= 5e-12 for p, s in zip(reply["pois"], stored["pois"]))
        assert not any(p.get("scorer_version", "") is None or p.get("links") == [] for p in reply["pois"])  # empty fields are left out to save context
        assert len(reply["pois"]) == min(reply["total"], args.get("limit", 200))
    assert call(c, "list_pois", map=map_key)["total"] == 5
    assert call(c, "list_pois", map=map_key, kind="eds_point", limit=1)["total"] == 3
    assert call(c, "list_pois", map=map_key, kind="all")["total"] == 5  # what small models ask for
    assert call(c, "list_pois", map=map_key, kind="eds_point, note, revisit")["total"] == 5
    missing = call(c, "list_pois", map=map_key, kind="poi")
    assert missing["total"] == 0 and missing["kinds_on_map"] == ["eds_point", "note"] and missing["total_on_map"] == 5
    assert call(c, "list_pois", map=map_key, box_m=[0, 0, 0, 0])["total"] == 5  # small models send this to mean no box
    outside = call(c, "list_pois", map=map_key, box_m=[x1 + 1e-6, x1 + 2e-6, y0, y1])
    assert outside["total"] == 0 and outside["total_on_map"] == 5
    # A box shrunk to one point with an unknown kind returns the nearest first.
    second = x0 + 2 * (x1 - x0) / 7
    near = call(c, "list_pois", map=map_key, box_m=[second, second, (y0 + y1) / 2, (y0 + y1) / 2], kind="revisit")
    assert near["pois"][0]["x_m"] == pytest.approx(second, abs=5e-12) and near["pois"][0]["distance_m"] == pytest.approx(0.0, abs=1e-12)
    assert near["total"] == 5 and near["kinds_on_map"] == ["eds_point", "note"] and "nearest" in near["hint"]
    notes = call(c, "list_pois", map=map_key, box_m=[x0, x0, y0, y0], kind="note")
    assert {p["kind"] for p in notes["pois"]} == {"note"} and "kinds_on_map" not in notes


def test_every_tool_call_is_traced_with_its_arguments_and_outcome(tmp_path, run_file):
    trace = tmp_path / "analysis.jsonl"
    c, tools, loop = open_client(local_roots=[tmp_path], trace_path=str(trace))
    try:
        call(c, "list_pois", map="no_such_map", kind="candidate")
    except Exception:
        pass
    call(c, "list_scorers")
    lines = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [r["tool"] for r in lines] == ["list_pois", "list_scorers"]
    assert lines[0]["arguments"] == {"map": "no_such_map", "kind": "candidate"} and lines[0]["ok"] is False and lines[0]["error"]
    assert lines[1]["ok"] is True and lines[1]["source"] == "analysis" and lines[1]["result"]
    # Lines are numbered like every asyncroscopy trace file, so the merged trace reads them in order.
    assert [r["seq"] for r in lines] == [1, 2] and all("written" in r and "duration_s" in r for r in lines)
    loop.run_until_complete(c.__aexit__(None, None, None))


def test_assemble_progress_only_goes_up(tmp_path):
    from pytemlib_mcp.analysis_server.map_store import MapStore
    from pytemlib_mcp.analysis_server.map_tools import PROGRESS_STEPS, MapTools

    from .test_analysis_map_store_reads import write_layout2

    path, _ = write_layout2(tmp_path / "run.h5")
    tools = MapTools(MapStore(local_roots=[tmp_path]))
    seen = []
    tools._run_assemble({"run": path, "eds_pois": True}, lambda done, total=None: seen.append(done))
    assert seen == sorted(seen) and seen[-1] == PROGRESS_STEPS


def test_a_registered_scorer_is_listed_and_used_by_score_map(client, run_file):
    from pytemlib_mcp.analysis_server.map_tools import SCORERS

    class Brightest:
        name, version, description = "brightest", "0.1.0", "mean intensity (a test scorer)"

        def score(self, region, metadata):
            return float(np.mean(region)) / 65535.0, "mean intensity"

    SCORERS.register(Brightest)
    try:
        c, tools = client
        assert {"name": "brightest", "version": "0.1.0", "description": "mean intensity (a test scorer)"} in call(c, "list_scorers")["result"]
        map_key = finish(c, tools, call(c, "assemble_map", run=run_file)["job_id"])["result_keys"]["map"]
        added = finish(c, tools, call(c, "score_map", map=map_key, scorer="brightest", top_k=2)["job_id"])["summary"]["added_pois"]
        assert len(added) == 2 and {p["source"] for p in added} == {"scorer:brightest"} and {p["scorer_version"] for p in added} == {"0.1.0"}
    finally:
        SCORERS.unregister("brightest")


def test_a_map_tool_given_a_made_up_key_answers_with_the_newest_keys(tmp_path):
    class Keys(list):
        def tail(self, n):
            return self[-n:]

    class Tiled(dict):
        def keys(self):
            return Keys(super().keys())

    mcp, tools = create_server(local_roots=[tmp_path], include_basic=False)
    tools.store._client = Tiled({f"map_{i}_map": None for i in range(3)})

    async def calls():
        async with Client(mcp) as c:
            return (await c.call_tool("map_info", {"map": "new_map"}, raise_on_error=False),
                    await c.call_tool("list_pois", {"map": "new_map"}, raise_on_error=False),
                    await c.call_tool("get_region", {"map": "new_map", "box_m": [0, 1, 0, 1], "level": 0}, raise_on_error=False))

    loop = asyncio.new_event_loop()
    try:
        replies = loop.run_until_complete(calls())
    finally:
        loop.close()
    for reply in replies:
        error = json.loads(reply.content[0].text)["error"]
        assert not reply.is_error and error["code"] == "unknown_dataset" and "'new_map'" in error["message"]
        assert error["newest_keys"] == ["map_2_map", "map_1_map", "map_0_map"]


def test_a_drift_fit_is_reported_only_when_image_matches_measure_it():
    drift = {"times_s": [0.0, 83.4, 166.7, 250.1], "total_nm": 145.6, "mean_rate_nm_per_s": 0.58}
    assert not drift_measured({"matches_used": 2, "drift": drift})
    assert drift_measured({"matches_used": 8, "drift": drift})
    assert not drift_measured({"matches_used": 400})
    unmeasured = measured_drift_only({"matches_used": 2, "fallback_tiles": 98, "drift": drift})
    assert "drift" not in unmeasured and "unknown" in unmeasured["drift_unmeasured"] and unmeasured["fallback_tiles"] == 98
    assert measured_drift_only({"matches_used": 8, "drift": drift})["drift"] == drift
    assert measured_drift_only({"matches_used": 0}) == {"matches_used": 0, "drift_unmeasured": unmeasured["drift_unmeasured"]}


# collect_map_run gathers tiles taken one by one
class _Node:
    structure_family = "array"

    def __init__(self, array, metadata=None):
        self._array, self.metadata = array, dict(metadata or {})

    def structure(self):
        return SimpleNamespace(shape=self._array.shape)

    def read(self):
        return self._array


class _Container(dict):
    structure_family = "container"

    def __init__(self, metadata=None):
        super().__init__()
        self.metadata = dict(metadata or {})

    def create_container(self, key, metadata=None):
        self[key] = _Container(metadata)
        return self[key]

    def write_array(self, array, key, metadata=None):
        self[key] = _Node(np.asarray(array), metadata)

    def delete_contents(self, key, **_):
        self.pop(key, None)


class _TiledStore:
    """A stand-in for the parts of a Tiled MapStore that collect_run uses."""

    def __init__(self, client):
        self.client = client

    def map_metadata(self, key):
        raise KeyError(key)

    def sample_of(self, key):
        return {}


def test_images_that_record_no_position_are_placed_by_the_callers_stage_readings():
    client = _Container()
    for k in range(2):  # saved the way a real instrument does with vendor metadata only
        client.create_container(f"img{k}.h5").create_container("image").write_array(np.full((8, 16), k), "HAADF", {"Detector": "HAADF"})
    tools = MapTools(_TiledStore(client))
    with pytest.raises(ValueError, match="does not record where it was taken"):
        tools.collect_run(["img0.h5", "img1.h5"])
    with pytest.raises(ValueError, match="needs fov_m"):
        tools.collect_run(["img0.h5", "img1.h5"], stage_xy_m=[[0, 0], [1e-7, 0]])
    with pytest.raises(ValueError, match="one .x, y. per image"):
        tools.collect_run(["img0.h5", "img1.h5"], stage_xy_m=[[0, 0]], fov_m=1.6e-7)
    assert [k for k in client if k.startswith("map_")] == []

    out = tools.collect_run(["img0.h5", "img1.h5"], {"workflow": "map_grid"}, stage_xy_m=[[1e-6, 2e-6], [1.1e-6, 2e-6]], fov_m=1.6e-7)
    assert out["tiles"] == 2 and out["placed_by_caller"] == 2 and "do not record" in out["note"]
    assert out["pixel_size_m"] == pytest.approx(1e-8)  # field of view divided by image columns
    tiles = client[out["run"]]["tiles"]
    assert [tiles[f"tile_{k:05d}"].metadata["stage_reported_m"] for k in range(2)] == [[1e-6, 2e-6], [1.1e-6, 2e-6]]
    assert tiles["tile_00001"].metadata["source_key"] == "img1.h5" and int(tiles["tile_00001"].read()[0, 0]) == 1


def test_tiles_reached_by_image_shift_are_placed_by_the_callers_reading_when_the_shift_is_not_recorded():
    client = _Container()
    placed = {"pixel_size_m": 1e-8, "stage_reported_m": "[1e-6, 2e-6, 0.0, 0.0, 0.0]", "axes": "rows=+y, columns=+x", "scan_region": "[0.0, 0.0, 1.0, 1.0]"}
    client.create_container("img0.h5").create_container("image").write_array(np.zeros((8, 16)), "HAADF", placed)
    client.create_container("img1.h5").create_container("image").write_array(np.ones((8, 16)), "HAADF", placed)
    client.create_container("img2.h5").create_container("image").write_array(np.ones((8, 16)), "HAADF", {**placed, "image_shift_m": "[3e-7, 0.0]"})
    tools = MapTools(_TiledStore(client))
    # The stage did not move so only the caller knows the shift.
    out = tools.collect_run(["img0.h5", "img1.h5", "img2.h5"], stage_xy_m=[[0.9e-6, 2e-6], [1.1e-6, 2e-6], [5e-6, 5e-6]], fov_m=1.6e-7)
    tiles = client[out["run"]]["tiles"]
    centres = [tiles[f"tile_{k:05d}"].metadata["stage_reported_m"] for k in range(3)]
    assert centres[0] == pytest.approx([0.9e-6, 2e-6]) and centres[1] == pytest.approx([1.1e-6, 2e-6])
    assert centres[2] == pytest.approx([1.3e-6, 2e-6])  # a recorded shift wins over the caller's value
    assert out["placed_by_caller"] == 2 and "image shift" in out["note"]


def test_a_gathered_run_and_its_map_are_saved_as_local_files_when_the_server_is_told_to(tmp_path):
    from pytemlib_mcp.analysis_server.map_store import MapStore

    tiles, _truth = make_tiles(stage_error_px=6.0, grid=2)
    remote = _Container()  # the instrument's Tiled is only read
    for t in tiles:
        remote.create_container(f"img{t.index}.h5").create_container("image").write_array(t.image.astype(np.uint16), "HAADF", {})
    store = MapStore(None, local_roots=[tmp_path])
    store._client = remote
    tools = MapTools(store, local_runs=tmp_path / "maps")
    before = set(remote)
    out = tools.collect_run([f"img{t.index}.h5" for t in tiles], {"workflow": "map_grid"},
                            stage_xy_m=[[t.stage_xy_m[0] + TILE * PIXEL / 2, t.stage_xy_m[1] + TILE * PIXEL / 2] for t in tiles], fov_m=TILE * PIXEL)
    run = out["run"]
    assert Path(run).parent == (tmp_path / "maps").resolve() and Path(run).exists() and set(remote) == before
    assert len(store.scan_run(run)) == len(tiles)
    result = tools._run_assemble({"run": run, "refine": True}, lambda *a: None)
    made = Path(result["result_keys"]["map"])
    assert made.parent == Path(run).parent and made.exists() and set(remote) == before
    assert tools.map_info(str(made))["tiles"] == len(tiles)

    outside = MapTools(MapStore(None, local_roots=[tmp_path / "elsewhere"]), local_runs=tmp_path / "maps2")
    outside.store._client = remote
    with pytest.raises(ValueError, match="outside the directories"):
        outside.collect_run(["img0.h5"], stage_xy_m=[[0.0, 0.0]], fov_m=TILE * PIXEL)
    assert not (tmp_path / "maps2").exists() or not list((tmp_path / "maps2").iterdir())
