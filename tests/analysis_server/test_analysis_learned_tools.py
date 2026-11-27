"""segment_sam_and_save and find_atoms_blobnet_and_save: pyTEMlib's learned tools on a Tiled image, with the result
written back to a real Tiled server. The models themselves are replaced, so no torch or weights are needed."""

import asyncio
import json

import numpy as np
import pytest
from fastmcp import Client

mcp_analysis = pytest.importorskip("pyTEMlib.mcp_analysis")
if not hasattr(mcp_analysis, "segment_sam"):
    pytest.skip("this pyTEMlib has no learned tools", allow_module_level=True)
from pyTEMlib.image_tools import image_learned  # noqa: E402
from tiled.client import from_uri  # noqa: E402
from tiled.server import SimpleTiledServer  # noqa: E402

from pyTEMlib.analysis_server import create_server  # noqa: E402


@pytest.fixture(scope="module")
def tiled(tmp_path_factory):
    server = SimpleTiledServer(directory=tmp_path_factory.mktemp("tiled"))
    client = from_uri(server.uri)
    run = client.create_container("run_7", metadata={"acquisition_type": "image", "sample_id": "S1", "sample_load_id": "L3"})
    run.write_array(np.random.default_rng(0).random((64, 80)), key="haadf", metadata={"pixel_size_m": 5e-10})
    yield server, client
    server.close()


@pytest.fixture
def models(monkeypatch):
    labels = np.zeros((64, 80), dtype=np.uint16)
    labels[10:20, 10:20], labels[40:50, 30:44] = 1, 2
    areas = [{"id": 2, "area_px": 140, "x": 44.5, "y": 36.5, "equiv_diameter_px": 13.3, "circularity": 0.8, "confidence": 0.9},
             {"id": 1, "area_px": 100, "x": 14.5, "y": 14.5, "equiv_diameter_px": 11.3, "circularity": 0.79, "confidence": 0.95}]
    calls = []

    def sam(values, model, *args):
        calls.append(("sam", values.shape, model, args))
        return labels, [dict(a) for a in areas]

    def blobnet(values, threshold, distance):
        calls.append(("blobnet", values.shape, threshold, distance))
        return np.array([[5.0, 6.0, 0.9], [20.0, 30.0, 0.8], [40.0, 70.0, 0.6]]), np.zeros(values.shape)

    monkeypatch.setattr(image_learned, "sam2_segment", sam)
    monkeypatch.setattr(image_learned, "blobnet_find_atoms", blobnet)
    return labels, calls


def call(server, tool, **arguments):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(go())


def test_the_saving_tools_are_offered_and_marked_as_writing(tiled):
    mcp, _ = create_server(tiled_uri=tiled[0].uri)

    async def listed():
        async with Client(mcp) as client:
            return {tool.name: tool for tool in await client.list_tools()}

    tools = asyncio.run(listed())
    for name in ("segment_sam_and_save", "find_atoms_blobnet_and_save"):
        assert name in tools and not (tools[name].annotations and tools[name].annotations.readOnlyHint)


def test_a_segmentation_is_saved_beside_its_source_with_what_made_it(tiled, models):
    server, client = tiled
    labels, calls = models
    mcp, _ = create_server(tiled_uri=server.uri)
    result = call(mcp, "segment_sam_and_save", dataset="run_7#haadf", model="tiny", max_areas=1)
    assert [block.type for block in result.content] == ["text", "image"]  # the picture still reaches the chat
    reply = json.loads(result.content[0].text)
    assert reply["total"] == 2 and reply["returned"] == 1 and len(reply["areas"]) == 1 and "labels_base64" not in reply
    assert calls[0][2] == "tiny"
    saved = client[reply["saved"]]
    assert reply["saved"].startswith("run_7_haadf_sam_")
    meta = saved.metadata
    assert meta["kind"] == "segmentation" and meta["source"] == "run_7#haadf" and meta["parents"] == ["run_7"]
    assert meta["parameters"]["model"] == "tiny" and meta["sample_id"] == "S1" and meta["sample_load_id"] == "L3"
    np.testing.assert_array_equal(saved["labels"].read(), labels)
    assert len(saved["areas"].read()) == 2  # every region, not only the one the reply listed
    assert saved["marked"].read().shape[-1] == 3


def test_atoms_are_saved_with_every_position(tiled, models):
    server, client = tiled
    mcp, _ = create_server(tiled_uri=server.uri)
    result = call(mcp, "find_atoms_blobnet_and_save", dataset="run_7#haadf", max_atoms=2, include_image=False)
    reply = json.loads(result.content[0].text)
    assert [block.type for block in result.content] == ["text"] and reply["returned"] == 2
    saved = client[reply["saved"]]
    assert saved.metadata["kind"] == "atom_positions" and saved.metadata["parameters"]["threshold_rel"] == 0.35
    atoms = saved["atoms"].read()
    assert len(atoms) == 3 and list(atoms["x"]) == [5.0, 20.0, 40.0]
    assert "marked" in saved  # kept even when the reply leaves the picture out


def test_without_tiled_nothing_runs_and_the_reply_says_why(models):
    mcp, _ = create_server(tiled_uri=None)
    reply = json.loads(call(mcp, "segment_sam_and_save", dataset="run_7#haadf").content[0].text)
    assert reply["error"]["code"] == "no_tiled" and "segment_sam" in reply["error"]["message"]
    assert models[1] == []


def test_a_model_that_is_missing_is_refused_and_nothing_is_written(tiled, monkeypatch):
    server, client = tiled

    def missing(*args):
        raise image_learned.MissingModel("set PYTEMLIB_BLOBNET_CHECKPOINT")

    monkeypatch.setattr(image_learned, "blobnet_find_atoms", missing)
    before = set(client)
    mcp, _ = create_server(tiled_uri=server.uri)
    reply = json.loads(call(mcp, "find_atoms_blobnet_and_save", dataset="run_7#haadf").content[0].text)
    assert reply["error"]["code"] == "model_unavailable" and set(client) == before
