"""to_stage / to_pixel: an image's pixels placed on the stage from what its acquisition recorded."""

import pytest

from pyTEMlib.mcp_server.frames import FrameError, frame_of, pixel_of, register, stage_of


class Node:
    def __init__(self, metadata=None, shape=None, children=None):
        self.metadata = metadata or {}
        self._shape = shape
        self._children = children or {}
        self.structure_family = "container" if children is not None else "array"

    def structure(self):
        return type("S", (), {"shape": self._shape})()

    def __contains__(self, name):
        return name in self._children

    def __getitem__(self, name):
        return self._children[name]

    def __iter__(self):
        return iter(self._children)


class Store:
    def __init__(self, nodes, sample=None):
        self.client = nodes
        self.sample = sample or {}

    def map_metadata(self, key):
        raise KeyError(key)

    def sample_of(self, key):
        return dict(self.sample)


IMAGE = {"pixel_size_m": 1e-9, "stage_reported_m": "[1e-6, -2e-6, 0.0, 0.0, 0.0]", "axes": "rows=+y, columns=+x", "scan_region": "[0.0, 0.0, 1.0, 1.0]"}


def store_with(metadata, shape=(101, 201)):
    return Store({"img.h5": Node(children={"image": Node(children={"HAADF": Node(metadata, shape=shape)})})})


def test_an_image_centre_is_where_the_stage_was_and_pixels_step_along_x_and_y():
    frame = frame_of(store_with(IMAGE), "img.h5")
    assert stage_of(frame, 50, 100) == pytest.approx((1e-6, -2e-6))
    assert stage_of(frame, 50, 101) == pytest.approx((1e-6 + 1e-9, -2e-6))
    assert stage_of(frame, 51, 100) == pytest.approx((1e-6, -2e-6 + 1e-9))
    assert stage_of(frame, 0, 0) == pytest.approx((1e-6 - 100e-9, -2e-6 - 50e-9))
    for row, col in ((0, 0), (12.5, 170.25), (100, 200)):
        assert pixel_of(frame, *stage_of(frame, row, col)) == pytest.approx((row, col))


@pytest.mark.parametrize("change, reason", [
    ({"stage_reported_m": "[0, 0, 0, 10.0, 0]", "stage_commanded_m": None}, "tilt"),
    ({"scan_region": "[0.25, 0.25, 0.75, 0.75]"}, "part of the scan field"),
    ({"axes": "rows=-y, columns=+x"}, "axes"),
    ({"pixel_size_m": None}, "does not record"),
    ({"stage_reported_m": "[NaN, 0, 0, 0, 0]", "stage_commanded_m": None}, "not a number"),
    ({"stage_reported_m": "[0, 0, 0, NaN, 0]", "stage_commanded_m": None}, "not a number"),
])
def test_images_it_cannot_place_are_refused(change, reason):
    metadata = {**IMAGE, **change}
    with pytest.raises(FrameError, match=reason):
        frame_of(store_with({k: v for k, v in metadata.items() if v is not None}), "img.h5")


def test_the_commanded_position_is_used_when_none_was_reported():
    metadata = {**IMAGE, "stage_commanded_m": "[3e-6, 0, 0, 0, 0]"}
    del metadata["stage_reported_m"]
    assert stage_of(frame_of(store_with(metadata), "img.h5"), 50, 100) == pytest.approx((3e-6, 0))


def test_an_unknown_key_raises_key_error():
    with pytest.raises(KeyError):
        frame_of(Store({}), "nope")


def test_to_stage_names_the_sample_load_so_a_move_can_be_checked_first():
    import asyncio

    from fastmcp import FastMCP

    def reply(store):
        mcp = FastMCP("t")
        register(mcp, store)
        return asyncio.run(mcp.call_tool("to_stage", {"dataset": "img.h5", "row": 50, "col": 100})).structured_content

    loaded = Store({"img.h5": Node(children={"image": Node(children={"HAADF": Node(IMAGE, shape=(101, 201))})})}, {"sample_id": "grid 12", "sample_load_id": "abc"})
    placed = reply(loaded)
    assert placed["sample_load_id"] == "abc" and placed["sample_id"] == "grid 12" and "get_sample" in placed["note"]
    unknown = reply(store_with(IMAGE))
    assert "sample_load_id" not in unknown and "does not record which sample load" in unknown["note"]


def test_the_sample_of_an_array_inside_a_file_is_read_from_the_file():
    from pyTEMlib.mcp_server.map_store import MapStore

    class Meta:
        def __init__(self, metadata):
            self.metadata = metadata

    class Fake(MapStore):
        client = None

    store = Fake.__new__(Fake)
    store.local_roots = []
    store.client = {"run.h5": Meta({"sample_id": "g", "sample_load_id": "L1"}), "run.h5/image": Meta({}), "run.h5/image/HAADF": Meta({"pixel_size_m": 1e-9})}
    assert store.sample_of("run.h5/image/HAADF") == {"sample_id": "g", "sample_load_id": "L1"}
    assert store.sample_of("other") == {}
