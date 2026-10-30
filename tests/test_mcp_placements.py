"""place_image / list_placements: a finer image placed on a coarser one and the record kept."""

import json

import numpy as np
import pytest

from pyTEMlib.mcp_server.frames import FrameError
from pyTEMlib.mcp_server.placement_tools import PlacementLog, place

from .test_register_child import STEP, fine_scene, parent_of

AXES = "rows=+y, columns=+x"


class Array:
    def __init__(self, data, metadata):
        self.data, self.metadata, self.structure_family = data, metadata, "array"

    def structure(self):
        return type("S", (), {"shape": self.data.shape})()

    def read(self):
        return self.data

    def __getitem__(self, index):
        return self.data[index]


class Store:
    def __init__(self, nodes):
        self.client = nodes

    def map_metadata(self, key):
        raise KeyError(key)

    def sample_of(self, key):
        return {"sample_id": "grid-7"}


def image(data, pixel_m, centre_xy):
    meta = {"pixel_size_m": pixel_m, "stage_reported_m": json.dumps([*centre_xy, 0.0, 0.0, 0.0]), "axes": AXES, "scan_region": "[0.0, 0.0, 1.0, 1.0]"}
    return Array(np.asarray(data, dtype=np.float64), meta)


@pytest.fixture
def setup(tmp_path):
    scene = fine_scene()  # fine pixels of 1 nm; the parent's are STEP nm
    parent = parent_of(scene)
    centre_px = np.array([1100.4, 900.7])  # (row, col) of the child's centre in fine pixels
    rr, cc = np.meshgrid(centre_px[0] + np.arange(256) - 127.5, centre_px[1] + np.arange(256) - 127.5, indexing="ij")
    from scipy import ndimage

    child = ndimage.map_coordinates(scene, [rr, cc], order=3)
    parent_centre_xy = ((parent.shape[1] - 1) / 2 * STEP * 1e-9, (parent.shape[0] - 1) / 2 * STEP * 1e-9)
    # Where the child's centre really is, in metres: the parent's pixel (r, c) centre is at (c, r) * STEP nm here.
    true_xy = (((centre_px[1] + 0.5) / STEP - 0.5) * STEP * 1e-9, ((centre_px[0] + 0.5) / STEP - 0.5) * STEP * 1e-9)
    reported = (true_xy[0] + 30e-9, true_xy[1] - 25e-9)  # the stage reports it 30 and 25 nm off
    nodes = {"survey.h5": image(parent, STEP * 1e-9, parent_centre_xy), "close.h5": image(child, 1e-9, reported), "blank.h5": image(np.full((256, 256), 3.0), 1e-9, reported)}
    return Store(nodes), PlacementLog(tmp_path / "placements.jsonl"), np.array(true_xy)


def test_a_stage_aimed_image_is_placed_where_it_really_is_and_logged(setup):
    store, log, true_xy = setup
    record = place(store, log, "close.h5", "survey.h5", window_m=80e-9)
    assert record["status"] == "placed", record["reason"]
    assert np.hypot(*(np.array(record["centre_xy_m"]) - true_xy)) < 0.3 * STEP * 1e-9
    assert record["sample_id"] == "grid-7" and record["child"] == "close.h5" and record["parent"] == "survey.h5"
    assert log.read()[-1]["id"] == record["id"]


def test_an_image_the_parent_cannot_place_is_logged_unplaced_with_the_reason(setup):
    store, log, _ = setup
    record = place(store, log, "blank.h5", "survey.h5", window_m=80e-9)
    assert record["status"] == "unplaced" and "blank" in record["reason"] and "centre_xy_m" not in record
    assert [r["status"] for r in log.read()] == ["unplaced"]


def test_a_window_too_small_for_the_stage_error_does_not_place_it_somewhere_else(setup):
    store, log, true_xy = setup
    record = place(store, log, "close.h5", "survey.h5", window_m=8e-9)  # one parent pixel; the stage is 5 pixels off
    assert record["status"] == "unplaced" or np.hypot(*(np.array(record["centre_xy_m"]) - true_xy)) < 0.3 * STEP * 1e-9


def test_a_coarser_image_cannot_be_placed_on_a_finer_one(setup):
    store, log, _ = setup
    with pytest.raises(FrameError, match="not finer"):
        place(store, log, "survey.h5", "close.h5")
