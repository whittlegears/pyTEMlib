"""Placing a finer image on its coarser parent (docs/designs/multiscale_maps.md in asyncroscopy, P3)."""

import numpy as np
import pytest
from scipy import ndimage

from pyTEMlib.image_tools.register import register_child, simulate_parent_view

STEP = 8  # parent pixels are 8 child pixels wide


def fine_scene(size=2400, seed=0):
    rng = np.random.default_rng(seed)
    image = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 6)
    blobs = np.zeros((size, size))
    blobs[rng.integers(0, size, 300), rng.integers(0, size, 300)] = 1.0
    return image / image.std() + 8 * ndimage.gaussian_filter(blobs, 10) / ndimage.gaussian_filter(blobs, 10).max()


def parent_of(scene):
    """What a coarse scan sees: the scene smoothed, then sampled at every STEP-th pixel centre."""
    smooth = ndimage.gaussian_filter(scene, 0.4 * STEP)
    centres = np.arange(STEP // 2, scene.shape[0], STEP) - 0.5
    rr, cc = np.meshgrid(centres, centres, indexing="ij")
    return ndimage.map_coordinates(smooth, [rr, cc], order=1)


@pytest.mark.parametrize("centre", [(1000.3, 1210.7), (640.0, 400.5), (1500.6, 1800.2)])
def test_a_child_lands_where_it_was_taken_to_a_fraction_of_a_parent_pixel(centre):
    scene = fine_scene()
    parent = parent_of(scene)
    child = ndimage.map_coordinates(scene, np.meshgrid(centre[0] + np.arange(256) - 127.5, centre[1] + np.arange(256) - 127.5, indexing="ij"), order=3)
    true = (np.array(centre) + 0.5) / STEP - 0.5  # the child's centre in parent pixel coordinates
    placed = register_child(child, 1.0, parent, float(STEP), predicted_px=true + [6.0, -5.0], window_px=12)
    assert placed.status == "placed", placed.reason
    assert np.hypot(*(np.array(placed.translation_px) - true)) < 0.3
    assert placed.score > 0.8 and placed.ambiguity < 0.8 and placed.scale == pytest.approx(1 / STEP)


def test_a_blank_child_is_unplaced_and_says_why():
    parent = parent_of(fine_scene())
    placed = register_child(np.full((256, 256), 3.0), 1.0, parent, float(STEP), predicted_px=(120, 120), window_px=10)
    assert placed.status == "unplaced" and "blank" in placed.reason and placed.translation_px is None


def test_a_child_on_a_repeating_pattern_is_a_guess_and_is_unplaced():
    rows, cols = np.mgrid[0:2400, 0:2400]
    lattice = np.cos(2 * np.pi * rows / 48.0) + np.cos(2 * np.pi * cols / 48.0)  # a repeat of 6 parent pixels
    parent = parent_of(lattice)
    child = lattice[872:1128, 872:1128]
    placed = register_child(child, 1.0, parent, float(STEP), predicted_px=(125, 125), window_px=15)
    assert placed.status == "unplaced" and "almost as well" in placed.reason


def test_a_child_outside_the_search_window_is_not_matched_to_something_else():
    scene = fine_scene()
    parent = parent_of(scene)
    child = scene[1872:2128, 1872:2128]
    true = np.array([2000.0, 2000.0]) / STEP
    placed = register_child(child, 1.0, parent, float(STEP), predicted_px=true - 30, window_px=8)
    assert placed.status == "unplaced"


def test_the_record_names_its_parent_tile_and_inputs():
    scene = fine_scene()
    child = scene[900:1156, 900:1156]
    placed = register_child(child, 1.0, parent_of(scene), float(STEP), predicted_px=(128, 128), window_px=8, parent_tile={"run": "run_1", "index": 4, "map_version": "run_1_map_v"})
    record = placed.to_dict()
    assert record["parent"]["index"] == 4 and len(record["inputs_sha256"]) == 64 and record["method"] == "simulated-parent-ncc"


def test_the_parent_view_has_the_parents_pixel_count():
    view = simulate_parent_view(np.random.default_rng(0).normal(size=(256, 256)), 1 / 16)
    assert view.shape == (16, 16)
    with pytest.raises(ValueError):
        simulate_parent_view(np.zeros((8, 8)), 2.0)
