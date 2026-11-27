"""register_to_parent: placing a finer image on the coarser image it was taken inside (one step of a multiscale map)."""

import numpy as np
import pytest
from scipy import ndimage

from pyTEMlib.image_tools import register_to_parent
from pyTEMlib.image_tools.image_registration import ChildPlacement, simulate_parent_view

from .conftest import PIXEL, STEP, fine_scene, image_dataset, parent_of


@pytest.mark.parametrize("centre", [(1000.3, 1210.7), (640.0, 400.5), (1500.6, 1800.2)])
def test_a_child_lands_where_it_was_taken_to_a_fraction_of_a_parent_pixel(centre):
    scene = fine_scene()
    parent = parent_of(scene)
    child = ndimage.map_coordinates(scene, np.meshgrid(centre[0] + np.arange(256) - 127.5, centre[1] + np.arange(256) - 127.5, indexing="ij"), order=3)
    true = (np.array(centre) + 0.5) / STEP - 0.5  # the child's centre in parent pixel coordinates
    placed = register_to_parent(image_dataset(child, PIXEL), image_dataset(parent, STEP * PIXEL), predicted_px=true + [6.0, -5.0], window_px=12)
    assert placed.status == "placed", placed.reason
    assert np.hypot(*(np.array(placed.translation_px) - true)) < 0.3
    assert placed.score > 0.8 and placed.ambiguity < 0.8 and placed.scale == pytest.approx(1 / STEP)


def test_a_blank_child_is_unplaced_and_says_why():
    parent = parent_of(fine_scene())
    placed = register_to_parent(image_dataset(np.full((256, 256), 3.0), PIXEL), image_dataset(parent, STEP * PIXEL), predicted_px=(120, 120), window_px=10)
    assert placed.status == "unplaced" and "blank" in placed.reason and placed.translation_px is None


def test_a_child_on_a_repeating_pattern_is_a_guess_and_is_unplaced():
    rows, cols = np.mgrid[0:2400, 0:2400]
    lattice = np.cos(2 * np.pi * rows / 48.0) + np.cos(2 * np.pi * cols / 48.0)  # a repeat of 6 parent pixels
    parent = parent_of(lattice)
    child = lattice[872:1128, 872:1128]
    placed = register_to_parent(image_dataset(child, PIXEL), image_dataset(parent, STEP * PIXEL), predicted_px=(125, 125), window_px=15)
    assert placed.status == "unplaced" and "almost as well" in placed.reason


def test_a_repeat_just_outside_a_narrow_window_still_makes_the_match_a_guess():
    rows, cols = np.mgrid[0:2400, 0:2400]
    lattice = np.cos(2 * np.pi * rows / 48.0) + np.cos(2 * np.pi * cols / 48.0)  # a repeat of 6 parent pixels
    parent = parent_of(lattice)
    child = lattice[872:1128, 872:1128]
    placed = register_to_parent(image_dataset(child, PIXEL), image_dataset(parent, STEP * PIXEL), predicted_px=(125, 125), window_px=4)
    assert placed.status == "unplaced" and "almost as well" in placed.reason


def test_a_child_outside_the_search_window_is_not_matched_to_something_else():
    scene = fine_scene()
    parent = parent_of(scene)
    child = scene[1872:2128, 1872:2128]
    true = np.array([2000.0, 2000.0]) / STEP
    placed = register_to_parent(image_dataset(child, PIXEL), image_dataset(parent, STEP * PIXEL), predicted_px=true - 30, window_px=8)
    assert placed.status == "unplaced"


def test_the_record_names_its_method_and_inputs_and_leaves_provenance_to_the_caller():
    scene = fine_scene()
    child = scene[900:1156, 900:1156]
    placed = register_to_parent(image_dataset(child, PIXEL), image_dataset(parent_of(scene), STEP * PIXEL), predicted_px=(128, 128), window_px=8)
    record = placed.to_dict()
    assert isinstance(placed, ChildPlacement) and "parent" not in record
    assert len(record["inputs_sha256"]) == 64 and record["method"] == "simulated-parent-ncc"


def test_pixel_sizes_come_from_the_datasets_in_any_length_unit():
    scene = fine_scene()
    parent = image_dataset(parent_of(scene), STEP * PIXEL)
    for axis in range(2):  # the same parent, its dimensions in Angstrom
        dim = parent.get_dimension_by_number(axis)[0]
        parent.set_dimension(axis, type(dim)(np.asarray(dim.values) * 10.0, name=dim.name, units="Å", quantity="Length", dimension_type="spatial"))
    centre = (1000.3, 1210.7)
    child = ndimage.map_coordinates(scene, np.meshgrid(centre[0] + np.arange(256) - 127.5, centre[1] + np.arange(256) - 127.5, indexing="ij"), order=3)
    placed = register_to_parent(image_dataset(child, PIXEL), parent, predicted_px=(np.array(centre) + 0.5) / STEP - 0.5, window_px=8)
    assert placed.status == "placed" and placed.scale == pytest.approx(1 / STEP)
    with pytest.raises(TypeError):
        register_to_parent(child, parent, predicted_px=(0, 0), window_px=8)


def test_the_parent_view_has_the_parents_pixel_count():
    view = simulate_parent_view(np.random.default_rng(0).normal(size=(256, 256)), 1 / 16)
    assert view.shape == (16, 16)
    with pytest.raises(ValueError):
        simulate_parent_view(np.zeros((8, 8)), 2.0)
