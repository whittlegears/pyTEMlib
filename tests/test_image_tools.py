# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np
import ase
import ase.build
import sys
import os
sys.path.insert(0, "../")

import pyTEMlib.image_tools as it
import pyTEMlib.file_tools as ft

class TestUtilityFunctions(unittest.TestCase):

    def test_get_wavelength(self):
        wavelength =  it.get_wavelength(200000)
        self.assertTrue(np.allclose(wavelength, 0.0025079340450548005, atol=1e-3))
    """
    def test_read_image_info(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.dm3')
        dataset = ft.open_file(file_name)
        metadata = it.read_image_info(dataset)
    """


# ---------------------------------------------------------------- find_blobs

def test_find_blobs_gives_the_distinct_blobs_strongest_first():
    from .conftest import gaussian_spots, image_dataset

    found = it.find_blobs(image_dataset(gaussian_spots(), 1e-9), n=5)  # dimensions in nm
    assert found.shape[1] == 4
    assert [(round(r), round(c)) for r, c in found[:2, :2]] == [(60, 80), (180, 150)]
    assert found[0, 3] > found[1, 3] >= 3.0
    assert 8.0 < found[0, 2] < 30.0  # diameter in nm
    assert all(not (abs(r - 128) < 5 and abs(c - 230) < 5) for r, c in found[:, :2])  # 2 x the noise: not distinct
    assert len(it.find_blobs(image_dataset(gaussian_spots(spots=()), 1e-9))) == 0
    assert len(it.find_blobs(image_dataset(gaussian_spots(), 1e-9), n=1)) == 1
    assert len(it.find_blobs(image_dataset(gaussian_spots(), 1e-9), min_size=1e9)) == 0  # wider than the image


# ---------------------------------------------------------------- pixel_to_stage, stage_to_pixel

def test_an_image_centre_is_its_stage_position_and_pixels_step_along_x_and_y():
    from .conftest import image_dataset

    dataset = image_dataset(np.zeros((101, 201)), 1e-9, (1e-6, -2e-6))
    assert np.allclose(it.get_pixel_size_m(dataset), [1e-9, 1e-9])
    assert np.allclose(it.pixel_to_stage(dataset, 50, 100), (1e-6, -2e-6))
    assert np.allclose(it.pixel_to_stage(dataset, 50, 101), (1e-6 + 1e-9, -2e-6))
    assert np.allclose(it.pixel_to_stage(dataset, 51, 100), (1e-6, -2e-6 + 1e-9))
    for row, col in ((0, 0), (12.5, 170.25), (100, 200)):
        assert np.allclose(it.stage_to_pixel(dataset, *it.pixel_to_stage(dataset, row, col)), (row, col))
    dataset.metadata['experiment'] = {'stage_position': {'X': 1e-6, 'Y': -2e-6, 'Z': 0.0}, 'image_shift': [3e-8, -4e-8]}
    assert np.allclose(it.get_image_centre_m(dataset), (1e-6 + 3e-8, -2e-6 - 4e-8))


def test_an_image_without_a_stage_position_or_calibration_is_refused():
    import pytest
    import sidpy

    from .conftest import image_dataset

    with pytest.raises(ValueError, match='stage_position'):
        it.pixel_to_stage(image_dataset(np.zeros((8, 8))), 0, 0)
    uncalibrated = sidpy.Dataset.from_array(np.zeros((8, 8)))
    uncalibrated.data_type = 'IMAGE'
    with pytest.raises(ValueError):
        it.get_pixel_size_m(uncalibrated)


def test_asyncroscopy_metadata_becomes_experiment_keys():
    experiment = it.experiment_from_asyncroscopy({'stage_reported_m': '[1e-6, 2e-6, 0, 0, 0]', 'image_shift_m': [1e-8, 0.0],
                                                  'stage_commanded_m': '[9, 9, 0, 0, 0]'})
    assert experiment == {'stage_position': [1e-6, 2e-6, 0.0, 0.0, 0.0], 'image_shift': [1e-8, 0.0]}
    assert it.experiment_from_asyncroscopy({'stage_commanded_m': [3e-6, 0.0]}) == {'stage_position': [3e-6, 0.0]}
    assert it.experiment_from_asyncroscopy({'stage_reported_m': 'not json'}) == {}


if __name__ == '__main__':
    unittest.main()