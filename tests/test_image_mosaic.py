"""Tile maps: placement, refinement, stitching and label merging against a known scene."""
import os
import tempfile
import unittest

import numpy as np
from scipy import ndimage

from pyTEMlib.image_tools import image_mosaic, image_segmentation


def _scene():
    rng = np.random.default_rng(5)
    rows, columns = np.indices((300, 420))
    scene = 0.1 + 0.05 * ndimage.gaussian_filter(rng.random((300, 420)), 2)
    centres = [(60, 70), (150, 205), (150, 140), (240, 330), (80, 300), (220, 100)]
    for row, column in centres:
        scene[(rows - row) ** 2 + (columns - column) ** 2 < 18 ** 2] += 1.0
    return scene, centres


def _tiles(scene, tile=160, step=120):
    for top in range(0, scene.shape[0] - tile + 1, step - 40 if step > 40 else step):
        for left in range(0, scene.shape[1] - tile + 1, step):
            yield top, left, scene[top:top + tile, left:left + tile]


class TestTileMap(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.map_dir = os.path.join(self.directory, 'map')
        self.scene, self.centres = _scene()

    def build(self, stage_error_px=0.0, refine=True, with_labels=True):
        image_mosaic.create_map(self.map_dir, pixel_size=2e-9)
        rng = np.random.default_rng(1)
        results = []
        for top, left, tile in _tiles(self.scene):
            error = rng.uniform(-stage_error_px, stage_error_px, 2) if results else np.zeros(2)
            centre_row, centre_column = top + tile.shape[0] / 2 + error[0], left + tile.shape[1] / 2 + error[1]
            position = (centre_column * 2e-9, centre_row * 2e-9)
            labels = image_segmentation.segment_particles(tile)[0] if with_labels else None
            results.append(image_mosaic.add_tile(self.map_dir, tile, position, labels=labels, refine=refine,
                                                 min_overlap_px=10))
        return results

    def test_exact_positions_reproduce_the_scene(self):
        self.build(refine=False, with_labels=False)
        stitched = image_mosaic.map_image(self.map_dir)
        covered = np.isfinite(stitched)
        self.assertGreater(covered.mean(), 0.95)
        rows, columns = stitched.shape
        np.testing.assert_allclose(stitched[covered], self.scene[:rows, :columns][covered], atol=1e-12)

    def test_refinement_corrects_stage_errors(self):
        results = self.build(stage_error_px=6.0, refine=True, with_labels=False)
        stitched = image_mosaic.map_image(self.map_dir)
        covered = np.isfinite(stitched)
        rows, columns = stitched.shape
        self.assertLess(np.abs(stitched[covered] - self.scene[:rows, :columns][covered]).mean(), 0.01)
        self.assertTrue(any(np.hypot(*result['refinement_px']) > 1 for result in results[1:]))

    def test_without_refinement_stage_errors_blur_the_map(self):
        self.build(stage_error_px=6.0, refine=False, with_labels=False)
        stitched = image_mosaic.map_image(self.map_dir)
        covered = np.isfinite(stitched)
        rows, columns = stitched.shape
        self.assertGreater(np.abs(stitched[covered] - self.scene[:rows, :columns][covered]).mean(), 0.02)

    def test_particles_cut_by_tile_edges_become_one_region(self):
        results = self.build(stage_error_px=4.0, refine=True, with_labels=True)
        self.assertGreater(sum(result['regions_joined'] for result in results), 0)
        summary = image_mosaic.map_summary(self.map_dir)
        self.assertEqual(len(summary['regions']), len(self.centres))
        stitched_rows, stitched_columns = summary['map_shape']
        inside = [(row, column) for row, column in self.centres
                  if row + 18 < stitched_rows and column + 18 < stitched_columns]
        whole = [region for region in summary['regions'] if region['area_px'] > 600]
        self.assertEqual(len(whole), len(inside))
        for region in whole:
            self.assertAlmostEqual(region['equivalent_radius'] / 2e-9, 18, delta=2)

    def test_labels_are_optional_and_validated(self):
        image_mosaic.create_map(self.map_dir, pixel_size=1.0)
        with self.assertRaises(ValueError):
            image_mosaic.add_tile(self.map_dir, np.zeros((10, 10)), (0, 0), labels=np.zeros((5, 5)))
        result = image_mosaic.add_tile(self.map_dir, np.ones((10, 10)), (0, 0))
        self.assertEqual(result['regions_added'], 0)
        with self.assertRaises(FileExistsError):
            image_mosaic.create_map(self.map_dir, pixel_size=1.0)

    def test_tiles_placed_left_and_above_grow_the_map(self):
        image_mosaic.create_map(self.map_dir, pixel_size=1.0)
        image_mosaic.add_tile(self.map_dir, np.ones((20, 20)), (100, 100), refine=False)
        result = image_mosaic.add_tile(self.map_dir, 2 * np.ones((20, 20)), (70, 60), refine=False)
        self.assertEqual(result['map_shape'], [60, 50])
        stitched = image_mosaic.map_image(self.map_dir)
        self.assertEqual(stitched[0, 0], 2.0)
        self.assertEqual(stitched[-1, -1], 1.0)

    def test_axis_signs_mirror_placement(self):
        image_mosaic.create_map(self.map_dir, pixel_size=1.0, axis_signs=(-1.0, 1.0))
        image_mosaic.add_tile(self.map_dir, np.ones((20, 20)), (0, 0), refine=False)
        result = image_mosaic.add_tile(self.map_dir, 2 * np.ones((20, 20)), (30, 0), refine=False)
        self.assertEqual(result['top_left_px'][1], 0)


class TestMergeLabels(unittest.TestCase):
    def test_bridging_region_joins_two_existing_regions(self):
        map_labels = np.zeros((10, 30), dtype=np.int64)
        map_labels[2:8, 0:10] = 1
        map_labels[2:8, 20:30] = 2
        tile = np.zeros((10, 22), dtype=np.int64)
        tile[2:8, :] = 1
        next_label, added, joined = image_mosaic.merge_labels(map_labels, tile, (0, 4), 3, min_overlap_px=5)
        self.assertEqual((added, joined), (1, 1))
        self.assertEqual(len(np.unique(map_labels[map_labels > 0])), 1)

    def test_small_overlap_does_not_join(self):
        map_labels = np.zeros((10, 10), dtype=np.int64)
        map_labels[0:3, 0:3] = 1
        tile = np.zeros((10, 10), dtype=np.int64)
        tile[2:6, 2:6] = 1
        image_mosaic.merge_labels(map_labels, tile, (0, 0), 2, min_overlap_px=5)
        self.assertEqual(sorted(np.unique(map_labels[map_labels > 0])), [1, 2])


if __name__ == '__main__':
    unittest.main()
