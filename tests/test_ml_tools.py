"""Optional ML tools: Bayesian optimisation and SAM2 segmentation.

Skipped when torch / botorch / SAM2 are not installed. Every test has a known answer.
"""
import importlib.util
import unittest

import numpy as np

has_bo = all(importlib.util.find_spec(name) for name in ('torch', 'gpytorch', 'botorch'))
has_sam2 = has_bo and importlib.util.find_spec('sam2') is not None


@unittest.skipUnless(has_bo, 'pyTEMlib[ml] not installed')
class TestBayesianOptimization(unittest.TestCase):
    def setUp(self):
        from pyTEMlib.ml_tools import bayesian_optimization
        self.bo = bayesian_optimization

    def test_finds_a_two_parameter_optimum(self):
        target = np.array([3.0, -2.0])

        def score(point):
            return float(np.exp(-np.sum((np.asarray(point) - target) ** 2) / 8))

        points = [[0, 0], [2, 0], [0, 2], [-2, 0], [0, -2]]
        scores = [score(point) for point in points]
        for _ in range(12):
            suggestion = self.bo.suggest_setting(points, scores, [[-10, 10], [-10, 10]])
            self.assertTrue(all(-10 <= value <= 10 for value in suggestion['next_point']))
            points.append(suggestion['next_point'])
            scores.append(score(suggestion['next_point']))
        np.testing.assert_allclose(points[int(np.argmax(scores))], target, atol=0.3)

    def test_minimisation_within_bounds(self):
        points = [[1.0], [4.0], [7.0]]
        suggestion = self.bo.suggest_setting(points, [(p[0] - 5) ** 2 for p in points], [[0, 6]], maximize=False)
        self.assertTrue(0 <= suggestion['next_point'][0] <= 6)
        self.assertEqual(suggestion['best_measured_point'], [4.0])

    def test_pareto_set(self):
        result = self.bo.suggest_pareto_setting([[0.0], [1.0], [2.0], [3.0], [1.5]],
                                                [[0, 3], [1, 2], [2, 1], [3, 0], [0.5, 0.5]], [[0, 3]])
        self.assertEqual(result['pareto_indices'], [0, 1, 2, 3])
        self.assertTrue(0 <= result['next_point'][0] <= 3)

    def test_bad_inputs_are_refused(self):
        with self.assertRaises(self.bo.SuggestionError):
            self.bo.suggest_setting([[0.0]], [1.0], [[0, 1]])
        with self.assertRaises(self.bo.SuggestionError):
            self.bo.suggest_setting([[0.0], [1.0]], [1.0, 2.0], [[1, 0]])


@unittest.skipUnless(has_sam2, 'SAM2 not installed')
class TestSam2Helpers(unittest.TestCase):
    def setUp(self):
        from pyTEMlib.ml_tools import sam2_segmentation
        self.sam2 = sam2_segmentation

    def test_prepare_image_is_8bit_rgb(self):
        prepared = self.sam2.prepare_image(np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32))
        self.assertEqual(prepared.shape, (2, 2, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        np.testing.assert_array_equal(prepared[..., 0], prepared[..., 2])
        with self.assertRaises(ValueError):
            self.sam2.prepare_image(np.zeros((2, 3, 4, 5)))

    def test_smaller_masks_win_overlaps(self):
        large = np.array([[True, True], [True, False]])
        small = np.array([[False, True], [False, False]])
        labels = self.sam2.label_areas([{'segmentation': large}, {'segmentation': small}], (2, 2))
        np.testing.assert_array_equal(labels, [[1, 2], [1, 0]])

    def test_area_statistics(self):
        mask = np.array([[True, True], [False, False]])
        statistics = self.sam2.area_statistics([{'segmentation': mask, 'area': 2, 'predicted_iou': 0.75}])
        self.assertEqual(statistics[0]['id'], 1)
        self.assertEqual(statistics[0]['confidence'], 0.75)
        self.assertEqual(statistics[0]['centroid_xy_px'], [0.5, 0.0])

    def test_tiny_model_finds_disks(self):
        rows, columns = np.indices((256, 256))
        image = np.full((256, 256), 30.0)
        disks = [(60, 60, 20), (150, 100, 30), (200, 200, 15), (80, 190, 25)]
        for row, column, radius in disks:
            image[(rows - row) ** 2 + (columns - column) ** 2 < radius ** 2] = 230.0
        labels, statistics = self.sam2.segment(image, 'facebook/sam2-hiera-tiny', 'cpu', points_per_side=16,
                                               min_area_px=50)
        for row, column, radius in disks:
            label = labels[row, column]
            self.assertGreater(label, 0)
            area = int((labels == label).sum())
            self.assertAlmostEqual(area, np.pi * radius ** 2, delta=0.15 * np.pi * radius ** 2)
        self.assertEqual(labels[5, 5], 0)


if __name__ == '__main__':
    unittest.main()
