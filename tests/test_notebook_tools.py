"""Tests of the analysis tools extracted from the asyncroscopy notebooks.

Every test builds synthetic data with a known answer and checks that the
tool recovers it.
"""
import unittest

import numpy as np
from scipy import ndimage

from pyTEMlib import optimization_tools
from pyTEMlib import utilities
from pyTEMlib.diffraction_tools import four_d_stem, ring_analysis
from pyTEMlib.image_tools import (image_alignment, image_metrics, image_segmentation,
                                  sparse_reconstruction)


def _lattice_image(size=128, period=8.0):
    rows, columns = np.indices((size, size), dtype=float)
    return 2 + np.cos(2 * np.pi * rows / period) * np.cos(2 * np.pi * columns / period)


def _disks(shape, centers, radii):
    rows, columns = np.indices(shape, dtype=float)
    image = np.zeros(shape)
    for (row, column), radius in zip(centers, radii):
        image[np.hypot(rows - row, columns - column) <= radius] = 1.0
    return image


class TestImageMetrics(unittest.TestCase):
    def test_blur_lowers_every_focus_metric(self):
        sharp = 1 + ndimage.gaussian_filter(np.random.default_rng(0).random((128, 128)), 0.7)
        blurred = ndimage.gaussian_filter(sharp, 2.5)
        sharp_metrics = image_metrics.image_quality_metrics(sharp)
        blurred_metrics = image_metrics.image_quality_metrics(blurred)
        for name in ('contrast', 'high_frequency_power', 'edge_sharpness'):
            self.assertGreater(sharp_metrics[name], blurred_metrics[name], name)

    def test_low_dose_noise_does_not_reverse_the_ranking(self):
        rows, columns = np.indices((128, 128), dtype=float)
        lattice = np.cos(2 * np.pi * rows / 5) * np.cos(2 * np.pi * columns / 5) + 1.2
        rng = np.random.default_rng(3)
        for sigma_sharp, sigma_blurred in ((0.6, 1.2), (0.8, 2.0)):
            scores = []
            for sigma in (sigma_sharp, sigma_blurred):
                image = ndimage.gaussian_filter(lattice, sigma)
                counts = rng.poisson(image / image.mean() * 1.3).astype(float)
                scores.append(image_metrics.high_frequency_power(counts + rng.normal(0, 0.5, counts.shape) + 1))
            self.assertGreater(scores[0], scores[1])

    def test_stack_returns_one_value_per_frame(self):
        stack = np.stack([_lattice_image(), ndimage.gaussian_filter(_lattice_image(), 2)])
        result = image_metrics.image_quality_metrics(stack, metrics=['contrast'])
        self.assertEqual(len(result['contrast']), 2)
        self.assertGreater(result['contrast'][0], result['contrast'][1])

    def test_entropy_bounds(self):
        rng = np.random.default_rng(0)
        self.assertGreater(image_metrics.image_entropy(rng.random((64, 64))), 0.9)
        self.assertEqual(image_metrics.image_entropy(np.ones((16, 16))), 0.0)

    def test_compare_images_psnr(self):
        reference = np.zeros((10, 10))
        reference[0, 0] = 1.0
        estimate = reference.copy()
        estimate[5, 5] = 0.1
        result = image_metrics.compare_images(reference, estimate)
        self.assertAlmostEqual(result['MSE'], 0.01 / 100)
        self.assertAlmostEqual(result['PSNR'], 40.0)

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):
            image_metrics.image_quality_metrics(_lattice_image(), metrics=['sharpnes'])


class TestSegmentation(unittest.TestCase):
    def test_otsu_separates_two_levels(self):
        image = np.r_[np.full(500, 10.0), np.full(500, 50.0)]
        threshold = image_segmentation.otsu_threshold(image)
        self.assertTrue(10 < threshold < 50)

    def test_particle_count_size_and_shape(self):
        image = _disks((200, 200), [(50, 50), (140, 120)], [12, 20])
        image = image + np.random.default_rng(1).normal(0, 0.05, image.shape)
        labels, _ = image_segmentation.segment_particles(image)
        stats = image_segmentation.particle_statistics(labels, pixel_size=0.5)
        self.assertEqual(len(stats), 2)
        self.assertAlmostEqual(stats[0]['equivalent_radius'], 20 * 0.5, delta=0.5)
        self.assertAlmostEqual(stats[1]['equivalent_radius'], 12 * 0.5, delta=0.5)
        self.assertAlmostEqual(stats[0]['com_y_px'], 140, delta=0.5)
        self.assertAlmostEqual(stats[0]['com_x_px'], 120, delta=0.5)
        self.assertLess(stats[0]['aspect_ratio'], 1.1)

    def test_ellipse_aspect_ratio_and_orientation(self):
        rows, columns = np.indices((200, 200), dtype=float)
        ellipse = ((rows - 100) / 15) ** 2 + ((columns - 100) / 45) ** 2 <= 1
        stats = image_segmentation.particle_statistics(ellipse.astype(int))
        self.assertAlmostEqual(stats[0]['aspect_ratio'], 3.0, delta=0.1)
        self.assertAlmostEqual(abs(stats[0]['orientation_deg']) % 180, 0.0, delta=1.0)

    def test_dark_features_with_polarity_below(self):
        image = 1.0 - _disks((100, 100), [(50, 50)], [10])
        labels, _ = image_segmentation.segment_particles(image, polarity='below')
        self.assertEqual(labels.max(), 1)

    def test_remove_small_regions_and_radial_path(self):
        mask = _disks((100, 100), [(20, 20), (60, 60)], [2, 15]).astype(bool)
        cleaned = image_segmentation.remove_small_regions(mask, 50)
        self.assertEqual(image_segmentation.mask_summary(cleaned)['n_regions'], 1)
        path = image_segmentation.radial_path(cleaned.astype(int), 1, [0, 1])
        self.assertAlmostEqual(np.linalg.norm(path[1] - path[0]), 15, delta=1)


class TestSparseReconstruction(unittest.TestCase):
    def setUp(self):
        rows, columns = np.indices((64, 64), dtype=float)
        self.image = 100 + 30 * np.sin(rows / 7) * np.cos(columns / 9)

    def test_mask_fraction(self):
        for method in ('random', 'grid'):
            mask = sparse_reconstruction.make_sampling_mask((64, 64), 0.2, method)
            self.assertAlmostEqual(mask.mean(), 0.2, delta=0.03)
        guided = sparse_reconstruction.make_sampling_mask((64, 64), 0.2, 'preview-guided', preview=self.image)
        self.assertEqual(guided.sum(), round(0.2 * 64 * 64))

    def test_reconstruction_beats_mean_and_keeps_units(self):
        mask = sparse_reconstruction.make_sampling_mask(self.image.shape, 0.2, seed=3)
        baseline = image_metrics.compare_images(self.image, np.full_like(self.image, self.image[mask].mean()))
        for method in ('nearest', 'dct', 'tv'):
            estimate = sparse_reconstruction.sparse_reconstruct(self.image * mask, mask, method)
            result = image_metrics.compare_images(self.image, estimate)
            self.assertLess(result['RMSE'], 0.5 * baseline['RMSE'], method)
            self.assertAlmostEqual(float(np.mean(estimate)), float(np.mean(self.image)), delta=3)

    def test_extra_axes_are_reconstructed_together(self):
        cube = np.stack([self.image, 2 * self.image, self.image[::-1]], axis=-1)
        mask = sparse_reconstruction.make_sampling_mask(self.image.shape, 0.3, seed=1)
        estimate = sparse_reconstruction.sparse_reconstruct(cube * mask[..., None], mask, 'dct')
        self.assertEqual(estimate.shape, cube.shape)
        self.assertLess(image_metrics.compare_images(cube, estimate)['RMSE'], 5)

    def test_uncertainty_guides_next_measurement(self):
        measured = np.zeros(self.image.shape, dtype=bool)
        measured[:10, :10] = True
        values = np.where(measured, self.image, 0)
        reconstruction, uncertainty = sparse_reconstruction.reconstruct_with_uncertainty(values, measured)
        self.assertTrue(np.all(uncertainty[measured] == 0))
        np.testing.assert_allclose(reconstruction[measured], self.image[measured])
        row, column = sparse_reconstruction.next_measurement_locations(uncertainty)[0]
        self.assertFalse(measured[row, column])
        _, distance_only = sparse_reconstruction.reconstruct_with_uncertainty(values, measured,
                                                                              disagreement_weight=0.0)
        self.assertEqual(sparse_reconstruction.next_measurement_locations(distance_only)[0], (63, 63))
        locations = sparse_reconstruction.next_measurement_locations(uncertainty, 3, min_separation=10)
        self.assertEqual(len(locations), 3)


class TestAlignment(unittest.TestCase):
    def test_subpixel_shift_is_recovered_and_undone(self):
        rng = np.random.default_rng(0)
        reference = ndimage.gaussian_filter(rng.random((96, 96)), 2)
        moving = ndimage.shift(reference, (3.4, -2.3), mode='wrap')
        shift = image_alignment.cross_correlation_shift(reference, moving)
        np.testing.assert_allclose(shift, (-3.4, 2.3), atol=0.25)
        restored = image_alignment.fourier_shift(moving, shift)
        self.assertLess(np.abs(restored - reference)[10:-10, 10:-10].mean(), 0.01)

    def test_fractional_shifts_without_wraparound(self):
        rng = np.random.default_rng(0)
        reference = ndimage.gaussian_filter(rng.random((64, 64)), 2)
        for offset in ((1.5, -1.5), (0.25, 0.7), (-7.6, 4.1)):
            moving = ndimage.shift(reference, offset, mode='nearest')
            shift = image_alignment.cross_correlation_shift(reference, moving)
            np.testing.assert_allclose(shift, -np.array(offset), atol=0.25)

    def test_max_shift_limits_search(self):
        rng = np.random.default_rng(1)
        reference = ndimage.gaussian_filter(rng.random((64, 64)), 1.5)
        moving = np.roll(reference, (20, 0), axis=(0, 1))
        shift = image_alignment.cross_correlation_shift(reference, moving, max_shift=5)
        self.assertTrue(np.all(np.abs(shift) <= 5.5))

    def test_stack_alignment(self):
        rng = np.random.default_rng(2)
        reference = ndimage.gaussian_filter(rng.random((80, 80)), 2)
        offsets = np.array([[0, 0], [2, 1], [4, -1], [6, -3]], dtype=float)
        stack = np.stack([ndimage.shift(reference, offset, mode='wrap') for offset in offsets])
        for cumulative in (False, True):
            shifts = image_alignment.align_image_stack(stack, cumulative=cumulative)
            np.testing.assert_allclose(shifts, -offsets, atol=0.3)
        average, _ = image_alignment.align_and_average(stack)
        self.assertEqual(average.shape, (80 - 12, 80 - 6))


class TestRingAnalysis(unittest.TestCase):
    def _gold_pattern(self, pixel_size_mrad=0.25, voltage=200e3, lattice=4.078, disk_radius=0):
        size = 256
        rows, columns = np.indices((size, size), dtype=float)
        center = (127.3, 128.6)
        radius = np.hypot(rows - center[0], columns - center[1])
        wavelength = utilities.get_wavelength(voltage, 'A')
        rings = np.zeros((size, size))
        for hkl in ((1, 1, 1), (2, 0, 0), (2, 2, 0)):
            d_spacing = lattice / np.sqrt(np.sum(np.square(hkl)))
            angle_mrad = 2 * np.arcsin(wavelength / (2 * d_spacing)) * 1e3
            rings += np.exp(-0.5 * ((radius - angle_mrad / pixel_size_mrad) / 0.8) ** 2)
        if disk_radius == 0:
            return rings + 50.0 * (radius <= 6), center, pixel_size_mrad
        disk_rows, disk_columns = np.indices((2 * disk_radius + 1,) * 2) - disk_radius
        disk = (np.hypot(disk_rows, disk_columns) <= disk_radius).astype(float)
        disk /= disk.sum()
        source = rings.copy()
        source[int(round(center[0])), int(round(center[1]))] += 3000.0
        return ndimage.convolve(source, disk, mode='constant'), center, pixel_size_mrad

    def test_direct_beam(self):
        pattern, center, _ = self._gold_pattern()
        beam = ring_analysis.find_direct_beam(pattern)
        np.testing.assert_allclose(beam['center'], center, atol=0.3)
        self.assertAlmostEqual(beam['radius_px'], 6, delta=1)

    def test_lattice_parameter_from_assigned_rings(self):
        pattern, _, pixel_size = self._gold_pattern()
        result = ring_analysis.analyze_ring_pattern(pattern, pixel_size, 200e3, deconvolve=False,
                                                    hkl=[(1, 1, 1), (2, 0, 0), (2, 2, 0)])
        self.assertEqual(len(result['rings']), 3)
        for ring in result['rings']:
            self.assertAlmostEqual(ring['lattice_parameter_A'], 4.078, delta=0.04)

    def test_deconvolution_keeps_ring_position(self):
        pattern, _, pixel_size = self._gold_pattern(disk_radius=4)
        hkl = [(1, 1, 1), (2, 0, 0), (2, 2, 0)]
        result = ring_analysis.analyze_ring_pattern(pattern, pixel_size, 200e3, deconvolve=True,
                                                    deconvolution_iterations=40, hkl=hkl)
        self.assertEqual(len(result['rings']), 3)
        for ring in result['rings']:
            self.assertAlmostEqual(ring['lattice_parameter_A'], 4.078, delta=0.05)

    def test_d_spacing_matches_bragg(self):
        wavelength = utilities.get_wavelength(300e3, 'A')
        angle = 2 * np.arcsin(wavelength / (2 * 2.0)) * 1e3
        self.assertAlmostEqual(float(ring_analysis.angle_to_d_spacing(angle, 300e3)), 2.0, places=6)


class TestFourDStem(unittest.TestCase):
    def _cube(self, scan_shift=(0, 0), pattern_shift=(0, 0)):
        rng = np.random.default_rng(4)
        sample = ndimage.gaussian_filter(rng.random((24, 24)), 1.5)
        sample = ndimage.shift(sample, scan_shift, mode='wrap')
        rows, columns = np.indices((32, 32), dtype=float)
        disk = np.hypot(rows - 15.5 - pattern_shift[0], columns - 15.5 - pattern_shift[1]) <= 4
        ring = np.abs(np.hypot(rows - 15.5 - pattern_shift[0], columns - 15.5 - pattern_shift[1]) - 10) < 1.5
        return disk[None, None] * (1 - sample)[..., None, None] + ring[None, None] * sample[..., None, None]

    def test_virtual_images(self):
        cube = self._cube()
        bright_field = four_d_stem.virtual_image(cube, 0, 4)
        dark_field = four_d_stem.virtual_image(cube, 8, 12)
        self.assertLess(np.corrcoef(bright_field.ravel(), dark_field.ravel())[0, 1], -0.9)
        com = four_d_stem.center_of_mass_shifts(self._cube(pattern_shift=(2, -1)))
        np.testing.assert_allclose(com.mean(axis=(0, 1)), (2, -1), atol=0.2)

    def test_maped_recovers_both_shifts(self):
        cubes = [self._cube(), self._cube(scan_shift=(2, 1), pattern_shift=(1.5, -1))]
        result = four_d_stem.maped_average(cubes)
        np.testing.assert_allclose(result['reciprocal_shifts'][1], (-1.5, 1), atol=0.3)
        np.testing.assert_allclose(result['real_shifts'][1], (-2, -1), atol=0.3)
        self.assertEqual(result['average'].shape, (20, 22, 32, 32))

    def test_tilt_series_map(self):
        stack = np.random.default_rng(0).random((5, 16, 20))
        self.assertEqual(four_d_stem.tilt_series_map(stack).shape, (5, 20))


class TestOptimization(unittest.TestCase):
    def test_quadratic_fit_is_exact_for_quadratic(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(30, 3))
        scores = 1 - np.sum((points - [0.2, -0.1, 0.3]) ** 2, axis=1)
        predict = optimization_tools.fit_quadratic_response(points, scores, ridge=0)
        np.testing.assert_allclose(predict(points), scores, atol=1e-8)

    def test_optimizer_finds_focus(self):
        optimum = 37e-9

        def score(z):
            return float(np.exp(-((z - optimum) / 60e-9) ** 2))

        optimizer = optimization_tools.TrustRegionOptimizer(start=0.0, step=20e-9, max_iterations=12,
                                                            bounds=[(-2e-6, 2e-6)])
        while (setting := optimizer.ask()) is not None:
            optimizer.tell(setting, score(setting[0]))
        best, _ = optimizer.best()
        self.assertAlmostEqual(best[0], optimum, delta=5e-9)

    def test_optimizer_finds_a_sharp_peak_with_flat_tails(self):
        optimum = 30e-9

        def score(z):
            distance_nm = min(abs(z - optimum) * 1e9, 72.0)
            return 0.05 + np.exp(-(distance_nm / 25.0) ** 2)

        optimizer = optimization_tools.TrustRegionOptimizer(start=120e-9, step=40e-9, max_iterations=8)
        while (setting := optimizer.ask()) is not None:
            optimizer.tell(setting, score(setting[0]))
        self.assertAlmostEqual(optimizer.best()[0][0], optimum, delta=4e-9)

    def test_local_points_keep_enough_for_a_quadratic(self):
        points = np.array([[0.0], [10.0], [20.0], [100.0], [200.0]])
        chosen = optimization_tools.local_points(points, [0.1, 0.9, 0.3, 0.0, 0.0], trust_radius=5.0)
        self.assertEqual(sorted(chosen.tolist()), [0, 1, 2])

    def test_optimizer_two_parameters_minimize(self):
        target = np.array([3.0, -2.0])
        optimizer = optimization_tools.TrustRegionOptimizer(start=[0, 0], step=[2, 2], maximize=False,
                                                            max_iterations=15, max_step=[4, 4])
        while (setting := optimizer.ask()) is not None:
            optimizer.tell(setting, float(np.sum((setting - target) ** 2)))
        best, _ = optimizer.best()
        np.testing.assert_allclose(best, target, atol=0.3)

    def test_bounds_are_respected(self):
        proposal = optimization_tools.propose_next_point([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 5.0,
                                                         bounds=[0.0, 2.5])
        self.assertTrue(0.0 <= proposal[0] <= 2.5)

    def test_parabola_vertex_and_trend(self):
        x = np.linspace(-8, 8, 9)
        vertex = optimization_tools.parabola_vertex(x, 5 - (x - 1.5) ** 2)
        self.assertAlmostEqual(vertex['vertex_x'], 1.5)
        self.assertLess(vertex['curvature'], 0)
        trend = optimization_tools.linear_trend(x, 2 * x + 1 + np.random.default_rng(0).normal(0, 0.1, 9))
        self.assertTrue(trend['slope_ci_low'] < 2 < trend['slope_ci_high'])


if __name__ == '__main__':
    unittest.main()
