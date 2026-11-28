"""Calls every analysis MCP tool through a real FastMCP client (in-memory transport)."""
import asyncio
import json
import os
import tempfile
import unittest

import numpy as np
from scipy import ndimage

from pyTEMlib import utilities

try:
    from fastmcp import Client, FastMCP
    from pyTEMlib import mcp_analysis_tools
    fastmcp_available = True
except ImportError:
    fastmcp_available = False


@unittest.skipUnless(fastmcp_available, 'fastmcp not installed')
class TestAnalysisMcpTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.server = FastMCP('test')
        cls.names = mcp_analysis_tools.register(cls.server, {})

    def path(self, name):
        return os.path.join(self.directory, name)

    def call(self, name, arguments):
        async def run():
            async with Client(self.server) as client:
                return await client.call_tool(name, arguments)
        return json.loads(asyncio.run(run()).content[0].text)

    def test_every_tool_is_listed(self):
        async def run():
            async with Client(self.server) as client:
                return {tool.name for tool in await client.list_tools()}
        self.assertEqual(asyncio.run(run()), set(self.names))

    def test_segmentation_and_metrics(self):
        rows, columns = np.indices((120, 120))
        np.save(self.path('disks.npy'), (((rows - 40) ** 2 + (columns - 40) ** 2) < 100)
                + 2.0 * (((rows - 85) ** 2 + (columns - 80) ** 2) < 225))
        result = self.call('segment_image', {'source': self.path('disks.npy'), 'pixel_size': 0.5})
        self.assertEqual(result['n_regions'], 2)
        self.assertEqual(np.load(result['labels_file']).max(), 2)
        metrics = self.call('compute_image_metrics', {'source': self.path('disks.npy'), 'metrics': ['contrast']})
        self.assertGreater(metrics['metrics']['contrast'], 0)

    def test_shift_and_stack(self):
        reference = ndimage.gaussian_filter(np.random.default_rng(0).random((64, 64)), 2)
        np.save(self.path('reference.npy'), reference)
        np.save(self.path('moving.npy'), ndimage.shift(reference, (2.5, -1.0), mode='nearest'))
        result = self.call('measure_image_shift', {'reference': self.path('reference.npy'),
                                                   'moving': self.path('moving.npy'),
                                                   'pixel_size': 0.1})
        np.testing.assert_allclose(result['shift_px'], (-2.5, 1.0), atol=0.25)
        np.testing.assert_allclose(result['shift_physical'], (-0.25, 0.1), atol=0.025)

    def test_sparse_workflow(self):
        rows, columns = np.indices((48, 48))
        image = 10 + np.sin(rows / 6) * np.cos(columns / 8)
        mask = self.call('plan_sampling_mask', {'rows': 48, 'columns': 48, 'fraction': 0.25,
                                                'output_filename': self.path('mask.npy')})
        np.save(self.path('image.npy'), image * np.load(mask['mask_file']))
        result = self.call('reconstruct_sparse_data', {'source': self.path('image.npy'),
                                                       'mask_filename': mask['mask_file']})
        self.assertLess(np.abs(np.load(result['reconstruction_file']) - image).mean(), 0.1)
        nxt = self.call('suggest_next_measurements', {'rows': 8, 'columns': 8, 'count': 2,
                                                      'measurements': [[0, 0, 1.0], [7, 7, 2.0]]})
        self.assertEqual(len(nxt['next_positions']), 2)
        self.assertNotIn([0, 0], nxt['next_positions'])

    def test_diffraction_rings(self):
        rows, columns = np.indices((256, 256))
        radius = np.hypot(rows - 127.5, columns - 127.5)
        wavelength = utilities.get_wavelength(300e3, 'A')
        ring_radius = 2 * np.arcsin(wavelength / (2 * 4.078 / np.sqrt(3))) * 1e3 / 0.2
        np.save(self.path('ring.npy'), 50.0 * (radius <= 5) + np.exp(-0.5 * ((radius - ring_radius) / 0.8) ** 2))
        result = self.call('analyze_diffraction_rings', {'source': self.path('ring.npy'), 'pixel_size_mrad': 0.2,
                                                         'acceleration_voltage': 300e3, 'deconvolve': False,
                                                         'hkl': [[1, 1, 1]]})
        self.assertAlmostEqual(result['rings'][0]['lattice_parameter_A'], 4.078, delta=0.04)

    def test_four_d_stem(self):
        rows, columns = np.indices((16, 16))
        disk = (np.hypot(rows - 7.5, columns - 7.5) <= 3).astype(float)
        sample = ndimage.gaussian_filter(np.random.default_rng(1).random((12, 12)), 1)
        cube = sample[..., None, None] * disk
        np.save(self.path('cube_a.npy'), cube)
        np.save(self.path('cube_b.npy'), ndimage.shift(sample, (1, 0), mode='wrap')[..., None, None] * disk)
        image = self.call('four_d_stem_virtual_image', {'source': self.path('cube_a.npy'),
                                                        'inner_radius': 0, 'outer_radius': 4})
        np.testing.assert_allclose(np.load(image['image_file']), sample * disk.sum())
        maped = self.call('maped_average_cubes', {'sources': [self.path('cube_a.npy'), self.path('cube_b.npy')],
                                                  'dark_field_inner': 0.0, 'dark_field_outer': 0.3})
        np.testing.assert_allclose(maped['real_shifts_px'][1], (-1, 0), atol=0.25)

    def test_optimisation_helpers(self):
        result = self.call('propose_next_setting', {'points': [[0], [-1], [1]], 'scores': [1.0, 0.5, 0.9],
                                                    'trust_radius': [1]})
        self.assertTrue(-1 <= result['next_point'][0] <= 1)
        vertex = self.call('fit_parabola_vertex', {'x': [-1, 0, 1, 2], 'y': [1, 0, 1, 4]})
        self.assertAlmostEqual(vertex['vertex_x'], 0.0, places=6)
        trend = self.call('fit_linear_trend', {'x': [0, 1, 2, 3], 'y': [1, 3, 5, 7.1]})
        self.assertAlmostEqual(trend['slope'], 2.03, places=2)


if __name__ == '__main__':
    unittest.main()


def _tiled_available():
    import shutil
    import importlib.util
    if shutil.which('tiled') is None or importlib.util.find_spec('h5py') is None:
        return False
    try:
        import tiled.server.app  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(fastmcp_available and _tiled_available(), 'fastmcp, tiled server or h5py not installed')
class TestDataKeysFromTiled(unittest.TestCase):
    """Tools read instrument data by key from a real Tiled server, laid out as acquisitions write it."""

    @classmethod
    def setUpClass(cls):
        import socket
        import subprocess
        import time
        import h5py
        cls.directory = tempfile.mkdtemp()
        cls.output = tempfile.mkdtemp()
        rows, columns = np.indices((96, 96))
        with h5py.File(os.path.join(cls.directory, 'stem_image_001.h5'), 'w') as handle:
            handle['image/HAADF'] = (((rows - 48) ** 2 + (columns - 40) ** 2) < 200).astype(float)
            handle['image/BF'] = 1.0 - handle['image/HAADF'][()]
        radius = np.hypot(*(np.indices((256, 256)) - 127.5))
        wavelength = utilities.get_wavelength(200e3, 'A')
        ring = 2 * np.arcsin(wavelength / (2 * 4.078 / np.sqrt(3))) * 1e3 / 0.25
        with h5py.File(os.path.join(cls.directory, 'camera_image_001.h5'), 'w') as handle:
            dataset = handle.create_dataset('image', data=50.0 * (radius <= 5)
                                            + np.exp(-0.5 * ((radius - ring) / 0.8) ** 2))
            dataset.attrs['pixel_size_mrad'] = 0.25
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        cls.process = subprocess.Popen(['tiled', 'serve', 'directory', cls.directory, '--public', '--port', str(port)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.uri = f'http://127.0.0.1:{port}'
        from tiled.client import from_uri
        deadline = time.time() + 60
        while True:
            try:
                if {'stem_image_001', 'camera_image_001'} <= set(from_uri(cls.uri)):
                    break
            except Exception:
                pass
            if time.time() > deadline:
                cls.process.kill()
                raise RuntimeError('Tiled server did not start')
            time.sleep(0.5)
        cls.server = FastMCP('test')
        mcp_analysis_tools.register(cls.server, {}, tiled_uri=cls.uri, output_dir=cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(10)

    def call(self, name, arguments):
        async def run():
            async with Client(self.server) as client:
                return await client.call_tool(name, arguments, raise_on_error=False)
        return asyncio.run(run())

    def test_multi_detector_key_needs_a_detector(self):
        result = self.call('compute_image_metrics', {'source': 'stem_image_001'})
        self.assertTrue(result.is_error)
        self.assertIn('image/HAADF', result.content[0].text)
        full_path = json.loads(self.call('compute_image_metrics', {'source': 'stem_image_001',
                                                                   'dataset_key': 'image/BF'}).content[0].text)
        self.assertIn('contrast', full_path['metrics'])

    def test_detector_path_inside_a_key(self):
        result = json.loads(self.call('segment_image', {'source': 'stem_image_001',
                                                        'dataset_key': 'HAADF'}).content[0].text)
        self.assertEqual(result['n_regions'], 1)
        self.assertTrue(result['labels_file'].startswith(self.output))

    def test_calibration_comes_from_metadata(self):
        result = json.loads(self.call('analyze_diffraction_rings', {'source': 'camera_image_001', 'deconvolve': False,
                                                                    'hkl': [[1, 1, 1]]}).content[0].text)
        self.assertAlmostEqual(result['rings'][0]['lattice_parameter_A'], 4.078, delta=0.04)

    def test_unknown_key_is_reported(self):
        result = self.call('compute_image_metrics', {'source': 'no_such_acquisition'})
        self.assertTrue(result.is_error)
        self.assertIn('no_such_acquisition', result.content[0].text)


@unittest.skipUnless(fastmcp_available, 'fastmcp not installed')
class TestTileMapTools(unittest.TestCase):
    """Stitching with per-tile segmentation, entirely through MCP calls."""

    def test_segment_each_tile_while_stitching(self):
        directory = tempfile.mkdtemp()
        server = FastMCP('test')
        mcp_analysis_tools.register(server, {}, output_dir=directory)
        rows, columns = np.indices((120, 200))
        scene = 0.1 + 0.05 * ndimage.gaussian_filter(np.random.default_rng(0).random((120, 200)), 2)
        scene[(rows - 60) ** 2 + (columns - 100) ** 2 < 15 ** 2] += 1.0
        for name, (left, error) in {'left': (0, 0.0), 'right': (80, 3.0)}.items():
            np.save(os.path.join(directory, f'{name}.npy'), scene[:, left:left + 120])

        async def run():
            async with Client(server) as client:
                async def call(tool, arguments):
                    return json.loads((await client.call_tool(tool, arguments)).content[0].text)
                await call('map_create', {'map_dir': os.path.join(directory, 'map'), 'pixel_size': 1.0})
                results = []
                for name, (left, error) in {'left': (0, 0.0), 'right': (80, 3.0)}.items():
                    tile = os.path.join(directory, f'{name}.npy')
                    labels = await call('segment_image', {'source': tile})
                    results.append(await call('map_add_tile', {
                        'map_dir': os.path.join(directory, 'map'), 'source': tile,
                        'position': [left + 60 + error, 60], 'labels': labels['labels_file']}))
                summary = await call('map_summary', {'map_dir': os.path.join(directory, 'map')})
                capabilities = await call('analysis_capabilities', {})
                return results, summary, capabilities

        results, summary, capabilities = asyncio.run(run())
        self.assertEqual(results[1]['regions_joined'], 1)
        self.assertEqual(results[1]['refinement_px'], [0.0, -3.0])
        self.assertEqual(summary['map_shape'], [120, 200])
        self.assertEqual(len(summary['regions']), 1)
        self.assertAlmostEqual(summary['regions'][0]['equivalent_radius'], 15, delta=1.5)
        self.assertEqual(np.load(summary['map_labels_file']).max(), 1)
        self.assertEqual(set(capabilities), {'bayesian_optimization', 'sam2_segmentation'})
