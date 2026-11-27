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
        result = self.call('segment_image', {'filename': self.path('disks.npy'), 'pixel_size': 0.5})
        self.assertEqual(result['n_regions'], 2)
        self.assertEqual(np.load(result['labels_file']).max(), 2)
        metrics = self.call('compute_image_metrics', {'filename': self.path('disks.npy'), 'metrics': ['contrast']})
        self.assertGreater(metrics['metrics']['contrast'], 0)

    def test_shift_and_stack(self):
        reference = ndimage.gaussian_filter(np.random.default_rng(0).random((64, 64)), 2)
        np.save(self.path('reference.npy'), reference)
        np.save(self.path('moving.npy'), ndimage.shift(reference, (2.5, -1.0), mode='nearest'))
        result = self.call('measure_image_shift', {'reference_filename': self.path('reference.npy'),
                                                   'moving_filename': self.path('moving.npy'),
                                                   'pixel_size': 0.1})
        np.testing.assert_allclose(result['shift_px'], (-2.5, 1.0), atol=0.25)
        np.testing.assert_allclose(result['shift_physical'], (-0.25, 0.1), atol=0.025)

    def test_sparse_workflow(self):
        rows, columns = np.indices((48, 48))
        image = 10 + np.sin(rows / 6) * np.cos(columns / 8)
        mask = self.call('plan_sampling_mask', {'rows': 48, 'columns': 48, 'fraction': 0.25,
                                                'output_filename': self.path('mask.npy')})
        np.save(self.path('image.npy'), image * np.load(mask['mask_file']))
        result = self.call('reconstruct_sparse_data', {'filename': self.path('image.npy'),
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
        result = self.call('analyze_diffraction_rings', {'filename': self.path('ring.npy'), 'pixel_size_mrad': 0.2,
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
        image = self.call('four_d_stem_virtual_image', {'filename': self.path('cube_a.npy'),
                                                        'inner_radius': 0, 'outer_radius': 4})
        np.testing.assert_allclose(np.load(image['image_file']), sample * disk.sum())
        maped = self.call('maped_average_cubes', {'filenames': [self.path('cube_a.npy'), self.path('cube_b.npy')],
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
