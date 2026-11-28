# -*- coding: utf-8 -*-
"""
Created on December 28 2021

@author: Gerd Duscher
"""
# -*- coding: utf-8 -*-
"""
Created on December 28 2021

@author: Gerd Duscher
"""
from tracemalloc import start
import unittest

import os
import numpy as np
import sidpy
print(sidpy.__version__)
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

import pyTEMlib
print(pyTEMlib.__version__)


def get_dataset():
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_path, '../example_data/EDS-STO.emd')
    datasets = pyTEMlib.file_tools.open_file(file_name)
    dataset = datasets['SuperX']
    return dataset
spectrum = get_dataset()

class TestFileFunctions(unittest.TestCase):
   
    def test_dataset(self):
        """Test loading dataset"""
        
        start = np.searchsorted(spectrum.energy_scale.values, 100)
        energy_scale = spectrum.energy_scale.values[start:]
        detector_Efficiency= pyTEMlib.eds_tools.detector_response(spectrum)  # tags, spectrum.energy_scale.values[start:])
        if 'start_energy' not in spectrum.metadata['EDS']['detector']:
            spectrum.metadata['EDS']['detector']['start_energy'] = 120
        
        self.assertIsInstance(spectrum, sidpy.Dataset)

    def test_find_elements(self):
        """Test finding elements in EDS spectrum"""     
        minor_peaks = pyTEMlib.eds_tools.detect_peaks(spectrum, minimum_number_of_peaks=10)

        spectrum.metadata.setdefault('EDS', {}).setdefault('elements', {})
        del spectrum.metadata['EDS']['elements']

        elements = pyTEMlib.eds_tools.peaks_element_correlation(spectrum, minor_peaks)
        spectrum.metadata['EDS'].update(pyTEMlib.eds_tools.get_x_ray_lines(spectrum, elements))
        
        self.assertIsInstance(elements, list)
        self.assertTrue('Ti' in elements)

    def test_fit_spectrum(self):
        """Test fitting EDS spectrum"""
        peaks, _ = pyTEMlib.eds_tools.fit_model(spectrum, use_detector_efficiency=True)
        model = pyTEMlib.eds_tools.get_model(spectrum)
        self.assertIsInstance(model, np.ndarray)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertTrue(np.sum(peaks) > .9)

    def test_quantify_xsection(self):
        """Test quantification using cross sections"""
        pyTEMlib.eds_tools.quantify_eds(spectrum, mask=['Cu'])
        self.assertIn('elements', spectrum.metadata['EDS'])
        self.assertIn('Cu', spectrum.metadata['EDS']['elements'])

    def test_quantify_kfactors(self):
        """Test quantification using k-factors"""
        q_dict = pyTEMlib.eds_tools.load_k_factors()
        pyTEMlib.eds_tools.quantify_eds(spectrum, q_dict, mask=['Cu'])
        self.assertIsInstance(q_dict, dict)
        self.assertIn('elements', spectrum.metadata['EDS'])
        self.assertIn('Cu', spectrum.metadata['EDS']['elements'])

    def test_r_absorption(self):
        """Test absorption correction"""
        pyTEMlib.eds_tools.apply_absorption_correction(spectrum, 30)
        self.assertIn('corrected-atom%', spectrum.metadata['EDS']['elements']['Ti'])


class TestDetectPeaks(unittest.TestCase):

    def test_a_peak_is_found_on_its_channel_when_the_smoothing_width_comes_out_even(self):
        energy = np.arange(0.0, 10000.0, 10.0)
        counts = 900.0 * np.exp(-0.5 * ((energy - 1740.0) / (130.0 / 2.3548)) ** 2)
        dataset = sidpy.Dataset.from_array(counts)
        dataset.data_type = 'SPECTRUM'
        dataset.set_dimension(0, sidpy.Dimension(energy, name='energy_scale', units='eV',
                                                 quantity='energy', dimension_type='spectral'))
        dataset.metadata['EDS'] = {'detector': {'energy_resolution': 136.0, 'start_channel': 12}}
        peaks = pyTEMlib.eds_tools.detect_peaks(dataset, minimum_number_of_peaks=1)
        self.assertIn(174, list(peaks))
