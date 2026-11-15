"""The analysis functions behind the MCP server (pyTEMlib.mcp_analysis): dataset references, reference data,
EDS, EELS, peak fitting, images and the probe. Synthetic inputs have known answers; the example files are real data."""

import base64
import json
import os

import numpy as np
import pytest
from scipy import ndimage

from pyTEMlib import mcp_analysis as analysis

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_data')
EDS_STO = os.path.join(EXAMPLES, 'EDS-STO.emd')  # the SrTiO3 Super-X spectrum (on a copper grid)
LOW_LOSS = os.path.join(EXAMPLES, 'AL-DFoffset0.00.dm3')  # aluminium low-loss, plasmon at 15 eV
CORE_LOSS = os.path.join(EXAMPLES, 'EELS_STO.dm3')  # SrTiO3 core-loss, Ti-L and O-K
IMAGE = os.path.join(EXAMPLES, 'p1-3-hr3.dm3')
MAP = os.path.join(EXAMPLES, 'SI HAADF 1402.emd')


def gaussian(x, centre, fwhm, height):
    return height * np.exp(-0.5 * ((x - centre) / (fwhm / 2.3548)) ** 2)


def save_spectrum(tmp_path, name, energy, counts):
    path = tmp_path / name
    np.savetxt(path, np.column_stack([energy, counts]), delimiter=',', header='energy,counts')
    return str(path)


def save_array(tmp_path, name, array):
    path = tmp_path / name
    np.save(path, array)
    return str(path)


def ok(result):
    assert 'error' not in result, result
    json.dumps({key: value for key, value in result.items() if key != '_png'})  # every result must be JSON
    return result


def refused(result, code):
    assert result.get('error', {}).get('code') == code, result
    assert result['error']['message']
    return result


# ------------------------------------------------------------------ dataset references and reading
# Opening files is the host's job, so there is no tool for it; the analysis tools still read the dataset a reference names.

def not_resolved(reference, code):
    with pytest.raises(analysis.AnalysisError) as raised:
        analysis.resolve(reference)
    assert raised.value.code == code and raised.value.message


def test_a_reference_without_a_key_names_the_first_dataset_of_the_file():
    dataset = analysis.resolve(EDS_STO)
    assert dataset.data_type.name == 'SPECTRUM' and dataset.shape == (2048,)
    axis = dataset._axes[0]
    assert axis.units == 'eV' and axis.dimension_type.name == 'SPECTRAL'
    assert axis.values[0] == pytest.approx(-250.0) and axis.values[-1] == pytest.approx(20281.4, abs=1)


def test_a_key_names_one_dataset_of_a_file_with_several():
    dataset = analysis.resolve(f'{MAP}#Channel_004')
    assert dataset.data_type.name == 'IMAGE_STACK' and dataset.shape == (63, 178, 215)
    not_resolved(f'{MAP}#Channel_099', 'unknown_key')


def test_a_missing_file_and_an_unknown_format_are_refused_with_a_reason(tmp_path):
    not_resolved(str(tmp_path / 'nothing.dm3'), 'file_not_found')
    odd = tmp_path / 'notes.xyz'
    odd.write_text('hello')
    not_resolved(str(odd), 'unsupported_format')
    not_resolved('', 'bad_reference')
    assert analysis.find_spectrum_peaks(str(tmp_path / 'nothing.dm3'))['error']['code'] == 'file_not_found'


def test_numpy_and_text_files_read_as_spectrum_image_and_stack(tmp_path):
    energy = np.linspace(0, 100, 201)
    assert analysis.resolve(save_spectrum(tmp_path, 's.csv', energy, energy * 2)).data_type.name == 'SPECTRUM'
    assert analysis.resolve(save_array(tmp_path, 'a1.npy', np.arange(50.0))).data_type.name == 'SPECTRUM'
    assert analysis.resolve(save_array(tmp_path, 'a2.npy', np.zeros((8, 9)))).data_type.name == 'IMAGE'
    assert analysis.resolve(save_array(tmp_path, 'a3.npy', np.zeros((3, 8, 9)))).data_type.name == 'IMAGE_STACK'
    not_resolved(save_array(tmp_path, 'a4.npy', np.zeros((2, 2, 2, 2))), 'unsupported_shape')


def test_a_text_spectrum_keeps_its_energy_axis(tmp_path):
    energy = np.linspace(-5, 95, 501)
    dataset = analysis.resolve(save_spectrum(tmp_path, 's.csv', energy, np.sin(energy)))
    np.testing.assert_allclose(dataset._axes[0].values, energy, atol=1e-9)
    np.testing.assert_allclose(np.asarray(dataset), np.sin(energy), atol=1e-6)


def test_an_emsa_spectrum_reads_with_the_energy_scale_of_its_header(tmp_path):
    path = tmp_path / 'spectrum.msa'
    counts = [10.0, 12.0, 30.0, 80.0, 31.0, 11.0]
    path.write_text('#FORMAT      : EMSA/MAS Spectral Data File\n#VERSION     : 1.0\n#NPOINTS     : 6\n#XUNITS      : eV\n'
                    '#XPERCHAN    : 10.0\n#OFFSET      : -20.0\n#DATATYPE    : XY\n#SPECTRUM    : Spectral Data Starts Here\n'
                    + ''.join(f'{-20.0 + 10.0 * index}, {value}\n' for index, value in enumerate(counts))
                    + '#ENDOFDATA   : \n')
    dataset = analysis.resolve(str(path))
    assert dataset.data_type.name == 'SPECTRUM' and dataset.shape == (6,)
    assert list(dataset._axes[0].values) == [-20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    np.testing.assert_allclose(np.asarray(dataset), counts)


def test_a_file_is_reread_after_it_changes(tmp_path):
    path = save_array(tmp_path, 'a.npy', np.zeros(10))
    assert analysis.resolve(path).shape == (10,)
    np.save(path, np.zeros(20))
    os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
    assert analysis.resolve(path).shape == (20,)


def test_allowed_roots_keep_reads_inside_them(tmp_path):
    inside = save_array(tmp_path, 'in.npy', np.zeros(5))
    analysis.set_allowed_roots([str(tmp_path)])
    try:
        assert analysis.resolve(inside).shape == (5,)
        analysis.loaded_datasets.pop(EDS_STO, None)
        not_resolved(EDS_STO, 'path_not_allowed')
    finally:
        analysis.set_allowed_roots(None)


def test_a_host_resolver_supplies_datasets_for_its_own_references(tmp_path):
    held = analysis._dataset_from_array(gaussian(np.arange(200.0), 80, 10, 5), 'held', axis_values=np.arange(200.0))

    def resolver(reference):
        return held if reference == 'catalogue/run-7/spectrum' else None

    analysis.add_resolver(resolver)
    try:
        assert analysis.resolve('catalogue/run-7/spectrum').shape == (200,)
        peaks = ok(analysis.find_spectrum_peaks('catalogue/run-7/spectrum', max_peaks=1))['peaks']
        assert peaks[0]['position'] == pytest.approx(80, abs=1)
    finally:
        analysis._resolvers.remove(resolver)


def test_analysis_never_changes_the_cached_dataset():
    before = json.dumps(dict(analysis.resolve(EDS_STO).metadata), sort_keys=True, default=str)
    ok(analysis.eds_quantify(EDS_STO, ['Sr', 'Ti', 'O']))
    assert json.dumps(dict(analysis.resolve(EDS_STO).metadata), sort_keys=True, default=str) == before


# ------------------------------------------------------------------ reference data

def test_emission_lines_of_iron_are_where_they_should_be():
    lines = {line['name']: line for line in ok(analysis.xray_emission_lines('Fe'))['lines']}
    assert lines['K-L3']['position_ev'] == pytest.approx(6404, abs=5) and lines['K-L3']['weight'] == 1.0
    assert lines['K-M3']['position_ev'] == pytest.approx(7058, abs=5) and lines['K-L3']['family'] == 'K'
    assert ok(analysis.xray_emission_lines(26))['element'] == 'Fe'
    only_k = ok(analysis.xray_emission_lines('Fe', energy_min_ev=6000, energy_max_ev=8000))['lines']
    assert only_k and all(line['family'] == 'K' for line in only_k)
    refused(analysis.xray_emission_lines('Xx'), 'unknown_element')


def test_absorption_edges_come_highest_first():
    edges = ok(analysis.xray_absorption_edges('Ti'))['edges']
    assert edges[0]['shell'] == 'K1' and edges[0]['onset_ev'] == pytest.approx(4966, abs=3)
    assert [edge['onset_ev'] for edge in edges] == sorted((edge['onset_ev'] for edge in edges), reverse=True)
    assert {edge['shell']: edge['onset_ev'] for edge in edges}['L3'] == pytest.approx(455.5, abs=1)


def test_element_reference_data():
    titanium = ok(analysis.element_reference_data('Ti'))
    assert titanium == {'symbol': 'Ti', 'atomic_number': 22, 'atomic_weight': pytest.approx(47.88, abs=0.02),
                        'density_g_cm3': pytest.approx(4.5, abs=0.1), 'fluorescence_yield': {'K': pytest.approx(0.19, abs=0.03)}}


def test_element_lines_for_drawing_are_sorted_and_labelled():
    lines = ok(analysis.eds_element_lines(['O', 'Ti'], energy_min_ev=0, energy_max_ev=6000))['lines']
    assert {line['symbol'] for line in lines} == {'O', 'Ti'}
    assert [line['position_ev'] for line in lines] == sorted(line['position_ev'] for line in lines)
    assert all(0 <= line['position_ev'] <= 6000 and line['weight'] >= 0.01 for line in lines)


# ------------------------------------------------------------------ EDS: the SrTiO3 spectrum the app's fixture test used

@pytest.mark.parametrize('label, expected_ev', [('Ti K-L3', 4511.0), ('Ti K-M3', 4931.0), ('Cu K-L3', 8048.0),
                                                ('Sr K-L3', 14165.0), ('Sr L3-M5', 1806.0)])
def test_peak_search_finds_the_known_lines_of_the_srtio3_spectrum(label, expected_ev):
    peaks = ok(analysis.eds_find_peaks(EDS_STO, number_of_peaks=10))['peaks']
    assert any(abs(peak['position_ev'] - expected_ev) < 40 for peak in peaks), f'no peak within 40 eV of {label}'
    assert all(peak['position_ev'] >= 120 for peak in peaks)


def test_identification_finds_the_sample_and_its_support_grid():
    found = [element['symbol'] for element in ok(analysis.eds_identify_elements(EDS_STO))['elements']]
    assert set(found) == {'Sr', 'Ti', 'O', 'Cu'}


def test_describing_titanium_reports_its_k_family_at_the_right_energy():
    titanium = ok(analysis.eds_describe_elements(EDS_STO, ['Ti']))['elements'][0]
    assert titanium['symbol'] == 'Ti' and titanium['atomic_number'] == 22
    family = titanium['families']['K']
    assert family['main_line'] == 'K-L3' and family['height'] > 1000
    assert {line['name']: line['position_ev'] for line in family['lines']}['K-L3'] == pytest.approx(4510.8, abs=2)


def test_quantifying_srtio3_gives_a_plausible_oxide():
    result = ok(analysis.eds_quantify(EDS_STO, ['Sr', 'Ti', 'O'], include_curves=True))
    percent = {element['symbol']: element['atomic_percent'] for element in result['elements']}
    assert sum(percent.values()) == pytest.approx(100.0)
    # SrTiO3 is 20/20/60; without absorption correction oxygen comes out low, as pyTEMlib's own notebooks show
    assert 15 < percent['Sr'] < 35 and 20 < percent['Ti'] < 40 and 35 < percent['O'] < 65
    assert all(element['family'] == 'K' and element['cross_section'] > 0 for element in result['elements'])
    assert len(result['curves']['model']['x']) == len(result['curves']['model']['y']) == 2048
    assert 'curves' not in ok(analysis.eds_quantify(EDS_STO, ['Sr', 'Ti', 'O']))


def test_quantifying_with_a_chosen_family_uses_it():
    result = ok(analysis.eds_quantify(EDS_STO, [{'symbol': 'Sr', 'family': 'L'}, 'Ti', 'O']))
    assert {element['symbol']: element['family'] for element in result['elements']} == {'Sr': 'L', 'Ti': 'K', 'O': 'K'}
    refused(analysis.eds_quantify(EDS_STO, [{'symbol': 'O', 'family': 'L'}]), 'no_lines_in_range')
    refused(analysis.eds_quantify(EDS_STO, ['Sr'], beam_energy_ev=120000), 'unsupported_beam_energy')


def test_net_intensity_recovers_a_known_peak_area_over_a_sloped_background(tmp_path):
    energy = np.arange(6000.0, 6800.0, 2.0)
    peak = gaussian(energy, 6404.0, 130.0, 5000.0)
    path = save_spectrum(tmp_path, 'fe.csv', energy, 50.0 + 0.1 * (energy - 6000.0) + peak)
    inside = np.abs(energy - 6404.0) <= 200.0
    assert ok(analysis.eds_net_intensity(path, 6404.0, 400.0))['net_intensity'] == pytest.approx(peak[inside].sum(), rel=0.02)
    ramp = save_spectrum(tmp_path, 'ramp.csv', energy, 0.1 * (energy - 6000.0))
    assert ok(analysis.eds_net_intensity(ramp, 6404.0, 400.0))['net_intensity'] == pytest.approx(0.0, abs=1e-6)
    refused(analysis.eds_net_intensity(path, 9000.0, 400.0), 'bad_window')


def test_peak_search_ends_on_a_spectrum_with_fewer_peaks_than_asked_for(tmp_path):
    energy = np.arange(0.0, 10000.0, 10.0)
    path = save_spectrum(tmp_path, 'two.csv', energy, gaussian(energy, 1740, 130, 900) + gaussian(energy, 6404, 150, 400))
    peaks = ok(analysis.eds_find_peaks(path, number_of_peaks=30))['peaks']
    assert {round(peak['position_ev'], -1) for peak in peaks} >= {1740.0, 6400.0}


# ------------------------------------------------------------------ EELS low loss

def low_loss_spectrum(tmp_path, zero_at=0.4, plasmon=0.0):
    energy = np.arange(-10.0, 80.0, 0.05)
    lorentz = 1.0 / (1.0 + ((energy - zero_at) / 0.25) ** 2)
    counts = 2.0e6 * lorentz * lorentz + plasmon * gaussian(energy, 15.0 + zero_at, 3.0, 1.0)
    return save_spectrum(tmp_path, 'low_loss.csv', energy, counts), energy, counts


def test_the_zero_loss_peak_is_found_where_it_was_put(tmp_path):
    path, _, _ = low_loss_spectrum(tmp_path, zero_at=0.4)
    found = ok(analysis.eels_find_zero_loss_peak(path))
    assert found['position_ev'] == pytest.approx(0.4, abs=0.03) and found['height'] == pytest.approx(2.0e6, rel=0.01)
    assert found['fwhm_ev'] == pytest.approx(0.32, abs=0.1)  # the product of two Lorentzians of half-width 0.25 eV
    aligned = ok(analysis.eels_align_zero_loss(path, include_curves=True))
    assert aligned['shift_ev'] == pytest.approx(0.4, abs=0.03)
    curve = aligned['curves']['aligned']
    assert abs(curve['x'][int(np.argmax(curve['y']))]) < 0.06


def test_a_core_loss_spectrum_has_no_zero_loss_peak():
    refused(analysis.eels_find_zero_loss_peak(CORE_LOSS), 'no_zero_loss')
    refused(analysis.eels_relative_thickness(CORE_LOSS), 'no_zero_loss')


def test_the_zero_loss_of_real_data_is_found_and_fitted():
    found = ok(analysis.eels_find_zero_loss_peak(LOW_LOSS))
    assert found['position_ev'] == pytest.approx(-0.136, abs=0.02) and 0.1 < found['fwhm_ev'] < 0.3
    fitted = ok(analysis.eels_resolution_function(LOW_LOSS, include_curves=True))
    assert len(fitted['parameters']) == 6 and 0.1 < fitted['fwhm_ev'] < 0.3
    zero_loss, spectrum = fitted['curves']['zero_loss'], fitted['curves']['spectrum']
    assert max(zero_loss['y']) == pytest.approx(max(spectrum['y']), rel=0.05)


def test_relative_thickness_agrees_between_the_fit_and_a_window_on_real_data():
    by_fit = ok(analysis.eels_relative_thickness(LOW_LOSS))
    by_window = ok(analysis.eels_relative_thickness(LOW_LOSS, -1.5, 1.5))
    assert by_fit['zero_loss_from'] == 'fitted resolution function' and by_window['zero_loss_from'] == 'window'
    assert by_fit['t_over_lambda'] == pytest.approx(by_window['t_over_lambda'], abs=0.02)
    assert 0.1 < by_window['t_over_lambda'] < 0.3
    subtracted = ok(analysis.eels_subtract_zero_loss(LOW_LOSS, include_curves=True))
    assert subtracted['t_over_lambda'] == pytest.approx(by_fit['t_over_lambda'])
    assert subtracted['inelastic_integral'] == pytest.approx(subtracted['total_integral'] - subtracted['zero_loss_integral'])
    assert set(subtracted['curves']) == {'zero_loss', 'difference'}
    refused(analysis.eels_relative_thickness(LOW_LOSS, zero_loss_start_ev=-1.0), 'bad_window')


def test_relative_thickness_of_a_known_window_ratio(tmp_path):
    energy = np.arange(-10.0, 60.0, 0.1)
    elastic, inelastic = gaussian(energy, 0.0, 1.0, 1.0e5), gaussian(energy, 20.0, 5.0, 4.0e3)
    path = save_spectrum(tmp_path, 'known.csv', energy, elastic + inelastic)
    expected = np.log((elastic.sum() + inelastic.sum()) / elastic.sum())
    measured = ok(analysis.eels_relative_thickness(path, -5.0, 5.0, align_zero_loss=False))
    assert measured['t_over_lambda'] == pytest.approx(expected, rel=1e-3)


def test_the_aluminium_plasmon_is_at_fifteen_ev():
    fitted = ok(analysis.eels_fit_drude_plasmon(LOW_LOSS, 12, 18, include_curves=True))
    assert fitted['peak_position_ev'] == pytest.approx(15.0, abs=0.3)
    assert fitted['assumed_collection_angle_mrad'] == 30.0  # the file carries no angles, and the result says so
    fit, spectrum = fitted['curves']['fit'], fitted['curves']['spectrum']
    peak = int(np.argmax(fit['y']))
    assert fit['x'][peak] == pytest.approx(15.0, abs=0.3) and fit['y'][peak] == pytest.approx(spectrum['y'][peak], rel=0.2)
    refused(analysis.eels_fit_drude_plasmon(LOW_LOSS, 18, 12), 'bad_window')
    refused(analysis.eels_fit_drude_plasmon(LOW_LOSS, -2, 12), 'bad_window')


def test_multiple_scattering_gives_a_thickness_near_the_log_ratio_one():
    fitted = ok(analysis.eels_fit_multiple_scattering(LOW_LOSS, 12, 18, max_energy_ev=35, include_curves=True))
    assert fitted['peak_position_ev'] == pytest.approx(15.0, abs=0.3)
    assert 0.1 < fitted['thickness_over_mfp'] < 0.35  # the log-ratio method gives 0.18 for this spectrum
    assert set(fitted['curves']) == {'spectrum', 'fit', 'zero_loss'}
    refused(analysis.eels_fit_multiple_scattering(LOW_LOSS, 12, 18, max_energy_ev=15), 'bad_window')


# ------------------------------------------------------------------ EELS core loss

def edge_spectrum(tmp_path, amplitude=5.0e9, exponent=3.0, edge_at=500.0, edge_height=40.0):
    energy = np.arange(300.0, 800.0, 0.5)
    background = amplitude * energy ** -exponent
    return save_spectrum(tmp_path, 'edge.csv', energy, background + edge_height * (energy >= edge_at)), energy, background


def test_a_power_law_background_recovers_its_amplitude_and_exponent(tmp_path):
    path, energy, background = edge_spectrum(tmp_path)
    fitted = ok(analysis.eels_fit_power_law_background(path, 350, 480, include_curves=True))
    assert fitted['exponent'] == pytest.approx(3.0, rel=1e-3) and fitted['amplitude'] == pytest.approx(5.0e9, rel=1e-2)
    assert fitted['noise_level'] < 1e-3
    np.testing.assert_allclose(fitted['curves']['background']['y'], background, rtol=1e-3)
    assert 'curves' not in ok(analysis.eels_fit_power_law_background(path, 350, 480))


def test_subtracting_the_background_leaves_the_edge(tmp_path):
    path, energy, _ = edge_spectrum(tmp_path, edge_height=40.0)
    difference = np.array(ok(analysis.eels_subtract_power_law_background(path, 350, 480, include_curves=True))['curves']['difference']['y'])
    assert np.abs(difference[energy < 500]).max() < 0.05
    np.testing.assert_allclose(difference[energy >= 500], 40.0, atol=0.05)


def test_integrating_a_step_edge_gives_its_height_times_the_channels(tmp_path):
    path, energy, _ = edge_spectrum(tmp_path, edge_height=40.0)
    integrated = ok(analysis.eels_integrate_signal(path, 350, 480, 500, 550))
    channels = np.count_nonzero((energy >= 500) & (energy < 550))
    assert integrated['signal_integral'] == pytest.approx(40.0 * channels, rel=1e-2)
    assert integrated['raw_integral'] == pytest.approx(integrated['signal_integral'] + integrated['background_integral'])
    refused(analysis.eels_integrate_signal(path, 350, 520, 500, 550), 'bad_window')
    refused(analysis.eels_fit_power_law_background(path, 350, 350.5), 'bad_window')


def test_core_loss_quantification_of_srtio3_gives_one_titanium_to_three_oxygen():
    result = ok(analysis.eels_quantify(CORE_LOSS, ['Ti', 'O'], include_curves=True))
    percent = {element['element']: element['atomic_percent'] for element in result['elements']}
    assert percent['Ti'] == pytest.approx(25.0, abs=5.0) and percent['O'] == pytest.approx(75.0, abs=5.0)
    edges = {element['element']: element['edge'] for element in result['elements']}
    assert edges == {'Ti': 'L3', 'O': 'K1'}
    model, spectrum = np.array(result['curves']['model']['y']), np.array(result['curves']['spectrum']['y'])
    assert np.abs(model - spectrum)[400:].mean() < 0.1 * spectrum[400:].mean()
    refused(analysis.eels_quantify(CORE_LOSS, ['Au']), 'no_edge_in_range')
    refused(analysis.eels_quantify(CORE_LOSS, []), 'bad_argument')


# ------------------------------------------------------------------ generic peaks

def three_peaks(tmp_path):
    x = np.arange(0.0, 200.0, 0.25)
    y = gaussian(x, 50.0, 6.0, 100.0) + gaussian(x, 110.0, 10.0, 60.0) + gaussian(x, 160.0, 4.0, 30.0)
    return save_spectrum(tmp_path, 'peaks.csv', x, y)


def test_the_most_prominent_peaks_are_found_with_their_widths(tmp_path):
    peaks = ok(analysis.find_spectrum_peaks(three_peaks(tmp_path), max_peaks=3))['peaks']
    assert [round(peak['position']) for peak in peaks] == [50, 110, 160]
    assert [peak['height'] for peak in peaks] == pytest.approx([100.0, 60.0, 30.0], rel=0.02)
    assert [peak['fwhm'] for peak in peaks] == pytest.approx([6.0, 10.0, 4.0], rel=0.25)
    assert len(ok(analysis.find_spectrum_peaks(three_peaks(tmp_path), max_peaks=1))['peaks']) == 1
    only_last = ok(analysis.find_spectrum_peaks(three_peaks(tmp_path), max_peaks=3, start_ev=140))['peaks']
    assert [round(peak['position']) for peak in only_last] == [160]


def test_gaussians_are_fitted_by_the_mixture_model(tmp_path):
    fitted = ok(analysis.fit_spectrum_peaks(three_peaks(tmp_path), max_peaks=3, include_curves=True))
    assert fitted['method'] == 'pyTEMlib Gaussian mixture model' and fitted['rms_residual'] < 0.1
    assert [peak['position'] for peak in fitted['peaks']] == pytest.approx([50.0, 110.0, 160.0], abs=0.05)
    assert [peak['width'] for peak in fitted['peaks']] == pytest.approx([6.0, 10.0, 4.0], rel=0.02)
    assert [peak['height'] for peak in fitted['peaks']] == pytest.approx([100.0, 60.0, 30.0], rel=0.02)
    assert fitted['peaks'][0]['area'] == pytest.approx(100.0 * 6.0 * 1.0645, rel=0.02)
    assert set(fitted['curves']) == {'fit', 'residual'}


def test_a_lorentzian_and_a_voigt_are_fitted_from_given_starts(tmp_path):
    x = np.arange(0.0, 200.0, 0.25)
    lorentzian = 80.0 / (1.0 + ((x - 70.0) / 4.0) ** 2)  # FWHM 8
    path = save_spectrum(tmp_path, 'mixed.csv', x, lorentzian + gaussian(x, 140.0, 12.0, 40.0))
    start = [{'type': 'Lorentzian', 'position': 68.0, 'width': 6.0, 'amplitude': 60.0},
             {'type': 'Gaussian', 'position': 143.0, 'width': 9.0, 'amplitude': 30.0}]
    fitted = ok(analysis.fit_spectrum_peaks(path, start))
    first, second = fitted['peaks']
    assert fitted['method'] == 'scipy least squares' and fitted['rms_residual'] < 0.05
    assert (first['type'], second['type']) == ('Lorentzian', 'Gaussian')
    assert (first['position'], first['width'], first['height']) == pytest.approx((70.0, 8.0, 80.0), rel=0.01)
    assert (second['position'], second['width'], second['height']) == pytest.approx((140.0, 12.0, 40.0), rel=0.01)
    assert ok(analysis.fit_spectrum_peaks(path, [{'type': 'Voigt', 'position': 70, 'width': 8, 'amplitude': 80}],
                                          start_ev=40, end_ev=100))['peaks'][0]['position'] == pytest.approx(70.0, abs=0.2)
    refused(analysis.fit_spectrum_peaks(path, [{'type': 'Sinc', 'position': 70, 'width': 8, 'amplitude': 1}]), 'bad_argument')
    refused(analysis.fit_spectrum_peaks(path, [{'type': 'Gaussian', 'position': 70, 'width': 0, 'amplitude': 1}]), 'bad_argument')


def test_spectrum_tools_refuse_an_image_and_image_tools_refuse_a_spectrum():
    refused(analysis.find_spectrum_peaks(IMAGE), 'not_a_spectrum')
    refused(analysis.image_fft(LOW_LOSS), 'not_an_image')
    refused(analysis.find_atoms(LOW_LOSS), 'not_an_image')


def test_a_spectral_image_is_summed_over_its_pixels_and_says_so():
    found = ok(analysis.eds_find_peaks(f'{MAP}#SuperX', number_of_peaks=10))
    assert found['summed_spectra'] == 215 * 178 and found['peaks']


# ------------------------------------------------------------------ images

def lattice(size=128, period=8.0):
    rows, columns = np.mgrid[:size, :size]
    return 100.0 + 50.0 * np.cos(2 * np.pi * rows / period) * np.cos(2 * np.pi * columns / period)


def png_size(png):
    from PIL import Image
    import io
    return Image.open(io.BytesIO(png)).size


def test_the_fft_of_a_lattice_is_a_png_with_its_reciprocal_scale(tmp_path):
    result = ok(analysis.image_fft(save_array(tmp_path, 'lattice.npy', lattice())))
    assert (result['width'], result['height']) == (128, 128) and result['reciprocal_pixel'] == pytest.approx([1 / 128, 1 / 128])
    assert result['_png'][:8] == b'\x89PNG\r\n\x1a\n' and png_size(result['_png']) == (128, 128)
    big = ok(analysis.image_fft(IMAGE, max_side=256))
    assert (big['width'], big['height']) == (2048, 2048) and max(png_size(big['_png'])) <= 256


def test_the_fft_comes_as_numbers_for_a_viewer_that_sets_its_own_contrast(tmp_path):
    path = save_array(tmp_path, 'lattice.npy', lattice())
    numbers = ok(analysis.image_fft(path, include_array=True))
    assert '_png' not in numbers and numbers['stride'] == 1 and (numbers['width'], numbers['height']) == (128, 128)
    magnitude = np.frombuffer(base64.b64decode(numbers['magnitude_base64']), dtype='<f4').reshape(128, 128)
    assert np.unravel_index(int(np.argmax(magnitude)), magnitude.shape) == (64, 64)
    assert magnitude.min() >= 0
    strided = ok(analysis.image_fft(IMAGE, max_side=256, include_array=True))
    assert strided['full_shape'] == [2048, 2048] and strided['stride'] == 8 and (strided['width'], strided['height']) == (256, 256)
    assert len(base64.b64decode(strided['magnitude_base64'])) == 4 * 256 * 256


def test_registration_recovers_a_known_shift(tmp_path):
    rng = np.random.default_rng(3)
    scene = ndimage.gaussian_filter(rng.normal(size=(160, 160)), 3)
    reference = save_array(tmp_path, 'reference.npy', scene[20:148, 20:148])
    moving = save_array(tmp_path, 'moving.npy', scene[15:143, 27:155])  # the same scene seen 5 rows up, 7 columns right
    shift = ok(analysis.register_images(reference, moving))
    assert (shift['shift_axis0_px'], shift['shift_axis1_px']) == pytest.approx((-5.0, 7.0), abs=0.2)
    assert shift['confidence'] > 0.5
    same = ok(analysis.register_images(reference, reference))
    assert (same['shift_axis0_px'], same['shift_axis1_px']) == (0.0, 0.0) and same['confidence'] == pytest.approx(1.0, abs=1e-3)
    refused(analysis.register_images(reference, save_array(tmp_path, 'small.npy', scene[:64, :64])), 'shape_mismatch')


def atom_image(positions, size=96, sigma=2.0):
    rows, columns = np.mgrid[:size, :size]
    image = np.random.default_rng(5).normal(0.0, 1.0, (size, size))
    for row, column in positions:
        image += 100.0 * np.exp(-((rows - row) ** 2 + (columns - column) ** 2) / (2 * sigma ** 2))
    return image


def test_atoms_are_found_and_refined_to_where_they_were_put(tmp_path):
    truth = [(20.3, 22.7), (20.6, 60.2), (55.4, 40.1), (75.8, 70.5)]
    path = save_array(tmp_path, 'atoms.npy', atom_image(truth))
    found = ok(analysis.find_atoms(path, atom_size=6.0, threshold=0.1))
    assert found['total'] == 4 and found['pixel_size'] == 1.0
    for atom in found['atoms']:  # the blob finder is good to a pixel
        assert min(np.hypot(atom['x'] - row, atom['y'] - column) for row, column in truth) < 1.0
    refined = ok(analysis.refine_atoms(path, found['atoms'], radius_px=5))
    assert refined['refined'] == 4 and refined['total'] == 4
    for atom in refined['atoms']:  # the Gaussian fit is good to a twentieth of one
        assert atom['inside'] and atom['refined']
        assert min(np.hypot(atom['x'] - row, atom['y'] - column) for row, column in truth) < 0.05
        assert atom['width'] == pytest.approx(2.0, rel=0.05)
    from_pairs = ok(analysis.refine_atoms(path, [[20, 23], [55, 40]], radius_px=5))
    assert from_pairs['refined'] == 2 and from_pairs['atoms'][0]['x'] == pytest.approx(20.3, abs=0.05)
    edge = ok(analysis.refine_atoms(path, [{'x': 1, 'y': 1}], radius_px=5))['atoms'][0]
    assert not edge['inside'] and not edge['refined'] and (edge['x'], edge['y']) == (1.0, 1.0)
    refused(analysis.refine_atoms(path, []), 'bad_argument')
    refused(analysis.find_atoms(path, atom_size=0), 'bad_argument')


def test_atoms_are_found_in_a_real_image():
    found = ok(analysis.find_atoms(IMAGE, atom_size=0.1))
    assert found['total'] > 50 and found['units'] == 'nm' and found['pixel_size'] == pytest.approx(0.0389, abs=0.001)


# ------------------------------------------------------------------ probe

def test_an_unaberrated_probe_is_close_to_the_diffraction_limit():
    probe = ok(analysis.simulate_probe(200, 30, include_image=True))
    assert probe['wavelength_nm'] == pytest.approx(0.002508, abs=1e-5)
    assert probe['diffraction_limit_nm'] == pytest.approx(0.61 * 0.002508 / 0.030, rel=1e-3)
    assert probe['half_intensity_radius_nm'] < probe['diffraction_limit_nm']
    assert probe['_png'][:4] == b'\x89PNG' and png_size(probe['_png']) == (256, 256)
    assert '_png' not in ok(analysis.simulate_probe(200, 30))


def test_aberrations_spread_the_probe():
    sharp = ok(analysis.simulate_probe(200, 30))
    defocused = ok(analysis.simulate_probe(200, 30, defocus_nm=-20))
    spherical = ok(analysis.simulate_probe(200, 30, cs_mm=1.0))
    astigmatic = ok(analysis.simulate_probe(200, 30, aberrations={'C12a': 20.0}))
    for blurred in (defocused, spherical, astigmatic):
        assert blurred['half_intensity_radius_nm'] > 2 * sharp['half_intensity_radius_nm']
        assert blurred['peak_intensity_fraction'] < sharp['peak_intensity_fraction']
    refused(analysis.simulate_probe(0, 30), 'bad_argument')
    refused(analysis.simulate_probe(200, 30, aberrations={'defocus': 1.0}), 'bad_argument')
    refused(analysis.simulate_probe(200, 30, size_px=8), 'bad_argument')
