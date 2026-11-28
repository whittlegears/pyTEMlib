"""EDS k-factor tables and the Dirac GOS file, read from the files pyTEMlib ships or keeps in its config directory.

A test whose file is not there skips and names the file.
"""

import csv
import glob
import os

import pytest

from pyTEMlib import config_dir
from pyTEMlib.eds_tools import eds_xsections

THERMO_CSV = 'k-factors-Spectra300UTK200keV.csv'
THERMO_JSON = 'k_factors_Thermo_200keV.json'
DIRAC_GOS = 'Dirac_GOS.gosh'


def shipped(name):
    """The path of a file pyTEMlib ships (pyTEMlib/data) or keeps in its config directory; skip when neither has it."""
    for folder in (config_dir.origin_path, config_dir.config_path):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    pytest.skip(f'{name} is in neither {config_dir.origin_path} nor {config_dir.config_path}')


def csv_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle, delimiter=',', quotechar='|'))
    column = rows[0].index('K-factor')
    return [(row[0], row[column]) for row in rows[1:] if row[0].count('-') == 1 and row[0].split('-')[0]]


def test_thermo_csv_k_factors_are_the_file_values():
    path = shipped(THERMO_CSV)
    k_factors, metadata = eds_xsections.read_csv_k_factors(path, reduced=False)
    rows = csv_rows(path)
    assert rows
    for identifier, value in rows:
        element, line = identifier.split('-')
        assert k_factors[element][line] == value
    assert metadata['microscope'] == 'ThermoFisher' and metadata['source_file'] == path
    assert metadata['name'] == THERMO_JSON


def test_reduced_thermo_k_factors_keep_only_the_first_line_of_each_family():
    k_factors, _ = eds_xsections.read_csv_k_factors(shipped(THERMO_CSV), reduced=True)
    lines = [line for table in k_factors.values() for line in table]
    assert lines and all(line.endswith('1') for line in lines)


def test_the_shipped_k_factor_tables_are_in_the_config_directory():
    shipped_tables = {os.path.basename(path) for path in glob.glob(os.path.join(config_dir.origin_path, 'k_factors*'))}
    if not shipped_tables:
        pytest.skip(f'no k_factors file in {config_dir.origin_path}')
    assert shipped_tables <= set(eds_xsections.get_k_factor_files())


def test_a_missing_esl_file_reads_as_none(tmp_path):
    k_factors, name = eds_xsections.read_esl_k_factors(str(tmp_path / 'missing.esl'))
    assert k_factors is None and name == 'k_factors_Bruker_15keV.json'


def test_bruker_esl_k_factors_name_the_primary_energy():
    found = [path for folder in (config_dir.origin_path, config_dir.config_path)
             for path in glob.glob(os.path.join(folder, '*.esl'))]
    if not found:
        pytest.skip(f'no Bruker .esl k-factor file in {config_dir.origin_path} or {config_dir.config_path}')
    k_factors, metadata = eds_xsections.read_esl_k_factors(found[0])
    assert k_factors and metadata['microscope'] == 'Bruker'
    assert metadata['name'] == f"k_factors_Bruker_{metadata['acceleration_voltage']}keV.json"


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="write_k_factors passes encoding='utf-8' to json.dump, which does not take it")
def test_a_converted_csv_is_written_to_the_config_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(eds_xsections, 'config_path', str(tmp_path))
    eds_xsections.convert_k_factor_file(shipped(THERMO_CSV), new_name='converted.json')
    assert (tmp_path / 'converted.json').is_file()


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='read_k_factors unpacks the JSON object into two names, which takes its first two keys')
def test_the_shipped_thermo_json_reads_as_the_csv_table():
    table, _ = eds_xsections.read_k_factors(shipped(THERMO_JSON))
    expected, _ = eds_xsections.read_csv_k_factors(shipped(THERMO_CSV), reduced=True)
    assert {k: v for k, v in table.items() if k} == {k: v for k, v in expected.items() if k}


def test_the_dirac_gos_file_holds_a_table_per_edge():
    h5py = pytest.importorskip('h5py')
    path = shipped(DIRAC_GOS)
    with h5py.File(path, 'r') as gos_file:
        edge = gos_file['Si']['L3']
        gos = edge['data'][:].squeeze().T
        free_energies = edge['free_energies'][:]
        q_axis = edge['q'][:]
        ionization_energy = float(edge['metadata'].attrs['ionization_energy'])
    assert gos.shape == (free_energies.size, q_axis.size)
    assert ionization_energy > 0
