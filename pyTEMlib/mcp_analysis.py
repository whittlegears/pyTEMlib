"""
Analysis functions behind the pyTEMlib MCP server (pyTEMlib/mcp_server.py registers each as a tool).

Every function here is plain Python: it takes a dataset reference and numbers and returns a
JSON-serialisable dict, so it can be tested and used without MCP.

Dataset references
------------------
A ``dataset`` argument is a string: a file path, optionally followed by ``#<key>`` to name one
dataset of a file that holds several (``example_data/EDS-STO.emd#SuperX``). Without a key the
first dataset of the file is used. A host that keeps data elsewhere (asyncroscopy's Tiled
catalogue) adds a resolver with :func:`add_resolver`, and limits which folders are readable with
:func:`set_allowed_roots`; the tools then take that host's references unchanged.

Returns
-------
* a dict of numbers, strings and short lists;
* a refusal is ``{"error": {"code": ..., "message": ...}}`` (raise :class:`AnalysisError`);
* 1-D curves (a fit, a background) only with ``include_curves=True``, as
  ``{"curves": {name: {"x": [...], "y": [...]}}}``: an agent gets numbers, a GUI gets the curves;
* a 2-D result is a PNG: its bytes are in the dict under the key ``_png``, which the server takes
  out and sends as an image content block.

The numerics are pyTEMlib's own (eels_tools, eds_tools, image_tools, probe_tools). Where pyTEMlib
has no function, the docstring of the tool says what is used instead.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import functools
import io
import math
import os
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy
import sidpy
import skimage.registration
from PIL import Image

from . import file_tools, utilities, probe_tools
from . import eds_tools, image_tools
from .eels_tools import zero_loss_tools, low_loss_tools, core_loss_tools, peak_fit_tools
from .eels_tools.eels_database import read_msa

__all__ = ['AnalysisError', 'add_resolver', 'set_allowed_roots', 'loaded_datasets', 'READ_ONLY_TOOLS']

# file name as given -> {key: sidpy.Dataset}; shared with the older tools of mcp_server.py
loaded_datasets: dict[str, dict] = {}
_file_stamp: dict[str, float] = {}
_resolvers: list[Callable[[str], Any]] = []
_allowed_roots: Optional[list[str]] = None

TEXT_EXTENSIONS = ('.csv', '.txt', '.tsv', '.dat', '.xy')
MSA_EXTENSIONS = ('.msa', '.emsa')
SUPPORTED_EXTENSIONS = ('.dm3', '.dm4', '.emd', '.emi', '.hf5', '.ndata', '.h5', '.rto', '.mrc', '.npy'
                        ) + TEXT_EXTENSIONS + MSA_EXTENSIONS
MAX_PNG_SIDE = 1024


class AnalysisError(Exception):
    """A refusal a tool reports as ``{"error": {"code", "message"}}`` instead of a traceback."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def tool(function):
    """Turn AnalysisError (and unexpected exceptions) into the error dict and make the result JSON-safe."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # pyTEMlib prints progress; a stdio server must not
                return _jsonable(function(*args, **kwargs))
        except AnalysisError as error:
            return {'error': {'code': error.code, 'message': error.message}}
        except Exception as error:  # a library failure is a refusal with its reason, not a crash of the server
            return {'error': {'code': 'analysis_failed', 'message': f'{type(error).__name__}: {error}'}}

    return wrapper


def _jsonable(value):
    """numpy scalars and arrays to Python numbers and lists; NaN and infinity to None."""
    if isinstance(value, dict):
        return {str(key): (item if key == '_png' else _jsonable(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def add_resolver(resolver: Callable[[str], Any]) -> None:
    """Add a function ``reference -> sidpy.Dataset | {key: sidpy.Dataset} | None`` tried before files.

    This is how a host with its own data store (a Tiled catalogue) makes its references work in
    every tool here. The resolver is asked for the whole reference ("key#path" included) and, when it does not
    return a single dataset for that, for the part before '#'. Return None for a reference that is not yours, and
    raise AnalysisError for one that is yours but cannot be analysed.
    """
    _resolvers.append(resolver)


def set_allowed_roots(roots: Optional[list[str]]) -> None:
    """Folders files may be read from; None (the default) allows any path the process can read."""
    global _allowed_roots
    _allowed_roots = None if roots is None else [os.path.realpath(root) for root in roots]


def split_reference(reference: str) -> tuple[str, Optional[str]]:
    """'path#key' -> (path, key); a path that exists as written is never split."""
    if not isinstance(reference, str) or not reference.strip():
        raise AnalysisError('bad_reference', 'dataset must be a non-empty string: a file path, optionally with #<key>')
    reference = reference.strip()
    if '#' in reference and not os.path.exists(reference) and reference not in loaded_datasets:
        path, key = reference.rsplit('#', 1)
        return path, (key or None)
    return reference, None


def _check_allowed(path: str) -> None:
    if _allowed_roots is None:
        return
    real = os.path.realpath(path)
    for root in _allowed_roots:
        if real == root or real.startswith(root + os.sep):
            return
    raise AnalysisError('path_not_allowed', f'{path} is outside the folders this server may read')


def _dataset_from_array(array: np.ndarray, title: str, axis_values: Optional[np.ndarray] = None) -> sidpy.Dataset:
    array = np.asarray(array)
    if array.ndim not in (1, 2, 3):
        raise AnalysisError('unsupported_shape', f'an array with {array.ndim} dimensions is not a spectrum, image or stack')
    dataset = sidpy.Dataset.from_array(array.astype(float), title=title)
    if array.ndim == 1:
        dataset.data_type = 'SPECTRUM'
        values = np.arange(array.shape[0], dtype=float) if axis_values is None else np.asarray(axis_values, dtype=float)
        dataset.set_dimension(0, sidpy.Dimension(values, name='energy_scale', units='eV' if axis_values is not None else 'channel',
                                                 quantity='energy', dimension_type='spectral'))
    else:
        offset = 0
        if array.ndim == 3:
            dataset.data_type = 'IMAGE_STACK'
            dataset.set_dimension(0, sidpy.Dimension(np.arange(array.shape[0], dtype=float), name='frame', units='frame',
                                                     quantity='time', dimension_type='temporal'))
            offset = 1
        else:
            dataset.data_type = 'IMAGE'
        for index, name in enumerate(('x', 'y')):
            dataset.set_dimension(index + offset, sidpy.Dimension(np.arange(array.shape[index + offset], dtype=float), name=name,
                                                                  units='pixel', quantity='distance', dimension_type='spatial'))
    return dataset


def _read_text_spectrum(path: str) -> sidpy.Dataset:
    """One or two numeric columns (energy, counts), comma, tab, semicolon or space separated; '#' and text lines skipped."""
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [part for part in line.replace(';', ',').replace('\t', ',').replace(' ', ',').split(',') if part]
            try:
                rows.append([float(part) for part in parts[:2]])
            except ValueError:
                continue
    if len(rows) < 3:
        raise AnalysisError('unreadable_file', f'{path} holds fewer than three numeric rows')
    width = min(len(row) for row in rows)
    table = np.array([row[:width] for row in rows])
    title = os.path.splitext(os.path.basename(path))[0]
    if width == 1:
        return _dataset_from_array(table[:, 0], title)
    return _dataset_from_array(table[:, 1], title, axis_values=table[:, 0])


def _read_msa(path: str) -> sidpy.Dataset:
    """EMSA/MAS spectrum through pyTEMlib.eels_tools.eels_database.read_msa."""
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        parameters = read_msa(handle.read())
    counts = np.asarray(parameters.get('data', parameters.get('spectrum', [])), dtype=float)
    energy = parameters.get('energy_scale', parameters.get('energy'))
    if counts.size < 3:
        raise AnalysisError('unreadable_file', f'{path} holds no spectrum pyTEMlib can read')
    if energy is None:
        offset = float(parameters.get('OFFSET', 0.0))
        per_channel = float(parameters.get('XPERCHAN', 1.0))
        energy = offset + per_channel * np.arange(counts.size)
    return _dataset_from_array(counts, os.path.splitext(os.path.basename(path))[0], axis_values=np.asarray(energy, dtype=float))


def _open_path(path: str) -> dict:
    _check_allowed(path)
    if not os.path.isfile(path):
        raise AnalysisError('file_not_found', f'no file at {path}')
    extension = os.path.splitext(path)[1].lower()
    if extension == '.npy':
        return {'Channel_000': _dataset_from_array(np.load(path, allow_pickle=False), os.path.basename(path)[:-4])}
    if extension in TEXT_EXTENSIONS:
        return {'Channel_000': _read_text_spectrum(path)}
    if extension in MSA_EXTENSIONS:
        return {'Channel_000': _read_msa(path)}
    try:
        datasets = file_tools.open_file(path)
    except NotImplementedError:
        raise AnalysisError('unsupported_format', f'{extension or "no extension"} is not a format pyTEMlib reads '
                            f'({", ".join(SUPPORTED_EXTENSIONS)})') from None
    datasets = {key: value for key, value in (datasets or {}).items() if isinstance(value, sidpy.Dataset)}
    if not datasets:
        raise AnalysisError('unreadable_file', f'pyTEMlib found no dataset in {path}')
    return datasets


def load_datasets(source: str) -> dict:
    """Every dataset behind a reference (without its #key): {key: sidpy.Dataset}. Files are read once per change."""
    for resolver in _resolvers:
        resolved = resolver(source)
        if resolved is not None:
            return resolved if isinstance(resolved, dict) else {'Channel_000': resolved}
    stamp = os.path.getmtime(source) if os.path.isfile(source) else None
    if source in loaded_datasets and (stamp is None or _file_stamp.get(source) == stamp):
        return loaded_datasets[source]
    datasets = _open_path(source)
    loaded_datasets[source] = datasets
    _file_stamp[source] = stamp
    return datasets


def resolve(reference: str) -> sidpy.Dataset:
    """The sidpy dataset a reference names. Treat it as read-only: it is the cached copy.

    A resolver is first asked for the whole reference, key included; one that returns a single sidpy.Dataset for it
    has named it. Otherwise the reference is a source and a key, as for a file.
    """
    for resolver in _resolvers:
        resolved = resolver(reference)
        if isinstance(resolved, sidpy.Dataset):
            return resolved
    source, key = split_reference(reference)
    datasets = load_datasets(source)
    if key is None:
        return next(iter(datasets.values()))
    if key not in datasets:
        raise AnalysisError('unknown_key', f'{source} has no dataset {key!r}; it has {", ".join(datasets)}')
    return datasets[key]


def _curve(x, y) -> dict:
    return {'x': np.asarray(x, dtype=float), 'y': np.asarray(y, dtype=float)}


def _with_curves(result: dict, include_curves: bool, **curves) -> dict:
    if include_curves:
        result['curves'] = {name: _curve(*xy) for name, xy in curves.items()}
    return result


def spectrum_of(reference: str, axis_name: str = 'energy_loss') -> tuple[sidpy.Dataset, dict]:
    """A private 1-D copy of a spectrum (a spectral image is summed over its pixels) with its axis named axis_name.

    pyTEMlib's spectrum functions write their results into metadata, so they get a copy, never the cached dataset.
    Returns the copy and a note dict (what was summed), which the tools merge into their result.
    """
    source = resolve(reference)
    spectral = source.get_spectral_dims()
    notes = {}
    if len(spectral) != 1:
        raise AnalysisError('not_a_spectrum', f'{reference} is a {source.data_type.name} with shape {list(source.shape)}; '
                            'this tool needs a spectrum or a spectral image')
    axis = source._axes[spectral[0]]
    values = np.asarray(source, dtype=float)
    if source.ndim > 1:
        other = tuple(index for index in range(source.ndim) if index != spectral[0])
        notes['summed_spectra'] = int(np.prod([source.shape[index] for index in other]))
        values = values.sum(axis=other)
    energy = np.asarray(axis.values, dtype=float)
    if energy.size < 3:
        raise AnalysisError('not_a_spectrum', f'{reference} has fewer than three channels')
    spectrum = sidpy.Dataset.from_array(values, title=str(source.title))
    spectrum.data_type = 'SPECTRUM'
    spectrum.units = source.units
    spectrum.quantity = source.quantity
    spectrum.set_dimension(0, sidpy.Dimension(energy, name=axis_name, units=axis.units, quantity=axis.quantity,
                                              dimension_type='spectral'))
    spectrum.metadata = copy.deepcopy(dict(source.metadata))
    return spectrum, notes


def image_of(reference: str) -> tuple[sidpy.Dataset, dict]:
    """A private 2-D IMAGE copy with spatial dimensions named x and y (a stack is averaged over its frames)."""
    source = resolve(reference)
    notes = {}
    values = np.asarray(source, dtype=float)
    if source.data_type.name == 'IMAGE_STACK' and source.ndim == 3:
        frame_axes = [index for index in range(3) if source._axes[index].dimension_type.name == 'TEMPORAL']
        frame_axis = frame_axes[0] if frame_axes else 0
        notes['averaged_frames'] = int(source.shape[frame_axis])
        values = values.mean(axis=frame_axis)
        axes = [source._axes[index] for index in range(3) if index != frame_axis]
    elif source.ndim == 2:
        axes = [source._axes[0], source._axes[1]]
    else:
        raise AnalysisError('not_an_image', f'{reference} is a {source.data_type.name} with shape {list(source.shape)}; '
                            'this tool needs an image or an image stack')
    image = sidpy.Dataset.from_array(values, title=str(source.title))
    image.data_type = 'IMAGE'
    image.units = source.units
    image.quantity = source.quantity
    for index, name in enumerate(('x', 'y')):
        image.set_dimension(index, sidpy.Dimension(np.asarray(axes[index].values, dtype=float), name=name,
                                                   units=axes[index].units, quantity=axes[index].quantity,
                                                   dimension_type='spatial'))
    image.metadata = copy.deepcopy(dict(source.metadata))
    return image, notes


def to_png(array: np.ndarray, max_side: int = MAX_PNG_SIDE) -> bytes:
    """An 8-bit greyscale PNG of a 2-D array, scaled between its 0.5 and 99.5 percentiles and strided to max_side."""
    array = np.asarray(array, dtype=float)
    step = max(1, -(-max(array.shape) // max_side))
    array = array[::step, ::step]
    finite = array[np.isfinite(array)]
    low, high = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((np.nan_to_num(array, nan=low) - low) / (high - low), 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8), mode='L').save(buffer, format='PNG')
    return buffer.getvalue()


def _window(energy: np.ndarray, start_ev: float, end_ev: float, what: str = 'window') -> tuple[int, int]:
    """Channel range [start, end) of an energy window, refused when it holds fewer than three channels."""
    if not end_ev > start_ev:
        raise AnalysisError('bad_window', f'the {what} must end above its start ({start_ev} to {end_ev} eV)')
    start, end = int(np.searchsorted(energy, start_ev)), int(np.searchsorted(energy, end_ev))
    if end - start < 3:
        raise AnalysisError('bad_window', f'the {what} {start_ev} to {end_ev} eV covers fewer than three channels of the '
                            f'{energy[0]:.1f} to {energy[-1]:.1f} eV scale')
    return start, end


def _element(element: Union[str, int]) -> tuple[int, dict]:
    try:
        z = utilities.get_z(element.strip() if isinstance(element, str) else element)
    except (ValueError, TypeError):
        z = 0
    table = utilities.get_x_sections(z) if z > 0 else {}
    if not table:
        raise AnalysisError('unknown_element', f'{element!r} is not an element pyTEMlib has tables for')
    return z, table


def _emission_lines(table: dict, energy_min_ev: float, energy_max_ev: float, min_weight: float) -> list[dict]:
    lines = []
    for name, line in table.get('lines', {}).items():
        position, weight = float(line.get('position', 0.0)), float(line.get('weight', 0.0))
        if energy_min_ev <= position <= energy_max_ev and weight >= min_weight:
            lines.append({'name': name, 'family': name[0], 'position_ev': position, 'weight': weight})
    return sorted(lines, key=lambda line: line['position_ev'])


@tool
def xray_emission_lines(element: Union[str, int], energy_min_ev: float = 0.0, energy_max_ev: float = 30000.0,
                        min_weight: float = 0.01) -> dict:
    """Characteristic X-ray emission lines of an element from pyTEMlib's cross-section tables."""
    z, table = _element(element)
    return {'element': table['name'], 'atomic_number': z,
            'lines': _emission_lines(table, energy_min_ev, energy_max_ev, min_weight)}


@tool
def xray_absorption_edges(element: Union[str, int]) -> dict:
    """Ionization (absorption) edge onsets of an element, highest energy first."""
    z, table = _element(element)
    edges = [{'shell': key, 'family': key[0], 'onset_ev': float(value['onset'])}
             for key, value in table.items()
             if len(key) == 2 and key[0] in 'KLMNO' and key[1].isdigit() and isinstance(value, dict) and 'onset' in value]
    return {'element': table['name'], 'atomic_number': z, 'edges': sorted(edges, key=lambda edge: -edge['onset_ev'])}


@tool
def element_reference_data(element: Union[str, int]) -> dict:
    """Atomic weight, nominal density and fluorescence yields of an element."""
    z, table = _element(element)
    return {'symbol': table['name'], 'atomic_number': z, 'atomic_weight': table.get('atomic_weight'),
            'density_g_cm3': table.get('nominal_density'),
            'fluorescence_yield': dict(table.get('fluorescent_yield', {}))}


def _eds_spectrum(reference: str, resolution_ev: Optional[float] = None, start_energy_ev: Optional[float] = None,
                  beam_energy_ev: Optional[float] = None) -> tuple[sidpy.Dataset, np.ndarray, dict]:
    """A spectrum copy with the metadata pyTEMlib.eds_tools needs; the notes say which values were assumed."""
    spectrum, notes = spectrum_of(reference, axis_name='energy_scale')
    energy = spectrum.energy_scale.values
    detector = spectrum.metadata.setdefault('EDS', {}).setdefault('detector', {})
    if resolution_ev is not None:
        detector['energy_resolution'] = float(resolution_ev)
    elif 'energy_resolution' not in detector:
        detector['energy_resolution'] = 136.0
        notes['assumed_resolution_ev'] = 136.0
    start_energy = float(start_energy_ev) if start_energy_ev is not None else float(detector.get('start_energy', 120.0))
    detector['start_energy'] = start_energy
    detector['start_channel'] = int(np.searchsorted(energy, start_energy))
    experiment = spectrum.metadata.setdefault('experiment', {})
    if beam_energy_ev is not None:
        experiment['acceleration_voltage'] = float(beam_energy_ev)
    elif not experiment.get('acceleration_voltage'):
        experiment['acceleration_voltage'] = 200000.0
        notes['assumed_beam_energy_ev'] = 200000.0
    spectrum.metadata['EDS'].pop('elements', None)
    return spectrum, energy, notes


def _families(element_tags: dict) -> dict:
    families = {}
    for family_key, family in element_tags.items():
        if not family_key.endswith('-family') or not isinstance(family, dict):
            continue
        lines = [{'name': name, 'position_ev': float(family[name]['position']), 'weight': float(family[name]['weight'])}
                 for name in family.get('lines', []) if isinstance(family.get(name), dict)]
        families[family_key[0]] = {'main_line': family.get('main'), 'height': family.get('height'),
                                   'lines': sorted(lines, key=lambda line: line['position_ev'])}
    return families


def _described_elements(tags: dict) -> list[dict]:
    return [{'symbol': symbol, 'atomic_number': int(element.get('Z', 0)), 'families': _families(element)}
            for symbol, element in tags.items() if symbol]


@tool
def eds_find_peaks(dataset: str, start_energy_ev: float = 120.0, min_prominence: float = 10.0,
                   number_of_peaks: int = 30, resolution_ev: Optional[float] = None) -> dict:
    """Peaks of an EDS spectrum (pyTEMlib.eds_tools.detect_peaks)."""
    spectrum, energy, notes = _eds_spectrum(dataset, resolution_ev, start_energy_ev)
    if number_of_peaks < 1:
        raise AnalysisError('bad_argument', 'number_of_peaks must be at least 1')
    channels = eds_tools.detect_peaks(spectrum, minimum_number_of_peaks=int(number_of_peaks), prominence=float(min_prominence))
    values = np.asarray(spectrum)
    peaks = [{'position_ev': float(energy[channel]), 'height': float(values[channel]), 'channel': int(channel)}
             for channel in sorted(int(channel) for channel in channels)]
    return {'peaks': peaks, **notes}


@tool
def eds_identify_elements(dataset: str, sensitivity: float = 3.0, resolution_ev: Optional[float] = None) -> dict:
    """Elements in an EDS spectrum (pyTEMlib.eds_tools.get_elements: take the strongest peak, name its element,
    subtract that element's lines, repeat until what is left is below sensitivity * sqrt(maximum))."""
    spectrum, _, notes = _eds_spectrum(dataset, resolution_ev)
    tags = eds_tools.get_elements(spectrum, float(sensitivity))
    return {'elements': _described_elements(tags), **notes}


@tool
def eds_describe_elements(dataset: str, symbols: list[str], resolution_ev: Optional[float] = None) -> dict:
    """The line families of the named elements as they sit in this spectrum (pyTEMlib.eds_tools.get_x_ray_lines):
    for each family its lines below the end of the energy scale and the spectrum height at its main line."""
    spectrum, _, notes = _eds_spectrum(dataset, resolution_ev)
    names = [_element(symbol)[1]['name'] for symbol in symbols]
    if not names:
        raise AnalysisError('bad_argument', 'symbols must name at least one element')
    return {'elements': _described_elements(eds_tools.get_x_ray_lines(spectrum, names)), **notes}


@tool
def eds_element_lines(symbols: list[str], energy_min_ev: float = 0.0, energy_max_ev: float = 30000.0,
                      min_weight: float = 0.01) -> dict:
    """Emission lines of several elements in an energy range, for drawing markers on a spectrum."""
    lines = []
    for symbol in symbols:
        z, table = _element(symbol)
        lines += [{'symbol': table['name'], 'atomic_number': z, **line}
                  for line in _emission_lines(table, energy_min_ev, energy_max_ev, min_weight)]
    return {'lines': sorted(lines, key=lambda line: line['position_ev'])}


@tool
def eds_net_intensity(dataset: str, peak_position_ev: float, window_ev: float) -> dict:
    """Counts in a window around a peak above a straight line through the window's two end channels.

    pyTEMlib has no window integration (it fits a model: use eds_quantify), so this is plain numpy.
    """
    spectrum, energy, notes = _eds_spectrum(dataset)
    start, end = _window(energy, peak_position_ev - window_ev / 2.0, peak_position_ev + window_ev / 2.0)
    counts = np.asarray(spectrum)[start:end]
    background = np.linspace(counts[0], counts[-1], counts.size)
    return {'net_intensity': max(0.0, float((counts - background).sum())), 'gross_intensity': float(counts.sum()),
            'background': float(background.sum()), 'window_start_ev': float(energy[start]),
            'window_end_ev': float(energy[end - 1]), **notes}


@tool
def eds_quantify(dataset: str, elements: Optional[list] = None, beam_energy_ev: Optional[float] = None,
                 resolution_ev: Optional[float] = None, include_curves: bool = False) -> dict:
    """Composition from an EDS spectrum: fit pyTEMlib's line-family model (eds_tools.fit_model), then quantify with
    Bote-Salvat ionization cross-sections (eds_tools.quantify_eds). Without elements they are identified first."""
    spectrum, energy, notes = _eds_spectrum(dataset, resolution_ev, beam_energy_ev=beam_energy_ev)
    chosen = {}
    for entry in (elements or []):
        symbol, family = (entry.get('symbol'), entry.get('family')) if isinstance(entry, dict) else (entry, None)
        chosen[_element(symbol)[1]['name']] = family
    if chosen:
        tags = eds_tools.get_x_ray_lines(spectrum, list(chosen))
    else:
        tags = eds_tools.get_elements(spectrum)
        tags.pop('', None)
        notes['elements_identified'] = list(tags)
    usable = {symbol: tag for symbol, tag in tags.items() if any(key.endswith('-family') for key in tag)}
    missing = sorted(set(tags) - set(usable))
    if missing:
        raise AnalysisError('no_lines_in_range', f'no K, L or M line of {", ".join(missing)} lies on this energy scale '
                            f'({energy[0]:.0f} to {energy[-1]:.0f} eV)')
    if not usable:
        raise AnalysisError('no_elements', 'no element was given and none was found in the spectrum')
    spectrum.metadata['EDS']['elements'] = usable
    for symbol, family in chosen.items():
        if family:
            if f'{family}-family' not in usable[symbol]:
                raise AnalysisError('no_lines_in_range', f'{symbol} has no {family} lines on this energy scale')
            usable[symbol]['symmetry'] = f'{family}-family'
    kilovolts = int(spectrum.metadata['experiment']['acceleration_voltage'] / 1000)
    eds_tools.fit_model(spectrum)
    try:
        eds_tools.quantify_eds(spectrum, mask=[])
    except FileNotFoundError:
        raise AnalysisError('unsupported_beam_energy', f'pyTEMlib has no ionization cross-section table for {kilovolts} kV '
                            '(it ships 60, 100, 200 and 300 kV)') from None
    quantified = []
    model = np.zeros(energy.size)
    for symbol, element in spectrum.metadata['EDS']['elements'].items():
        family = element.get('symmetry')
        quantified.append({'symbol': symbol, 'atomic_number': int(element.get('Z', 0)), 'family': family[0] if family else None,
                           'intensity': element.get(family, {}).get('areal_density'),
                           'cross_section': element.get('cross_section'), 'atomic_percent': element.get('atom%'),
                           'weight_percent': element.get('weight%')})
        for family_key, entry in element.items():
            if family_key.endswith('-family') and isinstance(entry.get('peaks'), np.ndarray):
                model += entry['peaks'] * entry.get('areal_density', 0.0)
    result = {'elements': quantified, 'beam_energy_ev': spectrum.metadata['experiment']['acceleration_voltage'],
              'method': 'Bote-Salvat cross-sections on fitted line-family intensities', **notes}
    return _with_curves(result, include_curves, spectrum=(energy, np.asarray(spectrum)), model=(energy, model))


def _eels_spectrum(reference: str, align: bool = False, optics: bool = False) -> tuple[sidpy.Dataset, np.ndarray, dict]:
    """A spectrum copy, optionally zero-loss aligned. With optics, the beam energy and angles pyTEMlib.eels_tools
    reads from metadata['experiment'] are filled in when the file has none, and the notes say which were assumed."""
    spectrum, notes = spectrum_of(reference, axis_name='energy_loss')
    experiment = spectrum.metadata.setdefault('experiment', {})
    if optics:
        if not experiment.get('acceleration_voltage'):
            experiment['acceleration_voltage'] = 200000.0
            notes['assumed_beam_energy_ev'] = 200000.0
        # the same fallback pyTEMlib.eels_tools.analyse_low_loss uses when a file carries no angles
        if not experiment.get('convergence_angle') and not experiment.get('collection_angle'):
            experiment['convergence_angle'], experiment['collection_angle'] = 10.0, 30.0
            notes['assumed_convergence_angle_mrad'], notes['assumed_collection_angle_mrad'] = 10.0, 30.0
        experiment.setdefault('convergence_angle', 10.0)
        experiment.setdefault('collection_angle', 30.0)
    if align:
        shift = _zero_loss_shift(spectrum)
        aligned = zero_loss_tools.shift_energy(spectrum, np.array([shift]))
        aligned.metadata = spectrum.metadata
        aligned.data_type = 'SPECTRUM'
        spectrum = aligned
        notes['zero_loss_shift_ev'] = shift
    return spectrum, spectrum.energy_loss.values, notes


def _zero_loss_shift(spectrum: sidpy.Dataset) -> float:
    energy = spectrum.energy_loss.values
    if not energy[0] < 0 < energy[-1]:
        raise AnalysisError('no_zero_loss', f'the energy scale ({energy[0]:.1f} to {energy[-1]:.1f} eV) does not include 0 eV, '
                            'so this is not a low-loss spectrum with a zero-loss peak')
    return float(zero_loss_tools.get_zero_loss_energy(spectrum)[0])


@tool
def eels_find_zero_loss_peak(dataset: str) -> dict:
    """Position and width of the zero-loss peak (pyTEMlib zero_loss_tools: a Gaussian fitted to the top of the peak)."""
    spectrum, energy, notes = _eels_spectrum(dataset)
    position = _zero_loss_shift(spectrum)
    values = np.asarray(spectrum)
    channel = int(np.argmax(values))
    half = values >= values[channel] / 2.0
    left, right = channel, channel
    while left > 0 and half[left - 1]:
        left -= 1
    while right < values.size - 1 and half[right + 1]:
        right += 1
    width = max(8, 2 * (right - left + 1))
    fwhm, _ = zero_loss_tools.get_channel_zero(values, energy, width)
    return {'channel': channel, 'position_ev': position, 'fwhm_ev': abs(float(fwhm)), 'window_width': int(width),
            'height': float(values[channel]), **notes}


@tool
def eels_align_zero_loss(dataset: str, include_curves: bool = False) -> dict:
    """Shift the energy scale so the zero-loss peak sits at 0 eV (pyTEMlib zero_loss_tools.align_zero_loss).

    Nothing is stored: the low-loss tools take align_zero_loss=True and do this themselves.
    """
    spectrum, energy, notes = _eels_spectrum(dataset)
    shift = _zero_loss_shift(spectrum)
    aligned = np.asarray(zero_loss_tools.shift_energy(spectrum, np.array([shift])))
    return _with_curves({'shift_ev': shift, 'channel': int(np.argmax(aligned)), **notes}, include_curves, aligned=(energy, aligned))


def _resolution_function(spectrum: sidpy.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """The fitted zero-loss peak and its six parameters (zero_loss_tools.get_resolution_function, the fit
    analyse_low_loss uses), refused when the fit did not land on the peak."""
    values = np.asarray(spectrum)
    with np.errstate(all='ignore'):
        zero_loss = np.asarray(zero_loss_tools.get_resolution_function(spectrum), dtype=float)
    parameters = np.asarray(spectrum.metadata['zero_loss']['fit']['parameters'], dtype=float)
    peak = values.max()
    landed = abs(int(np.argmax(zero_loss)) - int(np.argmax(values))) <= max(3, values.size // 100)
    if not np.all(np.isfinite(zero_loss)) or not 0.5 * peak < zero_loss.max() < 1.5 * peak or not landed:
        raise AnalysisError('zero_loss_fit_failed', 'the product-of-two-Lorentzians fit did not converge on the zero-loss peak '
                            '(pyTEMlib starts it from fixed values that suit an aligned spectrum in counts)')
    return zero_loss, parameters


def _fwhm(energy: np.ndarray, curve: np.ndarray) -> Optional[float]:
    above = np.nonzero(curve >= curve.max() / 2.0)[0]
    return float(energy[above[-1]] - energy[above[0]]) if above.size > 1 else None


@tool
def eels_resolution_function(dataset: str, align_zero_loss: bool = True, include_curves: bool = False) -> dict:
    """The zero-loss peak as a product of two Lorentzians fitted to the spectrum
    (pyTEMlib zero_loss_tools.get_resolution_function)."""
    spectrum, energy, notes = _eels_spectrum(dataset, align=align_zero_loss)
    zero_loss, parameters = _resolution_function(spectrum)
    result = {'parameters': parameters, 'function': 'product of two Lorentzians (centre, amplitude, width) x 2',
              'fwhm_ev': _fwhm(energy, zero_loss), 'zero_loss_integral': float(zero_loss.sum()), **notes}
    return _with_curves(result, include_curves, spectrum=(energy, np.asarray(spectrum)), zero_loss=(energy, zero_loss))


@tool
def eels_subtract_zero_loss(dataset: str, align_zero_loss: bool = True, include_curves: bool = False) -> dict:
    """The spectrum minus its fitted zero-loss peak (see eels_resolution_function)."""
    spectrum, energy, notes = _eels_spectrum(dataset, align=align_zero_loss)
    zero_loss, parameters = _resolution_function(spectrum)
    values = np.asarray(spectrum)
    difference = values - zero_loss
    total, elastic = float(values.sum()), float(zero_loss.sum())
    result = {'parameters': parameters, 'zero_loss_integral': elastic, 'total_integral': total,
              'inelastic_integral': float(difference.sum()),
              't_over_lambda': math.log(total / elastic) if total > elastic > 0 else None, **notes}
    return _with_curves(result, include_curves, zero_loss=(energy, zero_loss), difference=(energy, difference))


@tool
def eels_relative_thickness(dataset: str, zero_loss_start_ev: Optional[float] = None, zero_loss_end_ev: Optional[float] = None,
                            align_zero_loss: bool = True) -> dict:
    """Relative thickness t/lambda = ln(total counts / zero-loss counts) (log-ratio method).

    The zero-loss counts are the sum over the given window, or, without a window, the integral of the fitted
    resolution function. The formula is plain arithmetic; pyTEMlib has no separate function for it.
    """
    spectrum, energy, notes = _eels_spectrum(dataset, align=align_zero_loss)
    values = np.asarray(spectrum)
    if zero_loss_start_ev is None and zero_loss_end_ev is None:
        elastic = float(_resolution_function(spectrum)[0].sum())
        notes['zero_loss_from'] = 'fitted resolution function'
    elif zero_loss_start_ev is None or zero_loss_end_ev is None:
        raise AnalysisError('bad_window', 'give both zero_loss_start_ev and zero_loss_end_ev, or neither')
    else:
        start, end = _window(energy, zero_loss_start_ev, zero_loss_end_ev, 'zero-loss window')
        elastic = float(values[start:end].sum())
        notes['zero_loss_from'] = 'window'
    total = float(values.sum())
    if not total >= elastic > 0:
        raise AnalysisError('no_signal', 'the zero-loss integral is not positive, or is larger than the whole spectrum')
    return {'t_over_lambda': math.log(total / elastic), 'zero_loss_integral': elastic, 'total_integral': total, **notes}


def _power_law(dataset: str, fit_start_ev: float, fit_end_ev: float):
    spectrum, energy, notes = _eels_spectrum(dataset)
    start, end = _window(energy, fit_start_ev, fit_end_ev, 'background fit window')
    if energy[start] <= 0:
        raise AnalysisError('bad_window', 'a power-law background needs a fit window above 0 eV')
    values = np.asarray(spectrum)
    with np.errstate(all='ignore'):
        background, (amplitude, exponent) = core_loss_tools.power_law_background(values, energy, [fit_start_ev, fit_end_ev])
    background = np.where(energy > 0, np.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    noise = float(np.std(values[start:end] - background[start:end]))
    return energy, values, background, {'amplitude': float(amplitude), 'exponent': float(exponent), 'noise_level': noise,
                                        'fit_start_ev': float(fit_start_ev), 'fit_end_ev': float(fit_end_ev), **notes}


@tool
def eels_fit_power_law_background(dataset: str, fit_start_ev: float, fit_end_ev: float, include_curves: bool = False) -> dict:
    """Power-law background A * E^-r fitted in a window before an edge (pyTEMlib core_loss_tools.power_law_background)."""
    energy, _, background, result = _power_law(dataset, fit_start_ev, fit_end_ev)
    return _with_curves(result, include_curves, background=(energy, background))


@tool
def eels_subtract_power_law_background(dataset: str, fit_start_ev: float, fit_end_ev: float, include_curves: bool = False) -> dict:
    """The spectrum minus its fitted power-law background (see eels_fit_power_law_background)."""
    energy, values, background, result = _power_law(dataset, fit_start_ev, fit_end_ev)
    return _with_curves(result, include_curves, background=(energy, background), difference=(energy, values - background))


@tool
def eels_integrate_signal(dataset: str, fit_start_ev: float, fit_end_ev: float, signal_start_ev: float,
                          signal_end_ev: float) -> dict:
    """Edge counts above the power-law background, summed over a signal window that starts at or after the fit window."""
    if fit_end_ev > signal_start_ev:
        raise AnalysisError('bad_window', f'the background fit window ends at {fit_end_ev} eV, inside the signal window '
                            f'that starts at {signal_start_ev} eV: fit the background before the edge')
    energy, values, background, result = _power_law(dataset, fit_start_ev, fit_end_ev)
    start, end = _window(energy, signal_start_ev, signal_end_ev, 'signal window')
    raw, below = float(values[start:end].sum()), float(background[start:end].sum())
    return {**result, 'signal_integral': raw - below, 'background_integral': below, 'raw_integral': raw,
            'signal_start_ev': float(signal_start_ev), 'signal_end_ev': float(signal_end_ev)}


@tool
def eels_quantify(dataset: str, elements: list, fit_start_ev: Optional[float] = None, fit_end_ev: Optional[float] = None,
                  include_curves: bool = False) -> dict:
    """Relative composition from a core-loss spectrum: pyTEMlib's model-based fit (core_loss_tools.fit_dataset) of a
    power-law background plus one X-ray-photoabsorption cross-section per element, over the fit range."""
    spectrum, energy, notes = _eels_spectrum(dataset, optics=True)
    if not elements:
        raise AnalysisError('bad_argument', 'elements must name at least one element')
    for element in elements:
        z, table = _element(element)
        if not core_loss_tools.add_element_to_dataset(spectrum, z):
            raise AnalysisError('no_edge_in_range', f'{table["name"]} has no major edge on this energy scale '
                                f'({energy[0]:.0f} to {energy[-1]:.0f} eV)')
    fit_area = spectrum.metadata['core_loss'].setdefault('fit_area', {})
    if fit_start_ev is not None:
        fit_area['fit_start'] = float(fit_start_ev)
    if fit_end_ev is not None:
        fit_area['fit_end'] = float(fit_end_ev)
    core_loss_tools.fit_dataset(spectrum)
    core_loss = spectrum.metadata['core_loss']
    edges = [edge for key, edge in core_loss['edges'].items() if str(key).isdigit()]
    total = sum(abs(edge['areal_density']) for edge in edges)
    described = [{'element': edge['element'], 'atomic_number': int(edge['z']), 'edge': edge['symmetry'],
                  'onset_ev': float(edge['onset']), 'areal_density': abs(float(edge['areal_density'])),
                  'atomic_percent': 100.0 * abs(edge['areal_density']) / total if total > 0 else None} for edge in edges]
    model = core_loss['model']
    result = {'elements': described, 'fit_start_ev': float(model['fit_area_start']), 'fit_end_ev': float(model['fit_area_end']),
              'background_amplitude': float(model['background-A']), 'background_exponent': -float(model['background-r']),
              'cross_sections': 'XRPA (pyTEMlib), effective collection angle 30 mrad as fixed in fit_dataset', **notes}
    return _with_curves(result, include_curves, spectrum=(energy, np.asarray(spectrum)), model=(energy, model['spectrum']))


def _plasmon(dataset: str, fit_start_ev: float, fit_end_ev: float, align: bool, subtract_zero_loss: bool):
    spectrum, energy, notes = _eels_spectrum(dataset, align=align, optics=True)
    _window(energy, fit_start_ev, fit_end_ev, 'plasmon fit window')
    if fit_start_ev <= 0:
        raise AnalysisError('bad_window', 'the plasmon fit window must start above 0 eV')
    zero_loss = np.zeros(energy.size)
    if subtract_zero_loss:
        zero_loss, _ = _resolution_function(spectrum)
        working = spectrum - zero_loss
        working.metadata = spectrum.metadata
        working.data_type = 'SPECTRUM'
        spectrum = working
    experiment = spectrum.metadata['experiment']
    effective = utilities.effective_collection_angle(energy, experiment['convergence_angle'], experiment['collection_angle'],
                                                     experiment['acceleration_voltage'])
    # fit_plasmon takes the angle of its angular term from metadata['experiment']['convergence_angle']
    experiment['convergence_angle'] = effective
    anglog = low_loss_tools.get_anglog(energy, experiment['acceleration_voltage'], effective)
    with np.errstate(all='ignore'):
        _, parameters = low_loss_tools.fit_plasmon(spectrum, float(fit_start_ev), float(fit_end_ev))
    parameters = np.asarray(parameters, dtype=float)
    fit = low_loss_tools.energy_loss_function(energy, parameters, anglog)
    fit[energy <= 0] = 0.0
    notes['effective_collection_angle_mrad'] = float(effective)
    return spectrum, energy, zero_loss, anglog, parameters, fit, notes


def _drude(parameters: np.ndarray) -> dict:
    """pyTEMlib's Drude parameters by the names its drude() gives them."""
    return {'peak_position_ev': parameters[0], 'peak_width_ev': parameters[1], 'damping_ev': parameters[2],
            'amplitude': parameters[3]}


@tool
def eels_fit_drude_plasmon(dataset: str, fit_start_ev: float, fit_end_ev: float, align_zero_loss: bool = True,
                           subtract_zero_loss: bool = False, include_curves: bool = False) -> dict:
    """Volume plasmon as a Drude energy-loss function fitted in a window (pyTEMlib low_loss_tools.fit_plasmon).

    As in pyTEMlib's analyse_low_loss the fit is to the aligned spectrum; subtract_zero_loss fits what is left
    after the fitted zero-loss peak is removed.
    """
    spectrum, energy, _, _, parameters, fit, notes = _plasmon(dataset, fit_start_ev, fit_end_ev, align_zero_loss, subtract_zero_loss)
    start, end = _window(energy, fit_start_ev, fit_end_ev)
    residual = np.asarray(spectrum)[start:end] - fit[start:end]
    result = {**_drude(parameters), 'fit_start_ev': float(fit_start_ev), 'fit_end_ev': float(fit_end_ev),
              'rms_residual': float(np.sqrt(np.mean(residual ** 2))), **notes}
    return _with_curves(result, include_curves, spectrum=(energy, np.asarray(spectrum)), fit=(energy, fit))


@tool
def eels_fit_multiple_scattering(dataset: str, fit_start_ev: float, fit_end_ev: float, max_energy_ev: float = 55.0,
                                 align_zero_loss: bool = True, include_curves: bool = False) -> dict:
    """Plural plasmon scattering: a Drude plasmon (fitted in the window) convolved with itself by Poisson statistics
    and fitted to the spectrum without its zero-loss peak up to max_energy_ev, which gives t/lambda
    (pyTEMlib low_loss_tools.fit_plasmon then fit_multiple_scattering, the sequence of analyse_low_loss)."""
    if max_energy_ev <= fit_end_ev:
        raise AnalysisError('bad_window', 'max_energy_ev must be above the plasmon fit window')
    spectrum, energy, _, anglog, _, _, notes = _plasmon(dataset, fit_start_ev, fit_end_ev, align_zero_loss, False)
    zero_loss, _ = _resolution_function(spectrum)
    inelastic = spectrum - zero_loss
    inelastic.metadata = spectrum.metadata
    inelastic.data_type = 'SPECTRUM'
    with np.errstate(all='ignore'):
        fit = np.asarray(low_loss_tools.fit_multiple_scattering(inelastic, anglog, end_fit_energy=float(max_energy_ev)), dtype=float)
    parameters = np.asarray(inelastic.metadata['plasmon']['multiple_scattering_fit']['parameters'], dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise AnalysisError('fit_failed', 'the multiple-scattering fit did not converge')
    result = {**_drude(parameters), 'thickness_over_mfp': parameters[4], 'max_energy_ev': float(max_energy_ev), **notes}
    return _with_curves(result, include_curves, spectrum=(energy, np.asarray(inelastic)), fit=(energy, fit),
                        zero_loss=(energy, zero_loss))


PEAK_TYPES = ('Gaussian', 'Lorentzian', 'Voigt')
_SIGMA_PER_FWHM = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _peak_shape(x: np.ndarray, kind: str, position: float, fwhm: float, height: float) -> np.ndarray:
    fwhm = max(abs(fwhm), 1e-12)
    gaussian = np.exp(-0.5 * ((x - position) / (fwhm * _SIGMA_PER_FWHM)) ** 2)
    lorentzian = 1.0 / (1.0 + ((x - position) / (fwhm / 2.0)) ** 2)
    if kind == 'Gaussian':
        return height * gaussian
    if kind == 'Lorentzian':
        return height * lorentzian
    return height * 0.5 * (gaussian + lorentzian)  # pseudo-Voigt with equal parts


def _area(kind: str, fwhm: float, height: float) -> float:
    gaussian = height * abs(fwhm) * _SIGMA_PER_FWHM * math.sqrt(2.0 * math.pi)
    lorentzian = height * math.pi * abs(fwhm) / 2.0
    return {'Gaussian': gaussian, 'Lorentzian': lorentzian}.get(kind, 0.5 * (gaussian + lorentzian))


def _range(energy: np.ndarray, start_ev: Optional[float], end_ev: Optional[float]) -> tuple[int, int]:
    if start_ev is None and end_ev is None:
        return 0, energy.size
    return _window(energy, energy[0] if start_ev is None else start_ev, energy[-1] + 1.0 if end_ev is None else end_ev)


@tool
def find_spectrum_peaks(dataset: str, max_peaks: int = 10, start_ev: Optional[float] = None,
                        end_ev: Optional[float] = None) -> dict:
    """The most prominent peaks of any spectrum (pyTEMlib peak_fit_tools.find_maxima on the smoothed spectrum),
    with height and full width at half prominence (scipy.signal.peak_widths)."""
    spectrum, notes = spectrum_of(dataset)
    energy, values = spectrum.energy_loss.values, np.asarray(spectrum)
    if max_peaks < 1:
        raise AnalysisError('bad_argument', 'max_peaks must be at least 1')
    start, end = _range(energy, start_ev, end_ev)
    channels = np.asarray(peak_fit_tools.find_maxima(values[start:end], int(max_peaks)), dtype=int)
    smoothed = scipy.ndimage.gaussian_filter(values[start:end], sigma=2)
    widths = scipy.signal.peak_widths(smoothed, channels, rel_height=0.5)[0] if channels.size else []
    step = float(np.mean(np.diff(energy)))
    peaks = [{'position': float(energy[start + channel]), 'height': float(values[start + channel]),
              'fwhm': float(width * abs(step)), 'channel': int(start + channel)} for channel, width in zip(channels, widths)]
    return {'peaks': peaks, 'units': spectrum._axes[0].units, **notes}


@tool
def fit_spectrum_peaks(dataset: str, initial: Optional[list] = None, start_ev: Optional[float] = None,
                       end_ev: Optional[float] = None, max_peaks: int = 5, include_curves: bool = False) -> dict:
    """Fit a sum of peaks to a spectrum (or to a range of it).

    initial: [{type: Gaussian | Lorentzian | Voigt, position, width (FWHM), amplitude (height)}]; without it the
    max_peaks most prominent peaks are fitted as Gaussians. Gaussian-only fits use pyTEMlib's Gaussian mixture model
    (peak_fit_tools.fit_gmm); pyTEMlib has no Lorentzian or Voigt peak fit, so a fit with one of those uses
    scipy.optimize.least_squares on the same sum-of-peaks model (Voigt is a pseudo-Voigt, half Gaussian half Lorentzian).
    """
    spectrum, notes = spectrum_of(dataset)
    energy, values = spectrum.energy_loss.values, np.asarray(spectrum)
    start, end = _range(energy, start_ev, end_ev)
    x, y = energy[start:end], values[start:end]
    if not initial:
        found = find_spectrum_peaks(dataset, max_peaks=max_peaks, start_ev=start_ev, end_ev=end_ev)
        if 'error' in found:
            return found
        initial = [{'type': 'Gaussian', 'position': peak['position'], 'width': peak['fwhm'], 'amplitude': peak['height']}
                   for peak in found['peaks']]
        notes['initial_peaks_found'] = len(initial)
    if not initial:
        raise AnalysisError('no_peaks', 'no peak was found to start the fit from; give initial peaks')
    kinds = []
    for peak in initial:
        kind = str(peak.get('type', 'Gaussian')).capitalize()
        if kind not in PEAK_TYPES:
            raise AnalysisError('bad_argument', f'peak type {peak.get("type")!r} is not one of {", ".join(PEAK_TYPES)}')
        if peak.get('width', 0) <= 0:
            raise AnalysisError('bad_argument', 'every initial peak needs a positive width (FWHM)')
        kinds.append(kind)
    if all(kind == 'Gaussian' for kind in kinds):
        # pyTEMlib's gmm takes (centre, amplitude, w) with sigma = 2.3548 * w
        start_parameters = np.ravel([[peak['position'], peak['amplitude'], peak['width'] * _SIGMA_PER_FWHM / 2.3548]
                                     for peak in initial]).astype(float)
        fitted = np.asarray(peak_fit_tools.fit_gmm(x, y, list(start_parameters))).reshape(-1, 3)
        triples = [(row[0], abs(row[2]) * 2.3548 / _SIGMA_PER_FWHM, row[1]) for row in fitted]
        notes['method'] = 'pyTEMlib Gaussian mixture model'
    else:
        start_parameters = np.ravel([[peak['position'], peak['width'], peak['amplitude']] for peak in initial]).astype(float)

        def residuals(parameters):
            return sum(_peak_shape(x, kind, *parameters[3 * index:3 * index + 3]) for index, kind in enumerate(kinds)) - y

        fitted = scipy.optimize.least_squares(residuals, start_parameters, x_scale='jac').x.reshape(-1, 3)
        triples = [(row[0], abs(row[1]), row[2]) for row in fitted]
        notes['method'] = 'scipy least squares'
    model = sum(_peak_shape(energy, kind, *triple) for kind, triple in zip(kinds, triples))
    peaks = [{'type': kind, 'position': float(position), 'width': float(width), 'amplitude': float(height),
              'height': float(height), 'area': _area(kind, width, height)} for kind, (position, width, height) in zip(kinds, triples)]
    residual = values - model
    result = {'peaks': sorted(peaks, key=lambda peak: peak['position']), 'units': spectrum._axes[0].units,
              'rms_residual': float(np.sqrt(np.mean(residual[start:end] ** 2))), **notes}
    return _with_curves(result, include_curves, fit=(energy, model), residual=(energy, residual))


@tool
def image_fft(dataset: str, max_side: int = 512, include_array: bool = False) -> dict:
    """Fourier transform magnitude of an image, log(1 + |F|) with zero frequency at the centre
    (pyTEMlib image_tools.fourier_transform).

    By default the magnitude comes as a PNG. With include_array=True it comes instead as numbers, for a viewer that sets
    its own contrast: 'magnitude_base64' is float32, little-endian, row-major, shaped [height, width], strided so its
    longer side is at most max_side; 'stride' says by how much.
    """
    image, notes = image_of(dataset)
    transform = image_tools.fourier_transform(image)
    magnitude = np.log1p(np.abs(np.asarray(transform)))
    u, v = np.asarray(transform._axes[0].values), np.asarray(transform._axes[1].values)
    side = max(16, min(int(max_side), 2048))
    result = {'reciprocal_pixel': [float(u[1] - u[0]), float(v[1] - v[0])],
              'reciprocal_units': [transform._axes[0].units, transform._axes[1].units],
              'display': 'log(1 + |F|), zero frequency at the centre', **notes}
    if include_array:
        stride = max(1, -(-max(magnitude.shape) // side))
        shown = np.ascontiguousarray(magnitude[::stride, ::stride], dtype='<f4')
        return {**result, 'width': int(shown.shape[1]), 'height': int(shown.shape[0]), 'stride': stride,
                'full_shape': [int(size) for size in magnitude.shape],
                'magnitude_base64': base64.b64encode(shown.tobytes()).decode('ascii')}
    return {**result, 'width': int(magnitude.shape[1]), 'height': int(magnitude.shape[0]), '_png': to_png(magnitude, side)}


@tool
def register_images(reference: str, moving: str, upsample_factor: int = 10) -> dict:
    """The shift, in pixels, that moves `moving` onto `reference`, by cross-correlation in Fourier space.

    pyTEMlib's rigid_registration does this for the frames of a stack with skimage.registration.phase_cross_correlation
    (unnormalised, its default); this calls the same routine on two separate images, with sub-pixel refinement.
    confidence is 1 minus the routine's normalised RMS error between the images after the shift.
    """
    first, notes = image_of(reference)
    second, _ = image_of(moving)
    if first.shape != second.shape:
        raise AnalysisError('shape_mismatch', f'the images differ in shape: {list(first.shape)} and {list(second.shape)}')
    shift, error, _ = skimage.registration.phase_cross_correlation(np.asarray(first), np.asarray(second),
                                                                   upsample_factor=max(1, int(upsample_factor)), normalization=None)
    return {'shift_axis0_px': float(shift[0]), 'shift_axis1_px': float(shift[1]),
            'confidence': float(max(0.0, 1.0 - error)), 'axes': [first._axes[0].name, first._axes[1].name], **notes}


@tool
def find_atoms(dataset: str, atom_size: float = 0.1, threshold: float = 0.0, max_atoms: int = 5000) -> dict:
    """Atom columns of an image by Laplacian-of-Gaussian blob detection (pyTEMlib image_tools.find_atoms).

    atom_size is the visible blob diameter in the units of the image scale (nm); threshold 0 uses the image's RMS contrast.
    """
    image, notes = image_of(dataset)
    if atom_size <= 0:
        raise AnalysisError('bad_argument', 'atom_size must be positive')
    blobs = np.asarray(image_tools.find_atoms(image, atom_size=float(atom_size), threshold=float(threshold)))
    values = np.asarray(image)
    atoms = [{'x': float(row), 'y': float(column), 'sigma': float(sigma),
              'intensity': float(values[int(row), int(column)])} for row, column, sigma in blobs[:max_atoms]]
    return {'atoms': atoms, 'total': int(len(blobs)), 'returned': len(atoms), 'pixel_size': float(np.gradient(image.x.values)[0]),
            'units': image._axes[0].units, 'coordinates': 'pixels along axis 0 (x) and axis 1 (y)', **notes}


@tool
def refine_atoms(dataset: str, atoms: list, radius_px: float = 3.0, max_dist_px: float = 4.0) -> dict:
    """Refine atom positions by fitting a 2-D Gaussian in a disc around each (pyTEMlib image_tools.atom_refine).

    An atom whose fit fails, moves more than max_dist_px, or lies within radius_px of the edge keeps its starting
    pixel and is marked refined: false.
    """
    image, notes = image_of(dataset)
    if not atoms:
        raise AnalysisError('bad_argument', 'atoms must list at least one position')
    if radius_px < 1:
        raise AnalysisError('bad_argument', 'radius_px must be at least 1')
    positions = np.array([[atom['x'], atom['y']] if isinstance(atom, dict) else list(atom)[:2] for atom in atoms], dtype=float)
    with contextlib.redirect_stderr(io.StringIO()), np.errstate(all='ignore'):  # atom_refine draws a progress bar
        refined = image_tools.atom_refine(np.asarray(image), positions, float(radius_px), max_dist=float(max_dist_px))
    result = []
    for index, (x, y) in enumerate(refined['atoms']):
        width = float(refined['gauss_width'][index])
        fitted = width != 0.0
        # atom_refine fits on a grid whose origin is half a pixel off the window's centre pixel (pyTEMlib's own
        # test adds the 0.5 back), so a fitted position is half a pixel low in both directions
        offset = 0.5 if fitted else 0.0
        result.append({'x': float(x) + offset, 'y': float(y) + offset, 'width': abs(width),
                       'amplitude': float(refined['gauss_amplitude'][index]),
                       'volume': float(refined['gauss_volume'][index]), 'inside': refined['inside'][index] == 1,
                       'refined': fitted})
    return {'atoms': result, 'refined': sum(atom['refined'] for atom in result), 'total': len(result), **notes}


@tool
def simulate_probe(voltage_kv: float, aperture_mrad: float, defocus_nm: float = 0.0, cs_mm: float = 0.0,
                   fov_nm: float = 4.0, size_px: int = 256, aberrations: Optional[dict] = None,
                   include_image: bool = False, include_array: bool = False) -> dict:
    """STEM probe intensity from an aberration function and an aperture (pyTEMlib probe_tools.get_probe).

    defocus_nm is C10 and cs_mm is C30; further coefficients (C12a, C21a, C50, ... in nm, pyTEMlib's notation) go in
    aberrations. Chromatic aberration is not included: pyTEMlib's chromatic probe (probe_tools.probe2) does not run.
    include_image adds a PNG; include_array adds the intensity as 'probe_base64', float32, little-endian, row-major,
    [size_px, size_px], for a viewer that sets its own contrast.
    """
    if voltage_kv <= 0 or aperture_mrad <= 0 or fov_nm <= 0:
        raise AnalysisError('bad_argument', 'voltage_kv, aperture_mrad and fov_nm must be positive')
    if not 16 <= size_px <= 2048:
        raise AnalysisError('bad_argument', 'size_px must be between 16 and 2048')
    ab = {'acceleration_voltage': float(voltage_kv) * 1000.0, 'convergence_angle': float(aperture_mrad), 'FOV': float(fov_nm),
          'C10': float(defocus_nm), 'C30': float(cs_mm) * 1e6}
    for name, value in (aberrations or {}).items():
        if not (isinstance(name, str) and name.startswith('C') and isinstance(value, (int, float))):
            raise AnalysisError('bad_argument', f'aberration {name!r} must be a coefficient name such as C12a with a number in nm')
        ab[name] = float(value)
    probe, _, _ = probe_tools.get_probe(ab, int(size_px), int(size_px), verbose=False)
    probe = np.asarray(probe, dtype=float)
    total = probe.sum()
    pixel = fov_nm / size_px
    centre = np.unravel_index(int(np.argmax(probe)), probe.shape)
    rows, columns = np.indices(probe.shape)
    radius = np.hypot(rows - centre[0], columns - centre[1]) * pixel
    order = np.argsort(radius, axis=None)
    enclosed = np.cumsum(probe.ravel()[order]) / total
    wavelength_nm = float(utilities.get_wavelength(ab['acceleration_voltage'], 'nm'))
    result = {'wavelength_nm': wavelength_nm, 'fov_nm': float(fov_nm), 'pixel_nm': pixel, 'size_px': int(size_px),
              'half_intensity_radius_nm': float(radius.ravel()[order][int(np.searchsorted(enclosed, 0.5))]),
              'peak_intensity_fraction': float(probe.max() / total),
              'diffraction_limit_nm': 0.61 * wavelength_nm / (aperture_mrad / 1000.0)}
    if include_image:
        result['_png'] = to_png(probe, 512)
    if include_array:
        result['probe_base64'] = base64.b64encode(np.ascontiguousarray(probe, dtype='<f4').tobytes()).decode('ascii')
    return result


# every tool here only computes; mcp_server.py marks them readOnlyHint
READ_ONLY_TOOLS = (xray_emission_lines, xray_absorption_edges, element_reference_data,
                   eds_find_peaks, eds_identify_elements, eds_describe_elements, eds_element_lines, eds_net_intensity,
                   eds_quantify, eels_find_zero_loss_peak, eels_align_zero_loss, eels_resolution_function,
                   eels_subtract_zero_loss, eels_relative_thickness, eels_fit_power_law_background,
                   eels_subtract_power_law_background, eels_integrate_signal, eels_quantify,
                   eels_fit_drude_plasmon, eels_fit_multiple_scattering, find_spectrum_peaks, fit_spectrum_peaks,
                   image_fft, register_images, find_atoms, refine_atoms, simulate_probe)
