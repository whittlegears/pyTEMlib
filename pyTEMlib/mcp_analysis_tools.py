"""
MCP tool wrappers for the general-purpose analysis tools

Registers agent-callable tools on a FastMCP server for image metrics,
segmentation, alignment, sparse reconstruction / active sampling,
diffraction ring analysis, 4D-STEM virtual imaging and MAPED averaging,
and measurement-driven optimisation.

Agents cannot pass large arrays, so array inputs are file paths (.npy, image
files, or any file pyTEMlib.file_tools.open_file reads, with an optional
dataset key) and large array outputs are written to .npy files whose paths
are returned. Every tool returns a JSON string.

part of pyTEMlib
"""
import json
import os

import numpy as np
from PIL import Image

from . import optimization_tools
from .diffraction_tools import four_d_stem, ring_analysis
from .image_tools import image_alignment, image_metrics, image_segmentation, sparse_reconstruction

image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def _json(payload):
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'{type(value).__name__} is not JSON serialisable')
    return json.dumps(payload, default=default)


def load_array(filename, dataset_key=None, loaded_datasets=None):
    """Read an array from .npy, a common image file, or a pyTEMlib-readable file.

    Returns
    -------
    data: numpy.ndarray
    pixel_size: float or None, from the first spatial / reciprocal dimension of a sidpy dataset
    units: str or None
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'file not found: {filename}')
    extension = os.path.splitext(filename)[1].lower()
    if extension == '.npy':
        return np.load(filename, allow_pickle=False), None, None
    if extension in image_extensions:
        with Image.open(filename) as source:
            frames = []
            for index in range(getattr(source, 'n_frames', 1)):
                source.seek(index)
                frames.append(np.asarray(source.convert('F') if source.mode not in ('F', 'I;16', 'I') else source,
                                         dtype=np.float64))
        return (frames[0] if len(frames) == 1 else np.stack(frames)), None, None
    if loaded_datasets is not None and filename in loaded_datasets:
        datasets = loaded_datasets[filename]
    else:
        from . import file_tools
        datasets = file_tools.open_file(filename)
        if loaded_datasets is not None:
            loaded_datasets[filename] = datasets
    if dataset_key is None:
        dataset_key = next(iter(datasets))
    if dataset_key not in datasets:
        raise KeyError(f'dataset {dataset_key!r} not found; available {list(datasets)}')
    dataset = datasets[dataset_key]
    pixel_size, units = None, None
    for axis in range(dataset.ndim):
        dimension = dataset._axes[axis]
        if str(dimension.dimension_type).split('.')[-1].upper() in ('SPATIAL', 'RECIPROCAL') and len(dimension) > 1:
            pixel_size = float(dimension.values[1] - dimension.values[0])
            units = dimension.units
            break
    return np.asarray(dataset), pixel_size, units


def _output_path(filename, suffix, output_filename):
    if output_filename:
        return output_filename
    return f'{os.path.splitext(filename)[0]}_{suffix}.npy'


def register(mcp, loaded_datasets=None):
    """Register all analysis tools on a FastMCP server instance."""

    @mcp.tool()
    def compute_image_metrics(filename: str, dataset_key: str = None, metrics: list = None) -> str:
        """Focus / quality metrics of an image or of every frame of an image stack.

        Works for any 2D image (HAADF, BF, camera, diffraction, AFM channel).
        metrics: subset of contrast, entropy, high_frequency_power, edge_sharpness (all when empty).
        Larger is better for every metric, so any can drive an optimisation.
        """
        data, _, _ = load_array(filename, dataset_key, loaded_datasets)
        result = {'metrics': image_metrics.image_quality_metrics(data, metrics or None)}
        if data.ndim == 2:
            result['statistics'] = image_metrics.image_statistics(data)
        return _json(result)

    @mcp.tool()
    def segment_image(filename: str, dataset_key: str = None, method: str = 'otsu', value: float = None,
                      polarity: str = 'above', smooth_sigma: float = 0.8, opening_size: int = 2,
                      min_area_px: int = None, pixel_size: float = None, max_regions: int = 50,
                      output_filename: str = None) -> str:
        """Threshold-segment a 2D image into regions and report per-region shape statistics.

        method: otsu | percentile (value 0-100) | absolute (value in image units).
        polarity: above (bright features) | below (dark features).
        pixel_size: physical pixel size; read from the dataset when omitted, else 1 (pixels).
        The label image is written to a .npy file whose path is returned.
        """
        data, file_pixel_size, units = load_array(filename, dataset_key, loaded_datasets)
        if data.ndim != 2:
            raise ValueError(f'segment_image needs a 2D image, got shape {data.shape}')
        labels, threshold = image_segmentation.segment_particles(data, method, value, polarity, smooth_sigma,
                                                                 opening_size, min_area_px)
        size = pixel_size or file_pixel_size or 1.0
        statistics = image_segmentation.particle_statistics(labels, data, size)
        path = _output_path(filename, 'labels', output_filename)
        np.save(path, labels)
        return _json({'threshold': threshold, 'n_regions': len(statistics),
                      'coverage': float(np.mean(labels > 0)),
                      'pixel_size': size,
                      'length_units': 'as given' if pixel_size else (units if file_pixel_size else 'pixels'),
                      'labels_file': path, 'regions': statistics[:max_regions],
                      'regions_truncated': len(statistics) > max_regions})

    @mcp.tool()
    def align_image_series(filename: str, dataset_key: str = None, reference_index: int = 0,
                           max_shift: float = None, cumulative: bool = False,
                           output_filename: str = None) -> str:
        """Register every frame of an image stack (frame, row, column) by sub-pixel cross-correlation.

        Returns per-frame (row, column) shifts in pixels and writes the aligned average to .npy.
        cumulative: register each frame to the previous one (for tilt series / slowly changing contrast).
        """
        data, _, _ = load_array(filename, dataset_key, loaded_datasets)
        if data.ndim != 3:
            raise ValueError(f'align_image_series needs a (frame, row, column) stack, got {data.shape}')
        shifts = image_alignment.align_image_stack(data, reference_index, max_shift, cumulative=cumulative)
        aligned, _ = image_alignment.apply_shifts(data, shifts)
        path = _output_path(filename, 'aligned_average', output_filename)
        np.save(path, aligned.mean(axis=0))
        return _json({'shifts_px': shifts, 'average_file': path, 'average_shape': aligned.shape[1:]})

    @mcp.tool()
    def measure_image_shift(reference_filename: str, moving_filename: str, max_shift: float = None,
                            pixel_size: float = None) -> str:
        """Sub-pixel (row, column) shift that registers the moving image onto the reference.

        With pixel_size, also returns the shift in physical units for drift / image-shift correction.
        """
        reference, reference_pixel, units = load_array(reference_filename, None, loaded_datasets)
        moving, _, _ = load_array(moving_filename, None, loaded_datasets)
        shift = image_alignment.cross_correlation_shift(reference, moving, max_shift)
        result = {'shift_px': shift}
        size = pixel_size or reference_pixel
        if size:
            result['shift_physical'] = shift * size
            result['units'] = units if pixel_size is None else 'as given'
        return _json(result)

    @mcp.tool()
    def plan_sampling_mask(rows: int, columns: int, fraction: float, method: str = 'random', seed: int = 0,
                           preview_filename: str = None, output_filename: str = None) -> str:
        """Choose which scan positions to measure for a sparse / compressed-sensing acquisition.

        method: random | grid | preview-guided (needs preview_filename, e.g. a fast low-dose scan).
        Writes the boolean mask to .npy and returns its path and the number of positions.
        """
        preview = None if preview_filename is None else load_array(preview_filename, None, loaded_datasets)[0]
        mask = sparse_reconstruction.make_sampling_mask((rows, columns), fraction, method, seed, preview)
        path = output_filename or os.path.abspath(f'sampling_mask_{rows}x{columns}_{method}.npy')
        np.save(path, mask)
        return _json({'mask_file': path, 'measured_positions': int(mask.sum()), 'fraction': float(mask.mean())})

    @mcp.tool()
    def reconstruct_sparse_data(filename: str, mask_filename: str, method: str = 'dct',
                                sample_axes: list = None, iterations: int = 40, sparsity_fraction: float = 0.08,
                                tv_step: float = 0.08, loss: str = 'l2', dataset_key: str = None,
                                output_filename: str = None) -> str:
        """Fill in unmeasured positions of sparsely sampled data (image, spectrum image or 4D-STEM).

        mask_filename: .npy boolean mask, True where measured, shaped like the two sample_axes.
        method: nearest | dct | tv. loss: l2 | l1 | huber (l1 / huber resist outliers).
        sample_axes: the two sampled axes, default [0, 1].
        """
        data, _, _ = load_array(filename, dataset_key, loaded_datasets)
        mask = np.load(mask_filename, allow_pickle=False).astype(bool)
        axes = tuple(sample_axes or (0, 1))
        estimate = sparse_reconstruction.sparse_reconstruct(data, mask, method, axes, iterations,
                                                            sparsity_fraction, tv_step, loss)
        path = _output_path(filename, f'{method}_reconstruction', output_filename)
        np.save(path, estimate)
        return _json({'reconstruction_file': path, 'shape': estimate.shape,
                      'measured_fraction': float(mask.mean())})

    @mcp.tool()
    def suggest_next_measurements(rows: int, columns: int, measurements: list, count: int = 1,
                                  min_separation: float = 1.0, value_bounds: list = None,
                                  region_mask_filename: str = None) -> str:
        """Active sampling of a scalar map: where to measure next given the values measured so far.

        measurements: list of [row, column, value] (e.g. a lattice parameter, thickness or intensity
        per probe position). Returns the next positions and a summary of the current reconstruction.
        The score is a heuristic ranking (model disagreement + distance to data), not a confidence.
        """
        values = np.zeros((rows, columns))
        measured = np.zeros((rows, columns), dtype=bool)
        for row, column, value in measurements:
            values[int(row), int(column)] = float(value)
            measured[int(row), int(column)] = True
        region = None if region_mask_filename is None else np.load(region_mask_filename).astype(bool)
        reconstruction, uncertainty = sparse_reconstruction.reconstruct_with_uncertainty(
            values, measured, region, None if value_bounds is None else tuple(value_bounds))
        locations = sparse_reconstruction.next_measurement_locations(uncertainty, count, min_separation)
        return _json({'next_positions': [list(location) for location in locations],
                      'measured': int(measured.sum()),
                      'reconstruction_mean': float(np.nanmean(reconstruction)),
                      'reconstruction_range': [float(np.nanmin(reconstruction)), float(np.nanmax(reconstruction))],
                      'mean_uncertainty': float(np.nanmean(uncertainty))})

    @mcp.tool()
    def analyze_diffraction_rings(filename: str, pixel_size_mrad: float, acceleration_voltage: float = 200e3,
                                  dataset_key: str = None, hkl: list = None, deconvolve: bool = True,
                                  max_angle_mrad: float = None, max_rings: int = None,
                                  prominence_fraction: float = 0.03) -> str:
        """Direct beam, diffraction rings, d-spacings and (cubic) lattice parameters of a 2D pattern.

        pixel_size_mrad: angular size of a detector pixel (full scattering angle).
        hkl: reflections assigned to the first rings in order, e.g. [[1,1,1],[2,0,0]]; for each a cubic
        lattice parameter is returned. deconvolve: Richardson-Lucy with the direct beam as PSF
        (use for convergent-beam / nano-diffraction patterns).
        """
        data, _, _ = load_array(filename, dataset_key, loaded_datasets)
        if data.ndim != 2:
            raise ValueError(f'analyze_diffraction_rings needs a 2D pattern, got {data.shape}')
        result = ring_analysis.analyze_ring_pattern(data, pixel_size_mrad, acceleration_voltage, max_angle_mrad,
                                                    deconvolve, prominence_fraction=prominence_fraction,
                                                    max_rings=max_rings, hkl=hkl)
        result.pop('radial_profile')
        result.pop('radial_angle_mrad')
        return _json(result)

    @mcp.tool()
    def four_d_stem_virtual_image(filename: str, inner_radius: float, outer_radius: float,
                                  center: list = None, dataset_key: str = None,
                                  output_filename: str = None) -> str:
        """Virtual bright-field (inner_radius 0) or annular dark-field image of a 4D-STEM cube.

        The cube is (scan_row, scan_column, k_row, k_column); radii are in diffraction pixels.
        Writes the image to .npy and returns its path plus the image's quality metrics.
        """
        data, _, _ = load_array(filename, dataset_key, loaded_datasets)
        image = four_d_stem.virtual_image(data, inner_radius, outer_radius, center)
        path = _output_path(filename, f'virtual_{inner_radius:g}_{outer_radius:g}', output_filename)
        np.save(path, image)
        return _json({'image_file': path, 'shape': image.shape,
                      'statistics': image_metrics.image_statistics(image)})

    @mcp.tool()
    def maped_average_cubes(filenames: list, dark_field_inner: float = 0.15, dark_field_outer: float = 0.45,
                            output_filename: str = None) -> str:
        """Align 4D-STEM cubes taken at different beam tilts (reciprocal then real space) and average (MAPED).

        dark_field_inner / _outer: virtual dark-field radii as fractions of the pattern size, used for
        real-space alignment. Writes the averaged cube to .npy.
        """
        cubes = [load_array(name, None, loaded_datasets)[0] for name in filenames]
        result = four_d_stem.maped_average(cubes, (dark_field_inner, dark_field_outer))
        path = _output_path(filenames[0], 'maped_average', output_filename)
        np.save(path, result['average'])
        return _json({'average_file': path, 'average_shape': result['average'].shape,
                      'reciprocal_shifts_px': result['reciprocal_shifts'],
                      'real_shifts_px': result['real_shifts']})

    @mcp.tool()
    def propose_next_setting(points: list, scores: list, trust_radius: list, bounds: list = None,
                             maximize: bool = True) -> str:
        """Next instrument setting to try in a measurement-driven optimisation (focus, stigmation, stage z...).

        points: settings measured so far, one list per measurement (any number of parameters).
        scores: measured metric for each point. trust_radius: search radius per parameter around the best
        point. bounds: [[low, high], ...] hard limits per parameter. Fits a quadratic response surface and
        returns the best predicted unmeasured point, or null when the trust region is exhausted.
        """
        proposal = optimization_tools.propose_next_point(np.asarray(points, dtype=float), scores,
                                                         np.asarray(trust_radius, dtype=float), bounds, maximize)
        best_index = int(np.argmax(scores) if maximize else np.argmin(scores))
        return _json({'next_point': None if proposal is None else proposal,
                      'best_measured_point': points[best_index], 'best_measured_score': scores[best_index]})

    @mcp.tool()
    def fit_parabola_vertex(x: list, y: list) -> str:
        """Vertex of a quadratic fit, e.g. best defocus from a defocus-contrast series."""
        return _json(optimization_tools.parabola_vertex(x, y))

    @mcp.tool()
    def fit_linear_trend(x: list, y: list, confidence: float = 0.95, smoothstep_x: bool = False) -> str:
        """Slope of y versus x with a Student-t confidence interval.

        smoothstep_x: fit against smoothstep(x) for centre-to-edge profiles with x in [0, 1].
        """
        transform = optimization_tools.smoothstep if smoothstep_x else None
        return _json(optimization_tools.linear_trend(x, y, confidence, transform))

    return ['compute_image_metrics', 'segment_image', 'align_image_series', 'measure_image_shift',
            'plan_sampling_mask', 'reconstruct_sparse_data', 'suggest_next_measurements',
            'analyze_diffraction_rings', 'four_d_stem_virtual_image', 'maped_average_cubes',
            'propose_next_setting', 'fit_parabola_vertex', 'fit_linear_trend']
