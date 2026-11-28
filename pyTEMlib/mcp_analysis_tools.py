"""
MCP tool wrappers for the general-purpose analysis tools

Registers agent-callable tools on a FastMCP server for image metrics,
segmentation, alignment, sparse reconstruction / active sampling,
diffraction ring analysis, 4D-STEM virtual imaging and MAPED averaging,
and measurement-driven optimisation.

Agents cannot pass large arrays, so every array input is a *source*: either a
file path (.npy, image files, or any file pyTEMlib.file_tools.open_file reads)
or a data key returned by an instrument acquisition tool, which is read from the
Tiled data server over HTTP. Large array outputs are written to .npy files whose
paths are returned. Every tool returns a JSON string.

The Tiled server is configured with ``register(..., tiled_uri=...)`` or the
PYTEMLIB_TILED_URI environment variable; nothing here talks to Tango.

part of pyTEMlib
"""
import json
import os

import numpy as np
from PIL import Image

from . import optimization_tools
from .diffraction_tools import four_d_stem, ring_analysis
from .image_tools import image_alignment, image_metrics, image_mosaic, image_segmentation, sparse_reconstruction

try:
    from .ml_tools import bayesian_optimization
    bayesian_optimization_missing = None
except ImportError as error:
    bayesian_optimization = None
    bayesian_optimization_missing = f"{error}; install with pip install 'pyTEMlib[ml]'"

try:
    from .ml_tools import sam2_segmentation
    sam2_missing = None
except ImportError as error:
    sam2_segmentation = None
    sam2_missing = (f"{error}; install with pip install 'pyTEMlib[ml]' "
                    "'git+https://github.com/facebookresearch/sam2.git' huggingface-hub")

image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def _json(payload):
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'{type(value).__name__} is not JSON serialisable')
    return json.dumps(payload, default=default)


def _read_tiled(uri, key, path=None):
    try:
        from tiled.client import from_uri
    except ImportError as exc:
        raise ImportError("reading data keys needs the Tiled client: pip install 'pyTEMlib[tiled]'") from exc
    try:
        node = from_uri(uri)[key]
    except KeyError as exc:
        raise KeyError(f'data key {key!r} not found on Tiled server {uri}') from exc
    trail = []
    for part in (path or '').strip('/').split('/'):
        if not part:
            continue
        children = list(node.keys())
        if part not in children and len(trail) == 0 and 'image' in children and part in node['image'].keys():
            node = node['image']
            trail.append('image')
        elif part not in children:
            raise KeyError(f'{"/".join([key, *trail])} has no {part!r}; it holds {children}')
        node = node[part]
        trail.append(part)
    while hasattr(node, 'keys'):
        children = list(node.keys())
        preferred = [name for name in ('image', 'stem_data') if name in children]
        if preferred:
            choice = preferred[0]
        elif len(children) == 1:
            choice = children[0]
        else:
            options = ['/'.join([*trail, child]) for child in children]
            raise ValueError(f'data key {key!r} holds several datasets; pass dataset_key, one of {options}')
        node = node[choice]
        trail.append(choice)
    return np.asarray(node.read()), dict(getattr(node, 'metadata', {}) or {}), '/'.join([key, *trail])


def load_array(source, dataset_key=None, loaded_datasets=None, tiled_uri=None):
    """Read an array from a file or from a data key on the Tiled data server.

    Parameters
    ----------
    source: str
        a file path (.npy, common image files, or any file pyTEMlib.file_tools.open_file
        reads), or a data key returned by an acquisition (read from Tiled)
    dataset_key: str or None
        dataset inside a multi-dataset file, or path inside a data key such as
        'image/HAADF'; when None a single dataset (or the 'image' / 'stem_data' group)
        is chosen, and an ambiguous choice raises an error naming the options
    tiled_uri: str or None
        Tiled server; PYTEMLIB_TILED_URI when None

    Returns
    -------
    data: numpy.ndarray
    pixel_size: float or None
    units: str or None
    metadata: dict (Tiled node metadata, or {} for files)
    """
    if not os.path.exists(source):
        uri = tiled_uri or os.environ.get('PYTEMLIB_TILED_URI')
        if not uri:
            raise FileNotFoundError(f'{source!r} is not a file, and no Tiled server is configured to read '
                                    'it as a data key (set PYTEMLIB_TILED_URI)')
        data, metadata, _ = _read_tiled(uri, source, dataset_key)
        if 'pixel_size_mrad' in metadata:
            return data, float(metadata['pixel_size_mrad']), 'mrad', metadata
        return data, None, None, metadata
    extension = os.path.splitext(source)[1].lower()
    if extension == '.npy':
        return np.load(source, allow_pickle=False), None, None, {}
    if extension in image_extensions:
        with Image.open(source) as image:
            frames = []
            for index in range(getattr(image, 'n_frames', 1)):
                image.seek(index)
                frames.append(np.asarray(image.convert('F') if image.mode not in ('F', 'I;16', 'I') else image,
                                         dtype=np.float64))
        return (frames[0] if len(frames) == 1 else np.stack(frames)), None, None, {}
    if loaded_datasets is not None and source in loaded_datasets:
        datasets = loaded_datasets[source]
    else:
        from . import file_tools
        datasets = file_tools.open_file(source)
        if loaded_datasets is not None:
            loaded_datasets[source] = datasets
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
    return np.asarray(dataset), pixel_size, units, {}


def _output_path(source, suffix, output_filename, output_dir):
    if output_filename:
        return output_filename
    if os.path.exists(source):
        return f'{os.path.splitext(source)[0]}_{suffix}.npy'
    safe = ''.join(character if character.isalnum() or character in '-_.' else '_' for character in source)
    return os.path.join(output_dir, f'{safe}_{suffix}.npy')


def register(mcp, loaded_datasets=None, tiled_uri=None, output_dir=None):
    """Register all analysis tools on a FastMCP server instance.

    tiled_uri: Tiled server for data keys (PYTEMLIB_TILED_URI when None)
    output_dir: where results computed from data keys are written; PYTEMLIB_OUTPUT_DIR, else the
        current directory, when None
    """
    def load(source, dataset_key=None):
        return load_array(source, dataset_key, loaded_datasets, tiled_uri)

    def results_dir():
        return os.path.abspath(output_dir or os.environ.get('PYTEMLIB_OUTPUT_DIR') or os.getcwd())

    def output_path(source, suffix, output_filename):
        return _output_path(source, suffix, output_filename, results_dir())

    @mcp.tool()
    def compute_image_metrics(source: str, dataset_key: str = None, metrics: list = None) -> str:
        """Focus / quality metrics of an image or of every frame of an image stack.

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        Works for any 2D image (HAADF, BF, camera, diffraction, AFM channel).
        metrics: subset of contrast, entropy, high_frequency_power, edge_sharpness (all when empty).
        Larger is better for every metric, so any can drive an optimisation.
        """
        data = load(source, dataset_key)[0]
        result = {'metrics': image_metrics.image_quality_metrics(data, metrics or None)}
        if data.ndim == 2:
            result['statistics'] = image_metrics.image_statistics(data)
        return _json(result)

    @mcp.tool()
    def segment_image(source: str, dataset_key: str = None, method: str = 'otsu', value: float = None,
                      polarity: str = 'above', smooth_sigma: float = 0.8, opening_size: int = 2,
                      min_area_px: int = None, pixel_size: float = None, max_regions: int = 50,
                      output_filename: str = None) -> str:
        """Threshold-segment a 2D image into regions and report per-region shape statistics.

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        method: otsu | percentile (value 0-100) | absolute (value in image units).
        polarity: above (bright features) | below (dark features).
        pixel_size: physical pixel size; read from the dataset when omitted, else 1 (pixels).
        The label image is written to a .npy file whose path is returned.
        """
        data, file_pixel_size, units, _ = load(source, dataset_key)
        if data.ndim != 2:
            raise ValueError(f'segment_image needs a 2D image, got shape {data.shape}')
        labels, threshold = image_segmentation.segment_particles(data, method, value, polarity, smooth_sigma,
                                                                 opening_size, min_area_px)
        size = pixel_size or file_pixel_size or 1.0
        statistics = image_segmentation.particle_statistics(labels, data, size)
        path = output_path(source, 'labels', output_filename)
        np.save(path, labels)
        return _json({'threshold': threshold, 'n_regions': len(statistics),
                      'coverage': float(np.mean(labels > 0)),
                      'pixel_size': size,
                      'length_units': 'as given' if pixel_size else (units if file_pixel_size else 'pixels'),
                      'labels_file': path, 'regions': statistics[:max_regions],
                      'regions_truncated': len(statistics) > max_regions})

    @mcp.tool()
    def align_image_series(source: str, dataset_key: str = None, reference_index: int = 0,
                           max_shift: float = None, cumulative: bool = False,
                           output_filename: str = None) -> str:
        """Register every frame of an image stack (frame, row, column) by sub-pixel cross-correlation.

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        Returns per-frame (row, column) shifts in pixels and writes the aligned average to .npy.
        cumulative: register each frame to the previous one (for tilt series / slowly changing contrast).
        """
        data = load(source, dataset_key)[0]
        if data.ndim != 3:
            raise ValueError(f'align_image_series needs a (frame, row, column) stack, got {data.shape}')
        shifts = image_alignment.align_image_stack(data, reference_index, max_shift, cumulative=cumulative)
        aligned, _ = image_alignment.apply_shifts(data, shifts)
        path = output_path(source, 'aligned_average', output_filename)
        np.save(path, aligned.mean(axis=0))
        return _json({'shifts_px': shifts, 'average_file': path, 'average_shape': aligned.shape[1:]})

    @mcp.tool()
    def measure_image_shift(reference: str, moving: str, max_shift: float = None,
                            pixel_size: float = None) -> str:
        """Sub-pixel (row, column) shift that registers the moving image onto the reference.

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        With pixel_size, also returns the shift in physical units for drift / image-shift correction.
        """
        reference, reference_pixel, units, _ = load(reference)
        moving = load(moving)[0]
        shift = image_alignment.cross_correlation_shift(reference, moving, max_shift)
        result = {'shift_px': shift}
        size = pixel_size or reference_pixel
        if size:
            result['shift_physical'] = shift * size
            result['units'] = units if pixel_size is None else 'as given'
        return _json(result)

    @mcp.tool()
    def plan_sampling_mask(rows: int, columns: int, fraction: float, method: str = 'random', seed: int = 0,
                           preview: str = None, output_filename: str = None) -> str:
        """Choose which scan positions to measure for a sparse / compressed-sensing acquisition.

        method: random | grid | preview-guided (needs preview: a file path or data key of e.g. a fast
        low-dose scan).
        Writes the boolean mask to .npy and returns its path and the number of positions.
        """
        preview = None if preview is None else load(preview)[0]
        mask = sparse_reconstruction.make_sampling_mask((rows, columns), fraction, method, seed, preview)
        path = output_filename or os.path.join(results_dir(), f'sampling_mask_{rows}x{columns}_{method}.npy')
        np.save(path, mask)
        return _json({'mask_file': path, 'measured_positions': int(mask.sum()), 'fraction': float(mask.mean())})

    @mcp.tool()
    def reconstruct_sparse_data(source: str, mask_filename: str, method: str = 'dct',
                                sample_axes: list = None, iterations: int = 40, sparsity_fraction: float = 0.08,
                                tv_step: float = 0.08, loss: str = 'l2', dataset_key: str = None,
                                output_filename: str = None) -> str:
        """Fill in unmeasured positions of sparsely sampled data (image, spectrum image or 4D-STEM).

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        mask_filename: .npy boolean mask, True where measured, shaped like the two sample_axes.
        method: nearest | dct | tv. loss: l2 | l1 | huber (l1 / huber resist outliers).
        sample_axes: the two sampled axes, default [0, 1].
        """
        data = load(source, dataset_key)[0]
        mask = np.load(mask_filename, allow_pickle=False).astype(bool)
        axes = tuple(sample_axes or (0, 1))
        estimate = sparse_reconstruction.sparse_reconstruct(data, mask, method, axes, iterations,
                                                            sparsity_fraction, tv_step, loss)
        path = output_path(source, f'{method}_reconstruction', output_filename)
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
    def analyze_diffraction_rings(source: str, pixel_size_mrad: float = None, acceleration_voltage: float = 200e3,
                                  dataset_key: str = None, hkl: list = None, deconvolve: bool = True,
                                  max_angle_mrad: float = None, max_rings: int = None,
                                  prominence_fraction: float = 0.03) -> str:
        """Direct beam, diffraction rings, d-spacings and (cubic) lattice parameters of a 2D pattern.

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        pixel_size_mrad: angular size of a detector pixel (full scattering angle); taken from the
        data's metadata when omitted, and the call fails if neither gives it.
        hkl: reflections assigned to the first rings in order, e.g. [[1,1,1],[2,0,0]]; for each a cubic
        lattice parameter is returned. deconvolve: Richardson-Lucy with the direct beam as PSF
        (use for convergent-beam / nano-diffraction patterns).
        """
        data, stored_pixel_size, stored_units, _ = load(source, dataset_key)
        if pixel_size_mrad is None:
            if stored_units != 'mrad':
                raise ValueError('pixel_size_mrad is required: the data carries no angular calibration')
            pixel_size_mrad = stored_pixel_size
        if data.ndim != 2:
            raise ValueError(f'analyze_diffraction_rings needs a 2D pattern, got {data.shape}')
        result = ring_analysis.analyze_ring_pattern(data, pixel_size_mrad, acceleration_voltage, max_angle_mrad,
                                                    deconvolve, prominence_fraction=prominence_fraction,
                                                    max_rings=max_rings, hkl=hkl)
        result.pop('radial_profile')
        result.pop('radial_angle_mrad')
        return _json(result)

    @mcp.tool()
    def four_d_stem_virtual_image(source: str, inner_radius: float, outer_radius: float,
                                  center: list = None, dataset_key: str = None,
                                  output_filename: str = None) -> str:
        """Virtual bright-field (inner_radius 0) or annular dark-field image of a 4D-STEM cube.

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        The cube is (scan_row, scan_column, k_row, k_column); radii are in diffraction pixels.
        Writes the image to .npy and returns its path plus the image's quality metrics.
        """
        data = load(source, dataset_key)[0]
        image = four_d_stem.virtual_image(data, inner_radius, outer_radius, center)
        path = output_path(source, f'virtual_{inner_radius:g}_{outer_radius:g}', output_filename)
        np.save(path, image)
        return _json({'image_file': path, 'shape': image.shape,
                      'statistics': image_metrics.image_statistics(image)})

    @mcp.tool()
    def maped_average_cubes(sources: list, dark_field_inner: float = 0.15, dark_field_outer: float = 0.45,
                            output_filename: str = None) -> str:
        """Align 4D-STEM cubes taken at different beam tilts (reciprocal then real space) and average (MAPED).

        source: a file path, or a data key returned by an
        acquisition tool (read from the Tiled data server).
        dark_field_inner / _outer: virtual dark-field radii as fractions of the pattern size, used for
        real-space alignment. Writes the averaged cube to .npy.
        """
        cubes = [load(name)[0] for name in sources]
        result = four_d_stem.maped_average(cubes, (dark_field_inner, dark_field_outer))
        path = output_path(sources[0], 'maped_average', output_filename)
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

    @mcp.tool()
    def map_create(map_dir: str, pixel_size: float, stage_to_image: list = None) -> str:
        """Start an empty tile map in a directory, to stitch images taken at known stage positions.

        pixel_size: image pixel size, in the same unit as the stage positions passed to map_add_tile.
        stage_to_image: 2x2 matrix mapping a stage move (x, y) to an image move (row, column); default
        [[0, 1], [1, 0]] (x along columns, y along rows). Instrument-specific: measure it once by moving
        the stage and measuring the image shift. Each map step is a separate call; the map is stored
        in the directory.
        """
        manifest = image_mosaic.create_map(map_dir, pixel_size, stage_to_image or ((0.0, 1.0), (1.0, 0.0)))
        return _json({'map_dir': os.path.abspath(map_dir), 'pixel_size': manifest['pixel_size'],
                      'stage_to_image': manifest['stage_to_image']})

    @mcp.tool()
    def map_add_tile(map_dir: str, source: str, position: list, dataset_key: str = None, labels: str = None,
                     refine: bool = True, max_shift_px: float = None, min_overlap_px: int = 10) -> str:
        """Add one image tile to a tile map, optionally with its segmentation labels.

        source: the tile image, a file path or data key. position: [x, y] stage position of the
        tile centre. refine: correct the stage placement by matching the overlap with the map.
        labels: optional label image of this tile (the labels_file from segment_image or
        segment_image_sam2); its regions are joined with map regions they overlap by at least
        min_overlap_px pixels, so a particle cut by a tile edge becomes one region.
        """
        image = load(source, dataset_key)[0]
        label_image = None if labels is None else load(labels)[0]
        return _json(image_mosaic.add_tile(map_dir, image, position, label_image, refine, max_shift_px,
                                           min_overlap_px))

    @mcp.tool()
    def map_summary(map_dir: str, max_regions: int = 50) -> str:
        """Stitched map and merged regions so far: writes map_image.npy and map_labels.npy into the map
        directory and returns their paths, the map size and per-region statistics in map coordinates."""
        summary = image_mosaic.map_summary(map_dir)
        if summary['tiles']:
            np.save(os.path.join(map_dir, 'map_image.npy'), image_mosaic.map_image(map_dir))
            np.save(os.path.join(map_dir, 'map_labels.npy'), image_mosaic.map_labels(map_dir))
            summary['map_image_file'] = os.path.abspath(os.path.join(map_dir, 'map_image.npy'))
            summary['map_labels_file'] = os.path.abspath(os.path.join(map_dir, 'map_labels.npy'))
        summary['regions_truncated'] = len(summary['regions']) > max_regions
        summary['regions'] = summary['regions'][:max_regions]
        return _json(summary)

    @mcp.tool()
    def analysis_capabilities() -> str:
        """Which optional tool groups this server offers, and how to install the missing ones."""
        return _json({'bayesian_optimization': bayesian_optimization_missing or 'available',
                      'sam2_segmentation': sam2_missing or 'available'})

    names = ['compute_image_metrics', 'segment_image', 'align_image_series', 'measure_image_shift',
             'plan_sampling_mask', 'reconstruct_sparse_data', 'suggest_next_measurements',
             'analyze_diffraction_rings', 'four_d_stem_virtual_image', 'maped_average_cubes',
             'propose_next_setting', 'fit_parabola_vertex', 'fit_linear_trend',
             'map_create', 'map_add_tile', 'map_summary', 'analysis_capabilities']

    if bayesian_optimization is not None:
        @mcp.tool()
        def bo_suggest_next_setting(points: list, scores: list, bounds: list, maximize: bool = True,
                                    acquisition: str = 'log_ei', beta: float = 2.0, seed: int = 0) -> str:
            """Bayesian optimisation: the next setting to try for one score.

            points: settings measured so far, one list per measurement (any number of parameters, at
            least two measurements). scores: the measured score of each. bounds: [[low, high], ...] per
            parameter; the suggestion stays inside them. acquisition: log_ei (expected improvement) or
            ucb (upper confidence bound, beta sets exploration). Returns the suggestion with the model's
            predicted score and standard deviation there. It only suggests; nothing is moved.
            """
            return _json(bayesian_optimization.suggest_setting(points, scores, bounds, maximize, acquisition,
                                                               beta, seed))

        @mcp.tool()
        def bo_suggest_pareto_setting(points: list, objectives: list, bounds: list, maximize: list = None,
                                      ref_point: list = None, seed: int = 0) -> str:
            """Multi-objective Bayesian optimisation (expected hypervolume improvement).

            objectives: one row per measured point with two or more scores (for example image contrast
            and an FFT score). maximize: one flag per score, all true by default. Returns the next
            setting, its predicted scores and which measured points are currently Pareto-optimal.
            """
            return _json(bayesian_optimization.suggest_pareto_setting(points, objectives, bounds, maximize,
                                                                      ref_point, seed))

        names += ['bo_suggest_next_setting', 'bo_suggest_pareto_setting']

    if sam2_segmentation is not None:
        @mcp.tool()
        def segment_image_sam2(source: str, dataset_key: str = None, model_id: str = 'facebook/sam2-hiera-large',
                               compute_device: str = 'auto', points_per_side: int = 48, iou_threshold: float = 0.57,
                               stability_threshold: float = 0.75, min_area_px: int = 200, max_regions: int = 50,
                               output_filename: str = None) -> str:
            """Segment an image with SAM2 (Segment Anything 2) and report each region.

            source: a file path or data key. model_id: a Hugging Face SAM2 model, e.g.
            facebook/sam2-hiera-large (default) or facebook/sam2-hiera-tiny (faster). The model is
            loaded once and reused. Regions smaller than min_area_px or covering half the image are
            dropped. Writes the label image to .npy (labels_file), which map_add_tile accepts.
            """
            image = load(source, dataset_key)[0]
            labels, statistics = sam2_segmentation.segment(image, model_id, compute_device, points_per_side,
                                                           iou_threshold, stability_threshold, min_area_px)
            path = output_path(source, 'sam2_labels', output_filename)
            np.save(path, labels)
            return _json({'labels_file': path, 'n_regions': len(statistics), 'model_id': model_id,
                          'regions': statistics[:max_regions], 'regions_truncated': len(statistics) > max_regions})

        names.append('segment_image_sam2')

    return names
