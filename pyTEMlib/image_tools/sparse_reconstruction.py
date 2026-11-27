"""
Sparse sampling and compressed-sensing reconstruction

Measure only some pixel positions, fill in the rest, and decide where to
measure next. The sampled axes can be embedded in a larger array, so the
same code reconstructs a 2D image, an RGB image, a spectrum image
(y, x, energy) or a 4D-STEM data set (y, x, ky, kx) from a subset of
probe positions, as well as a scalar map (e.g. a lattice parameter per
probe position) built one measurement at a time.

Extracted and generalised from the asyncroscopy notebook
10_Compressed_Sensing and the digital-twin hackathon notebooks. Unlike the
notebook versions, values are not assumed to lie in [0, 1]: data are
rescaled with the measured range internally and returned in original units.

part of pyTEMlib
"""
import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn


def make_sampling_mask(shape, fraction, method='random', seed=0, preview=None, floor=0.15):
    """Boolean mask selecting which positions to measure.

    Parameters
    ----------
    shape: (int, int)
        shape of the sampled grid (rows, columns)
    fraction: float
        fraction of positions to measure, 0 < fraction <= 1
    method: str
        'random' - uniform random positions (classic compressed sensing)
        'grid' - regular sub-grid, simple for scan hardware
        'preview-guided' - random positions weighted by the gradient of ``preview``
        (for example a fast low-dose scan or a digital-twin image)
    seed: int
        random seed
    preview: 2D array_like or None
        required for 'preview-guided'; shape must equal ``shape``
    floor: float
        minimum relative sampling weight for 'preview-guided'

    Returns
    -------
    numpy.ndarray of bool with the given shape
    """
    height, width = (int(shape[0]), int(shape[1]))
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    pixel_count = height * width
    sample_count = max(1, int(round(fraction * pixel_count)))
    rng = np.random.default_rng(seed)
    mask = np.zeros(pixel_count, dtype=bool)
    if method == 'random':
        mask[rng.choice(pixel_count, sample_count, replace=False)] = True
    elif method == 'grid':
        row_count = max(1, int(round(np.sqrt(sample_count * height / width))))
        column_count = max(1, int(round(sample_count / row_count)))
        rows = np.linspace(0, height - 1, min(row_count, height), dtype=int)
        columns = np.linspace(0, width - 1, min(column_count, width), dtype=int)
        mask.reshape(height, width)[np.ix_(rows, columns)] = True
    elif method == 'preview-guided':
        if preview is None:
            raise ValueError("method 'preview-guided' needs a preview image")
        preview = np.nan_to_num(np.asarray(preview, dtype=np.float64))
        if preview.ndim == 3:
            preview = preview.mean(axis=-1)
        if preview.shape != (height, width):
            raise ValueError(f'preview shape {preview.shape} does not match {(height, width)}')
        gradient_y, gradient_x = np.gradient(ndimage.gaussian_filter(preview, sigma=2.0))
        importance = np.hypot(gradient_x, gradient_y)
        importance = floor + importance / max(importance.max(), 1e-12)
        probabilities = importance.ravel() / importance.sum()
        mask[rng.choice(pixel_count, sample_count, replace=False, p=probabilities)] = True
    else:
        raise ValueError(f'unknown sampling method {method!r}')
    return mask.reshape(height, width)


def _to_planes(array, sample_axes):
    moved = np.moveaxis(np.asarray(array, dtype=np.float64), sample_axes, (-2, -1))
    return moved.reshape(-1, *moved.shape[-2:]), moved.shape


def _from_planes(planes, moved_shape, sample_axes):
    return np.moveaxis(planes.reshape(moved_shape), (-2, -1), sample_axes)


def _loss_update(residual, loss, delta):
    if loss == 'l2':
        return residual
    if loss == 'l1':
        return delta * np.sign(residual)
    if loss == 'huber':
        return np.clip(residual, -delta, delta)
    raise ValueError(f'unknown loss {loss!r}; choose l2, l1 or huber')


def _nearest_planes(planes, mask):
    _, indices = ndimage.distance_transform_edt(~mask, return_indices=True)
    return planes[:, indices[0], indices[1]]


def _dct_planes(observed, mask, iterations, sparsity_fraction, loss, delta):
    estimate = _nearest_planes(observed, mask)
    keep_count = max(1, int(round(sparsity_fraction * mask.size)))
    for _ in range(iterations):
        coefficients = dctn(estimate, axes=(-2, -1), norm='ortho')
        magnitudes = np.abs(coefficients).reshape(len(coefficients), -1)
        threshold = np.partition(magnitudes, -keep_count, axis=1)[:, -keep_count]
        proposal = idctn(coefficients * (np.abs(coefficients) >= threshold[:, None, None]),
                         axes=(-2, -1), norm='ortho')
        proposal[:, mask] += _loss_update(observed[:, mask] - proposal[:, mask], loss, delta)
        estimate = np.clip(proposal, 0.0, 1.0)
    return estimate


def _tv_planes(observed, mask, iterations, step, loss, delta):
    estimate = _nearest_planes(observed, mask)
    for _ in range(iterations):
        gradient_x = np.diff(estimate, axis=-1, append=estimate[..., -1:])
        gradient_y = np.diff(estimate, axis=-2, append=estimate[..., -1:, :])
        magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1e-6)
        direction_x = gradient_x / magnitude
        direction_y = gradient_y / magnitude
        divergence = direction_x.copy()
        divergence[..., 1:] -= direction_x[..., :-1]
        divergence += direction_y
        divergence[..., 1:, :] -= direction_y[..., :-1, :]
        proposal = np.clip(estimate + step * divergence, 0.0, 1.0)
        proposal[:, mask] += _loss_update(observed[:, mask] - proposal[:, mask], loss, delta)
        estimate = np.clip(proposal, 0.0, 1.0)
    return estimate


def sparse_reconstruct(sampled, mask, method='dct', sample_axes=(0, 1), iterations=40,
                       sparsity_fraction=0.08, tv_step=0.08, loss='l2', delta=0.05,
                       value_range=None):
    """Fill in unmeasured positions of a sparsely sampled array.

    Parameters
    ----------
    sampled: array_like
        data with measured values at ``mask`` positions; other values are ignored
    mask: 2D bool array
        True where measured; its shape equals the shape of ``sample_axes``
    method: str
        'nearest' - copy the nearest measurement (baseline)
        'dct' - iterative hard thresholding in the 2D DCT basis
        'tv' - total-variation smoothing that preserves edges
    sample_axes: tuple of two ints
        axes of ``sampled`` that were sparsely sampled, e.g. (0, 1) for
        (y, x), (y, x, channel) or (y, x, ky, kx)
    iterations: int
    sparsity_fraction: float
        fraction of DCT coefficients kept ('dct')
    tv_step: float
        gradient step size ('tv'), in units of the normalised data
    loss: str
        'l2', 'l1' or 'huber' data-consistency update; l1 / huber resist outliers
    delta: float
        step for 'l1' and transition for 'huber', in units of the normalised data
    value_range: (float, float) or None
        data range used to normalise to [0, 1]; measured min / max when None

    Returns
    -------
    numpy.ndarray with the shape of ``sampled`` in original units
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('at least one measured position is required')
    planes, moved_shape = _to_planes(sampled, sample_axes)
    if planes.shape[-2:] != mask.shape:
        raise ValueError(f'mask shape {mask.shape} does not match sampled axes {planes.shape[-2:]}')
    if value_range is None:
        measured = planes[:, mask]
        value_range = (float(np.nanmin(measured)), float(np.nanmax(measured)))
    low, high = value_range
    span = high - low if high > low else 1.0
    normalized = np.clip(np.nan_to_num((planes - low) / span), 0.0, 1.0)
    if method == 'nearest':
        result = _nearest_planes(normalized, mask)
    elif method == 'dct':
        result = _dct_planes(normalized, mask, iterations, sparsity_fraction, loss, delta)
    elif method == 'tv':
        result = _tv_planes(normalized, mask, iterations, tv_step, loss, delta)
    else:
        raise ValueError(f'unknown reconstruction method {method!r}; choose nearest, dct or tv')
    return _from_planes(low + span * result, moved_shape, sample_axes)


def reconstruct_with_uncertainty(values, measured_mask, region_mask=None, value_bounds=None,
                                 iterations=30, sparsity_choices=(0.04, 0.08, 0.16),
                                 disagreement_weight=0.7):
    """Reconstruct a sparsely measured scalar map and score where to measure next.

    Several DCT reconstructions with different sparsity are averaged. The
    acquisition score combines where they disagree with the distance to the
    nearest measurement. It is a heuristic ranking for active sampling, not a
    calibrated confidence interval.

    Parameters
    ----------
    values: 2D array_like
        measured values at ``measured_mask`` positions
    measured_mask: 2D bool array
    region_mask: 2D bool array or None
        positions that may be reconstructed / measured; everywhere when None
    value_bounds: (float, float) or None
        physical range of the quantity; measured min / max when None
    iterations: int
    sparsity_choices: sequence of float
        DCT keep fractions of the ensemble members
    disagreement_weight: float in [0, 1]
        weight of model disagreement; 1 - weight goes to distance from evidence

    Returns
    -------
    reconstruction: 2D array in units of ``values``, NaN outside region
    uncertainty: 2D array in [0, 1], 0 at measured positions, NaN outside region
    """
    values = np.asarray(values, dtype=np.float64)
    measured_mask = np.asarray(measured_mask, dtype=bool)
    region_mask = np.ones_like(measured_mask) if region_mask is None else np.asarray(region_mask, dtype=bool)
    if not measured_mask.any():
        raise ValueError('at least one measurement is required')
    if value_bounds is None:
        value_bounds = (float(values[measured_mask].min()), float(values[measured_mask].max()))
    candidates = np.asarray([sparse_reconstruct(values, measured_mask, 'dct', iterations=iterations,
                                                sparsity_fraction=sparsity, value_range=value_bounds)
                             for sparsity in sparsity_choices])
    for candidate in candidates:
        candidate[measured_mask] = values[measured_mask]
    reconstruction = candidates.mean(axis=0)
    span = value_bounds[1] - value_bounds[0] if value_bounds[1] > value_bounds[0] else 1.0
    disagreement = candidates.std(axis=0) / span
    disagreement /= max(float(disagreement[region_mask].max()), 1e-12)
    distance = ndimage.distance_transform_edt(~measured_mask)
    distance /= max(float(distance[region_mask].max()), 1e-12)
    uncertainty = disagreement_weight * disagreement + (1 - disagreement_weight) * distance
    uncertainty[measured_mask] = 0.0
    uncertainty[~region_mask] = np.nan
    reconstruction[~region_mask] = np.nan
    return reconstruction, uncertainty


def next_measurement_locations(uncertainty, count=1, min_separation=1.0):
    """Highest-uncertainty positions, at least ``min_separation`` pixels apart.

    Returns
    -------
    list of (row, column) tuples, best first
    """
    score = np.array(uncertainty, dtype=np.float64)
    score[~np.isfinite(score)] = -np.inf
    rows, columns = np.indices(score.shape)
    chosen = []
    for _ in range(int(count)):
        index = int(np.argmax(score))
        if not np.isfinite(score.flat[index]) or score.flat[index] <= 0:
            break
        row, column = np.unravel_index(index, score.shape)
        chosen.append((int(row), int(column)))
        score[np.hypot(rows - row, columns - column) < max(min_separation, 1.0)] = -np.inf
    return chosen
