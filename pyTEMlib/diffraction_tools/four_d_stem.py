"""
4D-STEM helpers: virtual detectors, pattern alignment and tilt-series averaging

A 4D-STEM data cube is indexed (scan_row, scan_column, k_row, k_column).
Any array with that layout works, including sidpy datasets.

``maped_average`` implements multi-angle precession-style averaging of
4D-STEM data recorded at several beam tilts (MAPED): diffraction patterns
of each tilt are aligned in reciprocal space, the virtual dark-field images
are aligned in real space, and the aligned cubes are averaged. Extracted
and generalised from the asyncroscopy 15_MAPED notebook, without the
quantem dependency.

part of pyTEMlib
"""
import numpy as np

from ..image_tools.image_alignment import cross_correlation_shift, fourier_shift


def _cube(data):
    cube = np.asarray(data, dtype=np.float64)
    if cube.ndim != 4:
        raise ValueError(f'expected a 4D (scan_row, scan_column, k_row, k_column) array, got {cube.shape}')
    return cube


def pattern_center(cube):
    """Geometric centre (row, column) of the diffraction patterns."""
    return tuple((np.array(np.shape(cube)[-2:]) - 1) / 2)


def mean_diffraction_pattern(cube):
    """Diffraction pattern averaged over all probe positions."""
    return _cube(cube).mean(axis=(0, 1))


def max_diffraction_pattern(cube):
    """Pixel-wise maximum over all probe positions (shows weak, local reflections)."""
    return _cube(cube).max(axis=(0, 1))


def virtual_detector_mask(pattern_shape, center, inner_radius, outer_radius):
    """Annular detector mask; inner_radius 0 gives a disk (bright field)."""
    rows, columns = np.indices(pattern_shape, dtype=np.float64)
    radius = np.hypot(rows - center[0], columns - center[1])
    return (radius >= inner_radius) & (radius <= outer_radius)


def virtual_image(cube, inner_radius, outer_radius, center=None):
    """Virtual bright-field (inner_radius 0) or annular dark-field image.

    Radii are in diffraction-pattern pixels; ``center`` defaults to the pattern centre.
    """
    cube = _cube(cube)
    center = pattern_center(cube) if center is None else center
    mask = virtual_detector_mask(cube.shape[-2:], center, inner_radius, outer_radius)
    return cube[..., mask].sum(axis=-1)


def center_of_mass_shifts(cube, center=None):
    """Centre-of-mass displacement of every pattern from ``center`` (DPC-style).

    Returns
    -------
    numpy.ndarray of shape (scan_row, scan_column, 2): (k_row, k_column) shifts in pixels
    """
    cube = _cube(cube)
    center = pattern_center(cube) if center is None else center
    rows, columns = np.indices(cube.shape[-2:], dtype=np.float64)
    total = np.maximum(cube.sum(axis=(-2, -1)), 1e-12)
    com_row = (cube * rows).sum(axis=(-2, -1)) / total - center[0]
    com_column = (cube * columns).sum(axis=(-2, -1)) / total - center[1]
    return np.stack([com_row, com_column], axis=-1)


def align_cube_patterns(cube, shift):
    """Shift every diffraction pattern of a cube by (k_row, k_column) pixels."""
    return np.clip(fourier_shift(_cube(cube), shift, axes=(-2, -1)), 0, None)


def align_cube_scan(cube, shift):
    """Shift the scan (real-space) axes of a cube by (row, column) probe positions."""
    return np.clip(fourier_shift(_cube(cube), shift, axes=(0, 1)), 0, None)


def maped_average(cubes, dark_field_radii=(0.15, 0.45), reference_index=0):
    """Align 4D-STEM cubes recorded at different beam tilts and average them.

    Parameters
    ----------
    cubes: sequence of 4D arrays with identical shape
    dark_field_radii: (float, float)
        inner and outer radius of the virtual annular dark-field detector used for
        real-space alignment, as fractions of the diffraction-pattern size
    reference_index: int
        index of the cube used as reference for both alignments

    Returns
    -------
    dict with average (4D array, cropped to the common field of view),
    reciprocal_shifts and real_shifts (arrays of shape (n, 2), pixels) and
    dark_field (virtual dark-field image of the average)
    """
    cubes = [_cube(cube) for cube in cubes]
    if len({cube.shape for cube in cubes}) != 1:
        raise ValueError('all cubes must have the same shape')
    reference = reference_index
    mean_patterns = [cube.mean(axis=(0, 1)) for cube in cubes]
    reciprocal_shifts = np.array([cross_correlation_shift(mean_patterns[reference], pattern, window=False)
                                  for pattern in mean_patterns])
    reciprocal_shifts -= reciprocal_shifts[reference]
    reciprocal_aligned = [align_cube_patterns(cube, shift) for cube, shift in zip(cubes, reciprocal_shifts)]

    size = min(cubes[0].shape[-2:])
    inner, outer = dark_field_radii[0] * size, dark_field_radii[1] * size
    dark_fields = [virtual_image(cube, inner, outer) for cube in reciprocal_aligned]
    real_shifts = np.array([cross_correlation_shift(dark_fields[reference], image) for image in dark_fields])
    real_shifts -= real_shifts[reference]
    real_aligned = [align_cube_scan(cube, shift) for cube, shift in zip(reciprocal_aligned, real_shifts)]

    margin_row, margin_column = np.ceil(np.max(np.abs(real_shifts), axis=0)).astype(int)
    rows = slice(margin_row, cubes[0].shape[0] - margin_row)
    columns = slice(margin_column, cubes[0].shape[1] - margin_column)
    average = np.mean([cube[rows, columns] for cube in real_aligned], axis=0)
    return {'average': average,
            'reciprocal_shifts': reciprocal_shifts,
            'real_shifts': real_shifts,
            'dark_field': virtual_image(average, inner, outer)}


def tilt_series_map(patterns, axis=-2, log_scale=True):
    """Collapse a stack of diffraction patterns (tilt, k_row, k_column) into a 2D map.

    Each pattern is averaged along ``axis`` so a tilt series becomes an image of
    tilt versus detector coordinate (e.g. an alpha-tilt diffraction map).
    """
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim != 3:
        raise ValueError('patterns must be a (tilt, k_row, k_column) stack')
    values = np.log1p(np.clip(patterns, 0, None)) if log_scale else patterns
    return values.mean(axis=axis if axis < 0 else axis + 1)
