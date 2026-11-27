"""
Direct beam, radial profile and d-spacing analysis of diffraction patterns

Works on any 2D diffraction pattern (convergent-beam nano-diffraction,
selected-area ring patterns, mean patterns of 4D-STEM data): locate the
direct beam, optionally deconvolve it out with Richardson-Lucy, integrate
azimuthally, find the Bragg rings and convert their angles to d-spacings
and lattice parameters.

Extracted and generalised from the asyncroscopy digital-twin hackathon
notebook, which was restricted to the first {200} ring of an FCC crystal at
200 kV; here the voltage, angular calibration, search range and the
reflection assigned to each ring are parameters.

part of pyTEMlib
"""
import numpy as np
from scipy import ndimage, signal

from ..utilities import get_wavelength


def find_direct_beam(pattern, search_fraction=0.08, threshold_fraction=0.5, center_guess=None):
    """Locate the direct (zero-order) beam.

    Parameters
    ----------
    pattern: 2D array_like
    search_fraction: float
        radius of the search region around ``center_guess`` as fraction of the smaller side
    threshold_fraction: float
        pixels above this fraction of the local maximum belong to the beam
    center_guess: (row, column) or None
        expected beam position; image centre when None

    Returns
    -------
    dict with center (row, column), radius_px (99th percentile of the disk
    radius) and mask (bool array of the disk)
    """
    pattern = np.nan_to_num(np.asarray(pattern, dtype=np.float64))
    if pattern.ndim != 2 or pattern.max() <= 0:
        raise ValueError('pattern must be a 2D array with positive intensity')
    rows, columns = np.indices(pattern.shape, dtype=np.float64)
    if center_guess is None:
        center_guess = ((pattern.shape[0] - 1) / 2, (pattern.shape[1] - 1) / 2)
    radius = np.hypot(rows - center_guess[0], columns - center_guess[1])
    roi = radius < search_fraction * min(pattern.shape)
    candidates = roi & (pattern >= threshold_fraction * pattern[roi].max())
    labels, _ = ndimage.label(candidates)
    brightest = np.unravel_index(int(np.argmax(np.where(roi, pattern, -np.inf))), pattern.shape)
    beam_label = labels[int(round(center_guess[0])), int(round(center_guess[1]))] or labels[brightest]
    if beam_label == 0:
        raise ValueError('direct beam could not be isolated')
    mask = ndimage.binary_fill_holes(labels == beam_label)
    weights = np.where(mask, pattern, 0.0)
    center = (float((rows * weights).sum() / weights.sum()), float((columns * weights).sum() / weights.sum()))
    distance = np.hypot(rows - center[0], columns - center[1])
    return {'center': center, 'radius_px': float(np.percentile(distance[mask], 99)), 'mask': mask}


def deconvolve_direct_beam(pattern, beam_mask, iterations=15, padding=2):
    """Richardson-Lucy deconvolution using the direct beam as point-spread function.

    Sharpens Bragg disks of convergent-beam patterns into spots / rings.
    """
    pattern = np.clip(np.nan_to_num(np.asarray(pattern, dtype=np.float64)), 0, None)
    beam_rows, beam_columns = np.nonzero(beam_mask)
    psf = np.where(beam_mask, pattern, 0.0)[
        max(0, beam_rows.min() - padding):beam_rows.max() + padding + 1,
        max(0, beam_columns.min() - padding):beam_columns.max() + padding + 1]
    psf = psf / psf.sum()
    mirrored = psf[::-1, ::-1]
    estimate = np.full_like(pattern, max(float(pattern.mean()), 1e-6))
    for _ in range(iterations):
        blurred = signal.fftconvolve(estimate, psf, mode='same')
        estimate *= signal.fftconvolve(pattern / np.maximum(blurred, 1e-8), mirrored, mode='same')
        estimate = np.clip(estimate, 0.0, None)
    return estimate


def radial_profile(pattern, center, bin_width=1.0, statistic='sum', mask=None):
    """Azimuthal integration around ``center``.

    Parameters
    ----------
    pattern: 2D array_like
    center: (row, column)
    bin_width: float, pixels
    statistic: 'sum' or 'mean'
    mask: 2D bool array or None, pixels to include

    Returns
    -------
    radius_px: 1D array of bin centres (pixels); bin i holds radii [i, i + 1) * bin_width
    profile: 1D array
    """
    pattern = np.nan_to_num(np.asarray(pattern, dtype=np.float64))
    rows, columns = np.indices(pattern.shape, dtype=np.float64)
    bins = np.floor(np.hypot(rows - center[0], columns - center[1]) / bin_width).astype(int)
    include = np.ones(pattern.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    profile = np.bincount(bins[include], weights=pattern[include])
    if statistic == 'mean':
        profile = profile / np.maximum(np.bincount(bins[include], minlength=len(profile)), 1)
    elif statistic != 'sum':
        raise ValueError("statistic must be 'sum' or 'mean'")
    return (np.arange(len(profile)) + 0.5) * bin_width, profile


def find_profile_peaks(profile, start=0, stop=None, prominence_fraction=0.03, min_distance=1,
                       smooth_sigma=1.5, max_peaks=None):
    """Peaks of a 1D profile with parabolic sub-bin refinement.

    Returns
    -------
    list of dict with position (fractional bin index), height and
    prominence_fraction (prominence relative to the searched maximum),
    ordered by position
    """
    profile = np.asarray(profile, dtype=np.float64)
    if smooth_sigma:
        profile = ndimage.gaussian_filter1d(profile, sigma=smooth_sigma)
    stop = len(profile) if stop is None else min(int(stop), len(profile))
    start = max(int(start), 1)
    segment = profile[start:stop]
    if segment.size < 3 or segment.max() <= 0:
        return []
    indices, properties = signal.find_peaks(segment, prominence=prominence_fraction * segment.max(),
                                            distance=max(1, int(min_distance)))
    peaks = []
    for index, prominence in zip(indices, properties['prominences']):
        center = start + int(index)
        left, middle, right = profile[center - 1:center + 2]
        denominator = left - 2 * middle + right
        offset = 0.5 * (left - right) / denominator if denominator != 0 else 0.0
        peaks.append({'position': center + float(np.clip(offset, -0.5, 0.5)),
                      'height': float(middle),
                      'prominence_fraction': float(prominence / segment.max())})
    if max_peaks is not None:
        strongest = sorted(peaks, key=lambda peak: peak['prominence_fraction'], reverse=True)[:max_peaks]
        peaks = sorted(strongest, key=lambda peak: peak['position'])
    return peaks


def angle_to_d_spacing(angle_mrad, acceleration_voltage):
    """Bragg d-spacing in Angstrom for a full scattering angle 2θ given in mrad."""
    wavelength = get_wavelength(float(acceleration_voltage), 'A')
    return wavelength / (2 * np.sin(np.asarray(angle_mrad, dtype=np.float64) * 1e-3 / 2))


def cubic_lattice_parameter(d_spacing, hkl):
    """Lattice parameter of a cubic crystal from the d-spacing of reflection ``hkl``."""
    return float(d_spacing) * float(np.sqrt(np.sum(np.square(hkl))))


def analyze_ring_pattern(pattern, pixel_size_mrad, acceleration_voltage=200e3, max_angle_mrad=None,
                         deconvolve=True, deconvolution_iterations=15, prominence_fraction=0.03,
                         min_ring_separation_mrad=1.0, max_rings=None, hkl=None,
                         center_guess=None):
    """Find the diffraction rings of a pattern and convert them to d-spacings.

    Parameters
    ----------
    pattern: 2D array_like
    pixel_size_mrad: float
        angular size of one detector pixel (full scattering angle)
    acceleration_voltage: float, V
    max_angle_mrad: float or None
        outer limit of the search; edge of the detector when None
    deconvolve: bool
        Richardson-Lucy deconvolution with the direct beam as PSF (for convergent beams)
    deconvolution_iterations: int
    prominence_fraction: float
        minimum peak prominence relative to the strongest feature
    min_ring_separation_mrad: float
    max_rings: int or None
        keep only the most prominent rings
    hkl: sequence of three ints, or list of them, or None
        reflection(s) assigned to the first ring(s), in order; a cubic lattice
        parameter is reported for every assigned ring
    center_guess: (row, column) or None

    Returns
    -------
    dict with direct_beam (center, radius_mrad), rings (angle_mrad, d_spacing_A,
    prominence_fraction and, when assigned, hkl and lattice_parameter_A),
    radial_angle_mrad and radial_profile
    """
    pattern = np.nan_to_num(np.asarray(pattern, dtype=np.float64))
    beam = find_direct_beam(pattern, center_guess=center_guess)
    work = deconvolve_direct_beam(pattern, beam['mask'], deconvolution_iterations) if deconvolve else pattern.copy()
    rows, columns = np.indices(pattern.shape, dtype=np.float64)
    distance = np.hypot(rows - beam['center'][0], columns - beam['center'][1])
    exclusion_px = 1.05 * beam['radius_px']
    work[distance <= exclusion_px] = 0.0
    radius_px, profile = radial_profile(work, beam['center'])
    stop = None if max_angle_mrad is None else int(np.ceil(max_angle_mrad / pixel_size_mrad))
    peaks = find_profile_peaks(profile, start=int(np.ceil(exclusion_px)), stop=stop,
                               prominence_fraction=prominence_fraction,
                               min_distance=round(min_ring_separation_mrad / pixel_size_mrad),
                               max_peaks=max_rings)
    if hkl is not None and np.ndim(hkl) == 1:
        hkl = [hkl]
    rings = []
    for index, peak in enumerate(peaks):
        peak['position'] += 0.5
        angle = peak['position'] * pixel_size_mrad
        d_spacing = float(angle_to_d_spacing(angle, acceleration_voltage))
        ring = {'angle_mrad': float(angle), 'radius_px': peak['position'],
                'd_spacing_A': d_spacing, 'prominence_fraction': peak['prominence_fraction']}
        if hkl is not None and index < len(hkl):
            ring['hkl'] = [int(value) for value in hkl[index]]
            ring['lattice_parameter_A'] = cubic_lattice_parameter(d_spacing, hkl[index])
        rings.append(ring)
    return {'direct_beam': {'center': list(beam['center']),
                            'radius_mrad': beam['radius_px'] * pixel_size_mrad},
            'rings': rings,
            'radial_angle_mrad': radius_px * pixel_size_mrad,
            'radial_profile': ndimage.gaussian_filter1d(profile, 1.5)}
