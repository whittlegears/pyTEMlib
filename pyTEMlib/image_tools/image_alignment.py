"""
Translation alignment of images and image series

Sub-pixel cross-correlation shifts, Fourier shifting along any two axes,
and alignment / averaging of image stacks. Used for drift tracking during
tilt series, for aligning repeated scans, and (through
pyTEMlib.diffraction_tools.four_d_stem) for aligning diffraction patterns.

Extracted and generalised from the asyncroscopy MAPED and alpha-tilt
diffraction-map notebooks, without the quantem dependency.

part of pyTEMlib
"""
import numpy as np


def _upsampled_correlation(cross_power, upsample_factor, region_size, offset):
    """Cross-correlation on a fine grid around ``offset`` by matrix-multiply DFT.

    Guizar-Sicairos, Thurman and Fienup, Opt. Lett. 33, 156 (2008).
    """
    rows, columns = cross_power.shape
    samples = np.arange(region_size) - region_size // 2
    row_kernel = np.exp(2j * np.pi / (rows * upsample_factor)
                        * (samples[:, None] + offset[0] * upsample_factor)
                        * np.fft.fftfreq(rows, 1 / rows)[None, :])
    column_kernel = np.exp(2j * np.pi / (columns * upsample_factor)
                           * np.fft.fftfreq(columns, 1 / columns)[:, None]
                           * (samples[None, :] + offset[1] * upsample_factor))
    return np.abs(row_kernel @ cross_power @ column_kernel)


def cross_correlation_shift(reference, moving, max_shift=None, phase_correlation=False,
                            window=True, upsample_factor=20):
    """Translation that aligns ``moving`` onto ``reference``.

    Parameters
    ----------
    reference, moving: 2D array_like of identical shape
    max_shift: float or None
        largest accepted shift in pixels along each axis
    phase_correlation: bool
        whiten the cross-power spectrum; robust to contrast changes but less
        accurate than plain cross-correlation for smooth, low-pass images
    window: bool
        apply a Hann window to suppress edge artefacts
    upsample_factor: int
        sub-pixel precision is 1 / upsample_factor pixel (upsampled-DFT refinement)

    Returns
    -------
    numpy.ndarray (row_shift, column_shift) in pixels. Applying it to
    ``moving`` with :func:`fourier_shift` (or scipy.ndimage.shift) registers
    it with ``reference``.
    """
    reference = np.nan_to_num(np.asarray(reference, dtype=np.float64))
    moving = np.nan_to_num(np.asarray(moving, dtype=np.float64))
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError('reference and moving must be 2D arrays of the same shape')
    reference = reference - reference.mean()
    moving = moving - moving.mean()
    if window:
        hann = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
        reference = reference * hann
        moving = moving * hann
    cross_power = np.fft.fft2(reference) * np.conj(np.fft.fft2(moving))
    if phase_correlation:
        cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(cross_power)))
    center = np.array(correlation.shape) // 2
    if max_shift is not None:
        rows, columns = np.indices(correlation.shape)
        outside = (np.abs(rows - center[0]) > max_shift) | (np.abs(columns - center[1]) > max_shift)
        correlation[outside] = -np.inf
    peak = np.array(np.unravel_index(int(np.argmax(correlation)), correlation.shape))
    shift = (peak - center).astype(np.float64)
    if upsample_factor > 1:
        region_size = int(np.ceil(1.5 * upsample_factor))
        fine = _upsampled_correlation(cross_power, upsample_factor, region_size, shift)
        fine_peak = np.array(np.unravel_index(int(np.argmax(fine)), fine.shape))
        shift = shift + (fine_peak - region_size // 2) / upsample_factor
    return shift


def fourier_shift(array, shift, axes=(-2, -1)):
    """Sub-pixel circular shift of ``array`` by ``shift`` pixels along ``axes``.

    Works on any N-D array; e.g. axes=(0, 1) shifts the scan axes of a
    4D-STEM cube and axes=(-2, -1) shifts every diffraction pattern.
    """
    array = np.asarray(array)
    axes = tuple(axis % array.ndim for axis in axes)
    phase = np.ones([array.shape[axis] if axis in axes else 1 for axis in range(array.ndim)],
                    dtype=np.complex128)
    for axis, value in zip(axes, shift):
        frequency_shape = [1] * array.ndim
        frequency_shape[axis] = array.shape[axis]
        frequency = np.fft.fftfreq(array.shape[axis]).reshape(frequency_shape)
        phase = phase * np.exp(-2j * np.pi * frequency * value)
    shifted = np.fft.ifftn(np.fft.fftn(array, axes=axes) * phase, axes=axes)
    return shifted.real if not np.iscomplexobj(array) else shifted


def align_image_stack(images, reference_index=0, max_shift=None, phase_correlation=False,
                      cumulative=False):
    """Shifts that register every image of a stack with a reference.

    Parameters
    ----------
    images: 3D array_like (frame, row, column)
    reference_index: int
        frame used as reference
    max_shift: float or None
        largest accepted shift per registration, pixels
    phase_correlation: bool
    cumulative: bool
        register each frame to its predecessor and accumulate (good for slowly
        changing contrast, e.g. tilt series) instead of registering to the reference

    Returns
    -------
    numpy.ndarray of shape (frames, 2): shift to apply to each frame, pixels
    """
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError('images must be a (frame, row, column) stack')
    shifts = np.zeros((len(images), 2))
    if cumulative:
        for index in range(1, len(images)):
            shifts[index] = shifts[index - 1] + cross_correlation_shift(
                images[index - 1], images[index], max_shift, phase_correlation)
        shifts -= shifts[reference_index]
    else:
        for index, image in enumerate(images):
            shifts[index] = cross_correlation_shift(images[reference_index], image, max_shift,
                                                    phase_correlation)
    return shifts


def apply_shifts(images, shifts, crop=True):
    """Apply per-frame shifts; optionally crop the wrapped-around margin.

    Returns
    -------
    aligned: numpy.ndarray (frame, row, column)
    crop_slices: tuple of two slices applied to the image axes
    """
    images = np.asarray(images, dtype=np.float64)
    aligned = np.stack([fourier_shift(image, shift) for image, shift in zip(images, shifts)])
    if not crop:
        return aligned, (slice(None), slice(None))
    margin_y, margin_x = np.ceil(np.max(np.abs(shifts), axis=0)).astype(int)
    slices = (slice(margin_y, images.shape[1] - margin_y), slice(margin_x, images.shape[2] - margin_x))
    return aligned[:, slices[0], slices[1]], slices


def align_and_average(images, reference_index=0, max_shift=None, cumulative=False):
    """Register a stack, average it and crop the margin that wrapped around.

    Returns
    -------
    average: 2D numpy.ndarray
    shifts: numpy.ndarray (frames, 2)
    """
    shifts = align_image_stack(images, reference_index, max_shift, cumulative=cumulative)
    aligned, _ = apply_shifts(images, shifts, crop=True)
    return aligned.mean(axis=0), shifts


def stage_drift_correction(shift_pixels, pixel_size, sign=(1.0, 1.0)):
    """Convert a (row, column) pixel shift to an (x, y) image-shift correction.

    ``sign`` maps image axes to the microscope's image-shift axes and must be
    calibrated once per instrument / scan rotation.
    """
    row_shift, column_shift = shift_pixels
    return np.asarray(sign, dtype=np.float64) * np.array([column_shift, row_shift]) * pixel_size
