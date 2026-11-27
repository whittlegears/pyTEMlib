"""
Image quality metrics

Scalar figures of merit for any 2D image (HAADF, BF, camera frame,
diffraction pattern, AFM channel, ...) and for stacks of images.
Larger values are better for every focus-type metric, so any of them can
drive an optimizer (focus, stigmation, aberration or stage-z searches).

Extracted and generalised from the asyncroscopy aberration / stage-z
hill-climbing and compressed-sensing notebooks.

part of pyTEMlib
"""
import numpy as np


def _as_float_array(image):
    return np.asarray(image, dtype=np.float64)


def _finite(image):
    values = image[np.isfinite(image)]
    if values.size == 0:
        raise ValueError('image contains no finite values')
    return values


def normalize_image(image, percentiles=(1, 99)):
    """Robustly rescale an image to [0, 1].

    Values below / above the given percentiles are clipped; NaN becomes 0.

    Parameters
    ----------
    image: array_like
        any N-dimensional array
    percentiles: tuple of two floats
        lower and upper percentile used for clipping, (0, 100) is plain min-max

    Returns
    -------
    numpy.ndarray of float64 in [0, 1]
    """
    image = _as_float_array(image)
    low, high = np.percentile(_finite(image), percentiles)
    if high <= low:
        return np.zeros_like(image)
    return np.nan_to_num(np.clip((image - low) / (high - low), 0.0, 1.0))


def _clip_percentiles(image, percentiles):
    image = _as_float_array(image)
    low, high = np.percentile(_finite(image), percentiles)
    return np.clip(image, low, high), low, high


def image_contrast(image, percentiles=(1, 99)):
    """Normalised RMS contrast: std / |mean| of the percentile-clipped image."""
    clipped, _, _ = _clip_percentiles(image, percentiles)
    clipped = clipped[np.isfinite(clipped)]
    return float(np.std(clipped) / (abs(np.mean(clipped)) + 1e-12))


def image_entropy(image, bins=256, percentiles=(1, 99)):
    """Shannon entropy of the intensity histogram, normalised to [0, 1]."""
    clipped, low, high = _clip_percentiles(image, percentiles)
    if high <= low:
        return 0.0
    histogram, _ = np.histogram(clipped[np.isfinite(clipped)], bins=bins, range=(low, high))
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(bins))


def high_frequency_power(image, cutoff=0.20):
    """Fraction of Fourier power above a spatial-frequency cutoff.

    Parameters
    ----------
    image: 2D array_like
    cutoff: float
        radial frequency in cycles / pixel (Nyquist is 0.5)
    """
    image = np.nan_to_num(_as_float_array(image))
    if image.ndim != 2:
        raise ValueError('high_frequency_power needs a 2D image')
    centered = image - np.mean(image)
    power = np.abs(np.fft.rfft2(centered)) ** 2
    frequency_y = np.fft.fftfreq(centered.shape[0])[:, None]
    frequency_x = np.fft.rfftfreq(centered.shape[1])[None, :]
    high = np.hypot(frequency_x, frequency_y) >= cutoff
    return float(power[high].sum() / (power.sum() + 1e-12))


def edge_sharpness(image, percentiles=(1, 99)):
    """RMS gradient magnitude of the mean-normalised, percentile-clipped image."""
    clipped, _, _ = _clip_percentiles(image, percentiles)
    if clipped.ndim != 2:
        raise ValueError('edge_sharpness needs a 2D image')
    clipped = np.nan_to_num(clipped, nan=float(np.nanmean(clipped)))
    normalized = clipped / (abs(np.mean(clipped)) + 1e-12)
    gradient_y, gradient_x = np.gradient(normalized)
    return float(np.sqrt(np.mean(gradient_x ** 2 + gradient_y ** 2)))


def image_statistics(image):
    """Basic intensity statistics of an image (NaN-aware)."""
    values = _finite(_as_float_array(image))
    return {'shape': list(np.shape(image)),
            'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'std': float(values.std()),
            'p01': float(np.percentile(values, 1)), 'p10': float(np.percentile(values, 10)),
            'p50': float(np.percentile(values, 50)), 'p90': float(np.percentile(values, 90)),
            'p99': float(np.percentile(values, 99)),
            'nan_fraction': float(1 - values.size / np.size(image))}


metric_functions = {'contrast': image_contrast,
                    'entropy': image_entropy,
                    'high_frequency_power': high_frequency_power,
                    'edge_sharpness': edge_sharpness}


def image_quality_metrics(image, metrics=None, image_axes=(-2, -1)):
    """Evaluate focus / quality metrics on an image or on every frame of a stack.

    Parameters
    ----------
    image: array_like
        2D image, or N-D array of images whose image plane is given by ``image_axes``
    metrics: list of str or None
        subset of ``metric_functions`` keys; all of them when None
    image_axes: tuple of two ints
        the two axes that form one image

    Returns
    -------
    dict
        metric name -> float for a 2D image, or metric name -> list of floats
        (one per frame, in C order of the remaining axes) for a stack
    """
    names = list(metric_functions) if metrics is None else list(metrics)
    unknown = [name for name in names if name not in metric_functions]
    if unknown:
        raise ValueError(f'unknown metrics {unknown}; choose from {list(metric_functions)}')
    image = _as_float_array(image)
    if image.ndim < 2:
        raise ValueError('image needs at least two dimensions')
    if image.ndim == 2:
        return {name: metric_functions[name](image) for name in names}
    moved = np.moveaxis(image, image_axes, (-2, -1))
    planes = moved.reshape(-1, *moved.shape[-2:])
    return {name: [metric_functions[name](plane) for plane in planes] for name in names}


def compare_images(reference, estimate, data_range=None):
    """Error metrics between a reference and an estimate of the same shape.

    Parameters
    ----------
    reference, estimate: array_like
    data_range: float or None
        peak-to-peak value used for PSNR; the reference's range when None

    Returns
    -------
    dict with MAE, MSE, RMSE and PSNR (dB)
    """
    reference = _as_float_array(reference)
    estimate = _as_float_array(estimate)
    if reference.shape != estimate.shape:
        raise ValueError(f'shape mismatch {reference.shape} vs {estimate.shape}')
    valid = np.isfinite(reference) & np.isfinite(estimate)
    error = reference[valid] - estimate[valid]
    mse = float(np.mean(error ** 2))
    if data_range is None:
        data_range = float(np.ptp(reference[valid]))
    psnr = float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse)) if data_range > 0 else float('nan')
    return {'MAE': float(np.mean(np.abs(error))), 'MSE': mse, 'RMSE': float(np.sqrt(mse)), 'PSNR': psnr}
