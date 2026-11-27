"""
Threshold segmentation and particle statistics

Works on any 2D image: HAADF overviews of nanoparticles, BF images,
AFM height / phase channels, virtual images from 4D-STEM, ...
Pure numpy / scipy.ndimage.

Extracted and generalised from the asyncroscopy digital-twin hackathon
notebook (particle segmentation and shape statistics) and the
SPM_agent_simple segmentation agent notebook.

part of pyTEMlib
"""
import numpy as np
from scipy import ndimage

from .image_metrics import normalize_image


def otsu_threshold(image, bins=256):
    """Otsu threshold of an image in its own intensity units (NaN-aware).

    Returns
    -------
    float
        threshold that maximises the between-class variance
    """
    values = np.asarray(image, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('image contains no finite values')
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return low
    histogram, edges = np.histogram(values, bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_background = np.cumsum(histogram)
    weight_foreground = histogram.sum() - weight_background
    sum_background = np.cumsum(histogram * centers)
    sum_foreground = np.sum(histogram * centers) - sum_background
    valid = (weight_background > 0) & (weight_foreground > 0)
    variance = np.zeros_like(centers)
    variance[valid] = (weight_background[valid] * weight_foreground[valid]
                       * (sum_background[valid] / weight_background[valid]
                          - sum_foreground[valid] / weight_foreground[valid]) ** 2)
    return float(centers[int(np.argmax(variance))])


def gradient_magnitude(image, normalize=True):
    """Sobel gradient magnitude; useful to segment thin boundaries (grain / domain walls)."""
    image = np.nan_to_num(np.asarray(image, dtype=np.float64))
    magnitude = np.hypot(ndimage.sobel(image, axis=0), ndimage.sobel(image, axis=1))
    return normalize_image(magnitude, (0, 100)) if normalize else magnitude


def threshold_image(image, method='otsu', value=None, polarity='above', smooth_sigma=0.0):
    """Binary mask from a threshold.

    Parameters
    ----------
    image: 2D array_like
    method: str
        'otsu' (automatic), 'percentile' (value in 0-100) or 'absolute' (value in image units)
    value: float or None
        required for 'percentile' and 'absolute'
    polarity: str
        'above' keeps pixels >= threshold (bright features), 'below' keeps darker pixels
    smooth_sigma: float
        Gaussian smoothing in pixels applied before thresholding; 0 disables it

    Returns
    -------
    mask: numpy.ndarray of bool
    threshold: float, in units of the (smoothed) image
    """
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    work = np.where(finite, image, np.nanmean(image))
    if smooth_sigma > 0:
        work = ndimage.gaussian_filter(work, sigma=smooth_sigma)
    if method == 'otsu':
        threshold = otsu_threshold(work[finite])
    elif method == 'percentile':
        if value is None:
            raise ValueError("method 'percentile' needs value in 0-100")
        threshold = float(np.percentile(work[finite], value))
    elif method == 'absolute':
        if value is None:
            raise ValueError("method 'absolute' needs value in image units")
        threshold = float(value)
    else:
        raise ValueError(f'unknown threshold method {method!r}')
    if polarity == 'above':
        mask = work >= threshold
    elif polarity == 'below':
        mask = work <= threshold
    else:
        raise ValueError("polarity must be 'above' or 'below'")
    return mask & finite, threshold


morphology_operations = {'open': ndimage.binary_opening, 'close': ndimage.binary_closing,
                         'dilate': ndimage.binary_dilation, 'erode': ndimage.binary_erosion,
                         'fill_holes': None}


def clean_mask(mask, operation='open', size=2):
    """Morphological cleanup of a binary mask with a square structuring element."""
    if operation not in morphology_operations:
        raise ValueError(f'unknown operation {operation!r}; choose from {list(morphology_operations)}')
    mask = np.asarray(mask, dtype=bool)
    if operation == 'fill_holes':
        return ndimage.binary_fill_holes(mask)
    structure = np.ones((int(size), int(size)), dtype=bool)
    return morphology_operations[operation](mask, structure=structure)


def remove_small_regions(mask, min_pixels):
    """Drop 8-connected components smaller than ``min_pixels``."""
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return labels > 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_pixels)
    keep[0] = False
    return keep[labels]


def mask_summary(mask):
    """Region count, coverage and largest region of a binary mask."""
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())[1:]
    return {'n_regions': int(count), 'coverage': float(np.mean(labels > 0)),
            'largest_region_px': int(sizes.max()) if count else 0}


def segment_particles(image, method='otsu', value=None, polarity='above', smooth_sigma=0.8,
                      opening_size=2, min_area_px=None, fill_holes=True):
    """Threshold segmentation into labelled 8-connected regions.

    Parameters
    ----------
    image: 2D array_like
    method, value, polarity, smooth_sigma:
        passed to :func:`threshold_image`
    opening_size: int
        size of the binary opening that removes specks; 0 disables it
    min_area_px: int or None
        regions smaller than this are removed; default max(12, 2e-5 * image size)
    fill_holes: bool
        fill enclosed holes in every region

    Returns
    -------
    labels: numpy.ndarray of int32, 0 is background, regions numbered 1..n
    threshold: float
    """
    mask, threshold = threshold_image(image, method, value, polarity, smooth_sigma)
    if opening_size and opening_size > 1:
        mask = clean_mask(mask, 'open', opening_size)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if min_area_px is None:
        min_area_px = max(12, int(0.00002 * np.size(image)))
    mask = remove_small_regions(mask, min_area_px)
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    return labels.astype(np.int32), threshold


def region_boundary(region_mask):
    """Boolean mask of the 8-connected boundary pixels of a region."""
    region_mask = np.asarray(region_mask, dtype=bool)
    return region_mask & ~ndimage.binary_erosion(region_mask, structure=np.ones((3, 3), dtype=bool),
                                                 border_value=0)


def particle_statistics(labels, image=None, pixel_size=1.0, min_area_px=0):
    """Rotation-invariant size and shape statistics of every labelled region.

    Parameters
    ----------
    labels: 2D int array
        region labels, 0 is background (e.g. from :func:`segment_particles`)
    image: 2D array_like or None
        intensities for mean / integrated intensity per region
    pixel_size: float
        physical size of one pixel; lengths are reported in that unit, areas in its square
    min_area_px: int
        regions with fewer pixels are skipped

    Returns
    -------
    list of dict, sorted by decreasing area. Positions (com_x_px, com_y_px)
    are in pixels; every other length is in units of ``pixel_size``.
    """
    labels = np.asarray(labels)
    intensities = None if image is None else np.asarray(image, dtype=np.float64)
    rows = []
    for label_id, region_slice in enumerate(ndimage.find_objects(labels), start=1):
        if region_slice is None:
            continue
        offset_y, offset_x = region_slice[0].start, region_slice[1].start
        padded_slice = (slice(max(offset_y - 1, 0), region_slice[0].stop + 1),
                        slice(max(offset_x - 1, 0), region_slice[1].stop + 1))
        offset_y, offset_x = padded_slice[0].start, padded_slice[1].start
        mask = labels[padded_slice] == label_id
        area_px = int(np.count_nonzero(mask))
        if area_px < max(min_area_px, 1):
            continue
        yx = np.column_stack(np.nonzero(mask)).astype(np.float64)
        cy, cx = yx.mean(axis=0)
        boundary_yx = np.column_stack(np.nonzero(region_boundary(mask))).astype(np.float64)
        distances = np.hypot(boundary_yx[:, 0] - cy, boundary_yx[:, 1] - cx)
        rmin_px, rmax_px = float(distances.min()), float(distances.max())

        covariance = np.cov((yx - [cy, cx]).T) if area_px > 2 else np.eye(2)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
        major_axis = eigenvalues[-1]
        orientation = 0.0
        if area_px > 2:
            vectors = np.linalg.eigh(covariance)[1]
            orientation = float(np.degrees(np.arctan2(vectors[0, -1], vectors[1, -1])))

        padded = np.pad(mask.astype(np.int8), 1)
        perimeter_px = float(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
        row = {'label': int(label_id),
               'area_px': area_px,
               'area': area_px * pixel_size ** 2,
               'com_x_px': float(cx + offset_x),
               'com_y_px': float(cy + offset_y),
               'equivalent_radius': float(np.sqrt(area_px / np.pi) * pixel_size),
               'rmin': rmin_px * pixel_size,
               'rmax': rmax_px * pixel_size,
               'rmax_over_rmin': rmax_px / max(rmin_px, 1e-9),
               'radial_cv': float(np.std(distances) / max(np.mean(distances), 1e-9)),
               'aspect_ratio': float(np.sqrt(major_axis / max(eigenvalues[0], 1e-9))),
               'orientation_deg': orientation,
               'circularity': float(4 * np.pi * area_px / max(perimeter_px ** 2, 1e-9)),
               'perimeter': perimeter_px * pixel_size,
               'touches_border': bool(region_slice[0].start == 0 or region_slice[1].start == 0
                                      or region_slice[0].stop == labels.shape[0]
                                      or region_slice[1].stop == labels.shape[1])}
        if intensities is not None:
            values = intensities[padded_slice][mask]
            row['mean_intensity'] = float(np.nanmean(values))
            row['integrated_intensity'] = float(np.nansum(values))
        rows.append(row)
    return sorted(rows, key=lambda item: item['area_px'], reverse=True)


def radial_path(labels, label_id, fractions, center_yx=None):
    """Points on a line from a region's centre towards its farthest boundary pixel.

    Parameters
    ----------
    labels: 2D int array
    label_id: int
    fractions: sequence of float
        0 is the centre, 1 the farthest boundary pixel; values > 1 go beyond the edge
    center_yx: (float, float) or None
        start point; centre of mass of the region when None

    Returns
    -------
    numpy.ndarray of shape (len(fractions), 2) with (row, column) positions in pixels
    """
    mask = np.asarray(labels) == label_id
    if not mask.any():
        raise ValueError(f'label {label_id} not present')
    if center_yx is None:
        center_yx = np.column_stack(np.nonzero(mask)).mean(axis=0)
    center_yx = np.asarray(center_yx, dtype=np.float64)
    boundary_yx = np.column_stack(np.nonzero(region_boundary(mask))).astype(np.float64)
    edge_yx = boundary_yx[np.argmax(np.linalg.norm(boundary_yx - center_yx, axis=1))]
    return center_yx[None, :] + np.asarray(fractions, dtype=np.float64)[:, None] * (edge_yx - center_yx)[None, :]


def label_value_map(labels, statistics, key):
    """Image where every region is painted with one of its statistics (NaN for background)."""
    labels = np.asarray(labels)
    lookup = np.full(int(labels.max()) + 1, np.nan)
    for row in statistics:
        lookup[row['label']] = row[key]
    return lookup[labels]
