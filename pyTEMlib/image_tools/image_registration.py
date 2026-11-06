"""
image_registration.py
by Gerd Duscher, UTK
part of pycroscopy.image
MIT license except where stated differently
"""

import hashlib
import typing
from dataclasses import asdict, dataclass

import numpy as np
import skimage
import scipy
from scipy import ndimage
from skimage.feature import match_template

from tqdm.auto import trange

import sidpy

from ..version import __version__
from .image_utilities import get_pixel_size_m

_SIMPLEITK_PRESENT = True
try:
    import SimpleITK
except ModuleNotFoundError:
    _SIMPLEITK_PRESENT = False
if not _SIMPLEITK_PRESENT:
    print('SimpleITK not installed; Registration Functions for Image Stacks not available')

#####################################################
# Registration Functions
#####################################################

def complete_registration(main_dataset: sidpy.Dataset) -> typing.Tuple[sidpy.Dataset,
                                                                       sidpy.Dataset]:
    """Rigid and then non-rigid (demon) registration

    Performs rigid and then non-rigid registration, please see individual functions:
    - rigid_registration
    - demon_registration

    Parameters
    ----------
    main_dataset: sidpy.Dataset
        dataset of data_type 'IMAGE_STACK' to be registered

    Returns
    -------
    non_rigid_registered: sidpy.Dataset
    rigid_registered_dataset: sidpy.Dataset

    """

    if main_dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    rigid_registered_dataset = rigid_registration(main_dataset)

    rigid_registered_dataset.data_type = 'IMAGE_STACK'

    non_rigid_registered = demon_registration(rigid_registered_dataset)
    return non_rigid_registered, rigid_registered_dataset


def demon_registration(dataset: sidpy.Dataset, verbose: bool=False) -> sidpy.Dataset:
    """
    Diffeomorphic Demon Non-Rigid Registration

    Depends on:
        simpleITK and numpy
    Please Cite: http://www.simpleitk.org/SimpleITK/project/parti.html
    and T. Vercauteren, X. Pennec, A. Perchant and N. Ayache
    Diffeomorphic Demons Using ITK\'s Finite Difference Solver Hierarchy
    The Insight Journal, http://hdl.handle.net/1926/510 2007

    Parameters
    ----------
    dataset: sidpy.Dataset
        stack of image after rigid registration and cropping
    verbose: boolean
        optional for increased output
    Returns
    -------
        dem_reg: stack of images with non-rigid registration

    Example
    -------
    dem_reg = demon_reg(stack_dataset, verbose=False)
    """

    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    dem_reg = np.zeros(dataset.shape)
    nimages = dataset.shape[0]
    if verbose:
        print(nimages)
    # create fixed image by summing over rigid registration

    fixed_np = np.average(np.array(dataset), axis=0)

    if not _SIMPLEITK_PRESENT:
        print('This feature is not available:')
        print('Please install simpleITK with: conda install simpleitk -c simpleitk')

    fixed = SimpleITK.GetImageFromArray(fixed_np)
    fixed = SimpleITK.DiscreteGaussian(fixed, 2.0)

    demons = SimpleITK.DiffeomorphicDemonsRegistrationFilter()

    demons.SetNumberOfIterations(200)
    demons.SetStandardDeviations(1.0)

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(SimpleITK.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    for i in trange(nimages):
        moving = SimpleITK.GetImageFromArray(dataset[i])
        moving_f = SimpleITK.DiscreteGaussian(moving, 2.0)
        displacement_field = demons.Execute(fixed, moving_f)
        out_tx = SimpleITK.DisplacementFieldTransform(displacement_field)
        resampler.SetTransform(out_tx)
        out = resampler.Execute(moving)
        dem_reg[i, :, :] = SimpleITK.GetArrayFromImage(out)

    print(':-)')
    print('You have successfully completed Diffeomorphic Demons Registration')

    demon_registered = dataset.like_data(dem_reg)
    demon_registered.title = 'Non-Rigid Registration'
    demon_registered.source = dataset.title

    demon_registered.metadata =dataset.metadata.copy()
    if 'analysis' not in demon_registered.metadata:
        demon_registered.metadata['analysis'] = {}
    demon_registered.metadata['analysis']['non_rigid_demon_registration'] = {'package': 'simpleITK',
                                                                             'method': 'DiscreteGaussian',
                                                                             'variance': 2,
                                                                             'input_dataset': dataset.source}
    demon_registered.data_type = 'IMAGE_STACK'
    return demon_registered


# ##############################
# Rigid Registration New 05/09/2024
# ##############################
def rigid_registration(dataset: sidpy.Dataset, normalization: typing.Optional[str] = None) -> sidpy.Dataset:
    """
    Rigid registration of image stack with pixel accuracy

    Uses simple cross_correlation
    (we determine drift from one image to next)

    Parameters
    ----------
    dataset: sidpy.Dataset
        sidpy dataset with image_stack dataset
    normalization: str or None
        if 'phase' then phase cross correlation is used, otherwise
        normalized cross correlation is used

    Returns
    -------
    rigid_registered: sidpy.Dataset
        Registered Stack and drift (with respect to center image)
    """

    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    if isinstance (normalization, str):
        if normalization.lower() != 'phase':
            normalization = None
    else:
        normalization = None
    image_dimensions = dataset.get_image_dims(return_axis=True)
    if dataset.get_dimensions_by_type('TEMPORAL')[0] != 0:
        x = image_dimensions[0]
        y = image_dimensions[1]
        z = dataset.get_dimensions_by_type('TEMPORAL', return_axis=True)[0]
        metadata = dataset.metadata.copy()
        original_metadata = dataset.original_metadata.copy()
        arr = np.rollaxis(np.array(dataset), 2, 0)
        dataset = sidpy.Dataset.from_array(arr, title=dataset.title, data_type='IMAGE_STACK',
                                           quantity=dataset.quantity, units=dataset.units)
        dataset.set_dimension(0, sidpy.Dimension(z.values, name='frame', units='frame', quantity='time',
                                                  dimension_type='temporal'))
        dataset.set_dimension(1, x)
        dataset.set_dimension(2, y)
        dataset.metadata = metadata
        dataset.original_metadata = original_metadata

    stack_dim = dataset.get_dimensions_by_type('TEMPORAL', return_axis=True)[0]
    image_dim = dataset.get_image_dims(return_axis=True)
    if len(image_dim) != 2:
        raise ValueError('need at least two SPATIAL dimension for an image stack')

    relative_drift = [[0., 0.]]
    im1 = np.fft.fft2(np.array(dataset[0]))
    for i in range(1, len(stack_dim)):
        im2 = np.fft.fft2(np.array(dataset[i]))
        shift, error, _ = skimage.registration.phase_cross_correlation(im1, im2,
                                                                       normalization=normalization,
                                                                       space='fourier')
        im1 = im2.copy()
        relative_drift.append(shift)

    rig_reg, drift = rig_reg_drift(dataset, relative_drift)
    crop_reg, input_crop = crop_image_stack(rig_reg, drift)

    rigid_registered = sidpy.Dataset.from_array(crop_reg,
                                                title='Rigid Registration',
                                                data_type='IMAGE_STACK',
                                                quantity=dataset.quantity,
                                                units=dataset.units)
    rigid_registered.title = 'Rigid_Registration'
    rigid_registered.source = dataset.title
    rigid_registered.metadata['analysis'] = {'rigid_registration': {'drift': drift,
                                 'input_crop': input_crop, 'input_shape': dataset.shape[1:]}}

    if 'experiment' in dataset.metadata:
        rigid_registered.metadata['experiment'] = dataset.metadata['experiment'].copy()
    rigid_registered.set_dimension(0, sidpy.Dimension(np.arange(rigid_registered.shape[0]),
                                          name='frame', units='frame', quantity='time',
                                          dimension_type='temporal'))

    array_x = image_dim[0].values[input_crop[0]:input_crop[1]]
    rigid_registered.set_dimension(1, sidpy.Dimension(array_x, name='x',
                                                      units='nm', quantity='Length',
                                                      dimension_type='spatial'))
    array_y =image_dim[1].values[input_crop[2]:input_crop[3]]
    rigid_registered.set_dimension(2, sidpy.Dimension(array_y, name='y',
                                                      units='nm', quantity='Length',
                                                      dimension_type='spatial'))
    rigid_registered.data_type = 'IMAGE_STACK'
    return rigid_registered.rechunk({0: 'auto', 1: -1, 2: -1})


def rig_reg_drift(dset: sidpy.Dataset,
                  rel_drift: typing.Union[typing.List[typing.List[float]], np.ndarray]
                  ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """ Shifting images on top of each other

    Uses relative drift to shift images on top of each other,
    with center image as reference.
    Shifting is done with shift routine of ndimage from scipy.
    This function is used by rigid_registration routine

    Parameters
    ----------
    dset: sidpy.Dataset
        dataset with image_stack
    rel_drift:
        relative_drift from image to image as list of [shiftx, shifty]

    Returns
    -------
    stack: numpy array
    drift: list of drift in pixel
    """

    frame_dim = []
    spatial_dim = []
    selection = []

    for i, axis in dset._axes.items():
        if axis.dimension_type.name == 'SPATIAL':
            spatial_dim.append(i)
            selection.append(slice(None))
        else:
            frame_dim.append(i)
            selection.append(slice(0, 1))

    if len(spatial_dim) != 2:
        print('need two spatial dimensions')
    if len(frame_dim) != 1:
        print('need one frame dimensions')

    rig_reg = np.zeros([dset.shape[frame_dim[0]], dset.shape[spatial_dim[0]], dset.shape[spatial_dim[1]]])

    # absolute drift
    drift = np.array(rel_drift).copy()

    drift[0] = [0, 0]
    for i in range(1, drift.shape[0]):
        drift[i] = drift[i - 1] + rel_drift[i]
    center_drift = drift[int(drift.shape[0] / 2)]
    drift = drift - center_drift
    # Shift images
    for i in range(rig_reg.shape[0]):
        selection[frame_dim[0]] = slice(i, i+1)
        # Now we shift
        rig_reg[i, :, :] = scipy.ndimage.shift(dset[tuple(selection)].squeeze().compute(),
                                         [drift[i, 0], drift[i, 1]], order=3)
    return rig_reg, drift



def crop_image_stack(rig_reg: np.ndarray, drift: typing.Union[np.ndarray, list]
                     ) -> typing.Tuple[np.ndarray, list[int]]:
    """Crop images in stack according to drift

    This function is used by rigid_registration routine

    Parameters
    ----------
    rig_reg: numpy array (N,x,y)
    drift: list (2,B)

    Returns
    -------
    numpy array
    """
    xpmax = int(rig_reg.shape[1] - -np.floor(np.min(np.array(drift)[:, 0])))
    xpmin = int(np.ceil(np.max(np.array(drift)[:, 0])))
    ypmax = int(rig_reg.shape[1] - -np.floor(np.min(np.array(drift)[:, 1])))
    ypmin = int(np.ceil(np.max(np.array(drift)[:, 1])))

    return rig_reg[:, xpmin:xpmax, ypmin:ypmax:], [xpmin, xpmax, ypmin, ypmax]


#####################################################
# Placing a finer image on its coarser parent
#####################################################

METHOD = "simulated-parent-ncc"
MIN_TEMPLATE_PX = 8  # a child seen as fewer parent pixels across than this says too little to place it
ANTI_ALIAS_PARENT_PX = 0.4  # the blur (sigma, parent pixels) that keeps detail finer than a parent pixel from aliasing
FINE_SMOOTH_PX = 0.5  # the light smoothing the sub-pixel position is taken from
FINE_SEARCH_PX = 2  # how far from the smoothed peak the fine peak may be
AMBIGUITY_EXCLUDE_PX = 2  # a rival peak must be further than this (parent pixels) from the best to count


@dataclass
class ChildPlacement:
    """Where a child image sits in its parent image, or why it could not be placed

    Shaped like an OME-NGFF coordinate transform (scale, rotation, translation). The translation is
    measured; the scale and rotation between the two images come from their pixel sizes and the
    scan rotation given, and are recorded as given. Callers add where the parent came from.

    Attributes
    ----------
    status: str
        "placed" or "unplaced"
    translation_px: list of float or None
        (row, col) of the child's centre in parent pixels, when placed
    scale: float
        parent pixels per child pixel
    rotation_rad: float
        the child's scan rotation relative to the parent's, as given
    score: float or None
        normalised cross-correlation at the peak (low on faint images even when right)
    ambiguity: float or None
        best rival peak / best peak
    uncertainty_px: list of float or None
        rough 1-sigma of the translation, from the peak's sharpness
    reason: str
        why the child was not placed
    method, method_version, inputs_sha256: str
        how the placement was made, and a hash of both images
    """

    status: str  # "placed" or "unplaced"
    translation_px: list[float] | None  # (row, col) of the child's centre in parent tile pixels
    scale: float  # parent pixels per child pixel (given: child pixel size / parent pixel size)
    rotation_rad: float  # given: the child's scan rotation relative to the parent's
    score: float | None = None  # normalised cross-correlation at the peak (low on faint images even when right)
    ambiguity: float | None = None  # best rival peak / best peak
    uncertainty_px: list[float] | None = None  # rough 1-sigma of the translation, from the peak's sharpness
    reason: str = ""
    method: str = METHOD
    method_version: str = __version__
    inputs_sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_parent_view(child: np.ndarray, scale: float, rotation_rad: float = 0.0, probe_px: float = 0.0) -> np.ndarray:
    """A child image as its parent would have imaged it

    Sampled at the parent's pixel spacing, after a blur to the parent's probe
    and the area a parent pixel spans, turned into the parent's frame.

    Parameters
    ----------
    child: numpy array (2D)
        the finer image
    scale: float
        parent pixels per child pixel (child pixel size / parent pixel size), between 0 and 1
    rotation_rad: float
        the child's scan rotation relative to the parent's
    probe_px: float
        the parent's probe size (Gaussian sigma) in parent pixels

    Returns
    -------
    view: numpy array (2D)
        the child at the parent's pixel spacing
    """
    child = np.asarray(child, dtype=np.float64)
    if not 0 < scale < 1:
        raise ValueError(f"scale must be the parent pixels per child pixel, between 0 and 1; got {scale}")
    # A parent pixel spans 1 / scale child pixels; blur by that span and the probe before sampling, so the sample
    # is not aliased (the parent's own pixels are point samples of an object already smoothed by its probe).
    sigma = np.hypot(ANTI_ALIAS_PARENT_PX, probe_px) / scale  # in child pixels
    smooth = ndimage.gaussian_filter(child, sigma)
    rows, cols = child.shape
    out = (np.array(child.shape) * scale).astype(int)
    if min(out) < 1:
        raise ValueError("the child is smaller than one parent pixel")
    centre_out = (out - 1) / 2.0
    centre_in = (np.array(child.shape) - 1) / 2.0
    r, c = np.meshgrid(np.arange(out[0]) - centre_out[0], np.arange(out[1]) - centre_out[1], indexing="ij")
    cos, sin = np.cos(-rotation_rad), np.sin(-rotation_rad)
    rr = (cos * r - sin * c) / scale + centre_in[0]
    cc = (sin * r + cos * c) / scale + centre_in[1]
    return ndimage.map_coordinates(smooth, [rr, cc], order=3, mode="nearest")


def register_to_parent(child: sidpy.Dataset, parent: sidpy.Dataset, predicted_px: typing.Sequence[float],
                       window_px: float, rotation_rad: float = 0.0, probe_px: float = 0.0, min_score: float = 0.5,
                       max_ambiguity: float = 0.8, smooth_px: float = 1.0) -> ChildPlacement:
    """Place a finer image on the coarser image it was taken inside

    One magnification step of a multiscale map: a stage cannot put a 10 to 50 nm
    field where it was asked to within its own error of hundreds of nanometres,
    so the child is placed by matching it to its parent. The child is resampled
    at the parent's pixel spacing, blurred to the parent's probe (see
    ``simulate_parent_view``) and searched for in the parent within ``window_px``
    parent pixels of ``predicted_px``, by normalised cross-correlation with a
    sub-pixel peak.

    Survey images are faint (a few counts a pixel), so both images are first
    smoothed by ``smooth_px`` parent pixels: pixel noise would otherwise keep a
    right match's correlation low. A placement is accepted only when it is
    clearly the best one: its score reaches ``min_score`` and no other peak in or
    just beyond the window (further away than the smoothed peak's own width)
    reaches ``max_ambiguity`` of it. Otherwise the result is "unplaced", with the
    reason: a wrong placement is worse than none.

    Parameters
    ----------
    child: sidpy.Dataset
        the finer image, with calibrated spatial dimensions
    parent: sidpy.Dataset
        the coarser image (or the part of it around the prediction), with calibrated spatial dimensions
    predicted_px: (float, float)
        (row, col) in parent pixels where the child's centre is expected, pixel centres at integers
    window_px: float
        how far from the prediction to search, in parent pixels
    rotation_rad: float
        the child's scan rotation relative to the parent's
    probe_px: float
        the parent's probe size (Gaussian sigma) in parent pixels
    min_score: float
        normalised cross-correlation the best match must reach
    max_ambiguity: float
        the most a rival peak may reach, as a share of the best
    smooth_px: float
        smoothing (Gaussian sigma, parent pixels) of both images before the match is judged

    Returns
    -------
    placement: ChildPlacement
        status "placed" with the translation (the child's centre in parent pixels), score, ambiguity and
        uncertainty, or "unplaced" with the reason
    """
    child_pixel_m = _square_pixel_m(child, 'child')
    parent_pixel_m = _square_pixel_m(parent, 'parent')
    child = np.asarray(child, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    scale = child_pixel_m / parent_pixel_m
    record = ChildPlacement(
        status="unplaced", translation_px=None, scale=scale, rotation_rad=float(rotation_rad),
        inputs_sha256=_hash(child, parent),
    )
    try:
        template = simulate_parent_view(child, scale, rotation_rad, probe_px)
    except ValueError as exc:
        record.reason = str(exc)
        return record
    if min(template.shape) < MIN_TEMPLATE_PX:
        record.reason = f"the child is only {min(template.shape)} parent pixels across; at least {MIN_TEMPLATE_PX} are needed to place it"
        return record
    if template.std() <= 1e-9 * (abs(float(template.mean())) + 1.0):
        record.reason = "the child is blank as the parent would see it: nothing to match"
        return record
    half = (np.array(template.shape) - 1) / 2.0
    predicted = np.asarray(predicted_px, dtype=np.float64)
    # The part of the parent where the template's top-left can fall when its centre is within the window, and as far
    # again as a rival must be from the best: a rival just outside the window still makes the match a guess.
    exclude = int(np.ceil(AMBIGUITY_EXCLUDE_PX + 3.0 * smooth_px))  # the smoothed peak's own shoulders are not rivals
    reach = window_px + exclude
    lo = np.floor(predicted - half - reach).astype(int)
    hi = np.ceil(predicted - half + reach).astype(int) + np.array(template.shape)
    lo_c, hi_c = np.maximum(lo, 0), np.minimum(hi, parent.shape)
    region = parent[lo_c[0] : hi_c[0], lo_c[1] : hi_c[1]]
    if region.shape[0] < template.shape[0] + 2 or region.shape[1] < template.shape[1] + 2:
        record.reason = "the search window falls outside the parent image"
        return record
    # Smoothed images decide whether there is a clear match; lightly smoothed ones say where it is to a fraction
    # of a pixel (smoothing that quiets noise enough to judge a match also blunts the peak it is judged on).
    fine = match_template(ndimage.gaussian_filter(region, FINE_SMOOTH_PX), ndimage.gaussian_filter(template, FINE_SMOOTH_PX))
    if smooth_px > 0:
        region = ndimage.gaussian_filter(region, smooth_px)
        template = ndimage.gaussian_filter(template, smooth_px)
    scores = match_template(region, template)
    rows, cols = np.indices(scores.shape)
    inside = (np.abs(lo_c[0] + rows + half[0] - predicted[0]) <= window_px) & (np.abs(lo_c[1] + cols + half[1] - predicted[1]) <= window_px)
    if not inside.any():
        record.reason = "the search window falls outside the parent image"
        return record
    peak = np.array(np.unravel_index(int(np.argmax(np.where(inside, scores, -np.inf))), scores.shape))
    best = float(scores[tuple(peak)])
    climbing = scores[max(0, peak[0] - 1) : peak[0] + 2, max(0, peak[1] - 1) : peak[1] + 2].max() > best
    others = scores.copy()
    r0, c0 = np.maximum(peak - exclude, 0)
    others[r0 : peak[0] + exclude + 1, c0 : peak[1] + exclude + 1] = -np.inf
    rival = float(others.max()) if np.isfinite(others).any() else -np.inf
    record.score = round(best, 4)
    record.ambiguity = round(rival / best, 4) if best > 0 and np.isfinite(rival) else 0.0
    near = fine[max(0, peak[0] - FINE_SEARCH_PX) : peak[0] + FINE_SEARCH_PX + 1, max(0, peak[1] - FINE_SEARCH_PX) : peak[1] + FINE_SEARCH_PX + 1]
    fine_peak = np.array(np.unravel_index(int(np.argmax(near)), near.shape)) + np.maximum(peak - FINE_SEARCH_PX, 0)
    offset, sharpness = _subpixel(fine, fine_peak)
    peak = fine_peak
    centre = lo_c + peak + offset + half
    if climbing or peak[0] in (0, scores.shape[0] - 1) or peak[1] in (0, scores.shape[1] - 1):
        record.reason = "the best match is on the edge of the search window: the child may lie outside it"
        return record
    if np.hypot(*(centre - predicted)) > window_px * np.sqrt(2.0) + 1.0:
        record.reason = "the best match is outside the search window"
        return record
    if best < min_score:
        record.reason = f"the best match scores {best:.2f}, below {min_score}: the parent shows too little of what the child shows"
        return record
    if record.ambiguity > max_ambiguity:
        record.reason = f"another place matches almost as well ({record.ambiguity:.2f} of the best): a guess between places, not a placement"
        return record
    record.status = "placed"
    record.translation_px = [round(float(v), 4) for v in centre]
    record.uncertainty_px = [round(float(v), 4) for v in sharpness]
    return record


def _subpixel(scores: np.ndarray, peak: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parabola through the peak and its neighbours on each axis: (offset in pixels, a rough 1-sigma from the curvature)."""
    offset, sigma = np.zeros(2), np.full(2, 0.5)
    for axis in range(2):
        step = np.zeros(2, int)
        step[axis] = 1
        if peak[axis] in (0, scores.shape[axis] - 1):
            continue
        y0, y1, y2 = (float(scores[tuple(peak + d * step)]) for d in (-1, 0, 1))
        curvature = y0 - 2 * y1 + y2
        if curvature < 0:
            offset[axis] = np.clip(0.5 * (y0 - y2) / curvature, -0.5, 0.5)
            sigma[axis] = min(0.5, float(np.sqrt(max(1e-6, 1.0 - y1) / -curvature)) * 0.5)
    return offset, sigma


def _square_pixel_m(dataset: sidpy.Dataset, name: str) -> float:
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError(f'the {name} must be a sidpy.Dataset')
    size = get_pixel_size_m(dataset)
    if abs(size[0] - size[1]) > 1e-6 * size[0]:
        raise ValueError(f'the {name} needs square pixels; it has {size[0]:.4g} x {size[1]:.4g} m')
    return float(size[0])


def _hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()
