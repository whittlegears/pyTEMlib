"""
Place a finer image on the coarser image it was taken inside (one magnification step of a multiscale map).

A stage cannot put a 10 to 50 nm field where it was asked to within its own error of hundreds of nanometres,
so an image taken at a finer tier is placed by matching it to its parent image, never by the stage. The child
covers only a small patch of the parent (32 to 51 parent pixels at a 16x step), so the match is made the way
the parent saw that patch: the child is resampled at the parent's pixel spacing, blurred to the parent's
probe, and searched for in the parent around where it was predicted, by normalised cross-correlation with a
sub-pixel peak.

A placement is accepted only when it is clearly the best one: its score passes ``min_score`` and no other
peak in the search window (further than a few parent pixels away) reaches ``max_ambiguity`` of it. Otherwise
the result says "unplaced", with the reason: a wrong placement is worse than none, because everything placed
on top of the child would be wrong with it.

The record (``Placement``) is shaped like an OME-NGFF coordinate transform (scale, rotation, translation)
and names the parent tile it is anchored to, so a re-stitch of the parent map carries the child along.
Version 1 measures the translation; the scale and rotation between tiers come from the scan settings
(the ratio of the two pixel sizes and the scan rotation, both given), and are recorded as given.

Coordinates: images are [row, column]; positions are (row, col) in parent tile pixels, pixel centres at integers.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from skimage.feature import match_template

from pyTEMlib.version import __version__

METHOD = "simulated-parent-ncc"
MIN_TEMPLATE_PX = 8  # a child seen as fewer parent pixels across than this says too little to place it
ANTI_ALIAS_PARENT_PX = 0.4  # the blur (sigma, parent pixels) that keeps detail finer than a parent pixel from aliasing
FINE_SMOOTH_PX = 0.5  # the light smoothing the sub-pixel position is taken from
FINE_SEARCH_PX = 2  # how far from the smoothed peak the fine peak may be
AMBIGUITY_EXCLUDE_PX = 2  # a rival peak must be further than this (parent pixels) from the best to count


@dataclass
class Placement:
    """Where a child image sits in its parent tile, or why it could not be placed."""

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
    parent: dict = field(default_factory=dict)  # the tile it is anchored to: run, index, map version

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_parent_view(child: np.ndarray, scale: float, rotation_rad: float = 0.0, probe_px: float = 0.0) -> np.ndarray:
    """The child as the parent would have imaged it: sampled at the parent's pixel spacing (``scale`` parent pixels
    per child pixel, below 1), after a blur to the parent's probe (``probe_px``, in parent pixels) and the area a
    parent pixel spans, turned by ``rotation_rad`` into the parent's frame."""
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


def register_child(
    child: np.ndarray,
    child_pixel_m: float,
    parent: np.ndarray,
    parent_pixel_m: float,
    predicted_px: tuple[float, float],
    window_px: float,
    rotation_rad: float = 0.0,
    probe_px: float = 0.0,
    min_score: float = 0.5,
    max_ambiguity: float = 0.8,
    smooth_px: float = 1.0,
    parent_tile: dict | None = None,
) -> Placement:
    """Place ``child`` in ``parent``: its centre is searched within ``window_px`` parent pixels of ``predicted_px``
    (row, col in parent pixels, e.g. from the parent's placement plus the stage move, or its own prediction from
    the tier above). Returns a Placement with status "placed" or "unplaced" and the reason.

    Survey images are faint (a few counts a pixel), so both images are first smoothed by ``smooth_px`` parent
    pixels: pixel noise would otherwise keep a right match's correlation low. A match must reach ``min_score``,
    and no other peak further than the smoothed peak's own width may reach ``max_ambiguity`` of it."""
    scale = float(child_pixel_m) / float(parent_pixel_m)
    record = Placement(
        status="unplaced", translation_px=None, scale=scale, rotation_rad=float(rotation_rad),
        inputs_sha256=_hash(child, parent), parent=dict(parent_tile or {}),
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
    parent = np.asarray(parent, dtype=np.float64)
    half = (np.array(template.shape) - 1) / 2.0
    predicted = np.asarray(predicted_px, dtype=np.float64)
    # The part of the parent where the template's top-left can fall when its centre is within the window.
    lo = np.floor(predicted - half - window_px).astype(int)
    hi = np.ceil(predicted - half + window_px).astype(int) + np.array(template.shape)
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
    peak = np.array(np.unravel_index(int(np.argmax(scores)), scores.shape))
    best = float(scores[tuple(peak)])
    others = scores.copy()
    exclude = int(np.ceil(AMBIGUITY_EXCLUDE_PX + 3.0 * smooth_px))  # the smoothed peak's own shoulders are not rivals
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
    if peak[0] in (0, scores.shape[0] - 1) or peak[1] in (0, scores.shape[1] - 1):
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


def _hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()
