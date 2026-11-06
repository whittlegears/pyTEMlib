"""
image_scoring part of image package of pycroscopy

Rank the regions of an image by how good they are as a field of view.

A scorer takes one region (a 2-D array) and its metadata and returns a value
between 0 and 1 and a short human-readable reason; it carries a name and a
version, so scores from different scorers (and versions) can be told apart
later. Trained models, hand-written rules and LLM judges can all implement the
``Scorer`` protocol. ``score_windows`` scores every window of a sidpy image
dataset and ``select_regions`` keeps the best windows that do not overlap.

The baseline scorer: in-focus feature edges, measured per feature.

A deliberately simple, explainable stand-in until trained or physics-based
scorers exist. It ranks a region higher when it holds features whose edges are
sharp (in focus) and stand out from the noise. Empty support film, defocused
and contaminated regions score near 0.

How 3.0.0 measures it, per feature rather than over the whole region:

1. Smooth by 1 pixel and take the gradient magnitude G. Edge points are
   ridges of G across the edge, at 1 and at ``EDGE_PROBE_PX`` pixels, where G
   is at least ``EDGE_SIGNIFICANCE`` times its noise level. The noise is
   estimated locally (median absolute high-pass residual in a 7 x 7
   neighbourhood), so bright, noisier particles are judged against their own
   noise. The inside of a large domed particle (strong but monotonic gradient)
   is not a ridge at 2 pixels and is left out.
2. Edge points are grouped into features (connected edge contours of at least
   ``MIN_EDGE_POINTS`` points; shorter ones are noise).
3. Each feature's edge width comes from how fast G falls off across its edge,
   ``EDGE_PROBE_PX`` pixels out on the darker side (medians over its edge
   points): for an edge blurred by a Gaussian of width s,
   G(t) / G(0) = exp(-t^2 / 2 s^2). This ratio does not depend on the height of
   the step, so overlapping particles (whose summed thickness doubles the
   height) or a bright particle next to a faint one do not read as blur, and it
   does not depend on how much of the region the features cover. A faint,
   short edge is judged by the blurred end of its uncertainty
   (``UNCERTAINTY_K`` standard errors), because the noise draws that get
   detected are the ones that look sharp.
4. The region's edge width and edge contrast are medians over features, each
   feature weighted by its edge points capped at ``FEATURE_WEIGHT_CAP`` (so one
   large particle does not outvote several small ones, and a single small
   particle counts like a large one) times its reliability
   (significance / ``SIGNIFICANCE_CAP``, squared, at most 1).

score = sharpness(edge width) x contrast(edge significance), each in 0-1.

This is a detection proxy, checked against synthetic maps with known
particles of several kinds (separated, crowded and overlapping, single small,
large, different pixel sampling, low contrast) and deliberately defocused,
contaminated and empty regions; it has not been checked against what a
microscopist would choose. Edge widths are in pixels: a feature must be
resolved at the pixel size scored, and a finite probe heavily oversampled
would read as blur.

History: 1.0.0 measured the fraction of the region covered by edges and
rewarded blur. 2.0.0 measured one global edge width (99th-percentile height /
steepest gradient); overlapping particles doubled the height and read as blur,
and a feature covering under ~1 % of the region was not reached by the 99th
percentile, so crowded regions and fields sampled finer than the particles
ranked at the bottom.
"""

import typing

import numpy as np
from scipy import ndimage

import sidpy

from .image_utilities import get_pixel_size_m

# For white noise of standard deviation 1 smoothed by a 1-pixel Gaussian: the high-pass residual
# (data - smooth) has std 0.872 and each np.gradient component of the smoothed image has std 0.159.
_RESIDUAL_STD = 0.872
_GRADIENT_STD = 0.159

EDGE_SIGNIFICANCE = 5.0  # an edge point's gradient must be this many noise standard deviations
EDGE_PROBE_PX = 2.0  # sample the gradient this far to either side of the edge
MIN_EDGE_POINTS = 8  # shorter edge contours are noise (or a hot pixel); a 2-pixel-radius particle has ~14
FEATURE_WEIGHT_CAP = 24  # edge points counted per feature when weighting the median
UNCERTAINTY_K = 1.5  # judge a feature by its ratio plus this many standard errors
SIGNIFICANCE_CAP = 20.0  # features with edges this far above noise count fully; fainter ones as (significance / cap)^2
# Edge width (Gaussian sigma, pixels, including the 1-pixel smoothing) where sharpness is 0.5.
# Tuned on seeds 0-2 of synthetic-fov-2 only: an in-focus edge measures ~1.2-1.7 px; a defocus
# blur of 3-5 px or a contamination layer measures ~1.9-5 px (after the uncertainty margin).
EDGE_WIDTH_MID_PX = 1.8
EDGE_WIDTH_SCALE_PX = 0.12
CONTRAST_SCALE = 8.0  # contrast = 1 - exp(-significance / CONTRAST_SCALE)


def _sigmoid_low(x: float, mid: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(np.clip((x - mid) / scale, -50, 50))))


def _feature_weight(feature: dict) -> float:
    """Edge points (capped, so size does not dominate) x reliability (the width's variance falls as significance^2, capped)."""
    reliability = min(feature["significance"] / SIGNIFICANCE_CAP, 1.0) ** 2
    return float(min(feature["edge_points"], FEATURE_WEIGHT_CAP) * reliability)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


@typing.runtime_checkable
class Scorer(typing.Protocol):
    """How good a region of an image is as a field of view

    Attributes
    ----------
    name: str
        short name of the scorer
    version: str
        changes whenever the scorer's values change
    description: str
        what it rewards, in a sentence or two
    """

    name: str
    version: str
    description: str

    def score(self, region: np.ndarray, metadata: dict) -> typing.Tuple[float, str]:
        """Return (value in [0, 1], reason) for one region; metadata holds at least pixel_size_m."""


class BaselineScorer:
    """In-focus feature edges: per-feature edge sharpness times edge contrast (see the module docstring)."""

    name = "baseline"
    version = "3.0.0"
    description = (
        "Per-feature edge sharpness x edge contrast, each mapped to 0-1: the median edge width over detected features, "
        "independent of feature height, size and coverage. Empty, defocused or contaminated regions score low. A detection proxy."
    )

    def measure(self, region: np.ndarray) -> dict:
        """Edge measurements behind the score: per-feature edge widths (px) and significances, and the noise."""
        data = np.asarray(region, dtype=np.float64)
        smooth = ndimage.gaussian_filter(data, 1.0, mode="nearest")
        residual = data - smooth
        noise_global = 1.4826 * float(np.median(np.abs(residual))) / _RESIDUAL_STD + 1e-9
        local = 1.4826 * ndimage.median_filter(np.abs(residual), size=7, mode="nearest") / _RESIDUAL_STD
        gradient_noise = _GRADIENT_STD * np.maximum(local, 0.25 * noise_global)
        gy, gx = np.gradient(smooth)
        magnitude = np.hypot(gy, gx)
        significance = magnitude / gradient_noise
        candidates = np.argwhere(significance >= EDGE_SIGNIFICANCE)
        result = {"noise": noise_global, "features": []}
        if len(candidates) == 0:
            return result
        rows, cols = candidates[:, 0].astype(np.float64), candidates[:, 1].astype(np.float64)
        g0 = magnitude[candidates[:, 0], candidates[:, 1]]
        ny, nx = gy[candidates[:, 0], candidates[:, 1]] / g0, gx[candidates[:, 0], candidates[:, 1]] / g0

        def sample(t: float) -> np.ndarray:
            return ndimage.map_coordinates(magnitude, [rows + t * ny, cols + t * nx], order=1, mode="nearest")

        t = EDGE_PROBE_PX
        near, far = (sample(1.0), sample(-1.0)), (sample(t), sample(-t))
        # An edge is a ridge of G at both 1 and t pixels; the inside of a large, thick particle has a strong
        # but monotonic gradient (rising towards its rim), which is not an edge and is left out here.
        ridge = (g0 >= near[0]) & (g0 >= near[1]) & (g0 > far[0]) & (g0 > far[1])
        if not ridge.any():
            return result
        points = candidates[ridge]
        g0, uphill, downhill = g0[ridge], far[0][ridge], far[1][ridge]
        mask = np.zeros(data.shape, bool)
        mask[points[:, 0], points[:, 1]] = True
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        point_label = labels[points[:, 0], points[:, 1]]
        point_significance = significance[points[:, 0], points[:, 1]]
        for index in range(1, n + 1):
            mine = point_label == index
            count = int(mine.sum())
            if count < MIN_EDGE_POINTS:
                continue
            # The width comes from the feature's edge profile (medians over its edge points), not from each point:
            # a single point near the noise floor, picked because noise raised its gradient, would look sharp.
            # Measured on the darker side of the edge (the support around a bright particle): the bright side of a
            # domed particle keeps rising towards its centre, and inside a small blob the gradient falls to zero.
            edge = np.median(g0[mine])
            ratio = np.median(downhill[mine]) / edge
            # Faint, short edges give a noisy ratio, and the brightest noise draws are the ones detected, which look
            # sharp. Judge them by the blurred end of their uncertainty: the ratio's standard error is about the
            # gradient noise over the edge strength, over the root of the independent edge points (~1 in 3).
            noise_ratio = float(np.median(g0[mine] / point_significance[mine])) / edge
            ratio += UNCERTAINTY_K * noise_ratio * np.sqrt(1.0 + ratio * ratio) / np.sqrt(count / 3.0)
            width = t / np.sqrt(2.0 * np.log(1.0 / np.clip(ratio, 1e-3, 0.995)))
            result["features"].append({
                "edge_points": count,
                "edge_width_px": float(width),
                "significance": float(np.median(point_significance[mine])),
            })
        return result

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        data = np.asarray(region, dtype=np.float64)
        if data.size < 16 or min(data.shape[:2]) < 8 or not np.isfinite(data).all():
            return 0.0, "region too small or not finite"
        measured = self.measure(data)
        features = measured["features"]
        if not features:
            return 0.0, f"no feature edges above {EDGE_SIGNIFICANCE:g} x noise (noise {measured['noise']:.3g})"
        weights = np.array([_feature_weight(f) for f in features])
        width = _weighted_median(np.array([f["edge_width_px"] for f in features]), weights)
        strength = _weighted_median(np.array([f["significance"] for f in features]), weights)
        sharpness = _sigmoid_low(width, EDGE_WIDTH_MID_PX, EDGE_WIDTH_SCALE_PX)
        contrast = 1.0 - float(np.exp(-strength / CONTRAST_SCALE))
        sharp_count = sum(f["edge_width_px"] < EDGE_WIDTH_MID_PX for f in features)
        value = float(np.clip(sharpness * contrast, 0.0, 1.0))
        reason = (
            f"{len(features)} features, {sharp_count} with sharp edges; median edge width {width:.2f} px "
            f"(sharpness {sharpness:.2f}); edge contrast {strength:.1f} x noise ({contrast:.2f})"
        )
        return value, reason


def score_windows(dataset: sidpy.Dataset, scorer: typing.Optional[Scorer] = None, window: int = 64,
                  stride: typing.Optional[int] = None) -> typing.List[dict]:
    """Score every window of an image

    Parameters
    ----------
    dataset: sidpy.Dataset
        image (data_type IMAGE) with calibrated spatial dimensions
    scorer: Scorer or None
        defaults to BaselineScorer
    window: int
        window side in pixels (shrunk to the image when the image is smaller)
    stride: int or None
        step between windows in pixels; defaults to the window (no overlap)

    Returns
    -------
    scores: list of dict
        one per window: ``row``, ``col`` (top-left pixel), ``window`` (pixels), ``value``
        (in [0, 1]) and ``reason``. The scorer is given the window's pixels and
        ``{'pixel_size_m', 'row', 'col', 'window'}``, the pixel size read from the
        spatial dimensions.
    """
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if dataset.data_type.name != 'IMAGE':
        raise TypeError('We need sidpy.Dataset of sidpy.Datatype: IMAGE')
    scorer = BaselineScorer() if scorer is None else scorer
    pixel_size_m = float(np.mean(get_pixel_size_m(dataset)))
    image = np.asarray(dataset)
    window = int(window)
    stride = int(stride or window)
    if window < 1 or stride < 1:
        raise ValueError('window and stride must be at least one pixel')
    h, w = image.shape[:2]
    window = min(window, h, w)
    out = []
    for r in range(0, max(1, h - window + 1), stride):
        for c in range(0, max(1, w - window + 1), stride):
            metadata = {'pixel_size_m': pixel_size_m, 'row': r, 'col': c, 'window': window}
            value, reason = scorer.score(image[r:r + window, c:c + window], metadata)
            out.append({'row': r, 'col': c, 'window': window, 'value': float(np.clip(value, 0.0, 1.0)), 'reason': reason})
    return out


def select_regions(scores: typing.Sequence[dict], top_k: int = 5,
                   min_separation: typing.Optional[int] = None) -> typing.List[dict]:
    """The best-scoring windows, each at least ``min_separation`` away from every better one

    Greedy non-maximum suppression: windows are taken best first, and a window is
    skipped when both its row and its column are within ``min_separation`` pixels
    of a window already taken (so half-overlapping windows do not all pick the
    same feature).

    Parameters
    ----------
    scores: list of dict
        as returned by ``score_windows`` (``row``, ``col``, ``window``, ``value``)
    top_k: int
        how many windows to keep at most
    min_separation: int or None
        in pixels; defaults to each window's own size

    Returns
    -------
    regions: list of dict
        the kept entries of ``scores``, best first
    """
    ranked = sorted(scores, key=lambda s: s['value'], reverse=True)
    kept: typing.List[dict] = []
    for s in ranked:
        if len(kept) >= int(top_k):
            break
        apart = s['window'] if min_separation is None else min_separation
        if all(abs(s['row'] - t['row']) >= apart or abs(s['col'] - t['col']) >= apart for t in kept):
            kept.append(s)
    return kept
