"""
Baseline scorer: in-focus feature edges, measured per feature.

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

This is a detection proxy. ``pyTEMlib.scoring.evaluate`` checks it against
synthetic maps with known particles of several kinds (separated, crowded and
overlapping, single small, large, different pixel sampling, low contrast) and
deliberately defocused, contaminated and empty regions; it has not been checked
against what a microscopist would choose. Edge widths are in pixels: a feature
must be resolved at the pixel size scored (``score_map`` picks the level), and
a finite probe heavily oversampled would read as blur.

History: 1.0.0 measured the fraction of the region covered by edges and
rewarded blur. 2.0.0 measured one global edge width (99th-percentile height /
steepest gradient); overlapping particles doubled the height and read as blur,
and a feature covering under ~1 % of the region was not reached by the 99th
percentile, so crowded regions and fields sampled finer than the particles
ranked at the bottom. 2.0.0 stays selectable as ``baseline_v2`` for comparison.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

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


class BaselineScorer:
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


# ---------------------------------------------------------------- 2.0.0, kept for comparison

V2_EDGE_WIDTH_MID_PX = 4.8
V2_EDGE_WIDTH_SCALE_PX = 0.3


class BaselineScorerV2:
    """Baseline 2.0.0: one global edge width (99th-percentile height / steepest gradient). Superseded by 3.0.0."""

    name = "baseline_v2"
    version = "2.0.0"
    description = "Superseded baseline 2.0.0 (global edge width), kept for comparison: crowded, overlapping or small features read as blurred."

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        data = np.asarray(region, dtype=np.float64)
        if data.size < 16 or min(data.shape[:2]) < 4 or not np.isfinite(data).all():
            return 0.0, "region too small or not finite"
        smooth = ndimage.gaussian_filter(data, 1.0)
        background = float(np.median(smooth))
        spread = float(np.percentile(smooth, 99) - background)
        noise = float(np.std(data - smooth)) + 1e-9
        contrast = 1.0 - np.exp(-max(spread, 0.0) / (6.0 * noise))
        steepest = float(np.percentile(np.hypot(*np.gradient(smooth)), 99))
        edge_width = max(spread, 1e-12) / max(steepest, 1e-12)
        sharpness = _sigmoid_low(edge_width, V2_EDGE_WIDTH_MID_PX, V2_EDGE_WIDTH_SCALE_PX)
        value = float(np.clip(contrast * sharpness, 0.0, 1.0))
        reason = f"contrast {contrast:.2f} (features {spread:.3g} above background, noise {noise:.3g}); edge width {edge_width:.1f} px, sharpness {sharpness:.2f}"
        return value, reason
