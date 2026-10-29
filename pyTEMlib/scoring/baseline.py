"""
Baseline scorer: contrast and edge sharpness.

A deliberately simple, explainable stand-in until trained or physics-based
scorers exist. It ranks a region higher when it has (a) features that stand
out from the background (relative contrast) and (b) steep edges for their
height, i.e. features in focus rather than blurred. Empty support film scores
near 0; a field of well-separated, in-focus particles scores high.

This is a detection proxy. ``pyTEMlib.scoring.evaluate`` checks it against
synthetic maps with known particles and deliberately defocused or contaminated
regions; it has not been checked against what a microscopist would choose.

Version 1.0.0 measured sharpness as the fraction of the region covered by
edges. Blurring a particle widens its edges, so defocused regions scored
*higher* than sharp ones (precision@5 = 0.0 on the evaluation set). 2.0.0
measures the edge width instead.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Width in pixels (feature height / steepest gradient, after a 1-pixel smoothing) where sharpness is 0.5.
# An in-focus particle a few pixels or more across measures ~3.5-4.5 px; a defocus blur of sigma >= 3 px
# or a thick contamination layer gives >= 5 px. Edges are measured in pixels, so a feature must be
# resolved at the pixel size scored (score_map's level choice).
EDGE_WIDTH_MID_PX = 4.8
EDGE_WIDTH_SCALE_PX = 0.3


class BaselineScorer:
    name = "baseline"
    version = "2.0.0"
    description = "Contrast x edge sharpness (edge width), each mapped to 0-1. Empty, blurred or featureless regions score low. A detection proxy."

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        data = np.asarray(region, dtype=np.float64)
        if data.size < 16 or min(data.shape[:2]) < 4 or not np.isfinite(data).all():
            return 0.0, "region too small or not finite"
        smooth = ndimage.gaussian_filter(data, 1.0)  # suppress shot noise before measuring edges
        background = float(np.median(smooth))
        spread = float(np.percentile(smooth, 99) - background)
        noise = float(np.std(data - smooth)) + 1e-9
        contrast = 1.0 - np.exp(-max(spread, 0.0) / (6.0 * noise))
        steepest = float(np.percentile(np.hypot(*np.gradient(smooth)), 99))
        edge_width = max(spread, 1e-12) / max(steepest, 1e-12)
        sharpness = float(1.0 / (1.0 + np.exp(np.clip((edge_width - EDGE_WIDTH_MID_PX) / EDGE_WIDTH_SCALE_PX, -50, 50))))
        value = float(np.clip(contrast * sharpness, 0.0, 1.0))
        reason = f"contrast {contrast:.2f} (features {spread:.3g} above background, noise {noise:.3g}); edge width {edge_width:.1f} px, sharpness {sharpness:.2f}"
        return value, reason
