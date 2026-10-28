"""
Baseline scorer: contrast and edge content.

A deliberately simple, explainable stand-in until trained or physics-based
scorers exist. It ranks a region higher when it has (a) features that stand
out from the background (relative contrast) and (b) sharp edges rather than
smooth gradients or noise. Empty support film scores near 0; a field of
well-separated particles scores high.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class BaselineScorer:
    name = "baseline"
    version = "1.0.0"
    description = "Contrast x edge sharpness, each mapped to 0-1. Empty or featureless regions score near 0."

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        data = np.asarray(region, dtype=np.float64)
        if data.size < 16 or not np.isfinite(data).all():
            return 0.0, "region too small or not finite"
        smooth = ndimage.gaussian_filter(data, 1.0)  # suppress shot noise before measuring edges
        background = float(np.median(smooth))
        spread = float(np.percentile(smooth, 99) - background)
        noise = float(np.std(data - smooth)) + 1e-9
        contrast = 1.0 - np.exp(-max(spread, 0.0) / (6.0 * noise))
        edges = np.hypot(*np.gradient(smooth))
        edge_fraction = float((edges > 3.0 * noise / 2.0).mean())
        sharpness = 1.0 - np.exp(-edge_fraction / 0.05)
        value = float(np.clip(contrast * sharpness, 0.0, 1.0))
        reason = f"contrast {contrast:.2f} (features {spread:.3g} above background, noise {noise:.3g}); edges cover {edge_fraction:.1%}"
        return value, reason
