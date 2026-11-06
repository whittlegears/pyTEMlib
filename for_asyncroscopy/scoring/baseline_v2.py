"""Baseline scorer 2.0.0, superseded by pyTEMlib's BaselineScorer 3.0.0 and kept only so the evaluation can show why.

2.0.0 measured one global edge width (99th-percentile height / steepest gradient): overlapping particles doubled
the height and read as blur, and a feature covering under ~1 % of the region was not reached by the 99th
percentile, so crowded regions and fields sampled finer than the particles ranked at the bottom.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _sigmoid_low(x: float, mid: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(np.clip((x - mid) / scale, -50, 50))))

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
