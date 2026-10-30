"""
Stitch a grid of microscope tiles into one map, using where the stage said each tile was.

Survey maps are taken at magnifications where there are no atomic columns to
match on, the stage never lands exactly where it was told, and the sample
drifts while a large map is acquired (hours, for a map of a million tiles).
So placement starts from each tile's reported stage position and refines it:

1. Overlapping neighbours are found with a spatial grid (O(n), not every pair).
   Each pair is matched by masked normalised cross-correlation, but the peak is
   searched only inside a bound around the expected offset: the stage's stated
   uncertainty plus the most the sample can drift between the two tiles'
   acquisition times. A peak outside the bound, or a weak one, is no match; the
   tile is never "appended" at a wrong place by a distant correlation peak.
2. One sparse least-squares solve finds every tile position and a smooth drift
   of the sample over time (piecewise linear, one knot per ``drift_knot_s``),
   from three kinds of equations: each tile sits at its stage position plus the
   drift at its acquisition time (weak prior), confident matches (strong), and
   a prior on the drift rate. Drift is estimated from pairs acquired far apart
   in time (row to row in a serpentine map), so error does not accumulate over
   thousands of tiles. Matches that disagree with the solution are dropped and
   the solve repeated. The pairs are then matched again with each window
   centred on the stage position plus the fitted drift, and solved again, so
   pairs whose drift moved them out of the first window still count. A tile
   with no usable match keeps its drift-corrected stage position, is flagged
   in ``Placement.fallback`` and is drawn only where no placed tile is
   (vacuum, holes and blank film rarely match); the map's metadata lists it.
   Once placed, the matched tiles also give two things the stage cannot: the
   scan's shared stretch, turn and shear against the stage
   (``estimate_scan_linear``, applied only when measured clearly) and each
   tile's gain and offset (``solve_gain_offset``), so tiles taken far apart in
   time, while the detector drifted, meet without a brightness step.
3. The canvas is written block by block (``block`` x ``block`` pixels) in
   quadtree order. Each tile is resampled onto the canvas at its sub-pixel
   position (cubic spline), with the scan's shared distortion undone, its
   gain and offset applied and a weight falling linearly to its edges, so
   overlaps cross-fade. Each pyramid level's block is made from the four blocks
   below it as soon as they are done. Peak memory is a few blocks per level,
   whatever the map size; tiles are read through the sequence passed in, so a
   lazy sequence (see pyTEMlib.mcp_server.map_store) keeps them out of memory
   too.

Coordinates: images are [row, column] with rows along +y and columns along
+x; stage positions are (x, y) in metres and mark the tile centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr, spsolve
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

# No stage reports positions better than about a pixel at survey magnification (readout, scan
# distortion); a smaller sigma would make the prior rigid and leave no room to measure drift.
MIN_STAGE_SIGMA_PX = 1.0
HUBER_PX = 0.5  # matches agree with the solution to ~0.1 px; beyond this their weight falls off
ROBUST_ROUNDS = 10
SCALE_ROUNDS = 2  # solves with the stage prior in the tiles' own scale (see register_tiles)
READMIT_ROUNDS = 3  # times dropped matches that agree with the solution are taken back and the solve repeated
LINK_CONFIDENCE = 0.7  # one link alone may join two groups of tiles only this confident
DIRECT_SOLVE_MAX = 400_000  # unknowns solved directly; more use lsqr
LSQR_ITERATIONS = 50_000
LINEAR_SIGNIFICANCE = 3.0  # the scan's shared distortion is removed only when measured this many standard errors from none
LINEAR_MIN_TILES = 9
GAIN_MIN_OVERLAP_PX = 16  # a shared area narrower than this says too little about two tiles' brightness
GAIN_PRIOR_SIGMA = 0.1  # a detector's gain is expected within about 10 % of nominal (offsets: 10 % of the signal level)
GAIN_RATE_SHARE = 0.2  # ...and to change by at most a fifth of that between two knots
GAIN_KNOT_S = 300.0  # gain and offset are modelled at one knot per this many seconds of acquisition
GAIN_LIMITS = (0.8, 1.25)  # never correct a tile's gain beyond these, whatever the overlaps say
OFFSET_LIMIT_SHARE = 0.3  # nor its offset beyond this share of the signal level
FALLBACK_WEIGHT = 1e-3  # an unplaced tile is drawn only where no placed tile is (vacuum, holes, blank film)
LINEAR_MIN_EFFECT_PX = 0.1  # the scan's shared distortion is removed only when it moves a tile corner at least this much
WHOLE_PIXEL_PX = 0.01  # a tile this close to a whole-pixel position is copied rather than resampled
SPLINE_MARGIN_PX = 12  # tile pixels kept around the part a block needs; the cubic spline's reach decays well within it
STRIP_PX = 8192  # pairs are matched in vertical strips this wide, so the tiles read stay few (see pairwise_offsets)


@dataclass
class Tile:
    image: np.ndarray
    stage_xy_m: tuple[float, float]
    pixel_size_m: float
    index: int = 0
    attrs: dict = field(default_factory=dict)
    acquired_s: float | None = None  # when the tile was taken; enables the drift model

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.image.shape[:2])


@dataclass
class PairMatch:
    i: int
    j: int
    offset_px: np.ndarray  # measured correction (row, col) of j relative to i, beyond the stage prediction
    confidence: float
    used: bool = False  # kept by the global solve (not rejected as an outlier)


@dataclass
class Placement:
    positions_px: np.ndarray  # (n, 2) top-left (row, col) of each tile on the canvas
    stage_positions_px: np.ndarray
    matches: list[PairMatch]
    canvas_shape: tuple[int, int]
    origin_xy_m: tuple[float, float]  # stage (x, y) of canvas pixel (0, 0) centre
    pixel_size_m: float
    fallback: np.ndarray | None = None  # (n,) True where a tile had no usable match and kept its drift-corrected stage position
    drift_times_s: np.ndarray | None = None  # knot times, seconds after the first tile
    drift_px: np.ndarray | None = None  # (knots, 2) sample drift (row, col) at each knot
    solver: dict | None = None  # how the last solve went: method, converged, iterations
    # Canvas pixels -> tile pixels about a tile's centre: the scan's shared stretch, turn and shear, removed when
    # drawing (identity when it was not measured to be different from none; see estimate_scan_linear).
    linear: np.ndarray = field(default_factory=lambda: np.eye(2))
    linear_estimate: np.ndarray | None = None  # what was measured, applied or not
    linear_stderr: np.ndarray | None = None
    gains: np.ndarray | None = None  # (n,) drawn value = gain * counts + offset (see solve_gain_offset)
    offsets: np.ndarray | None = None

    def centres_px(self, shapes: np.ndarray) -> np.ndarray:
        """(n, 2) canvas (row, col) of each tile's centre."""
        return self.positions_px + (np.asarray(shapes, dtype=np.float64) - 1) / 2.0

    def pixel_to_stage(self, row: float, col: float) -> tuple[float, float]:
        return (self.origin_xy_m[0] + col * self.pixel_size_m, self.origin_xy_m[1] + row * self.pixel_size_m)

    def stage_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return ((y - self.origin_xy_m[1]) / self.pixel_size_m, (x - self.origin_xy_m[0]) / self.pixel_size_m)

    def drift_summary(self) -> dict | None:
        """Fitted drift in metres, for the map's metadata (None without acquisition times)."""
        if self.drift_px is None or self.drift_times_s is None or len(self.drift_times_s) < 2:
            return None
        xy = self.drift_px[:, ::-1] * self.pixel_size_m  # (row, col) -> (x, y)
        span = float(self.drift_times_s[-1] - self.drift_times_s[0])
        total = xy[-1] - xy[0]
        return {
            "times_s": [round(float(t), 3) for t in self.drift_times_s],
            "xy_m": [[float(v) for v in p] for p in xy],
            "total_nm": float(np.hypot(*total) * 1e9),
            "mean_rate_nm_per_s": float(np.hypot(*total) * 1e9 / span) if span > 0 else 0.0,
        }


def stage_layout(centres_xy_m, shapes, pixel_sizes_m) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Top-left canvas positions (row, col) from tile centres and image shapes alone (no pixels needed).

    Returns (positions, pixel size, origin); see ``place_by_stage``.
    """
    if len(centres_xy_m) == 0:
        raise ValueError("no tiles to place")
    pixel_sizes = np.asarray(pixel_sizes_m, dtype=np.float64)
    pixel = float(pixel_sizes[0])
    if np.any(np.abs(pixel_sizes - pixel) > 1e-6 * pixel):
        raise ValueError("all tiles must share one pixel size; resample first")
    centres = np.asarray(centres_xy_m, dtype=np.float64)
    shapes = np.asarray(shapes, dtype=np.float64)[:, :2]
    top_left_xy = centres - shapes[:, ::-1] * pixel / 2.0  # (x, y) of each tile's corner
    origin = top_left_xy.min(axis=0)
    rows = (top_left_xy[:, 1] - origin[1]) / pixel
    cols = (top_left_xy[:, 0] - origin[0]) / pixel
    # Origin is the centre of canvas pixel (0, 0).
    return np.stack([rows, cols], axis=1), pixel, (float(origin[0] + pixel / 2), float(origin[1] + pixel / 2))


def stage_canvas_shape(centres_xy_m, shapes, pixel_sizes_m) -> tuple[int, int]:
    """Canvas (rows, cols) of the stage-only placement, from metadata alone.

    Refinement moves tiles by at most the search bound, so the stitched canvas
    is within a few pixels of this.
    """
    positions, _pixel, _origin = stage_layout(centres_xy_m, shapes, pixel_sizes_m)
    extent = (positions + np.asarray(shapes, dtype=np.float64)[:, :2]).max(axis=0)
    return int(np.ceil(extent[0])), int(np.ceil(extent[1]))


def tile_arrays(tiles: Sequence[Tile]) -> dict:
    """Stage centres (n, 2), shapes (n, 2), pixel sizes (n,) and acquisition times (n,) or None of a tile sequence.

    A sequence that has these as arrays already (``arrays()``, e.g. a run read from a
    store) gives them without making an object per tile.
    """
    if hasattr(tiles, "arrays"):
        return tiles.arrays()
    times = [t.acquired_s for t in tiles]
    return {
        "stage_xy_m": np.array([t.stage_xy_m for t in tiles], dtype=np.float64).reshape(-1, 2),
        "shapes": np.array([t.shape for t in tiles], dtype=np.int64).reshape(-1, 2),
        "pixel_sizes_m": np.array([t.pixel_size_m for t in tiles], dtype=np.float64),
        "acquired_s": None if any(v is None for v in times) else np.asarray(times, dtype=np.float64),
    }


def place_by_stage(tiles: Sequence[Tile]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Top-left canvas positions (row, col) from stage positions; returns (positions, pixel size, origin)."""
    if not len(tiles):
        raise ValueError("no tiles to place")
    arrays = tile_arrays(tiles)
    return stage_layout(arrays["stage_xy_m"], arrays["shapes"], arrays["pixel_sizes_m"])


def overlapping_pairs(positions: np.ndarray, shapes: np.ndarray, min_overlap_px: int = 16) -> list[tuple[int, int, int, int]]:
    """(i, j, overlap rows, overlap cols) for every pair of tiles that overlap by at least ``min_overlap_px``.

    Tiles are bucketed in a grid of cells as large as the largest tile, so only
    tiles in neighbouring cells are compared: O(n) for a map, not O(n^2).
    """
    n = len(positions)
    if n < 2:
        return []
    cell = shapes.max(axis=0).astype(np.float64)
    keys = np.floor(positions / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for k, (a, b) in enumerate(keys):
        buckets.setdefault((int(a), int(b)), []).append(k)
    pairs = []
    for i in range(n):
        a, b = int(keys[i, 0]), int(keys[i, 1])
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for j in buckets.get((a + da, b + db), ()):
                    if j <= i:
                        continue
                    top = max(positions[i, 0], positions[j, 0])
                    left = max(positions[i, 1], positions[j, 1])
                    bottom = min(positions[i, 0] + shapes[i, 0], positions[j, 0] + shapes[j, 0])
                    right = min(positions[i, 1] + shapes[i, 1], positions[j, 1] + shapes[j, 1])
                    height, width = int(bottom - top), int(right - left)
                    if height >= min_overlap_px and width >= min_overlap_px:
                        pairs.append((i, j, height, width))
    pairs.sort()
    return pairs


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _overlap_ncc(a: np.ndarray, b: np.ndarray, rel: np.ndarray, min_overlap_px: int) -> float | None:
    """NCC of the region a and b share when b sits at ``rel`` (row, col) relative to a."""
    dr, dc = int(round(rel[0])), int(round(rel[1]))
    r0, r1 = max(0, dr), min(a.shape[0], dr + b.shape[0])
    c0, c1 = max(0, dc), min(a.shape[1], dc + b.shape[1])
    if r1 - r0 < min_overlap_px or c1 - c0 < min_overlap_px:
        return None
    return _ncc(a[r0:r1, c0:c1], b[r0 - dr : r1 - dr, c0 - dc : c1 - dc])


def bounded_shift(a: np.ndarray, b: np.ndarray, predicted: np.ndarray, bound_px: float, overlap_ratio: float) -> np.ndarray | None:
    """Position of b relative to a (row, col), searched only within ``bound_px`` of ``predicted``.

    Masked normalised cross-correlation over the whole (partial) overlap, with
    the peak taken inside the bound and refined to subpixel by a parabola. A
    peak on the edge of the bound means the true one may lie outside it: None.
    """
    xcorr = cross_correlate_masked(b, a, np.ones(b.shape, bool), np.ones(a.shape, bool), axes=(0, 1), mode="full", overlap_ratio=overlap_ratio)
    # Index k of xcorr is the shift (shape - 1 - k) of b relative to a.
    centre = np.array(a.shape, dtype=np.float64) - 1.0 - np.asarray(predicted, dtype=np.float64)
    lo = np.maximum(np.floor(centre - bound_px).astype(int), 0)
    hi = np.minimum(np.ceil(centre + bound_px).astype(int) + 1, np.array(xcorr.shape))
    if np.any(hi - lo < 3):
        return None
    window = xcorr[lo[0] : hi[0], lo[1] : hi[1]]
    if not np.isfinite(window).any():
        return None
    peak = np.unravel_index(int(np.nanargmax(window)), window.shape)
    if peak[0] in (0, window.shape[0] - 1) or peak[1] in (0, window.shape[1] - 1):
        return None
    sub = []
    for axis in range(2):
        step = np.zeros(2, int)
        step[axis] = 1
        y0, y1, y2 = (float(window[tuple(np.array(peak) + d * step)]) for d in (-1, 0, 1))
        denom = y0 - 2 * y1 + y2
        sub.append(0.5 * (y0 - y2) / denom if denom < 0 and np.isfinite(denom) else 0.0)
    k = lo + np.array(peak) + np.clip(sub, -0.5, 0.5)
    return np.array(a.shape, dtype=np.float64) - 1.0 - k


def _overlap_crops(a_shape, b_shape, predicted: np.ndarray, bound: float) -> tuple[tuple[tuple[slice, slice], np.ndarray], tuple[tuple[slice, slice], np.ndarray]]:
    """The parts of tiles a and b that can overlap when b sits within ``bound`` of ``predicted`` relative to a:
    ((slices of a, their origin in a), (slices of b, their origin in b))."""
    margin = int(np.ceil(bound)) + 2
    a_slices, b_slices, a_origin, b_origin = [], [], [], []
    for axis in range(2):
        shift = int(np.round(predicted[axis]))
        lo, hi = max(0, shift) - margin, min(a_shape[axis], shift + b_shape[axis]) + margin  # in a's pixels
        a_lo, a_hi = max(0, lo), min(a_shape[axis], hi)
        b_lo, b_hi = max(0, lo - shift), min(b_shape[axis], hi - shift)
        a_slices.append(slice(a_lo, a_hi))
        b_slices.append(slice(b_lo, b_hi))
        a_origin.append(a_lo)
        b_origin.append(b_lo)
    return (tuple(a_slices), np.array(a_origin, dtype=np.float64)), (tuple(b_slices), np.array(b_origin, dtype=np.float64))


def pairwise_offsets(
    tiles: Sequence[Tile],
    positions: np.ndarray,
    min_overlap_px: int = 16,
    max_correction_px: float | None = None,
    on_pair: Callable[[int, int], None] | None = None,
    stage_sigma_px: float | None = None,
    max_drift_px_per_s: float = 0.0,
    min_confidence: float = 0.3,
) -> list[PairMatch]:
    """Measure how far each overlapping pair is from where the stage put it, within a bound.

    The search bound for a pair is ``max_correction_px`` if given, else four
    stage sigmas (for the two tiles together) plus ``max_drift_px_per_s`` times
    the time between them, and never more than 0.6 of the overlap. Matches
    below ``min_confidence`` (NCC of the overlap) are not returned.

    ``on_pair(done, total)`` is called before each overlapping pair is matched
    and once at the end; it may raise (e.g. a job's abort) to stop early.
    """
    shapes = tile_arrays(tiles)["shapes"]
    pairs = overlapping_pairs(positions, shapes, min_overlap_px)
    # Match strip by strip (vertical strips STRIP_PX wide), top to bottom within each: the tiles
    # a pair needs are then the current and the previous row of one strip, so a bounded tile
    # cache reads each block of tiles about once however wide the map is.
    strip = (positions[:, 1] // STRIP_PX).astype(np.int64)
    pairs.sort(key=lambda p: (strip[p[0]], positions[p[0], 0], positions[p[0], 1], p[1]))
    times = _times(tiles)
    sigma = max(stage_sigma_px, MIN_STAGE_SIGMA_PX) if stage_sigma_px is not None else 0.1 * float(np.median(shapes))
    matches: list[PairMatch] = []
    for done, (i, j, height, width) in enumerate(pairs):
        if on_pair is not None:
            on_pair(done, len(pairs))  # may raise to stop registration
        a = np.asarray(tiles[i].image, dtype=np.float64)
        b = np.asarray(tiles[j].image, dtype=np.float64)
        if a.std() == 0 or b.std() == 0:
            continue
        predicted = positions[j] - positions[i]
        if max_correction_px is not None:
            bound = float(max_correction_px)
        else:
            dt = abs(times[j] - times[i]) if times is not None else 0.0
            bound = 4.0 * np.sqrt(2.0) * sigma + max_drift_px_per_s * dt + 2.0
        bound = min(bound, 0.6 * min(height, width))
        # Correlate only the predicted overlap plus the search bound on each side: the rest of each tile
        # cannot meet the other within the bound, and leaving it out makes each correlation several times smaller.
        (ca, oa), (cb, ob) = _overlap_crops(a.shape, b.shape, predicted, bound)
        a_part, b_part = a[ca], b[cb]
        overlap_ratio = 0.5 * min(height * width / a_part.size, height * width / b_part.size)
        shift = bounded_shift(a_part, b_part, predicted + ob - oa, bound, min(overlap_ratio, 1.0))
        if shift is None:
            continue
        shift = shift - ob + oa
        confidence = _overlap_ncc(a, b, shift, min_overlap_px)
        if confidence is None or confidence < min_confidence:
            continue
        matches.append(PairMatch(i, j, shift - predicted, confidence))
    if on_pair is not None:
        on_pair(len(pairs), len(pairs))
    return matches


def _times(tiles: Sequence[Tile]) -> np.ndarray | None:
    """Acquisition times relative to the first tile, or None if any tile lacks one."""
    times = tile_arrays(tiles)["acquired_s"]
    if times is None or not np.isfinite(times).all():
        return None
    return times - times.min()


def _knots(times: np.ndarray | None, knot_s: float, n: int) -> np.ndarray | None:
    if times is None or times.max() <= 0:
        return None
    count = int(np.clip(np.ceil(times.max() / knot_s), 1, max(1, n // 10))) + 1
    return np.linspace(0.0, times.max(), count)


def _drift_at(times: np.ndarray, knots: np.ndarray, drift: np.ndarray) -> np.ndarray:
    left, frac = _interp_weights(times, knots)
    return drift[left] * (1.0 - frac)[:, None] + drift[left + 1] * frac[:, None]


def _interp_weights(times: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each time, the knot index on its left and the weight of the knot on its right."""
    k = np.clip(np.searchsorted(knots, times, side="right") - 1, 0, len(knots) - 2)
    span = knots[k + 1] - knots[k]
    frac = np.where(span > 0, (times - knots[k]) / np.where(span > 0, span, 1.0), 0.0)
    return k, frac


def solve_positions(
    stage_positions: np.ndarray,
    matches: Sequence[PairMatch],
    stage_sigma_px: float,
    match_sigma_px: float = 0.5,
    min_confidence: float = 0.3,
    outlier_px: float = 3.0,
    times_s: np.ndarray | None = None,
    drift_knot_s: float = 120.0,
    max_drift_px_per_s: float = 0.0,
    bound_sigmas: float = 3.5,
    info: dict | None = None,
) -> tuple[np.ndarray, list[PairMatch], np.ndarray | None, np.ndarray | None]:
    """Weighted least squares over tile positions and (with times) a piecewise-linear drift.

    Stage positions plus the drift at each tile's time are a weak prior,
    confident matches strong constraints, and the drift rate has a prior of
    about ``max_drift_px_per_s`` (drift starts at 0 at the first tile).
    The solve is repeated with Huber weights (``HUBER_PX``); matches whose
    residual exceeds ``outlier_px`` and five robust sigmas, or that disagree
    with the drift-corrected stage by more than ``bound_sigmas`` combined
    stage sigmas, are dropped. Returns
    (positions, matches used, knot times, drift at knots).
    """
    candidates = [m for m in matches if m.confidence >= min_confidence]
    knots = _knots(times_s, drift_knot_s, len(stage_positions)) if max_drift_px_per_s > 0 else None
    # A match must also agree with the stage once the fitted drift is removed: the two stage
    # readings differ by at most bound_sigmas combined stage sigmas. This catches a wrong peak in a
    # low-texture overlap that nothing else contradicts (its only other constraint is the weak prior).
    bound = bound_sigmas * np.sqrt(2.0) * max(stage_sigma_px, MIN_STAGE_SIGMA_PX) + 1.0

    def check(group, positions, drift):
        """Each match's residual against the solve and its disagreement with the drift-corrected stage."""
        i = np.array([m.i for m in group])
        j = np.array([m.j for m in group])
        offsets = np.array([m.offset_px for m in group])
        residuals = np.hypot(*((positions[j] - positions[i]) - (stage_positions[j] - stage_positions[i] + offsets)).T)
        drift_at = _drift_at(times_s, knots, drift) if drift is not None else np.zeros_like(stage_positions)
        return residuals, np.hypot(*(offsets - (drift_at[j] - drift_at[i])).T)

    def cutoff_of(residuals):
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        return max(outlier_px, float(np.median(residuals)) + 5.0 * 1.4826 * mad)

    def robust_solve(used, trust_drift):
        robust = np.ones(len(used))
        positions, drift = _solve_once(stage_positions, used, stage_sigma_px, match_sigma_px, times_s, knots, max_drift_px_per_s, robust, info)
        for round_ in range(ROBUST_ROUNDS):
            if not used:
                break
            residuals, disagreement = check(used, positions, drift)
            # Drop the worst first: a wrong match bends its neighbours in the solve, so correct matches around it
            # look bad too until it is gone. Only matches within a factor two of the largest residual go this
            # round; the rest are judged again once the solve no longer carries the worst.
            keep = residuals <= max(cutoff_of(residuals), 0.5 * float(residuals.max()))
            if trust_drift or round_ > 0:  # the drift is only trusted once a first robust solve has fitted it
                keep &= disagreement <= bound
            # Huber weights: a match a little off (a weak, slightly wrong peak) counts less instead of
            # bending its neighbours and the drift fit; one far off is dropped outright.
            new_robust = np.minimum(1.0, HUBER_PX / np.maximum(residuals, 1e-9))[keep]
            if keep.all() and np.allclose(new_robust, robust, atol=0.02):
                break
            used = [m for m, k in zip(used, keep) if k]
            robust = new_robust
            positions, drift = _solve_once(stage_positions, used, stage_sigma_px, match_sigma_px, times_s, knots, max_drift_px_per_s, robust, info)
        return positions, used, drift

    positions, used, drift = robust_solve(candidates, trust_drift=False)
    # A correct match is not wrong for good because it was dropped while wrong ones still bent the solve. Two ways
    # back, until nothing changes: a match that agrees with the solution found without the wrong ones; and, where the
    # used matches split the tiles into separate groups (a few tiles whose wrong links outnumbered their right ones),
    # the links between two groups that agree with each other on how far one group must move.
    for _ in range(READMIT_ROUNDS):
        if not used or not candidates:
            break
        residuals, disagreement = check(candidates, positions, drift)
        cutoff = cutoff_of(check(used, positions, drift)[0])
        back = [m for m, r, d in zip(candidates, residuals, disagreement) if r <= cutoff and d <= bound]
        back += _consensus_links(candidates, back, positions, stage_positions, len(stage_positions), outlier_px)
        if {id(m) for m in back} == {id(m) for m in used}:
            break
        positions, used, drift = robust_solve(back, trust_drift=True)
    return positions, used, knots, drift


def _consensus_links(candidates, used, positions, stage_positions, n: int, radius_px: float) -> list[PairMatch]:
    """Links between groups of tiles the used matches do not join, kept when they agree on the groups' offset.

    Each unused link between two groups says how far one group must move against the other. The largest set
    of links between the same two groups that agree within ``radius_px`` is kept when it has at least two links,
    or one of confidence LINK_CONFIDENCE or more; links that contradict each other join nothing.
    """
    if not used:
        return []
    i = np.array([m.i for m in used])
    j = np.array([m.j for m in used])
    graph = sparse.coo_matrix((np.ones(len(used)), (i, j)), shape=(n, n))
    _count, group = connected_components(graph, directed=False)
    taken = {id(m) for m in used}
    between: dict[tuple[int, int], list[tuple[PairMatch, np.ndarray]]] = {}
    for m in candidates:
        if id(m) in taken or group[m.i] == group[m.j]:
            continue
        correction = (positions[m.j] - positions[m.i]) - (stage_positions[m.j] - stage_positions[m.i] + m.offset_px)
        key, sign = ((group[m.i], group[m.j]), 1.0) if group[m.i] < group[m.j] else ((group[m.j], group[m.i]), -1.0)
        between.setdefault(key, []).append((m, sign * correction))
    joined: list[PairMatch] = []
    for links in between.values():
        vectors = np.array([v for _m, v in links])
        agree = [np.nonzero(np.hypot(*(vectors - v).T) <= radius_px)[0] for v in vectors]
        best = max(agree, key=len)
        if len(best) >= 2 or (len(best) == 1 and links[best[0]][0].confidence >= LINK_CONFIDENCE):
            joined += [links[k][0] for k in best]
    return joined


def _solve_once(stage_positions, matches, stage_sigma_px, match_sigma_px, times_s, knots, max_drift_px_per_s, robust, info=None):
    """One weighted least-squares solve; returns (positions, drift at knots or None).

    The drift at the first knot is fixed at zero by leaving it out of the
    unknowns (a heavily weighted anchor equation would ill-condition the
    system). Columns are scaled to unit norm before an iterative solve.
    """
    n = len(stage_positions)
    k_count = 0 if knots is None else len(knots)
    free = max(k_count - 1, 0)  # drift unknowns: knots 1 .. K-1
    rows, cols, vals = [], [], []
    rhs: list[np.ndarray] = []
    w_stage = 1.0 / max(stage_sigma_px, MIN_STAGE_SIGMA_PX)
    idx = np.arange(n)
    # Stage prior: p_i - drift(t_i) = s_i.
    rows.append(idx), cols.append(idx), vals.append(np.full(n, w_stage))
    if k_count:
        left, frac = _interp_weights(times_s, knots)
        for knot, weight in ((left, 1.0 - frac), (left + 1, frac)):
            keep = knot > 0
            rows.append(idx[keep]), cols.append(n + knot[keep] - 1), vals.append(-w_stage * weight[keep])
    rhs.append(w_stage * stage_positions)
    eq = n
    if matches:
        mi = np.array([m.i for m in matches])
        mj = np.array([m.j for m in matches])
        w = np.array([m.confidence for m in matches]) * robust / match_sigma_px
        target = stage_positions[mj] - stage_positions[mi] + np.array([m.offset_px for m in matches])
        e = eq + np.arange(len(matches))
        rows += [e, e]
        cols += [mj, mi]
        vals += [w, -w]
        rhs.append(w[:, None] * target)
        eq += len(matches)
    if free:
        # The drift rate between knots has a prior of about max_drift_px_per_s.
        dt = np.diff(knots)
        w_rate = 1.0 / np.maximum(max_drift_px_per_s * dt, 1e-6)
        e = eq + np.arange(free)
        rows.append(e), cols.append(n + np.arange(free)), vals.append(w_rate)
        later = np.arange(1, free)
        rows.append(eq + later), cols.append(n + later - 1), vals.append(-w_rate[1:])
        rhs.append(np.zeros((free, 2)))
        eq += free
    a = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(eq, n + free))
    b = np.vstack(rhs)
    x0 = np.concatenate([stage_positions, np.zeros((free, 2))])
    # Solve for the correction to the stage placement. The normal equations are sparse (a grid
    # graph plus the drift knots): a direct sparse solve is about 10x faster than lsqr here.
    # Beyond DIRECT_SOLVE_MAX unknowns lsqr keeps memory down; its convergence is checked.
    if a.shape[1] <= DIRECT_SOLVE_MAX:
        ata = (a.T @ a).tocsc()
        solved = np.stack([x0[:, axis] + spsolve(ata, a.T @ (b[:, axis] - a @ x0[:, axis])) for axis in range(2)], axis=1)
        if info is not None:
            info.update({"method": "direct", "converged": bool(np.isfinite(solved).all())})
    else:
        scale = 1.0 / np.maximum(np.sqrt(np.asarray(a.multiply(a).sum(axis=0))).ravel(), 1e-12)
        scaled = a @ sparse.diags(scale)
        results = [lsqr(scaled, b[:, axis] - a @ x0[:, axis], atol=1e-10, btol=1e-10, iter_lim=LSQR_ITERATIONS) for axis in range(2)]
        solved = np.stack([x0[:, axis] + scale * r[0] for axis, r in enumerate(results)], axis=1)
        if info is not None:
            # istop 1 and 2: solved to the tolerances; 7: the iteration limit was reached first.
            info.update({"method": "lsqr", "converged": all(r[1] in (1, 2) for r in results), "iterations": int(max(r[2] for r in results)), "istop": [int(r[1]) for r in results]})
    drift = None
    if k_count:
        drift = np.vstack([np.zeros((1, 2)), solved[n:]])
    return solved[:n], drift


def register_tiles(
    tiles: Sequence[Tile],
    stage_sigma_m: float = 200e-9,
    min_confidence: float = 0.3,
    refine: bool = True,
    on_pair: Callable[[int, int], None] | None = None,
    max_drift_m_per_s: float = 0.0,
    drift_knot_s: float = 120.0,
    scan_linear: np.ndarray | None = None,
) -> Placement:
    """Place tiles by stage position, then refine with bounded image matching and (with times) a drift model.

    ``stage_sigma_m``: 1-sigma error of the reported stage positions (sets the
    search bound and the prior). ``max_drift_m_per_s``: the most the sample
    drifts relative to the stage; 0 turns the drift model off. Tiles need
    ``acquired_s`` for the drift model. ``scan_linear``: a calibration of the
    scan's shared stretch, turn and shear (the 2 x 2 map from stage pixels to
    tile pixels, as ``Placement.linear_estimate`` reports it), measured once
    for this scan setting, for example on a large map; it is applied as given.
    Without it the map measures its own, which only a large map can do: the
    stage's error hides a 1 % scale error on a few dozen tiles.
    """
    stage_px, pixel, origin = place_by_stage(tiles)
    sigma_px = stage_sigma_m / pixel
    drift_px_s = max_drift_m_per_s / pixel
    times = _times(tiles)
    matches = pairwise_offsets(tiles, stage_px, on_pair=on_pair, stage_sigma_px=sigma_px, max_drift_px_per_s=drift_px_s, min_confidence=min_confidence) if refine else []
    knots, drift = None, None
    info: dict = {}
    if matches:
        positions, used, knots, drift = solve_positions(stage_px, matches, sigma_px, min_confidence=min_confidence, times_s=times, drift_knot_s=drift_knot_s, max_drift_px_per_s=drift_px_s, info=info)
        if drift is not None:
            # Match again with each search window centred on the stage position plus the drift found
            # so far. The first pass, centred on the stage alone, misses pairs whose drift difference
            # pushed the true offset past the window, and those are the pairs that say most about
            # the drift: without this pass the fitted drift is biased low on long maps.
            expected = stage_px + _drift_at(times, knots, drift)
            matches = pairwise_offsets(tiles, expected, on_pair=on_pair, stage_sigma_px=sigma_px, max_drift_px_per_s=0.0, min_confidence=min_confidence)
            correction = expected - stage_px
            for m in matches:
                m.offset_px = m.offset_px + (correction[m.j] - correction[m.i])  # relative to the stage again, as the solve expects
            if matches:
                positions, used, knots, drift = solve_positions(stage_px, matches, sigma_px, min_confidence=min_confidence, times_s=times, drift_knot_s=drift_knot_s, max_drift_px_per_s=drift_px_s, info=info)
        # The stage prior is in the stage's scale and the matches in the tiles' own. When the scan is stretched or
        # turned against the stage (1 % is common), the prior pulls a large map toward the stage's scale and bends it
        # by about a pixel, and the bend makes the stretch look smaller than it is. So put the prior in the tiles'
        # scale and solve again, until the stretch measured from the solve stops changing.
        half = (tile_arrays(tiles)["shapes"].astype(np.float64) - 1) / 2.0
        base = {id(m): m.offset_px.copy() for m in matches}
        for _ in range(SCALE_ROUNDS):
            matched_now = np.zeros(len(tiles), bool)
            for m in used:
                matched_now[m.i] = matched_now[m.j] = True
            corrected = stage_px + _drift_at(times, knots, drift) if drift is not None else stage_px
            if scan_linear is not None:
                scale = np.asarray(scan_linear, dtype=np.float64).reshape(2, 2)
            else:
                scale, _stderr, trusted = estimate_scan_linear(corrected + half, positions + half, matched_now)
                if not (trusted and _linear_matters(scale, half * 2 + 1)):
                    break
            centre = (stage_px + half).mean(axis=0)
            prior = (stage_px + half - centre) @ scale.T + centre - half
            for m in matches:
                m.offset_px = base[id(m)] + (stage_px[m.j] - stage_px[m.i]) - (prior[m.j] - prior[m.i])
            positions, used, knots, drift = solve_positions(prior, matches, sigma_px, min_confidence=min_confidence, times_s=times, drift_knot_s=drift_knot_s, max_drift_px_per_s=drift_px_s, info=info)
            if scan_linear is not None:
                break
        for m in matches:  # offsets are reported relative to the stage again
            m.offset_px = base[id(m)]
    else:
        positions, used = stage_px.copy(), []
    used_ids = {id(m) for m in used}
    matched = np.zeros(len(tiles), bool)
    for m in matches:
        m.used = id(m) in used_ids
        if m.used:
            matched[m.i] = matched[m.j] = True
    shapes = tile_arrays(tiles)["shapes"].astype(np.float64)
    gains, offsets = solve_gain_offset(tiles, positions, [m for m in matches if m.used])
    # Against the stage plus the fitted drift: in a raster, drift grows with position and would look like a stretch.
    expected = stage_px + _drift_at(times, knots, drift) if drift is not None else stage_px
    linear, linear_stderr, applied = estimate_scan_linear(expected + (shapes - 1) / 2.0, positions + (shapes - 1) / 2.0, matched)
    estimate = linear
    applied = applied and _linear_matters(linear, shapes)
    if scan_linear is not None:
        linear, applied = np.asarray(scan_linear, dtype=np.float64).reshape(2, 2), True
    if applied:
        # The tiles' own pixels are stretched and turned relative to the stage (field-of-view calibration, scan
        # rotation, shear): put the tile centres back on the stage's scale; assembly resamples each tile to match.
        centres = positions + (shapes - 1) / 2.0
        middle = centres[matched].mean(axis=0)
        positions = (centres - middle) @ np.linalg.inv(linear).T + middle - (shapes - 1) / 2.0
    corners = _footprint_corners(linear if applied else np.eye(2), shapes)
    # Shift so the canvas starts at (0, 0), and move the origin with it.
    shift = (positions + (shapes - 1) / 2.0 + corners[0]).min(axis=0) + 0.5
    positions = positions - shift
    stage_px = stage_px - shift
    origin = (origin[0] + shift[1] * pixel, origin[1] + shift[0] * pixel)
    canvas = tuple(int(np.ceil(v)) for v in (positions + (shapes - 1) / 2.0 + corners[1] + 0.5).max(axis=0))
    return Placement(
        positions, stage_px, matches, canvas, origin, pixel, fallback=~matched if refine else None, drift_times_s=knots, drift_px=drift,
        solver=info or None, linear=linear if applied else np.eye(2), linear_estimate=estimate, linear_stderr=linear_stderr, gains=gains, offsets=offsets,
    )


def _footprint_corners(linear: np.ndarray, shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lowest, highest) (row, col) reach of each tile's pixels from its centre on the canvas, per tile."""
    inverse = np.linalg.inv(linear)
    half = shapes / 2.0
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float64)
    reach = np.stack([(half * s) @ inverse.T for s in signs])  # (4, n, 2)
    return reach.min(axis=0), reach.max(axis=0)


def estimate_scan_linear(stage_centres: np.ndarray, centres: np.ndarray, matched: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """The scan's own geometry against the stage's: the 2 x 2 map A with tile centres = A @ stage centres + b.

    Measured from the matched tiles, whose centres follow their images: a field-of-view calibration error
    stretches A, scan rotation turns it, shear skews it. The stage's random error averages out over many
    tiles, so A is only trusted (``applied``) when it differs from no distortion by more than
    LINEAR_SIGNIFICANCE standard errors; a small map leaves it alone. Returns (A, standard errors, applied).
    """
    identity = np.eye(2)
    n = int(matched.sum())
    if n < LINEAR_MIN_TILES:
        return identity, np.full((2, 2), np.inf), False
    # The stage readings carry the noise and the tile centres hardly any, so the stage is regressed on the centres
    # (noise in the regressor would shrink the fitted slope toward zero) and the result inverted.
    design = np.hstack([centres[matched], np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(design, stage_centres[matched], rcond=None)
    residual = stage_centres[matched] - design @ coef
    variance = (residual**2).sum(axis=0) / max(n - 3, 1)  # per output axis
    unscaled = np.diag(np.linalg.pinv(design.T @ design))[:2]
    stderr = np.sqrt(np.outer(variance, unscaled))  # of the stage-from-tile map; A is near identity, so A's are the same
    linear = np.linalg.inv(coef[:2].T)
    applied = bool(np.any(np.abs(linear - identity) > LINEAR_SIGNIFICANCE * stderr))
    return linear, stderr, applied


def _linear_matters(linear: np.ndarray, shapes: np.ndarray) -> bool:
    """Whether undoing ``linear`` moves a tile corner by at least LINEAR_MIN_EFFECT_PX."""
    return float(np.abs(linear - np.eye(2)).max() * shapes.max()) >= LINEAR_MIN_EFFECT_PX


def solve_gain_offset(tiles: Sequence[Tile], positions_px: np.ndarray, matches: Sequence[PairMatch]) -> tuple[np.ndarray, np.ndarray]:
    """Gain and offset per tile that make overlapping tiles agree in brightness (drawn = gain * counts + offset).

    For each used pair the shared area's spread gives the ratio of the two tiles' gains (the same sample seen
    twice, so its contrast differs only by the gain) and its mean the offsets once the gains are known.

    A detector drifts in time, not in space, so when every tile has an acquisition time the gain and offset are
    smooth functions of time (piecewise linear, one knot per GAIN_KNOT_S) with a prior on how fast they change;
    per-tile values would let each pair's small error add up along chains of tiles across a large map. Without
    times each tile gets its own values. Either way a prior pulls gain to 1 and offset to 0, gains are capped to
    GAIN_LIMITS, and tiles in no used pair keep (1, 0) unless the time model covers them.
    """
    n = len(tiles)
    pairs = []
    for m in matches:
        a, b = np.asarray(tiles[m.i].image), np.asarray(tiles[m.j].image)
        d = np.round(positions_px[m.j] - positions_px[m.i]).astype(int)
        r0, c0 = max(0, d[0]), max(0, d[1])
        r1, c1 = min(a.shape[0], d[0] + b.shape[0]), min(a.shape[1], d[1] + b.shape[1])
        if r1 - r0 < GAIN_MIN_OVERLAP_PX or c1 - c0 < GAIN_MIN_OVERLAP_PX:
            continue
        pa = a[r0:r1, c0:c1].astype(np.float64)  # slice, then convert: never a whole tile in float64
        pb = b[r0 - d[0] : r1 - d[0], c0 - d[1] : c1 - d[1]].astype(np.float64)
        sa, sb = float(pa.std()), float(pb.std())
        if sa > 0 and sb > 0:
            pairs.append((m.i, m.j, float(pa.mean()), float(pb.mean()), np.log(sb / sa), pa.size, (sa + sb) / 2.0))
    if not pairs:
        return np.ones(n), np.zeros(n)
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    size = np.array([p[5] for p in pairs], dtype=np.float64)
    spread = np.array([p[6] for p in pairs])
    times = _times(tiles)
    knots = _knots(times, GAIN_KNOT_S, max(n, 20)) if times is not None else None
    if knots is not None:  # unknowns: the values at the knots; tile k is a mix of the two knots around its time
        left, frac = _interp_weights(times, knots)
        count = len(knots)
        mix = sparse.csr_matrix(
            (np.concatenate([1.0 - frac, frac]), (np.concatenate([np.arange(n)] * 2), np.concatenate([left, left + 1]))), shape=(n, count)
        )
    else:
        count = n
        mix = sparse.identity(n, format="csr")
    difference = mix[i] - mix[j]  # one row per pair: unknowns of i minus unknowns of j

    def solve(target: np.ndarray, sigma: np.ndarray, prior_sigma: float) -> np.ndarray:
        rows = [sparse.diags(1.0 / sigma) @ difference, sparse.identity(count, format="csr") / prior_sigma]
        rhs = [target / sigma, np.zeros(count)]
        if knots is not None and count > 1:  # how fast the detector may change between knots
            step = sparse.diags([-np.ones(count - 1), np.ones(count - 1)], [0, 1], shape=(count - 1, count)) / (prior_sigma * GAIN_RATE_SHARE)
            rows.append(step)
            rhs.append(np.zeros(count - 1))
        return lsqr(sparse.vstack(rows).tocsr(), np.concatenate(rhs), atol=1e-12, btol=1e-12)[0]

    # The log of a spread measured over N pixels is uncertain by about sqrt(1 / (2 N)) per tile, so a pair by 1 / sqrt(N).
    log_gain = mix @ solve(np.array([p[4] for p in pairs]), 1.0 / np.sqrt(size), GAIN_PRIOR_SIGMA)
    gains = np.clip(np.exp(log_gain), *GAIN_LIMITS)
    level = float(np.median([abs(p[2]) + abs(p[3]) for p in pairs])) / 2.0 + 1.0
    target = gains[j] * np.array([p[3] for p in pairs]) - gains[i] * np.array([p[2] for p in pairs])
    offsets = mix @ solve(target, spread / np.sqrt(size) + 1e-9, GAIN_PRIOR_SIGMA * level)
    offsets = np.clip(offsets, -OFFSET_LIMIT_SHARE * level, OFFSET_LIMIT_SHARE * level)
    return gains, offsets


def _cast(array: np.ndarray, dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(dtype)
    return array.astype(dtype)


def pyramid_levels(canvas_shape: tuple[int, int], min_size: int = 256) -> int:
    levels = 1
    h, w = canvas_shape
    while max(h, w) > min_size:
        h, w = max(1, h // 2), max(1, w // 2)
        levels += 1
    return levels


def level_shapes(canvas_shape: tuple[int, int], levels: int) -> list[tuple[int, int]]:
    """Shape of each pyramid level: each is the one before halved (rounded down), at least 1 x 1."""
    shapes = [tuple(int(v) for v in canvas_shape)]
    for _ in range(1, levels):
        h, w = shapes[-1]
        shapes.append((max(1, h // 2), max(1, w // 2)))
    return shapes


class LevelWriter(Protocol):
    """Where ``assemble`` puts the pyramid: one array per level, written block by block."""

    def create(self, level: int, shape: tuple[int, int], dtype: np.dtype, block: int) -> None: ...

    def write(self, level: int, row0: int, col0: int, data: np.ndarray) -> None: ...


class ArrayWriter:
    """A LevelWriter that keeps every level in memory (for small maps and tests)."""

    def __init__(self):
        self.levels: list[np.ndarray] = []

    def create(self, level, shape, dtype, block):
        self.levels.append(np.zeros(shape, dtype=dtype))

    def write(self, level, row0, col0, data):
        self.levels[level][row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]] = data


def assemble(
    tiles: Sequence[Tile],
    placement: Placement,
    writer: LevelWriter,
    block: int = 512,
    levels: int | None = None,
    dtype=None,
    on_block: Callable[[int, int], None] | None = None,
) -> int:
    """Blend tiles onto the canvas and write every pyramid level to ``writer``, block by block.

    Level-0 blocks blend only the tiles that touch them (found with a grid
    index); a level-k block is the 2 x 2 average of its four level-(k-1)
    blocks, made right after them (quadtree order), so peak memory is a few
    blocks per level. ``on_block(done, total)`` may raise to stop. Returns the
    number of levels written.
    """
    shapes = level_shapes(placement.canvas_shape, levels or pyramid_levels(placement.canvas_shape))
    levels = len(shapes)
    dtype = np.dtype(dtype or np.result_type(*{np.asarray(tiles[k].image).dtype for k in range(min(len(tiles), 4))}))
    for level, shape in enumerate(shapes):
        writer.create(level, shape, dtype, block)
    tile_shapes = tile_arrays(tiles)["shapes"].astype(np.float64)
    centres = placement.centres_px(tile_shapes)
    linear = np.asarray(placement.linear, dtype=np.float64)
    low, high = _footprint_corners(linear, tile_shapes)
    lows = np.floor(centres + low).astype(np.int64)
    highs = np.ceil(centres + high).astype(np.int64)
    # An unplaced tile (no usable match: vacuum, a hole, blank film) is drawn only where no placed tile is.
    tile_weight = np.where(placement.fallback, FALLBACK_WEIGHT, 1.0) if placement.fallback is not None else np.ones(len(tiles))
    drawn = np.ones(len(tiles), bool)
    corner = centres - (tile_shapes - 1) / 2.0
    whole = np.allclose(linear, np.eye(2)) & np.all(np.abs(corner - np.round(corner)) < WHOLE_PIXEL_PX, axis=1)  # copied, not resampled
    gains = placement.gains if placement.gains is not None else np.ones(len(tiles))
    offsets = placement.offsets if placement.offsets is not None else np.zeros(len(tiles))
    index: dict[tuple[int, int], list[int]] = {}
    for k in np.nonzero(drawn)[0]:
        for bi in range(max(0, lows[k][0]) // block, max(0, highs[k][0]) // block + 1):
            for bj in range(max(0, lows[k][1]) // block, max(0, highs[k][1]) // block + 1):
                index.setdefault((bi, bj), []).append(int(k))

    def sample(k: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Tile k's values at tile coordinates (u, v), by cubic spline over just the part of the tile they need."""
        th, tw = (int(x) for x in tile_shapes[k])
        u0, u1 = max(0, int(np.floor(u.min())) - SPLINE_MARGIN_PX), min(th, int(np.ceil(u.max())) + SPLINE_MARGIN_PX + 1)
        v0, v1 = max(0, int(np.floor(v.min())) - SPLINE_MARGIN_PX), min(tw, int(np.ceil(v.max())) + SPLINE_MARGIN_PX + 1)
        part = np.asarray(tiles[k].image)[u0:u1, v0:v1].astype(np.float64)
        coefficients = ndimage.spline_filter(part, order=3, mode="mirror")
        return ndimage.map_coordinates(coefficients, [u - u0, v - v0], order=3, prefilter=False, mode="mirror")

    total = sum(-(-h // block) * -(-w // block) for h, w in shapes)
    done = [0]

    def emit(level: int, row0: int, col0: int, data: np.ndarray) -> None:
        writer.write(level, row0, col0, _cast(data, dtype))
        done[0] += 1
        if on_block is not None:
            on_block(done[0], total)

    def base(bi: int, bj: int) -> np.ndarray:
        h, w = shapes[0]
        r0, c0 = bi * block, bj * block
        r1, c1 = min(r0 + block, h), min(c0 + block, w)
        acc = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
        wsum = np.zeros_like(acc)
        for k in index.get((bi, bj), ()):
            rr0, rr1 = max(r0, lows[k][0]), min(r1, highs[k][0] + 1)
            cc0, cc1 = max(c0, lows[k][1]), min(c1, highs[k][1] + 1)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            # Each canvas pixel looks up where it falls in the tile (sub-pixel, with the scan's shared distortion
            # undone) and takes the tile's value there by cubic-spline interpolation: no rounding to whole pixels.
            th, tw = tile_shapes[k]
            if whole[k]:
                top, left = (int(v) for v in np.round(corner[k]))
                rr0, rr1, cc0, cc1 = max(rr0, top), min(rr1, top + int(th)), max(cc0, left), min(cc1, left + int(tw))
                if rr0 >= rr1 or cc0 >= cc1:
                    continue
                u = np.arange(rr0 - top, rr1 - top, dtype=np.float64)[:, None]
                v = np.arange(cc0 - left, cc1 - left, dtype=np.float64)[None, :]
                weight = np.minimum(u + 1.0, th - u) * np.minimum(v + 1.0, tw - v) * tile_weight[k]
                sub = np.asarray(tiles[k].image)[rr0 - top : rr1 - top, cc0 - left : cc1 - left].astype(np.float64) * gains[k] + offsets[k]
                acc[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] += sub * weight
                wsum[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] += weight
                continue
            dr = (np.arange(rr0, rr1) - centres[k][0]).astype(np.float32)[:, None]  # float32 is 1e-4 px at 512 px
            dc = (np.arange(cc0, cc1) - centres[k][1]).astype(np.float32)[None, :]
            u = np.float32(linear[0, 0]) * dr + np.float32(linear[0, 1]) * dc + np.float32((th - 1) / 2.0)
            v = np.float32(linear[1, 0]) * dr + np.float32(linear[1, 1]) * dc + np.float32((tw - 1) / 2.0)
            # The cross-fade weight: a linear ramp to zero half a pixel outside the tile's edge pixels.
            weight = np.clip(np.minimum(u + 1.0, th - u), 0.0, None) * np.clip(np.minimum(v + 1.0, tw - v), 0.0, None)
            weight[(u < -0.5) | (u > th - 0.5) | (v < -0.5) | (v > tw - 0.5)] = 0.0
            weight *= tile_weight[k]
            if not weight.any():
                continue
            sub = sample(k, u, v) * gains[k] + offsets[k]
            acc[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] += sub * weight
            wsum[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] += weight
        out = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0).astype(np.float32)
        emit(0, r0, c0, out)
        return out

    def node(level: int, bi: int, bj: int) -> np.ndarray:
        if level == 0:
            return base(bi, bj)
        h, w = shapes[level]
        ph, pw = shapes[level - 1]
        r0, c0 = bi * block, bj * block
        r1, c1 = min(r0 + block, h), min(c0 + block, w)
        buf = np.zeros((2 * (r1 - r0), 2 * (c1 - c0)), dtype=np.float32)
        for di in (0, 1):
            for dj in (0, 1):
                ci, cj = 2 * bi + di, 2 * bj + dj
                if ci * block >= ph or cj * block >= pw:
                    continue
                child = node(level - 1, ci, cj)
                rr, cc = di * block, dj * block
                if rr >= buf.shape[0] or cc >= buf.shape[1]:
                    continue  # the odd last row/column of the level below is dropped by halving
                take = child[: buf.shape[0] - rr, : buf.shape[1] - cc]
                buf[rr : rr + take.shape[0], cc : cc + take.shape[1]] = take
        out = buf.reshape(r1 - r0, 2, c1 - c0, 2).mean(axis=(1, 3))
        emit(level, r0, c0, out)
        return out

    top_h, top_w = shapes[-1]
    for bi in range(-(-top_h // block)):
        for bj in range(-(-top_w // block)):
            node(levels - 1, bi, bj)
    _finish_leftovers(shapes, block, node, levels)
    return levels


def _finish_leftovers(shapes, block, node, levels):
    """Blocks of lower levels not under any top-level block (a level whose odd last row or column was dropped)."""
    # Every block of level k lies under block (bi // 2, bj // 2) of level k + 1 unless it is in the odd
    # last row/column that halving drops; those still have to be written at their own level.
    for level in range(levels - 2, -1, -1):
        h, w = shapes[level]
        nh, nw = shapes[level + 1]
        for bi in range(-(-h // block)):
            for bj in range(-(-w // block)):
                if bi * block >= 2 * nh or bj * block >= 2 * nw:
                    node(level, bi, bj)


def stitch_to_arrays(tiles: Sequence[Tile], placement: Placement | None = None, **kwargs) -> tuple[list[np.ndarray], Placement]:
    """Convenience: register (if needed) and return every pyramid level as an in-memory array."""
    placement = placement or register_tiles(tiles)
    writer = ArrayWriter()
    assemble(tiles, placement, writer, **kwargs)
    return writer.levels, placement
