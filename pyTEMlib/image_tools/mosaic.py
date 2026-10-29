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
   with no usable match keeps its drift-corrected stage position and is
   flagged in ``Placement.fallback``.
3. The canvas is written block by block (``block`` x ``block`` pixels) in
   quadtree order: each pyramid level's block is made from the four blocks
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
from scipy import sparse
from scipy.sparse.linalg import lsqr, spsolve
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked

# No stage reports positions better than about a pixel at survey magnification (readout, scan
# distortion); a smaller sigma would make the prior rigid and leave no room to measure drift.
MIN_STAGE_SIGMA_PX = 1.0
HUBER_PX = 0.5  # matches agree with the solution to ~0.1 px; beyond this their weight falls off
ROBUST_ROUNDS = 10
DIRECT_SOLVE_MAX = 400_000  # unknowns solved directly; more use lsqr
LSQR_ITERATIONS = 50_000
FALLBACK_WEIGHT = 0.25  # blend weight of a tile with no usable match: its neighbours' pixels win where they overlap
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
        overlap_ratio = 0.5 * min(height * width / a.size, height * width / b.size)
        shift = bounded_shift(a, b, predicted, bound, overlap_ratio)
        if shift is None:
            continue
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
    used = [m for m in matches if m.confidence >= min_confidence]
    knots = _knots(times_s, drift_knot_s, len(stage_positions)) if max_drift_px_per_s > 0 else None
    robust = np.ones(len(used))
    positions, drift = _solve_once(stage_positions, used, stage_sigma_px, match_sigma_px, times_s, knots, max_drift_px_per_s, robust, info)
    # A match must also agree with the stage once the fitted drift is removed: the two stage
    # readings differ by at most bound_sigmas combined stage sigmas. This catches a wrong peak in a
    # low-texture overlap that nothing else contradicts (its only other constraint is the weak prior).
    bound = bound_sigmas * np.sqrt(2.0) * max(stage_sigma_px, MIN_STAGE_SIGMA_PX) + 1.0
    for round_ in range(ROBUST_ROUNDS):
        if not used:
            break
        i = np.array([m.i for m in used])
        j = np.array([m.j for m in used])
        offsets = np.array([m.offset_px for m in used])
        residuals = np.hypot(*((positions[j] - positions[i]) - (stage_positions[j] - stage_positions[i] + offsets)).T)
        drift_at = _drift_at(times_s, knots, drift) if drift is not None else np.zeros_like(stage_positions)
        disagreement = np.hypot(*(offsets - (drift_at[j] - drift_at[i])).T)
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        cutoff = max(outlier_px, float(np.median(residuals)) + 5.0 * 1.4826 * mad)
        keep = residuals <= cutoff
        if round_ > 0:  # the drift is only trusted once a first robust solve has fitted it
            keep &= disagreement <= bound
        # Huber weights: a match a little off (a weak, slightly wrong peak) counts less instead of
        # bending its neighbours and the drift fit; one far off is dropped outright.
        new_robust = np.minimum(1.0, HUBER_PX / np.maximum(residuals, 1e-9))[keep]
        if keep.all() and np.allclose(new_robust, robust, atol=0.02):
            break
        used = [m for m, k in zip(used, keep) if k]
        robust = new_robust
        positions, drift = _solve_once(stage_positions, used, stage_sigma_px, match_sigma_px, times_s, knots, max_drift_px_per_s, robust, info)
    return positions, used, knots, drift


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
) -> Placement:
    """Place tiles by stage position, then refine with bounded image matching and (with times) a drift model.

    ``stage_sigma_m``: 1-sigma error of the reported stage positions (sets the
    search bound and the prior). ``max_drift_m_per_s``: the most the sample
    drifts relative to the stage; 0 turns the drift model off. Tiles need
    ``acquired_s`` for the drift model.
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
    else:
        positions, used = stage_px.copy(), []
    used_ids = {id(m) for m in used}
    matched = np.zeros(len(tiles), bool)
    for m in matches:
        m.used = id(m) in used_ids
        if m.used:
            matched[m.i] = matched[m.j] = True
    # Shift so the canvas starts at (0, 0), and move the origin with it.
    shift = positions.min(axis=0)
    positions = positions - shift
    stage_px = stage_px - shift
    origin = (origin[0] + shift[1] * pixel, origin[1] + shift[0] * pixel)
    shapes = tile_arrays(tiles)["shapes"]
    canvas = tuple(int(np.ceil(v)) for v in (positions + shapes).max(axis=0))
    return Placement(positions, stage_px, matches, canvas, origin, pixel, fallback=~matched if refine else None, drift_times_s=knots, drift_px=drift, solver=info or None)


def _cast(array: np.ndarray, dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(dtype)
    return array.astype(dtype)


def _blend_weight(shape: tuple[int, int]) -> np.ndarray:
    """Linear ramp to the tile edges, so overlaps cross-fade instead of showing a seam."""
    r = np.minimum(np.arange(shape[0]) + 1, shape[0] - np.arange(shape[0])).astype(np.float32)
    c = np.minimum(np.arange(shape[1]) + 1, shape[1] - np.arange(shape[1])).astype(np.float32)
    return np.outer(r, c)


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
    tops = np.round(placement.positions_px).astype(np.int64)
    tile_shapes = tile_arrays(tiles)["shapes"]
    gains = np.where(placement.fallback, FALLBACK_WEIGHT, 1.0) if placement.fallback is not None else np.ones(len(tops))
    index: dict[tuple[int, int], list[int]] = {}
    for k, ((top, left), (h, w)) in enumerate(zip(tops, tile_shapes)):
        for bi in range(max(0, top) // block, (top + h - 1) // block + 1):
            for bj in range(max(0, left) // block, (left + w - 1) // block + 1):
                index.setdefault((bi, bj), []).append(k)
    weight_by_shape: dict[tuple[int, int], np.ndarray] = {}
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
            top, left = tops[k]
            th, tw = tile_shapes[k]
            rr0, rr1 = max(r0, top), min(r1, top + th)
            cc0, cc1 = max(c0, left), min(c1, left + tw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            shape = (int(th), int(tw))
            weight = weight_by_shape.get(shape)
            if weight is None:
                weight = weight_by_shape[shape] = _blend_weight(shape)
            image = np.asarray(tiles[k].image)
            sub = image[rr0 - top : rr1 - top, cc0 - left : cc1 - left].astype(np.float64)
            ws = weight[rr0 - top : rr1 - top, cc0 - left : cc1 - left] * gains[k]
            acc[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] += sub * ws
            wsum[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] += ws
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
