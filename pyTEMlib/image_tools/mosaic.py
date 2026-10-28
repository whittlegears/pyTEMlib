"""
Stitch a grid of microscope tiles into one map, using where the stage said each tile was.

Survey maps are taken at magnifications where there are no atomic columns to
match on, and the stage never lands exactly where it was told. So placement
starts from each tile's reported stage position, then refines it:

1. Neighbouring tiles that overlap are matched by phase correlation of the
   overlap region. Each match gives a measured offset and a confidence.
2. A weighted least-squares solve combines the stage positions (weak prior)
   with the confident matches (strong constraints). A tile with no good match
   keeps its stage position, so a blank area cannot pull a tile away.
3. The canvas is written in horizontal bands with linear blending in the
   overlaps, and each pyramid level is made from the one above by 2x2 averaging
   band by band, so ``assemble`` itself needs memory for one band per level,
   not for the whole map. The tiles are held in memory, and a sink that keeps
   every band (``stitch_to_arrays``, the MCP server's ``assemble_map``) still
   ends up holding the whole map; only a sink that writes bands out as they
   arrive keeps the total bounded.

Coordinates: images are [row, column] with rows along +y and columns along
+x; stage positions are (x, y) in metres and mark the tile centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage.registration import phase_cross_correlation


@dataclass
class Tile:
    image: np.ndarray
    stage_xy_m: tuple[float, float]
    pixel_size_m: float
    index: int = 0
    attrs: dict = field(default_factory=dict)


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

    def pixel_to_stage(self, row: float, col: float) -> tuple[float, float]:
        return (self.origin_xy_m[0] + col * self.pixel_size_m, self.origin_xy_m[1] + row * self.pixel_size_m)

    def stage_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return ((y - self.origin_xy_m[1]) / self.pixel_size_m, (x - self.origin_xy_m[0]) / self.pixel_size_m)


def stage_layout(centres_xy_m, shapes, pixel_sizes_m) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Top-left canvas positions (row, col) from tile centres and image shapes alone (no pixels needed).

    Returns (positions, pixel size, origin); see ``place_by_stage``.
    """
    if len(centres_xy_m) == 0:
        raise ValueError("no tiles to place")
    pixel = float(pixel_sizes_m[0])
    if any(abs(p - pixel) > 1e-6 * pixel for p in pixel_sizes_m):
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

    Refinement moves tiles by at most a fraction of their overlap, so the
    stitched canvas is within a few pixels of this.
    """
    positions, _pixel, _origin = stage_layout(centres_xy_m, shapes, pixel_sizes_m)
    extent = (positions + np.asarray(shapes, dtype=np.float64)[:, :2]).max(axis=0)
    return int(np.ceil(extent[0])), int(np.ceil(extent[1]))


def place_by_stage(tiles: Sequence[Tile]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Top-left canvas positions (row, col) from stage positions; returns (positions, pixel size, origin)."""
    if not tiles:
        raise ValueError("no tiles to place")
    return stage_layout([t.stage_xy_m for t in tiles], [t.image.shape[:2] for t in tiles], [t.pixel_size_m for t in tiles])


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def pairwise_offsets(
    tiles: Sequence[Tile],
    positions: np.ndarray,
    min_overlap_px: int = 16,
    max_correction_px: float | None = None,
    on_pair: Callable[[int, int], None] | None = None,
) -> list[PairMatch]:
    """Measure how far each overlapping pair is from where the stage put it.

    Whole tiles are compared with masked normalised cross-correlation, which
    allows any partial overlap, so a stage error almost as large as the
    overlap itself is still measured (cropping both tiles to the predicted
    overlap strip and phase-correlating that biases or loses large shifts).

    ``on_pair(done, total)`` is called before each overlapping pair is matched
    and once at the end; it may raise (e.g. a job's abort) to stop early.
    """
    matches: list[PairMatch] = []
    shapes = np.array([t.image.shape[:2] for t in tiles])
    # Overlapping pairs first (bounding boxes only, cheap), so progress can report a total.
    pairs: list[tuple[int, int, int, int]] = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            top = max(positions[i, 0], positions[j, 0])
            left = max(positions[i, 1], positions[j, 1])
            bottom = min(positions[i, 0] + shapes[i, 0], positions[j, 0] + shapes[j, 0])
            right = min(positions[i, 1] + shapes[i, 1], positions[j, 1] + shapes[j, 1])
            height, width = int(bottom - top), int(right - left)
            if height >= min_overlap_px and width >= min_overlap_px:
                pairs.append((i, j, height, width))
    for done, (i, j, height, width) in enumerate(pairs):
        if on_pair is not None:
            on_pair(done, len(pairs))  # may raise to stop registration
        a = tiles[i].image.astype(np.float64)
        b = tiles[j].image.astype(np.float64)
        if a.std() == 0 or b.std() == 0:
            continue
        predicted = positions[j] - positions[i]
        overlap_ratio = 0.5 * min(height * width / a.size, height * width / b.size)
        shift = phase_cross_correlation(a, b, reference_mask=np.ones_like(a, bool), moving_mask=np.ones_like(b, bool), overlap_ratio=overlap_ratio)
        shift = np.asarray(shift[0] if isinstance(shift, tuple) else shift, dtype=np.float64)
        # shift moves b onto a, so it is b's position relative to a.
        correction = shift - predicted
        limit = max_correction_px if max_correction_px is not None else 0.6 * min(height, width)
        if np.hypot(*correction) > limit:
            continue
        confidence = _overlap_ncc(a, b, shift, min_overlap_px)
        if confidence is None:
            continue
        matches.append(PairMatch(i, j, correction, confidence))
    if on_pair is not None:
        on_pair(len(pairs), len(pairs))
    return matches


def _overlap_ncc(a: np.ndarray, b: np.ndarray, rel: np.ndarray, min_overlap_px: int) -> float | None:
    """NCC of the region a and b share when b sits at ``rel`` (row, col) relative to a."""
    dr, dc = int(round(rel[0])), int(round(rel[1]))
    r0, r1 = max(0, dr), min(a.shape[0], dr + b.shape[0])
    c0, c1 = max(0, dc), min(a.shape[1], dc + b.shape[1])
    if r1 - r0 < min_overlap_px or c1 - c0 < min_overlap_px:
        return None
    return _ncc(a[r0:r1, c0:c1], b[r0 - dr : r1 - dr, c0 - dc : c1 - dc])


def solve_positions(
    stage_positions: np.ndarray,
    matches: Sequence[PairMatch],
    stage_sigma_px: float,
    match_sigma_px: float = 0.5,
    min_confidence: float = 0.3,
    outlier_px: float = 3.0,
) -> tuple[np.ndarray, list[PairMatch]]:
    """Weighted least squares: stage positions as a weak prior, confident matches as strong constraints.

    A match that disagrees with the solution by more than ``outlier_px`` (and
    by more than three times the typical disagreement) is dropped and the
    solve repeated, so one wrong match cannot drag its neighbours. Returns the
    positions and the matches that were used.
    """
    used = [m for m in matches if m.confidence >= min_confidence]
    positions = stage_positions.copy()
    for _ in range(5):
        positions = _solve_once(stage_positions, used, stage_sigma_px, match_sigma_px)
        if not used:
            break
        residuals = np.array([np.hypot(*((positions[m.j] - positions[m.i]) - (stage_positions[m.j] - stage_positions[m.i] + m.offset_px))) for m in used])
        cutoff = max(outlier_px, 3.0 * float(np.median(residuals)))
        worst = int(np.argmax(residuals))
        if residuals[worst] <= cutoff:
            break
        used = [m for k, m in enumerate(used) if k != worst]
    return positions, used


def _solve_once(stage_positions: np.ndarray, matches: Sequence[PairMatch], stage_sigma_px: float, match_sigma_px: float) -> np.ndarray:
    n = len(stage_positions)
    rows, cols, vals, rhs_r, rhs_c = [], [], [], [], []
    w_stage = 1.0 / max(stage_sigma_px, 1e-6)
    for k in range(n):
        rows.append(k), cols.append(k), vals.append(w_stage)
        rhs_r.append(w_stage * stage_positions[k, 0])
        rhs_c.append(w_stage * stage_positions[k, 1])
    eq = n
    for m in matches:
        w = m.confidence / match_sigma_px
        target = stage_positions[m.j] - stage_positions[m.i] + m.offset_px
        rows += [eq, eq]
        cols += [m.j, m.i]
        vals += [w, -w]
        rhs_r.append(w * target[0])
        rhs_c.append(w * target[1])
        eq += 1
    a = sparse.csr_matrix((vals, (rows, cols)), shape=(eq, n))
    solved_r = lsqr(a, np.array(rhs_r), atol=1e-12, btol=1e-12)[0]
    solved_c = lsqr(a, np.array(rhs_c), atol=1e-12, btol=1e-12)[0]
    return np.stack([solved_r, solved_c], axis=1)


def register_tiles(
    tiles: Sequence[Tile],
    stage_sigma_m: float = 200e-9,
    min_confidence: float = 0.3,
    refine: bool = True,
    on_pair: Callable[[int, int], None] | None = None,
) -> Placement:
    """Place tiles by stage position, then refine with image matching.

    ``on_pair(done, total)`` reports matching progress over the overlapping
    pairs (the slow part) and may raise to abort; see ``pairwise_offsets``.
    """
    stage_px, pixel, origin = place_by_stage(tiles)
    matches = pairwise_offsets(tiles, stage_px, on_pair=on_pair) if refine else []
    positions, used = solve_positions(stage_px, matches, stage_sigma_m / pixel, min_confidence=min_confidence) if matches else (stage_px.copy(), [])
    used_ids = {id(m) for m in used}
    for m in matches:
        m.used = id(m) in used_ids
    # Shift so the canvas starts at (0, 0), and move the origin with it.
    shift = positions.min(axis=0)
    positions = positions - shift
    stage_px = stage_px - shift
    origin = (origin[0] + shift[1] * pixel, origin[1] + shift[0] * pixel)
    shapes = np.array([t.image.shape[:2] for t in tiles])
    canvas = tuple(int(np.ceil(v)) for v in (positions + shapes).max(axis=0))
    return Placement(positions, stage_px, matches, canvas, origin, pixel)


def _cast(array: np.ndarray, dtype) -> np.ndarray:
    return np.rint(array).astype(dtype) if np.issubdtype(dtype, np.integer) else array.astype(dtype)


def _blend_weight(shape: tuple[int, int]) -> np.ndarray:
    """Linear ramp to the tile edges, so overlaps cross-fade instead of showing a seam."""
    r = np.minimum(np.arange(shape[0]) + 1, shape[0] - np.arange(shape[0])).astype(np.float32)
    c = np.minimum(np.arange(shape[1]) + 1, shape[1] - np.arange(shape[1])).astype(np.float32)
    return np.outer(r, c)


class PyramidWriter:
    """Receives level-0 bands in order and emits every pyramid level band by band.

    ``sink(level, row0, band)`` is called for each finished band of each level.
    """

    def __init__(self, width: int, levels: int, sink: Callable[[int, int, np.ndarray], None]):
        self.sink = sink
        self.levels = levels
        self._pending: list[np.ndarray | None] = [None] * levels
        self._next_row = [0] * levels
        self._widths = [width]
        for _ in range(1, levels):
            self._widths.append(max(1, self._widths[-1] // 2))

    def push(self, level: int, band: np.ndarray) -> None:
        self.sink(level, self._next_row[level], band)
        self._next_row[level] += band.shape[0]
        if level + 1 >= self.levels:
            return
        pending = band if self._pending[level] is None else np.vstack([self._pending[level], band])
        usable = (pending.shape[0] // 2) * 2
        width = self._widths[level + 1] * 2
        if usable:
            block = pending[:usable, :width].astype(np.float32)
            down = block.reshape(usable // 2, 2, width // 2, 2).mean(axis=(1, 3))
            self.push(level + 1, _cast(down, band.dtype))
        self._pending[level] = pending[usable:] if usable < pending.shape[0] else None

    def flush(self) -> None:
        for level in range(self.levels - 1):
            pending = self._pending[level]
            if pending is not None and pending.shape[0]:
                width = self._widths[level + 1] * 2
                down = pending[:, :width].astype(np.float32).reshape(1, pending.shape[0], width // 2, 2).mean(axis=(1, 3))
                self._pending[level] = None
                self.push(level + 1, _cast(down, pending.dtype))


def pyramid_levels(canvas_shape: tuple[int, int], min_size: int = 256) -> int:
    levels = 1
    h, w = canvas_shape
    while max(h, w) > min_size:
        h, w = max(1, h // 2), max(1, w // 2)
        levels += 1
    return levels


def assemble(
    tiles: Sequence[Tile],
    placement: Placement,
    sink: Callable[[int, int, np.ndarray], None],
    band_rows: int = 1024,
    levels: int | None = None,
    dtype=None,
) -> int:
    """Blend tiles onto the canvas band by band and emit all pyramid levels to ``sink``.

    Returns the number of levels written. Peak memory is about
    ``band_rows x canvas width x 8 bytes`` plus the tiles themselves.
    """
    height, width = placement.canvas_shape
    levels = levels or pyramid_levels(placement.canvas_shape)
    dtype = dtype or np.result_type(*[t.image.dtype for t in tiles])
    writer = PyramidWriter(width, levels, sink)
    weight_by_shape = {shape: _blend_weight(shape) for shape in {t.image.shape[:2] for t in tiles}}  # one per shape, not per tile
    weights = [weight_by_shape[t.image.shape[:2]] for t in tiles]
    tops = np.round(placement.positions_px).astype(int)
    for row0 in range(0, height, band_rows):
        rows = min(band_rows, height - row0)
        acc = np.zeros((rows, width), dtype=np.float64)
        wsum = np.zeros((rows, width), dtype=np.float64)
        for t, w, (top, left) in zip(tiles, weights, tops):
            h, wd = t.image.shape[:2]
            r0, r1 = max(row0, top), min(row0 + rows, top + h)
            if r0 >= r1:
                continue
            c0, c1 = max(0, left), min(width, left + wd)
            tr0, tc0 = r0 - top, c0 - left
            sub = t.image[tr0 : tr0 + (r1 - r0), tc0 : tc0 + (c1 - c0)].astype(np.float64)
            ws = w[tr0 : tr0 + (r1 - r0), tc0 : tc0 + (c1 - c0)]
            acc[r0 - row0 : r1 - row0, c0:c1] += sub * ws
            wsum[r0 - row0 : r1 - row0, c0:c1] += ws
        band = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
        band = _cast(band, dtype)
        writer.push(0, band)
    writer.flush()
    return levels


def stitch_to_arrays(tiles: Sequence[Tile], placement: Placement | None = None, **kwargs) -> tuple[list[np.ndarray], Placement]:
    """Convenience: register (if needed) and return every pyramid level as an in-memory array."""
    placement = placement or register_tiles(tiles)
    bands: dict[int, list[tuple[int, np.ndarray]]] = {}
    assemble(tiles, placement, lambda level, row0, band: bands.setdefault(level, []).append((row0, band)), **kwargs)
    levels = [np.vstack([b for _, b in sorted(bands[k], key=lambda x: x[0])]) for k in sorted(bands)]
    return levels, placement
