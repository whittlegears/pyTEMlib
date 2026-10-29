"""How stitching accuracy, time and memory grow with the number of tiles, against ground truth.

A synthetic sample (Gaussian particles, sparse enough that some tiles are
almost blank) is imaged as a serpentine grid of tiles. Each tile's content
sits at its true position; the reported stage position adds a stage error
(``--stage-noise`` of the field of view, 1 sigma) and the sample drift at the
tile's acquisition time (a constant rate plus a random walk). Tiles are made
on demand, so maps far larger than memory can be registered.

Three placements are compared with the truth, after removing the common
offset: stage only, naive appending (each tile placed from the previous one's
match, the way an unbounded "append to the map" stitcher works), and
``register_tiles`` (bounded search plus a global solve with a drift model).

    python benchmarks/mosaic_scale.py --sides 10,32,100 --assemble

Prints one JSON line per size.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache

import numpy as np
import psutil

from pyTEMlib.image_tools.mosaic import Tile, assemble, bounded_shift, register_tiles

PIXEL = 1e-9


class Scene:
    """Particles spread over the map, looked up per tile through a grid (O(particles in the tile))."""

    def __init__(self, extent_px: float, density: float, seed: int):
        rng = np.random.default_rng(seed)
        count = int(density * extent_px * extent_px)
        self.xy = rng.uniform(-50, extent_px + 50, size=(count, 2))
        self.sigma = rng.uniform(1.5, 4.0, size=count)
        self.amp = rng.uniform(200, 900, size=count)
        self.cell = 64.0
        keys = np.floor(self.xy / self.cell).astype(np.int64)
        self.buckets: dict[tuple[int, int], np.ndarray] = {}
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys = keys[order]
        starts = np.flatnonzero(np.r_[True, np.any(np.diff(keys, axis=0) != 0, axis=1)])
        ends = np.r_[starts[1:], len(keys)]
        for s, e in zip(starts, ends):
            self.buckets[(int(keys[s, 0]), int(keys[s, 1]))] = order[s:e]

    def render(self, top: float, left: float, size: int, rng: np.random.Generator) -> np.ndarray:
        rows = top + np.arange(size)
        cols = left + np.arange(size)
        image = np.full((size, size), 50.0)
        r0, r1 = int(np.floor((top - 12) / self.cell)), int(np.floor((top + size + 12) / self.cell))
        c0, c1 = int(np.floor((left - 12) / self.cell)), int(np.floor((left + size + 12) / self.cell))
        for a in range(r0, r1 + 1):
            for b in range(c0, c1 + 1):
                for k in self.buckets.get((a, b), ()):
                    y, x = self.xy[k]
                    gy = np.exp(-0.5 * ((rows - y) / self.sigma[k]) ** 2)
                    gx = np.exp(-0.5 * ((cols - x) / self.sigma[k]) ** 2)
                    image += self.amp[k] * np.outer(gy, gx)
        return rng.poisson(image).astype(np.uint16)


class SyntheticRun:
    """A serpentine run of ``side`` x ``side`` tiles whose images are rendered when asked for."""

    def __init__(self, side: int, tile: int, overlap: float, stage_noise: float, drift_px_per_tile: float, seed: int, density: float = 8e-4):
        step = tile * (1 - overlap)
        self.tile = tile
        rng = np.random.default_rng(seed + 1)
        order = []
        for r in range(side):
            cols = range(side) if r % 2 == 0 else reversed(range(side))
            order += [(r, c) for c in cols]
        n = len(order)
        self.commanded = np.array([[r * step, c * step] for r, c in order], dtype=np.float64)  # top-left (row, col)
        self.times = np.arange(n, dtype=np.float64)  # one tile per second
        walk = np.cumsum(rng.normal(0, 0.15 * drift_px_per_tile, size=(n, 2)), axis=0)
        self.drift = np.outer(self.times, [drift_px_per_tile, 0.6 * drift_px_per_tile]) + walk
        self.truth = self.commanded + rng.normal(0, 0.05 * tile, size=(n, 2))  # where each tile's content is: the stage misses its target
        # What the stage reports: the truth, off by its readout error and by the sample drift since the start.
        self.reported = self.truth + self.drift + rng.normal(0, stage_noise * tile, size=(n, 2))
        extent = side * step + tile + abs(self.drift).max() + 100
        self.scene = Scene(extent, density=density, seed=seed)
        self.seed = seed

    def __len__(self):
        return len(self.truth)

    @lru_cache(maxsize=4096)
    def image(self, k: int) -> np.ndarray:
        top, left = self.truth[k]
        return self.scene.render(top, left, self.tile, np.random.default_rng(self.seed * 1_000_003 + k))

    def __getitem__(self, k):
        return LazyTile(self, k)


class LazyTile:
    def __init__(self, run: SyntheticRun, k: int):
        self.run, self.index = run, k
        top, left = run.reported[k]
        half = run.tile / 2
        self.stage_xy_m = ((left + half) * PIXEL, (top + half) * PIXEL)
        self.pixel_size_m = PIXEL
        self.attrs = {}
        self.acquired_s = float(run.times[k])
        self.shape = (run.tile, run.tile)

    @property
    def image(self):
        return self.run.image(self.index)


def mosaic_pairs(placement, run):
    from pyTEMlib.image_tools.mosaic import overlapping_pairs

    return overlapping_pairs(placement.stage_positions_px, np.full((len(run), 2), run.tile), 16)


def errors(placed: np.ndarray, truth: np.ndarray, subset: np.ndarray | None = None) -> dict:
    d = placed - truth
    d = d - np.median(d, axis=0)
    e = np.hypot(d[:, 0], d[:, 1])
    if subset is not None:
        e = e[subset]
    return {"median_px": round(float(np.median(e)), 3), "p95_px": round(float(np.percentile(e, 95)), 3), "max_px": round(float(e.max()), 3)}


def naive_append(run: SyntheticRun, stage_px: np.ndarray) -> np.ndarray:
    """Each tile placed from the previous tile by their match alone (no global solve, no bound beyond the overlap)."""
    placed = stage_px.copy()
    for k in range(1, len(run)):
        a = run.image(k - 1).astype(np.float64)
        b = run.image(k).astype(np.float64)
        predicted = stage_px[k] - stage_px[k - 1]
        shift = bounded_shift(a, b, predicted, 0.45 * run.tile, 0.1)
        placed[k] = placed[k - 1] + (shift if shift is not None else predicted)
    return placed


class DiscardWriter:
    def create(self, level, shape, dtype, block):
        pass

    def write(self, level, row0, col0, data):
        pass


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def run_size(side: int, args) -> dict:
    run = SyntheticRun(side, args.tile, args.overlap, args.stage_noise, args.drift, args.seed, args.density)
    tiles = [run[k] for k in range(len(run))]
    stage_px = run.reported - run.reported.min(axis=0)
    truth = run.truth
    row = {"tiles": len(run), "side": side, "tile_px": args.tile, "overlap": args.overlap, "density_per_px2": args.density, "stage_noise_fov": args.stage_noise, "drift_px_per_tile": args.drift}
    row["stage_only"] = errors(stage_px, truth)
    if len(run) <= args.naive_max:
        row["naive_append"] = errors(naive_append(run, stage_px), truth)
    before = rss_mb()
    t = time.time()
    placement = register_tiles(tiles, stage_sigma_m=args.stage_noise * args.tile * PIXEL, max_drift_m_per_s=3 * args.drift * PIXEL, drift_knot_s=args.knot)
    row["register_s"] = round(time.time() - t, 1)
    row["register_rss_growth_mb"] = round(rss_mb() - before)
    row["bounded_drift_solve"] = errors(placement.positions_px, truth)
    if placement.fallback is not None and (~placement.fallback).any():
        row["bounded_drift_solve"]["matched_tiles_median_px"] = errors(placement.positions_px, truth, ~placement.fallback)["median_px"]
    row["pairs_overlapping"] = len(mosaic_pairs(placement, run))
    row["solver"] = placement.solver
    row["matches_used"] = int(sum(m.used for m in placement.matches))
    row["fallback_tiles"] = int(placement.fallback.sum()) if placement.fallback is not None else None
    if placement.drift_px is not None:
        row["drift_fit_error_px"] = round(float(np.hypot(*(placement.drift_px[-1] - placement.drift_px[0] + (run.drift[-1] - run.drift[0])))), 2)
    if args.assemble:
        before = rss_mb()
        t = time.time()
        assemble(tiles, placement, DiscardWriter(), block=512)
        row["assemble_s"] = round(time.time() - t, 1)
        row["assemble_rss_growth_mb"] = round(rss_mb() - before)
        row["canvas"] = list(placement.canvas_shape)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sides", default="10,32,100")
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--overlap", type=float, default=0.3, help="overlap fraction; the matcher needs at least 16 px of overlap")
    parser.add_argument("--stage-noise", type=float, default=0.05)
    parser.add_argument("--drift", type=float, default=0.02, help="drift in pixels per tile (one tile per second)")
    parser.add_argument("--knot", type=float, default=120.0)
    parser.add_argument("--naive-max", type=int, default=20000)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--density", type=float, default=8e-4, help="particles per pixel^2 of the synthetic sample")
    parser.add_argument("--out", default=None, help="append each row to this JSON-lines file")
    args = parser.parse_args()
    for side in (int(s) for s in args.sides.split(",")):
        row = json.dumps(run_size(side, args))
        print(row, flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as handle:
                handle.write(row + "\n")


if __name__ == "__main__":
    main()
