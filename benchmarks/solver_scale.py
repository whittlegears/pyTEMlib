"""The position-and-drift solve alone, at sizes too large to register images for in a test.

Matches are made from the ground truth with match noise (``--match-noise`` px) and stage positions
with stage noise and drift, as in mosaic_scale.py, so only the solve is measured: its method (direct
or lsqr), whether it converged, iterations, time, peak memory growth and placement error. As in
register_tiles, the pairs are found twice: around the stage positions, then around the stage
positions plus the drift the first solve fitted.

Each size runs in its own process under a wall-clock cap (``--cap-s``); a size that does not finish
in time is recorded as a timeout row instead of running on.

    python benchmarks/solver_scale.py --sides 100,320,650 --cap-s 1500 --out benchmarks/results/solver_scale.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import psutil

from pyTEMlib.image_tools import mosaic


def synthetic(side: int, tile: int, overlap: float, stage_noise_px: float, drift_px_per_tile: float, match_noise_px: float, seed: int):
    rng = np.random.default_rng(seed)
    step = tile * (1 - overlap)
    rows, cols = np.divmod(np.arange(side * side), side)
    cols = np.where(rows % 2 == 1, side - 1 - cols, cols)
    commanded = np.stack([rows * step, cols * step], axis=1).astype(np.float64)
    truth = commanded + rng.normal(0, 0.05 * tile, commanded.shape)
    times = np.arange(side * side, dtype=np.float64)
    drift = np.outer(times, [drift_px_per_tile, 0.6 * drift_px_per_tile])
    reported = truth + drift + rng.normal(0, stage_noise_px, truth.shape)
    s = reported - reported.min(axis=0)
    q = truth - reported.min(axis=0)
    def matches_around(expected):
        pairs = mosaic.overlapping_pairs(expected, np.full((len(s), 2), tile), 16)
        return [mosaic.PairMatch(i, j, (q[j] - q[i]) - (s[j] - s[i]) + rng.normal(0, match_noise_px, 2), 1.0) for i, j, _h, _w in pairs]

    return s, q, times, matches_around


def run_side(side: int, match_noise: float, out: str | None) -> None:
    process = psutil.Process(os.getpid())
    s, q, times, matches_around = synthetic(side, 64, 0.3, 3.2, 0.02, match_noise, 3)
    before = process.memory_info().rss
    info: dict = {}
    t0 = time.time()
    positions, used, knots, drift = mosaic.solve_positions(s, matches_around(s), 3.2, times_s=times, max_drift_px_per_s=0.06, info=info)
    matches = matches_around(s + mosaic._drift_at(times, knots, drift))
    positions, used, knots, drift = mosaic.solve_positions(s, matches, 3.2, times_s=times, max_drift_px_per_s=0.06, info=info)
    d = positions - q
    d -= np.median(d, axis=0)
    e = np.hypot(*d.T)
    row = {
        "tiles": side * side, "match_noise_px": match_noise, "unknowns": side * side + (0 if knots is None else len(knots) - 1), "matches": len(matches),
        "solver": info, "solve_s": round(time.time() - t0, 1), "rss_growth_mb": round((process.memory_info().rss - before) / 1e6),
        "median_px": round(float(np.median(e)), 3), "p95_px": round(float(np.percentile(e, 95)), 3), "max_px": round(float(e.max()), 3),
    }
    write(row, out)


def write(row: dict, out: str | None) -> None:
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sides", default="100,320,650")
    parser.add_argument("--match-noise", type=float, default=0.1)
    parser.add_argument("--cap-s", type=float, default=1500.0, help="wall-clock limit per size (s)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    for side in (int(v) for v in args.sides.split(",")):
        worker = multiprocessing.Process(target=run_side, args=(side, args.match_noise, args.out))
        t0 = time.time()
        worker.start()
        worker.join(args.cap_s)
        if worker.is_alive():
            worker.kill()
            worker.join()
            write({"tiles": side * side, "timeout_s": args.cap_s, "note": "did not finish within the cap; stopped"}, args.out)
        elif worker.exitcode:
            write({"tiles": side * side, "failed": f"exit code {worker.exitcode}", "after_s": round(time.time() - t0, 1)}, args.out)


if __name__ == "__main__":
    main()
