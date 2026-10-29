"""Bounded registration and the drift model against ground truth (synthetic runs from benchmarks/mosaic_scale.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyTEMlib.image_tools import mosaic

spec = importlib.util.spec_from_file_location("mosaic_scale", Path(__file__).resolve().parents[1] / "benchmarks" / "mosaic_scale.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_a_pair_whose_true_offset_lies_outside_the_bound_gives_no_match():
    rng = np.random.default_rng(3)
    scene = np.zeros((128, 260))
    for y, x in rng.uniform([8, 8], [120, 252], size=(60, 2)):
        rr, cc = np.ogrid[:128, :260]
        scene += 400 * np.exp(-((rr - y) ** 2 + (cc - x) ** 2) / 8.0)
    scene = rng.poisson(scene + 30).astype(np.float64)
    # b really sits 112 px right of a; the stage says 104 px, so the true correction is +8 px.
    a = mosaic.Tile(scene[:, :128], (64 * PIXEL, 64 * PIXEL), PIXEL, 0)
    b = mosaic.Tile(scene[:, 112:240], ((104 + 64) * PIXEL, 64 * PIXEL), PIXEL, 1)
    positions = np.array([[0.0, 0.0], [0.0, 104.0]])
    # With a 4 px bound the true peak is outside it, and whatever the window holds is noise: no match, not a wrong one.
    assert mosaic.pairwise_offsets([a, b], positions, max_correction_px=4.0) == []
    loose = mosaic.pairwise_offsets([a, b], positions, max_correction_px=12.0)
    assert len(loose) == 1 and np.allclose(loose[0].offset_px, [0.0, 8.0], atol=0.3)


PIXEL = 1e-9


def test_wrong_matches_are_dropped_by_the_robust_solve():
    run = bench.SyntheticRun(12, 64, 0.3, 0.05, 0.0, 5)
    s = run.reported - run.reported.min(axis=0)
    q = run.truth - run.reported.min(axis=0)
    pairs = mosaic.overlapping_pairs(s, np.full((len(s), 2), 64), 16)
    matches = [mosaic.PairMatch(i, j, (q[j] - q[i]) - (s[j] - s[i]), 1.0) for i, j, _h, _w in pairs]
    rng = np.random.default_rng(0)
    for m in rng.choice(matches, size=len(matches) // 10, replace=False):
        m.offset_px = m.offset_px + rng.choice([-1, 1], size=2) * rng.uniform(4, 9, size=2)  # a wrong peak inside the bound
    positions, used, _k, _d = mosaic.solve_positions(s, matches, 3.2)
    assert bench.errors(positions, q)["median_px"] < 0.5
    bad = {id(m) for m in matches if np.hypot(*((s[m.j] - s[m.i] + m.offset_px) - (q[m.j] - q[m.i]))) > 2}
    assert len(bad & {id(m) for m in used}) <= 0.1 * len(bad)


def test_the_drift_is_recovered_and_the_placement_is_under_a_pixel_on_a_textured_sample():
    run = bench.SyntheticRun(16, 64, 0.3, 0.05, 0.05, 7, density=2e-3)
    tiles = [run[k] for k in range(len(run))]
    placement = mosaic.register_tiles(tiles, stage_sigma_m=0.05 * 64 * bench.PIXEL, max_drift_m_per_s=0.15 * bench.PIXEL)
    true_total = run.drift[-1] - run.drift[0]
    fitted_total = -(placement.drift_px[-1] - placement.drift_px[0])
    assert np.hypot(*(fitted_total - true_total)) < 0.1 * np.hypot(*true_total)
    assert bench.errors(placement.positions_px, run.truth)["median_px"] < 1.0
    assert placement.solver["converged"]


def test_error_does_not_grow_with_the_number_of_tiles():
    medians = []
    for side in (8, 20):
        run = bench.SyntheticRun(side, 64, 0.3, 0.05, 0.02, 11, density=2e-3)
        tiles = [run[k] for k in range(len(run))]
        placement = mosaic.register_tiles(tiles, stage_sigma_m=0.05 * 64 * bench.PIXEL, max_drift_m_per_s=0.06 * bench.PIXEL)
        medians.append(bench.errors(placement.positions_px, run.truth)["median_px"])
    assert medians[1] < 1.0 and medians[1] <= medians[0] + 0.3, medians


def test_the_iterative_solver_agrees_with_the_direct_one_and_reports_convergence(monkeypatch):
    run = bench.SyntheticRun(20, 64, 0.3, 0.05, 0.02, 13)
    s = run.reported - run.reported.min(axis=0)
    q = run.truth - run.reported.min(axis=0)
    pairs = mosaic.overlapping_pairs(s, np.full((len(s), 2), 64), 16)
    matches = [mosaic.PairMatch(i, j, (q[j] - q[i]) - (s[j] - s[i]), 1.0) for i, j, _h, _w in pairs]
    direct_info, iterative_info = {}, {}
    direct, *_ = mosaic.solve_positions(s, matches, 3.2, times_s=run.times, max_drift_px_per_s=0.06, info=direct_info)
    monkeypatch.setattr(mosaic, "DIRECT_SOLVE_MAX", 10)
    iterative, *_ = mosaic.solve_positions(s, matches, 3.2, times_s=run.times, max_drift_px_per_s=0.06, info=iterative_info)
    assert direct_info["method"] == "direct" and iterative_info["method"] == "lsqr" and iterative_info["converged"]
    assert np.abs(direct - iterative).max() < 0.01
    monkeypatch.setattr(mosaic, "LSQR_ITERATIONS", 3)
    starved = {}
    mosaic.solve_positions(s, matches, 3.2, times_s=run.times, max_drift_px_per_s=0.06, info=starved)
    assert starved["converged"] is False and 7 in starved["istop"]  # an iteration limit is reported, not hidden


def test_a_tile_with_no_match_gives_way_to_its_matched_neighbour_where_they_overlap():
    a = mosaic.Tile(np.full((32, 32), 100.0), (16 * PIXEL, 16 * PIXEL), PIXEL, 0)
    b = mosaic.Tile(np.full((32, 32), 200.0), (32 * PIXEL, 16 * PIXEL), PIXEL, 1)
    placement = mosaic.register_tiles([a, b], refine=False)
    placement.fallback = np.array([False, True])
    levels, _ = mosaic.stitch_to_arrays([a, b], placement)
    overlap = levels[0][:, 16:32]
    assert overlap.mean() < 150  # equal weights would give 150 in the middle of the overlap
    assert levels[0][:, 40].mean() == 200  # where only the fallback tile is, it is still drawn


def test_assemble_progress_only_goes_up(tmp_path):
    from pyTEMlib.mcp_server.map_store import MapStore
    from pyTEMlib.mcp_server.map_tools import PROGRESS_STEPS, MapTools

    from .test_map_store_remote_reads import write_layout2

    path, _ = write_layout2(tmp_path / "run.h5")
    tools = MapTools(MapStore(local_roots=[tmp_path]))
    seen = []
    tools._run_assemble({"run": path, "eds_pois": True}, lambda done, total=None: seen.append(done))
    assert seen == sorted(seen) and seen[-1] == PROGRESS_STEPS


solver_spec = importlib.util.spec_from_file_location("solver_scale", Path(__file__).resolve().parents[1] / "benchmarks" / "solver_scale.py")
solver_bench = importlib.util.module_from_spec(solver_spec)
solver_spec.loader.exec_module(solver_bench)


def solve_error(side):
    s, q, times, matches_around = solver_bench.synthetic(side, 64, 0.3, 3.2, 0.02, 0.1, 3)
    info = {}
    _p, _u, knots, drift = mosaic.solve_positions(s, matches_around(s), 3.2, times_s=times, max_drift_px_per_s=0.06, info=info)
    positions, _u, _k, _d = mosaic.solve_positions(s, matches_around(s + mosaic._drift_at(times, knots, drift)), 3.2, times_s=times, max_drift_px_per_s=0.06, info=info)
    d = positions - q
    d -= np.median(d, axis=0)
    return float(np.median(np.hypot(*d.T))), info


def test_tight_matches_keep_the_map_from_stretching_along_the_slow_axis_as_it_grows():
    # 5 % FOV stage noise and drift, matches good to 0.1 px. With the matches weighted at an assumed
    # 0.5 px, the stage prior and the free drift let rows creep apart: 0.24 px at 30 x 30 rising to 0.66 px
    # at 100 x 100 and 17 px at 320 x 320. Weighted by their measured scatter it stays ~0.13 px.
    small, _ = solve_error(30)
    large, info = solve_error(100)
    assert large < 0.3 and large < 1.5 * small, (small, large)
    assert info["match_sigma_px"] == pytest.approx(0.1, abs=0.02)
