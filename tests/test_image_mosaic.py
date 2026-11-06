"""Stitching (image_mosaic): placement against known tile positions, bounded matching, the drift model,
the drawn map, and stitch_mosaic on sidpy datasets."""

import tracemalloc

import numpy as np
import pytest
from scipy import ndimage

from pyTEMlib.image_tools import image_mosaic as mosaic
from pyTEMlib.image_tools.image_mosaic import Tile, assemble, estimate_scan_linear, register_tiles, solve_gain_offset, stitch_to_arrays

from pyTEMlib.image_tools.image_utilities import get_pixel_size_m, pixel_to_stage

from .conftest import PIXEL, TILE, SyntheticRun, errors, make_tiles, image_dataset, placement_error, scene


def test_refinement_recovers_true_positions_with_stage_error():
    tiles, truth = make_tiles(stage_error_px=0.05 * TILE)
    placement = register_tiles(tiles, stage_sigma_m=0.05 * TILE * PIXEL)
    stage_only = placement_error(placement.stage_positions_px, truth)
    refined = placement_error(placement.positions_px, truth)
    assert np.median(refined) < 1.0
    assert np.median(stage_only) > 5.0


def test_refinement_no_worse_than_stage_when_stage_is_perfect():
    tiles, truth = make_tiles(stage_error_px=0.0)
    placement = register_tiles(tiles, stage_sigma_m=2 * PIXEL)
    assert np.median(placement_error(placement.positions_px, truth)) <= np.median(placement_error(placement.stage_positions_px, truth)) + 0.5


def test_blank_tiles_keep_stage_positions():
    blank = np.full((1300, 1300), 100.0)
    tiles, truth = make_tiles(stage_error_px=0.0, image=blank)
    rng = np.random.default_rng(3)
    for t in tiles:  # each acquisition has its own shot noise
        t.image = t.image + rng.normal(0, 1.0, t.image.shape)
    placement = register_tiles(tiles)
    assert all(m.confidence < 0.3 for m in placement.matches)
    assert np.allclose(placement.positions_px, placement.stage_positions_px)


def test_pixel_stage_round_trip():
    tiles, _truth = make_tiles(stage_error_px=3.0)
    placement = register_tiles(tiles)
    for row, col in [(0, 0), (100.5, 37.25), (placement.canvas_shape[0] - 1, 3)]:
        x, y = placement.pixel_to_stage(row, col)
        r2, c2 = placement.stage_to_pixel(x, y)
        assert abs(r2 - row) < 1e-6 and abs(c2 - col) < 1e-6


def test_stitched_map_matches_scene_and_pyramid_halves():
    image = scene()
    tiles, truth = make_tiles(stage_error_px=4.0, image=image)
    levels, placement = stitch_to_arrays(tiles, register_tiles(tiles, stage_sigma_m=4 * PIXEL))
    assert levels[0].shape == placement.canvas_shape
    for a, b in zip(levels, levels[1:]):
        assert b.shape[0] == a.shape[0] // 2 or b.shape[0] == (a.shape[0] + 1) // 2
    # The stitched map, aligned on the first tile, reproduces the scene.
    offset = np.round(truth[0] - placement.positions_px[0]).astype(int)
    h, w = levels[0].shape
    reference = image[offset[0] : offset[0] + h, offset[1] : offset[1] + w]
    overlap = levels[0][: reference.shape[0], : reference.shape[1]]
    covered = overlap != 0
    assert np.corrcoef(overlap[covered], reference[covered])[0, 1] > 0.98


class CountingWriter:
    def __init__(self):
        self.shapes, self.written = [], []

    def create(self, level, shape, dtype, block):
        self.shapes.append(shape)
        self.written.append(np.zeros(shape, bool))

    def write(self, level, row0, col0, data):
        assert not self.written[level][row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]].any()  # each pixel once
        self.written[level][row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]] = True


def test_assembly_memory_is_bounded_by_block_size_and_writes_every_pixel_once():
    rng = np.random.default_rng(5)
    tiles = [Tile(rng.integers(0, 1000, (512, 512), dtype=np.uint16), ((c * 480 + 256) * PIXEL, (r * 480 + 256) * PIXEL), PIXEL, r * 8 + c) for r in range(8) for c in range(8)]
    placement = register_tiles(tiles, refine=False)
    canvas_bytes = placement.canvas_shape[0] * placement.canvas_shape[1] * 2
    writer = CountingWriter()
    tracemalloc.start()
    assemble(tiles, placement, writer, block=256)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert all(level.all() for level in writer.written)
    assert peak < 0.25 * canvas_bytes * 4  # a float64 working copy of the canvas would be 4x canvas_bytes


def test_mixed_pixel_sizes_rejected():
    tiles, _ = make_tiles(stage_error_px=0.0, grid=2)
    tiles[1] = Tile(tiles[1].image, tiles[1].stage_xy_m, 2 * PIXEL)
    with pytest.raises(ValueError, match="pixel size"):
        register_tiles(tiles)


def test_registration_reports_pairs_and_can_be_stopped():
    tiles, _truth = make_tiles(stage_error_px=3.0, grid=3)
    seen = []
    register_tiles(tiles, on_pair=lambda done, total: seen.append((done, total)))
    total = seen[0][1]
    assert total >= 12 and seen == [(k, total) for k in range(total + 1)]

    class Stop(Exception):
        pass

    def stop_at_three(done, total):
        if done == 3:
            raise Stop

    with pytest.raises(Stop):
        register_tiles(tiles, on_pair=stop_at_three)


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



def test_wrong_matches_are_dropped_by_the_robust_solve():
    run = SyntheticRun(12, 64, 0.3, 0.05, 0.0, 5)
    s = run.reported - run.reported.min(axis=0)
    q = run.truth - run.reported.min(axis=0)
    pairs = mosaic.overlapping_pairs(s, np.full((len(s), 2), 64), 16)
    matches = [mosaic.PairMatch(i, j, (q[j] - q[i]) - (s[j] - s[i]), 1.0) for i, j, _h, _w in pairs]
    rng = np.random.default_rng(0)
    for m in rng.choice(matches, size=len(matches) // 10, replace=False):
        m.offset_px = m.offset_px + rng.choice([-1, 1], size=2) * rng.uniform(4, 9, size=2)  # a wrong peak inside the bound
    positions, used, _k, _d = mosaic.solve_positions(s, matches, 3.2)
    assert errors(positions, q)["median_px"] < 0.5
    bad = {id(m) for m in matches if np.hypot(*((s[m.j] - s[m.i] + m.offset_px) - (q[m.j] - q[m.i]))) > 2}
    assert len(bad & {id(m) for m in used}) <= 0.1 * len(bad)


def test_the_drift_is_recovered_and_the_placement_is_under_a_pixel_on_a_textured_sample():
    run = SyntheticRun(16, 64, 0.3, 0.05, 0.05, 7, density=2e-3)
    tiles = [run[k] for k in range(len(run))]
    placement = mosaic.register_tiles(tiles, stage_sigma_m=0.05 * 64 * PIXEL, max_drift_m_per_s=0.15 * PIXEL)
    true_total = run.drift[-1] - run.drift[0]
    fitted_total = -(placement.drift_px[-1] - placement.drift_px[0])
    assert np.hypot(*(fitted_total - true_total)) < 0.1 * np.hypot(*true_total)
    assert errors(placement.positions_px, run.truth)["median_px"] < 1.0
    assert placement.solver["converged"]


def test_error_does_not_grow_with_the_number_of_tiles():
    medians = []
    for side in (8, 20):
        run = SyntheticRun(side, 64, 0.3, 0.05, 0.02, 11, density=2e-3)
        tiles = [run[k] for k in range(len(run))]
        placement = mosaic.register_tiles(tiles, stage_sigma_m=0.05 * 64 * PIXEL, max_drift_m_per_s=0.06 * PIXEL)
        medians.append(errors(placement.positions_px, run.truth)["median_px"])
    assert medians[1] < 1.0 and medians[1] <= medians[0] + 0.3, medians


def test_the_iterative_solver_agrees_with_the_direct_one_and_reports_convergence(monkeypatch):
    run = SyntheticRun(20, 64, 0.3, 0.05, 0.02, 13)
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


def test_a_tile_with_no_match_is_drawn_only_where_no_placed_tile_is():
    a = mosaic.Tile(np.full((32, 32), 100.0), (16 * PIXEL, 16 * PIXEL), PIXEL, 0)
    b = mosaic.Tile(np.full((32, 32), 200.0), (32 * PIXEL, 16 * PIXEL), PIXEL, 1)
    placement = mosaic.register_tiles([a, b], refine=False)
    placement.fallback = np.array([False, True])
    levels, _ = mosaic.stitch_to_arrays([a, b], placement)
    assert np.allclose(levels[0][:, 17:31], 100.0, atol=1.0)  # the shared area is the placed tile
    assert np.allclose(levels[0][:, 40], 200.0)  # where only the unplaced tile is, it shows (vacuum is not black)


def test_a_map_where_nothing_matched_is_still_drawn():
    tile = mosaic.Tile(np.full((32, 32), 7.0), (16 * PIXEL, 16 * PIXEL), PIXEL, 0)
    levels, placement = mosaic.stitch_to_arrays([tile], mosaic.register_tiles([tile]))
    assert placement.fallback.all() and np.allclose(levels[0], 7.0)


def test_tiles_whose_wrong_links_outnumber_their_right_ones_are_still_placed():
    rng = np.random.default_rng(11)
    grid, step = 5, 200.0
    truth = np.array([[r * step, c * step] for r in range(grid) for c in range(grid)])
    stage = truth + rng.normal(0, 25.0, truth.shape)
    island = {18, 19, 23, 24}  # the bottom-right corner
    matches = []
    for a in range(grid * grid):
        for b in (a + 1, a + grid):
            if b >= grid * grid or (b == a + 1 and b % grid == 0):
                continue
            correction = (truth[b] - truth[a]) - (stage[b] - stage[a]) + rng.normal(0, 0.2, 2)
            matches.append(mosaic.PairMatch(a, b, correction, 0.8))
    # Three wrong links into the island, each confidently wrong in its own way, against its two right ones.
    for a, b in ((13, 18), (14, 19), (17, 18)):
        matches = [m for m in matches if (m.i, m.j) != (a, b)]
        matches.append(mosaic.PairMatch(a, b, (truth[b] - truth[a]) - (stage[b] - stage[a]) + rng.uniform(60, 120, 2) * rng.choice([-1, 1], 2), 0.6))
    positions, used, _knots, _drift = mosaic.solve_positions(stage, matches, 25.0)
    error = (positions - truth) - np.median(positions - truth, axis=0)
    assert np.hypot(*error[sorted(island)].T).max() < 1.0
    assert not any((m.i, m.j) in {(13, 18), (14, 19), (17, 18)} for m in used)


def _centroid(image: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(image.shape)
    return np.array([(image * rows).sum(), (image * cols).sum()]) / image.sum()


def test_a_tile_is_drawn_at_its_sub_pixel_position():
    blob = np.zeros((64, 64))
    blob[32, 32] = 1.0
    blob = ndimage.gaussian_filter(blob, 3.0)
    tile = Tile(blob, (32 * PIXEL, 32 * PIXEL), PIXEL, 0)
    placement = register_tiles([tile], refine=False)
    placement.positions_px = placement.positions_px + np.array([[0.3, 0.7]])
    levels, _ = stitch_to_arrays([tile], placement)
    drawn = _centroid(levels[0].astype(np.float64))
    assert np.allclose(drawn, _centroid(blob) + np.array([0.3, 0.7]), atol=0.02)  # rounding would put it 0.3 off


def test_the_scans_shared_stretch_and_turn_are_measured_and_only_trusted_when_clear():
    rng = np.random.default_rng(3)
    grid = np.array([[r, c] for r in range(12) for c in range(12)], dtype=float) * 300.0
    stage = grid + rng.normal(0, 20.0, grid.shape)  # where the stage says each tile is
    angle = np.deg2rad(0.3)
    a = 1.01 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    images = grid @ a.T  # where the images put them: stretched 1 % and turned 0.3 degrees
    linear, stderr, applied = estimate_scan_linear(stage, images, np.ones(len(grid), bool))
    assert applied and np.allclose(linear, a, atol=4 * stderr.max())
    linear, stderr, applied = estimate_scan_linear(stage, grid, np.ones(len(grid), bool))
    assert not applied  # no distortion: the estimate stays within its errors and nothing is undone
    assert not estimate_scan_linear(stage[:4], images[:4], np.ones(4, bool))[2]  # too few tiles to say


def test_a_stretched_scan_is_drawn_on_the_stages_scale():
    image = scene(size=1400, blobs=900)
    stretch = 1.02
    tiles = []
    for r in range(4):
        for c in range(4):
            centre = np.array([300 + r * 200, 300 + c * 200], dtype=float)
            # The scan sees 2 % more sample than it thinks: each tile pixel spans 1.02 sample pixels.
            u = (np.arange(TILE) - (TILE - 1) / 2) * stretch
            rows, cols = np.meshgrid(centre[0] + u, centre[1] + u, indexing="ij")
            tile = ndimage.map_coordinates(image, [rows, cols], order=3)
            tiles.append(Tile(tile, ((centre[1] + 0.5) * PIXEL, (centre[0] + 0.5) * PIXEL), PIXEL, len(tiles)))
    placement = register_tiles(tiles, stage_sigma_m=3 * PIXEL)
    assert np.allclose(np.diag(placement.linear), 1 / stretch * np.ones(2), atol=2e-3)  # applied, and measured right
    levels, placement = stitch_to_arrays(tiles, placement)
    centres = placement.centres_px(np.full((16, 2), TILE))
    spacing = np.diff(centres.reshape(4, 4, 2)[:, :, 1], axis=1)
    assert np.allclose(spacing, 200.0, atol=0.3)  # the stage's 200 px, not the images' 196


def test_gain_and_offset_drift_between_tiles_is_matched():
    tiles, _truth = make_tiles(stage_error_px=0.0, overlap=0.3)
    rng = np.random.default_rng(9)
    gains = 1.0 + rng.uniform(-0.1, 0.1, len(tiles))
    offsets = rng.uniform(-5, 5, len(tiles))
    drifted = [Tile(t.image * g + o, t.stage_xy_m, t.pixel_size_m, t.index) for t, g, o in zip(tiles, gains, offsets)]
    placement = register_tiles(drifted, stage_sigma_m=2 * PIXEL)
    # What each tile was multiplied by is undone up to one shared factor and shift (the map keeps a scale).
    recovered = placement.gains * gains
    assert np.std(recovered / recovered.mean()) < 0.003
    levels, _ = stitch_to_arrays(drifted, placement)
    plain, _ = stitch_to_arrays(drifted, placement.__class__(**{**placement.__dict__, "gains": None, "offsets": None}))

    def step(canvas):  # mean jump across the vertical tile edges, where neighbours meet
        edges = sorted({int(round(p[1])) + TILE for p in placement.positions_px if int(round(p[1])) + TILE < canvas.shape[1] - 1})
        return np.mean([np.abs(canvas[40:-40, e] - canvas[40:-40, e - 1]).mean() for e in edges])

    assert step(levels[0].astype(float)) < step(plain[0].astype(float))


def test_a_pair_with_too_small_an_overlap_says_nothing_about_gain():
    tiles, _ = make_tiles(stage_error_px=0.0, grid=2)
    gains, offsets = solve_gain_offset(tiles, np.array([[0.0, 0.0], [0.0, 250.0], [250.0, 0.0], [250.0, 250.0]]), [mosaic.PairMatch(0, 1, np.zeros(2), 1.0, True)])
    assert np.allclose(gains, 1.0) and np.allclose(offsets, 0.0)


def test_a_detector_drifting_in_time_is_followed_smoothly_and_not_invented_where_it_did_not_drift():
    tiles, _ = make_tiles(stage_error_px=0.0, overlap=0.3, grid=6, image=scene(size=1300, blobs=1500))
    times = np.arange(len(tiles)) * 60.0  # one tile a minute: half an hour of acquisition
    true_gain = 1.0 + 0.08 * np.sin(times / 1800.0 * np.pi)  # a slow swing of up to 8 %
    drifted = [Tile(t.image * g, t.stage_xy_m, t.pixel_size_m, t.index, acquired_s=float(s)) for t, g, s in zip(tiles, true_gain, times)]
    placement = register_tiles(drifted, stage_sigma_m=2 * PIXEL)
    corrected = placement.gains * true_gain
    assert np.std(corrected / corrected.mean()) < 0.01  # the drift is undone to within 1 %
    steady = [Tile(t.image, t.stage_xy_m, t.pixel_size_m, t.index, acquired_s=float(s)) for t, s in zip(tiles, times)]
    placement = register_tiles(steady, stage_sigma_m=2 * PIXEL)
    assert np.all(np.abs(placement.gains - 1.0) < 0.01)  # nothing invented on a steady detector


def test_a_calibrated_scan_is_undone_on_a_map_too_small_to_measure_it():
    image = scene(size=1400, blobs=900)
    stretch = 1.02
    rng = np.random.default_rng(4)
    tiles = []
    for r in range(3):
        for c in range(3):
            centre = np.array([400 + r * 200, 400 + c * 200], dtype=float)
            u = (np.arange(TILE) - (TILE - 1) / 2) * stretch
            rows, cols = np.meshgrid(centre[0] + u, centre[1] + u, indexing="ij")
            reported = centre + rng.normal(0, 15.0, 2)  # a stage too poor to see a 2 % stretch over 9 tiles
            tiles.append(Tile(ndimage.map_coordinates(image, [rows, cols], order=3), ((reported[1] + 0.5) * PIXEL, (reported[0] + 0.5) * PIXEL), PIXEL, len(tiles)))
    own = register_tiles(tiles, stage_sigma_m=45 * PIXEL)
    assert np.allclose(own.linear, np.eye(2))  # it cannot tell, so it changes nothing
    calibrated = register_tiles(tiles, stage_sigma_m=45 * PIXEL, scan_linear=np.eye(2) / stretch)
    spacing = np.diff(calibrated.centres_px(np.full((9, 2), TILE)).reshape(3, 3, 2)[:, :, 1], axis=1)
    assert np.allclose(spacing, 200.0, atol=0.5)  # the sample's 200 px, from the calibration


def _datasets(stage_error_px, grid=4, overlap=0.2, seed=1):
    """Images of a scene as sidpy datasets that record their stage positions; returns them and the true top-left."""
    tiles, truth = make_tiles(stage_error_px=stage_error_px, grid=grid, overlap=overlap, seed=seed)
    return [image_dataset(t.image, PIXEL, t.stage_xy_m, title=f"tile {k}") for k, t in enumerate(tiles)], truth


def test_a_tile_is_made_from_a_dataset_with_its_pixel_size_and_stage_position():
    dataset = image_dataset(np.ones((32, 48)), 0.5e-9, (1e-6, -2e-6), acquired_s=12.0)
    tile = Tile.from_dataset(dataset, index=3)
    assert tile.pixel_size_m == pytest.approx(0.5e-9) and tile.stage_xy_m == pytest.approx((1e-6, -2e-6))
    assert tile.shape == (32, 48) and tile.index == 3 and tile.acquired_s == 12.0
    dataset.metadata["experiment"]["image_shift"] = [3e-8, 0.0]
    assert Tile.from_dataset(dataset).stage_xy_m == pytest.approx((1e-6 + 3e-8, -2e-6))
    with pytest.raises(ValueError, match="stage_position"):
        Tile.from_dataset(image_dataset(np.ones((8, 8))))
    with pytest.raises(TypeError):
        Tile.from_dataset(np.ones((8, 8)))


def test_stitch_mosaic_returns_a_calibrated_map_whose_pixels_sit_on_the_stage():
    datasets, truth = _datasets(stage_error_px=4.0)
    mosaic_dataset = mosaic.stitch_mosaic(datasets, stage_sigma_m=4 * PIXEL)
    assert mosaic_dataset.data_type.name == "IMAGE"
    assert get_pixel_size_m(mosaic_dataset) == pytest.approx([PIXEL, PIXEL])
    placed = mosaic_dataset.metadata["mosaic"]
    assert placed["matches_used"] > 0 and placed["sources"] == [d.title for d in datasets]
    assert np.median(placement_error(np.array(placed["positions_px"]), truth)) < 1.0
    # The map's own stage position puts each of its pixels where the placement says it is.
    for row, col in ((0, 0), (100.5, 37.25), (mosaic_dataset.shape[0] - 1, 3)):
        expected = (placed["origin_xy_m"][0] + col * PIXEL, placed["origin_xy_m"][1] + row * PIXEL)
        assert pixel_to_stage(mosaic_dataset, row, col) == pytest.approx(expected, abs=1e-15)
    first = np.round(placed["positions_px"][0]).astype(int)
    drawn = np.asarray(mosaic_dataset)[first[0] + 40 : first[0] + 80, first[1] + 40 : first[1] + 80]
    assert np.corrcoef(drawn.ravel(), np.asarray(datasets[0])[40:80, 40:80].ravel())[0, 1] > 0.98
