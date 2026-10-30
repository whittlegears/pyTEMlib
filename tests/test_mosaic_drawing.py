"""What the drawn map gets right beyond tile positions: sub-pixel placement, the scan's shared distortion,
detector gain drift, and unplaced tiles left out (docs/designs/multiscale_maps.md in asyncroscopy, P2)."""

import numpy as np
import pytest
from scipy import ndimage

from pyTEMlib.image_tools import mosaic
from pyTEMlib.image_tools.mosaic import Tile, estimate_scan_linear, register_tiles, solve_gain_offset, stitch_to_arrays

from .test_mosaic import PIXEL, TILE, make_tiles, scene


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
