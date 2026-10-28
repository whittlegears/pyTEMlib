"""Stitching accuracy against known tile positions."""

import tracemalloc

import numpy as np
import pytest
from scipy import ndimage

from pyTEMlib.image_tools.mosaic import Tile, assemble, register_tiles, stitch_to_arrays

PIXEL = 1e-9
TILE = 256


def scene(size=1300, blobs=500, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size))
    rows, cols = rng.integers(0, size, blobs), rng.integers(0, size, blobs)
    image[rows, cols] = rng.uniform(50, 200, blobs)
    image = ndimage.gaussian_filter(image, 4) * 30 + 20
    return image + rng.normal(0, 1.0, image.shape)


def make_tiles(stage_error_px, overlap=0.2, grid=4, seed=1, reported_noise_px=0.0, image=None):
    """Tiles cut where the stage really was; positions reported where it was told to go."""
    rng = np.random.default_rng(seed)
    image = scene() if image is None else image
    step = int(TILE * (1 - overlap))
    tiles, true_top_left = [], []
    for r in range(grid):
        for c in range(grid):
            commanded = np.array([40 + r * step, 40 + c * step], dtype=float)
            true = commanded + rng.normal(0, stage_error_px, 2)
            ti = np.clip(np.round(true).astype(int), 0, image.shape[0] - TILE)
            tiles.append(image[ti[0] : ti[0] + TILE, ti[1] : ti[1] + TILE].copy())
            true_top_left.append(ti.astype(float))
            reported = commanded + rng.normal(0, reported_noise_px, 2) if reported_noise_px else commanded
            centre_xy = ((reported[1] + TILE / 2) * PIXEL, (reported[0] + TILE / 2) * PIXEL)
            tiles[-1] = Tile(tiles[-1], centre_xy, PIXEL, index=len(tiles) - 1)
    return tiles, np.array(true_top_left)


def placement_error(positions, truth):
    """Per-tile error after removing the unobservable global offset."""
    diff = positions - truth
    diff -= np.median(diff, axis=0)
    return np.hypot(diff[:, 0], diff[:, 1])


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


def test_assembly_memory_is_bounded_by_band_size():
    rng = np.random.default_rng(5)
    tiles = [Tile(rng.integers(0, 1000, (512, 512), dtype=np.uint16), ((c * 480 + 256) * PIXEL, (r * 480 + 256) * PIXEL), PIXEL, r * 8 + c) for r in range(8) for c in range(8)]
    placement = register_tiles(tiles, refine=False)
    canvas_bytes = placement.canvas_shape[0] * placement.canvas_shape[1] * 2
    written = []
    tracemalloc.start()
    assemble(tiles, placement, lambda level, row0, band: written.append(band.nbytes), band_rows=256)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sum(written) >= canvas_bytes
    assert peak < 0.5 * canvas_bytes * 4  # float64 working copy of the full canvas would be 4x canvas_bytes


def test_mixed_pixel_sizes_rejected():
    tiles, _ = make_tiles(stage_error_px=0.0, grid=2)
    tiles[1] = Tile(tiles[1].image, tiles[1].stage_xy_m, 2 * PIXEL)
    with pytest.raises(ValueError, match="pixel size"):
        register_tiles(tiles)
