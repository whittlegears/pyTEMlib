"""Tests for pyTEMlib.image_tools.stitching."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from pyTEMlib.image_tools.stitching import ImageEntry, ImageStitcher

def _checkerboard(h: int = 128, w: int = 128, sq: int = 16) -> np.ndarray:
    """Generate a greyscale checkerboard image (uint8)."""
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, sq * 2):
        for c in range(0, w, sq * 2):
            img[r : r + sq, c : c + sq] = 200
            img[r + sq : r + 2 * sq, c + sq : c + 2 * sq] = 200
    return img


def _make_tiles(n: int = 3, overlap: int = 32, sq: int = 16) -> list[np.ndarray]:
    """Split a checkerboard into *n* overlapping horizontal tiles."""
    full_w = 256
    tile_w = 128
    full = _checkerboard(128, full_w + (n - 1) * (tile_w - overlap), sq)
    tiles = []
    for i in range(n):
        x0 = i * (tile_w - overlap)
        tiles.append(full[:, x0 : x0 + tile_w].copy())
    return tiles

class TestImageEntry:
    def test_shape(self):
        img = np.zeros((64, 128), dtype=np.uint8)
        entry = ImageEntry(image=img)
        assert entry.shape == (64, 128)

    def test_default_fields(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        entry = ImageEntry(image=img)
        assert entry.transform is None
        assert entry.segmentation is None
        assert entry.position_hint is None

class TestImageStitcher:
    def test_add_image_returns_index(self):
        stitcher = ImageStitcher()
        img = np.zeros((64, 64), dtype=np.uint8)
        idx0 = stitcher.add_image(img)
        idx1 = stitcher.add_image(img)
        assert idx0 == 0
        assert idx1 == 1
        assert stitcher.tile_count == 2

    def test_clear_resets_state(self):
        stitcher = ImageStitcher()
        stitcher.add_image(np.zeros((32, 32), dtype=np.uint8))
        stitcher.clear()
        assert stitcher.tile_count == 0
        assert stitcher.get_canvas() is None

    def test_stitch_raises_without_images(self):
        stitcher = ImageStitcher()
        with pytest.raises(ValueError, match="No images"):
            stitcher.stitch()

    def test_stitch_single_image(self):
        img = _checkerboard(64, 64)
        stitcher = ImageStitcher()
        stitcher.add_image(img)
        canvas = stitcher.stitch()
        assert canvas is not None
        assert canvas.ndim in (2, 3)
        assert canvas.dtype == np.uint8

    def test_stitch_two_tiles_returns_canvas(self):
        tiles = _make_tiles(2, overlap=32)
        stitcher = ImageStitcher(min_match_count=4)
        for t in tiles:
            stitcher.add_image(t)
        canvas = stitcher.stitch()
        assert canvas is not None
        h, w = canvas.shape[:2]
        assert w >= tiles[0].shape[1]

    def test_stitch_with_position_hints(self):
        tile_w = 128
        tile_h = 128
        img1 = _checkerboard(tile_h, tile_w)
        img2 = _checkerboard(tile_h, tile_w)
        stitcher = ImageStitcher(min_match_count=4)
        stitcher.add_image(img1, position_hint=(0, 0))
        stitcher.add_image(img2, position_hint=(0, 96))
        canvas = stitcher.stitch()
        assert canvas is not None

    def test_stitch_colour_images(self):
        grey = _checkerboard(64, 64)
        colour = np.stack([grey, grey // 2, grey // 4], axis=2)
        stitcher = ImageStitcher(min_match_count=4)
        stitcher.add_image(colour)
        canvas = stitcher.stitch()
        assert canvas.ndim == 3

    def test_blend_modes(self):
        img = _checkerboard(64, 64)
        for mode in ("average", "max"):
            stitcher = ImageStitcher(blend_mode=mode, min_match_count=4)
            stitcher.add_image(img)
            canvas = stitcher.stitch()
            assert canvas is not None

    def test_progress_callback_called(self):
        calls: list[tuple] = []

        def cb(step, total, msg):
            calls.append((step, total, msg))

        img = _checkerboard(64, 64)
        stitcher = ImageStitcher()
        stitcher.add_image(img)
        stitcher.stitch(progress_callback=cb)
        assert len(calls) > 0

    def test_get_canvas_before_stitch(self):
        stitcher = ImageStitcher()
        assert stitcher.get_canvas() is None

    def test_get_canvas_after_stitch(self):
        img = _checkerboard(64, 64)
        stitcher = ImageStitcher()
        stitcher.add_image(img)
        stitcher.stitch()
        canvas = stitcher.get_canvas()
        assert canvas is not None
        assert canvas.dtype == np.uint8

    def test_unplaced_tiles_empty_before_stitch(self):
        stitcher = ImageStitcher()
        assert stitcher.unplaced_tiles == []

    def test_unmatchable_tile_without_hint_is_reported_not_placed(self):
        tile = _checkerboard(64, 64)
        blank = np.zeros((64, 64), dtype=np.uint8)
        stitcher = ImageStitcher(min_match_count=4)
        stitcher.add_image(tile)
        stitcher.add_image(blank)
        canvas = stitcher.stitch()
        assert stitcher.unplaced_tiles == [1]
        assert canvas.shape[:2] == tile.shape

    def test_unmatchable_tile_with_hints_is_placed_from_hints(self):
        tile = _checkerboard(64, 64)
        blank = np.zeros((64, 64), dtype=np.uint8)
        stitcher = ImageStitcher(min_match_count=4)
        stitcher.add_image(tile, position_hint=(0, 0))
        stitcher.add_image(blank, position_hint=(0, 48))
        canvas = stitcher.stitch()
        assert stitcher.unplaced_tiles == []
        assert canvas.shape[:2] == (64, 112)

    def test_later_tile_registers_against_last_placed_tile(self):
        tiles = _make_tiles(2, overlap=32)
        blank = np.zeros_like(tiles[0])
        stitcher = ImageStitcher(min_match_count=4)
        stitcher.add_image(tiles[0])
        stitcher.add_image(blank)
        stitcher.add_image(tiles[1])
        stitcher.stitch()
        assert 1 in stitcher.unplaced_tiles

    def test_clear_resets_unplaced(self):
        stitcher = ImageStitcher(min_match_count=4)
        stitcher.add_image(_checkerboard(64, 64))
        stitcher.add_image(np.zeros((64, 64), dtype=np.uint8))
        stitcher.stitch()
        stitcher.clear()
        assert stitcher.unplaced_tiles == []
