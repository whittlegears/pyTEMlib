"""Tests for pyTEMlib.image_tools.stitching.segmentation."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from pyTEMlib.image_tools.stitching.segmentation import (
    BlobSegmenter,
    Segmenter,
    SegmentationResult,
    SKIMAGE_AVAILABLE,
)

def _make_atom_image(h: int = 128, w: int = 128, n_atoms: int = 16) -> np.ndarray:
    """Synthetic greyscale image with Gaussian atom peaks."""
    rng = np.random.default_rng(42)
    img = np.zeros((h, w), dtype=np.float32)

    ys = rng.integers(10, h - 10, size=n_atoms)
    xs = rng.integers(10, w - 10, size=n_atoms)
    for y, x in zip(ys, xs):
        yy, xx = np.ogrid[:h, :w]
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2**2))

    img = (img / img.max() * 255).astype(np.uint8)
    return img

class TestSegmentationResult:
    def test_fields(self):
        pos = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = SegmentationResult(atom_positions=pos, method="test")
        assert result.atom_positions.shape == (2, 2)
        assert result.method == "test"
        assert result.segmentation_map is None
        assert result.confidence is None

    def test_optional_fields(self):
        pos = np.empty((0, 2))
        seg_map = np.zeros((64, 64))
        conf = np.array([0.9, 0.8])
        result = SegmentationResult(
            atom_positions=pos,
            segmentation_map=seg_map,
            confidence=conf,
        )
        assert result.segmentation_map is not None
        assert result.confidence is not None

@pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="scikit-image not installed")
class TestBlobSegmenter:
    def test_detects_atoms_in_synthetic_image(self):
        img = _make_atom_image(n_atoms=9)
        seg = BlobSegmenter(min_sigma=1.0, max_sigma=5.0, threshold=0.1)
        result = seg.detect(img)
        assert isinstance(result, SegmentationResult)
        assert len(result.atom_positions) > 0
        assert result.method == "blob_log"

    def test_returns_empty_on_blank_image(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        seg = BlobSegmenter(threshold=0.5)
        result = seg.detect(img)
        assert isinstance(result, SegmentationResult)
        assert result.atom_positions.shape[1] == 2

    def test_accepts_colour_image(self):
        grey = _make_atom_image(n_atoms=4)
        colour = np.stack([grey, grey, grey], axis=2)
        seg = BlobSegmenter()
        result = seg.detect(colour)
        assert result.atom_positions.ndim == 2

    def test_positions_within_image_bounds(self):
        h, w = 128, 128
        img = _make_atom_image(h=h, w=w, n_atoms=16)
        seg = BlobSegmenter(min_sigma=1.0, max_sigma=5.0, threshold=0.05)
        result = seg.detect(img)
        if len(result.atom_positions) > 0:
            rows = result.atom_positions[:, 0]
            cols = result.atom_positions[:, 1]
            assert (rows >= 0).all() and (rows < h).all()
            assert (cols >= 0).all() and (cols < w).all()

@pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="scikit-image not installed")
class TestSegmenter:
    def test_creates_successfully(self):
        seg = Segmenter(use_ml=False)
        assert "Blob" in seg.backend_name

    def test_detect_returns_result(self):
        img = _make_atom_image(n_atoms=9)
        seg = Segmenter(use_ml=False)
        result = seg.detect(img)
        assert isinstance(result, SegmentationResult)

    def test_blob_kwargs_forwarded(self):
        seg = Segmenter(use_ml=False, threshold=0.5)
        img = _make_atom_image(n_atoms=9)
        result = seg.detect(img)
        assert isinstance(result, SegmentationResult)

    def test_default_is_blob_backend(self):
        seg = Segmenter()
        assert "Blob" in seg.backend_name


class TestSegmenterMLRequiresModel:
    def test_use_ml_without_model_raises(self):
        with pytest.raises(ValueError, match="atomai_model"):
            Segmenter(use_ml=True)
