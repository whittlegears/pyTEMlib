"""
Stitch tiles by their image features, for tiles with no stage positions.

``ImageStitcher`` matches overlapping tiles with ORB descriptors, optionally
augmented with atom positions from a ``Segmenter``, fits a homography per
pair with RANSAC and blends the tiles onto one canvas. It came from
STEM-ML-Functions' ``stem_stitcher``.

Maps acquired with stage positions are stitched by ``image_tools.image_mosaic``
(``register_tiles``, ``assemble``), which also writes their pyramid levels.

Needs OpenCV (``pip install pyTEMlib[stitching]``); ``AtomAISegmenter`` also
needs ``atomai`` and a trained model.
"""

from .segmentation import (
    ATOMAI_AVAILABLE,
    SKIMAGE_AVAILABLE,
    AtomAISegmenter,
    BlobSegmenter,
    Segmenter,
    SegmentationResult,
)
from .stitcher import ImageEntry, ImageStitcher

__all__ = [
    "ImageStitcher",
    "ImageEntry",
    "Segmenter",
    "BlobSegmenter",
    "AtomAISegmenter",
    "SegmentationResult",
    "ATOMAI_AVAILABLE",
    "SKIMAGE_AVAILABLE",
]
