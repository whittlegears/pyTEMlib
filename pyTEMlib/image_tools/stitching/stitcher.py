"""
Image stitching engine for STEM tile images.

Workflow
--------
1. The user calls :meth:`ImageStitcher.add_image` for every tile that should be
   included in the mosaic.
2. :meth:`ImageStitcher.stitch` detects atoms (via the configured
   :class:`~pyTEMlib.image_tools.stitching.segmentation.Segmenter`), matches overlapping tiles
   using both classical feature descriptors (ORB) and atom-position proximity,
   computes affine/homography transforms via RANSAC, and blends the tiles onto
   a shared canvas.
3. :meth:`ImageStitcher.get_canvas` returns the stitched ``numpy`` array.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("pyTEMlib.image_tools.stitching needs OpenCV: pip install pyTEMlib[stitching]") from exc

from .segmentation import Segmenter, SegmentationResult

logger = logging.getLogger(__name__)

@dataclass
class ImageEntry:
    """One tile that participates in the mosaic.

    Attributes
    ----------
    image:
        Greyscale or colour tile image (uint8 or float32).
    position_hint:
        Optional ``(row, col)`` hint for where the top-left corner of this
        tile should land on the canvas (in pixels).  When provided, the
        stitcher uses this as the initial estimate for pairwise registration.
    transform:
        ``(3, 3)`` homography matrix mapping this tile to canvas coordinates.
        Set by the stitcher during :meth:`ImageStitcher.stitch`.
    segmentation:
        Atom segmentation result for this tile.  Set by the stitcher.
    """

    image: np.ndarray
    position_hint: Optional[Tuple[int, int]] = None
    transform: Optional[np.ndarray] = field(default=None)
    segmentation: Optional[SegmentationResult] = field(default=None)

    @property
    def shape(self) -> Tuple[int, int]:
        """``(height, width)`` of the tile."""
        h, w = self.image.shape[:2]
        return h, w

class ImageStitcher:
    """Stitches a collection of STEM tile images into a single mosaic.

    Parameters
    ----------
    segmenter:
        Atom segmenter used to find feature points.  A default
        :class:`~pyTEMlib.image_tools.stitching.segmentation.Segmenter` is created when
        ``None`` is passed.
    blend_mode:
        How overlapping regions are combined.  Supported values are
        ``"average"`` (linear blend) and ``"max"`` (maximum intensity).
    use_atom_features:
        When ``True``, atom positions from segmentation are added as extra
        feature points during pairwise matching, which improves accuracy on
        crystalline STEM images.
    min_match_count:
        Minimum number of good feature matches required to accept a pairwise
        registration.  Pairs below this threshold are rejected; the tile is
        then placed from its position hint if both it and its reference tile
        have one, otherwise it is left out of the mosaic and listed in
        :attr:`unplaced_tiles`.
    """

    def __init__(
        self,
        segmenter: Optional[Segmenter] = None,
        blend_mode: str = "average",
        use_atom_features: bool = True,
        min_match_count: int = 4,
    ) -> None:
        self._segmenter = segmenter or Segmenter()
        self._blend_mode = blend_mode
        self._use_atom_features = use_atom_features
        self._min_match_count = min_match_count
        self._entries: List[ImageEntry] = []
        self._canvas: Optional[np.ndarray] = None
        self._canvas_weight: Optional[np.ndarray] = None
        self._unplaced: List[int] = []

    def clear(self) -> None:
        """Remove all tiles and reset internal state."""
        self._entries.clear()
        self._canvas = None
        self._canvas_weight = None
        self._unplaced = []

    def add_image(
        self,
        image: np.ndarray,
        position_hint: Optional[Tuple[int, int]] = None,
    ) -> int:
        """Add a tile to the stitching queue.

        Parameters
        ----------
        image:
            2-D or 3-D numpy array representing the tile.
        position_hint:
            ``(row, col)`` pixel offset on the final canvas.  Providing hints
            for all tiles speeds up stitching significantly when the
            approximate tile layout is known in advance.

        Returns
        -------
        int
            Index of the newly added tile (0-based).
        """
        entry = ImageEntry(image=image, position_hint=position_hint)
        self._entries.append(entry)
        return len(self._entries) - 1

    def stitch(self, progress_callback=None) -> np.ndarray:
        """Run the stitching pipeline and return the mosaic canvas.

        Parameters
        ----------
        progress_callback:
            Optional callable ``f(step: int, total: int, message: str)``
            called periodically so the GUI can display a progress bar.

        Returns
        -------
        np.ndarray
            The stitched canvas (uint8, colour if any input was colour,
            greyscale otherwise).  Tiles that could not be registered are
            not drawn; their indices are in :attr:`unplaced_tiles`.

        Raises
        ------
        ValueError
            If fewer than one image has been added.
        """
        if not self._entries:
            raise ValueError("No images added.  Call add_image() first.")

        n = len(self._entries)

        def _report(step: int, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(step, n * 2 + 2, msg)

        _report(0, "Detecting atoms …")
        for i, entry in enumerate(self._entries):
            _report(i, f"Segmenting tile {i + 1}/{n} …")
            grey = self._to_grey(entry.image)
            entry.segmentation = self._segmenter.detect(grey)
            logger.debug(
                "Tile %d: %d atoms detected.", i, len(entry.segmentation.atom_positions)
            )

        self._unplaced = []
        for entry in self._entries:
            entry.transform = None
        self._entries[0].transform = np.eye(3, dtype=float)

        ref_idx = 0
        for i in range(1, n):
            _report(n + i, f"Registering tile {i + 1}/{n} …")
            ref = self._entries[ref_idx]
            H = self._register_pair(ref, self._entries[i], ref_idx, i)
            if H is None:
                self._unplaced.append(i)
                continue
            self._entries[i].transform = ref.transform @ H
            ref_idx = i

        if self._unplaced:
            logger.warning(
                "%d of %d tiles could not be registered and are not in the "
                "mosaic: %s",
                len(self._unplaced),
                n,
                self._unplaced,
            )

        _report(n * 2, "Rendering canvas …")
        canvas = self._render()
        self._unplaced.sort()

        _report(n * 2 + 1, "Done.")
        self._canvas = canvas
        return canvas

    def get_canvas(self) -> Optional[np.ndarray]:
        """Return the last stitched canvas, or ``None`` if :meth:`stitch` has
        not been called yet."""
        return self._canvas

    @property
    def tile_count(self) -> int:
        """Number of tiles currently in the queue."""
        return len(self._entries)

    @property
    def unplaced_tiles(self) -> List[int]:
        """Indices of tiles the last :meth:`stitch` could not register.

        These tiles are absent from the canvas.
        """
        return list(self._unplaced)

    @staticmethod
    def _to_grey(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        lo, hi = image.min(), image.max()
        if hi > lo:
            image = (image - lo) / (hi - lo) * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _is_valid_homography(H: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a homography matrix is numerically valid.
        
        Returns False if the matrix is singular, has NaN/Inf, or would
        cause invalid scaling during warpPerspective.
        """
        if H is None or H.shape != (3, 3):
            return False
        if not np.isfinite(H).all():
            return False
        det = np.linalg.det(H)
        if abs(det) < tolerance:
            return False
        return True

    def _extract_features(
        self, entry: ImageEntry
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extract ORB keypoints + descriptors, optionally augmented with
        atom positions from segmentation."""
        grey = self._to_uint8(self._to_grey(entry.image))
        orb = cv2.ORB_create(nfeatures=2000)
        raw_kps, descs = orb.detectAndCompute(grey, None)
        kps: List[cv2.KeyPoint] = list(raw_kps) if raw_kps is not None else []

        if (
            self._use_atom_features
            and entry.segmentation is not None
            and len(entry.segmentation.atom_positions) > 0
        ):
            atom_kps = [
                cv2.KeyPoint(float(c), float(r), 3.0)
                for r, c in entry.segmentation.atom_positions
            ]
            if descs is not None and len(atom_kps) > 0:
                _, atom_descs = orb.compute(grey, atom_kps)
                if atom_descs is not None and len(atom_descs) > 0:
                    kps = kps + atom_kps
                    descs = np.vstack([descs, atom_descs])

        return kps, descs

    def _register_pair(
        self,
        ref: ImageEntry,
        src: ImageEntry,
        ref_idx: int,
        src_idx: int,
    ) -> Optional[np.ndarray]:
        """Estimate the homography that maps *src* onto *ref*.

        Falls back to a pure translation from position hints when feature
        matching fails.  Returns ``None`` when matching fails and the pair
        has no position hints, so the tile is reported as unplaced.
        """
        kps_ref, descs_ref = self._extract_features(ref)
        kps_src, descs_src = self._extract_features(src)

        H_fallback = self._fallback_transform(ref, src)

        if (
            descs_ref is None
            or descs_src is None
            or len(descs_ref) < self._min_match_count
            or len(descs_src) < self._min_match_count
        ):
            logger.warning(
                "Tile pair (%d, %d): not enough descriptors – falling back to position hints.",
                ref_idx,
                src_idx,
            )
            return H_fallback

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(descs_src, descs_ref, k=2)
        good: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < self._min_match_count:
            logger.warning(
                "Tile pair (%d, %d): only %d good matches – falling back to position hints.",
                ref_idx,
                src_idx,
                len(good),
            )
            return H_fallback

        pts_src = np.float32(
            [kps_src[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        pts_ref = np.float32(
            [kps_ref[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)

        if H is None:
            logger.warning(
                "Tile pair (%d, %d): RANSAC failed – falling back to position hints.", ref_idx, src_idx
            )
            return H_fallback

        logger.debug(
            "Tile pair (%d, %d): %d/%d inliers.",
            ref_idx,
            src_idx,
            int(mask.sum()),
            len(good),
        )
        return H

    def _fallback_transform(
        self, ref: ImageEntry, src: ImageEntry
    ) -> Optional[np.ndarray]:
        """Pure translation from position hints, or ``None`` without them."""
        if src.position_hint is None or ref.position_hint is None:
            return None
        H = np.eye(3, dtype=float)
        dr = src.position_hint[0] - ref.position_hint[0]
        dc = src.position_hint[1] - ref.position_hint[1]
        H[0, 2] = float(dc)
        H[1, 2] = float(dr)
        return H

    def _render(self) -> np.ndarray:
        """Warp all placed tiles onto a shared canvas and blend them."""
        placed = [
            (i, e) for i, e in enumerate(self._entries) if e.transform is not None
        ]
        corners_all: list[np.ndarray] = []
        for _, entry in placed:
            h, w = entry.shape
            corners = np.float32(
                [[0, 0], [w, 0], [w, h], [0, h]]
            ).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, entry.transform)
            corners_all.append(warped.reshape(-1, 2))

        all_pts = np.vstack(corners_all)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)

        offset_x, offset_y = -float(x_min), -float(y_min)
        canvas_w = int(np.ceil(x_max - x_min))
        canvas_h = int(np.ceil(y_max - y_min))

        max_dim = 32768
        if canvas_w > max_dim or canvas_h > max_dim:
            scale = min(max_dim / canvas_w, max_dim / canvas_h)
            canvas_w = int(canvas_w * scale)
            canvas_h = int(canvas_h * scale)
            offset_x *= scale
            offset_y *= scale
            logger.warning("Canvas capped to %dx%d.", canvas_w, canvas_h)

        is_colour = any(e.image.ndim == 3 for e in self._entries)

        accumulator = np.zeros(
            (canvas_h, canvas_w, 3) if is_colour else (canvas_h, canvas_w),
            dtype=np.float64,
        )
        weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

        T = np.array(
            [[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=float
        )

        for idx, entry in placed:
            H = T @ entry.transform
            
            if not self._is_valid_homography(H):
                logger.warning("Skipping tile with invalid homography (det=%f)", np.linalg.det(H))
                self._unplaced.append(idx)
                continue
            
            tile = entry.image

            if is_colour and tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            elif not is_colour and tile.ndim == 3:
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            tile_f = tile.astype(np.float64)

            try:
                warped = cv2.warpPerspective(
                    tile_f,
                    H,
                    (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                mask_src = np.ones(entry.shape[:2], dtype=np.float64)
                warped_mask = cv2.warpPerspective(
                    mask_src,
                    H,
                    (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            except cv2.error as e:
                logger.warning("Failed to warp tile: %s. Skipping.", str(e))
                self._unplaced.append(idx)
                continue

            if self._blend_mode == "max":
                if is_colour:
                    for c in range(3):
                        accumulator[:, :, c] = np.maximum(
                            accumulator[:, :, c], warped[:, :, c]
                        )
                else:
                    accumulator = np.maximum(accumulator, warped)
                weight = np.maximum(weight, warped_mask)
            else:
                if is_colour:
                    for c in range(3):
                        accumulator[:, :, c] += warped[:, :, c] * warped_mask
                else:
                    accumulator += warped * warped_mask
                weight += warped_mask

        valid = weight > 0
        if self._blend_mode != "max":
            if is_colour:
                for c in range(3):
                    accumulator[:, :, c][valid] /= weight[valid]
            else:
                accumulator[valid] /= weight[valid]

        canvas = np.clip(accumulator, 0, 255).astype(np.uint8)
        return canvas
