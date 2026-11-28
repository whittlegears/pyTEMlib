"""
Atom segmentation for STEM images.

Provides ML-based and classical methods to detect atomic positions in STEM images.
AtomAI (https://github.com/pycroscopy/atomai) is used for deep-learning segmentation
when explicitly requested with a trained model; scikit-image blob detection is the
default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import atomai as aoi 

    _ATOMAI_AVAILABLE = True
    logger.info("AtomAI is available – ML segmentation requires a trained model.")
except ImportError:
    _ATOMAI_AVAILABLE = False
except Exception as _atomai_exc: 
    _ATOMAI_AVAILABLE = False
    logger.debug("AtomAI import raised an unexpected error: %s", _atomai_exc)

try:
    from skimage.feature import blob_dog, blob_log 
    from skimage.filters import gaussian 

    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
except Exception as _skimage_exc:  
    _SKIMAGE_AVAILABLE = False
    logger.debug("scikit-image import raised an unexpected error: %s", _skimage_exc)

ATOMAI_AVAILABLE: bool = _ATOMAI_AVAILABLE
SKIMAGE_AVAILABLE: bool = _SKIMAGE_AVAILABLE

@dataclass
class SegmentationResult:
    """Result returned by any segmenter.

    Attributes
    ----------
    atom_positions:
        ``(N, 2)`` float array of detected atom positions as ``(row, col)``
        (i.e., ``(y, x)`` in image coordinates).
    segmentation_map:
        Optional ``(H, W)`` probability / class map produced by an ML model.
    confidence:
        Optional per-atom confidence scores with shape ``(N,)``.
    method:
        Name of the segmentation method that produced this result.
    """

    atom_positions: np.ndarray
    segmentation_map: Optional[np.ndarray] = field(default=None)
    confidence: Optional[np.ndarray] = field(default=None)
    method: str = "unknown"

class BlobSegmenter:
    """Atom detector using Laplacian-of-Gaussian blob detection.

    This is the always-available fallback segmenter that works without any
    deep-learning framework.  It tries both bright-field and dark-field
    polarity automatically.

    Parameters
    ----------
    min_sigma:
        Minimum sigma (px) for the Gaussian kernel.
    max_sigma:
        Maximum sigma (px) for the Gaussian kernel.
    num_sigma:
        Number of intermediate sigma values to test.
    threshold:
        Detection threshold relative to the maximum blob response.
    overlap:
        Maximum allowed overlap between blobs (0–1).
    """

    def __init__(
        self,
        min_sigma: float = 1.0,
        max_sigma: float = 8.0,
        num_sigma: int = 10,
        threshold: float = 0.05,
        overlap: float = 0.5,
    ) -> None:
        if not _SKIMAGE_AVAILABLE:
            raise ImportError(
                "scikit-image is required for BlobSegmenter.  "
                "Install it with: pip install scikit-image"
            )
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=float)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        return img

    def detect(self, image: np.ndarray) -> SegmentationResult:
        """Detect atoms in *image* and return a :class:`SegmentationResult`.

        Parameters
        ----------
        image:
            2-D greyscale image (float or integer dtype).  If a colour image
            is passed, it will be converted to greyscale by averaging channels.

        Returns
        -------
        SegmentationResult
        """
        if image.ndim == 3:
            image = image.mean(axis=2)

        img = self._normalize(image)

        common = dict(
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap,
        )

        blobs_bf = blob_log(img, **common)
        blobs_df = blob_log(1.0 - img, **common)
        blobs = blobs_bf if len(blobs_bf) >= len(blobs_df) else blobs_df

        if len(blobs) == 0:
            positions = np.empty((0, 2), dtype=float)
        else:
            positions = blobs[:, :2].copy()

        return SegmentationResult(
            atom_positions=positions,
            method="blob_log",
        )

class AtomAISegmenter:
    """Atom detector backed by an AtomAI deep-learning model.

    Requires ``atomai`` to be installed::

        pip install atomai

    Parameters
    ----------
    model:
        Either a path to a trained AtomAI model file or an
        already-instantiated, trained ``atomai`` model object.  Required: an
        untrained network produces meaningless atom positions.
    """

    def __init__(self, model: "str | object | None" = None) -> None:
        if not _ATOMAI_AVAILABLE:
            raise ImportError(
                "atomai is required for AtomAISegmenter.  "
                "Install it with: pip install atomai"
            )
        if model is None:
            raise ValueError(
                "AtomAISegmenter needs a trained model: pass a model file path "
                "or a trained atomai model object."
            )
        self._model_arg = model
        self._model: object | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if isinstance(self._model_arg, str):
            try:
                self._model = aoi.models.load_model(self._model_arg)
            except Exception as exc: 
                raise RuntimeError(
                    f"Could not load AtomAI model '{self._model_arg}': {exc}"
                ) from exc
        else:
            self._model = self._model_arg

    def detect(self, image: np.ndarray) -> SegmentationResult:
        """Detect atoms via AtomAI model inference.

        Parameters
        ----------
        image:
            2-D greyscale STEM image.

        Returns
        -------
        SegmentationResult
        """
        self._load()

        img = np.asarray(image, dtype=float)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)

        nn_output, _ = self._model.predict(img, refine=False) 
        try:
            from atomai.utils import find_com 

            positions = find_com(nn_output)
        except Exception:  
            mask = nn_output > 0.5
            from scipy import ndimage 

            labelled, n = ndimage.label(mask)
            if n:
                positions = np.array(
                    ndimage.center_of_mass(mask, labelled, range(1, n + 1))
                )
            else:
                positions = np.empty((0, 2), dtype=float)

        return SegmentationResult(
            atom_positions=positions,
            segmentation_map=nn_output,
            method="atomai",
        )

class Segmenter:
    """Auto-selecting segmenter that picks the best available backend.

    With *use_ml* ``False`` (default) scikit-image blob detection is used.
    With *use_ml* ``True`` the AtomAI segmenter is used with the trained
    model given in *atomai_model*; there is no silent fallback to blob
    detection.

    Parameters
    ----------
    use_ml:
        When ``True``, use AtomAI with *atomai_model*.  Requires ``atomai``
        and a trained model.
    atomai_model:
        Trained model path or object forwarded to :class:`AtomAISegmenter`.
        Required when *use_ml* is ``True``.
    blob_kwargs:
        Keyword arguments forwarded to :class:`BlobSegmenter`.
    """

    def __init__(
        self,
        use_ml: bool = False,
        atomai_model: "str | object | None" = None,
        **blob_kwargs: object,
    ) -> None:
        self._backend: BlobSegmenter | AtomAISegmenter

        if use_ml:
            if atomai_model is None:
                raise ValueError(
                    "use_ml=True requires atomai_model: a path to a trained "
                    "AtomAI model or a trained model object.  Use "
                    "use_ml=False for blob detection."
                )
            self._backend = AtomAISegmenter(model=atomai_model)
            logger.info("Segmenter: using AtomAI backend.")
        elif _SKIMAGE_AVAILABLE:
            self._backend = BlobSegmenter(**blob_kwargs)
            logger.info("Segmenter: using BlobSegmenter backend.")
        else:
            raise RuntimeError(
                "No segmentation backend available.  "
                "Install scikit-image (pip install scikit-image) or "
                "atomai (pip install atomai)."
            )

    @property
    def backend_name(self) -> str:
        """Name of the active segmentation backend."""
        return type(self._backend).__name__

    def detect(self, image: np.ndarray) -> SegmentationResult:
        """Run atom detection on *image*.

        Parameters
        ----------
        image:
            2-D greyscale (or colour) STEM image.

        Returns
        -------
        SegmentationResult
        """
        return self._backend.detect(image)
