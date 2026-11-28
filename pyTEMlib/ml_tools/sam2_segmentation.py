"""
SAM2 automatic segmentation of microscope images (optional)

Needs torch and SAM2:
``pip install "git+https://github.com/facebookresearch/sam2.git" huggingface-hub``.
Moved from asyncroscopy's segmentation Tango device (asyncroscopy/mcp/segment.py),
with the same image preparation, mask filtering, labelling and area statistics, but
returning arrays instead of talking to Tango.

The model is loaded once per (model, device) and reused.

part of pyTEMlib
"""
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_hf
from scipy import ndimage

_models = {}


def prepare_image(image):
    """Normalise a 2D (or RGB) image to 8-bit and expand it to the RGB input SAM2 expects."""
    gray = np.squeeze(np.asarray(image)).astype(np.float32)
    if gray.ndim == 3 and gray.shape[-1] in {3, 4}:
        gray = np.mean(gray[..., :3], axis=-1)
    if gray.ndim != 2:
        raise ValueError(f'segmentation needs a 2D grayscale or RGB image; received shape {gray.shape}')
    finite = np.isfinite(gray)
    if not np.any(finite):
        raise ValueError('segmentation input contains no finite values')
    low, high = np.percentile(gray[finite], [0.5, 99.5])
    normalized = np.nan_to_num(np.clip((gray - low) / (high - low + 1e-8), 0, 1), nan=0.0, posinf=1.0, neginf=0.0)
    return np.repeat((normalized * 255).astype(np.uint8)[..., np.newaxis], 3, axis=-1)


def label_areas(masks, shape):
    """Combine SAM2 masks into one integer label image (later, smaller masks win overlaps)."""
    labels = np.zeros(shape, dtype=np.int32)
    for label, mask in enumerate(masks, start=1):
        segmentation = np.asarray(mask['segmentation'], dtype=bool)
        if segmentation.shape != tuple(shape):
            raise ValueError(f'SAM2 returned mask shape {segmentation.shape}; expected {tuple(shape)}')
        labels[segmentation] = label
    return labels


def area_statistics(masks):
    """Area, equivalent diameter, circularity, SAM2 confidence and centroid of each mask."""
    statistics = []
    for area_id, mask in enumerate(masks, start=1):
        segmentation = np.asarray(mask['segmentation'], dtype=bool)
        area_px = int(mask['area'])
        cy, cx = ndimage.center_of_mass(segmentation)
        perimeter = int(np.sum(segmentation & ~ndimage.binary_erosion(segmentation)))
        circularity = min(float(4 * np.pi * area_px / perimeter ** 2), 1.0) if perimeter > 0 else 0.0
        statistics.append({'id': area_id, 'area_px': area_px,
                           'equiv_diameter_px': float(2 * np.sqrt(area_px / np.pi)),
                           'circularity': circularity, 'confidence': float(mask['predicted_iou']),
                           'centroid_xy_px': [float(cx), float(cy)]})
    return statistics


def resolve_device(compute_device='auto'):
    """'auto' picks CUDA when available; otherwise 'cpu', 'cuda' or 'cuda:<index>'."""
    requested = compute_device.strip().lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError(f"unsupported compute_device {compute_device!r}; use 'auto', 'cpu', 'cuda' or 'cuda:<index>'")
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'compute_device is {compute_device!r}, but CUDA is unavailable in PyTorch {torch.__version__}')
    return str(device)


def load_model(model_id='facebook/sam2-hiera-large', compute_device='auto'):
    """Load (once) and return the SAM2 model and the device it runs on."""
    device = resolve_device(compute_device)
    key = (model_id, device)
    if key not in _models:
        _models[key] = build_sam2_hf(model_id, device=device)
    return _models[key], device


def segment(image, model_id='facebook/sam2-hiera-large', compute_device='auto', points_per_side=48,
            iou_threshold=0.57, stability_threshold=0.75, min_area_px=200, crop_n_layers=1):
    """Segment an image with SAM2's automatic mask generator.

    Masks smaller than ``min_area_px`` or covering half the image or more (background)
    are dropped. Defaults are those of the asyncroscopy segmentation device.

    Returns
    -------
    labels: 2D int32 array, 0 is unlabelled
    statistics: list of dict, one per label (see :func:`area_statistics`)
    """
    prepared = prepare_image(image)
    model, device = load_model(model_id, compute_device)
    generator = SAM2AutomaticMaskGenerator(model=model, points_per_side=points_per_side,
                                           pred_iou_thresh=iou_threshold,
                                           stability_score_thresh=stability_threshold,
                                           min_mask_region_area=min_area_px, crop_n_layers=crop_n_layers)
    with torch.inference_mode():
        if device.startswith('cuda'):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                masks = generator.generate(prepared)
        else:
            masks = generator.generate(prepared)
    masks.sort(key=lambda mask: mask['area'], reverse=True)
    total = prepared.shape[0] * prepared.shape[1]
    masks = [mask for mask in masks if min_area_px < mask['area'] < total * 0.5]
    return label_areas(masks, prepared.shape[:2]), area_statistics(masks)
