"""
Tile maps: stitch images acquired at known stage positions, optionally with labels

A map lives in a directory (``manifest.json`` plus ``.npy`` arrays), so every step is a
separate call and nothing depends on a long-running process. Each tile is placed from
its stage position, optionally refined by cross-correlating its overlap with what is
already in the map, and averaged into the map image. A tile may bring a label image
(from any segmentation: thresholds, SAM2, ...); its regions are merged with the map's
regions where they overlap, so a particle cut by a tile edge becomes one region once
both halves are in. Segmenting is therefore optional and independent of stitching.

part of pyTEMlib
"""
import json
import os

import numpy as np
from skimage.feature import match_template

from .image_segmentation import particle_statistics

manifest_name = 'manifest.json'


def _paths(map_dir):
    return {name: os.path.join(map_dir, f'{name}.npy') for name in ('sum', 'count', 'labels')}


def _read_manifest(map_dir):
    path = os.path.join(map_dir, manifest_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{map_dir} is not a tile map (no {manifest_name}); create it first')
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _write_manifest(map_dir, manifest):
    with open(os.path.join(map_dir, manifest_name), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=1)


def create_map(map_dir, pixel_size, stage_to_image=((0.0, 1.0), (1.0, 0.0))):
    """Start an empty map.

    Parameters
    ----------
    map_dir: str
        directory to hold the map; must not already contain one
    pixel_size: float
        size of an image pixel, in the same unit as the stage positions given to :func:`add_tile`
    stage_to_image: 2x2 matrix
        maps a stage displacement (x, y) to an image displacement (row, column), before dividing
        by pixel_size. The default [[0, 1], [1, 0]] means x moves along columns and y along rows;
        a sign flip or a rotated scan needs another matrix, measured once per instrument (move the
        stage, measure the image shift).
    """
    matrix = np.asarray(stage_to_image, dtype=float)
    if matrix.shape != (2, 2) or abs(np.linalg.det(matrix)) < 1e-9:
        raise ValueError('stage_to_image must be an invertible 2x2 matrix')
    if pixel_size <= 0:
        raise ValueError('pixel_size must be positive')
    os.makedirs(map_dir, exist_ok=True)
    if os.path.exists(os.path.join(map_dir, manifest_name)):
        raise FileExistsError(f'{map_dir} already holds a map')
    _write_manifest(map_dir, {'pixel_size': float(pixel_size), 'stage_to_image': matrix.tolist(),
                              'origin_position': None, 'map_origin_px': [0, 0], 'next_label': 1, 'tiles': []})
    return _read_manifest(map_dir)


def _grow(array, shape, offset, fill):
    grown = np.full(shape, fill, dtype=array.dtype)
    grown[offset[0]:offset[0] + array.shape[0], offset[1]:offset[1] + array.shape[1]] = array
    return grown


def _find(parent, label):
    while parent.get(label, label) != label:
        parent[label] = parent.get(parent[label], parent[label])
        label = parent[label]
    return label


def merge_labels(map_labels, tile_labels, top_left, first_label, min_overlap_px=10):
    """Write a tile's labels into the map's, joining regions that overlap.

    Parameters
    ----------
    map_labels: 2D int array, modified in place, 0 is unlabelled
    tile_labels: 2D int array of the tile, 0 is unlabelled
    top_left: (row, column) of the tile in the map
    first_label: first unused label number in the map
    min_overlap_px: overlap needed to join a tile region with a map region

    Returns
    -------
    next unused label, number of tile regions added, number joined to an existing region
    """
    rows = slice(top_left[0], top_left[0] + tile_labels.shape[0])
    columns = slice(top_left[1], top_left[1] + tile_labels.shape[1])
    window = map_labels[rows, columns]
    parent = {}
    added = joined = 0
    next_label = first_label
    for tile_label in np.unique(tile_labels):
        if tile_label == 0:
            continue
        region = tile_labels == tile_label
        new_label = next_label
        next_label += 1
        added += 1
        existing, counts = np.unique(window[region], return_counts=True)
        for label, count in zip(existing, counts):
            if label != 0 and count >= min_overlap_px:
                parent[_find(parent, new_label)] = _find(parent, int(label))
        if _find(parent, new_label) != new_label:
            joined += 1
        window[region & (window == 0)] = new_label
    if parent:
        lookup = np.arange(next_label, dtype=map_labels.dtype)
        for label in list(parent):
            lookup[label] = _find(parent, label)
        map_labels[...] = lookup[map_labels]
    return next_label, added, joined


def _overlap_shift(total, count, image, top_left, max_shift_px, min_overlap=16, min_score=0.5, min_gain=0.05):
    """Integer (row, column) correction of a tile's placement from its overlap with the map.

    Normalised cross-correlation of the tile's overlapping part against the map region
    around it, searched up to ``max_shift_px``; no wrap-around, so narrow overlap strips
    work. The correction is accepted only when the best match scores at least
    ``min_score`` and beats the stage-based placement by ``min_gain``: periodic or
    featureless overlaps give several near-equal matches, and trusting the stage is
    then better than picking one.

    Returns
    -------
    dict with shift (accepted correction or [0, 0]), accepted, score_best and
    score_stage, or None when the overlap is too small or mostly empty
    """
    rows = slice(max(top_left[0], 0), min(top_left[0] + image.shape[0], count.shape[0]))
    columns = slice(max(top_left[1], 0), min(top_left[1] + image.shape[1], count.shape[1]))
    if rows.stop - rows.start < min_overlap or columns.stop - columns.start < min_overlap:
        return None
    if (count[rows, columns] > 0).mean() < 0.5:
        return None
    limit = int(np.ceil(max_shift_px))
    search_rows = slice(max(rows.start - limit, 0), min(rows.stop + limit, count.shape[0]))
    search_columns = slice(max(columns.start - limit, 0), min(columns.stop + limit, count.shape[1]))
    covered = count[search_rows, search_columns] > 0
    search = total[search_rows, search_columns] / np.maximum(count[search_rows, search_columns], 1)
    search = np.where(covered, search, search[covered].mean())
    template = image[rows.start - top_left[0]:rows.stop - top_left[0],
                     columns.start - top_left[1]:columns.stop - top_left[1]]
    if np.std(template) == 0 or np.std(search) == 0:
        return None
    score = match_template(search, template)
    peak = np.unravel_index(int(np.argmax(score)), score.shape)
    nominal = (rows.start - search_rows.start, columns.start - search_columns.start)
    best, stage = float(score[peak]), float(score[nominal])
    accepted = best >= min_score and best - stage >= min_gain
    shift = [int(peak[0] - nominal[0]), int(peak[1] - nominal[1])] if accepted else [0, 0]
    return {'shift': shift, 'accepted': bool(accepted), 'score_best': best, 'score_stage': stage}


def add_tile(map_dir, image, position, labels=None, refine=True, max_shift_px=None, min_overlap_px=10):
    """Place one tile into the map.

    Parameters
    ----------
    map_dir: str
    image: 2D array
    position: (x, y) stage position of the tile centre, in the map's pixel_size unit
    labels: 2D int array with the shape of ``image``, or None
    refine: bool
        correct the stage-based placement by matching the overlap with the map; the correction
        is kept only when the match is clearly better than the stage placement (see overlap_match
        in the result)
    max_shift_px: float or None
        largest accepted refinement; a quarter of the smaller tile side by default
    min_overlap_px: overlap needed to join regions across tiles

    Returns
    -------
    dict with the tile's top-left position in the map, the refinement applied, the map
    shape and how many regions were added and joined
    """
    manifest = _read_manifest(map_dir)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f'a tile must be a 2D image, got shape {image.shape}')
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != image.shape:
            raise ValueError(f'labels shape {labels.shape} differs from image shape {image.shape}')
    paths = _paths(map_dir)
    if manifest['origin_position'] is None:
        manifest['origin_position'] = [float(position[0]), float(position[1])]
        total = np.zeros((0, 0))
        count = np.zeros((0, 0))
        region_map = np.zeros((0, 0), dtype=np.int64)
    else:
        total, count, region_map = (np.load(paths[name]) for name in ('sum', 'count', 'labels'))

    displacement = np.asarray(position, dtype=float) - np.asarray(manifest['origin_position'])
    centre = np.asarray(manifest['stage_to_image']) @ displacement / manifest['pixel_size']
    top_left = np.round(np.array(centre) - np.array(image.shape) / 2.0).astype(int) + np.array(manifest['map_origin_px'])

    refinement = [0.0, 0.0]
    match = None
    if refine and count.size:
        match = _overlap_shift(total, count, image, top_left,
                               max_shift_px if max_shift_px is not None else min(image.shape) / 4)
        if match is not None:
            refinement = [float(value) for value in match['shift']]
            top_left = top_left + np.asarray(match['shift'], dtype=int)

    low = np.minimum(top_left, 0)
    pad = -low
    needed = np.maximum(np.array(count.shape) + pad, top_left + pad + np.array(image.shape))
    if count.size == 0:
        total = np.zeros(needed)
        count = np.zeros(needed)
        region_map = np.zeros(needed, dtype=np.int64)
    elif np.any(pad > 0) or np.any(needed > np.array(count.shape)):
        total = _grow(total, needed, pad, 0.0)
        count = _grow(count, needed, pad, 0.0)
        region_map = _grow(region_map, needed, pad, 0)
    top_left = top_left + pad
    manifest['map_origin_px'] = [int(v) for v in np.array(manifest['map_origin_px']) + pad]
    for tile in manifest['tiles']:
        tile['top_left_px'] = [int(v) for v in np.array(tile['top_left_px']) + pad]

    rows = slice(top_left[0], top_left[0] + image.shape[0])
    columns = slice(top_left[1], top_left[1] + image.shape[1])
    total[rows, columns] += image
    count[rows, columns] += 1
    added = joined = 0
    if labels is not None:
        manifest['next_label'], added, joined = merge_labels(region_map, labels, top_left, manifest['next_label'],
                                                             min_overlap_px)
    manifest['tiles'].append({'position': [float(position[0]), float(position[1])],
                              'top_left_px': [int(v) for v in top_left], 'shape': list(image.shape),
                              'refinement_px': refinement, 'has_labels': labels is not None})
    np.save(paths['sum'], total)
    np.save(paths['count'], count)
    np.save(paths['labels'], region_map)
    _write_manifest(map_dir, manifest)
    return {'tile_index': len(manifest['tiles']) - 1, 'top_left_px': [int(v) for v in top_left],
            'refinement_px': refinement, 'overlap_match': match, 'map_shape': list(count.shape),
            'regions_added': added, 'regions_joined': joined}


def map_image(map_dir):
    """The stitched image (overlaps averaged, NaN where no tile was placed)."""
    paths = _paths(map_dir)
    total, count = np.load(paths['sum']), np.load(paths['count'])
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def map_labels(map_dir):
    """The merged label image, regions renumbered 1..n."""
    labels = np.load(_paths(map_dir)['labels'])
    present = np.unique(labels)
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[present[present > 0]] = np.arange(1, int((present > 0).sum()) + 1)
    return lookup[labels]


def map_summary(map_dir):
    """Tiles, map size and per-region statistics in map coordinates and physical units."""
    manifest = _read_manifest(map_dir)
    if not manifest['tiles']:
        return {'tiles': 0, 'map_shape': [0, 0], 'regions': []}
    labels = map_labels(map_dir)
    statistics = particle_statistics(labels, map_image(map_dir), manifest['pixel_size'])
    return {'tiles': len(manifest['tiles']), 'map_shape': list(labels.shape),
            'pixel_size': manifest['pixel_size'], 'regions': statistics}
