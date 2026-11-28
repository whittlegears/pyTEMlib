"""pytemlib_mcp/schemas/stitched_map.schema.json against what pyTEMlib itself produces: stitch_quality's output."""

import json
import os

import jsonschema
import numpy as np
from scipy import ndimage

from pyTEMlib.image_tools.image_mosaic import Tile, register_tiles, stitch_quality

schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'schemas', 'stitched_map.schema.json')
pixel_size_m = 1e-9
side = 96


def schema():
    with open(schema_path) as handle:
        return json.load(handle)


def quality_validator():
    full = schema()
    return jsonschema.Draft202012Validator({'$schema': full['$schema'], '$defs': full['$defs'], '$ref': '#/$defs/stitch_quality'})


def tiles(grid, with_times):
    rng = np.random.default_rng(4)
    step = int(side * 0.75)
    scene = ndimage.gaussian_filter(rng.normal(size=(side + step * grid, side + step * grid)), 2.0)
    out = []
    for r in range(grid):
        for c in range(grid):
            image = scene[r * step: r * step + side, c * step: c * step + side].copy()
            centre = ((c * step + side / 2) * pixel_size_m, (r * step + side / 2) * pixel_size_m)
            out.append(Tile(image, centre, pixel_size_m, len(out), acquired_s=10.0 * len(out) if with_times else None))
    return out


def test_the_schema_is_a_valid_draft_2020_12_schema():
    jsonschema.Draft202012Validator.check_schema(schema())


def test_stitch_quality_of_a_map_too_small_to_measure_the_scan_matches_the_schema():
    quality = stitch_quality(register_tiles(tiles(2, False), refine=False))
    quality_validator().validate(json.loads(json.dumps(quality, allow_nan=False)))
    assert quality['scan_linear']['stderr'] is None


def test_stitch_quality_of_a_refined_map_with_drift_matches_the_schema():
    quality = stitch_quality(register_tiles(tiles(4, True), stage_sigma_m=4 * pixel_size_m, max_drift_m_per_s=0.05 * pixel_size_m))
    quality_validator().validate(json.loads(json.dumps(quality, allow_nan=False)))
    assert quality['matches_used'] > 0 and 'drift' in quality and 'solver' in quality
    assert quality['scan_linear']['stderr'] is not None


def test_a_field_stitch_quality_does_not_give_is_refused():
    quality = stitch_quality(register_tiles(tiles(2, False), refine=False))
    errors = list(quality_validator().iter_errors({**quality, 'positions_px': []}))
    assert errors
