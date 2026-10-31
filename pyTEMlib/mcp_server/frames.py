"""Pixel <-> stage coordinates for a stitched map or a single image, so an agent never does the arithmetic.

A stitched map carries its own transform (origin and pixel size). A single image is placed from what its
acquisition recorded: the stage position (reported, else commanded) plus the image (beam) shift it was taken
with (``image_shift_m``) is the image centre, rows run along +y and columns along +x (the ``axes`` the twin and
instruments record), one pixel is ``pixel_size_m``. An image taken at a non-zero tilt, or over part of the scan
field, is refused: the flat transform would be wrong. An image that does not record its shift is placed as if it
had none, and the reply says so.

``find_features`` gives an agent that cannot see images where the distinct features of one are: bright blobs
(particles, clusters, atom columns at atomic scale) with their pixel and stage positions, size and contrast, so it
can aim a finer image at something a placement can match.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from .map_store import pixel_to_stage

MAX_TILT_DEG = 0.05
BLOB_DIAMETER_PER_SIGMA = 2.0 * math.sqrt(2.0)  # a Laplacian-of-Gaussian blob of scale sigma is this many sigmas across
MIN_CONTRAST = 3.0  # a feature's mean above the background, in units of the image's noise: below it, it is noise
AXES = "rows=+y, columns=+x"


class FrameError(ValueError):
    pass


def _vector(value) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
        return [float(v) for v in value]
    return None


def image_node(store, key: str):
    """The 2-D array of an acquisition: the key itself, or the first array under its ``image`` group."""
    node = store.client[key]
    if getattr(node, "structure_family", None) == "container":
        group = node["image"] if "image" in node else node
        arrays = [group[name] for name in group if getattr(group[name], "structure_family", None) == "array"]
        placed = [a for a in arrays if "pixel_size_m" in dict(a.metadata)]
        if not (placed or arrays):
            raise FrameError(f"{key} holds no image array")
        node = (placed or arrays)[0]
    return node


def frame_of(store, key: str) -> dict[str, Any]:
    """How ``key``'s pixels sit on the stage: {"kind", "origin_xy_m" or "centre_xy_m", "pixel_size_m", "shape"}."""
    try:
        meta = store.map_metadata(key)
    except (KeyError, ValueError, FileNotFoundError):
        meta = None
    if meta and meta.get("kind") == "stitched_map":
        return {"kind": "map", "metadata": meta, "pixel_size_m": float(meta["pixel_size_m"]), "shape": list(meta["canvas_shape"]), "sample": store.sample_of(key)}
    node = image_node(store, key)
    m = dict(node.metadata)
    stage = _vector(m.get("stage_reported_m")) or _vector(m.get("stage_commanded_m"))
    pixel = m.get("pixel_size_m")
    if not stage or len(stage) < 2 or not isinstance(pixel, (int, float)) or not pixel > 0:
        raise FrameError(f"{key} does not record where it was taken (stage_reported_m and pixel_size_m)")
    if m.get("axes") != AXES:  # an image that does not say which way its rows run cannot be placed
        raise FrameError(f"{key} records axes {m.get('axes')!r}; only {AXES!r} is understood")
    region = _vector(m.get("scan_region"))
    if region is not None and region != [0.0, 0.0, 1.0, 1.0]:
        raise FrameError(f"{key} covers part of the scan field ({region}); its pixels cannot be placed on the stage yet")
    if not all(math.isfinite(v) for v in stage[:5]):
        raise FrameError(f"{key} records a stage position that is not a number ({stage}); it cannot be placed on the stage")
    tilts = stage[3:5]
    if any(abs(t) > MAX_TILT_DEG for t in tilts):
        raise FrameError(f"{key} was taken at tilt {tilts} degrees; the flat pixel-to-stage transform would be wrong there")
    shape = list(node.structure().shape)[-2:]
    shift = _vector(m.get("image_shift_m"))
    if shift is not None and not all(math.isfinite(v) for v in shift[:2]):
        raise FrameError(f"{key} records an image shift that is not a number ({shift})")
    centre = [stage[0] + shift[0], stage[1] + shift[1]] if shift else stage[:2]
    return {"kind": "image", "centre_xy_m": centre, "pixel_size_m": float(pixel), "shape": shape, "sample": store.sample_of(key),
            "shift_recorded": shift is not None}


def stage_of(frame: dict, row: float, col: float) -> tuple[float, float]:
    if frame["kind"] == "map":
        return pixel_to_stage(frame["metadata"], row, col)
    rows, cols = frame["shape"]
    p = frame["pixel_size_m"]
    cx, cy = frame["centre_xy_m"]
    return cx + (col - (cols - 1) / 2.0) * p, cy + (row - (rows - 1) / 2.0) * p


def pixel_of(frame: dict, x: float, y: float) -> tuple[float, float]:
    p = frame["pixel_size_m"]
    if frame["kind"] == "map":
        ox, oy = frame["metadata"]["origin_xy_m"]
        return (y - oy) / p, (x - ox) / p
    rows, cols = frame["shape"]
    cx, cy = frame["centre_xy_m"]
    return (y - cy) / p + (rows - 1) / 2.0, (x - cx) / p + (cols - 1) / 2.0


def features_in(image: np.ndarray, pixel_m: float, n: int = 5, min_size_m: float | None = None, max_size_m: float | None = None) -> list[dict]:
    """The ``n`` most distinct bright blobs of ``image``: [{"row", "col", "diameter_m", "contrast"}], strongest first.

    Sizes default to 3 pixels up to an eighth of the image. Contrast is the blob's mean above the image's median, in
    units of its noise (the median absolute deviation); blobs under MIN_CONTRAST, or not wholly inside the image, are
    left out."""
    from skimage.feature import blob_log

    image = np.asarray(image, dtype=np.float64)
    rows, cols = image.shape
    background = float(np.median(image))
    noise = float(np.median(np.abs(image - background))) * 1.4826 or float(image.std()) or 1.0
    scaled = np.clip((image - background) / noise, 0.0, None)
    smallest = max(1.0, (min_size_m or 3.0 * pixel_m) / pixel_m / BLOB_DIAMETER_PER_SIGMA)
    largest = max(smallest, (max_size_m or min(rows, cols) * pixel_m / 8.0) / pixel_m / BLOB_DIAMETER_PER_SIGMA)
    found = []
    for row, col, sigma in blob_log(scaled, min_sigma=smallest, max_sigma=largest, num_sigma=8, threshold=None, threshold_rel=0.05):
        radius = sigma * BLOB_DIAMETER_PER_SIGMA / 2.0
        if row - radius < 0 or col - radius < 0 or row + radius > rows - 1 or col + radius > cols - 1:
            continue
        r0, r1 = int(max(0, row - radius)), int(min(rows, row + radius + 1))
        c0, c1 = int(max(0, col - radius)), int(min(cols, col + radius + 1))
        yy, xx = np.mgrid[r0:r1, c0:c1]
        inside = (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
        contrast = (float(image[r0:r1, c0:c1][inside].mean()) - background) / noise
        if contrast >= MIN_CONTRAST:
            found.append({"row": float(row), "col": float(col), "diameter_m": float(2.0 * radius * pixel_m), "contrast": round(contrast, 2)})
    return sorted(found, key=lambda f: -f["contrast"])[: max(0, int(n))]


def register(mcp, store) -> None:
    def finite(*values) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)

    @mcp.tool()
    def to_stage(dataset: str, row: float, col: float) -> dict[str, Any]:
        """Stage x, y (metres) of pixel (row, col) of a stitched map or a single image. Use it instead of doing the
        arithmetic yourself, e.g. before moving the stage to a feature you found in an image."""
        if not finite(row, col):
            return {"error": {"code": "bad_pixel", "message": "row and col must be finite numbers"}}
        try:
            frame = frame_of(store, dataset)
        except KeyError:
            return {"error": {"code": "unknown_dataset", "message": f"no map or image {dataset!r}"}}
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        x, y = stage_of(frame, row, col)
        rows, cols = frame["shape"]
        notes = ["this image does not record its image shift, so it is placed as if it had none"] if frame["kind"] == "image" and not frame["shift_recorded"] else []
        notes.append(
            "before moving there, check the instrument's get_sample gives this sample_load_id: another load puts the sample elsewhere"
            if frame["sample"].get("sample_load_id") else "this dataset does not record which sample load it was taken on"
        )
        return {"x_m": x, "y_m": y, "inside": -0.5 <= row <= rows - 0.5 and -0.5 <= col <= cols - 0.5, "kind": frame["kind"],
                **frame["sample"], "note": "; ".join(notes)}

    @mcp.tool()
    def find_features(image: str, n: int = 5, min_size_m: float | None = None, max_size_m: float | None = None) -> dict[str, Any]:
        """The most distinct bright features (particles, clusters; atom columns at atomic scale) of a single image,
        strongest first, each with its pixel (row, col), stage position (x_m, y_m), diameter_m and contrast (times the
        noise). Use it to choose where to aim a finer image or a spectrum: aim at x_m, y_m.
        n: how many (default 5). min_size_m, max_size_m: the feature sizes to look for (default 3 pixels to 1/8 of the
        image). No features: the image shows nothing distinct (bare support, or noise)."""
        try:
            frame = frame_of(store, image)
        except KeyError:
            return {"error": {"code": "unknown_dataset", "message": f"no image {image!r}"}}
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        if frame["kind"] != "image":
            return {"error": {"code": "not_an_image", "message": f"{image} is a stitched map: score_map ranks a map's regions"}}
        features = features_in(np.asarray(image_node(store, image).read()), frame["pixel_size_m"], n, min_size_m, max_size_m)
        for f in features:
            f["x_m"], f["y_m"] = stage_of(frame, f["row"], f["col"])
        note = {} if frame["shift_recorded"] else {"note": "this image does not record its image shift, so x_m, y_m assume it had none"}
        return {"features": features, "pixel_size_m": frame["pixel_size_m"], "shape": frame["shape"], **frame["sample"], **note}

    @mcp.tool()
    def to_pixel(dataset: str, x_m: float, y_m: float) -> dict[str, Any]:
        """Pixel (row, col) of stage position (x_m, y_m) on a stitched map or a single image, and whether it is inside."""
        if not finite(x_m, y_m):
            return {"error": {"code": "bad_position", "message": "x_m and y_m must be finite numbers"}}
        try:
            frame = frame_of(store, dataset)
        except KeyError:
            return {"error": {"code": "unknown_dataset", "message": f"no map or image {dataset!r}"}}
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        row, col = pixel_of(frame, x_m, y_m)
        rows, cols = frame["shape"]
        return {"row": row, "col": col, "inside": -0.5 <= row <= rows - 0.5 and -0.5 <= col <= cols - 0.5, "kind": frame["kind"], **frame["sample"]}
