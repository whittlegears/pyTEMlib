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

An image's frame holds it as a sidpy dataset (``image_dataset``): the transform and the blob finder are
pyTEMlib's ``image_tools.pixel_to_stage``, ``stage_to_pixel`` and ``find_blobs``, and asyncroscopy's flat
metadata is translated with ``image_tools.experiment_from_asyncroscopy``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import dask.array
import numpy as np
import sidpy

from pyTEMlib.image_tools import experiment_from_asyncroscopy, find_blobs, get_image_centre_m, pixel_to_stage, stage_to_pixel

from . import map_store

MAX_TILT_DEG = 0.05
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


def image_dataset(data, pixel_size_m: float, experiment: dict | None = None, title: str = "image") -> sidpy.Dataset:
    """A sidpy IMAGE dataset of ``data`` (a numpy or dask array) with square pixels of ``pixel_size_m``.

    The spatial dimensions are y (rows) and x (columns) in nm from the first pixel, the pyTEMlib convention;
    ``experiment`` becomes ``metadata['experiment']`` (``stage_position``, ``image_shift``, ...)."""
    dataset = sidpy.Dataset.from_array(data, title=title)
    dataset.data_type = "IMAGE"
    pixel_nm = float(pixel_size_m) * 1e9
    for axis, name in enumerate(("y", "x")):
        dataset.set_dimension(axis, sidpy.Dimension(np.arange(dataset.shape[axis]) * pixel_nm, name=name, units="nm",
                                                    quantity="Length", dimension_type="spatial"))
    dataset.metadata["experiment"] = dict(experiment or {})
    return dataset


NEWEST_KEYS = 8


def unknown_key(store, key: str, what: str) -> dict[str, Any]:
    """The refusal for a key the store does not have, with the newest keys it does have to copy from.

    Small agents passed words in place of keys (place_image on "new_map", find_features on "stitched_map_key"):
    a bare "no image" gave them nothing to correct the call with."""
    try:
        newest = list(store.client.keys().tail(NEWEST_KEYS))
    except Exception:
        newest = []
    message = f"no {what} {key!r}: pass a key exactly as a tool returned it"
    return {"error": {"code": "unknown_dataset", "message": message, **({"newest_keys": newest[::-1]} if newest else {})}}


def image_array(store, key: str) -> tuple[Any, str]:
    """The 2-D array of an acquisition and its key: the key itself, or the first array under its ``image`` group
    (one that records ``pixel_size_m`` first)."""
    node = store.client[key]
    if getattr(node, "structure_family", None) != "container":
        return node, key
    prefix = f"{key}/image" if "image" in node else key
    group = node["image"] if "image" in node else node
    arrays = [(name, group[name]) for name in group if getattr(group[name], "structure_family", None) == "array"]
    placed = [(name, a) for name, a in arrays if "pixel_size_m" in dict(a.metadata)]
    if not (placed or arrays):
        raise FrameError(f"{key} holds no image array")
    name, found = (placed or arrays)[0]
    return found, f"{prefix}/{name}"


def image_node(store, key: str):
    """The 2-D array of an acquisition: the key itself, or the first array under its ``image`` group."""
    return image_array(store, key)[0]


def image_descriptor(store, key: str) -> dict[str, Any]:
    """Which array of ``key`` is its image and how it sits on the stage: {"kind", "image", "shape", "dtype", "chunks",
    "frame"}. ``frame`` is {"pixel_size_m", "origin_xy_m", "centre_xy_m", "axes"}, where pixel (r, c) is centred at
    stage (origin_xy_m[0] + c * pixel_size_m, origin_xy_m[1] + r * pixel_size_m); it is null, with ``frame_refused``
    saying why, when the image cannot be placed. A stitched map's image is its level 0."""
    try:
        meta = store.map_metadata(key)
    except (KeyError, ValueError, FileNotFoundError):
        meta = None
    if meta and meta.get("kind") == "stitched_map":
        view = store.map_view(key, meta)
        level = view["pyramid"][0]
        rows, cols = level["shape"]
        origin = level["origin_xy_m"]
        pixel = level["pixel_size_m"]
        centre = [origin[0] + (cols - 1) / 2.0 * pixel, origin[1] + (rows - 1) / 2.0 * pixel]
        return {"kind": "map", "image": level["key"], "shape": level["shape"], "dtype": level["dtype"], "chunks": level["chunks"],
                "frame": {"pixel_size_m": pixel, "origin_xy_m": origin, "centre_xy_m": centre, "axes": view["axes"]}, **store.sample_of(key)}
    node, array_key = image_array(store, key)
    structure = node.structure()
    described = {"kind": "image", "image": array_key, "shape": [int(v) for v in structure.shape][-2:], "dtype": structure.data_type.to_numpy_dtype().str,
                 "chunks": [[int(v) for v in axis] for axis in structure.chunks]}
    try:
        frame = frame_of(store, key)
    except FrameError as exc:
        return {**described, "frame": None, "frame_refused": str(exc), **store.sample_of(key)}
    x0, y0 = pixel_to_stage(frame["dataset"], 0, 0)
    reply = {**described, "frame": {"pixel_size_m": frame["pixel_size_m"], "origin_xy_m": [x0, y0], "centre_xy_m": frame["centre_xy_m"], "axes": AXES},
             **frame["sample"]}
    if not frame["shift_recorded"]:
        reply["note"] = "this image does not record its image shift, so it is placed as if it had none"
    return reply


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
    experiment = experiment_from_asyncroscopy(m)
    stage = experiment.get("stage_position")
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
    shift = experiment.get("image_shift")
    if shift is not None and not (len(shift) >= 2 and all(math.isfinite(v) for v in shift[:2])):
        raise FrameError(f"{key} records an image shift that is not a number ({shift})")
    # The frame needs only the shape, calibration and position, so a lazy array of zeros stands in for the pixels.
    dataset = image_dataset(dask.array.zeros(shape, chunks=shape), float(pixel), experiment, title=key)
    try:
        current = float(m["beam_current_pa"]) if "beam_current_pa" in m else None
    except (TypeError, ValueError):
        current = None
    return {"kind": "image", "dataset": dataset, "centre_xy_m": [float(v) for v in get_image_centre_m(dataset)], "pixel_size_m": float(pixel),
            "shape": shape, "sample": store.sample_of(key), "shift_recorded": shift is not None, "beam_current_pa": current}


def stage_of(frame: dict, row: float, col: float) -> tuple[float, float]:
    if frame["kind"] == "map":
        return map_store.pixel_to_stage(frame["metadata"], row, col)
    return pixel_to_stage(frame["dataset"], row, col)


def pixel_of(frame: dict, x: float, y: float) -> tuple[float, float]:
    if frame["kind"] == "map":
        p = frame["pixel_size_m"]
        ox, oy = frame["metadata"]["origin_xy_m"]
        return (y - oy) / p, (x - ox) / p
    return stage_to_pixel(frame["dataset"], x, y)


def features_of(frame: dict, image: np.ndarray, n: int = 5, min_size_m: float | None = None, max_size_m: float | None = None) -> list[dict]:
    """The ``n`` most distinct bright blobs of an image (see ``image_tools.find_blobs``) with pixel and stage positions:
    [{"row", "col", "diameter_m", "contrast", "x_m", "y_m"}], strongest first. Sizes default to 3 pixels up to an eighth
    of the image."""
    dataset = image_dataset(np.asarray(image), frame["pixel_size_m"], frame["dataset"].metadata["experiment"])
    nm = 1e9  # image_dataset's dimensions are in nm
    blobs = find_blobs(dataset, n, None if min_size_m is None else min_size_m * nm, None if max_size_m is None else max_size_m * nm)
    features = []
    for row, col, diameter, contrast in blobs:
        x, y = pixel_to_stage(dataset, row, col)
        features.append({"row": float(row), "col": float(col), "diameter_m": float(diameter / nm), "contrast": round(float(contrast), 2), "x_m": x, "y_m": y})
    return features


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
            return unknown_key(store, dataset, "map or image")
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        rows, cols = frame["shape"]
        if not (-0.5 <= row <= rows - 0.5 and -0.5 <= col <= cols - 0.5):
            # Agents sometimes pass a point's nanometres as row and col.
            return {"error": {"code": "outside", "message": (
                f"pixel ({row:g}, {col:g}) is outside this {rows} x {cols} pixel {frame['kind']}: row and col are pixels, "
                f"0 to {rows - 1} and 0 to {cols - 1}. A position given in nanometres or metres is already a stage "
                "position (metres = nm x 1e-9): use it as it is, without to_stage.")}}
        x, y = stage_of(frame, row, col)
        notes = ["this image does not record its image shift, so it is placed as if it had none"] if frame["kind"] == "image" and not frame["shift_recorded"] else []
        reply = {"x_m": x, "y_m": y, "inside": True, "kind": frame["kind"], **frame["sample"]}
        load = frame["sample"].get("sample_load_id")
        if load:
            # Another load puts the sample elsewhere, so the instrument's sample record must match before moving there.
            reply["next_actions"] = [{"server": "instrument", "command": "get_sample", "args": {}, "expect": {"sample_load_id": load}}]
        else:
            notes.append("this dataset does not record which sample load it was taken on")
        if notes:
            reply["note"] = "; ".join(notes)
        return reply

    @mcp.tool()
    def find_features(image: str, n: int = 5, min_size_m: float | None = None, max_size_m: float | None = None) -> dict[str, Any]:
        """The most distinct bright features (particles, clusters; atom columns at atomic scale) of a single image,
        strongest first, each with its pixel (row, col), stage position (x_m, y_m), diameter_m and contrast (times the
        noise). Use it to choose where to aim a finer image or a spectrum: aim at x_m, y_m (an instrument's
        place_beam_at parks the beam there for a spectrum).
        n: how many (default 5). min_size_m, max_size_m: the feature sizes to look for (default 3 pixels to 1/8 of the
        image). No features: the image shows nothing distinct (bare support, or noise)."""
        try:
            frame = frame_of(store, image)
        except KeyError:
            return unknown_key(store, image, "image")
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        if frame["kind"] != "image":
            return {"error": {"code": "not_an_image", "message": f"{image} is a stitched map: score_map ranks a map's regions"}}
        features = features_of(frame, image_node(store, image).read(), n, min_size_m, max_size_m)
        note = {} if frame["shift_recorded"] else {"note": "this image does not record its image shift, so x_m, y_m assume it had none"}
        if not features and n > 0 and frame.get("beam_current_pa") == 0:
            # Only noise. The beam must be unblanked and the image taken again.
            note["warnings"] = [{"code": "beam_blanked", "beam_current_pa": 0}]
            note["next_actions"] = [{"server": "instrument", "command": "unblank_beam", "args": {}}]
        elif not features and n > 0:
            # Say why nothing was found and what would find it, so an agent does not repeat the same search.
            fov = frame["pixel_size_m"] * max(frame["shape"])
            note["hint"] = (f"nothing distinct in this {fov:.3g} m wide image. A feature wider than about a third of the field fills it "
                            "and cannot be told apart: to find particles, take an image whose field of view is several times their "
                            "size (200 to 600 nm for nanoparticles). Bare support or noise also shows nothing.")
        return {"features": features, "pixel_size_m": frame["pixel_size_m"], "shape": frame["shape"], **frame["sample"], **note}

    @mcp.tool()
    def image_info(dataset: str) -> dict[str, Any]:
        """Which array of a dataset (an acquisition's Tiled key, or a stitched map) is its image, and where it sits on
        the stage. Returns {"kind": "image" or "map", "image": the array's key, "shape": [rows, cols], "dtype", "chunks",
        "frame": {"pixel_size_m", "origin_xy_m": stage x, y of pixel (0, 0), "centre_xy_m", "axes"}}; frame is null,
        with "frame_refused" saying why, for an image that does not record where it was taken."""
        try:
            return image_descriptor(store, dataset)
        except KeyError:
            return unknown_key(store, dataset, "map or image")
        except (FrameError, FileNotFoundError) as exc:
            return {"error": {"code": "no_image", "message": str(exc)}}

    @mcp.tool()
    def to_pixel(dataset: str, x_m: float, y_m: float) -> dict[str, Any]:
        """Pixel (row, col) of stage position (x_m, y_m) on a stitched map or a single image, and whether it is inside."""
        if not finite(x_m, y_m):
            return {"error": {"code": "bad_position", "message": "x_m and y_m must be finite numbers"}}
        try:
            frame = frame_of(store, dataset)
        except KeyError:
            return unknown_key(store, dataset, "map or image")
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        row, col = pixel_of(frame, x_m, y_m)
        rows, cols = frame["shape"]
        return {"row": row, "col": col, "inside": -0.5 <= row <= rows - 0.5 and -0.5 <= col <= cols - 0.5, "kind": frame["kind"], **frame["sample"]}
