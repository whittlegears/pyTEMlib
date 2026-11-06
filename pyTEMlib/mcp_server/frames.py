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

from pyTEMlib.image_tools.image_atoms import find_blobs
from pyTEMlib.image_tools.image_utilities import experiment_from_asyncroscopy, get_image_centre_m, pixel_to_stage, stage_to_pixel

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
    # The frame needs the image's shape, calibration and position, not its pixels: a lazy array of zeros stands in.
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
            return {"error": {"code": "unknown_dataset", "message": f"no map or image {dataset!r}"}}
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        rows, cols = frame["shape"]
        if not (-0.5 <= row <= rows - 0.5 and -0.5 <= col <= cols - 0.5):
            # A small agent passed a point of interest's nanometres (x = -5.9, y = 379.4) as row and col.
            return {"error": {"code": "outside", "message": (
                f"pixel ({row:g}, {col:g}) is outside this {rows} x {cols} pixel {frame['kind']}: row and col are pixels, "
                f"0 to {rows - 1} and 0 to {cols - 1}. A position given in nanometres or metres is already a stage "
                "position (metres = nm x 1e-9): use it as it is, without to_stage.")}}
        x, y = stage_of(frame, row, col)
        notes =["this image does not record its image shift, so it is placed as if it had none"] if frame["kind"] == "image" and not frame["shift_recorded"] else []
        notes.append(
            "before moving there, check the instrument's get_sample gives this sample_load_id: another load puts the sample elsewhere"
            if frame["sample"].get("sample_load_id") else "this dataset does not record which sample load it was taken on"
        )
        return {"x_m": x, "y_m": y, "inside": True, "kind": frame["kind"],
                **frame["sample"], "note": "; ".join(notes)}

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
            return {"error": {"code": "unknown_dataset", "message": f"no image {image!r}"}}
        except FrameError as exc:
            return {"error": {"code": "no_frame", "message": str(exc)}}
        if frame["kind"] != "image":
            return {"error": {"code": "not_an_image", "message": f"{image} is a stitched map: score_map ranks a map's regions"}}
        features = features_of(frame, image_node(store, image).read(), n, min_size_m, max_size_m)
        note = {} if frame["shift_recorded"] else {"note": "this image does not record its image shift, so x_m, y_m assume it had none"}
        if not features and n > 0 and frame.get("beam_current_pa") == 0:
            note["hint"] = "this image was taken with the beam blanked (beam_current_pa 0), so it shows only noise: unblank_beam and take it again"
        elif not features and n > 0:
            # A small agent at a 20 nm field searched for particles over and over: say why nothing was found and what would find it.
            fov = frame["pixel_size_m"] * max(frame["shape"])
            note["hint"] = (f"nothing distinct in this {fov:.3g} m wide image. A feature wider than about a third of the field fills it "
                            "and cannot be told apart: to find particles, take an image whose field of view is several times their "
                            "size (200 to 600 nm for nanoparticles). Bare support or noise also shows nothing.")
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
