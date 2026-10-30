"""MCP tools that place a finer image on a coarser one and keep the placements (a multiscale map's links).

``place_image`` reads the child image and the part of its parent (a stitched map or a single image) around where
it should be, places it with ``pyTEMlib.image_tools.register.register_child``, and appends the result to an
append-only JSON-lines log (one record per call, placed or not, never rewritten). ``list_placements`` reads it.

Where a child should be comes from the stage position it records, or from ``centre_xy_m`` when it was aimed by
image shift (which an image does not record): the agent that shifted the beam knows where it pointed. The
search reaches ``window_m`` around that, so a stage-aimed child needs a window as large as the stage's error,
and a shift-aimed one only the shift's calibration error.

A record names the child, the parent (and the parent map's version), where the child's centre is in the parent's
pixels and on the stage, the score and ambiguity of the match, and, when it could not be placed, why. A child is
never placed by a guess: "unplaced" is an answer.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from pyTEMlib.image_tools.register import register_child

from .frames import FrameError, frame_of, image_node, pixel_of, stage_of

DEFAULT_WINDOW_M = 4e-7  # the stage's error on a typical instrument: a stage-aimed child is searched this far
EDGE_PX = 4  # parent pixels read beyond the window, so the correlation has room at the window's edge


class PlacementLog:
    """Placement records, appended one JSON line each and never rewritten."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> dict:
        record = {"id": uuid.uuid4().hex[:12], "at": time.time(), **record}
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self._lock, open(self.path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except ValueError:  # a line cut short by a crash: skipped, the rest still read
                    continue
        return out


def read_region(store, frame: dict, key: str, rows: slice, cols: slice) -> np.ndarray:
    if frame["kind"] == "map":
        return np.asarray(store.read_level(key, 0, rows, cols), dtype=np.float64)
    return np.asarray(image_node(store, key)[rows, cols], dtype=np.float64)


def place(store, log: PlacementLog, image: str, parent: str, centre_xy_m: list[float] | None = None,
          window_m: float = DEFAULT_WINDOW_M, rotation_deg: float = 0.0) -> dict[str, Any]:
    """Place ``image`` on ``parent`` and log the result; returns the record (status "placed" or "unplaced")."""
    child_frame = frame_of(store, image)
    if child_frame["kind"] != "image":
        raise FrameError(f"{image} is a stitched map; place_image places a single image on a map or a coarser image")
    parent_frame = frame_of(store, parent)
    child_px, parent_px = child_frame["pixel_size_m"], parent_frame["pixel_size_m"]
    if not child_px < parent_px:
        raise FrameError(f"{image} has {child_px:.3g} m pixels, not finer than {parent}'s {parent_px:.3g} m: nothing to place it in")
    centre = [float(v) for v in (centre_xy_m or child_frame["centre_xy_m"])]
    if not (len(centre) == 2 and all(math.isfinite(v) for v in centre)):
        raise FrameError("centre_xy_m must be [x, y] in metres")
    row, col = pixel_of(parent_frame, *centre)
    half = np.array(child_frame["shape"], dtype=np.float64) * child_px / parent_px / 2.0
    window_px = float(window_m) / parent_px
    rows_total, cols_total = parent_frame["shape"]
    r0 = int(max(0, math.floor(row - half[0] - window_px - EDGE_PX)))
    r1 = int(min(rows_total, math.ceil(row + half[0] + window_px + EDGE_PX) + 1))
    c0 = int(max(0, math.floor(col - half[1] - window_px - EDGE_PX)))
    c1 = int(min(cols_total, math.ceil(col + half[1] + window_px + EDGE_PX) + 1))
    record: dict[str, Any] = {
        "child": image, "parent": parent, "parent_kind": parent_frame["kind"], "predicted_xy_m": centre, "window_m": float(window_m),
        "child_pixel_m": child_px, "parent_pixel_m": parent_px, **child_frame.get("sample", {}),
    }
    if parent_frame["kind"] == "map":
        record["parent_version"] = parent_frame["metadata"].get("version") or parent
        record["parent_run"] = parent_frame["metadata"].get("run")
    if r1 - r0 < 3 or c1 - c0 < 3:
        result = {"status": "unplaced", "reason": f"{centre} is outside {parent}"}
        return log.append({**record, **result})
    child = np.asarray(image_node(store, image).read(), dtype=np.float64)
    region = read_region(store, parent_frame, parent, slice(r0, r1), slice(c0, c1))
    placed = register_child(
        child, child_px, region, parent_px, (row - r0, col - c0), window_px, rotation_rad=math.radians(rotation_deg),
        parent_tile={"key": parent, "version": record.get("parent_version", parent)},
    ).to_dict()
    result = {k: placed[k] for k in ("status", "reason", "score", "ambiguity", "scale", "rotation_rad", "uncertainty_px", "method", "method_version", "inputs_sha256")}
    if placed["status"] == "placed":
        in_parent = [placed["translation_px"][0] + r0, placed["translation_px"][1] + c0]
        result["parent_px"] = [round(v, 4) for v in in_parent]
        result["centre_xy_m"] = list(stage_of(parent_frame, *in_parent))
        result["shift_from_prediction_m"] = [result["centre_xy_m"][0] - centre[0], result["centre_xy_m"][1] - centre[1]]
    return log.append({**record, **result})


def register(mcp, store, log: PlacementLog) -> None:
    @mcp.tool()
    def place_image(image: str, parent: str, centre_xy_m: list[float] | None = None, window_m: float = DEFAULT_WINDOW_M, rotation_deg: float = 0.0) -> dict[str, Any]:
        """Place a finer image on the coarser one it was taken inside (a stitched map, or a single image), and keep the result.

        image: the finer image's key. parent: the map's or the coarser image's key.
        centre_xy_m: where the image's centre should be, [x, y] metres; leave it out to use the stage position the
        image records (aimed by image shift? give the point the beam was shifted to: images do not record the shift).
        window_m: how far from there to search (default 400 nm, a stage's error; a shift-aimed image needs much less).
        rotation_deg: the image's scan rotation relative to the parent's, if they differ.
        Returns the record: status "placed" (centre_xy_m and parent_px where it really is, score) or "unplaced" with the reason.
        Never a guess: an image matching more than one place is unplaced.
        """
        try:
            return place(store, log, image, parent, centre_xy_m, window_m, rotation_deg)
        except (FrameError, KeyError) as exc:
            return {"error": {"code": "cannot_place", "message": str(exc)}}

    @mcp.tool()
    def list_placements(parent: str | None = None, image: str | None = None, status: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Placements made by place_image, newest first: of images on ``parent``, of ``image``, or all; ``status`` "placed"
        or "unplaced". Returns {"total", "placements": [at most limit]}."""
        found = [r for r in log.read() if (parent is None or r.get("parent") == parent) and (image is None or r.get("child") == image)
                 and (status is None or r.get("status") == status)]
        found.reverse()
        return {"total": len(found), "placements": found[: max(0, int(limit))]}
