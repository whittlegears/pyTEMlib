"""
Read map runs and write stitched maps, in Tiled or in local HDF5 files.

A run (from asyncroscopy's Mapper, or any program that writes the same
layout) is one HDF5 file with ``tiles/tile_NNNNN`` image datasets whose
attributes carry ``stage_reported_m`` (x, y in metres, tile centre) and
``pixel_size_m``. The stitched map is stored next to it as ``<run>_map``:
``level_0`` .. ``level_N`` arrays (each half the size of the one before)
plus metadata holding the pixel-to-stage transform and the points of
interest.

The same functions work against a Tiled server (keys) or the filesystem
(paths). This module never imports asyncroscopy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pyTEMlib.image_tools.mosaic import Placement, Tile

AXES = "rows=+y, columns=+x; stage coordinates in metres"


@dataclass
class RunData:
    run: str
    tiles: list[Tile]
    plan: dict


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tile_from(index: int, image: np.ndarray, attrs: dict) -> Tile:
    attrs = {k: _decode(v) for k, v in attrs.items()}
    stage = attrs.get("stage_reported_m") or attrs.get("commanded_xy_m")
    if stage is None or "pixel_size_m" not in attrs:
        raise ValueError(f"tile {index} has no stage_reported_m/pixel_size_m attributes; cannot place it")
    return Tile(np.asarray(image), (float(stage[0]), float(stage[1])), float(attrs["pixel_size_m"]), int(attrs.get("index", index)), attrs)


class MapStore:
    """Where runs are read from and maps are written to."""

    def __init__(self, tiled_uri: str | None = None, api_key: str | None = None):
        self.tiled_uri = tiled_uri
        self.api_key = api_key or os.environ.get("ASYNCROSCOPY_TILED_API_KEY") or os.environ.get("TILED_API_KEY")
        self._client = None

    # -------------------------------------------------------------- helpers
    @property
    def client(self):
        if self._client is None:
            if not self.tiled_uri:
                raise FileNotFoundError("not a local file, and no Tiled server is configured to look it up as a key")
            from tiled.client import from_uri

            self._client = from_uri(self.tiled_uri, api_key=self.api_key)
        return self._client

    @staticmethod
    def is_local(source: str) -> bool:
        return Path(source).suffix.lower() in {".h5", ".hdf5"} and Path(source).exists()

    @staticmethod
    def map_name(run: str) -> str:
        stem = Path(run).name
        for suffix in (".h5", ".hdf5"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        return f"{stem}_map"

    def local_map_path(self, run: str) -> Path:
        return Path(run).with_name(self.map_name(run) + ".h5")

    # -------------------------------------------------------------- runs
    def open_run(self, run: str, box_m: list[float] | None = None) -> RunData:
        tiles: list[Tile] = []
        if self.is_local(run):
            with h5py.File(run, "r") as h5:
                plan = _decode(h5.attrs.get("plan", "{}")) or {}
                group = h5["tiles"]
                for k, name in enumerate(sorted(group)):
                    tiles.append(_tile_from(k, group[name][()], dict(group[name].attrs)))
        else:
            node = self.client[run]
            plan = _decode(dict(node.metadata).get("plan", "{}")) or {}
            group = node["tiles"]
            for k, name in enumerate(sorted(group)):
                child = group[name]
                tiles.append(_tile_from(k, child.read(), dict(child.metadata)))
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            tiles = [t for t in tiles if x0 <= t.stage_xy_m[0] <= x1 and y0 <= t.stage_xy_m[1] <= y1]
        if not tiles:
            raise ValueError(f"run {run!r} has no tiles" + (" in that box" if box_m else ""))
        return RunData(run, tiles, plan if isinstance(plan, dict) else {})

    # -------------------------------------------------------------- maps
    def write_map(self, run: str, levels: list[np.ndarray], placement: Placement, extra: dict | None = None) -> str:
        """Store pyramid levels with the pixel-to-stage transform; return the map's key or path."""
        confident = [m.confidence for m in placement.matches if m.confidence >= 0.3]
        metadata = {
            "kind": "stitched_map",
            "run": run,
            "axes": AXES,
            "levels": len(levels),
            "downsample": [2**k for k in range(len(levels))],
            "pixel_size_m": placement.pixel_size_m,
            "origin_xy_m": list(placement.origin_xy_m),
            "canvas_shape": list(placement.canvas_shape),
            "tile_positions_px": placement.positions_px.round(3).tolist(),
            "matches_used": len(confident),
            "mean_match_confidence": float(np.mean(confident)) if confident else None,
            "pois": [],
            **(extra or {}),
        }
        if self.is_local(run):
            path = self.local_map_path(run)
            with h5py.File(path, "w") as h5:
                for k, level in enumerate(levels):
                    h5.create_dataset(f"level_{k}", data=level, chunks=True)
                h5.attrs["metadata"] = json.dumps(metadata)
            return str(path)
        name = self.map_name(run)
        try:
            return self._write_tiled(name, levels, metadata)
        except Exception as exc:
            if "401" in str(exc) or "permission" in str(exc).lower():
                raise PermissionError(
                    f"Tiled at {self.tiled_uri} refused to store the map: {exc}. Start this server with the Tiled API key "
                    "(--api-key, or $ASYNCROSCOPY_TILED_API_KEY; an asyncroscopy stack uses 'secret' unless that variable is set)."
                ) from exc
            raise

    def _write_tiled(self, name: str, levels: list[np.ndarray], metadata: dict) -> str:
        if name in self.client:
            self.client[name].delete_tree() if hasattr(self.client[name], "delete_tree") else self.client.delete(name)
        container = self.client.create_container(name, metadata=metadata)
        for k, level in enumerate(levels):
            container.write_array(level, key=f"level_{k}", metadata={"downsample": 2**k})
        return name

    def map_metadata(self, map_key: str) -> dict:
        if self.is_local(map_key):
            with h5py.File(map_key, "r") as h5:
                return json.loads(h5.attrs["metadata"])
        return dict(self.client[map_key].metadata)

    def read_level(self, map_key: str, level: int, rows: slice = slice(None), cols: slice = slice(None)) -> np.ndarray:
        if self.is_local(map_key):
            with h5py.File(map_key, "r") as h5:
                return h5[f"level_{level}"][rows, cols]
        return np.asarray(self.client[map_key][f"level_{level}"][rows, cols])

    def level_shape(self, map_key: str, level: int) -> tuple[int, int]:
        if self.is_local(map_key):
            with h5py.File(map_key, "r") as h5:
                return tuple(h5[f"level_{level}"].shape)
        return tuple(self.client[map_key][f"level_{level}"].shape)

    def set_pois(self, map_key: str, pois: list[dict]) -> None:
        if self.is_local(map_key):
            with h5py.File(map_key, "a") as h5:
                metadata = json.loads(h5.attrs["metadata"])
                metadata["pois"] = pois
                h5.attrs["metadata"] = json.dumps(metadata)
            return
        node = self.client[map_key]
        metadata = dict(node.metadata)
        metadata["pois"] = pois
        node.replace_metadata(metadata=metadata) if hasattr(node, "replace_metadata") else node.update_metadata(metadata=metadata)


def stage_box_to_pixels(metadata: dict, box_m: list[float], level: int) -> tuple[slice, slice]:
    """Rows/cols of ``level`` covering the stage box [x_min, x_max, y_min, y_max] (clipped to the map)."""
    scale = 2**level
    p0 = metadata["pixel_size_m"]
    ox, oy = metadata["origin_xy_m"]
    h, w = (int(np.ceil(v / scale)) for v in metadata["canvas_shape"])
    x0, x1, y0, y1 = box_m

    def index(value: float, origin: float, limit: int, round_up: bool) -> int:
        level0 = (value - origin) / p0 + 0.5  # level-0 pixel edge coordinate
        idx = np.ceil(level0 / scale) if round_up else np.floor(level0 / scale)
        return int(np.clip(idx, 0, limit))

    return slice(index(y0, oy, h, False), index(y1, oy, h, True)), slice(index(x0, ox, w, False), index(x1, ox, w, True))


def pixel_to_stage(metadata: dict, row: float, col: float, level: int = 0) -> tuple[float, float]:
    """Stage (x, y) of the centre of pixel (row, col) of ``level``."""
    scale = 2**level
    p0 = metadata["pixel_size_m"]
    ox, oy = metadata["origin_xy_m"]
    return ox + (col * scale + (scale - 1) / 2.0) * p0, oy + (row * scale + (scale - 1) / 2.0) * p0
