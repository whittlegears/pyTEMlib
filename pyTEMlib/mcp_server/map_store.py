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
(paths). Local paths are accepted only under the store's ``local_roots``
(default: the working directory), so a client cannot read, or write a map
next to, an arbitrary HDF5 file on the host. This module never imports
asyncroscopy.

Re-stitching a run replaces its map's pixels and transform but keeps the
map's points of interest: they are stage coordinates, so they stay valid
after the tiles are re-registered.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from pyTEMlib.image_tools.mosaic import Placement, Tile

AXES = "rows=+y, columns=+x; stage coordinates in metres"


@dataclass
class RunData:
    run: str
    tiles: list[Tile]
    plan: dict


@dataclass
class TileInfo:
    """A tile's placement metadata and image shape, without its pixels."""

    index: int
    stage_xy_m: tuple[float, float]
    pixel_size_m: float
    shape: tuple[int, ...]
    attrs: dict
    name: str


@dataclass
class RunInfo:
    run: str
    tiles: list[TileInfo]
    plan: dict


class LocalPathRefused(ValueError):
    """A local file path outside the store's ``local_roots``."""


# Serialises read-modify-write of a map's points of interest (and re-writes of
# a whole map, which carry the points over) within this
# process. It does NOT protect against several server processes writing the
# same map; that would need a lock in the store (e.g. Tiled) itself.
_POI_LOCK = threading.Lock()


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


def _tile_info(index: int, name: str, shape: tuple[int, ...], attrs: dict) -> TileInfo:
    attrs = {k: _decode(v) for k, v in attrs.items()}
    stage = attrs.get("stage_reported_m") or attrs.get("commanded_xy_m")
    if stage is None or "pixel_size_m" not in attrs:
        raise ValueError(f"tile {index} has no stage_reported_m/pixel_size_m attributes; cannot place it")
    if len(shape) < 2:
        raise ValueError(f"tile {index} is not an image (shape {tuple(shape)})")
    return TileInfo(int(attrs.get("index", index)), (float(stage[0]), float(stage[1])), float(attrs["pixel_size_m"]), tuple(int(v) for v in shape), attrs, name)


def _tile_from(info: TileInfo, image: np.ndarray) -> Tile:
    return Tile(np.asarray(image), info.stage_xy_m, info.pixel_size_m, info.index, info.attrs)


class MapStore:
    """Where runs are read from and maps are written to."""

    def __init__(self, tiled_uri: str | None = None, api_key: str | None = None, local_roots: list[str | os.PathLike] | None = None):
        """``local_roots``: directories whose HDF5 files (and subdirectories') may be read and written as
        local paths; default the current working directory. Pass ``[]`` to accept no local paths."""
        self.tiled_uri = tiled_uri
        self.api_key = api_key or os.environ.get("ASYNCROSCOPY_TILED_API_KEY") or os.environ.get("TILED_API_KEY")
        self.local_roots = [Path(r).resolve() for r in ([Path.cwd()] if local_roots is None else local_roots)]
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

    def is_local(self, source: str) -> bool:
        """True for an existing .h5/.hdf5 file under one of ``local_roots``; False for anything else (a Tiled key).

        Raises LocalPathRefused for an existing HDF5 file outside the roots.
        """
        path = Path(source)
        if path.suffix.lower() not in {".h5", ".hdf5"} or not path.exists():
            return False
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in self.local_roots):
            return True
        allowed = ", ".join(str(r) for r in self.local_roots) or "none"
        raise LocalPathRefused(
            f"local file {source!r} is outside the directories this server may read and write ({allowed}); "
            "use a Tiled key, move the file under one of them, or start the server with --local-root <dir>"
        )

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
    def scan_run(self, run: str, box_m: list[float] | None = None) -> RunInfo:
        """Tile positions and shapes of a run, from metadata only (no pixels are read)."""
        infos: list[TileInfo] = []
        if self.is_local(run):
            with h5py.File(run, "r") as h5:
                plan = _decode(h5.attrs.get("plan", "{}")) or {}
                group = h5["tiles"]
                for k, name in enumerate(sorted(group)):
                    infos.append(_tile_info(k, name, group[name].shape, dict(group[name].attrs)))
        else:
            node = self.client[run]
            plan = _decode(dict(node.metadata).get("plan", "{}")) or {}
            group = node["tiles"]
            for k, name in enumerate(sorted(group)):
                child = group[name]
                infos.append(_tile_info(k, name, tuple(child.shape), dict(child.metadata)))
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            infos = [t for t in infos if x0 <= t.stage_xy_m[0] <= x1 and y0 <= t.stage_xy_m[1] <= y1]
        if not infos:
            raise ValueError(f"run {run!r} has no tiles" + (" in that box" if box_m else ""))
        return RunInfo(run, infos, plan if isinstance(plan, dict) else {})

    def open_run(self, run: str, box_m: list[float] | None = None) -> RunData:
        """Tiles of a run with their images (only tiles inside ``box_m`` are read). Slow for big runs: use in jobs."""
        info = self.scan_run(run, box_m)
        if self.is_local(run):
            with h5py.File(run, "r") as h5:
                group = h5["tiles"]
                tiles = [_tile_from(t, group[t.name][()]) for t in info.tiles]
        else:
            group = self.client[run]["tiles"]
            tiles = [_tile_from(t, group[t.name].read()) for t in info.tiles]
        return RunData(run, tiles, info.plan)

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
            "level_shapes": [list(level.shape[:2]) for level in levels],
            "tile_positions_px": placement.positions_px.round(3).tolist(),
            "matches_used": len(confident),
            "mean_match_confidence": float(np.mean(confident)) if confident else None,
            "pois": [],
            **(extra or {}),
        }
        local = self.is_local(run)
        # Under the POI lock, so an add_poi on the old map cannot slip in between reading its points and replacing it.
        with _POI_LOCK:
            if "pois" not in (extra or {}):
                metadata["pois"] = self._existing_pois(run, local)
            if local:
                path = self.local_map_path(run)
                with h5py.File(path, "w") as h5:
                    for k, level in enumerate(levels):
                        h5.create_dataset(f"level_{k}", data=level, chunks=True)
                    h5.attrs["metadata"] = json.dumps(metadata)
                return str(path)
            return self._write_tiled_checked(self.map_name(run), levels, metadata)

    def _existing_pois(self, run: str, local: bool) -> list[dict]:
        """Points of interest of the map previously stitched from ``run`` (empty if there is none)."""
        try:
            if local:
                path = self.local_map_path(run)
                if not path.exists():
                    return []
                with h5py.File(path, "r") as h5:
                    metadata = json.loads(h5.attrs["metadata"])
            else:
                name = self.map_name(run)
                if name not in self.client:
                    return []
                metadata = dict(self.client[name].metadata)
        except (OSError, KeyError, ValueError):  # unreadable or not a map: nothing to keep
            return []
        pois = _decode(metadata.get("pois", []))
        return list(pois) if isinstance(pois, list) else []

    def _write_tiled_checked(self, name: str, levels: list[np.ndarray], metadata: dict) -> str:
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
                meta = json.loads(h5.attrs["metadata"]) if "metadata" in h5.attrs else {}
        else:
            meta = dict(self.client[map_key].metadata)
        if meta.get("kind") != "stitched_map":
            raise ValueError(
                f"{map_key} is not a stitched map (a Mapper run is not one); stitch it with assemble_map "
                "and pass the key from its results (result_keys.map)."
            )
        return meta

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

    def _write_pois(self, map_key: str, pois: list[dict]) -> None:
        if self.is_local(map_key):
            with h5py.File(map_key, "a") as h5:
                metadata = json.loads(h5.attrs["metadata"])
                metadata["pois"] = pois
                h5.attrs["metadata"] = json.dumps(metadata)
            return
        node = self.client[map_key]  # fetched fresh, so the rest of the metadata is current
        metadata = dict(node.metadata)
        metadata["pois"] = pois
        node.replace_metadata(metadata=metadata) if hasattr(node, "replace_metadata") else node.update_metadata(metadata=metadata)

    def update_pois(self, map_key: str, change: Callable[[list[dict]], list[dict]]) -> list[dict]:
        """Apply ``change(current_pois) -> new_pois`` to a map's points of interest and store the result.

        The current list is re-read immediately before writing, under a
        process-wide lock, so concurrent add_poi / score_map calls in this
        server cannot lose each other's points. Several server processes
        writing the same map are not covered (see ``_POI_LOCK``).
        """
        with _POI_LOCK:
            pois = change(list(self.map_metadata(map_key).get("pois", [])))
            self._write_pois(map_key, pois)
            return pois

    def add_pois(self, map_key: str, new: list[dict]) -> list[dict]:
        """Append points of interest; returns the stored list."""
        return self.update_pois(map_key, lambda current: current + list(new))

    def set_pois(self, map_key: str, pois: list[dict]) -> None:
        """Replace all points of interest (use ``add_pois``/``update_pois`` to keep concurrent additions)."""
        self.update_pois(map_key, lambda _current: list(pois))


def level_shape_of(metadata: dict, level: int) -> tuple[int, int]:
    """(rows, cols) of pyramid ``level``: the stored ``level_shapes`` when the map has them.

    Maps written before ``level_shapes`` was stored fall back to computing it
    from ``canvas_shape`` with the pyramid writer's rule (rows round up,
    columns round down at each halving).
    """
    shapes = metadata.get("level_shapes")
    if shapes and 0 <= level < len(shapes):
        return int(shapes[level][0]), int(shapes[level][1])
    h, w = (int(v) for v in metadata["canvas_shape"])
    for _ in range(level):
        h, w = (h + 1) // 2, max(1, w // 2)
    return h, w


def stage_box_to_pixels(metadata: dict, box_m: list[float], level: int) -> tuple[slice, slice]:
    """Rows/cols of ``level`` covering the stage box [x_min, x_max, y_min, y_max] (clipped to the map)."""
    scale = 2**level
    p0 = metadata["pixel_size_m"]
    ox, oy = metadata["origin_xy_m"]
    h, w = level_shape_of(metadata, level)
    x0, x1, y0, y1 = box_m

    def index(value: float, origin: float, limit: int, round_up: bool) -> int:
        level0 = (value - origin) / p0 + 0.5  # level-0 pixel edge coordinate
        idx = np.ceil(level0 / scale) if round_up else np.floor(level0 / scale)
        return int(np.clip(idx, 0, limit))

    return slice(index(y0, oy, h, False), index(y1, oy, h, True)), slice(index(x0, ox, w, False), index(x1, ox, w, True))


def pixel_to_stage(metadata: dict, row: float, col: float, level: int = 0) -> tuple[float, float]:
    """Stage (x, y) of the centre of pixel (row, col) of ``level``.

    Level-k pixel (r, c) averages level-0 pixels [r*2^k, (r+1)*2^k) x [c*2^k, (c+1)*2^k),
    so this does not depend on the level's shape; ``level_shape_of`` gives that shape.
    """
    scale = 2**level
    p0 = metadata["pixel_size_m"]
    ox, oy = metadata["origin_xy_m"]
    return ox + (col * scale + (scale - 1) / 2.0) * p0, oy + (row * scale + (scale - 1) / 2.0) * p0
