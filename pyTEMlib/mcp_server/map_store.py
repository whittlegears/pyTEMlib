"""
Read map runs and write stitched maps, in Tiled or in local HDF5 files.

A run (from asyncroscopy's Mapper, or any program that writes the same
layout) is one HDF5 file. Two layouts are read:

* layout 2 (``run_layout`` attribute = 2), for maps of any size: ``tiles`` is
  one chunked array [n, rows, cols] (one chunk per tile), with per-tile arrays
  ``written`` [n], ``stage_reported_m`` [n, >=2], ``acquired_s`` [n] and
  ``row_col`` [n, 2]; ``tiles`` carries ``pixel_size_m``. A run of a million
  tiles is a handful of datasets, so Tiled lists it at once and a block of
  consecutive tiles is one request.
* layout 1 (no attribute): ``tiles/tile_NNNNN`` datasets whose attributes
  carry ``stage_reported_m`` and ``pixel_size_m``. Fine for small runs.

Tiles are read lazily in chunks of consecutive indices through a cache of
bounded size, so stitching never holds the whole run in memory.

The stitched map is stored next to the run as ``<run>_map``: ``level_0`` ..
``level_N`` chunked arrays (each half the size of the one before), written
block by block as the stitcher makes them, plus metadata holding the
pixel-to-stage transform and the stitching quality. Its points of interest
live in a ``poi`` layer beside the levels, in batches (one per add), so
adding points costs the size of the batch, not of every point so far.

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
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import numpy as np

AXES = "rows=+y, columns=+x; stage coordinates in metres"
CHUNK_TILES = 32  # tiles read per request
CACHE_BYTES = 512 * 2**20  # tile images kept in memory while stitching
V1_EXPORT_MAX_TILES = 20_000  # a layout-1 run from Tiled is exported in one request up to this size
POI_BATCH = "batch_"


@dataclass
class TileInfo:
    """A tile's placement metadata and image shape, without its pixels."""

    index: int
    stage_xy_m: tuple[float, float]
    pixel_size_m: float
    shape: tuple[int, ...]
    attrs: dict
    name: str | int
    acquired_s: float | None = None


@dataclass
class RunInfo:
    run: str
    tiles: list[TileInfo]
    plan: dict
    layout: int = 1


class LazyTile:
    """A tile whose image is read from its run when first needed (see ``TileSource``)."""

    def __init__(self, info: TileInfo, source: "TileSource", position: int):
        self.stage_xy_m = info.stage_xy_m
        self.pixel_size_m = info.pixel_size_m
        self.index = info.index
        self.attrs = info.attrs
        self.acquired_s = info.acquired_s
        self.shape = tuple(info.shape[:2])
        self._source = source
        self._position = position

    @property
    def image(self) -> np.ndarray:
        return self._source.image(self._position)


@dataclass
class RunData:
    run: str
    tiles: Sequence[LazyTile]
    plan: dict
    source: "TileSource"

    def close(self) -> None:
        self.source.close()


class LocalPathRefused(ValueError):
    """A local file path outside the store's ``local_roots``."""


# Serialises replacing a map's points of interest within this process (adding points is a new
# batch and needs no lock), and every open of a local map file: HDF5 refuses a file opened for
# writing in one thread while another has it open. Several server processes replacing the same
# map's points are not covered; that would need a lock in the store itself.
_POI_LOCK = threading.RLock()


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
    acquired = attrs.get("acquired")
    return TileInfo(int(attrs.get("index", index)), (float(stage[0]), float(stage[1])), float(attrs["pixel_size_m"]), tuple(int(v) for v in shape), attrs, name, float(acquired) if acquired is not None else None)


class TileSource:
    """Reads a run's tile images in chunks of consecutive tiles, keeping at most ``CACHE_BYTES`` of them."""

    def __init__(self, store: "MapStore", run: str, info: RunInfo):
        self.store, self.run, self.info = store, run, info
        self.local = store.is_local(run)
        self._cache: OrderedDict[int, list[np.ndarray]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._export: str | None = None

    def image(self, position: int) -> np.ndarray:
        chunk = position // CHUNK_TILES
        with self._lock:
            images = self._cache.get(chunk)
            if images is not None:
                self._cache.move_to_end(chunk)
                return images[position % CHUNK_TILES]
        tiles = self.info.tiles[chunk * CHUNK_TILES : (chunk + 1) * CHUNK_TILES]
        images = self._read(tiles)
        size = sum(a.nbytes for a in images)
        with self._lock:
            self._cache[chunk] = images
            self._bytes += size
            while self._bytes > CACHE_BYTES and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._bytes -= sum(a.nbytes for a in old)
        return images[position % CHUNK_TILES]

    def _read(self, tiles: list[TileInfo]) -> list[np.ndarray]:
        if self.info.layout >= 2:
            first, last = tiles[0].name, tiles[-1].name
            if self.local:
                with h5py.File(self.run, "r") as h5:
                    block = h5["tiles"][first : last + 1]
            else:
                block = np.asarray(self.store.client[self.run]["tiles"][first : last + 1])
            return [np.asarray(block[t.name - first]) for t in tiles]
        if self.local:
            with h5py.File(self.run, "r") as h5:
                return [h5["tiles"][t.name][()] for t in tiles]
        if len(self.info.tiles) <= V1_EXPORT_MAX_TILES:
            if self._export is None:
                self._export = self._export_v1()
            if self._export:
                with h5py.File(self._export, "r") as h5:
                    return [h5[t.name][()] for t in tiles]
        group = self.store.client[self.run]["tiles"]
        return [np.asarray(group[t.name].read()) for t in tiles]

    def _export_v1(self) -> str:
        """One HDF5 export of a layout-1 run's tiles group (about 13x faster than a request per tile); "" if refused."""
        handle, path = tempfile.mkstemp(suffix=".h5")
        os.close(handle)
        try:
            self.store.client[self.run]["tiles"].export(path, format="application/x-hdf5")
            return path
        except Exception:  # a Tiled that cannot export HDF5: read tile by tile
            os.unlink(path)
            return ""

    def dataset(self, name: str) -> np.ndarray | None:
        """A whole per-tile array of a layout-2 run (e.g. ``spectrum_keys``), or None if the run has none."""
        if self.info.layout < 2:
            return None
        if self.local:
            with h5py.File(self.run, "r") as h5:
                return h5[name][()] if name in h5 else None
        node = self.store.client[self.run]
        return np.asarray(node[name].read()) if name in node else None

    def close(self) -> None:
        self._cache.clear()
        if self._export:
            try:
                os.unlink(self._export)
            except OSError:
                pass
            self._export = ""


class _LocalMapWriter:
    def __init__(self, path: Path):
        self.path = path
        self.h5 = h5py.File(path, "w")

    def create(self, level, shape, dtype, block):
        self.h5.create_dataset(f"level_{level}", shape=shape, dtype=dtype, chunks=(min(block, shape[0]), min(block, shape[1])))

    def write(self, level, row0, col0, data):
        self.h5[f"level_{level}"][row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]] = data

    def finish(self, metadata: dict) -> str:
        self.h5.attrs["metadata"] = json.dumps(metadata)
        self.h5.close()
        return str(self.path)

    def abandon(self) -> None:
        self.h5.close()


class _TiledMapWriter:
    def __init__(self, container, name: str):
        self.container, self.name = container, name
        self.nodes: dict[int, Any] = {}
        self.block = 0

    def create(self, level, shape, dtype, block):
        from tiled.structures.array import ArrayStructure, BuiltinDtype
        from tiled.structures.core import StructureFamily
        from tiled.structures.data_source import DataSource

        def axis(n: int) -> tuple[int, ...]:
            full, rest = divmod(n, block)
            return (block,) * full + ((rest,) if rest else ())

        structure = ArrayStructure(shape=tuple(shape), chunks=(axis(shape[0]), axis(shape[1])), data_type=BuiltinDtype.from_numpy_dtype(np.dtype(dtype)))
        self.nodes[level] = self.container.new(
            StructureFamily.array,
            [DataSource(structure=structure, structure_family=StructureFamily.array)],
            key=f"level_{level}",
            metadata={"downsample": 2**level},
        )
        self.block = block

    def write(self, level, row0, col0, data):
        self.nodes[level].write_block(np.ascontiguousarray(data), block=(row0 // self.block, col0 // self.block))

    def finish(self, metadata: dict) -> str:
        _replace_metadata(self.container, metadata)
        return self.name

    def abandon(self) -> None:
        pass


def _replace_metadata(node, metadata: dict) -> None:
    node.replace_metadata(metadata=metadata) if hasattr(node, "replace_metadata") else node.update_metadata(metadata=metadata)


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
        if self.is_local(run):
            with h5py.File(run, "r") as h5:
                plan = _decode(h5.attrs.get("plan", "{}")) or {}
                if int(h5.attrs.get("run_layout", 1)) >= 2:
                    infos = _layout2_infos(h5["tiles"].shape, dict(h5["tiles"].attrs), {k: h5[k][()] for k in ("written", "stage_reported_m", "acquired_s", "row_col") if k in h5})
                    layout = 2
                else:
                    group = h5["tiles"]
                    infos = [_tile_info(k, name, group[name].shape, dict(group[name].attrs)) for k, name in enumerate(sorted(group))]
                    layout = 1
        else:
            node = self.client[run]
            meta = dict(node.metadata)
            plan = _decode(meta.get("plan", "{}")) or {}
            if int(_decode(meta.get("run_layout", 1)) or 1) >= 2:
                tiles = node["tiles"]
                arrays = {k: np.asarray(node[k].read()) for k in ("written", "stage_reported_m", "acquired_s", "row_col") if k in node}
                infos = _layout2_infos(tuple(tiles.shape), dict(tiles.metadata), arrays)
                layout = 2
            else:
                # One paged listing carries every tile's shape and metadata (not a request per tile).
                children = dict(node["tiles"].items())
                infos = [_tile_info(k, name, tuple(children[name].shape), dict(children[name].metadata)) for k, name in enumerate(sorted(children))]
                layout = 1
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            infos = [t for t in infos if x0 <= t.stage_xy_m[0] <= x1 and y0 <= t.stage_xy_m[1] <= y1]
        if not infos:
            raise ValueError(f"run {run!r} has no tiles" + (" in that box" if box_m else ""))
        return RunInfo(run, infos, plan if isinstance(plan, dict) else {}, layout)

    def open_run(self, run: str, box_m: list[float] | None = None) -> RunData:
        """The run's tiles, read lazily (see ``TileSource``); call ``close`` when done."""
        info = self.scan_run(run, box_m)
        source = TileSource(self, run, info)
        return RunData(run, [LazyTile(t, source, k) for k, t in enumerate(info.tiles)], info.plan, source)

    # -------------------------------------------------------------- maps
    def map_writer(self, run: str):
        """A LevelWriter for the map of ``run`` (see mosaic.assemble); pass it to ``write_map`` when assembled.

        An existing map of the run is replaced; read its points first with ``existing_pois`` to keep them.
        """
        if self.is_local(run):
            return _LocalMapWriter(self.local_map_path(run))
        name = self.map_name(run)
        if name in self.client:
            self.client[name].delete_tree() if hasattr(self.client[name], "delete_tree") else self.client.delete(name)
        container = self.client.create_container(name, metadata={"kind": "stitching", "run": run})
        container.create_container("poi", metadata={"kind": "poi_layer"})
        return _TiledMapWriter(container, name)

    def write_map(self, run: str, writer, metadata: dict, pois: list[dict]) -> str:
        """Finish a map written through ``map_writer``: store its metadata and points of interest."""
        metadata = {"kind": "stitched_map", "run": run, "axes": AXES, **metadata}
        try:
            key = writer.finish(metadata)
        except Exception as exc:
            if "401" in str(exc) or "permission" in str(exc).lower():
                raise PermissionError(
                    f"Tiled at {self.tiled_uri} refused to store the map: {exc}. Start this server with the Tiled API key "
                    "(--api-key, or $ASYNCROSCOPY_TILED_API_KEY; an asyncroscopy stack uses 'secret' unless that variable is set)."
                ) from exc
            raise
        if pois:
            self.add_pois(key, pois)
        return key

    def existing_pois(self, run: str) -> list[dict]:
        """Points of interest of the map previously stitched from ``run`` (empty if there is none)."""
        try:
            if self.is_local(run):
                key = str(self.local_map_path(run))
                if not Path(key).exists():
                    return []
            else:
                key = self.map_name(run)
                if key not in self.client:
                    return []
            return self.list_pois(key)
        except (OSError, KeyError, ValueError):  # unreadable or not a map: nothing to keep
            return []

    def map_metadata(self, map_key: str) -> dict:
        if self.is_local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
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
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                return h5[f"level_{level}"][rows, cols]
        return np.asarray(self.client[map_key][f"level_{level}"][rows, cols])

    def level_shape(self, map_key: str, level: int) -> tuple[int, int]:
        if self.is_local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                return tuple(h5[f"level_{level}"].shape)
        return tuple(self.client[map_key][f"level_{level}"].shape)

    # -------------------------------------------------------------- points of interest
    def list_pois(self, map_key: str, kind: str | None = None, box_m: list[float] | None = None) -> list[dict]:
        """Points of interest, optionally of one ``kind`` and inside a stage box [x_min, x_max, y_min, y_max]."""
        meta = self.map_metadata(map_key)
        pois = list(_decode(meta.get("pois", [])) or [])  # maps written before the layer kept them here
        for batch in self._batches(map_key):
            pois += batch
        if kind is not None:
            pois = [p for p in pois if p.get("kind") == kind]
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            pois = [p for p in pois if x0 <= p["x_m"] <= x1 and y0 <= p["y_m"] <= y1]
        return pois

    def poi_count(self, map_key: str) -> int:
        return len(self.list_pois(map_key))

    def _batches(self, map_key: str) -> list[list[dict]]:
        if self.is_local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                if "poi" not in h5:
                    return []
                return [json.loads(h5["poi"][name].attrs["pois"]) for name in sorted(h5["poi"])]
        node = self.client[map_key]
        if "poi" not in node:
            return []
        return [list(_decode(child.metadata.get("pois", [])) or []) for _, child in sorted(node["poi"].items(), key=lambda kv: kv[0])]

    def add_pois(self, map_key: str, new: list[dict]) -> list[dict]:
        """Store ``new`` points as one batch of the map's POI layer; returns them."""
        new = list(new)
        if not new:
            return new
        self.map_metadata(map_key)  # refuses a key that is not a map
        name = f"{POI_BATCH}{time.time_ns():020d}_{uuid.uuid4().hex[:6]}"
        if self.is_local(map_key):
            with _POI_LOCK, h5py.File(map_key, "a") as h5:
                h5.require_group("poi").create_group(name).attrs["pois"] = json.dumps(new)
            return new
        node = self.client[map_key]
        if "poi" not in node:
            node.create_container("poi", metadata={"kind": "poi_layer"})
        node["poi"].create_container(name, metadata={"pois": new})
        return new

    def update_pois(self, map_key: str, change: Callable[[list[dict]], list[dict]]) -> list[dict]:
        """Replace the map's points with ``change(current)`` (rewrites the layer; adding points needs no rewrite)."""
        with _POI_LOCK:
            pois = change(self.list_pois(map_key))
            if self.is_local(map_key):
                with _POI_LOCK, h5py.File(map_key, "a") as h5:
                    if "poi" in h5:
                        del h5["poi"]
                    meta = json.loads(h5.attrs["metadata"])
                    meta.pop("pois", None)
                    h5.attrs["metadata"] = json.dumps(meta)
            else:
                node = self.client[map_key]
                if "poi" in node:
                    layer = node["poi"]
                    for key in list(layer):
                        layer[key].delete_tree() if hasattr(layer[key], "delete_tree") else layer.delete(key)
                meta = dict(node.metadata)
                if "pois" in meta:
                    meta.pop("pois")
                    _replace_metadata(node, meta)
        self.add_pois(map_key, pois)
        return pois

    def set_pois(self, map_key: str, pois: list[dict]) -> None:
        """Replace all points of interest."""
        self.update_pois(map_key, lambda _current: list(pois))


def _layout2_infos(shape: tuple[int, ...], tile_attrs: dict, arrays: dict) -> list[TileInfo]:
    attrs = {k: _decode(v) for k, v in tile_attrs.items()}
    if "pixel_size_m" not in attrs:
        raise ValueError("the run's tiles array has no pixel_size_m attribute; cannot place its tiles")
    n = int(shape[0])
    written = np.asarray(arrays.get("written", np.ones(n, bool)), dtype=bool)
    stage = np.asarray(arrays["stage_reported_m"], dtype=np.float64)
    acquired = arrays.get("acquired_s")
    row_col = arrays.get("row_col")
    pixel = float(attrs["pixel_size_m"])
    infos = []
    for k in np.flatnonzero(written):
        extra = {"row": int(row_col[k][0]), "col": int(row_col[k][1])} if row_col is not None else {}
        infos.append(TileInfo(int(k), (float(stage[k, 0]), float(stage[k, 1])), pixel, tuple(int(v) for v in shape[1:3]), extra, int(k), float(acquired[k]) if acquired is not None else None))
    return infos


def level_shape_of(metadata: dict, level: int) -> tuple[int, int]:
    """(rows, cols) of pyramid ``level``: the stored ``level_shapes`` when the map has them.

    Maps written before ``level_shapes`` was stored fall back to computing it
    from ``canvas_shape`` with that pyramid writer's rule (rows round up,
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
