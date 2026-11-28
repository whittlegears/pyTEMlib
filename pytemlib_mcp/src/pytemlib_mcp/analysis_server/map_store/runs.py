"""Reading a run's tiles: run layouts, lazy tiles and the chunked, cached tile reader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from .store import MapStore

CHUNK_TILES = 32  # tiles read per request
CACHE_BYTES = 512 * 2**20  # tile images kept in memory while stitching
V1_EXPORT_MAX_TILES = 20_000  # largest layout 1 run exported from Tiled in one request


@dataclass
class RunInfo:
    """A run's tiles as arrays: one row per tile (only tiles that were written)."""

    run: str
    plan: dict
    layout: int
    index: np.ndarray  # tile index in the run, one per tile
    stage_xy_m: np.ndarray  # two values per tile
    shapes: np.ndarray  # two values per tile
    pixel_sizes_m: np.ndarray  # one value per tile
    acquired_s: np.ndarray | None  # one value per tile, or None when unknown
    names: list = field(default_factory=list)  # dataset names in layout 1
    row_col: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.index)

    def select(self, keep: np.ndarray) -> "RunInfo":
        return RunInfo(
            self.run, self.plan, self.layout, self.index[keep], self.stage_xy_m[keep], self.shapes[keep], self.pixel_sizes_m[keep],
            None if self.acquired_s is None else self.acquired_s[keep],
            [n for n, k in zip(self.names, keep) if k] if self.names else [],
            None if self.row_col is None else self.row_col[keep],
        )


class LazyTile:
    """One tile, made on demand; its image is read from the run when first needed (see ``TileSource``)."""

    __slots__ = ("stage_xy_m", "pixel_size_m", "index", "acquired_s", "shape", "attrs", "_source", "_position")

    def __init__(self, info: RunInfo, source: "TileSource", position: int):
        self.stage_xy_m = (float(info.stage_xy_m[position, 0]), float(info.stage_xy_m[position, 1]))
        self.pixel_size_m = float(info.pixel_sizes_m[position])
        self.index = int(info.index[position])
        self.acquired_s = None if info.acquired_s is None else float(info.acquired_s[position])
        self.shape = (int(info.shapes[position, 0]), int(info.shapes[position, 1]))
        self.attrs = {} if info.row_col is None else {"row": int(info.row_col[position, 0]), "col": int(info.row_col[position, 1])}
        self._source = source
        self._position = position

    @property
    def image(self) -> np.ndarray:
        return self._source.image(self._position)


class LazyTiles:
    """The tiles of a run as a sequence: arrays for placement, images on demand."""

    def __init__(self, info: RunInfo, source: "TileSource"):
        self.info, self.source = info, source

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, position: int) -> LazyTile:
        if position < 0:
            position += len(self.info)
        if not 0 <= position < len(self.info):
            raise IndexError(position)
        return LazyTile(self.info, self.source, position)

    def __iter__(self):
        return (self[k] for k in range(len(self)))

    def arrays(self) -> dict:
        return {"stage_xy_m": self.info.stage_xy_m, "shapes": self.info.shapes, "pixel_sizes_m": self.info.pixel_sizes_m, "acquired_s": self.info.acquired_s}


@dataclass
class RunData:
    run: str
    tiles: LazyTiles
    plan: dict
    source: "TileSource"

    def close(self) -> None:
        self.source.close()


class LocalPathRefused(ValueError):
    """A local file path outside the store's ``local_roots``."""


class _Locks:
    """One re-entrant lock per local file or Tiled key, made on first use.

    A map's (or point store's) lock serialises changes to its points of interest within this
    process, and every open of a local map file: HDF5 refuses a file opened for writing in one
    thread while another has it open. Work on different maps never waits on each other.
    Several server processes changing the same map's points are not covered.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    def __call__(self, name: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(name)
            if lock is None:
                lock = self._locks[name] = threading.RLock()
            return lock


# Local files are locked by resolved path across every store in the process, since two stores may open one file.
_FILE_LOCKS = _Locks()


def _file_lock(path: str | os.PathLike) -> threading.RLock:
    return _FILE_LOCKS(str(Path(path).resolve()))


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


def _batch_points(item) -> list[dict]:
    """The points of one stored batch: a string dataset of JSON, or (stores written before) a group whose
    ``pois`` attribute holds the JSON."""
    if isinstance(item, h5py.Dataset):
        value = item[()]
        return json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
    return json.loads(item.attrs["pois"])


def _layout1_info(run: str, plan: dict, names: list[str], shapes: list, attrs_list: list[dict]) -> RunInfo:
    rows = []
    for k, (name, shape, attrs) in enumerate(zip(names, shapes, attrs_list)):
        attrs = {key: _decode(v) for key, v in attrs.items()}
        stage = attrs.get("stage_reported_m") or attrs.get("commanded_xy_m")
        if stage is None or "pixel_size_m" not in attrs:
            raise ValueError(f"tile {k} has no stage_reported_m/pixel_size_m attributes; cannot place it")
        if len(shape) < 2:
            raise ValueError(f"tile {k} is not an image (shape {tuple(shape)})")
        rows.append((int(attrs.get("index", k)), float(stage[0]), float(stage[1]), int(shape[0]), int(shape[1]), float(attrs["pixel_size_m"]), attrs.get("acquired")))
    times = [r[6] for r in rows]
    return RunInfo(
        run, plan, 1,
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([[r[1], r[2]] for r in rows], dtype=np.float64).reshape(-1, 2),
        np.array([[r[3], r[4]] for r in rows], dtype=np.int64).reshape(-1, 2),
        np.array([r[5] for r in rows], dtype=np.float64),
        None if any(t is None for t in times) else np.array(times, dtype=np.float64),
        list(names),
    )


def _layout2_info(run: str, plan: dict, shape: tuple[int, ...], tile_attrs: dict, arrays: dict) -> RunInfo:
    attrs = {k: _decode(v) for k, v in tile_attrs.items()}
    if "pixel_size_m" not in attrs:
        raise ValueError("the run's tiles array has no pixel_size_m attribute; cannot place its tiles")
    n = int(shape[0])
    written = np.asarray(arrays.get("written", np.ones(n, bool)), dtype=bool)
    index = np.flatnonzero(written)
    stage = np.asarray(arrays["stage_reported_m"], dtype=np.float64)[index, :2]
    acquired = arrays.get("acquired_s")
    row_col = arrays.get("row_col")
    return RunInfo(
        run, plan, 2, index, stage,
        np.tile(np.array(shape[1:3], dtype=np.int64), (len(index), 1)),
        np.full(len(index), float(attrs["pixel_size_m"])),
        None if acquired is None else np.asarray(acquired, dtype=np.float64)[index],
        [], None if row_col is None else np.asarray(row_col)[index],
    )


class TileSource:
    """Reads a run's tile images in chunks of consecutive tiles, keeping at most ``CACHE_BYTES`` of them.

    ``reads`` counts the chunk reads, so a caller can see how often each chunk was read.
    """

    def __init__(self, store: "MapStore", run: str, info: RunInfo):
        self.store, self.run, self.info = store, run, info
        self.local = store.is_local(run)
        self._cache: OrderedDict[int, list[np.ndarray]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._export: str | None = None
        self.reads = 0

    def image(self, position: int) -> np.ndarray:
        chunk = position // CHUNK_TILES
        with self._lock:
            images = self._cache.get(chunk)
            if images is not None:
                self._cache.move_to_end(chunk)
                return images[position % CHUNK_TILES]
        positions = range(chunk * CHUNK_TILES, min((chunk + 1) * CHUNK_TILES, len(self.info)))
        images = self._read(positions)
        size = sum(a.nbytes for a in images)
        with self._lock:
            self.reads += 1
            self._cache[chunk] = images
            self._bytes += size
            while self._bytes > CACHE_BYTES and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._bytes -= sum(a.nbytes for a in old)
        return images[position % CHUNK_TILES]

    def _read(self, positions: range) -> list[np.ndarray]:
        info = self.info
        if info.layout >= 2:
            indices = info.index[positions.start : positions.stop]
            first, last = int(indices[0]), int(indices[-1])
            if self.local:
                with h5py.File(self.run, "r") as h5:
                    block = h5["tiles"][first : last + 1]
            else:
                block = np.asarray(self.store.client[self.run]["tiles"][first : last + 1])
            return [np.asarray(block[int(i) - first]) for i in indices]
        names = info.names[positions.start : positions.stop]
        if self.local:
            with h5py.File(self.run, "r") as h5:
                return [h5["tiles"][name][()] for name in names]
        if len(info) <= V1_EXPORT_MAX_TILES:
            if self._export is None:
                self._export = self._export_v1()
            if self._export:
                with h5py.File(self._export, "r") as h5:
                    return [h5[name][()] for name in names]
        group = self.store.client[self.run]["tiles"]
        return [np.asarray(group[name].read()) for name in names]

    def _export_v1(self) -> str:
        """One HDF5 export of a layout-1 run's tiles group (about 13x faster than a request per tile); "" if refused."""
        handle, path = tempfile.mkstemp(suffix=".h5")
        os.close(handle)
        try:
            self.store.client[self.run]["tiles"].export(path, format="application/x-hdf5")
            return path
        except Exception:  # this Tiled cannot export HDF5, so read tile by tile
            os.unlink(path)
            return ""

    def dataset(self, name: str) -> np.ndarray | None:
        """A whole per-tile array beside the run's tiles (e.g. ``spectrum_keys``), or None if the run has none."""
        if self.local:
            with h5py.File(self.run, "r") as h5:
                return h5[name][()] if name in h5 else None
        node = self.store.client[self.run]
        return np.asarray(node[name].read()) if name in node else None

    def attribute(self, name: str) -> Any:
        """One of the run's own attributes (local) or metadata (Tiled), decoded; None if the run has none."""
        if self.local:
            with h5py.File(self.run, "r") as h5:
                value = h5.attrs.get(name)
        else:
            value = dict(self.store.client[self.run].metadata).get(name)
        return None if value is None else _decode(value)

    def close(self) -> None:
        self._cache.clear()
        if self._export:
            try:
                os.unlink(self._export)
            except OSError:
                pass
            self._export = ""
