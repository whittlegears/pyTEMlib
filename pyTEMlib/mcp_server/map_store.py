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

A run is held as arrays (positions, shapes, times), not an object per tile,
and its tiles are read lazily in chunks of consecutive indices through a
cache of bounded size (``TileSource``, which counts its reads).

The stitched map is stored next to the run: ``level_0`` .. ``level_N``
chunked arrays (each half the size of the one before), written block by
block as the stitcher makes them, plus metadata holding the pixel-to-stage
transform and the stitching quality. Every stitch is a new version under its
own key (``<run>_map_<UTC time>_<hex>``, or that name with ``.h5`` beside a
local run) and no version is ever deleted or overwritten, so anything placed
on or cited from a map keeps it. ``list_maps`` lists a run's versions, newest
first; ``current_map`` is the newest. Maps stitched before versions existed
keep their key ``<run>_map``.

Points of interest belong to the run, not to a map version: they are in stage
metres, so one point store per run (``<run>_points``) serves every version of
its map. The store is built for a million points: EDS points are kept in a
grid of spatial cells (each with its count), so a query for a stage box reads
only the cells it touches and a count reads no points; the few other points
(scored regions, high magnification images, notes) are kept together as
"highlights". Adding points costs the points added; replacing them builds a
new layer and switches to it in one metadata update, and the replaced layer is
kept and listed in the store's ``layers``. Maps stitched before point stores
existed kept their points inside the map; those are read from there until the
run's store is made, which copies them in once (``migrated_from``).

The same functions work against a Tiled server (keys) or the filesystem
(paths). Local paths are accepted only under the store's ``local_roots``
(default: the working directory), so a client cannot read, or write a map
next to, an arbitrary HDF5 file on the host. This module never imports
asyncroscopy.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

AXES = "rows=+y, columns=+x; stage coordinates in metres"
CHUNK_TILES = 32  # tiles read per request
CACHE_BYTES = 512 * 2**20  # tile images kept in memory while stitching
V1_EXPORT_MAX_TILES = 20_000  # a layout-1 run from Tiled is exported in one request up to this size
POI_CELLS = 64  # the POI layer's grid is POI_CELLS x POI_CELLS cells over the map
CELL_KINDS = {"eds_point"}  # kinds kept in spatial cells; every other kind is a highlight
HIGHLIGHTS = "highlights"


@dataclass
class RunInfo:
    """A run's tiles as arrays: one row per tile (only tiles that were written)."""

    run: str
    plan: dict
    layout: int
    index: np.ndarray  # (n,) tile index in the run
    stage_xy_m: np.ndarray  # (n, 2)
    shapes: np.ndarray  # (n, 2)
    pixel_sizes_m: np.ndarray  # (n,)
    acquired_s: np.ndarray | None  # (n,) or None when unknown
    names: list = field(default_factory=list)  # layout 1: dataset names
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


# Serialises changes to a map's points of interest within this process, and every open of a
# local map file: HDF5 refuses a file opened for writing in one thread while another has it
# open. Several server processes changing the same map's points are not covered.
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
    """Writes a new map version to a temporary file; ``swap`` renames it to its own versioned name."""

    def __init__(self, path: Path):
        self.path = path
        self.name = str(path)
        self.tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        self.h5 = h5py.File(self.tmp, "w")

    def create(self, level, shape, dtype, block):
        self.h5.create_dataset(f"level_{level}", shape=shape, dtype=dtype, chunks=(min(block, shape[0]), min(block, shape[1])))

    def write(self, level, row0, col0, data):
        self.h5[f"level_{level}"][row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]] = data

    def finish(self, metadata: dict) -> str:
        self.h5.attrs["metadata"] = json.dumps(metadata)
        self.h5.close()
        return str(self.tmp)

    def swap(self) -> str:
        with _POI_LOCK:
            os.replace(self.tmp, self.path)
        return str(self.path)

    def abandon(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass
        try:
            self.tmp.unlink()
        except OSError:
            pass


class _TiledMapWriter:
    """Writes a new map version under its own key; earlier versions are left as they are."""

    def __init__(self, store: "MapStore", container, name: str):
        self.store, self.container, self.name = store, container, name
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
        self.metadata = metadata
        _replace_metadata(self.container, {**metadata, "kind": "stitching"})  # not listed as a map until swap
        return self.name

    def swap(self) -> str:
        _replace_metadata(self.container, self.metadata)
        return self.name

    def abandon(self) -> None:
        try:
            _delete_node(self.store.client[self.name])
        except Exception:
            pass


def _delete_node(node) -> None:
    """Delete a Tiled node and everything under it, data written through Tiled included. Used only to remove a
    map version whose writing failed, before anything could name it."""
    if hasattr(node, "delete_tree"):
        node.delete_tree()
    else:  # tiled 0.2: delete() removes this node; external_only=False also removes data Tiled wrote itself
        node.delete(recursive=True, external_only=False)


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

    def dataset_key(self, key: str) -> str | None:
        """The one name a dataset's notes are kept under: the resolved path of a local HDF5 file under the roots,
        or a key the Tiled server has; None when there is no such dataset. A Tiled error other than a missing
        key (server down, refused) raises, so it is reported rather than read as "no dataset"."""
        try:
            if self.is_local(key):
                return str(Path(key).resolve())
        except LocalPathRefused:
            return None
        if not self.tiled_uri:
            return None
        try:
            self.client[key]
        except KeyError:
            return None
        return key

    @staticmethod
    def _stem(run: str) -> str:
        stem = Path(run).name
        for suffix in (".h5", ".hdf5"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
        return stem

    @classmethod
    def map_name(cls, run: str) -> str:
        """The name every map version of ``run`` starts with (and the whole key of a map stitched before versions)."""
        return f"{cls._stem(run)}_map"

    @staticmethod
    def version_suffix() -> str:
        now = time.time()
        return f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now))}{int(now % 1 * 1000):03d}Z_{uuid.uuid4().hex[:4]}"

    def list_maps(self, run: str) -> list[dict]:
        """Every stitched version of ``run``'s map, newest first: {"map", "id", "stitched_at", "inputs_sha256", "tiles"}."""
        found: list[tuple[str, dict]] = []
        if self.is_local(run):
            name = self.map_name(run)
            for path in Path(run).parent.glob(f"{name}*.h5"):
                if path.name != f"{name}.h5" and not path.name.startswith(f"{name}_"):
                    continue
                try:
                    with _POI_LOCK, h5py.File(path, "r") as h5:
                        meta = json.loads(h5.attrs["metadata"])
                except (OSError, KeyError, ValueError):
                    continue
                if meta.get("kind") == "stitched_map" and Path(str(meta.get("run", ""))).resolve() == Path(run).resolve():
                    found.append((str(path), meta))
        else:
            from tiled.queries import Key

            for key, node in self.client.search(Key("kind") == "stitched_map").search(Key("run") == run).items():
                found.append((key, dict(node.metadata)))
        found.sort(key=lambda item: (float(_decode(item[1].get("stitched_at")) or 0.0), item[0]), reverse=True)
        return [
            {"map": key, "id": meta.get("id"), "stitched_at": _decode(meta.get("stitched_at")), "inputs_sha256": meta.get("inputs_sha256"), "tiles": _decode(meta.get("tiles"))}
            for key, meta in found
        ]

    def current_map(self, run: str) -> str | None:
        """Key or path of the newest map stitched from ``run``, or None."""
        maps = self.list_maps(run)
        return maps[0]["map"] if maps else None

    # -------------------------------------------------------------- runs
    def scan_run(self, run: str, box_m: list[float] | None = None) -> RunInfo:
        """Tile positions and shapes of a run, from metadata only (no pixels are read)."""
        if self.is_local(run):
            with h5py.File(run, "r") as h5:
                plan = _decode(h5.attrs.get("plan", "{}")) or {}
                if int(h5.attrs.get("run_layout", 1)) >= 2:
                    info = _layout2_info(run, plan, h5["tiles"].shape, dict(h5["tiles"].attrs), {k: h5[k][()] for k in ("written", "stage_reported_m", "acquired_s", "row_col") if k in h5})
                else:
                    group = h5["tiles"]
                    names = sorted(group)
                    info = _layout1_info(run, plan, names, [group[n].shape for n in names], [dict(group[n].attrs) for n in names])
        else:
            node = self.client[run]
            meta = dict(node.metadata)
            plan = _decode(meta.get("plan", "{}")) or {}
            if int(_decode(meta.get("run_layout", 1)) or 1) >= 2:
                tiles = node["tiles"]
                arrays = {k: np.asarray(node[k].read()) for k in ("written", "stage_reported_m", "acquired_s", "row_col") if k in node}
                info = _layout2_info(run, plan, tuple(tiles.shape), dict(tiles.metadata), arrays)
            else:
                # One paged listing carries every tile's shape and metadata (not a request per tile).
                children = dict(node["tiles"].items())
                names = sorted(children)
                info = _layout1_info(run, plan, names, [tuple(children[n].shape) for n in names], [dict(children[n].metadata) for n in names])
        info.plan = plan if isinstance(plan, dict) else {}
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            xy = info.stage_xy_m
            info = info.select((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
        if not len(info):
            raise ValueError(f"run {run!r} has no tiles" + (" in that box" if box_m else ""))
        return info

    def check_run(self, run: str) -> dict:
        """Cheap facts about a run (no per-tile data): layout and tile count; raises if it is not a run."""
        if self.is_local(run):
            with h5py.File(run, "r") as h5:
                if "tiles" not in h5:
                    raise ValueError(f"{run!r} is not a map run (it has no tiles)")
                layout = int(h5.attrs.get("run_layout", 1))
                count = int(h5["tiles"].shape[0]) if layout >= 2 else len(h5["tiles"])
            return {"layout": layout, "tiles": count}
        node = self.client[run]
        if "tiles" not in node:
            raise ValueError(f"{run!r} is not a map run (it has no tiles)")
        layout = int(_decode(dict(node.metadata).get("run_layout", 1)) or 1)
        count = int(node["tiles"].shape[0]) if layout >= 2 else len(node["tiles"])
        return {"layout": layout, "tiles": count}

    def open_run(self, run: str, box_m: list[float] | None = None) -> RunData:
        """The run's tiles, read lazily (see ``TileSource``); call ``close`` when done."""
        info = self.scan_run(run, box_m)
        source = TileSource(self, run, info)
        return RunData(run, LazyTiles(info, source), info.plan, source)

    # -------------------------------------------------------------- maps
    def map_writer(self, run: str):
        """A LevelWriter for a new map version of ``run`` (see mosaic.assemble); pass it to ``write_map`` when assembled.

        Earlier versions of the run's map are never touched.
        """
        name = f"{self.map_name(run)}_{self.version_suffix()}"
        if self.is_local(run):
            return _LocalMapWriter(Path(run).with_name(f"{name}.h5"))
        container = self.client.create_container(name, metadata={"kind": "stitching", "run": run})
        return _TiledMapWriter(self, container, name)

    def write_map(self, run: str, writer, metadata: dict, eds_points: list[dict] | None = None, previous: str | None = None) -> str:
        """Finish a map written through ``map_writer``: its metadata, the run's point store, then its final name.

        ``eds_points``: the run's EDS point spectra at this stitch's tile centres; those the run's store does not
        hold yet are added (by id), and those it holds keep the position of the stitch that first added them, so
        every version lists them at one place (a version's own tile positions are in its ``tile_positions_px``).
        ``previous``: the run's newest map before this one, whose points are copied into the run's point store if
        it was stitched before point stores existed. On Tiled the new version is listed only once all of this is
        done, so a failure here removes a version nothing can have named.
        """
        metadata = {"kind": "stitched_map", "run": run, "axes": AXES, "id": uuid.uuid4().hex, "parents": [run], "version": writer.name, **self.sample_of(run), **metadata}
        try:
            staged = writer.finish(metadata)
        except Exception as exc:
            writer.abandon()
            if "401" in str(exc) or "permission" in str(exc).lower():
                raise PermissionError(
                    f"Tiled at {self.tiled_uri} refused to store the map: {exc}. Start this server with the Tiled API key "
                    "(--api-key, or $ASYNCROSCOPY_TILED_API_KEY; an asyncroscopy stack uses 'secret' unless that variable is set)."
                ) from exc
            raise
        try:
            with _POI_LOCK:
                key, store = self._ensure_store(run, self.is_local(run), self._grid("", metadata), previous)
                if eds_points:
                    self._add_to(key, store, list(eds_points), dedupe=True)
        except Exception:
            writer.abandon()
            raise
        return writer.swap()

    def sample_of(self, key: str) -> dict:
        """{"sample_id", "sample_load_id"} a dataset recorded (asyncroscopy writes them on the file, and a stitched map
        copies its run's); {} when it recorded none or cannot be read. A key inside a file (``file.h5/image/HAADF``)
        is looked up on the file it is in."""
        try:
            found = self._own_sample(key)
            while not found and "/" in key.strip("/") and not self._local(key):
                key = key.strip("/").rsplit("/", 1)[0]
                found = self._own_sample(key)
        except Exception:  # a refused path or a Tiled error: the sample is unknown, which callers report
            return {}
        return found

    def _own_sample(self, key: str) -> dict:
        try:
            if self._local(key):
                with _POI_LOCK, h5py.File(key, "r") as h5:
                    meta = {k: h5.attrs[k] for k in h5.attrs}
            else:
                meta = dict(self.client[key].metadata)
        except (OSError, KeyError, ValueError, LocalPathRefused):
            return {}
        if "metadata" in meta:  # a local stitched map keeps its metadata as one JSON attribute
            meta = {**meta, **(_decode(meta["metadata"]) or {})}
        found = {k: meta.get(k) for k in ("sample_id", "sample_load_id")}
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in found.items() if isinstance(v, (str, bytes)) and v}

    def map_metadata(self, map_key: str) -> dict:
        if self.is_local(map_key) or map_key.endswith(".tmp"):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                meta = json.loads(h5.attrs["metadata"]) if "metadata" in h5.attrs else {}
        else:
            meta = dict(self.client[map_key].metadata)
        if meta.get("kind") != "stitched_map":
            try:
                stitched = self.current_map(map_key)
            except Exception:  # not a run either: no stitched map to name
                stitched = None
            if stitched:
                raise ValueError(f"{map_key} is a Mapper run, not a stitched map; its newest stitched map is {stitched!r} (list_maps lists every version): pass that.")
            raise ValueError(
                f"{map_key} is not a stitched map (a Mapper run is not one, and this one has not been stitched); "
                f"stitch it with assemble_map(run={map_key!r}) and pass the key from its results (result_keys.map)."
            )
        return meta

    def _local(self, map_key: str) -> bool:
        return map_key.endswith(".tmp") or self.is_local(map_key)

    # -------------------------------------------------------------- the run's point store
    def points_key(self, run: str, local: bool) -> str:
        """Where ``run``'s points of interest are kept: ``<run>_points.h5`` beside a local run, else the key ``<run>_points``."""
        name = f"{self._stem(run)}_points"
        return str(Path(run).with_name(f"{name}.h5")) if local else name

    def _store_of(self, map_key: str, meta: dict) -> str | None:
        """The run's point store for this map when it is ready; None for a map with no run or whose run has none yet."""
        run = meta.get("run")
        if not run:
            return None
        key = self.points_key(str(run), self._local(map_key))
        found = self._store_meta(key)
        if not found.get("ready"):
            return None
        self._check_store_run(key, found, str(run))
        return key

    def _store_meta(self, key: str) -> dict:
        """A point store's metadata; {} when there is no store (or a local file left without its metadata)."""
        if key.endswith(".h5"):
            if not Path(key).exists():
                return {}
            with _POI_LOCK, h5py.File(key, "r") as h5:
                return json.loads(h5.attrs["metadata"]) if "metadata" in h5.attrs else {}
        if key not in self.client:
            return {}
        return {k: _decode(v) for k, v in dict(self.client[key].metadata).items()}

    def _write_store_meta(self, key: str, meta: dict) -> None:
        if key.endswith(".h5"):
            with h5py.File(key, "a") as h5:
                h5.attrs["metadata"] = json.dumps(meta)
        else:
            _replace_metadata(self.client[key], meta)

    @staticmethod
    def _check_store_run(key: str, meta: dict, run: str) -> None:
        owner = str(meta.get("run") or "")
        same = Path(owner).resolve() == Path(run).resolve() if key.endswith(".h5") else owner == run
        if owner and not same:
            raise ValueError(
                f"the point store {key!r} belongs to run {owner!r}, not {run!r}: two runs share the name {Path(run).name!r}; "
                "rename one of them before adding points"
            )

    def _points_at(self, map_key: str) -> tuple[str, dict]:
        """Where to read a map's points: the run's store, or the map itself when its run has no store yet."""
        meta = self.map_metadata(map_key)
        store = self._store_of(map_key, meta)
        return (store, self._store_meta(store)) if store else (map_key, meta)

    def _ensure_points(self, map_key: str, legacy: str | None = "") -> tuple[str, dict]:
        """The run's point store for this map, made on first use (see ``_ensure_store``; ``legacy`` defaults to this
        map). A map with no run keeps its own points."""
        with _POI_LOCK:
            meta = self.map_metadata(map_key)
            if not meta.get("run"):
                return map_key, meta
            return self._ensure_store(str(meta["run"]), self._local(map_key), self._grid(map_key, meta), map_key if legacy == "" else legacy)

    def _ensure_store(self, run: str, local: bool, grid: dict, legacy: str | None) -> tuple[str, dict]:
        """``run``'s point store, made on first use with ``grid``. A new store copies in the points ``legacy`` kept
        inside itself before point stores existed, into a fresh layer, and is marked ready only when the copy is
        complete: until then readers keep reading the legacy map, and the next call copies again."""
        with _POI_LOCK:
            key = self.points_key(run, local)
            found = self._store_meta(key)
            self._check_store_run(key, found, run)
            if found.get("ready"):
                return key, found
            carried: list[dict] = []
            if legacy:
                try:
                    carried = self._list_in(legacy, self.map_metadata(legacy))
                except (OSError, KeyError, ValueError):
                    carried = []
            store_meta = {"kind": "poi_store", "run": run, "grid": grid, "poi_layer": f"poi_{uuid.uuid4().hex[:8]}", "layers": [], "ready": False}
            if carried:
                store_meta["migrated_from"] = legacy
            if found:
                self._write_store_meta(key, store_meta)
            elif key.endswith(".h5"):
                staging = Path(f"{key}.{uuid.uuid4().hex[:8]}.tmp")
                with h5py.File(staging, "w") as h5:
                    h5.attrs["metadata"] = json.dumps(store_meta)
                os.replace(staging, key)
            else:
                self.client.create_container(key, metadata=store_meta)
            if carried:
                self.add_pois_to_layer(key, store_meta, carried)
            store_meta["ready"] = True
            self._write_store_meta(key, store_meta)
            return key, store_meta

    def read_level(self, map_key: str, level: int, rows: slice = slice(None), cols: slice = slice(None)) -> np.ndarray:
        if self._local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                return h5[f"level_{level}"][rows, cols]
        return np.asarray(self.client[map_key][f"level_{level}"][rows, cols])

    def level_shape(self, map_key: str, level: int) -> tuple[int, int]:
        if self._local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                return tuple(h5[f"level_{level}"].shape)
        return tuple(self.client[map_key][f"level_{level}"].shape)

    # -------------------------------------------------------------- points of interest
    def _grid(self, map_key: str, meta: dict | None = None) -> dict:
        """The POI layer's cell grid: origin and cell size in metres, cells per side (a point store keeps the grid
        of the map it was made for)."""
        meta = meta or self.map_metadata(map_key)
        if "grid" in meta:
            return _decode(meta["grid"])
        h, w = (int(v) for v in meta["canvas_shape"])
        pixel = float(meta["pixel_size_m"])
        ox, oy = (float(v) for v in meta["origin_xy_m"])
        cell = max(h, w) * pixel / POI_CELLS
        return {"x0": ox - pixel / 2, "y0": oy - pixel / 2, "cell_m": cell, "n": POI_CELLS}

    @staticmethod
    def _cell_of(grid: dict, x: float, y: float) -> tuple[int, int]:
        n = grid["n"]
        i = int(np.clip((y - grid["y0"]) // grid["cell_m"], 0, n - 1))
        j = int(np.clip((x - grid["x0"]) // grid["cell_m"], 0, n - 1))
        return i, j

    def _layer_name(self, meta: dict) -> str:
        return meta.get("poi_layer", "poi")

    def list_pois(self, map_key: str, kind: str | None = None, box_m: list[float] | None = None, limit: int | None = None) -> list[dict]:
        """Points of interest, optionally of one ``kind`` and inside a stage box [x_min, x_max, y_min, y_max].

        With a box, only the spatial cells the box touches are read.
        """
        key, meta = self._points_at(map_key)
        return self._list_in(key, meta, kind, box_m, limit)

    def _list_in(self, map_key: str, meta: dict, kind: str | None = None, box_m: list[float] | None = None, limit: int | None = None) -> list[dict]:
        out = list(_decode(meta.get("pois", [])) or [])  # maps written before the layer kept them here
        grid = self._grid(map_key, meta)
        cells = None
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            i0, j0 = self._cell_of(grid, x0, y0)
            i1, j1 = self._cell_of(grid, x1, y1)
            cells = {(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}
        if kind is None or kind not in CELL_KINDS:
            out += self._read_group(map_key, meta, HIGHLIGHTS)
        if kind is None or kind in CELL_KINDS:
            for name in self._cell_names(map_key, meta) if cells is None else [f"c_{i}_{j}" for i, j in sorted(cells)]:
                out += self._read_group(map_key, meta, name)
                if limit is not None and box_m is None and kind is None and len(out) >= limit:
                    break
        if kind is not None:
            out = [p for p in out if p.get("kind") == kind]
        if box_m is not None:
            x0, x1, y0, y1 = box_m
            out = [p for p in out if x0 <= p["x_m"] <= x1 and y0 <= p["y_m"] <= y1]
        return out if limit is None else out[:limit]

    def poi_kinds(self, map_key: str) -> list[str]:
        """The kinds of point the map holds, from its highlights and stored cell counts (no cell is read)."""
        key, meta = self._points_at(map_key)
        found = {str(p.get("kind")) for p in (_decode(meta.get("pois", [])) or []) + self._read_group(key, meta, HIGHLIGHTS)}
        if any(n > 0 for n in self.poi_counts(map_key)["cells"].values()):
            found |= CELL_KINDS
        return sorted(found)

    def poi_counts(self, map_key: str) -> dict:
        """Points per cell and in total, from stored counts (no point is read): {"total", "highlights", "cells": {"i_j": n}, "grid"}."""
        map_key, meta = self._points_at(map_key)
        layer = self._layer_name(meta)
        cells: dict[str, int] = {}
        highlights = 0
        if self._local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                if layer in h5:
                    for name, group in h5[layer].items():
                        if name == HIGHLIGHTS:
                            highlights = int(group.attrs.get("count", 0))
                        else:
                            cells[name[2:]] = int(group.attrs.get("count", 0))
        else:
            node = self.client[map_key]
            if layer in node:
                for name, child in node[layer].items():
                    count = int(_decode(child.metadata.get("count", 0)) or 0)
                    if name == HIGHLIGHTS:
                        highlights = count
                    else:
                        cells[name[2:]] = count
        legacy = len(_decode(meta.get("pois", [])) or [])
        return {"total": legacy + highlights + sum(cells.values()), "highlights": legacy + highlights, "cells": cells, "grid": self._grid(map_key, meta)}

    def poi_count(self, map_key: str) -> int:
        return self.poi_counts(map_key)["total"]

    def _cell_names(self, map_key: str, meta: dict) -> list[str]:
        layer = self._layer_name(meta)
        if self._local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                return sorted(n for n in h5[layer] if n != HIGHLIGHTS) if layer in h5 else []
        node = self.client[map_key]
        return sorted(n for n in node[layer]) if layer in node else []

    def _read_group(self, map_key: str, meta: dict, name: str) -> list[dict]:
        layer = self._layer_name(meta)
        if self._local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                if layer not in h5 or name not in h5[layer]:
                    return []
                return [p for batch in sorted(h5[layer][name]) for p in json.loads(h5[layer][name][batch].attrs["pois"])]
        node = self.client[map_key]
        if layer not in node or name not in node[layer]:
            return []
        return [p for _, child in sorted(node[layer][name].items(), key=lambda kv: kv[0]) for p in (_decode(child.metadata.get("pois", [])) or [])]

    def add_pois(self, map_key: str, new: list[dict], dedupe: bool = False) -> list[dict]:
        """Store ``new`` points (EDS points in their spatial cells, others as highlights); returns those stored.

        ``dedupe``: skip points whose id is already on the map, so adding the same points again
        (a retried or resumed step) stores them once.
        """
        new = list(new)
        if not new:
            return new
        with _POI_LOCK:
            map_key, meta = self._ensure_points(map_key)
            return self._add_to(map_key, meta, new, dedupe)

    def _add_to(self, map_key: str, meta: dict, new: list[dict], dedupe: bool) -> list[dict]:
        grid = self._grid(map_key, meta)
        groups: dict[str, list[dict]] = {}
        for p in new:
            name = f"c_{'_'.join(map(str, self._cell_of(grid, p['x_m'], p['y_m'])))}" if p.get("kind") in CELL_KINDS else HIGHLIGHTS
            groups.setdefault(name, []).append(p)
        stored: list[dict] = []
        with _POI_LOCK:
            seen: set = set()
            if dedupe:  # a point whose position moved may now fall in another cell: look for its id in all of them
                cells = any(name != HIGHLIGHTS for name in groups)
                names = ([HIGHLIGHTS] if HIGHLIGHTS in groups else []) + (self._cell_names(map_key, meta) if cells else [])
                seen = {p.get("id") for name in names for p in self._read_group(map_key, meta, name)}
            for name, points in groups.items():
                if dedupe:
                    points = [p for p in points if p.get("id") not in seen]
                if points:
                    self._append(map_key, meta, name, points)
                    stored += points
        return stored

    def _append(self, map_key: str, meta: dict, name: str, points: list[dict]) -> None:
        layer = self._layer_name(meta)
        batch = f"b_{time.time_ns():020d}_{uuid.uuid4().hex[:6]}"
        if self._local(map_key):
            with h5py.File(map_key, "a") as h5:
                group = h5.require_group(layer).require_group(name)
                group.create_group(batch).attrs["pois"] = json.dumps(points)
                group.attrs["count"] = int(group.attrs.get("count", 0)) + len(points)
            return
        node = self.client[map_key]
        if layer not in node:
            node.create_container(layer, metadata={"kind": "poi_layer", "grid": self._grid(map_key, meta)})
        layer_node = node[layer]
        if name not in layer_node:
            layer_node.create_container(name, metadata={"count": 0})
        group = layer_node[name]
        group.create_container(batch, metadata={"pois": points})
        _replace_metadata(group, {"count": int(_decode(group.metadata.get("count", 0)) or 0) + len(points)})

    def update_pois(self, map_key: str, change: Callable[[list[dict]], list[dict]]) -> list[dict]:
        """Replace the run's points with ``change(current)``.

        A new layer is built beside the current one and the store's metadata is switched to it in
        one update, so readers see either all the old points or all the new ones. The old layer is
        kept and listed in the store's ``layers`` with the time it was replaced.
        """
        with _POI_LOCK:
            map_key, meta = self._ensure_points(map_key)
            pois = change(self._list_in(map_key, meta))
            old = self._layer_name(meta)
            replaced = [{"layer": old, "replaced_at": time.time()}] if self._has_layer(map_key, old) else []
            staged = {**meta, "poi_layer": f"poi_{uuid.uuid4().hex[:8]}", "layers": list(_decode(meta.get("layers")) or []) + replaced}
            staged.pop("pois", None)
            self.add_pois_to_layer(map_key, staged, pois)
            if self._local(map_key):
                with h5py.File(map_key, "a") as h5:
                    h5.attrs["metadata"] = json.dumps(staged)
            else:
                _replace_metadata(self.client[map_key], staged)
        return pois

    def _has_layer(self, map_key: str, layer: str) -> bool:
        if self._local(map_key):
            with _POI_LOCK, h5py.File(map_key, "r") as h5:
                return layer in h5
        return layer in self.client[map_key]

    def add_pois_to_layer(self, map_key: str, meta: dict, pois: list[dict]) -> None:
        grid = self._grid(map_key, meta)
        groups: dict[str, list[dict]] = {}
        for p in pois:
            name = f"c_{'_'.join(map(str, self._cell_of(grid, p['x_m'], p['y_m'])))}" if p.get("kind") in CELL_KINDS else HIGHLIGHTS
            groups.setdefault(name, []).append(p)
        if not groups and not self._local(map_key):
            node = self.client[map_key]
            node.create_container(meta["poi_layer"], metadata={"kind": "poi_layer", "grid": grid})
        for name, points in groups.items():
            self._append(map_key, meta, name, points)

    def set_pois(self, map_key: str, pois: list[dict]) -> None:
        """Replace all points of interest."""
        self.update_pois(map_key, lambda _current: list(pois))


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
