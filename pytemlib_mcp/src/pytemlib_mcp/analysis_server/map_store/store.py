"""MapStore: Tiled access, runs and maps, versions."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

import h5py
import numpy as np

from .runs import LazyTiles, LocalPathRefused, RunData, RunInfo, TileSource, _Locks, _decode, _file_lock, _layout1_info, _layout2_info
from .poi import PoiStore
from .pyramid import _LocalMapWriter, _TiledMapWriter

AXES = "rows=+y, columns=+x; stage coordinates in metres"


class MapStore(PoiStore):
    """Where runs are read from and maps are written to."""

    def __init__(self, tiled_uri: str | None = None, api_key: str | None = None, local_roots: list[str | os.PathLike] | None = None):
        """``local_roots``: directories whose HDF5 files (and subdirectories') may be read and written as
        local paths; None or ``[]`` accepts no local paths. ``api_key``: the Tiled API key (the command line
        reads it from the environment when it is not given)."""
        self.tiled_uri = tiled_uri
        self.api_key = api_key
        self.local_roots = [Path(r).resolve() for r in (local_roots or [])]
        self._client = None
        self._tiled_locks = _Locks()

    def _lock(self, key: str) -> threading.RLock:
        """The lock of one map, point store or local file (see ``_Locks``)."""
        if str(key).lower().endswith((".h5", ".hdf5", ".tmp")):
            return _file_lock(key)
        return self._tiled_locks(str(key))

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

    def list_maps(self, run: str | None = None) -> list[dict]:
        """Every stitched version of ``run``'s map, newest first: {"map", "id", "stitched_at", "inputs_sha256", "tiles", "run"}.

        Without ``run``, every stitched map the store has: in Tiled, and in the HDF5 files named ``*_map*.h5`` under
        ``local_roots``."""
        found = self._stitched(run)
        found.sort(key=lambda item: (float(_decode(item[1].get("stitched_at")) or 0.0), item[0]), reverse=True)
        return [
            {"map": key, "id": meta.get("id"), "stitched_at": _decode(meta.get("stitched_at")), "inputs_sha256": meta.get("inputs_sha256"),
             "tiles": _decode(meta.get("tiles")), "run": meta.get("run")}
            for key, meta in found
        ]

    def _stitched(self, run: str | None) -> list[tuple[str, dict]]:
        if run is None:
            found = self._local_maps([path for root in self.local_roots for path in root.rglob("*_map*.h5")], None)
            return found + (self._tiled_maps(None) if self.tiled_uri else [])
        if self.is_local(run):
            name = self.map_name(run)
            return self._local_maps([p for p in Path(run).parent.glob(f"{name}*.h5") if p.name == f"{name}.h5" or p.name.startswith(f"{name}_")], run)
        return self._tiled_maps(run)

    @staticmethod
    def _local_maps(paths: list[Path], run: str | None) -> list[tuple[str, dict]]:
        found = []
        for path in paths:
            try:
                with _file_lock(path), h5py.File(path, "r") as h5:
                    meta = json.loads(h5.attrs["metadata"])
            except (OSError, KeyError, ValueError):
                continue
            if meta.get("kind") == "stitched_map" and (run is None or Path(str(meta.get("run", ""))).resolve() == Path(run).resolve()):
                found.append((str(path), meta))
        return found

    def _tiled_maps(self, run: str | None) -> list[tuple[str, dict]]:
        from tiled.queries import Key

        search = self.client.search(Key("kind") == "stitched_map")
        if run is not None:
            search = search.search(Key("run") == run)
        return [(key, dict(node.metadata)) for key, node in search.items()]

    def current_map(self, run: str) -> str | None:
        """Key or path of the newest map stitched from ``run``, or None."""
        maps = self.list_maps(run)
        return maps[0]["map"] if maps else None

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
                # One paged listing carries every tile's shape and metadata, which avoids a request per tile.
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

    def map_writer(self, run: str):
        """A LevelWriter for a new map version of ``run`` (see image_mosaic.assemble); pass it to ``write_map`` when assembled.

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
        metadata = {"kind": "stitched_map", "run": run, "axes": AXES, "id": uuid.uuid4().hex, "parents": [run], "version": writer.name,
                    **self.sample_of(run), **metadata}
        try:
            writer.finish(metadata)
        except Exception as exc:
            writer.abandon()
            if "401" in str(exc) or "permission" in str(exc).lower():
                raise PermissionError(
                    f"Tiled at {self.tiled_uri} refused to store the map: {exc}. Start this server with the Tiled API key "
                    "(--api-key, or $ASYNCROSCOPY_TILED_API_KEY; an asyncroscopy stack uses 'secret' unless that variable is set)."
                ) from exc
            raise
        try:
            local = self.is_local(run)
            with self._lock(self.points_key(run, local)):
                key, store = self._ensure_store(run, local, self._grid("", metadata), previous)
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
        except Exception:  # a refused path or a Tiled error leaves the sample unknown, and callers report that
            return {}
        return found

    def _own_sample(self, key: str) -> dict:
        try:
            if self._local(key):
                with self._lock(key), h5py.File(key, "r") as h5:
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
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                meta = json.loads(h5.attrs["metadata"]) if "metadata" in h5.attrs else {}
        else:
            meta = dict(self.client[map_key].metadata)
        if meta.get("kind") != "stitched_map":
            try:
                stitched = self.current_map(map_key)
            except Exception:  # not a run either, so there is no stitched map to name
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

    def read_level(self, map_key: str, level: int, rows: slice = slice(None), cols: slice = slice(None)) -> np.ndarray:
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                return h5[f"level_{level}"][rows, cols]
        return np.asarray(self.client[map_key][f"level_{level}"][rows, cols])

    def level_shape(self, map_key: str, level: int) -> tuple[int, int]:
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                return tuple(h5[f"level_{level}"].shape)
        return tuple(self.client[map_key][f"level_{level}"].shape)

    def map_view(self, map_key: str, meta: dict | None = None) -> dict:
        """What a client needs to draw a stitched map: {"title", "canvas_shape", "pixel_size_m", "origin_xy_m", "axes",
        "stage_box_m", "local", "pyramid"}. ``pyramid`` has one descriptor per level, read from the stored arrays:
        {"level", "key", "shape", "dtype", "chunks", "downsample", "pixel_size_m", "origin_xy_m"}. Level k's pixel (r, c)
        is centred at stage (origin_xy_m[0] + c * pixel_size_m, origin_xy_m[1] + r * pixel_size_m) of that level."""
        meta = meta or self.map_metadata(map_key)
        pixel = float(meta["pixel_size_m"])
        ox, oy = (float(v) for v in meta["origin_xy_m"])
        rows, cols = (int(v) for v in meta["canvas_shape"])
        local = self._local(map_key)
        arrays = self._level_arrays(map_key, int(meta["levels"]), local)
        pyramid = []
        for level, (key, shape, dtype, chunks) in enumerate(arrays):
            factor = 2**level
            pyramid.append({"level": level, "key": key, "shape": shape, "dtype": dtype, "chunks": chunks, "downsample": factor,
                            "pixel_size_m": pixel * factor, "origin_xy_m": [ox + (factor - 1) / 2.0 * pixel, oy + (factor - 1) / 2.0 * pixel]})
        return {"title": self.map_title(map_key, meta), "canvas_shape": [rows, cols], "pixel_size_m": pixel, "origin_xy_m": [ox, oy],
                "axes": meta.get("axes"), "stage_box_m": [ox - pixel / 2, ox + (cols - 0.5) * pixel, oy - pixel / 2, oy + (rows - 0.5) * pixel],
                "local": local, "pyramid": pyramid}

    def map_title(self, map_key: str, meta: dict) -> str:
        """The run's name and the UTC time the map was stitched, or the map's own name when it records neither."""
        name = self._stem(str(meta["run"])) if meta.get("run") else self._stem(map_key)
        stitched = _decode(meta.get("stitched_at"))
        if not isinstance(stitched, (int, float)):
            return name
        return f"{name}, stitched {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(float(stitched)))} UTC"

    def _level_arrays(self, map_key: str, levels: int, local: bool) -> list[tuple[str, list[int], str, list[list[int]]]]:
        """(key, shape, dtype, chunks per axis) of each level array: a Tiled key ``<map>/level_<k>``, or ``<file>#level_<k>``."""
        out = []
        if local:
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                for level in range(levels):
                    data = h5[f"level_{level}"]
                    sides = data.chunks or data.shape
                    chunks = [[min(side, n - start) for start in range(0, n, side)] for n, side in zip(data.shape, sides)]
                    out.append((f"{map_key}#level_{level}", [int(v) for v in data.shape], data.dtype.str, chunks))
            return out
        node = self.client[map_key]
        for level in range(levels):
            structure = node[f"level_{level}"].structure()
            out.append((f"{map_key}/level_{level}", [int(v) for v in structure.shape], structure.data_type.to_numpy_dtype().str,
                        [[int(v) for v in axis] for axis in structure.chunks]))
        return out
