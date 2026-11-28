"""A map's points of interest: kept once per run, in spatial cells and a highlights group."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from .runs import _batch_points, _decode
from .pyramid import _replace_metadata

POI_CELLS = 64  # cells along each side of the POI layer's grid over the map
CELL_KINDS = {"eds_point": "EDS point spectrum"}  # kinds kept in spatial cells, with their labels; every other kind is a highlight
HIGHLIGHTS = "highlights"
highlight_kind_labels = {"candidate": "Scored region", "note": "Note"}


def poi_kind_labels(kinds: list[str]) -> list[dict]:
    """``kinds`` as {"kind", "label", "in_cells"}: the label of a kind this server writes (eds_pois, score_map, add_poi's
    default), null for a kind a caller named itself; ``in_cells`` for the kinds kept in spatial cells."""
    labels = {**highlight_kind_labels, **CELL_KINDS}
    return [{"kind": kind, "label": labels.get(kind), "in_cells": kind in CELL_KINDS} for kind in kinds]


class PoiStore:
    """MapStore's points of interest, kept once per run for all its map versions."""

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
            with self._lock(key), h5py.File(key, "r") as h5:
                return json.loads(h5.attrs["metadata"]) if "metadata" in h5.attrs else {}
        if key not in self.client:
            return {}
        return {k: _decode(v) for k, v in dict(self.client[key].metadata).items()}

    def _write_store_meta(self, key: str, meta: dict) -> None:
        if self._local(key):
            with self._lock(key), h5py.File(key, "a") as h5:
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
        map). A map with no run keeps its own points. Call with the lock of ``_points_target(map_key)`` held."""
        meta = self.map_metadata(map_key)
        if not meta.get("run"):
            return map_key, meta
        return self._ensure_store(str(meta["run"]), self._local(map_key), self._grid(map_key, meta), map_key if legacy == "" else legacy)

    def _ensure_store(self, run: str, local: bool, grid: dict, legacy: str | None) -> tuple[str, dict]:
        """``run``'s point store, made on first use with ``grid``. A new store copies in the points ``legacy`` kept
        inside itself before point stores existed, into a fresh layer, and is marked ready only when the copy is
        complete: until then readers keep reading the legacy map, and the next call copies again."""
        key = self.points_key(run, local)
        with self._lock(key):
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

    @classmethod
    def cells_in(cls, grid: dict, box_m: list[float]) -> list[tuple[int, int]]:
        """The cells (i, j) a stage box [x_min, x_max, y_min, y_max] touches, in order."""
        x0, x1, y0, y1 = box_m
        i0, j0 = cls._cell_of(grid, x0, y0)
        i1, j1 = cls._cell_of(grid, x1, y1)
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    @staticmethod
    def cell_box(grid: dict, i: int, j: int) -> list[float]:
        """The stage box [x_min, x_max, y_min, y_max] of cell (i, j)."""
        cell = grid["cell_m"]
        return [grid["x0"] + j * cell, grid["x0"] + (j + 1) * cell, grid["y0"] + i * cell, grid["y0"] + (i + 1) * cell]

    def _layer_name(self, meta: dict) -> str:
        return meta.get("poi_layer", "poi")

    @staticmethod
    def _highlights_name(meta: dict) -> str:
        """The current highlights group of the layer: ``highlights`` until ``update_highlights`` replaced it."""
        return str(meta.get("highlights") or HIGHLIGHTS)

    def _group_of(self, grid: dict, meta: dict, poi: dict) -> str:
        """The group a point is kept in: its spatial cell for an EDS point, else the highlights."""
        if poi.get("kind") in CELL_KINDS:
            return f"c_{'_'.join(map(str, self._cell_of(grid, poi['x_m'], poi['y_m'])))}"
        return self._highlights_name(meta)

    def _points_target(self, map_key: str) -> str:
        """The key a map's points are kept under (its run's point store, or the map itself when it has no run), so
        a change can take that key's lock before it reads them."""
        meta = self.map_metadata(map_key)
        run = meta.get("run")
        return self.points_key(str(run), self._local(map_key)) if run else map_key

    def list_pois(self, map_key: str, kind: str | None = None, box_m: list[float] | None = None, limit: int | None = None,
                  cells: bool = True) -> list[dict]:
        """Points of interest, optionally of one ``kind`` and inside a stage box [x_min, x_max, y_min, y_max].

        With a box, only the stored spatial cells the box touches are read; ``cells=False`` reads none of them (the highlights only).
        """
        key, meta = self._points_at(map_key)
        return self._list_in(key, meta, kind, box_m, limit, cells)

    def _list_in(self, map_key: str, meta: dict, kind: str | None = None, box_m: list[float] | None = None, limit: int | None = None,
                 cells: bool = True) -> list[dict]:
        out = list(_decode(meta.get("pois", [])) or [])  # older maps keep their points here
        grid = self._grid(map_key, meta)
        touched = None if box_m is None else {f"c_{i}_{j}" for i, j in self.cells_in(grid, box_m)}
        if kind is None or kind not in CELL_KINDS:
            out += self._read_group(map_key, meta, self._highlights_name(meta))
        if cells and (kind is None or kind in CELL_KINDS):
            for name in [n for n in self._cell_names(map_key, meta) if touched is None or n in touched]:
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
        found = {str(p.get("kind")) for p in (_decode(meta.get("pois", [])) or []) + self._read_group(key, meta, self._highlights_name(meta))}
        if any(n > 0 for n in self.poi_counts(map_key)["cells"].values()):
            found |= set(CELL_KINDS)
        return sorted(found)

    def poi_counts(self, map_key: str) -> dict:
        """Points per cell and in total, from stored counts (no point is read): {"total", "highlights", "cells": {"i_j": n}, "grid"}."""
        map_key, meta = self._points_at(map_key)
        layer = self._layer_name(meta)
        current = self._highlights_name(meta)
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                children = {name: int(group.attrs.get("count", 0)) for name, group in h5[layer].items()} if layer in h5 else {}
        else:
            node = self.client[map_key]
            children = {name: int(_decode(child.metadata.get("count", 0)) or 0) for name, child in node[layer].items()} if layer in node else {}
        highlights = children.get(current, 0)
        cells = {name[2:]: count for name, count in children.items() if name.startswith("c_")}
        legacy = len(_decode(meta.get("pois", [])) or [])
        return {"total": legacy + highlights + sum(cells.values()), "highlights": legacy + highlights, "cells": cells, "grid": self._grid(map_key, meta)}

    def poi_count(self, map_key: str) -> int:
        return self.poi_counts(map_key)["total"]

    def _cell_names(self, map_key: str, meta: dict) -> list[str]:
        layer = self._layer_name(meta)
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                return sorted(n for n in h5[layer] if n.startswith("c_")) if layer in h5 else []
        node = self.client[map_key]
        return sorted(n for n in node[layer] if n.startswith("c_")) if layer in node else []

    def _read_group(self, map_key: str, meta: dict, name: str) -> list[dict]:
        layer = self._layer_name(meta)
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                if layer not in h5 or name not in h5[layer]:
                    return []
                group = h5[layer][name]
                return [p for batch in sorted(group) for p in _batch_points(group[batch])]
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
        with self._lock(self._points_target(map_key)):
            map_key, meta = self._ensure_points(map_key)
            return self._add_to(map_key, meta, new, dedupe)

    def _add_to(self, map_key: str, meta: dict, new: list[dict], dedupe: bool) -> list[dict]:
        """Append ``new`` to the points kept at ``map_key``; call with its lock held."""
        grid = self._grid(map_key, meta)
        current = self._highlights_name(meta)
        groups: dict[str, list[dict]] = {}
        for p in new:
            groups.setdefault(self._group_of(grid, meta, p), []).append(p)
        stored: list[dict] = []
        seen: set = set()
        if dedupe:  # a point that moved may now fall in another cell, so look for its id in all of them
            cells = any(name != current for name in groups)
            names = ([current] if current in groups else []) + (self._cell_names(map_key, meta) if cells else [])
            seen = {p.get("id") for name in names for p in self._read_group(map_key, meta, name)}
        for name, points in groups.items():
            if dedupe:
                points = [p for p in points if p.get("id") not in seen]
            if points:
                self._append(map_key, meta, name, points)
                stored += points
        return stored

    def _append(self, map_key: str, meta: dict, name: str, points: list[dict]) -> None:
        """One batch of points in group ``name`` of the current layer (both made when missing). A local batch is
        a UTF-8 string dataset holding the points as JSON, so a batch of any size fits (an HDF5 attribute holds
        at most 64 KB)."""
        layer = self._layer_name(meta)
        batch = f"b_{time.time_ns():020d}_{uuid.uuid4().hex[:6]}"
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "a") as h5:
                group = h5.require_group(layer).require_group(name)
                if points:
                    group.create_dataset(batch, data=json.dumps(points), dtype=h5py.string_dtype("utf-8"))
                group.attrs["count"] = int(group.attrs.get("count", 0)) + len(points)
            return
        node = self.client[map_key]
        if layer not in node:
            node.create_container(layer, metadata={"kind": "poi_layer", "grid": self._grid(map_key, meta)})
        layer_node = node[layer]
        if name not in layer_node:
            layer_node.create_container(name, metadata={"count": 0})
        if not points:
            return
        group = layer_node[name]
        group.create_container(batch, metadata={"pois": points})
        _replace_metadata(group, {"count": int(_decode(group.metadata.get("count", 0)) or 0) + len(points)})

    def update_pois(self, map_key: str, change: Callable[[list[dict]], list[dict]]) -> list[dict]:
        """Replace every one of the run's points with ``change(current)``.

        This reads and rewrites every point, EDS cells included; to change only the highlights (scored
        regions, notes) use ``update_highlights``. A new layer is built beside the current one and the
        store's metadata is switched to it in one update, so readers see either all the old points or all
        the new ones. The old layer is kept and listed in the store's ``layers`` with the time it was replaced.
        """
        with self._lock(self._points_target(map_key)):
            map_key, meta = self._ensure_points(map_key)
            pois = change(self._list_in(map_key, meta))
            old = self._layer_name(meta)
            replaced = [{"layer": old, "replaced_at": time.time()}] if self._has_layer(map_key, old) else []
            staged = {**meta, "poi_layer": f"poi_{uuid.uuid4().hex[:8]}", "layers": list(_decode(meta.get("layers")) or []) + replaced}
            for key in ("pois", "highlights", "replaced_highlights"):  # the new layer keeps its highlights in its own group
                staged.pop(key, None)
            self.add_pois_to_layer(map_key, staged, pois)
            self._write_store_meta(map_key, staged)
        return pois

    def update_highlights(self, map_key: str, change: Callable[[list[dict]], list[dict]]) -> list[dict]:
        """Replace the run's highlights (every point that is not an EDS point) with ``change(current)``; the EDS
        cells are neither read nor rewritten. Returns the new highlights.

        The new highlights are written to a new group of the current layer and the store's metadata is
        switched to it in one update, so readers see the old highlights or the new ones, never a mix. The
        replaced group is kept and listed in ``replaced_highlights``.
        """
        with self._lock(self._points_target(map_key)):
            map_key, meta = self._ensure_points(map_key)
            if _decode(meta.get("pois")):  # an older map keeps its points inline, so rewrite them all
                kept = self.update_pois(map_key, lambda current: [p for p in current if p.get("kind") in CELL_KINDS]
                                        + list(change([p for p in current if p.get("kind") not in CELL_KINDS])))
                return [p for p in kept if p.get("kind") not in CELL_KINDS]
            old = self._highlights_name(meta)
            new = list(change(self._read_group(map_key, meta, old)))
            wrong = sorted({str(p.get("kind")) for p in new if p.get("kind") in CELL_KINDS})
            if wrong:
                raise ValueError(f"update_highlights cannot add points of kind {wrong}: they are kept in the spatial cells")
            name = f"{HIGHLIGHTS}_{uuid.uuid4().hex[:8]}"
            self._append(map_key, meta, name, new)
            replaced = list(_decode(meta.get("replaced_highlights")) or [])
            if self._has_group(map_key, meta, old):
                replaced.append({"group": old, "replaced_at": time.time()})
            self._write_store_meta(map_key, {**meta, "highlights": name, "replaced_highlights": replaced})
        return new

    def _has_layer(self, map_key: str, layer: str) -> bool:
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                return layer in h5
        return layer in self.client[map_key]

    def _has_group(self, map_key: str, meta: dict, name: str) -> bool:
        layer = self._layer_name(meta)
        if self._local(map_key):
            with self._lock(map_key), h5py.File(map_key, "r") as h5:
                return layer in h5 and name in h5[layer]
        node = self.client[map_key]
        return layer in node and name in node[layer]

    def add_pois_to_layer(self, map_key: str, meta: dict, pois: list[dict]) -> None:
        grid = self._grid(map_key, meta)
        groups: dict[str, list[dict]] = {}
        for p in pois:
            groups.setdefault(self._group_of(grid, meta, p), []).append(p)
        if not groups and not self._local(map_key):
            node = self.client[map_key]
            node.create_container(meta["poi_layer"], metadata={"kind": "poi_layer", "grid": grid})
        for name, points in groups.items():
            self._append(map_key, meta, name, points)

    def set_pois(self, map_key: str, pois: list[dict]) -> None:
        """Replace all points of interest."""
        self.update_pois(map_key, lambda _current: list(pois))
