"""Writing a map's pyramid (a local HDF5 file or Tiled nodes) and converting between stage and pixel positions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import os
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .runs import _file_lock

if TYPE_CHECKING:
    from .store import MapStore

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
        with _file_lock(self.path):
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
    else:  # tiled 0.2 needs external_only=False to also remove data Tiled wrote itself
        node.delete(recursive=True, external_only=False)


def _replace_metadata(node, metadata: dict) -> None:
    node.replace_metadata(metadata=metadata) if hasattr(node, "replace_metadata") else node.update_metadata(metadata=metadata)


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
