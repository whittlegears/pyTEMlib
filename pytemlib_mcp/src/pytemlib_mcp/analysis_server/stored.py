"""Find the arrays inside a stored acquisition (a Tiled node or an HDF5 group) without reading their values.

This is the part of asyncroscopy's ``asyncroscopy.data.views`` that ``tiled_datasets`` needs (finding a dataset
in a key, its attributes, its energy axis), copied so that pyTEMlib does not depend on asyncroscopy; it reads the
layouts asyncroscopy writes the same way views does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterator

import numpy as np

PREVIEW_MAX_SIDE = 512
MAX_LISTED = 200  # datasets walked inside one container
MAX_GROUPS = 40  # entries listed per level of a container
MAX_DEPTH = 3
IMAGE_GROUPS = ("image", "tiles", "channels")  # looked at first, so a run's pictures come before its spectra
SPECTRUM_GROUPS = ("spectrum_image", "spectrum", "spectra")


@dataclass(frozen=True)
class Dataset:
    """One array (or table) inside a key, described from its structure, without reading its values."""

    key: str
    path: str
    node: Any
    shape: tuple[int, ...]
    dtype: np.dtype
    attrs: dict
    parent_attrs: dict = field(default_factory=dict)
    family: str = "array"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize if self.shape else self.dtype.itemsize

    @property
    def name(self) -> str:
        return f"{self.key}:{self.path}" if self.path else self.key


@dataclass(frozen=True)
class Axis:
    """A linear axis: channel i is at ``offset + i * scale``."""

    offset: float
    scale: float
    label: str
    units: str

    def values(self, start: int, stop: int, step: int = 1) -> np.ndarray:
        return self.offset + self.scale * np.arange(start, stop, step, dtype=np.float64)


def numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [numpy_to_python(v) for v in obj]
        return tuple(conv) if isinstance(obj, tuple) else conv
    return obj


def _decode(value: Any) -> Any:
    """data_writer json-encodes list and dict attributes; decode them back."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _attrs(node: Any) -> dict:
    return {str(k): _decode(numpy_to_python(v)) for k, v in dict(getattr(node, "metadata", None) or {}).items()}


def _family(node: Any) -> str:
    family = getattr(node, "structure_family", None)
    if family is not None:
        return str(getattr(family, "value", family))
    if callable(getattr(node, "read", None)) and not callable(getattr(node, "keys", None)):
        return "array"
    return "container" if callable(getattr(node, "keys", None)) else "unknown"


class _Loaded:
    """An array already in memory, with the node interface (for nodes that do not report their shape)."""

    def __init__(self, array: np.ndarray, metadata: dict) -> None:
        self.array, self.metadata = array, metadata
        self.shape, self.dtype = array.shape, array.dtype

    def read(self, slices: Any = None) -> np.ndarray:
        return self.array if slices is None else self.array[slices]


def as_dataset(node: Any, key: str, path: str = "", parent_attrs: dict | None = None) -> Dataset:
    family = _family(node)
    shape = getattr(node, "shape", None)
    if family == "array" and shape is None:  # a reader that cannot report its shape, so read it once
        node = _Loaded(np.asarray(node.read()), dict(getattr(node, "metadata", None) or {}))
        shape = node.shape
    try:
        dtype = np.dtype(getattr(node, "dtype", None) or np.float64) if family == "array" else np.dtype(object)
    except TypeError:
        dtype = np.dtype(object)
    return Dataset(key, path, node, tuple(int(s) for s in (shape or ())), dtype, _attrs(node), dict(parent_attrs or {}), family)


def _children(node: Any) -> Iterator[tuple[str, Any]]:
    items = getattr(node, "items", None)
    if callable(items):  # Tiled pages carry each child's structure, so no request per child is needed
        return ((str(k), v) for k, v in items())
    return ((str(k), node[k]) for k in node.keys())


def _count(node: Any) -> int | None:
    try:
        return len(node)
    except TypeError:
        return None


def _join(base: str, name: str) -> str:
    return f"{base}/{name}" if base else name


def datasets(node: Any, key: str, limit: int = MAX_LISTED) -> list[Dataset]:
    """The arrays and tables inside ``node`` (itself, if it is one), at most ``limit``, at most 3 levels deep."""
    out: list[Dataset] = []

    def walk(current: Any, path: str, depth: int, parent: dict) -> None:
        if len(out) >= limit:
            return
        family = _family(current)
        if family in ("array", "table"):
            out.append(as_dataset(current, key, path, parent))
            return
        if family != "container" or depth > MAX_DEPTH:
            return
        attrs = _attrs(current)
        for child, value in _children(current):
            walk(value, _join(path, child), depth + 1, attrs)
            if len(out) >= limit:
                return

    walk(node, "", 0, {})
    return out


def _first_dataset(node: Any, key: str, path: str = "", parent: dict | None = None, depth: int = 0) -> Dataset | None:
    """The first array or table found depth-first, reading as little as possible."""
    family = _family(node)
    if family in ("array", "table"):
        return as_dataset(node, key, path, parent)
    if family != "container" or depth > MAX_DEPTH:
        return None
    attrs = _attrs(node)
    for child, value in islice(_children(node), MAX_GROUPS):
        found = _first_dataset(value, key, _join(path, child), attrs, depth + 1)
        if found is not None:
            return found
    return None


def is_map(node: Any) -> bool:
    return _family(node) == "container" and _attrs(node).get("kind") == "stitched_map"


def resolve(node: Any, key: str, path: str) -> Any:
    """The node at ``path`` inside ``key`` (a dataset or a group), or a ValueError listing what is there."""
    current = node
    for part in path.strip("/").split("/"):
        try:
            current = current[part]
        except (KeyError, TypeError):
            raise ValueError(f"{key!r} has no dataset {path!r}; it has:\n{container_listing(node)}") from None
    return current


def pick(node: Any, key: str, dataset: str | None = None, max_side: int = PREVIEW_MAX_SIDE) -> Dataset:
    """The dataset to show for ``key``: ``dataset`` if given (inside it, if it is a group), else a picture
    (image, a run's tiles or channels), a spectrum, a map level, else the first dataset."""
    base = ""
    if dataset:
        base = dataset.strip("/")
        target = resolve(node, key, base)
        if _family(target) in ("array", "table"):
            return as_dataset(target, key, base, _attrs(node))
        node = target
    if _family(node) in ("array", "table"):
        return as_dataset(node, key)
    if is_map(node):
        levels = sorted((d for d in datasets(node, key, limit=40) if d.path.startswith("level_") and d.ndim == 2), key=lambda d: d.nbytes)
        fitting = [d for d in levels if max(d.shape) <= 2 * max_side]
        if levels:
            return fitting[-1] if fitting else levels[0]
    for name in IMAGE_GROUPS + SPECTRUM_GROUPS:
        try:
            child = node[name]
        except (KeyError, TypeError):
            continue
        found = _first_dataset(child, key, _join(base, name), _attrs(node))
        if found is not None:
            return found
    found = _first_dataset(node, key, base)
    if found is None:
        raise ValueError(f"{key!r}{' / ' + base if base else ''} holds no arrays or tables")
    return found


def container_listing(node: Any, prefix: str = "", depth: int = 0) -> str:
    """Every group of a container, breadth-first: 'tiles/: 400 x (128, 128) uint16 (shape of tile_00000)'."""
    lines: list[str] = []
    children = list(islice(_children(node), MAX_GROUPS + 1))
    for name, child in children[:MAX_GROUPS]:
        family = _family(child)
        if family in ("array", "table"):
            ds = as_dataset(child, "", name)
            lines.append(f"{prefix}{name}: {tuple(ds.shape)} {ds.dtype}" if family == "array" else f"{prefix}{name}: table")
            continue
        if family != "container":
            lines.append(f"{prefix}{name}: {family}")
            continue
        count = _count(child)
        counted = count if count is not None else "?"
        first = next(iter(_children(child)), None)
        if first is None:
            lines.append(f"{prefix}{name}/: empty")
        elif _family(first[1]) in ("array", "table"):
            ds = as_dataset(first[1], "", first[0])
            lines.append(f"{prefix}{name}/: {counted} x {tuple(ds.shape)} {ds.dtype} (shape of {first[0]})")
        elif depth < MAX_DEPTH:
            lines.append(f"{prefix}{name}/: {counted} groups")
            lines.append(container_listing(child, f"{prefix}{name}/", depth + 1))
        else:
            lines.append(f"{prefix}{name}/: {counted} groups (deeper levels not listed; open one with dataset)")
    if len(children) > MAX_GROUPS:
        total = _count(node)
        lines.append(f"{prefix}... and {(total - MAX_GROUPS) if total else 'more'} more entries")
    return "\n".join(line for line in lines if line)


def energy_axis(attrs: dict) -> Axis | None:
    """The calibrated axis of a 1-D dataset, from any naming convention this repository writes."""
    try:
        if "energy_offset_ev" in attrs and "energy_dispersion_ev" in attrs:
            return Axis(float(attrs["energy_offset_ev"]), float(attrs["energy_dispersion_ev"]), "energy", "eV")
        if "dispersion" in attrs and ("start_energy" in attrs or "offset" in attrs):
            return Axis(float(attrs.get("start_energy", attrs.get("offset"))), float(attrs["dispersion"]), "energy", str(attrs.get("energy_units", "eV")))
        if "Dispersion" in attrs and "OffsetEnergy" in attrs:
            # Thermo Fisher EDS files saved through AutoScript store the first channel energy and the channel width as text in eV.
            if float(attrs["Dispersion"]) > 0:
                return Axis(float(attrs["OffsetEnergy"]), float(attrs["Dispersion"]), "energy", "eV")
        if "axis_offset" in attrs and "axis_scale" in attrs:
            return Axis(float(attrs["axis_offset"]), float(attrs["axis_scale"]), str(attrs.get("axis_label", "axis")), str(attrs.get("axis_units", "")))
    except (TypeError, ValueError):
        return None
    return None
