"""Stored acquisitions as sidpy datasets, so pyTEMlib's analysis tools take Tiled keys.

pyTEMlib's tools (``pytemlib_mcp.analysis``) take a dataset reference and read it themselves. This module adds a
resolver to them: a reference ``key`` or ``key#path`` that names something in the Tiled catalogue becomes a
sidpy dataset with the dimensions, units and data type the acquisition recorded:

* an array with an energy scale (any naming ``stored.energy_axis`` reads) is a SPECTRUM, or a SPECTRAL_IMAGE when it
  has positions before its channels;
* a 2-D array is an IMAGE, a 3-D one an IMAGE_STACK (frames first); with ``pixel_size_m`` its axes are in nm,
  without it in pixels;
* leading axes of length one (a streamed acquisition of one row) are dropped.

An array that is none of these, or a spectrum without an energy scale, is refused with the reason, never guessed.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import h5py
import numpy as np
import sidpy

from pytemlib_mcp import analysis as mcp_analysis
from pyTEMlib.image_tools import experiment_from_asyncroscopy

from . import stored

MAX_BYTES = 1 << 30


def _refuse(code: str, message: str) -> Exception:
    return mcp_analysis.AnalysisError(code, message)


def _pixel_nm(attrs: dict) -> float | None:
    value = attrs.get("pixel_size_m")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value * 1e9 if np.isfinite(value) and value > 0 else None


def _spatial(size: int, name: str, pixel_nm: float | None) -> sidpy.Dimension:
    if pixel_nm is None:
        return sidpy.Dimension(np.arange(size, dtype=float), name=name, units="pixel", quantity="Length", dimension_type="spatial")
    return sidpy.Dimension(np.arange(size) * pixel_nm, name=name, units="nm", quantity="Length", dimension_type="spatial")


def _experiment(attrs: dict) -> dict:
    experiment = experiment_from_asyncroscopy(attrs)
    for name in ("acceleration_voltage_v", "acceleration_voltage", "high_tension_v"):
        try:
            experiment["acceleration_voltage"] = float(attrs[name])
            break
        except (KeyError, TypeError, ValueError):
            continue
    return experiment


def to_sidpy(ds: stored.Dataset) -> sidpy.Dataset:
    """The sidpy dataset of one stored array, read whole (refused above ``MAX_BYTES``)."""
    if ds.family != "array":
        raise _refuse("not_an_array", f"{ds.name} is a {ds.family}, not an array pyTEMlib can analyse")
    if ds.nbytes > MAX_BYTES:
        raise _refuse("too_large", f"{ds.name} is {ds.nbytes / 2**20:.0f} MiB; the analysis tools read at most {MAX_BYTES // 2**20} MiB")
    attrs = {**ds.parent_attrs, **ds.attrs}
    values = np.asarray(ds.node.read(), dtype=float)
    while values.ndim > 1 and values.shape[0] == 1:
        values = values[0]
    energy = stored.energy_axis(ds.attrs) or stored.energy_axis(ds.parent_attrs)
    pixel_nm = _pixel_nm(attrs)
    dataset = sidpy.Dataset.from_array(values, title=ds.name)
    if energy is not None:
        channels = values.shape[-1]
        spectral = sidpy.Dimension(energy.values(0, channels), name="energy", units=energy.units or "eV", quantity="energy",
                                   dimension_type="spectral")
        if values.ndim == 1:
            dataset.data_type = "SPECTRUM"
            dataset.set_dimension(0, spectral)
        elif values.ndim in (2, 3):
            dataset.data_type = "SPECTRAL_IMAGE"
            names = ("y", "x") if values.ndim == 3 else ("position",)
            for index, name in enumerate(names):
                dataset.set_dimension(index, _spatial(values.shape[index], name, pixel_nm))
            dataset.set_dimension(values.ndim - 1, spectral)
        else:
            raise _refuse("unsupported_shape", f"{ds.name} has an energy scale but shape {list(values.shape)}")
    elif values.ndim == 2:
        dataset.data_type = "IMAGE"
        dataset.set_dimension(0, _spatial(values.shape[0], "y", pixel_nm))
        dataset.set_dimension(1, _spatial(values.shape[1], "x", pixel_nm))
    elif values.ndim == 3 and "spectrum_image_axes" not in attrs:
        dataset.data_type = "IMAGE_STACK"
        dataset.set_dimension(0, sidpy.Dimension(np.arange(values.shape[0], dtype=float), name="frame", units="frame",
                                                 quantity="time", dimension_type="temporal"))
        dataset.set_dimension(1, _spatial(values.shape[1], "y", pixel_nm))
        dataset.set_dimension(2, _spatial(values.shape[2], "x", pixel_nm))
    elif values.ndim in (1, 3):
        raise _refuse("uncalibrated", f"{ds.name} looks like spectra but records no energy scale "
                      "(energy_offset_ev/energy_dispersion_ev, dispersion/start_energy or axis_offset/axis_scale)")
    else:
        raise _refuse("unsupported_shape", f"{ds.name} has shape {list(values.shape)}, which is not a spectrum, image or stack")
    dataset.metadata = {"experiment": _experiment(attrs)}
    dataset.original_metadata = dict(attrs)
    return dataset


_installed: list[Callable[[str], sidpy.Dataset | None]] = []


def install(client: Callable[[], Any], local_roots: list | None) -> None:
    """Make pyTEMlib's analysis tools in this process read Tiled keys from ``client()`` and files only under
    ``local_roots`` (None: no files). Installing again replaces the previous resolver."""
    for previous in _installed:
        if previous in mcp_analysis._resolvers:
            mcp_analysis._resolvers.remove(previous)
    _installed.clear()
    mcp_analysis.set_allowed_roots([str(root) for root in local_roots] if local_roots else [])
    for resolver in (tiled_resolver(client), h5_resolver(mcp_analysis._check_allowed)):
        mcp_analysis.add_resolver(resolver)
        _installed.append(resolver)


_ASYNCROSCOPY_MARKS = ("acquisition_type", "pixel_size_m", "energy_offset_ev", "spectrum_image_axes", "instrument_id")


def _written_by_asyncroscopy(h5) -> bool:
    if any(mark in h5.attrs for mark in _ASYNCROSCOPY_MARKS):
        return True
    found = []
    h5.visititems(lambda _name, node: found.append(node) if isinstance(node, h5py.Dataset) and any(m in node.attrs for m in _ASYNCROSCOPY_MARKS) else None)
    return bool(found)


def h5_resolver(check_allowed: Callable[[str], None]) -> Callable[[str], sidpy.Dataset | None]:
    """A pyTEMlib resolver for ``path.h5#dataset`` (or ``path.h5``: its first numeric dataset) in an HDF5 file asyncroscopy
    wrote, read with the same rules as a Tiled key. Other HDF5 files (NSID, Nion) are left to pyTEMlib (None)."""

    def resolve(reference: str) -> sidpy.Dataset | None:
        if not isinstance(reference, str):
            return None
        path, _, name = reference.strip().rpartition("#") if "#" in reference and not os.path.exists(reference) else (reference.strip(), "", "")
        if not path.lower().endswith((".h5", ".hdf5")) or not os.path.isfile(path):
            return None
        check_allowed(path)
        with h5py.File(path, "r") as h5:
            if not _written_by_asyncroscopy(h5):
                return None
            if not name:
                numeric = []
                h5.visititems(lambda item, node: numeric.append(item) if isinstance(node, h5py.Dataset) and node.dtype.kind in "biuf" else None)
                if not numeric:
                    raise _refuse("unreadable_file", f"{path} holds no numeric dataset")
                name = numeric[0]
            if name not in h5 or not isinstance(h5[name], h5py.Dataset):
                raise _refuse("unknown_key", f"{path} has no dataset {name!r}")
            node = h5[name]
            loaded = stored._Loaded(np.asarray(node), {**dict(h5.attrs), **dict(node.attrs)})
            parent = {str(k): stored._decode(v) for k, v in dict(node.parent.attrs).items()}
            return to_sidpy(stored.as_dataset(loaded, os.path.basename(path), name, parent))

    return resolve


def tiled_resolver(client: Callable[[], Any]) -> Callable[[str], sidpy.Dataset | None]:
    """A pyTEMlib resolver for references ``key`` or ``key#path`` into the catalogue ``client()`` returns.

    A reference that is an existing file, or whose key is not in the catalogue, is left to pyTEMlib (None).
    """

    def resolve(reference: str) -> sidpy.Dataset | None:
        if not isinstance(reference, str) or not reference.strip() or os.path.exists(reference):
            return None
        key, _, path = reference.strip().partition("#")
        try:
            catalogue = client()
            if catalogue is None or key not in catalogue:
                return None
            node = catalogue[key]
        except Exception:
            return None
        try:
            ds = stored.pick(node, key, path or None)
        except ValueError as exc:
            raise _refuse("unknown_key", str(exc)) from None
        return to_sidpy(ds)

    return resolve
