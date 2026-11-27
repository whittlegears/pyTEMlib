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
"highlights". Points are appended in batches; a local batch is one string
dataset of JSON, so its size has no limit (an HDF5 attribute's is 64 KB).
Adding points costs the points added. Replacing the highlights (a scorer's
tagged regions scored again) writes a new highlights group and switches to it
in one metadata update, without touching the EDS cells; replacing every point
builds a new layer the same way. Replaced groups and layers are kept and
listed in the store's ``replaced_highlights`` and ``layers``. Maps stitched
before point stores existed kept their points inside the map; those are read
from there until the run's store is made, which copies them in once
(``migrated_from``).

Each map, point store and local file has its own lock (``_Locks``), so work on
one map never waits for another.

The same functions work against a Tiled server (keys) or the filesystem
(paths). Local paths are accepted only under the store's ``local_roots``
(none unless given), so a client cannot read, or write a map next to, an
arbitrary HDF5 file on the host.
"""

from __future__ import annotations

from . import pyramid, runs
from .poi import CELL_KINDS, HIGHLIGHTS, POI_CELLS
from .pyramid import level_shape_of, pixel_to_stage, stage_box_to_pixels
from .runs import LazyTile, LazyTiles, LocalPathRefused, RunData, RunInfo, TileSource
from .store import AXES, MapStore

__all__ = [
    "AXES",
    "CELL_KINDS",
    "HIGHLIGHTS",
    "LazyTile",
    "LazyTiles",
    "LocalPathRefused",
    "MapStore",
    "POI_CELLS",
    "RunData",
    "RunInfo",
    "TileSource",
    "level_shape_of",
    "pixel_to_stage",
    "pyramid",
    "runs",
    "stage_box_to_pixels",
]
