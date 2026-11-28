"""
The analysis MCP server: stitching, scoring, points of interest, notes and placements.

Two sets of tools:

* pyTEMlib's own MCP tools (``pytemlib_mcp.server``: file loading, wavelength and probe
  calculations, element data, diffraction), mounted unchanged.
* the map tools: stitch microscope map runs into zoomable maps, view regions, score regions with
  pluggable scorers and keep points of interest, place finer images on coarser ones, turn pixels
  into stage positions and put notes on datasets, reading and writing through a Tiled server or
  local HDF5 files.

The algorithms are pyTEMlib's (``pyTEMlib.image_tools``: ``register_tiles``, ``assemble``,
``score_windows``, ``register_to_parent``, ``find_blobs``, ``pixel_to_stage``); this package
stores, schedules and serves them. Run it with::

    python -m pytemlib_mcp.analysis_server --tiled http://localhost:9091 --api-key secret --port 8003

Local HDF5 paths are accepted only under ``--local-root`` directories (repeatable; default the
working directory). ``assemble_map`` reads tiles lazily and writes the pyramid block by block, so
its memory does not grow with the map; ``--max-canvas-megapixels`` (default 400000) only catches a
run with wrong metadata. See ``map_tools`` for the details.

``create_server`` builds the same server in-process.
"""

from __future__ import annotations

from .jobs import JobRunner
from .map_store import MapStore
from .map_tools import MapTools
from .server import create_server

__all__ = ["JobRunner", "MapStore", "MapTools", "create_server"]
