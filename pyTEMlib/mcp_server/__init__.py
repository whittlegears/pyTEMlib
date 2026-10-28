"""
pyTEMlib MCP server.

Two sets of tools:

* ``basic_tools``: file loading, wavelength and probe calculations, element
  data and diffraction (the original pyTEMlib MCP server).
* ``map_tools``: stitch microscope map runs into zoomable maps, view regions,
  score regions with pluggable scorers and keep points of interest, reading
  and writing through a Tiled server or local HDF5 files.

Run it with::

    python -m pyTEMlib.mcp_server --tiled http://localhost:9091 --port 8003

``create_server`` builds the same server in-process.
"""

from __future__ import annotations

from fastmcp import FastMCP

from .jobs import ENVELOPE_SCHEMA_PATH, ENVELOPE_VERSION, JobRunner
from .map_store import MapStore
from .map_tools import MapTools


def create_server(tiled_uri: str | None = None, api_key: str | None = None, name: str = "pyTEMlib-MCP", include_basic: bool = True) -> tuple[FastMCP, MapTools]:
    """Build the MCP server; returns (FastMCP app, MapTools) so callers can also use the tools directly."""
    if include_basic:
        from . import basic_tools

        mcp = basic_tools.mcp
    else:
        mcp = FastMCP(name)
    tools = MapTools(MapStore(tiled_uri, api_key))
    tools.register(mcp)
    return mcp, tools


__all__ = ["ENVELOPE_SCHEMA_PATH", "ENVELOPE_VERSION", "JobRunner", "MapStore", "MapTools", "create_server"]
