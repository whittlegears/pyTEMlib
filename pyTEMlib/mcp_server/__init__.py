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

Local HDF5 paths are accepted only under ``--local-root`` directories
(repeatable; default the working directory). ``assemble_map`` reads tiles
lazily and writes the pyramid block by block, so its memory does not grow
with the map; ``--max-canvas-megapixels`` (default 400000) only catches a
run with wrong metadata. See ``map_tools`` for the details.

``create_server`` builds the same server in-process; ``from pyTEMlib.mcp_server import mcp``
gives a default instance (local files, no Tiled URI) built on first use.
"""

from __future__ import annotations

import threading

from fastmcp import FastMCP

from .jobs import ENVELOPE_SCHEMA_PATH, ENVELOPE_VERSION, JobRunner
from .map_store import MapStore
from .map_tools import DEFAULT_MAX_CANVAS_PIXELS, MapTools


def create_server(
    tiled_uri: str | None = None,
    api_key: str | None = None,
    name: str = "pyTEMlib-MCP",
    include_basic: bool = True,
    local_roots: list | None = None,
    max_canvas_pixels: int = DEFAULT_MAX_CANVAS_PIXELS,
    trace_path: str | None = None,
    annotations_path: str | None = None,
) -> tuple[FastMCP, MapTools]:
    """Build a new MCP server; returns (FastMCP app, MapTools) so callers can also use the tools directly.

    ``local_roots``: directories whose HDF5 files may be used as local runs and
    maps (default: the current working directory). ``max_canvas_pixels``:
    largest level-0 map ``assemble_map`` will stitch (it is held in memory). ``trace_path``: a JSON-lines
    file that gets one line per tool call (tool, arguments, the calling agent and run, outcome).
    ``annotations_path``: the JSON-lines event file of the notes people and agents put on datasets
    (annotate, list_annotations, ...); without it the note tools are not offered.

    Every call returns a fresh FastMCP instance, so calling it twice never
    registers a tool twice. The basic tools are mounted from
    ``basic_tools.mcp``, which is left unchanged.
    """
    mcp = FastMCP(name)
    if trace_path:
        from .trace import ToolTrace

        mcp.add_middleware(ToolTrace(trace_path))
    if include_basic:
        from . import basic_tools

        mcp.mount(basic_tools.mcp)
    tools = MapTools(MapStore(tiled_uri, api_key, local_roots=local_roots), max_canvas_pixels=max_canvas_pixels)
    tools.register(mcp)
    from .frames import register as register_frames

    register_frames(mcp, tools.store)
    if annotations_path:
        from .annotation_store import AnnotationStore, register as register_notes

        register_notes(mcp, AnnotationStore(annotations_path), tools.store.dataset_key, tools.store.sample_of)
    return mcp, tools


_default: tuple[FastMCP, MapTools] | None = None
_default_lock = threading.Lock()


def default_server() -> tuple[FastMCP, MapTools]:
    """The module-level server (basic + map tools, local files; Tiled API key from the environment), built once."""
    global _default
    with _default_lock:
        if _default is None:
            _default = create_server()
        return _default


def __getattr__(name: str):
    # ``from pyTEMlib.mcp_server import mcp`` gives the default server, built on first use
    # so importing this package does not load the basic tools' dependencies.
    if name == "mcp":
        return default_server()[0]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ENVELOPE_SCHEMA_PATH", "ENVELOPE_VERSION", "JobRunner", "MapStore", "MapTools", "create_server", "default_server", "mcp"]
