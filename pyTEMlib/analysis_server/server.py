"""Build the analysis MCP server: pyTEMlib's basic tools, the map tools, pixel/stage frames, and optionally
notes, placements and a per-call tool log."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.transforms import GetToolNext, Transform
from fastmcp.tools import ToolResult
from mcp.types import ToolAnnotations

from ..mcp_analysis import READ_ONLY_TOOLS as _ANALYSIS_TOOLS
from .frames import unknown_key
from .jobs import DEFAULT_MAX_WORKERS
from .map_store import MapStore
from .map_tools import DEFAULT_MAX_CANVAS_PIXELS, MapTools
from .names import CALLER_HEADER, RUN_HEADER
from .tool_log import ToolCallLog


def caller_and_run() -> tuple[str | None, str | None]:
    """The calling agent and its run, from the X-Asyncroscopy-Caller and X-Asyncroscopy-Run headers agents send;
    (None, None) outside an HTTP request."""
    try:
        headers = {k.lower(): v for k, v in (get_http_headers(include_all=True) or {}).items()}
    except Exception:
        return None, None
    return headers.get(CALLER_HEADER.lower()) or None, headers.get(RUN_HEADER.lower()) or None


class ToolLog(Middleware):
    """One line per tool call in a ``ToolCallLog``: tool, arguments, the calling agent and run, outcome.

    The asyncroscopy instrument endpoint keeps the same kind of log, so an analysis call (score_map with
    odd arguments, a stitch that failed) can be read back in the merged trace and matched to the agent
    run that made it. Lines carry ``source: analysis``.
    """

    def __init__(self, path: str | Path):
        self.log = ToolCallLog(path)

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        caller, run = caller_and_run()
        params = context.message
        started, began = time.time(), time.perf_counter()
        try:
            result = await call_next(context)
        except Exception as exc:
            self.log.record(params.name, dict(params.arguments or {}), started, time.perf_counter() - began, error=exc,
                            source="analysis", caller=caller, run=run)
            raise
        text = " ".join(getattr(block, "text", "") for block in (getattr(result, "content", None) or []))
        self.log.record(params.name, dict(params.arguments or {}), started, time.perf_counter() - began, result=text,
                        source="analysis", caller=caller, run=run)
        return result



def missing_key(exc: BaseException) -> str | None:
    """The key of the KeyError behind a tool's error (FastMCP wraps it in a ToolError), or None."""
    seen = exc
    while seen is not None:
        if isinstance(seen, KeyError) and seen.args and isinstance(seen.args[0], str):
            return seen.args[0]
        seen = seen.__cause__ or seen.__context__
    return None


class UnknownKeyReply(Middleware):
    """A tool given a key the store does not have answers with the newest keys to copy, not a bare KeyError.

    qwen3:8b asked list_pois and map_info about "new_map"; both failed with "Error calling tool 'list_pois':
    'new_map'", and it told the user the map did not exist without naming one that does. Only a missing key
    that is one of the call's own arguments is answered this way; any other error is raised as it was."""

    def __init__(self, store):
        self.store = store

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        try:
            return await call_next(context)
        except Exception as exc:
            missing = missing_key(exc)
            if missing is None or missing not in [v for v in (context.message.arguments or {}).values() if isinstance(v, str)]:
                raise
            reply = unknown_key(self.store, missing, "map or dataset")
            return ToolResult(content=json.dumps(reply), structured_content=reply)


# Tools that only read. Clients that gate tool calls read MCP's readOnlyHint and run these without asking.
# The basic tools come from pyTEMlib, which marks none, so they are named here too.
# A test fails when a tool is in neither set.
READ_ONLY = frozenset({
    "get_pytemlib_status", "calculate_electron_wavelength", "list_supported_file_formats", "open_tem_file",
    "list_loaded_datasets", "get_dataset_info", "calculate_probe_size", "get_element_info",
    "analyze_diffraction_pattern", "create_visualization_script", "get_available_tools",
    "analysis_job_status", "analysis_job_results", "map_info", "list_maps", "get_region", "list_pois",
    "list_scorers", "list_tiles", "to_stage", "find_features", "to_pixel", "list_placements",
    "list_annotations", "annotation_history", "match_tiles",
})
READ_ONLY = READ_ONLY | frozenset(tool.__name__ for tool in _ANALYSIS_TOOLS)  # pyTEMlib's analysis tools only compute
WRITES = frozenset({
    "assemble_map", "collect_map_run", "score_map", "analysis_job_abort", "add_poi", "jobs_envelope", "place_image",
    "annotate", "edit_annotation", "delete_annotation", "restore_annotation",
})
if {"segment_sam", "find_atoms_blobnet"} <= {tool.__name__ for tool in _ANALYSIS_TOOLS}:  # learned_tools adds these
    WRITES = WRITES | frozenset({"segment_sam_and_save", "find_atoms_blobnet_and_save"})


class MarkReadOnly(Transform):
    """Lists the tools named in ``names`` with MCP's readOnlyHint set."""

    def __init__(self, names: frozenset[str]):
        self.names = names

    def mark(self, tool):
        if tool is None or tool.name not in self.names:
            return tool
        return tool.model_copy(update={"annotations": ToolAnnotations(read_only_hint=True)})

    async def list_tools(self, tools):
        return [self.mark(tool) for tool in tools]

    async def get_tool(self, name: str, call_next: GetToolNext, *, version=None):
        return self.mark(await call_next(name, version=version))


def create_server(
    tiled_uri: str | None = None,
    api_key: str | None = None,
    name: str = "asyncroscopy-analysis",
    include_basic: bool = True,
    local_roots: list | None = None,
    max_canvas_pixels: int = DEFAULT_MAX_CANVAS_PIXELS,
    trace_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    placements_path: str | Path | None = None,
    max_jobs: int = DEFAULT_MAX_WORKERS,
    local_runs: str | Path | None = None,
) -> tuple[FastMCP, MapTools]:
    """Build a new MCP server; returns (FastMCP app, MapTools) so callers can also use the tools directly.

    ``local_roots``: directories whose HDF5 files may be used as local runs and maps (None: no local
    files, only Tiled keys). ``max_canvas_pixels``: largest level-0 map ``assemble_map`` will stitch.
    ``trace_path``: a JSON-lines file that gets one line per tool call. ``annotations_path``: the
    JSON-lines event file of the notes people and agents put on datasets (annotate, list_annotations,
    ...); without it the note tools are not offered. ``placements_path``: the JSON-lines log of finer
    images placed on coarser ones (place_image, list_placements); without it those tools are not
    offered. ``max_jobs``: stitching and scoring jobs run at once; more wait as "queued".
    ``local_runs``: a folder under ``local_roots`` where collect_map_run saves runs as files. Their maps
    are saved beside them and nothing is written to Tiled.

    Nothing is written anywhere this call is not given: the command line (``python -m
    pyTEMlib.analysis_server``) puts the files under the outputs folder (``$ASYNCROSCOPY_OUTPUTS``). Every call returns a fresh FastMCP instance, so calling it twice never
    registers a tool twice. The basic tools are pyTEMlib's ``pyTEMlib.mcp_server``, mounted unchanged:
    its analysis tools (EELS, EDS, FFT, atoms, peaks, probe) read Tiled keys through ``tiled_datasets`` and
    files only under ``local_roots``. When that pyTEMlib has the learned tools (segment_sam, find_atoms_blobnet),
    ``learned_tools`` adds versions of them that also save the result in Tiled.
    """
    mcp = FastMCP(name)
    if trace_path:
        mcp.add_middleware(ToolLog(trace_path))
    if include_basic:
        from pyTEMlib import mcp_server as basic_tools

        mcp.mount(basic_tools.mcp)
    tools = MapTools(MapStore(tiled_uri, api_key, local_roots=local_roots), max_canvas_pixels=max_canvas_pixels, max_jobs=max_jobs,
                     local_runs=local_runs)
    mcp.add_middleware(UnknownKeyReply(tools.store))
    tools.register(mcp)
    if include_basic and _ANALYSIS_TOOLS:
        from .tiled_datasets import install as install_tiled_datasets

        install_tiled_datasets(lambda: tools.store.client if tiled_uri else None, local_roots)
        from .learned_tools import register as register_learned

        register_learned(mcp, lambda: tools.store.client if tiled_uri else None, tools.store.sample_of, tools.store.is_local,
                         tools.store.version_suffix)
    from .frames import register as register_frames

    register_frames(mcp, tools.store)
    if annotations_path:
        from .annotation_store import AnnotationStore, register as register_notes

        register_notes(mcp, AnnotationStore(annotations_path), tools.store.dataset_key, tools.store.sample_of)
    if placements_path:
        from .placement_tools import PlacementLog, register as register_placements

        register_placements(mcp, tools.store, PlacementLog(placements_path))
    mcp.add_transform(MarkReadOnly(READ_ONLY))
    return mcp, tools
