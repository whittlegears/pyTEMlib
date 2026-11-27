"""Command line: ``python -m pyTEMlib.analysis_server --tiled http://localhost:9091 --api-key secret --port 8003``.

The tool log, the note events and the placement log go under the outputs folder
(``$ASYNCROSCOPY_OUTPUTS``, default ``outputs`` here) unless given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import environment
from .jobs import DEFAULT_MAX_WORKERS
from .map_tools import DEFAULT_MAX_CANVAS_PIXELS
from .server import create_server


def parser(env: environment.Environment) -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(prog="python -m pyTEMlib.analysis_server", description="pyTEMlib analysis MCP server: map, scoring, placement and note tools.")
    out.add_argument("--tiled", default="", help="Tiled server URI; without it, tools take local file paths.")
    out.add_argument("--api-key", default=None, help="Tiled API key (default: $ASYNCROSCOPY_TILED_API_KEY or $TILED_API_KEY).")
    out.add_argument("--host", default="127.0.0.1")
    out.add_argument("--port", type=int, default=8003)
    out.add_argument("--transport", default="streamable-http", choices=["streamable-http", "sse", "stdio"])
    out.add_argument(
        "--local-root", action="append", default=None, metavar="DIR",
        help="Directory whose HDF5 files tools may read and write as local paths (repeatable; default: the working directory).",
    )
    out.add_argument(
        "--max-canvas-megapixels", type=float, default=DEFAULT_MAX_CANVAS_PIXELS / 1e6,
        help="Largest full-resolution map assemble_map will stitch, a guard against wrong run metadata (default: %(default)g).",
    )
    out.add_argument(
        "--local-runs", default="", metavar="DIR",
        help="Folder (under a --local-root) where collect_map_run writes its runs as HDF5 files, with their maps stitched beside them; "
             "nothing is then written to Tiled, which is only read (empty: runs and maps go to Tiled).",
    )
    out.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_WORKERS, help="Stitching and scoring jobs run at once; more wait queued (default: %(default)s).")
    out.add_argument(
        "--trace", default=str(env.output("trace", "analysis.jsonl")),
        help="JSON-lines file with one line per tool call: tool, arguments, calling agent and run, outcome (empty: none; default: %(default)s).",
    )
    out.add_argument(
        "--annotations", default=str(env.output("annotations", "notes.jsonl")),
        help="JSON-lines event file of the notes people and agents put on datasets (empty: no note tools; default: %(default)s).",
    )
    out.add_argument(
        "--placements", default=str(env.output("placements", "placements.jsonl")),
        help="JSON-lines log of finer images placed on coarser ones (empty: no placement tools; default: %(default)s).",
    )
    return out


def build(args: argparse.Namespace, env: environment.Environment):
    """Build the server from the parsed options. Returns the FastMCP app and the MapTools."""
    api_key = args.api_key or env.tiled_api_key or env.plain_tiled_api_key
    local_roots = args.local_root if args.local_root is not None else [str(Path.cwd())]
    return create_server(
        args.tiled or None, api_key, local_roots=local_roots,
        max_canvas_pixels=int(args.max_canvas_megapixels * 1e6), trace_path=args.trace or None,
        annotations_path=args.annotations or None, placements_path=args.placements or None, max_jobs=args.max_jobs,
        local_runs=args.local_runs or None,
    )


def serve(mcp, tools, args: argparse.Namespace, note: str = "") -> None:
    """Serve until stopped. ``note`` is an extra line printed at startup."""
    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        else:
            print(f"pyTEMlib analysis server at http://{args.host}:{args.port}/mcp (Tiled: {args.tiled or 'none, local files'}; "
                  f"local roots: {', '.join(map(str, tools.store.local_roots))}"
                  + (f"; gathered runs and their maps are saved in {Path(args.local_runs).resolve()}" if args.local_runs else "") + ")", flush=True)
            if note:
                print(note, flush=True)
            mcp.run(transport=args.transport, host=args.host, port=args.port)
    finally:
        tools.jobs.shutdown()


def main(argv: list[str] | None = None) -> None:
    env = environment.get()
    args = parser(env).parse_args(argv)
    mcp, tools = build(args, env)
    serve(mcp, tools, args)


if __name__ == "__main__":
    main()
