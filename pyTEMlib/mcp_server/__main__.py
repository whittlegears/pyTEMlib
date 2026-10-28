"""Command line: ``python -m pyTEMlib.mcp_server --tiled http://localhost:9091 --port 8003``."""

from __future__ import annotations

import argparse

from . import create_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="pyTEMlib MCP server: analysis and map tools.")
    parser.add_argument("--tiled", default="", help="Tiled server URI; without it, tools take local file paths.")
    parser.add_argument("--api-key", default=None, help="Tiled API key (default: $ASYNCROSCOPY_TILED_API_KEY or $TILED_API_KEY).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--transport", default="streamable-http", choices=["streamable-http", "sse", "stdio"])
    args = parser.parse_args(argv)
    mcp, _tools = create_server(args.tiled or None, args.api_key)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        print(f"pyTEMlib MCP server at http://{args.host}:{args.port}/mcp (Tiled: {args.tiled or 'none, local files'})", flush=True)
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
