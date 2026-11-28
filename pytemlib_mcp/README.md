# pytemlib-mcp

pyTEMlib's analysis exposed as [MCP](https://modelcontextprotocol.io) tools, so an agent or a GUI can
call it over a socket. It lives beside pyTEMlib rather than inside it, so pyTEMlib itself does not
depend on `fastmcp`.

- `pytemlib_mcp.analysis`: the analysis functions (EELS, EDS, peak fitting, images, probe, reference
  data). Plain Python, usable and testable without MCP; each takes a dataset reference and returns a dict.
- `pytemlib_mcp.server`: the FastMCP server that registers them as tools.
- `pytemlib_mcp.analysis_server`: the analysis server that agents and Angstrom connect to. It mounts
  `pytemlib_mcp.server` and adds the map tools: run reading from Tiled or local HDF5, stitching jobs,
  map versions and pyramids, points of interest, frames, placements and notes. `docs/map_contract.md`
  describes the map data it reads and writes.

## Install and run

```bash
pip install -e ./pytemlib_mcp
python -m pytemlib_mcp                                  # SSE on http://127.0.0.1:8003/mcp
python -m pytemlib_mcp --transport streamable-http
```

```bash
python -m pytemlib_mcp.analysis_server --tiled http://localhost:9091 --api-key secret --port 8003
```

## Tests

```bash
pip install -e "./pytemlib_mcp[test]"
python -m pytest pytemlib_mcp/tests   # from the pyTEMlib root; the map tests share tests/conftest.py
```

`examples/` holds a hand-written SSE client and the notebook that uses it.
