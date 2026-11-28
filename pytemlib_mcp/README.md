# pytemlib-mcp

pyTEMlib's analysis exposed as [MCP](https://modelcontextprotocol.io) tools, so an agent or a GUI can
call it over a socket. It lives beside pyTEMlib rather than inside it, so pyTEMlib itself does not
depend on `fastmcp`.

- `pytemlib_mcp.analysis`: the analysis functions (EELS, EDS, peak fitting, images, probe, reference
  data). Plain Python, usable and testable without MCP; each takes a dataset reference and returns a dict.
- `pytemlib_mcp.server`: the FastMCP server that registers them as tools.

## Install and run

```bash
pip install -e ./pytemlib_mcp
python -m pytemlib_mcp                                  # SSE on http://127.0.0.1:8003/mcp
python -m pytemlib_mcp --transport streamable-http
```

asyncroscopy's analysis server mounts `pytemlib_mcp.server.mcp` alongside its map tools.

## Tests

```bash
pip install -e "./pytemlib_mcp[test]"
pytest pytemlib_mcp/tests
```

`examples/` holds a hand-written SSE client and the notebook that uses it.
