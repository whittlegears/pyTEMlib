# pytemlib-mcp

An [MCP](https://modelcontextprotocol.io) server that lets an agent open TEM files with
pyTEMlib and look up electron-optics and element reference data. It lives beside pyTEMlib
rather than inside it so pyTEMlib does not depend on `fastmcp`.

## Install and run

```bash
pip install -e ./pytemlib_mcp
pytemlib-mcp                     # Streamable HTTP on http://127.0.0.1:8003/mcp
pytemlib-mcp --transport sse     # legacy SSE, same address
pytemlib-mcp --transport stdio
```

## Tools

| Tool | What it does |
| :-- | :-- |
| `get_pytemlib_status` | Installed pyTEMlib version |
| `calculate_electron_wavelength` | Relativistic wavelength via `utilities.get_wavelength` |
| `calculate_probe_size` | Diffraction-limited radius 0.61·λ/α (no aberrations) |
| `get_element_info` | Z, atomic weight and edge onsets from `utilities.get_x_sections` |
| `open_tem_file` | Reads a file with `file_tools.open_file` and keeps it loaded |
| `list_loaded_datasets` | Datasets from opened files |
| `get_dataset_info` | Type, shape, axes and metadata keys of one dataset |
| `create_visualization_script` | A script that reopens the dataset and calls `dataset.plot()` |

## Client

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8003/mcp") as client:
        print(await client.call_tool("get_element_info", {"element": "Ti"}))

asyncio.run(main())
```

## Tests

```bash
pip install -e "./pytemlib_mcp[test]"
pytest pytemlib_mcp/tests
```
