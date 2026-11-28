"""Where the pyTEMlib MCP servers listen unless a flag or environment variable says otherwise.

asyncroscopy's launcher (``startup_scripts/run_pytemlib.py``) checks that its own defaults match these.
"""

HOST = "127.0.0.1"
PORT = 8003
MCP_PATH = "/mcp"
