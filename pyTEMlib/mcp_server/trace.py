"""One JSON line per tool call: which agent asked for what, with which arguments, and what came of it.

The asyncroscopy instrument endpoint keeps the same kind of trace; with this one an
analysis call (score_map with odd arguments, a stitch that failed) can be read back
and matched to the agent run that made it, by the X-Asyncroscopy-Caller and
X-Asyncroscopy-Run headers agents send.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext

ARGUMENT_CHARS = 2000  # arguments longer than this (a pasted array) are cut in the trace
RESULT_CHARS = 500


class ToolTrace(Middleware):
    """Appends {ts, tool, caller, run, arguments, ok, seconds, result | error} to ``path`` for every tool call."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        headers = {k.lower(): v for k, v in (get_http_headers(include_all=True) or {}).items()}
        params = context.message
        record: dict[str, Any] = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "analysis",
            "tool": params.name,
            "caller": headers.get("x-asyncroscopy-caller"),
            "run": headers.get("x-asyncroscopy-run"),
            "arguments": json.dumps(params.arguments or {}, default=str)[:ARGUMENT_CHARS],
        }
        began = time.perf_counter()
        try:
            result = await call_next(context)
        except Exception as exc:
            record.update(ok=False, seconds=round(time.perf_counter() - began, 3), error=f"{type(exc).__name__}: {exc}"[:RESULT_CHARS])
            self._write(record)
            raise
        text = " ".join(getattr(block, "text", "") for block in (getattr(result, "content", None) or []))
        record.update(ok=True, seconds=round(time.perf_counter() - began, 3), result=text[:RESULT_CHARS])
        self._write(record)
        return result

    def _write(self, record: dict) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
