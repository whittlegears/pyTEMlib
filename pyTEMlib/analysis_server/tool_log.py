"""The per-call tool log: one JSON line per tool call, in the format of asyncroscopy's ``ToolCallLog``
(``asyncroscopy.trace.writer``), so asyncroscopy's merged trace reads this server's calls with its own."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp.server.dependencies import get_http_headers

from . import environment
from .fileio import SequencedAppender
from .names import CALLER_HEADER, RUN_HEADER

RESULT_PREVIEW_CHARS = 500


def _plain(value: Any) -> Any:
    """numpy scalars and arrays as plain numbers and lists; anything else as text."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_plain))


def _appender(path: Path) -> SequencedAppender:
    """An appender for ``path`` that rotates as the environment says (``ASYNCROSCOPY_LOG_ROTATE_BYTES``, ``_KEEP``)."""
    env = environment.get()
    return SequencedAppender(path, rotate_bytes=env.log_rotate_bytes, keep=env.log_rotate_keep)


def caller_and_run() -> tuple[str | None, str | None]:
    """The calling agent and its run, from the X-Asyncroscopy-Caller and X-Asyncroscopy-Run headers agents send;
    (None, None) outside an HTTP request."""
    try:
        headers = {k.lower(): v for k, v in (get_http_headers(include_all=True) or {}).items()}
    except Exception:
        return None, None
    return headers.get(CALLER_HEADER.lower()) or None, headers.get(RUN_HEADER.lower()) or None


class ToolCallLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._out = _appender(self.path)

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        started: float,
        duration_s: float,
        result: Any = None,
        error: BaseException | None = None,
        **context: Any,
    ) -> None:
        """One line per call. ``context``: e.g. caller and run, written when not None."""
        # Never let a logging failure fail the device call.
        try:
            row = {
                "ts": started,
                "iso": datetime.fromtimestamp(started).isoformat(),
                "tool": tool,
                "arguments": _json_safe(arguments),
                "duration_s": round(duration_s, 6),
                "ok": error is None,
                "error": f"{type(error).__name__}: {error}" if error is not None else None,
                "result": None if result is None else (result if isinstance(result, str) else json.dumps(_json_safe(result), default=str))[:RESULT_PREVIEW_CHARS],
                **{k: _json_safe(v) for k, v in context.items() if v is not None},
            }
            self._out.append(row)
        except Exception as exc:
            warnings.warn(f"tool log {self.path} not written: {exc}", RuntimeWarning, stacklevel=2)
