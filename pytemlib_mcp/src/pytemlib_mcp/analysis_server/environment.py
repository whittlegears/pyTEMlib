"""The environment variables the analysis server reads, with the names and defaults asyncroscopy gives them
(``asyncroscopy.settings``), so a server started by asyncroscopy's launcher behaves as it did inside asyncroscopy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

OUTPUTS_ENV = "ASYNCROSCOPY_OUTPUTS"
PLUGINS_ENV = "ASYNCROSCOPY_PLUGINS"
DEFAULT_LOG_ROTATE_BYTES = 64 * 2**20
DEFAULT_LOG_ROTATE_KEEP = 20


def _number(env: Mapping[str, str], name: str, default, kind):
    """``env[name]`` as ``kind``; a missing or blank value is ``default``. A value that is not a number names the variable."""
    text = env.get(name, "")
    if not text.strip():
        return default
    try:
        return kind(text)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {text!r}") from None


@dataclass(frozen=True)
class Environment:
    outputs_root: Path
    log_rotate_bytes: int
    log_rotate_keep: int
    tiled_api_key: str | None = field(repr=False)
    plain_tiled_api_key: str | None = field(repr=False)
    plugins: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Environment":
        env = os.environ if environ is None else environ
        return cls(
            outputs_root=Path(env.get(OUTPUTS_ENV) or "outputs").resolve(),
            log_rotate_bytes=_number(env, "ASYNCROSCOPY_LOG_ROTATE_BYTES", DEFAULT_LOG_ROTATE_BYTES, int),
            log_rotate_keep=_number(env, "ASYNCROSCOPY_LOG_ROTATE_KEEP", DEFAULT_LOG_ROTATE_KEEP, int),
            tiled_api_key=env.get("ASYNCROSCOPY_TILED_API_KEY"),
            plain_tiled_api_key=env.get("TILED_API_KEY") or None,
            plugins=env.get(PLUGINS_ENV, ""),
        )

    def output(self, *parts: str) -> Path:
        """A path under ``outputs_root``, e.g. ``output("trace", "analysis.jsonl")``."""
        return self.outputs_root.joinpath(*parts)


def get() -> Environment:
    """The environment as it is now."""
    return Environment.from_env()
