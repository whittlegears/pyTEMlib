"""
Pluggable "how interesting is this region?" scorers for map exploration.

A scorer takes one map region (a 2-D array) and its metadata and returns a
value between 0 and 1, a short human-readable reason, and its own version so
scores from different scorers (and versions) can be compared later. Trained
models, hand-written rules and LLM judges all implement the same interface.

Register a scorer from any installed package with an entry point:

    [project.entry-points."pytemlib.scorers"]
    grain_boundaries = "my_package.scorers:GrainBoundaryScorer"

Evaluate any scorer by name with ``python -m pyTEMlib.scoring.bench --scorer <name>``
(``pyTEMlib.scoring.evaluate``): precision@5, overall and per feature kind,
against synthetic maps with known particles (separated, crowded, small, large,
other pixel sizes, low contrast) and deliberately defocused, contaminated or
empty regions. That is a
detection proxy, not a check against what a microscopist would choose; see
``README.md`` in this folder.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

import numpy as np

from .baseline import BaselineScorer, BaselineScorerV2

ENTRY_POINT_GROUP = "pytemlib.scorers"


@runtime_checkable
class Scorer(Protocol):
    name: str
    version: str
    description: str

    def score(self, region: np.ndarray, metadata: dict) -> tuple[float, str]:
        """Return (value in [0, 1], reason)."""


def available_scorers() -> dict[str, Scorer]:
    """Built-in scorers plus any registered under the ``pytemlib.scorers`` entry point."""
    scorers: dict[str, Scorer] = {BaselineScorer.name: BaselineScorer(), BaselineScorerV2.name: BaselineScorerV2()}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry.load()
            scorer = loaded() if isinstance(loaded, type) else loaded
        except Exception as exc:  # a broken plug-in must not take the server down
            print(f"[pyTEMlib.scoring] could not load scorer {entry.name!r}: {exc}")
            continue
        if isinstance(scorer, Scorer):
            scorers[getattr(scorer, "name", entry.name)] = scorer
    return scorers


def score_windows(image: np.ndarray, scorer: Scorer, window: int, stride: int | None = None, metadata: dict | None = None) -> list[dict]:
    """Score every window of a map level; returns dicts with row, col (top-left), value, reason."""
    stride = stride or window
    out = []
    h, w = image.shape[:2]
    if h < window or w < window:
        window = min(h, w)
    for r in range(0, max(1, h - window + 1), stride):
        for c in range(0, max(1, w - window + 1), stride):
            value, reason = scorer.score(image[r : r + window, c : c + window], {**(metadata or {}), "row": r, "col": c, "window": window})
            out.append({"row": r, "col": c, "window": window, "value": float(np.clip(value, 0.0, 1.0)), "reason": reason})
    return out


__all__ = ["BaselineScorer", "BaselineScorerV2", "Scorer", "available_scorers", "score_windows"]
