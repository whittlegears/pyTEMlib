"""
Command-line bench for field-of-view scorers.

    python -m pyTEMlib.scoring.bench --scorer baseline
    python -m pyTEMlib.scoring.bench --scorer random
    python -m pyTEMlib.scoring.bench --scorer baseline_v2
    python -m pyTEMlib.scoring.bench --scorer my_package.scorers:MyScorer --seeds 0 1 2 --out results.jsonl

``--scorer`` takes a registered name (see ``pyTEMlib.scoring.available_scorers``;
the reference ``random`` is always there) or ``module:attribute`` naming a
scorer class or instance, so a new scorer can be benched before it is
packaged. Prints the result as JSON; ``--out`` also appends it as one line to
a JSONL file. The default seeds are the held-out ones (3-29); seeds 0-2 are
for tuning. Exits 1 when precision@k falls below the threshold on any seed or,
averaged over the seeds, for any feature kind.

The number is a detection proxy (sharp, clean particles found), not a
measure of what a microscopist would choose.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .evaluate import DEFAULT_K, DEFAULT_SEEDS, DEFAULT_THRESHOLD, evaluate, resolve_scorer


def load_scorer(spec: str):
    if ":" not in spec:
        return resolve_scorer(spec)
    module, _, attr = spec.partition(":")
    loaded = getattr(importlib.import_module(module), attr)
    return loaded() if isinstance(loaded, type) else loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyTEMlib.scoring.bench", description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--scorer", default="baseline", help="registered name, 'random', or module:attribute")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", type=Path, help="append the result to this JSONL file")
    parser.add_argument("--brief", action="store_true", help="print only the summary, not every seed")
    args = parser.parse_args(argv)

    result = evaluate(load_scorer(args.scorer), seeds=args.seeds, k=args.k, threshold=args.threshold)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    shown = {key: value for key, value in result.items() if key != "per_seed"} if args.brief else result
    print(json.dumps(shown, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
