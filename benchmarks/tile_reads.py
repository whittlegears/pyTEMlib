"""How often stitching reads each block of tiles through a bounded cache, with a remote-like read cost.

A synthetic run (see mosaic_scale.py) is written as a layout-2 run file, opened through MapStore,
whose TileSource reads CHUNK_TILES consecutive tiles per request, each request delayed by
``--read-latency-ms`` as a remote Tiled would be, with a cache much smaller than the run. It reports
chunk reads per chunk (1.0 = each read once per matching pass) and the time spent waiting.

    python benchmarks/tile_reads.py --sides 40,100 --cache-mb 16 --read-latency-ms 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from pyTEMlib.image_tools import mosaic
from pyTEMlib.mcp_server import map_store
from pyTEMlib.mcp_server.map_store import MapStore

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mosaic_scale", here / "mosaic_scale.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def write_run(path: Path, run) -> None:
    n, tile = len(run), run.tile
    with h5py.File(path, "w") as h5:
        h5.attrs["run_layout"] = 2
        data = h5.create_dataset("tiles", shape=(n, tile, tile), dtype=np.uint16, chunks=(1, tile, tile))
        data.attrs["pixel_size_m"] = bench.PIXEL
        for k in range(n):
            data[k] = run.image(k)
        h5["written"] = np.ones(n, bool)
        stage = np.zeros((n, 5))
        stage[:, 0] = (run.reported[:, 1] + tile / 2) * bench.PIXEL
        stage[:, 1] = (run.reported[:, 0] + tile / 2) * bench.PIXEL
        h5["stage_reported_m"] = stage
        h5["acquired_s"] = run.times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sides", default="40,100")
    parser.add_argument("--cache-mb", type=float, default=16)
    parser.add_argument("--read-latency-ms", type=float, default=20)
    parser.add_argument("--strip-px", type=int, default=mosaic.STRIP_PX)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    map_store.CACHE_BYTES = int(args.cache_mb * 2**20)
    mosaic.STRIP_PX = args.strip_px
    real_read = map_store.TileSource._read

    def slow_read(self, positions):
        time.sleep(args.read_latency_ms / 1000)
        return real_read(self, positions)

    map_store.TileSource._read = slow_read
    for side in (int(v) for v in args.sides.split(",")):
        run = bench.SyntheticRun(side, 64, 0.3, 0.05, 0.02, 3, density=2e-3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.h5"
            write_run(path, run)
            store = MapStore(local_roots=[folder])
            data = store.open_run(str(path))
            t0 = time.time()
            placement = mosaic.register_tiles(data.tiles, stage_sigma_m=0.05 * 64 * bench.PIXEL, max_drift_m_per_s=0.06 * bench.PIXEL)
            elapsed = time.time() - t0
            chunks = -(-len(run) // map_store.CHUNK_TILES)
            row = {
                "tiles": len(run), "chunk_tiles": map_store.CHUNK_TILES, "chunks": chunks, "cache_mb": args.cache_mb,
                "run_mb": round(len(run) * 64 * 64 * 2 / 2**20, 1), "strip_px": args.strip_px,
                "chunk_reads": data.source.reads, "reads_per_chunk": round(data.source.reads / chunks, 2),
                "read_wait_s": round(data.source.reads * args.read_latency_ms / 1000, 1), "register_s": round(elapsed, 1),
                "median_px": bench.errors(placement.positions_px, run.truth)["median_px"],
            }
            data.close()
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
