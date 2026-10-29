"""
MCP tools for stitching, viewing, scoring and annotating microscope maps.

Agents never receive full arrays: tools take and return keys (Tiled keys or
file paths), stage coordinates in metres, scores and short previews. Pixels
move between Tiled and this server directly.

Memory: ``assemble_map`` reads the run's tiles lazily in chunks through a
bounded cache and writes the pyramid block by block as the stitcher makes it
(see ``mosaic.assemble`` and ``map_store``), so its memory does not grow
with the map. ``max_canvas_pixels`` (default 400 gigapixels;
``--max-canvas-megapixels``) only guards against a run whose metadata is
wrong; stitch part of a run with ``box_m`` to get a result sooner.

Stitching accuracy: the search for each overlap is bounded by the stage's
stated uncertainty (``stage_sigma_m``) plus the most the sample can drift
between the two tiles (``max_drift_m_per_s``), and a drift over time is fitted
with the tile positions (``mosaic.register_tiles``). The map's metadata
records the fitted drift, the residuals of the matches used and the tiles
that fell back to their stage position.
"""

from __future__ import annotations

import io
import json
import time
import uuid
import weakref
from typing import Any

import numpy as np
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image as MCPImage

from pyTEMlib.image_tools.mosaic import assemble, level_shapes, pyramid_levels, register_tiles, stage_canvas_shape
from pyTEMlib.scoring import available_scorers, score_windows

from .jobs import JobRunner
from .map_store import MapStore, level_shape_of, pixel_to_stage, stage_box_to_pixels

PREVIEW_MAX_SIDE = 512
MAX_STATUS_WAIT_S = 25.0  # analysis_job_status waits stay under MCP client timeouts
DEFAULT_MAX_CANVAS_PIXELS = 400_000_000_000  # level-0 pixels; memory is bounded, so this only catches wrong metadata
SCORE_MAX_SIDE = 4096  # score_map reads the finest level no larger than this, never a giant level 0 whole
BLOCK = 512  # pyramid block and chunk side

# FastMCP app -> the MapTools whose tools it serves; guards against registering the tools twice.
_REGISTERED: "weakref.WeakKeyDictionary[FastMCP, MapTools]" = weakref.WeakKeyDictionary()


def _png(array: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    from PIL import Image

    data = np.asarray(array, dtype=np.float64)
    step = max(1, int(np.ceil(max(data.shape) / max_side)))
    data = data[::step, ::step]
    lo, hi = np.percentile(data, [1, 99.5]) if data.size else (0.0, 1.0)
    scaled = np.clip((data - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    buffer = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _loads(request: str | dict) -> dict:
    if isinstance(request, dict):
        return request
    return json.loads(request) if request.strip().startswith("{") else {"id": request.strip()}


def eds_pois(run, placement) -> list[dict]:
    """A point of interest for every EDS point spectrum of a layout-2 run, at its tile's stitched centre."""
    keys = run.source.dataset("spectrum_keys")
    if keys is None:
        return []
    out = []
    for k, tile in enumerate(run.tiles):
        key = keys[tile.index]
        key = key.decode() if isinstance(key, bytes) else str(key)
        if not key:
            continue
        row, col = placement.positions_px[k] + np.array(tile.shape) / 2.0 - 0.5
        x, y = placement.pixel_to_stage(row, col)
        out.append({"id": uuid.uuid4().hex[:8], "x_m": float(x), "y_m": float(y), "kind": "eds_point", "source": "run", "score": None, "scorer_version": None, "reason": f"EDS point spectrum of tile {tile.index}", "links": [key]})
    return out


def stitch_quality(placement) -> dict:
    """How well the tiles agree after the solve, for the map's metadata and the job summary."""
    used = [m for m in placement.matches if m.used]
    out = {"matches_used": len(used), "matches_found": len(placement.matches)}
    if used:
        i = np.array([m.i for m in used])
        j = np.array([m.j for m in used])
        offsets = np.array([m.offset_px for m in used])
        p, s = placement.positions_px, placement.stage_positions_px
        residuals = np.hypot(*((p[j] - p[i]) - (s[j] - s[i] + offsets)).T)
        out["median_residual_px"] = round(float(np.median(residuals)), 3)
        out["p95_residual_px"] = round(float(np.percentile(residuals, 95)), 3)
    if placement.fallback is not None:
        out["fallback_tiles"] = int(placement.fallback.sum())
    drift = placement.drift_summary()
    if drift is not None:
        out["drift"] = drift
    return out


class MapTools:
    def __init__(self, store: MapStore, max_canvas_pixels: int = DEFAULT_MAX_CANVAS_PIXELS):
        self.store = store
        self.max_canvas_pixels = int(max_canvas_pixels)
        self.jobs = JobRunner()
        self.jobs.register("assemble_map", self._check_assemble, self._run_assemble)
        self.jobs.register("score_map", self._check_score, self._run_score)

    # -------------------------------------------------------------- tasks
    def _check_assemble(self, plan: dict) -> dict:
        # Metadata and shapes only: reading every tile image here would block the request; _run_assemble loads them.
        run = self.store.scan_run(plan["run"], plan.get("box_m"))
        canvas = stage_canvas_shape([t.stage_xy_m for t in run.tiles], [t.shape[:2] for t in run.tiles], [t.pixel_size_m for t in run.tiles])
        self._check_canvas(canvas)
        return {"steps": len(run.tiles) + 1, "tiles": len(run.tiles), "canvas_shape": list(canvas), "canvas_megapixels": round(canvas[0] * canvas[1] / 1e6, 3)}

    def _check_canvas(self, canvas: tuple[int, int]) -> None:
        pixels = int(canvas[0]) * int(canvas[1])
        if pixels > self.max_canvas_pixels:
            raise ValueError(
                f"the stitched map would be {canvas[0]} x {canvas[1]} = {pixels / 1e6:.1f} megapixels at full resolution, over this "
                f"server's limit of {self.max_canvas_pixels / 1e6:.1f} megapixels; check the run's tile metadata, "
                "stitch part of the run with box_m, or start the server with a larger --max-canvas-megapixels"
            )

    def _run_assemble(self, plan: dict, progress) -> dict:
        run = self.store.open_run(plan["run"], plan.get("box_m"))
        try:
            return self._assemble(run, plan, progress)
        finally:
            run.close()

    def _assemble(self, run, plan: dict, progress) -> dict:
        # Progress: the overlapping pairs matched (the slow part), then the pyramid blocks written.
        steps = {"pairs": 0}

        def on_pair(done: int, total: int) -> None:
            steps["pairs"] = total
            progress(done, 2 * total + 1)  # raises Aborted when the job is aborted, which stops registration

        progress(0, len(run.tiles) + 1)
        refine = bool(plan.get("refine", True))
        placement = register_tiles(
            run.tiles,
            stage_sigma_m=float(plan.get("stage_sigma_m", 200e-9)),
            refine=refine,
            on_pair=on_pair,
            max_drift_m_per_s=float(plan.get("max_drift_m_per_s", 0.0)),
            drift_knot_s=float(plan.get("drift_knot_s", 120.0)),
        )
        pairs = steps["pairs"]
        self._check_canvas(placement.canvas_shape)  # refinement can grow the canvas by a few pixels
        pois = self.store.existing_pois(plan["run"])
        if plan.get("eds_pois"):
            # Every EDS point spectrum of the run, at its tile's stitched centre, linked to its key;
            # the previous stitch's EDS points are replaced, not repeated.
            pois = [p for p in pois if not (p.get("kind") == "eds_point" and p.get("source") == "run")] + eds_pois(run, placement)
        levels = pyramid_levels(placement.canvas_shape)
        writer = self.store.map_writer(plan["run"])
        try:
            assemble(run.tiles, placement, writer, block=BLOCK, levels=levels, on_block=lambda done, total: progress(pairs + done * pairs // max(total, 1), 2 * pairs + 1))
        except BaseException:
            writer.abandon()
            raise
        quality = stitch_quality(placement)
        metadata = {
            "levels": levels,
            "downsample": [2**k for k in range(levels)],
            "pixel_size_m": placement.pixel_size_m,
            "origin_xy_m": list(placement.origin_xy_m),
            "canvas_shape": list(placement.canvas_shape),
            "level_shapes": [list(v) for v in level_shapes(placement.canvas_shape, levels)],
            "tiles": len(run.tiles),
            "stitched_at": time.time(),
            "refined": refine,
            **quality,
        }
        if len(run.tiles) <= 10_000:  # per-tile positions are kept for small maps; large ones keep the summary only
            metadata["tile_positions_px"] = placement.positions_px.round(3).tolist()
        key = self.store.write_map(plan["run"], writer, metadata, pois)
        progress(2 * pairs + 1)
        summary = {"tiles": len(run.tiles), "levels": levels, "canvas_shape": list(placement.canvas_shape), **quality}
        if plan.get("eds_pois"):
            summary["eds_pois"] = sum(1 for p in pois if p.get("kind") == "eds_point" and p.get("source") == "run")
        return {"result_keys": {"map": key}, "summary": summary}

    def _check_score(self, plan: dict) -> dict:
        scorer = plan.get("scorer", "baseline")
        if scorer not in available_scorers():
            raise ValueError(f"unknown scorer {scorer!r}; available: {sorted(available_scorers())}")
        self.store.map_metadata(plan["map"])
        return {"steps": 1}

    def _run_score(self, plan: dict, progress) -> dict:
        meta = self.store.map_metadata(plan["map"])
        scorer = available_scorers()[plan.get("scorer", "baseline")]
        # The finest level that fits SCORE_MAX_SIDE, so scoring a map of any size reads a bounded image.
        fits = next((k for k in range(meta["levels"]) if max(level_shape_of(meta, k)) <= SCORE_MAX_SIDE), meta["levels"] - 1)
        level = max(int(plan.get("level", min(2, meta["levels"] - 1))), fits)
        image = self.store.read_level(plan["map"], level)
        pixel = meta["pixel_size_m"] * 2**level
        if "window_m" in plan:
            window = int(round(float(plan["window_m"]) / pixel))
        else:  # aim for about a 6 x 6 grid of candidate regions
            window = min(image.shape[:2]) // 6
        window = int(np.clip(window, 16, max(16, min(image.shape[:2]))))
        scores = score_windows(image, scorer, window, stride=max(1, window // 2), metadata={"level": level, "pixel_size_m": pixel})
        scores.sort(key=lambda s: s["value"], reverse=True)
        top: list[dict] = []
        for s in scores:  # keep chosen regions at least one window apart
            if all(abs(s["row"] - t["row"]) >= window or abs(s["col"] - t["col"]) >= window for t in top):
                top.append(s)
            if len(top) >= int(plan.get("top_k", 5)):
                break
        added = []
        for s in top:
            x, y = pixel_to_stage(meta, s["row"] + s["window"] / 2.0, s["col"] + s["window"] / 2.0, level)
            poi = {
                "id": uuid.uuid4().hex[:8], "x_m": x, "y_m": y, "kind": "candidate", "source": f"scorer:{scorer.name}",
                "score": s["value"], "scorer_version": scorer.version, "reason": s["reason"], "window_m": s["window"] * pixel, "links": [],
            }
            added.append(poi)
        self.store.add_pois(plan["map"], added)
        progress(1)
        return {"result_keys": {"map": plan["map"]}, "summary": {"scored_windows": len(scores), "added_pois": added}}

    # -------------------------------------------------------------- tools
    def register(self, mcp: FastMCP) -> None:
        """Add the map tools to ``mcp``. Registering the same MapTools again is a no-op; a second MapTools is refused."""
        owner = _REGISTERED.get(mcp)
        if owner is self:
            return
        if owner is not None:
            raise RuntimeError("this FastMCP app already has map tools from another MapTools; build a new app with create_server()")
        _REGISTERED[mcp] = self
        tools = self

        @mcp.tool()
        def assemble_map(
            run: str,
            refine: bool = True,
            box_m: list[float] | None = None,
            stage_sigma_m: float = 2e-7,
            max_drift_m_per_s: float = 0.0,
            eds_pois: bool = True,
        ) -> dict[str, Any]:
            """Stitch a map run into a zoomable multi-resolution map, in the background.

            run: the run's Tiled key or file path (Mapper results -> result_keys.run or run_file).
            refine: refine stage positions by matching overlaps (turn off for blank samples).
            box_m: optional [x_min, x_max, y_min, y_max] stage box (metres) to stitch only part.
            stage_sigma_m: 1-sigma error of the reported stage positions; bounds the overlap search.
            max_drift_m_per_s: most the sample drifts (m/s); above 0 a drift over time is fitted.
            eds_pois: mark every EDS point spectrum of the run as a point of interest linked to it.
            Returns a job_id; poll analysis_job_status, then analysis_job_results gives result_keys.map.
            """
            plan = {"task": "assemble_map", "run": run, "refine": refine, "stage_sigma_m": stage_sigma_m, "max_drift_m_per_s": max_drift_m_per_s, "eds_pois": eds_pois}
            if box_m:
                plan["box_m"] = box_m
            reserved = tools.jobs.reserve(plan)
            if "error" in reserved:
                return reserved
            return {**tools.jobs.start(reserved["reservation_id"]), "estimate": reserved["estimate"]}

        @mcp.tool()
        def score_map(map: str, scorer: str = "baseline", top_k: int = 5, window_m: float | list[float] | None = None, level: int | None = None) -> dict[str, Any]:
            """Rank regions of a stitched map with a scorer and add the best ones as points of interest, in the background.

            map: the map key or path (assemble_map results -> result_keys.map).
            scorer: one of list_scorers(). top_k: how many points of interest to add.
            window_m: side of a square region in metres (default: a sixth of the map, so about 6 x 6 candidate regions).
            Returns a job_id; analysis_job_results(job_id).summary.added_pois lists the new points.
            """
            plan = {"task": "score_map", "map": map, "scorer": scorer, "top_k": top_k}
            if isinstance(window_m, list):
                # Models often write a size as [width, height]; regions are square, so use the larger side.
                window_m = max((float(v) for v in window_m), default=None)
            if window_m:
                plan["window_m"] = window_m
            if level is not None:
                plan["level"] = level
            reserved = tools.jobs.reserve(plan)
            if "error" in reserved:
                return reserved
            return tools.jobs.start(reserved["reservation_id"])

        @mcp.tool()
        def analysis_job_status(job_id: str, wait_s: float = 0.0) -> dict[str, Any]:
            """Progress of a stitching or scoring job started by assemble_map or score_map; with wait_s (at most 25) it first waits for the job to finish. Not for map acquisitions: those started with Mapper_start use Mapper_job_status or wait_for_job."""
            if wait_s > 0:
                # Waiting here saves a small model dozens of polling turns.
                return tools.jobs.wait(job_id, timeout=min(float(wait_s), MAX_STATUS_WAIT_S))
            return tools.jobs.status(job_id)

        @mcp.tool()
        def analysis_job_results(job_id: str) -> dict[str, Any]:
            """Result of a stitching or scoring job (assemble_map, score_map): result_keys.map and a summary. For a map acquisition use Mapper_results."""
            return tools.jobs.results(job_id)

        @mcp.tool()
        def analysis_job_abort(job_id: str) -> dict[str, Any]:
            """Stop a running stitching or scoring job (not a map acquisition: use Mapper_abort)."""
            return tools.jobs.abort(job_id)

        @mcp.tool()
        def map_info(map: str) -> dict[str, Any]:
            """Size, levels, stage extent and number of points of interest of a stitched map."""
            meta = tools.store.map_metadata(map)
            h, w = meta["canvas_shape"]
            x_min, y_min = pixel_to_stage(meta, -0.5, -0.5)
            x_max, y_max = pixel_to_stage(meta, h - 0.5, w - 0.5)
            return {
                "map": map, "run": meta.get("run"), "levels": meta["levels"], "canvas_shape": meta["canvas_shape"],
                "pixel_size_m": meta["pixel_size_m"], "stage_box_m": [x_min, x_max, y_min, y_max],
                "points_of_interest": tools.store.poi_count(map),
                "stitching": {k: meta[k] for k in ("matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "drift") if k in meta},
            }

        @mcp.tool()
        def get_region(map: str, box_m: list[float], level: int | None = None) -> ToolResult:
            """Look at part of a map: a PNG preview of the stage box [x_min, x_max, y_min, y_max] (metres).

            level: 0 is full resolution; higher levels are 2x smaller each. Default picks the
            finest level that keeps the preview under 512 pixels.
            """
            meta = tools.store.map_metadata(map)
            if level is None:
                width_m = box_m[1] - box_m[0]
                level = 0
                while level < meta["levels"] - 1 and width_m / (meta["pixel_size_m"] * 2**level) > PREVIEW_MAX_SIDE:
                    level += 1
            rows, cols = stage_box_to_pixels(meta, box_m, level)
            region = tools.store.read_level(map, level, rows, cols)
            info = {"map": map, "level": level, "box_m": box_m, "shape": list(region.shape), "pixel_size_m": meta["pixel_size_m"] * 2**level}
            if region.size == 0:
                return ToolResult(content=[json.dumps({**info, "note": "box is outside the map"})], structured_content=info)
            info.update({"mean": float(region.mean()), "max": float(region.max())})
            return ToolResult(content=[json.dumps(info), MCPImage(data=_png(region), format="png")], structured_content=info)

        @mcp.tool()
        def list_pois(map: str, kind: str | None = None, box_m: list[float] | None = None, limit: int = 200) -> dict[str, Any]:
            """Points of interest on a map (stage x_m, y_m in metres, kind, source, score, reason, links).

            box_m: only those inside [x_min, x_max, y_min, y_max]. At most ``limit`` are returned; ``total`` says how many match.
            """
            pois = tools.store.list_pois(map, kind=kind, box_m=box_m)
            return {"total": len(pois), "pois": pois[: max(0, int(limit))]}

        @mcp.tool()
        def add_poi(map: str, x_m: float, y_m: float, kind: str = "note", reason: str = "", links: list[str] | None = None) -> dict[str, Any]:
            """Mark a stage position on a map, e.g. where an EDS spectrum or a high-magnification image was taken.

            links: Tiled keys of data acquired there. Refuses points outside the map.
            """
            meta = tools.store.map_metadata(map)
            h, w = meta["canvas_shape"]
            x_min, y_min = pixel_to_stage(meta, -0.5, -0.5)
            x_max, y_max = pixel_to_stage(meta, h - 0.5, w - 0.5)
            if not (x_min <= x_m <= x_max and y_min <= y_m <= y_max):
                return {"error": {"code": "outside_map", "message": f"({x_m:.3e}, {y_m:.3e}) is outside the map [{x_min:.3e}, {x_max:.3e}] x [{y_min:.3e}, {y_max:.3e}]"}}
            poi = {"id": uuid.uuid4().hex[:8], "x_m": float(x_m), "y_m": float(y_m), "kind": kind, "source": "agent", "score": None, "scorer_version": None, "reason": reason, "links": list(links or [])}
            tools.store.add_pois(map, [poi])
            return poi

        @mcp.tool()
        def list_scorers() -> list[dict[str, str]]:
            """Scorers available to score_map: name, version and what they look for."""
            return [{"name": s.name, "version": s.version, "description": s.description} for s in available_scorers().values()]

        @mcp.tool()
        def list_tiles(run: str, box_m: list[float] | None = None) -> list[dict[str, Any]]:
            """Tiles of a map run (index, stage position, source key), optionally only those in a stage box."""
            data = tools.store.scan_run(run, box_m)
            return [{"index": t.index, "stage_xy_m": list(t.stage_xy_m), "shape": list(t.shape), "source_key": t.attrs.get("source_key")} for t in data.tiles]

        @mcp.tool()
        def jobs_envelope(verb: str, request: str) -> dict[str, Any]:
            """Generic job envelope for other clients: verb is reserve, start, status, abort, results or release; request is its JSON."""
            body = _loads(request)
            if verb == "reserve":
                return tools.jobs.reserve(body.get("plan", {}), body.get("caller"))
            if verb in {"start", "release"}:
                return getattr(tools.jobs, verb)(body.get("reservation_id", body.get("id", "")))
            if verb in {"status", "abort", "results"}:
                return getattr(tools.jobs, verb)(body.get("job_id", body.get("id", "")))
            return tools.jobs.error("bad_request", f"unknown verb {verb!r}")
