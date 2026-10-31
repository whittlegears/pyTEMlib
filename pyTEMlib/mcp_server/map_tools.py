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

Nothing is deleted: every stitch is a new map version (``list_maps``), and a
run's points of interest are kept once for all its versions (see ``map_store``).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
import time
import uuid
import weakref
from typing import Any

import numpy as np
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image as MCPImage

from pyTEMlib.version import __version__
from pyTEMlib.image_tools.mosaic import assemble, level_shapes, pyramid_levels, register_tiles, stage_canvas_shape
from pyTEMlib.scoring import available_scorers, score_windows

from .jobs import JobRunner
from .map_store import MapStore, level_shape_of, pixel_to_stage, stage_box_to_pixels

PREVIEW_MAX_SIDE = 512
MAX_STATUS_WAIT_S = 25.0  # analysis_job_status waits stay under MCP client timeouts
PROGRESS_STEPS = 1000  # assemble_map reports progress in these units
DEFAULT_MAX_CANVAS_PIXELS = 400_000_000_000  # level-0 pixels; memory is bounded, so this only catches wrong metadata
SCORE_MAX_SIDE = 4096  # score_map reads the finest level no larger than this, never a giant level 0 whole
BLOCK = 512  # pyramid block and chunk side
UNPLACED_LISTED = 1000  # unplaced tile indices kept in a map's metadata (the count is always there)

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
    """A point of interest for every EDS point spectrum of a layout-2 run, at its tile's stitched centre.

    Ids are the run and tile, so stitching the same run again gives the same points.
    """
    keys = run.source.dataset("spectrum_keys")
    if keys is None:
        return []
    info = run.tiles.info
    centres = placement.positions_px + info.shapes / 2.0 - 0.5
    run_name = Path(str(run.run)).name
    out = []
    for k, index in enumerate(info.index):
        key = keys[int(index)]
        key = key.decode() if isinstance(key, bytes) else str(key)
        if not key:
            continue
        x, y = placement.pixel_to_stage(centres[k, 0], centres[k, 1])
        out.append({"id": f"{run_name}:eds:{int(index)}", "x_m": float(x), "y_m": float(y), "kind": "eds_point", "source": "run", "score": None, "scorer_version": None, "reason": f"EDS point spectrum of tile {int(index)}", "links": [key]})
    return out


def inputs_sha256(plan: dict, info, stage_sigma_m: float) -> str:
    """A hash of what a stitch was made from: the run, which of its tiles were used, the settings and this pyTEMlib.

    Two map versions with the same hash were stitched from the same inputs the same way."""
    settings = {k: plan.get(k) for k in ("run", "refine", "box_m", "max_drift_m_per_s", "drift_knot_s", "scan_linear")}
    digest = hashlib.sha256(json.dumps({**settings, "stage_sigma_m": float(stage_sigma_m), "pytemlib": __version__}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(np.asarray(info.index, dtype=np.int64)).tobytes())
    return digest.hexdigest()


def stitch_quality(placement) -> dict:
    """How well the tiles agree after the solve, for the map's metadata and the job summary."""
    used = [m for m in placement.matches if m.used]
    out = {"matches_used": len(used), "matches_found": len(placement.matches)}
    if used:
        i = np.array([m.i for m in used])
        j = np.array([m.j for m in used])
        offsets = np.array([m.offset_px for m in used])
        p, s = placement.positions_px, placement.stage_positions_px
        # Matches are measured in the tiles' own pixels: compare them with the drawn positions taken back there.
        residuals = np.hypot(*(((p[j] - p[i]) @ np.asarray(placement.linear).T) - (s[j] - s[i] + offsets)).T)
        out["median_residual_px"] = round(float(np.median(residuals)), 3)
        out["p95_residual_px"] = round(float(np.percentile(residuals, 95)), 3)
    if placement.fallback is not None:
        out["fallback_tiles"] = int(placement.fallback.sum())  # unplaced: listed, not drawn
        out["unplaced_tiles"] = [int(k) for k in np.nonzero(placement.fallback)[0][:UNPLACED_LISTED]]
    if placement.linear_estimate is not None:
        out["scan_linear"] = {
            "applied": not np.allclose(placement.linear, np.eye(2)),
            "matrix": np.round(placement.linear_estimate, 6).tolist(),
            "stderr": np.round(placement.linear_stderr, 6).tolist(),
        }
    if placement.gains is not None:
        out["tile_gain"] = {"min": round(float(placement.gains.min()), 4), "max": round(float(placement.gains.max()), 4)}
    if placement.solver:
        out["solver"] = placement.solver
    drift = placement.drift_summary()
    if drift is not None:
        out["drift"] = drift
    return out


class MapTools:
    def __init__(self, store: MapStore, max_canvas_pixels: int = DEFAULT_MAX_CANVAS_PIXELS):
        self.store = store
        self.max_canvas_pixels = int(max_canvas_pixels)
        self.jobs = JobRunner()
        # One job per run's map at a time: a second assemble (a retry after a timeout, a resumed
        # Mapper) is refused as busy with the running job's id, which the caller then waits for.
        self.jobs.register("assemble_map", self._check_assemble, self._run_assemble, resource=lambda plan: f"map of {plan['run']}")
        self.jobs.register("score_map", self._check_score, self._run_score, resource=lambda plan: f"scoring of {plan['map']}")

    # -------------------------------------------------------------- tasks
    def _check_assemble(self, plan: dict) -> dict:
        # Constant time, whatever the run's size: the request must answer at once. The job itself
        # reads the tile positions and checks the canvas before reading any pixel.
        facts = self.store.check_run(plan["run"])
        return {"steps": PROGRESS_STEPS, "tiles": facts["tiles"], "layout": facts["layout"]}

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
        # Progress in PROGRESS_STEPS units that only go up: registration (both matching passes)
        # takes the first 70 %, writing the pyramid the next 29 %, storing the map the rest.
        passes = {"pass": 0, "last": 0}

        def on_pair(done: int, total: int) -> None:
            if done < passes["last"]:
                passes["pass"] += 1
            passes["last"] = done
            share = min(1.0, (passes["pass"] + done / max(total, 1)) / 2.0)
            progress(int(0.7 * PROGRESS_STEPS * share), PROGRESS_STEPS)  # raises Aborted when the job is aborted

        progress(0, PROGRESS_STEPS)
        refine = bool(plan.get("refine", True))
        info = run.tiles.info
        self._check_canvas(stage_canvas_shape(info.stage_xy_m, info.shapes, info.pixel_sizes_m))
        sigma = plan.get("stage_sigma_m")
        if sigma is None:  # 5 % of the field of view: a stage error a usual 15 % overlap can still absorb
            sigma = 0.05 * float(np.median(info.shapes[:, 1])) * float(info.pixel_sizes_m[0])
        placement = register_tiles(
            run.tiles,
            stage_sigma_m=float(sigma),
            refine=refine,
            on_pair=on_pair,
            max_drift_m_per_s=float(plan.get("max_drift_m_per_s", 0.0)),
            drift_knot_s=float(plan.get("drift_knot_s", 120.0)),
            scan_linear=np.asarray(plan["scan_linear"], dtype=np.float64) if plan.get("scan_linear") is not None else None,
        )
        progress(int(0.7 * PROGRESS_STEPS), PROGRESS_STEPS)
        self._check_canvas(placement.canvas_shape)  # refinement can grow the canvas by a few pixels
        # Every EDS point spectrum of the run, at its tile's stitched centre, linked to its key; the run's
        # EDS points from an earlier stitch are replaced, not repeated. Every other point stays with the run.
        eds = eds_pois(run, placement) if plan.get("eds_pois") else None
        previous = self.store.current_map(plan["run"])
        levels = pyramid_levels(placement.canvas_shape)
        writer = self.store.map_writer(plan["run"])
        try:
            assemble(run.tiles, placement, writer, block=BLOCK, levels=levels, on_block=lambda done, total: progress(int((0.7 + 0.29 * done / max(total, 1)) * PROGRESS_STEPS), PROGRESS_STEPS))
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
            "stage_sigma_m": float(sigma),
            "inputs_sha256": inputs_sha256(plan, info, float(sigma)),
            "pytemlib_version": __version__,
            **quality,
        }
        if len(run.tiles) <= 10_000:  # per-tile positions are kept for small maps; large ones keep the summary only
            metadata["tile_positions_px"] = placement.positions_px.round(3).tolist()
            # With the raw tiles, these rebuild the map exactly: drawn value = gain * counts + offset.
            metadata["tile_gains"] = np.round(placement.gains, 5).tolist() if placement.gains is not None else None
            metadata["tile_offsets"] = np.round(placement.offsets, 4).tolist() if placement.offsets is not None else None
            metadata["scan_linear_applied"] = np.asarray(placement.linear).round(6).tolist()
        progress(int(0.99 * PROGRESS_STEPS), PROGRESS_STEPS)  # last chance to abort before the new version is stored
        key = self.store.write_map(plan["run"], writer, metadata, eds_points=eds, previous=previous)
        progress(PROGRESS_STEPS, PROGRESS_STEPS)
        summary = {"tiles": len(run.tiles), "levels": levels, "canvas_shape": list(placement.canvas_shape), **quality}
        if previous:
            summary["previous_map"] = previous
        if eds is not None:
            summary["eds_pois"] = len(eds)
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
        tag = plan.get("tag")
        added = []
        for k, s in enumerate(top):
            x, y = pixel_to_stage(meta, s["row"] + s["window"] / 2.0, s["col"] + s["window"] / 2.0, level)
            poi = {
                "id": f"{tag}:region:{k}" if tag else uuid.uuid4().hex[:8], "x_m": x, "y_m": y, "kind": "candidate", "source": f"scorer:{scorer.name}",
                "score": s["value"], "scorer_version": scorer.version, "reason": s["reason"], "window_m": s["window"] * pixel, "links": [],
            }
            if tag:
                poi["tag"] = tag
            added.append(poi)
        progress(0)  # an abort requested while scoring stops here, before any point is written
        if tag:
            # The same tag's earlier regions (a retried or resumed scoring) are replaced, not repeated.
            self.store.update_pois(plan["map"], lambda current: [p for p in current if p.get("tag") != tag] + added)
        else:
            self.store.add_pois(plan["map"], added)
        progress(1)
        return {"result_keys": {"map": plan["map"]}, "summary": {"scored_windows": len(scores), "added_pois": added}}

    def map_info(self, map_key: str) -> dict[str, Any]:
        meta = self.store.map_metadata(map_key)
        h, w = meta["canvas_shape"]
        x_min, y_min = pixel_to_stage(meta, -0.5, -0.5)
        x_max, y_max = pixel_to_stage(meta, h - 0.5, w - 0.5)
        info = {
            "map": map_key, "run": meta.get("run"), "tiles": meta.get("tiles"), "levels": meta["levels"], "canvas_shape": meta["canvas_shape"],
            "pixel_size_m": meta["pixel_size_m"], "stage_box_m": [x_min, x_max, y_min, y_max],
            "points_of_interest": self.store.poi_count(map_key),
            "stitching": {k: meta[k] for k in ("matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "unplaced_tiles", "scan_linear", "tile_gain", "drift") if k in meta},
            "id": meta.get("id"), "stitched_at": meta.get("stitched_at"), "inputs_sha256": meta.get("inputs_sha256"),
        }
        if meta.get("run"):
            try:
                versions = self.store.list_maps(meta["run"])
            except Exception:  # the run is gone or unreadable: the map itself is still described
                versions = []
            if versions:
                info["versions"] = {"count": len(versions), "newest": versions[0]["map"], "this_is_newest": versions[0]["map"] == map_key}
        return info

    def list_pois(self, map_key: str, kind: str | None = None, box_m: list[float] | None = None, limit: int = 200) -> dict[str, Any]:
        """{"total": how many match, "pois": at most ``limit`` of them}; the shape is the same for every filter,
        except that a filter matching nothing also gives how many points the map has (``total_on_map``) and, for a
        kind, the kinds it has (``kinds_on_map``). A box with no area ([0, 0, 0, 0]) is no box."""
        limit = max(0, int(limit))
        # Models ask for kind="all" or list several kinds in one string; a filter that matches nothing reads as "no points".
        kinds = {k for k in re.split(r"[\s,;|]+", kind or "") if k} - {"all", "any", "*"}
        # Models fill an optional box with zeros: a box with no area holds no point, so it means "no box".
        if box_m is not None and not (len(box_m) == 4 and box_m[1] > box_m[0] and box_m[3] > box_m[2]):
            box_m = None
        if box_m is None and not kinds:
            return {"total": self.store.poi_count(map_key), "pois": self.store.list_pois(map_key, limit=limit)}
        if not kinds:
            pois = self.store.list_pois(map_key, box_m=box_m)
        else:  # one read per kind, so a highlight kind never reads the map's spatial cells
            pois = [p for k in sorted(kinds) for p in self.store.list_pois(map_key, kind=k, box_m=box_m)]
        out = {"total": len(pois), "pois": pois[:limit]}
        if not pois:  # a filter that matches nothing must not read as an empty map
            out["total_on_map"] = self.store.poi_count(map_key)
            if kinds and box_m is None:
                out["kinds_on_map"] = self.store.poi_kinds(map_key)
        return out

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
            stage_sigma_m: float | None = None,
            max_drift_m_per_s: float = 0.0,
            eds_pois: bool = True,
            scan_linear: list[list[float]] | None = None,
        ) -> dict[str, Any]:
            """Stitch a map run into a zoomable multi-resolution map, in the background. Each call makes a new map
            version; earlier versions and the run's points of interest are kept.

            run: the run's Tiled key or file path (Mapper results -> result_keys.run or run_file).
            refine: refine stage positions by matching overlaps (turn off for blank samples).
            box_m: optional [x_min, x_max, y_min, y_max] stage box (metres) to stitch only part.
            stage_sigma_m: 1-sigma error of the reported stage positions; bounds the overlap search (default 5 % of the field of view).
            max_drift_m_per_s: most the sample drifts (m/s); above 0 a drift over time is fitted.
            eds_pois: mark every EDS point spectrum of the run as a point of interest linked to it.
            scan_linear: optional 2 x 2 calibration of the scan's shared stretch and turn for this scan setting (a large
            map's stitching.scan_linear.matrix); without it the map measures its own, which needs a large map.
            Returns a job_id; poll analysis_job_status, then analysis_job_results gives result_keys.map.
            """
            plan = {"task": "assemble_map", "run": run, "refine": refine, "max_drift_m_per_s": max_drift_m_per_s, "eds_pois": eds_pois}
            if stage_sigma_m is not None:
                plan["stage_sigma_m"] = stage_sigma_m
            if box_m:
                plan["box_m"] = box_m
            if scan_linear is not None:
                plan["scan_linear"] = scan_linear
            reserved = tools.jobs.reserve(plan)
            if "error" in reserved:
                return reserved
            return {**tools.jobs.start(reserved["reservation_id"]), "estimate": reserved["estimate"]}

        @mcp.tool()
        def score_map(map: str, scorer: str = "baseline", top_k: int = 5, window_m: float | list[float] | None = None, level: int | None = None, tag: str | None = None) -> dict[str, Any]:
            """Rank regions of a stitched map with a scorer and add the best ones as points of interest, in the background.

            map: the map key or path (assemble_map results -> result_keys.map).
            scorer: one of list_scorers(). top_k: how many points of interest to add.
            window_m: side of a square region in metres (default: a sixth of the map, so about 6 x 6 candidate regions).
            tag: optional; scoring again with the same tag replaces the regions it added before (safe to retry).
            Returns a job_id; analysis_job_results(job_id).summary.added_pois lists the new points.
            """
            plan = {"task": "score_map", "map": map, "scorer": scorer, "top_k": top_k}
            if tag:
                plan["tag"] = tag
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
            """Tiles, size, levels, stage extent and number of points of interest of a stitched map.

            Given a Mapper run instead, the refusal names the run's stitched map."""
            return tools.map_info(map)

        @mcp.tool()
        def list_maps(run: str, limit: int = 20) -> dict[str, Any]:
            """Every stitched version of a run's map, newest first (each assemble_map makes a new one; none is deleted).

            Returns {"total", "maps": [{"map", "id", "stitched_at", "inputs_sha256", "tiles"}, at most ``limit``]}."""
            maps = tools.store.list_maps(run)
            return {"total": len(maps), "maps": maps[: max(0, int(limit))]}

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

            kind: only these kinds (e.g. "eds_point" or "candidate,revisit"); leave it out for every kind.
            box_m: only those inside [x_min, x_max, y_min, y_max] (only that part of the map is read); leave it out for all.
            Returns {"total": how many match, "pois": [at most ``limit`` of them]}, always this shape; when a filter
            matches nothing, "total_on_map" says how many points the map has.
            """
            return tools.list_pois(map, kind=kind, box_m=box_m, limit=limit)

        @mcp.tool()
        def add_poi(map: str, x_m: float, y_m: float, kind: str = "note", reason: str = "", links: list[str] | None = None, id: str | None = None) -> dict[str, Any]:
            """Mark a stage position on a map, e.g. where an EDS spectrum or a high-magnification image was taken.

            links: Tiled keys of data acquired there. Refuses points outside the map.
            id: optional; adding a point with an id already on the map changes nothing (safe to retry).
            """
            meta = tools.store.map_metadata(map)
            h, w = meta["canvas_shape"]
            x_min, y_min = pixel_to_stage(meta, -0.5, -0.5)
            x_max, y_max = pixel_to_stage(meta, h - 0.5, w - 0.5)
            if not (x_min <= x_m <= x_max and y_min <= y_m <= y_max):
                return {"error": {"code": "outside_map", "message": f"({x_m:.3e}, {y_m:.3e}) is outside the map [{x_min:.3e}, {x_max:.3e}] x [{y_min:.3e}, {y_max:.3e}]"}}
            poi = {"id": id or uuid.uuid4().hex[:8], "x_m": float(x_m), "y_m": float(y_m), "kind": kind, "source": "agent", "score": None, "scorer_version": None, "reason": reason, "links": list(links or [])}
            tools.store.add_pois(map, [poi], dedupe=id is not None)
            return poi

        @mcp.tool()
        def list_scorers() -> list[dict[str, str]]:
            """Scorers available to score_map: name, version and what they look for."""
            return [{"name": s.name, "version": s.version, "description": s.description} for s in available_scorers().values()]

        @mcp.tool()
        def list_tiles(run: str, box_m: list[float] | None = None, limit: int = 200) -> dict[str, Any]:
            """Tiles of a map run (index, stage position, shape), optionally only those in a stage box; at most ``limit``, with the total."""
            info = tools.store.scan_run(run, box_m)
            shown = range(min(len(info), max(0, int(limit))))
            return {
                "total": len(info),
                "tiles": [{"index": int(info.index[k]), "stage_xy_m": [float(v) for v in info.stage_xy_m[k]], "shape": [int(v) for v in info.shapes[k]]} for k in shown],
            }

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
