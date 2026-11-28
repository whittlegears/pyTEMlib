"""
MCP tools for stitching, viewing, scoring and annotating microscope maps.

Agents never receive full arrays: tools take and return keys (Tiled keys or
file paths), stage coordinates in metres, scores and short previews. Pixels
move between Tiled and this server directly.

Memory: ``assemble_map`` reads the run's tiles lazily in chunks through a
bounded cache and writes the pyramid block by block as the stitcher makes it
(see ``image_mosaic.assemble`` and ``map_store``), so its memory does not grow
with the map. ``max_canvas_pixels`` (default 400 gigapixels;
``--max-canvas-megapixels``) only guards against a run whose metadata is
wrong; stitch part of a run with ``box_m`` to get a result sooner.

Stitching accuracy: the search for each overlap is bounded by the stage's
stated uncertainty (``stage_sigma_m``) plus the most the sample can drift
between the two tiles (``max_drift_m_per_s``), and a drift over time is fitted
with the tile positions (``image_mosaic.register_tiles``). The map's metadata
records the fitted drift, the residuals of the matches used and the tiles
that fell back to their stage position.

Nothing is deleted: every stitch is a new map version (``list_maps``), and a
run's points of interest are kept once for all its versions (see ``map_store``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image as MCPImage

from pyTEMlib import __version__
from pytemlib_mcp.analysis import to_png
from pyTEMlib.image_tools import (
    BaselineScorer,
    assemble,
    level_shapes,
    pyramid_levels,
    register_tiles,
    score_windows,
    select_regions,
    stage_canvas_shape,
    stitch_quality,
)
from pyTEMlib.image_tools.image_mosaic import pair_shift

from .frames import FrameError, frame_of, image_dataset, image_node
from .jobs import DEFAULT_MAX_WORKERS, JobRunner
from .map_store import CELL_KINDS, MapStore, level_shape_of, pixel_to_stage, stage_box_to_pixels
from .map_store.poi import poi_kind_labels
from .registry import Registry

PREVIEW_MAX_SIDE = 512
MAX_STATUS_WAIT_S = 25.0  # analysis_job_status waits stay under MCP client timeouts
PROGRESS_STEPS = 1000  # assemble_map reports progress in these units
DEFAULT_MAX_CANVAS_PIXELS = 400_000_000_000  # level 0 pixels. Memory is bounded, so this only catches wrong metadata
SCORE_MAX_SIDE = 4096  # score_map reads the finest level no larger than this
BLOCK = 512  # pyramid block and chunk side
MATCH_MIN_SCORE = 0.3  # lowest correlation match_tiles accepts, the same bar the stitcher uses
MATCH_MIN_OVERLAP_PX = 16

# The scorers score_map offers, by name. pyTEMlib's baseline is one of them.
# Plug-ins add theirs with @SCORERS.register or the asyncroscopy.scorers entry point.
SCORERS: Registry[type] = Registry("asyncroscopy.scorers", kind="scorer")
SCORERS.register(BaselineScorer)

# Maps each FastMCP app to the MapTools it serves, so the tools are never registered twice.
_REGISTERED: "weakref.WeakKeyDictionary[FastMCP, MapTools]" = weakref.WeakKeyDictionary()


def _loads(request: str | dict) -> dict:
    if isinstance(request, dict):
        return request
    return json.loads(request) if request.strip().startswith("{") else {"id": request.strip()}


def eds_pois(run, placement) -> list[dict]:
    """A point of interest for every EDS point spectrum of a run (its ``spectrum_keys``), where on its tile it was taken.

    That place is the run's ``spectra_at``, [x, y] fractions of the tile image from the left and from the top, the same
    for every tile. A run without it is a Mapper run, whose spectra are taken at the tile centre. The point follows the
    tile's stitched position and the scan's stretch and turn the stitch removed. Ids are the run and tile, so stitching
    the same run again gives the same points.
    """
    keys = run.source.dataset("spectrum_keys")
    if keys is None:
        return []
    at = run.source.attribute("spectra_at")
    fx, fy = (0.5, 0.5) if at is None else (float(at[0]), float(at[1]))
    info = run.tiles.info
    shapes = np.asarray(info.shapes, dtype=np.float64)
    offsets = (np.array([fy, fx]) - 0.5) * shapes @ np.linalg.inv(np.asarray(placement.linear, dtype=np.float64)).T
    points = placement.centres_px(shapes) + offsets
    run_name = Path(str(run.run)).name
    out = []
    for k, index in enumerate(info.index):
        key = keys[int(index)]
        key = key.decode() if isinstance(key, bytes) else str(key)
        if not key:
            continue
        x, y = placement.pixel_to_stage(points[k, 0], points[k, 1])
        out.append({"id": f"{run_name}:eds:{int(index)}", "x_m": float(x), "y_m": float(y), "kind": "eds_point", "source": "run", "score": None, "scorer_version": None, "reason": f"EDS point spectrum of tile {int(index)}", "links": [key]})
    return out


DRIFT_UNMEASURED = "too few image matches between tiles to measure drift: it is unknown, not zero"


def drift_measured(summary: dict) -> bool:
    """Whether the stitch's drift fit rests on image matches: at least two per knot (x and y at each).

    With fewer the fit is its prior, no drift: a 100-tile map with 1 match reported 0.00017 nm (qwen3:8b gave
    it to the user as metres), one with 2 matches 145.6 nm."""
    drift = summary.get("drift")
    if not isinstance(drift, dict):
        return False
    return int(summary.get("matches_used") or 0) >= 2 * len(drift.get("times_s") or [])


def measured_drift_only(quality: dict) -> dict:
    """A stitch's quality with a note that drift is unknown whenever its matches did not measure it (no fit, or a fit
    resting on too few matches, which is dropped).

    Left out silently, qwen3:8b read the missing drift as none and answered "no movement was detected"."""
    if drift_measured(quality):
        return quality
    return {**{k: v for k, v in quality.items() if k != "drift"}, "drift_unmeasured": DRIFT_UNMEASURED}


def finite_or_none(value: Any) -> Any:
    """Return ``value`` with each number that is not finite replaced by None, which JSON can carry."""
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def inputs_sha256(plan: dict, info, stage_sigma_m: float) -> str:
    """A hash of what a stitch was made from: the run, which of its tiles were used, the settings and this pyTEMlib.

    Two map versions with the same hash were stitched from the same inputs the same way."""
    settings = {k: plan.get(k) for k in ("run", "refine", "box_m", "max_drift_m_per_s", "drift_knot_s", "scan_linear")}
    digest = hashlib.sha256(json.dumps({**settings, "stage_sigma_m": float(stage_sigma_m), "pytemlib": __version__}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(np.asarray(info.index, dtype=np.int64)).tobytes())
    return digest.hexdigest()


class MapTools:
    def __init__(self, store: MapStore, max_canvas_pixels: int = DEFAULT_MAX_CANVAS_PIXELS, max_jobs: int = DEFAULT_MAX_WORKERS,
                 local_runs: str | Path | None = None):
        self.store = store
        # Gathered runs are saved as HDF5 files in this folder. None means they go to Tiled.
        self.local_runs = Path(local_runs) if local_runs else None
        self.max_canvas_pixels = int(max_canvas_pixels)
        self.jobs = JobRunner(max_workers=max_jobs)
        # Only one job per run's map at a time. A second assemble is refused as busy
        # with the running job's id, which the caller then waits for.
        self.jobs.register("assemble_map", self._check_assemble, self._run_assemble, resource=lambda plan: f"map of {plan['run']}")
        self.jobs.register("score_map", self._check_score, self._run_score, resource=lambda plan: f"scoring of {plan['map']}")

    def _check_assemble(self, plan: dict) -> dict:
        # This must answer at once whatever the run's size. The job itself
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
        # Progress only goes up. Registration takes the first 70 %,
        # writing the pyramid the next 29 % and storing the map the rest.
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
        if sigma is None:  # 5 % of the field of view, a stage error that a usual 15 % overlap can still absorb
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
        # Each EDS point spectrum of the run goes where it was taken on its stitched tile, linked to its key.
        # They replace the run's EDS points from an earlier stitch. Every other point stays with the run.
        eds = eds_pois(run, placement) if plan.get("eds_pois") else None
        previous = self.store.current_map(plan["run"])
        levels = pyramid_levels(placement.canvas_shape)
        writer = self.store.map_writer(plan["run"])
        try:
            assemble(run.tiles, placement, writer, block=BLOCK, levels=levels, on_block=lambda done, total: progress(int((0.7 + 0.29 * done / max(total, 1)) * PROGRESS_STEPS), PROGRESS_STEPS))
        except BaseException:
            writer.abandon()
            raise
        quality = measured_drift_only(finite_or_none(stitch_quality(placement)))
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
        if len(run.tiles) <= 10_000:  # small maps keep per-tile positions and large ones keep only the summary
            metadata["tile_positions_px"] = placement.positions_px.round(3).tolist()
            # With the raw tiles these rebuild the map exactly. The drawn value is gain * counts + offset.
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

    def match_tiles(self, image: str, neighbour: str, offset_xy_m: list[float], fov_m: float, search_m: float | None = None,
                    min_score: float = MATCH_MIN_SCORE) -> dict[str, Any]:
        """Check that ``image`` overlaps ``neighbour`` where it should. A tile loop calls this as each tile is taken.

        ``offset_xy_m`` is where the centre of ``image`` should be from the centre of ``neighbour``, as [x, y] in
        metres with rows along +y and columns along +x. ``fov_m`` is the width of both images. The overlap is searched
        within ``search_m`` of that offset, by default a tenth of the field of view, with the stitcher's pair matching.
        It is accepted when the two overlaps correlate to ``min_score``. Nothing is written.
        """
        if not (isinstance(fov_m, (int, float)) and math.isfinite(fov_m) and fov_m > 0):
            raise ValueError("fov_m must be the width of the images in metres, above 0")
        if not (isinstance(offset_xy_m, (list, tuple)) and len(offset_xy_m) >= 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in offset_xy_m[:2])):
            raise ValueError("offset_xy_m must be [x, y] in metres")
        try:
            a = np.asarray(image_node(self.store, neighbour).read(), dtype=np.float64)
            b = np.asarray(image_node(self.store, image).read(), dtype=np.float64)
        except (KeyError, FrameError) as exc:
            raise ValueError(f"an image could not be read: {exc}") from exc
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("match_tiles compares two single images")
        pixel = float(fov_m) / a.shape[1]
        predicted = np.array([float(offset_xy_m[1]) / pixel, float(offset_xy_m[0]) / pixel])
        height = int(min(a.shape[0], predicted[0] + b.shape[0]) - max(0.0, predicted[0]))
        width = int(min(a.shape[1], predicted[1] + b.shape[1]) - max(0.0, predicted[1]))
        asked = {"image": image, "neighbour": neighbour, "expected_xy_m": [float(v) for v in offset_xy_m[:2]]}
        if height < MATCH_MIN_OVERLAP_PX or width < MATCH_MIN_OVERLAP_PX:
            return {**asked, "matched": False, "score": None, "reason": "the two images do not overlap at that offset"}
        if not (a.std() > 0 and b.std() > 0):
            return {**asked, "matched": False, "score": None, "reason": "an image is blank"}
        bound = min((float(search_m) if search_m else 0.1 * float(fov_m)) / pixel, 0.6 * min(height, width))
        shift = pair_shift(a, b, predicted, bound, (height, width), MATCH_MIN_OVERLAP_PX)
        if shift is None:
            return {**asked, "matched": False, "score": None, "reason": "no clear match in the overlap"}
        dr, dc = int(round(shift[0])), int(round(shift[1]))
        r0, r1, c0, c1 = max(0, dr), min(a.shape[0], dr + b.shape[0]), max(0, dc), min(a.shape[1], dc + b.shape[1])
        if r1 - r0 < MATCH_MIN_OVERLAP_PX or c1 - c0 < MATCH_MIN_OVERLAP_PX:
            return {**asked, "matched": False, "score": None, "reason": "no clear match in the overlap"}
        one, two = a[r0:r1, c0:c1] - a[r0:r1, c0:c1].mean(), b[r0 - dr:r1 - dr, c0 - dc:c1 - dc] - b[r0 - dr:r1 - dr, c0 - dc:c1 - dc].mean()
        norm = math.sqrt(float((one * one).sum()) * float((two * two).sum()))
        score = float((one * two).sum() / norm) if norm > 0 else 0.0
        found = [float(shift[1]) * pixel, float(shift[0]) * pixel]
        result = {**asked, "matched": score >= float(min_score), "score": round(score, 4), "offset_xy_m": found,
                  "error_xy_m": [found[0] - float(offset_xy_m[0]), found[1] - float(offset_xy_m[1])]}
        if not result["matched"]:
            result["reason"] = f"the overlaps correlate to {score:.2f}, under {float(min_score):.2f}"
        return result

    def collect_run(self, image_keys: list[str], plan: dict | None = None, stage_xy_m: list[list[float]] | None = None,
                    fov_m: float | None = None, spectrum_keys: list[str] | None = None,
                    spectra_at: list[float] | None = None) -> dict[str, Any]:
        """Gather single images into one map run in Tiled. The run is a ``tiles`` container of placed arrays.

        Each image is placed by what its acquisition recorded, read with ``frame_of``. An image that does not say
        where it was taken is placed by the caller's reading instead. ``stage_xy_m[k]`` is the stage position image k
        was taken at and ``fov_m`` is the width of every image, with rows along +y and columns along +x. Without them
        such an image is refused by name before anything is written.

        ``spectrum_keys[k]`` is the key of the EDS point spectrum taken on image k, "" for none. ``spectra_at`` is
        where on every image they were taken, [x, y] fractions from the left and from the top; it is needed with them.
        The run keeps both (``spectrum_keys`` beside ``tiles``, ``spectra_at`` in its metadata), so its map gets an
        EDS point of interest for each.

        With ``local_runs`` set the run is an HDF5 file in that folder with the same layout, and the images are only
        read from Tiled.
        """
        keys = [str(k) for k in image_keys or []]
        if not keys:
            raise ValueError("image_keys is empty: give the data keys of the tiles to gather")
        if len(set(keys)) != len(keys):
            raise ValueError("image_keys names an image twice")
        if stage_xy_m is not None:
            if fov_m is None or not (isinstance(fov_m, (int, float)) and math.isfinite(fov_m) and fov_m > 0):
                raise ValueError("stage_xy_m needs fov_m, the width of every image in metres")
            if len(stage_xy_m) != len(keys):
                raise ValueError(f"stage_xy_m has {len(stage_xy_m)} positions for {len(keys)} images; give one [x, y] per image")
            for k, xy in enumerate(stage_xy_m):
                if not (isinstance(xy, (list, tuple)) and len(xy) >= 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy[:2])):
                    raise ValueError(f"stage_xy_m[{k}] is not [x, y] in metres: {str(xy)[:80]}")
        spectra = self._spectra(keys, spectrum_keys, spectra_at)
        client = getattr(self.store, "client", None)
        if client is None:
            raise ValueError("gathering a run needs the Tiled server; this analysis server was started without one")
        frames, given = [], []
        for k, key in enumerate(keys):
            try:
                frame = frame_of(self.store, key)
            except KeyError as exc:
                raise ValueError(f"{key} cannot be a tile: {exc}") from exc
            except FrameError as exc:
                if stage_xy_m is None:
                    raise ValueError(f"{key} cannot be a tile: {exc}") from exc
                try:
                    shape = list(image_node(self.store, key).structure().shape)[-2:]
                except FrameError as unreadable:
                    raise ValueError(f"{key} cannot be a tile: {unreadable}") from unreadable
                frame = {"kind": "image", "centre_xy_m": [float(v) for v in stage_xy_m[k][:2]], "pixel_size_m": float(fov_m) / shape[1],
                         "shape": shape, "sample": self.store.sample_of(key)}
                given.append(str(exc))
            if frame["kind"] != "image":
                raise ValueError(f"{key} is a stitched map, not a tile image")
            if stage_xy_m is not None and not frame.get("shift_recorded", True):
                # The image records the stage but not the image shift, so use the position the caller read.
                xy = [float(v) for v in stage_xy_m[k][:2]]
                if math.dist(xy, frame["centre_xy_m"]) > float(frame["pixel_size_m"]):
                    frame = {**frame, "centre_xy_m": xy}
                    given.append(f"{key} records no image shift")
            frames.append(frame)
        name = f"map_{uuid.uuid4().hex[:12]}"
        metadata = {
            "kind": "map_run", "run_layout": 1, "id": uuid.uuid4().hex, "plan": json.dumps(plan or {}), "source": "workflow",
            "tiles": len(keys), "created": time.time(), **(frames[0].get("sample") or {}),
        }
        if spectra is not None:
            metadata["spectra_at"] = spectra["spectra_at"]
        if self.local_runs is not None:
            name = self._write_local_run(name, metadata, keys, frames, spectra)
        else:
            self._write_tiled_run(client, name, metadata, keys, frames, spectra)
        out = {"run": name, "tiles": len(keys), "pixel_size_m": float(frames[0]["pixel_size_m"])}
        if spectra is not None:
            out["spectra"] = sum(1 for key in spectra["spectrum_keys"] if key)
        if given:
            out["placed_by_caller"] = len(given)
            out["note"] = (f"{len(given)} of {len(keys)} images do not record where they were taken or the image shift they were taken with "
                           f"({given[0][:160]}); they are placed at the "
                           "stage positions given, as full fields with rows along +y and columns along +x")
        return out

    @staticmethod
    def _spectra(keys: list[str], spectrum_keys: list[str] | None, spectra_at: list[float] | None) -> dict | None:
        """The checked spectrum keys and their place on the tile, or None when the run has no spectra."""
        if spectrum_keys is None:
            if spectra_at is not None:
                raise ValueError("spectra_at is where the spectra were taken; give spectrum_keys with it")
            return None
        if len(spectrum_keys) != len(keys):
            raise ValueError(f"spectrum_keys has {len(spectrum_keys)} entries for {len(keys)} images; give one per image, \"\" where none was taken")
        if not all(isinstance(key, str) for key in spectrum_keys):
            raise ValueError("spectrum_keys must be strings: a spectrum's data key, or \"\" where none was taken")
        if spectra_at is None:
            raise ValueError("spectrum_keys needs spectra_at, where on each image the spectra were taken: [x, y] fractions from the left and the top")
        if not (isinstance(spectra_at, (list, tuple)) and len(spectra_at) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in spectra_at)):
            raise ValueError(f"spectra_at must be [x, y], fractions of the image from 0 to 1, not {str(spectra_at)[:80]}")
        return {"spectrum_keys": list(spectrum_keys), "spectra_at": [float(v) for v in spectra_at]}

    def _write_tiled_run(self, client, name: str, metadata: dict, keys: list[str], frames: list[dict], spectra: dict | None) -> None:
        """Write the run as a Tiled container. A partly written run is deleted."""
        run = client.create_container(name, metadata=metadata)
        try:
            if spectra is not None:
                run.write_array(np.array(spectra["spectrum_keys"], dtype=str), key="spectrum_keys")
            tiles = run.create_container("tiles")
            for k, (key, frame) in enumerate(zip(keys, frames)):
                tiles.write_array(self._tile_array(key), key=f"tile_{k:05d}", metadata={
                    "index": k, "stage_reported_m": [float(v) for v in frame["centre_xy_m"]], "pixel_size_m": float(frame["pixel_size_m"]),
                    "source_key": key,
                })
        except BaseException:
            try:  # a partial run would stitch into a map with holes
                client.delete_contents(name, recursive=True, external_only=False)
            except Exception:
                pass
            raise

    def _tile_array(self, key: str) -> np.ndarray:
        array = np.asarray(image_node(self.store, key).read())
        while array.ndim > 2:  # drop a leading frame or channel axis
            array = array[0]
        return array

    def _write_local_run(self, name: str, metadata: dict, keys: list[str], frames: list[dict], spectra: dict | None) -> str:
        """Write the run as ``<local_runs>/<name>.h5`` and return its path. A partly written file is removed."""
        self.local_runs.mkdir(parents=True, exist_ok=True)
        path = (self.local_runs / f"{name}.h5").resolve()
        if not any(path.parent == root or root in path.parents for root in self.store.local_roots):
            raise ValueError(f"the local runs folder {self.local_runs} is outside the directories this server may write "
                             f"({', '.join(map(str, self.store.local_roots)) or 'none'}); start it with --local-root for that folder")
        partial = path.with_suffix(".h5.tmp")
        try:
            with h5py.File(partial, "w") as h5:
                for field, value in metadata.items():
                    h5.attrs[field] = value
                if spectra is not None:
                    h5.create_dataset("spectrum_keys", data=spectra["spectrum_keys"], dtype=h5py.string_dtype())
                tiles = h5.create_group("tiles")
                for k, (key, frame) in enumerate(zip(keys, frames)):
                    tile = tiles.create_dataset(f"tile_{k:05d}", data=self._tile_array(key))
                    tile.attrs["index"] = k
                    tile.attrs["stage_reported_m"] = [float(v) for v in frame["centre_xy_m"]]
                    tile.attrs["pixel_size_m"] = float(frame["pixel_size_m"])
                    tile.attrs["source_key"] = key
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return str(path)

    def _check_score(self, plan: dict) -> dict:
        scorer = plan.get("scorer", "baseline")
        SCORERS.get(scorer)  # raises a ValueError that names the registered scorers
        self.store.map_metadata(plan["map"])
        return {"steps": 1}

    def _run_score(self, plan: dict, progress) -> dict:
        meta = self.store.map_metadata(plan["map"])
        scorer = SCORERS.get(plan.get("scorer", "baseline"))()
        # Use the finest level that fits SCORE_MAX_SIDE, so scoring a map of any size reads a bounded image.
        fits = next((k for k in range(meta["levels"]) if max(level_shape_of(meta, k)) <= SCORE_MAX_SIDE), meta["levels"] - 1)
        level = max(int(plan.get("level", min(2, meta["levels"] - 1))), fits)
        image = self.store.read_level(plan["map"], level)
        pixel = meta["pixel_size_m"] * 2**level
        if "window_m" in plan:
            window = int(round(float(plan["window_m"]) / pixel))
        else:  # aim for about a 6 x 6 grid of candidate regions
            window = min(image.shape[:2]) // 6
        window = int(np.clip(window, 16, max(16, min(image.shape[:2]))))
        scores = score_windows(image_dataset(image, pixel, title=plan["map"]), scorer, window, stride=max(1, window // 2))
        top = select_regions(scores, int(plan.get("top_k", 5)), min_separation=window)  # chosen regions at least one window apart
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
            # Earlier regions with the same tag are replaced, not repeated.
            # Only the highlights are rewritten, never the run's EDS points.
            self.store.update_highlights(plan["map"], lambda current: [p for p in current if p.get("tag") != tag] + added)
        else:
            self.store.add_pois(plan["map"], added)
        progress(1)
        return {"result_keys": {"map": plan["map"]}, "summary": {"scored_windows": len(scores), "added_pois": added}}

    def map_info(self, map_key: str) -> dict[str, Any]:
        meta = self.store.map_metadata(map_key)
        view = self.store.map_view(map_key, meta)
        info = {
            "map": map_key, "run": meta.get("run"), "tiles": meta.get("tiles"), "levels": meta["levels"], **view,
            "points_of_interest": self.store.poi_count(map_key), "poi_kinds": poi_kind_labels(self.store.poi_kinds(map_key)),
            "stitching": {k: meta[k] for k in ("matches_used", "median_residual_px", "p95_residual_px", "fallback_tiles", "unplaced_tiles", "scan_linear", "tile_gain", "drift", "drift_unmeasured") if k in meta},
            "id": meta.get("id"), "stitched_at": meta.get("stitched_at"), "inputs_sha256": meta.get("inputs_sha256"),
        }
        if meta.get("run"):
            try:
                versions = self.store.list_maps(meta["run"])
            except Exception:  # the run is gone or unreadable, but the map itself is still described
                versions = []
            if versions:
                info["versions"] = {"count": len(versions), "newest": versions[0]["map"], "this_is_newest": versions[0]["map"] == map_key}
        return info

    def list_pois(self, map_key: str, kind: str | None = None, box_m: list[float] | None = None, limit: int = 200,
                  counts_above: int | None = None) -> dict[str, Any]:
        """{"total": how many match, "pois": at most ``limit`` of them}; the shape is the same for every filter,
        except that a filter matching nothing also gives how many points the map has (``total_on_map``) and, for a
        kind, the kinds it has (``kinds_on_map``). A box with no area ([0, 0, 0, 0]) is no box; one shrunk to a point
        ([x, x, y, y]) lists the points nearest it, closest first, each with ``distance_m`` (of any kind when none of
        the asked kinds is on the map). ``counts_above``: see ``_crowded``."""
        limit = max(0, int(limit))
        # Models ask for kind="all" or list several kinds in one string. A filter that matches nothing would read as no points.
        kinds = {k for k in re.split(r"[\s,;|]+", kind or "") if k} - {"all", "any", "*"}
        # Models fill an optional box with zeros, so a box with no area means no box.
        # A box shrunk to one point means the model wants the point there.
        point = None
        if box_m is not None and not (len(box_m) == 4 and box_m[1] > box_m[0] and box_m[3] > box_m[2]):
            if len(box_m) == 4 and box_m[0] == box_m[1] and box_m[2] == box_m[3] and any(box_m):
                point = (float(box_m[0]), float(box_m[2]))
            box_m = None
        if point is not None:
            return self._nearest(map_key, point, kinds, limit)
        if counts_above is not None and (not kinds or kinds & set(CELL_KINDS)):
            crowded = self._crowded(map_key, kinds, box_m, limit, max(0, int(counts_above)))
            if crowded is not None:
                return crowded
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

    def _crowded(self, map_key: str, kinds: set[str], box_m: list[float] | None, limit: int, counts_above: int) -> dict[str, Any] | None:
        """Per-cell counts in place of the points kept in cells, when more than ``counts_above`` of them are in the cells
        the box (or the whole map) touches; None when there are not that many. Counts come from the stored cell counts,
        so no cell is read: each is of the whole cell, every kind kept in cells. The highlights that match are still
        listed as points."""
        counts = self.store.poi_counts(map_key)
        grid = counts["grid"]
        touched = counts["cells"] if box_m is None else {f"{i}_{j}": counts["cells"].get(f"{i}_{j}", 0) for i, j in self.store.cells_in(grid, box_m)}
        cells = [{"cell": name, "count": n, "box_m": self.store.cell_box(grid, *(int(v) for v in name.split("_")))} for name, n in sorted(touched.items()) if n > 0]
        counted = sum(c["count"] for c in cells)
        if counted <= counts_above:
            return None
        highlight_kinds = sorted(kinds - set(CELL_KINDS))
        if kinds and not highlight_kinds:
            pois = []
        elif highlight_kinds:
            pois = [p for k in highlight_kinds for p in self.store.list_pois(map_key, kind=k, box_m=box_m)]
        else:
            pois = self.store.list_pois(map_key, box_m=box_m, cells=False)
        return {"total": len(pois) + counted, "pois": pois[:limit], "cells": cells, "cell_kinds": sorted(CELL_KINDS)}

    def _nearest(self, map_key: str, point: tuple[float, float], kinds: set[str], limit: int) -> dict[str, Any]:
        """The points nearest ``point``, closest first, of the asked kinds (of any kind when the map has none of them)."""
        x, y = point
        pois = [p for k in sorted(kinds) for p in self.store.list_pois(map_key, kind=k)] if kinds else []
        out: dict[str, Any] = {}
        if not pois:
            if kinds:
                out["kinds_on_map"] = self.store.poi_kinds(map_key)
            pois = self.store.list_pois(map_key)
        near = sorted(({**p, "distance_m": math.hypot(p["x_m"] - x, p["y_m"] - y)} for p in pois), key=lambda p: p["distance_m"])
        out = {"total": len(near), "pois": near[:limit], **out}
        out["hint"] = (f"points nearest ({x:.4g}, {y:.4g}) m first: the first is the point there; use its x_m and y_m as they are"
                       if near else "this map has no points of interest")
        return out

    def register(self, mcp: FastMCP) -> None:
        """Add the map tools to ``mcp``. Registering the same MapTools again is a no-op; a second MapTools is refused."""
        owner = _REGISTERED.get(mcp)
        if owner is self:
            return
        if owner is not None:
            raise RuntimeError("this FastMCP app already has map tools from another MapTools; build a new app with create_server()")
        _REGISTERED[mcp] = self
        self._register_stitch_tools(mcp)
        self._register_match_and_score_tools(mcp)
        self._register_job_tools(mcp)
        self._register_view_tools(mcp)
        self._register_poi_tools(mcp)
        self._register_listing_tools(mcp)
        self._register_envelope_tool(mcp)

    def _register_stitch_tools(self, mcp: FastMCP) -> None:
        """Add the tools that gather a run and stitch it."""
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

            run: the run's Tiled key or file path (Mapper results -> result_keys.run or run_file). Images taken one
            by one are not a run yet: call collect_map_run with their keys first, and pass the run it returns.
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
        def collect_map_run(image_keys: list[str], plan: dict | None = None, stage_xy_m: list[list[float]] | None = None,
                            fov_m: float | None = None, spectrum_keys: list[str] | None = None,
                            spectra_at: list[float] | None = None) -> dict[str, Any]:
            """Gather single scanned images into one map run, so tiles taken one by one (a workflow's tile loop) can
            be stitched with assemble_map. Not needed for a Mapper job: its results already name a run.

            image_keys: the tiles' data keys, in the order they were taken.
            plan: facts to keep with the run (size, field of view, the workflow's name).
            stage_xy_m, fov_m: optional; the stage [x, y] each image was taken at and the width of every image
            (metres). They place an image that does not record where it was taken; without them it is refused.
            spectrum_keys: optional; one per image, in the same order, the key of the EDS point spectrum taken on it,
            "" for none. assemble_map then marks each as a point of interest where it was taken.
            spectra_at: with spectrum_keys, where on every image the spectra were taken: [x, y] fractions of the image,
            x from the left, y from the top.
            Returns {"run": the run's key for assemble_map, "tiles": how many, "spectra": how many, with spectrum_keys}.
            """
            try:
                return tools.collect_run(image_keys, plan, stage_xy_m, fov_m, spectrum_keys, spectra_at)
            except ValueError as exc:
                return {"error": {"code": "bad_request", "message": str(exc)}}

    def _register_match_and_score_tools(self, mcp: FastMCP) -> None:
        """Add the tools that match tiles and score a map."""
        tools = self

        @mcp.tool()
        def match_tiles(image: str, neighbour: str, offset_xy_m: list[float], fov_m: float, search_m: float | None = None,
                        min_score: float = MATCH_MIN_SCORE) -> dict[str, Any]:
            """Check that a tile overlaps the tile before it where it should, as soon as it is taken.

            image: the new tile's data key. neighbour: the key of an overlapping tile already taken.
            offset_xy_m: where image's centre should be from neighbour's, [x, y] metres. fov_m: the width of both images, metres.
            search_m: how far from there to look (default a tenth of fov_m). min_score: the correlation of the two overlaps to accept.
            Returns {"matched": true or false, "score", "offset_xy_m": where it really is, "error_xy_m"}, with "reason"
            when it did not match (no overlap, a blank image, no clear match). Reads only; nothing is written.
            """
            try:
                return tools.match_tiles(image, neighbour, offset_xy_m, fov_m, search_m, min_score)
            except ValueError as exc:
                return {"error": {"code": "bad_request", "message": str(exc)}}

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
                # Models often write a size as width and height. Regions are square, so use the larger side.
                window_m = max((float(v) for v in window_m), default=None)
            if window_m:
                plan["window_m"] = window_m
            if level is not None:
                plan["level"] = level
            reserved = tools.jobs.reserve(plan)
            if "error" in reserved:
                return reserved
            return tools.jobs.start(reserved["reservation_id"])

    def _register_job_tools(self, mcp: FastMCP) -> None:
        """Add the tools that follow a stitching or scoring job."""
        tools = self

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

    def _register_view_tools(self, mcp: FastMCP) -> None:
        """Add the tools that describe a map and show part of it."""
        tools = self

        @mcp.tool()
        def map_info(map: str) -> dict[str, Any]:
            """Tiles, size, levels, stage extent and number of points of interest of a stitched map, the kinds of point on it
            with their labels (poi_kinds), and how to draw it: title, origin_xy_m, pixel_size_m, axes and pyramid (one entry
            per level: level, key, shape, dtype, chunks, downsample, pixel_size_m, origin_xy_m).

            Given a Mapper run instead, the refusal names the run's stitched map."""
            return tools.map_info(map)

        @mcp.tool()
        def list_maps(run: str | None = None, limit: int = 20) -> dict[str, Any]:
            """Every stitched version of a run's map, newest first (each assemble_map makes a new one; none is deleted).
            Leave run out to list every stitched map.

            Returns {"total", "maps": [at most ``limit``]}; each map has map, id, stitched_at, inputs_sha256, tiles, run,
            title, canvas_shape, pixel_size_m, origin_xy_m, axes, stage_box_m, local and pyramid (one entry per level:
            level, key, shape, dtype, chunks, downsample, pixel_size_m, origin_xy_m)."""
            maps = tools.store.list_maps(run)
            return {"total": len(maps), "maps": [{**entry, **tools.store.map_view(entry["map"])} for entry in maps[: max(0, int(limit))]]}

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
            return ToolResult(content=[json.dumps(info), MCPImage(data=to_png(region, PREVIEW_MAX_SIDE), format="png")], structured_content=info)

    def _register_poi_tools(self, mcp: FastMCP) -> None:
        """Add the tools that list and add points of interest."""
        tools = self

        @mcp.tool()
        def list_pois(map: str, kind: str | None = None, box_m: list[float] | None = None, limit: int = 200,
                      counts_above: int | None = None) -> dict[str, Any]:
            """Points of interest on a map (stage x_m, y_m in metres, kind, source, score, reason, links).

            kind: leave it out to list every point (the usual call). Give it only when the user asks for one kind:
            "candidate" (regions a scorer ranked), "eds_point", "revisit", "note", or several joined by commas.
            box_m: only those inside [x_min, x_max, y_min, y_max] (only that part of the map is read); leave it out for all.
            Returns {"total": how many match, "pois": [at most ``limit`` of them]}, always this shape; when a filter
            matches nothing, "total_on_map" says how many points the map has.
            counts_above: for drawing a crowded map. When more than this many points kept in cells (poi_kinds with
            in_cells) are in the box, they are not listed: "cells" gives [{"cell", "count", "box_m"}] for each cell the
            box touches (a count is of the whole cell), and "pois" lists only the other points.
            """
            listed = tools.list_pois(map, kind=kind, box_m=box_m, limit=limit, counts_above=counts_above)
            # A small model has little context, so empty fields are dropped and numbers are rounded.
            # Positions are given to 0.01 nm and scores to 0.01. The store keeps them exact.
            # A scored point keeps its scorer version, so a reader can tell which scorer found it.
            listed["pois"] = [{**{k: v for k, v in poi.items() if not (k in ("links", "scorer_version") and not v)},
                               **{k: round(poi[k], 11) for k in ("x_m", "y_m", "distance_m") if isinstance(poi.get(k), float)},
                               **({"score": round(poi["score"], 2)} if isinstance(poi.get("score"), float) else {})}
                              for poi in listed["pois"]]
            return listed

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

    def _register_listing_tools(self, mcp: FastMCP) -> None:
        """Add the tools that list scorers and tiles."""
        tools = self

        @mcp.tool()
        def list_scorers() -> list[dict[str, str]]:
            """Scorers available to score_map: name, version and what they look for."""
            return [{"name": s.name, "version": s.version, "description": s.description} for s in SCORERS.all().values()]

        @mcp.tool()
        def list_tiles(run: str, box_m: list[float] | None = None, limit: int = 200) -> dict[str, Any]:
            """Tiles of a map run (index, stage position, shape), optionally only those in a stage box; at most ``limit``, with the total."""
            info = tools.store.scan_run(run, box_m)
            shown = range(min(len(info), max(0, int(limit))))
            return {
                "total": len(info),
                "tiles": [{"index": int(info.index[k]), "stage_xy_m": [float(v) for v in info.stage_xy_m[k]], "shape": [int(v) for v in info.shapes[k]]} for k in shown],
            }

    def _register_envelope_tool(self, mcp: FastMCP) -> None:
        """Add the generic job envelope tool."""
        tools = self

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
