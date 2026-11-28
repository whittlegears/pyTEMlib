"""Run pyTEMlib's learned-model tools on a Tiled dataset and keep what they found in Tiled.

pyTEMlib's ``segment_sam`` and ``find_atoms_blobnet`` only compute: they reply with numbers and a marked-up PNG, and
nothing outlives the chat. The two tools here run the same pyTEMlib functions and also write the result to the Tiled
server, as a new container beside the source:

* ``segment_sam_and_save``: ``<source>_sam_<version>`` with ``labels`` (the label image, 0 for no region, shaped like
  the source), ``areas`` (a table of every region, not only those the reply lists) and ``marked`` (the PNG the
  reply shows, as an RGB array);
* ``find_atoms_blobnet_and_save``: ``<source>_blobnet_<version>`` with ``atoms`` (a table of every atom) and
  ``marked``.

The container's metadata says what made it (``kind``, ``tool``, the model and every parameter), names the source
(``source``, and its key in ``parents``) and copies the source's sample, so a later reader can find, trust and
redo it. The reply is pyTEMlib's, plus ``saved``: the new key. The source must be a Tiled key: a local file's results
are not written anywhere. pyTEMlib stays free of asyncroscopy; it is only called.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image as MCPImage
from PIL import Image

from pytemlib_mcp import analysis as mcp_analysis

KINDS = {"segment_sam": ("segmentation", "sam"), "find_atoms_blobnet": ("atom_positions", "blobnet")}
SAVING_TOOLS = ("segment_sam_and_save", "find_atoms_blobnet_and_save")
EVERY = 10**9  # ask pyTEMlib for every area or atom; the reply is cut to the caller's limit afterwards


class NotSaved(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _refused(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _rgb(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def _labels(result: dict) -> np.ndarray:
    dtype = "<u2" if result.get("labels_dtype") == "uint16" else "<u4"
    flat = np.frombuffer(base64.b64decode(result["labels_base64"]), dtype=dtype)
    return flat.reshape(int(result["height"]), int(result["width"]))


def _table(rows: list[dict], columns: tuple[str, ...]):
    return pd.DataFrame(rows, columns=list(dict.fromkeys([*columns, *(k for row in rows for k in row)])))


def result_name(source: str, suffix: str, version: str) -> str:
    """The new key: the source's key (its ``#path`` too, "/" and "#" made "_") then the model and a version."""
    stem = source.replace("#", "_").replace("/", "_").strip("_") or "dataset"
    return f"{stem}_{suffix}_{version}"


def save(client, source: str, tool: str, parameters: dict, result: dict, sample: dict, version: str) -> str:
    """Write one pyTEMlib result (with ``_png`` and, for SAM, the label image) as a new container; returns its key.
    A write that fails part way removes what it wrote."""
    kind, suffix = KINDS[tool]
    name = result_name(source, suffix, version)
    listed = {k: result[k] for k in ("total", "coverage", "pixel_size", "units", "coordinates") if k in result}
    metadata = {"kind": kind, "tool": tool, "source": source, "parents": [source.partition("#")[0]], "id": uuid.uuid4().hex,
                "version": version, "parameters": parameters, "model": parameters.get("model", "blobnet"), **listed, **sample}
    container = client.create_container(name, metadata=metadata)
    try:
        if tool == "segment_sam":
            container.write_array(np.ascontiguousarray(_labels(result)), key="labels",
                                  metadata={"description": "region id of each pixel of the source; 0 for no region"})
            container.write_table(_table(result.get("areas") or [], ("id", "area_px", "x", "y", "equiv_diameter_px",
                                                                        "circularity", "confidence")), key="areas")
        else:
            container.write_table(_table(result.get("atoms") or [], ("x", "y", "x_scaled", "y_scaled", "confidence")),
                                      key="atoms")
        if result.get("_png"):
            container.write_array(_rgb(result["_png"]), key="marked",
                                  metadata={"description": "the source with what was found drawn on it, as the reply showed it"})
    except Exception:
        try:
            client.delete_contents([name], recursive=True)
        except Exception:
            pass
        raise
    return name


def _reply(result: dict, saved: str, list_key: str, limit: int, include_image: bool):
    """pyTEMlib's reply, cut to ``limit`` entries, with the saved key; a ToolResult with the PNG when there is one."""
    reply = {k: v for k, v in result.items() if k not in ("_png", "labels_base64", "labels_dtype", "width", "height")}
    listed = reply.get(list_key) or []
    reply[list_key] = listed[: max(0, int(limit))]
    reply["returned"] = len(reply[list_key])
    reply["saved"] = saved
    png = result.get("_png") if include_image else None
    if not png:
        return reply
    return ToolResult(content=[json.dumps(reply), MCPImage(data=png, format="png")], structured_content=reply)


def register(mcp, client: Callable[[], Any], sample_of: Callable[[str], dict], is_local: Callable[[str], bool],
             version: Callable[[], str]) -> list[str]:
    """Add the saving tools to ``mcp`` when the installed pyTEMlib has the learned tools; returns the names added.

    ``client()`` is the Tiled client (None: no Tiled server, and the tools refuse); ``sample_of(key)`` the sample a
    dataset recorded; ``is_local(source)`` whether a reference is a local file; ``version()`` a new version suffix.
    """
    if not all(hasattr(mcp_analysis, tool) for tool in KINDS):
        return []

    async def run(tool: str, dataset: str, parameters: dict, list_key: str, limit: int, include_image: bool):
        try:
            if not isinstance(dataset, str) or not dataset.strip():
                raise NotSaved("bad_argument", "dataset must be a Tiled key")
            if is_local(dataset.partition("#")[0]):
                raise NotSaved("not_in_tiled", f"{dataset} is a local file; only results of a Tiled dataset are saved "
                               f"(use {tool} to see them without saving)")
            catalogue = client()
            if catalogue is None:
                raise NotSaved("no_tiled", f"this server has no Tiled server to save to (use {tool} to see the result without saving)")
        except NotSaved as exc:
            return _refused(exc.code, exc.message)
        except Exception as exc:  # a refused local path, a server that is down
            return _refused("no_tiled", f"{type(exc).__name__}: {exc}")
        function = getattr(mcp_analysis, tool)
        extra = {"include_labels": True, "max_areas": EVERY} if tool == "segment_sam" else {"max_atoms": EVERY}
        result = await asyncio.to_thread(function, dataset, **parameters, **extra, include_image=True)
        if "error" in result:
            return result
        try:
            saved = await asyncio.to_thread(save, catalogue, dataset.strip(), tool, parameters, result,
                                            sample_of(dataset.partition("#")[0]), version())
        except Exception as exc:
            return _refused("not_saved", f"the {KINDS[tool][0]} was found but could not be written to Tiled: {type(exc).__name__}: {exc}")
        return _reply(result, saved, list_key, limit, include_image)

    @mcp.tool()
    async def segment_sam_and_save(dataset: str, model: str = "large", points_per_side: int = 48, iou_threshold: float = 0.57,
                                   stability_threshold: float = 0.75, min_area_px: int = 200, max_areas: int = 500,
                                   include_image: bool = True):
        """segment_sam on a Tiled dataset, and keep the result in Tiled under a new key (the reply's 'saved'): the label
        image ('labels'), every region ('areas') and the marked-up picture ('marked'), with the model, its parameters
        and the source in the metadata. Use it instead of segment_sam when the segmentation should outlive the chat
        (to measure, compare or report it later); the reply is segment_sam's, with up to max_areas regions listed."""
        parameters = {"model": model, "points_per_side": points_per_side, "iou_threshold": iou_threshold,
                      "stability_threshold": stability_threshold, "min_area_px": min_area_px}
        return await run("segment_sam", dataset, parameters, "areas", max_areas, include_image)

    @mcp.tool()
    async def find_atoms_blobnet_and_save(dataset: str, threshold_rel: float = 0.35, min_distance_px: int = 3,
                                          max_atoms: int = 5000, include_image: bool = True):
        """find_atoms_blobnet on a Tiled dataset, and keep the result in Tiled under a new key (the reply's 'saved'):
        every atom ('atoms') and the marked-up picture ('marked'), with the parameters and the source in the metadata.
        Use it instead of find_atoms_blobnet when the positions should outlive the chat; the reply is
        find_atoms_blobnet's, with up to max_atoms atoms listed."""
        parameters = {"threshold_rel": threshold_rel, "min_distance_px": min_distance_px}
        return await run("find_atoms_blobnet", dataset, parameters, "atoms", max_atoms, include_image)

    return list(SAVING_TOOLS)
