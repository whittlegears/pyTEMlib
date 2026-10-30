"""Notes that people and agents put on datasets: a pin, a circle, a box, or an energy range.

Every change is an event appended to one JSON-lines file (create, edit, delete, restore), so a
note's history is never lost and a deleted note can be brought back. A note's current state is
its events folded in order. Each edit, delete or restore may name the version it was made
against (``expected_version``); if someone changed the note since, the change is refused rather
than silently overwriting theirs. Several server processes may share the file: each operation
takes an OS lock on ``<file>.lock`` and first reads what the others appended.

Coordinates are in the dataset's own frame: ``stage_m`` (metres, stage x/y, for maps), ``pixel``
(x = column, y = row, in pixels, for images) or ``energy_ev`` (for an energy range on a spectrum).
The author is the calling agent's ``X-Asyncroscopy-Caller`` header when there is one, otherwise the
name the caller reports; neither is verified, so readers should treat it as "reported by".
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

if os.name == "nt":
    import msvcrt
else:
    import fcntl

SHAPES = {
    "pin": ("x", "y"),
    "circle": ("x", "y", "r"),
    "box": ("x0", "y0", "x1", "y1"),
    "range": ("start", "end"),
}
FRAMES = {"stage_m", "pixel", "energy_ev"}
MAX_NOTE_CHARS = 2000
MAX_KIND_CHARS = 40
MAX_LINKS = 20
MAX_NOTES_PER_DATASET = 5000
MAX_NAME_CHARS = 80


class AnnotationError(ValueError):
    """A note that cannot be written: bad shape or coordinates, an unknown note, or a stale version."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _check_geometry(shape: str, frame: str, coords: dict) -> dict:
    if shape not in SHAPES:
        raise AnnotationError("bad_shape", f"shape must be one of {sorted(SHAPES)}, got {shape!r}")
    if frame not in FRAMES:
        raise AnnotationError("bad_frame", f"frame must be one of {sorted(FRAMES)}, got {frame!r}")
    if (shape == "range") != (frame == "energy_ev"):
        raise AnnotationError("bad_frame", "an energy range uses frame energy_ev, and energy_ev is only for a range")
    if not isinstance(coords, dict):
        raise AnnotationError("bad_coords", f"coords must be an object with {list(SHAPES[shape])}")
    missing = [k for k in SHAPES[shape] if k not in coords]
    extra = sorted(set(coords) - set(SHAPES[shape]))
    if missing or extra:
        raise AnnotationError("bad_coords", f"a {shape} needs exactly {list(SHAPES[shape])}" + (f"; missing {missing}" if missing else "") + (f"; unexpected {extra}" if extra else ""))
    out = {}
    for key in SHAPES[shape]:
        value = coords[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AnnotationError("bad_coords", f"{key} must be a finite number, got {value!r}")
        out[key] = float(value)
    if shape == "circle" and not out["r"] > 0:
        raise AnnotationError("bad_coords", "a circle's r must be > 0")
    if shape == "box" and not (out["x0"] < out["x1"] and out["y0"] < out["y1"]):
        raise AnnotationError("bad_coords", "a box needs x0 < x1 and y0 < y1")
    if shape == "range" and not out["start"] < out["end"]:
        raise AnnotationError("bad_coords", "a range needs start < end")
    return out


def _check_text(note: Any, kind: Any, links: Any) -> tuple[str, str, list[str]]:
    if not isinstance(note, str) or len(note) > MAX_NOTE_CHARS:
        raise AnnotationError("bad_note", f"note must be text of at most {MAX_NOTE_CHARS} characters")
    if not isinstance(kind, str) or not kind.strip() or len(kind) > MAX_KIND_CHARS:
        raise AnnotationError("bad_kind", f"kind must be a short word (at most {MAX_KIND_CHARS} characters), e.g. note, particle, defect")
    if links is None:
        links = []
    if not isinstance(links, list) or len(links) > MAX_LINKS or not all(isinstance(k, str) and k for k in links):
        raise AnnotationError("bad_links", f"links must be a list of at most {MAX_LINKS} data keys")
    return note, kind.strip(), list(links)


def _short(name: str | None) -> str | None:
    return name[:MAX_NAME_CHARS] if name else None


class AnnotationStore:
    """The event file and the notes folded from it, shared by threads and by processes using the same file."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._events: dict[str, list[dict]] = {}
        self._offset = 0
        self.unreadable_lines = 0
        with self._locked():
            pass

    @contextlib.contextmanager
    def _locked(self):
        """This process's lock, then the file lock, then the events other processes appended."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path.with_name(self.path.name + ".lock"), "a+b") as lock:
                lock.seek(0)
                if os.name == "nt":
                    while True:
                        try:
                            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                            break
                        except OSError:
                            time.sleep(0.05)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    self._catch_up()
                    yield
                finally:
                    lock.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _catch_up(self) -> None:
        """Read the complete lines appended since the last read. A last line without its newline (a write cut
        short by a crash) is left alone; the next append ends it first, so it then reads as one unreadable line."""
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        complete = data[: data.rfind(b"\n") + 1]
        self._offset += len(complete)
        for raw in complete.split(b"\n"):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
                self._events.setdefault(str(event["note_id"]), []).append(event)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                self.unreadable_lines += 1

    def _append(self, event: dict) -> None:
        line = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        with open(self.path, "ab") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > self._offset:
                f.write(b"\n")
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self._catch_up()

    @staticmethod
    def _fold(events: list[dict]) -> dict | None:
        """The note its events describe, or None when its create event is missing (a line lost to a crash)."""
        if not events or events[0].get("event") != "create":
            return None
        note: dict[str, Any] = {}
        for event in events:
            if event["event"] == "create":
                note = {"id": event["note_id"], "dataset": event["dataset"], **event["fields"],
                        "author": event.get("author"), "created_at": event["at"], "deleted": False}
            elif event["event"] == "edit":
                note.update(event["fields"])
                note["edited_at"], note["edited_by"] = event["at"], event.get("author")
            elif event["event"] in ("delete", "restore"):
                note["deleted"] = event["event"] == "delete"
                note[f"{event['event']}d_at"], note[f"{event['event']}d_by"] = event["at"], event.get("author")
        note["version"] = len(events)
        return note

    def _current(self, note_id: str) -> dict:
        note = self._fold(self._events.get(note_id, []))
        if note is None:
            raise AnnotationError("unknown_note", f"no note {note_id!r}")
        return note

    def _notes_of(self, dataset: str) -> list[dict]:
        folded = (self._fold(events) for events in self._events.values() if events and events[0].get("dataset") == dataset)
        return [n for n in folded if n is not None]

    def _event(self, kind: str, note_id: str, dataset: str, author: str | None, run: str | None, fields: dict | None = None) -> dict:
        return {"event": kind, "event_id": uuid.uuid4().hex, "note_id": note_id, "dataset": dataset,
                "at": time.time(), "author": _short(author), "run": _short(run), **({"fields": fields} if fields is not None else {})}

    def create(self, dataset: str, shape: str, coords: dict, frame: str, note: str = "", kind: str = "note",
               links: list[str] | None = None, author: str | None = None, run: str | None = None) -> dict:
        geometry = _check_geometry(shape, frame, coords)
        text, kind, links = _check_text(note, kind, links)
        note_id = uuid.uuid4().hex[:12]
        fields = {"shape": shape, "frame": frame, "coords": geometry, "note": text, "kind": kind, "links": links}
        with self._locked():
            if len(self._notes_of(dataset)) >= MAX_NOTES_PER_DATASET:
                raise AnnotationError("too_many", f"{dataset} already has {MAX_NOTES_PER_DATASET} notes")
            self._append(self._event("create", note_id, dataset, author, run, fields))
            return self._current(note_id)

    def _check_version(self, current: dict, expected_version: int | None) -> None:
        if expected_version is not None and int(expected_version) != current["version"]:
            raise AnnotationError(
                "conflict", f"note {current['id']} is at version {current['version']}, not {expected_version}: "
                f"{current.get('edited_by') or current.get('author') or 'someone'} changed it since you read it; "
                "read it again and repeat the change if it still applies",
            )

    def edit(self, note_id: str, changes: dict, author: str | None = None, run: str | None = None, expected_version: int | None = None) -> dict:
        allowed = {"shape", "frame", "coords", "note", "kind", "links"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise AnnotationError("bad_edit", f"only {sorted(allowed)} can be changed, not {unknown}")
        if not changes:
            raise AnnotationError("bad_edit", "nothing to change")
        with self._locked():
            current = self._current(note_id)
            self._check_version(current, expected_version)
            if current["deleted"]:
                raise AnnotationError("deleted", f"note {note_id} is deleted; restore it before editing")
            merged = {k: changes.get(k, current[k]) for k in allowed}
            geometry = _check_geometry(merged["shape"], merged["frame"], merged["coords"])
            text, kind, links = _check_text(merged["note"], merged["kind"], merged["links"])
            fields = {k: v for k, v in {"shape": merged["shape"], "frame": merged["frame"], "coords": geometry,
                                        "note": text, "kind": kind, "links": links}.items() if k in changes}
            self._append(self._event("edit", note_id, current["dataset"], author, run, fields))
            return self._current(note_id)

    def set_deleted(self, note_id: str, deleted: bool, author: str | None = None, run: str | None = None, expected_version: int | None = None) -> dict:
        with self._locked():
            current = self._current(note_id)
            self._check_version(current, expected_version)
            if current["deleted"] == deleted:
                return current
            self._append(self._event("delete" if deleted else "restore", note_id, current["dataset"], author, run))
            return self._current(note_id)

    def list(self, dataset: str, kind: str | None = None, include_deleted: bool = False) -> list[dict]:
        with self._locked():
            notes = self._notes_of(dataset)
        return sorted((n for n in notes if (include_deleted or not n["deleted"]) and (kind is None or n.get("kind") == kind)),
                      key=lambda n: n["created_at"])

    def history(self, note_id: str) -> list[dict]:
        with self._locked():
            if self._fold(self._events.get(note_id, [])) is None:
                raise AnnotationError("unknown_note", f"no note {note_id!r}")
            return [dict(e) for e in self._events[note_id]]


def _caller() -> tuple[str | None, str | None]:
    try:
        from fastmcp.server.dependencies import get_http_headers

        headers = {k.lower(): v for k, v in (get_http_headers(include_all=True) or {}).items()}
    except Exception:
        return None, None
    return headers.get("x-asyncroscopy-caller") or None, headers.get("x-asyncroscopy-run") or None


def _error(exc: AnnotationError) -> dict:
    return {"error": {"code": exc.code, "message": str(exc)}}


def register(mcp, store: AnnotationStore, dataset_key: Callable[[str], str | None]) -> None:
    """Add the note tools to ``mcp``. ``dataset_key(dataset)`` gives the one name a dataset's notes are kept under,
    or None when there is no such dataset."""

    def who(reported_by: str | None) -> tuple[str | None, str | None]:
        caller, run = _caller()
        return caller or (reported_by.strip() if isinstance(reported_by, str) and reported_by.strip() else None), run

    @mcp.tool()
    def annotate(dataset: str, shape: Literal["pin", "circle", "box", "range"], coords: dict[str, float],
                 frame: Literal["stage_m", "pixel", "energy_ev"] = "stage_m", note: str = "",
                 kind: str = "note", links: list[str] | None = None, reported_by: str | None = None) -> dict[str, Any]:
        """Put a note on a dataset (a map, an image or a spectrum) that people and agents both see.

        coords: pin {x, y}, circle {x, y, r}, box {x0, y0, x1, y1}, or range {start, end} on a spectrum.
        frame: metres for maps, pixels for images, or electronvolts (only for a range).
        kind: a short word such as note, particle, defect, revisit. links: data keys the note refers to.
        Returns the note with its id and version."""
        try:
            key = dataset_key(dataset)
        except Exception as exc:
            return {"error": {"code": "unavailable", "message": f"could not check that {dataset!r} exists: {exc}"}}
        if key is None:
            return {"error": {"code": "unknown_dataset", "message": f"no dataset or map {dataset!r} to annotate"}}
        author, run = who(reported_by)
        try:
            return store.create(key, shape, coords, frame, note, kind, links, author, run)
        except AnnotationError as exc:
            return _error(exc)

    @mcp.tool()
    def list_annotations(dataset: str, kind: str | None = None, include_deleted: bool = False, limit: int = 200) -> dict[str, Any]:
        """Notes on a dataset, oldest first: {"total", "notes": [at most limit]}. include_deleted shows deleted ones too."""
        try:
            key = dataset_key(dataset) or dataset
        except Exception as exc:
            return {"error": {"code": "unavailable", "message": f"could not look up {dataset!r}: {exc}"}}
        notes = store.list(key, kind=kind, include_deleted=include_deleted)
        return {"total": len(notes), "notes": notes[: max(0, int(limit))]}

    @mcp.tool()
    def edit_annotation(id: str, note: str | None = None, kind: str | None = None, coords: dict[str, float] | None = None,
                        shape: Literal["pin", "circle", "box", "range"] | None = None,
                        frame: Literal["stage_m", "pixel", "energy_ev"] | None = None, links: list[str] | None = None,
                        expected_version: int | None = None, reported_by: str | None = None) -> dict[str, Any]:
        """Change a note's text, kind, shape, coordinates or links; fields left out stay as they are.
        expected_version: the version you read; the edit is refused if someone changed the note since."""
        changes = {k: v for k, v in {"note": note, "kind": kind, "coords": coords, "shape": shape, "frame": frame, "links": links}.items() if v is not None}
        author, run = who(reported_by)
        try:
            return store.edit(id, changes, author, run, expected_version)
        except AnnotationError as exc:
            return _error(exc)

    @mcp.tool()
    def delete_annotation(id: str, expected_version: int | None = None, reported_by: str | None = None) -> dict[str, Any]:
        """Delete a note. Its history is kept and restore_annotation brings it back."""
        author, run = who(reported_by)
        try:
            return store.set_deleted(id, True, author, run, expected_version)
        except AnnotationError as exc:
            return _error(exc)

    @mcp.tool()
    def restore_annotation(id: str, expected_version: int | None = None, reported_by: str | None = None) -> dict[str, Any]:
        """Bring back a deleted note."""
        author, run = who(reported_by)
        try:
            return store.set_deleted(id, False, author, run, expected_version)
        except AnnotationError as exc:
            return _error(exc)

    @mcp.tool()
    def annotation_history(id: str) -> dict[str, Any]:
        """Every change to a note, oldest first: who, when, and what changed."""
        try:
            return {"id": id, "events": store.history(id)}
        except AnnotationError as exc:
            return _error(exc)
