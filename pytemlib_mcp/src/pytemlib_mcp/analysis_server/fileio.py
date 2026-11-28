"""A lock between processes and sequenced JSON-lines appends, for the tool log, the notes and the placements.

``SequencedAppender`` gives every line of a JSON-lines file the file's next ``seq`` and the Unix time it was
``written``, under a lock shared by every process appending to that file. It is asyncroscopy's
(``asyncroscopy.core.fileio``), copied so that pyTEMlib does not depend on asyncroscopy: the two write the same
lines and take the same lock, so asyncroscopy's trace reader merges this server's logs with its own.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Callable, Iterator

LOCK_TIMEOUT_S = 5.0  # longest one attempt waits for another process's append
LOCK_ATTEMPTS = 3  # attempts before an append gives up and raises
_LOCK_BYTE = 1 << 20  # Windows locks this byte of the .seq file. It sits past the data, so the data stays readable


def seq_path(path: str | Path) -> Path:
    """The sidecar holding a log's last ``seq`` and ``written`` time: ``<name>.seq`` next to it.

    Every process appending to the log shares it, and it outlives a rotation of the log, so
    ``seq`` keeps rising when ``jobs.jsonl`` is renamed to ``jobs.jsonl.1`` and a new one begun.
    """
    path = Path(path)
    return path.with_name(path.name + ".seq")


@contextmanager
def file_lock(handle: IO[bytes], deadline: float) -> Iterator[None]:
    """An exclusive lock between processes on ``handle`` (opened "a+b"), waited for until ``deadline``.

    Raises TimeoutError when another process holds it past ``deadline``.
    """
    if os.name == "nt":
        import msvcrt

        def attempt() -> None:
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

        def release() -> None:
            handle.seek(_LOCK_BYTE)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        def attempt() -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release() -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    while True:
        try:
            attempt()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("another process held the file's lock too long") from None
            time.sleep(0.002)
    try:
        yield
    finally:
        release()


def rotated_paths(path: str | Path) -> list[Path]:
    """The rotated files of a log (``<name>.1``, ``<name>.2``, ...), oldest first."""
    path = Path(path)
    prefix = path.name + "."
    found = []
    try:
        names = os.listdir(path.parent)
    except OSError:
        return []
    for name in names:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            found.append((int(name[len(prefix):]), name))
    return [path.with_name(name) for _, name in sorted(found)]


def rotate(path: str | Path, keep: int = 0) -> Path | None:
    """Rename ``path`` to the next free ``<name>.<n>`` and delete the oldest rotated files beyond ``keep`` (0: keep all).

    The caller holds the file's lock. Returns the new name, or None when the file could not be renamed (on
    Windows, while another process has it open): the file stays where it is and the next append tries again.
    Numbers only ever rise, so a rotated file is never renamed again and a reader can follow it by name.
    """
    path = Path(path)
    rotated = rotated_paths(path)
    number = int(rotated[-1].name.rsplit(".", 1)[1]) + 1 if rotated else 1
    target = path.with_name(f"{path.name}.{number}")
    try:
        os.rename(path, target)
    except OSError:
        return None
    if keep > 0:
        for old in rotated[: max(0, len(rotated) + 1 - keep)]:
            try:
                os.unlink(old)
            except OSError:  # still open in a reader, so it goes on a later rotation
                pass
    return target


def _last_seq_in(path: Path) -> tuple[int, float]:
    """(highest seq, latest written) already in ``path``, or in its newest rotated file when ``path`` is new:
    how a log without a .seq sidecar is continued."""
    seq, written = 0, 0.0
    if not path.exists() or path.stat().st_size == 0:
        rotated = rotated_paths(path)
        if not rotated:
            return seq, written
        path = rotated[-1]
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                if isinstance(row.get("seq"), int):
                    seq = max(seq, row["seq"])
                if isinstance(row.get("written"), (int, float)):
                    written = max(written, float(row["written"]))
    return seq, written


class SequencedAppender:
    """Appends JSON lines to one file, each with the file's next ``seq`` and its ``written`` time.

    ``seq`` rises by one per line of the file whichever thread or process wrote it: the sidecar
    (``seq_path``) is locked while a line is numbered and appended, so lines land in ``seq``
    order. ``written`` is the Unix time of the append, never earlier than the line before it;
    the merging reader orders the lines of several files by it. The file is opened for each
    line and closed again, so it can be renamed (rotated) between lines, on Windows too.

    ``rotate_bytes``: above 0, a line that would take the file past this size first renames the file
    to ``<name>.<n>`` (``rotate``), under the same lock, so ``seq`` carries on in the new file.
    ``keep``: rotated files kept (0: all). ``may_rotate``: asked before rotating; False waits for a
    later append (a job device reading its own events back by offset). It runs under the lock, so
    it must not wait on anything that may itself be appending.

    A line is never written without its ``seq``: when the lock cannot be had after
    ``LOCK_ATTEMPTS`` tries the append raises TimeoutError and writes nothing.
    """

    def __init__(
        self, path: str | Path, lock_timeout_s: float = LOCK_TIMEOUT_S, attempts: int = LOCK_ATTEMPTS,
        rotate_bytes: int = 0, keep: int = 0, may_rotate: Callable[[], bool] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = lock_timeout_s
        self.attempts = max(1, int(attempts))
        self.rotate_bytes = int(rotate_bytes)
        self.keep = int(keep)
        self.may_rotate = may_rotate
        self._lock = threading.Lock()

    def _rotate_for(self, incoming: int) -> None:
        if self.rotate_bytes <= 0:
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size == 0 or size + incoming <= self.rotate_bytes:
            return
        if self.may_rotate is None or self.may_rotate():
            rotate(self.path, self.keep)

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        """Number ``row``, append it as one line and return it.

        Raises TimeoutError when another process holds the lock through every attempt, and
        OSError when the file cannot be written.
        """
        with self._lock:
            for attempt in range(self.attempts):
                try:
                    with seq_path(self.path).open("a+b") as counter, file_lock(counter, time.monotonic() + self.lock_timeout_s):
                        counter.seek(0)
                        stored = counter.read(100).decode("ascii", errors="replace").split()
                        try:
                            last_seq, last_written = int(stored[0]), float(stored[1])
                        except (IndexError, ValueError):  # a new or damaged sidecar, so continue from the log itself
                            last_seq, last_written = _last_seq_in(self.path)
                        row["seq"] = last_seq + 1
                        row["written"] = max(time.time(), last_written)
                        line = json.dumps(row) + "\n"
                        self._rotate_for(len(line.encode("utf-8")))
                        with self.path.open("a", encoding="utf-8") as handle:
                            handle.write(line)
                        counter.seek(0)
                        counter.truncate()
                        counter.write(f"{row['seq']} {row['written']!r}".encode("ascii"))
                    return row
                except TimeoutError:
                    if attempt + 1 == self.attempts:
                        raise TimeoutError(f"{self.path}: another process held its lock through {self.attempts} attempts; line not written") from None
        raise AssertionError("unreachable")  # pragma: no cover
