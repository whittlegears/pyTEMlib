"""Named plug-in classes (the scorers ``score_map`` offers), found from decorators, plug-in modules and entry points.

This is asyncroscopy's ``Registry`` and plug-in loader (``asyncroscopy.registry``, ``asyncroscopy.plugins``), copied
so that pyTEMlib does not depend on asyncroscopy. Classes arrive three ways:

- ``@registry.register`` on the class, when its module is imported;
- ``ASYNCROSCOPY_PLUGINS``: modules or ``.py`` files, separated by commas or ``os.pathsep``, imported once per process;
- the package entry-point ``group``, for installed packages.

A source that fails to load is remembered with its error, and the "unknown name" error lists those failures.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from pathlib import Path
from typing import Generic, TypeVar

from . import environment
from .environment import PLUGINS_ENV

def plugin_specs(value: str | None = None) -> list[str]:
    """The entries of ``value`` (default: the ``ASYNCROSCOPY_PLUGINS`` variable), in order, without blanks or repeats."""
    text = environment.get().plugins if value is None else value
    parts = re.split("[,%s]" % re.escape(os.pathsep), text)
    seen: list[str] = []
    for part in (p.strip() for p in parts):
        if part and part not in seen:
            seen.append(part)
    return seen


def is_file_spec(spec: str) -> bool:
    """True when ``spec`` names a Python file rather than a module."""
    return spec.endswith(".py") or "/" in spec or "\\" in spec or Path(spec).is_file()


def _import_file(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file {path}")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"\W", "_", path.stem)
    name = f"asyncroscopy_plugin_{stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling look the module up while it runs
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class Plugins:
    """The plug-in modules named by ``ASYNCROSCOPY_PLUGINS`` (or by ``source``), each imported at most once."""

    def __init__(self, source: Callable[[], str] | None = None) -> None:
        self.source = source or (lambda: environment.get().plugins)
        self._tried: set[str] = set()
        self._errors: dict[str, str] = {}

    def specs(self) -> list[str]:
        """The entries named now."""
        return plugin_specs(self.source())

    def load(self, specs: list[str] | str | None = None) -> dict[str, str]:
        """Import every plug-in not yet tried in this process; returns {entry: error} for those that failed.

        ``specs`` defaults to the source, which is read again on each call, so one set
        later in the process is still picked up.
        """
        if specs is None:
            entries = self.specs()
        elif isinstance(specs, str):
            entries = plugin_specs(specs)
        else:
            entries = list(specs)
        for spec in entries:
            if spec in self._tried:
                continue
            self._tried.add(spec)
            try:
                if is_file_spec(spec):
                    _import_file(Path(spec))
                else:
                    importlib.import_module(spec)
            except Exception as exc:  # a broken plug-in must not take the device down, so it is reported instead
                self._errors[spec] = f"{type(exc).__name__}: {exc}"
                print(f"[pyTEMlib.analysis_server.registry] could not import {spec!r}: {self._errors[spec]}", file=sys.stderr)
        return {spec: self._errors[spec] for spec in entries if spec in self._errors}

    def errors(self) -> dict[str, str]:
        """{entry: error} for every plug-in that failed to import."""
        return dict(self._errors)

    def hint(self) -> str:
        """Text for an "unknown name" error: what failed to import, or how to make a plug-in visible."""
        if self._errors:
            return "; plug-ins that failed to import: " + ", ".join(f"{k} ({v})" for k, v in self._errors.items())
        if not self.specs():
            return f"; a class defined in your own module is found only if {PLUGINS_ENV} names it"
        return ""


PLUGINS = Plugins()

T = TypeVar("T", bound=type)


class Registry(Generic[T]):
    """Named classes of one kind, found from decorators, built-in modules, plug-ins and entry points.

    ``kind`` names the thing in error messages ("instrument profile"). ``key`` is the class attribute
    holding a class's registered name (``name`` for most, ``source`` for specimens).
    """

    def __init__(
        self,
        group: str,
        builtins: Mapping[str, str] | None = None,
        *,
        kind: str,
        key: str = "name",
        plugins: Plugins | None = None,
    ) -> None:
        self.group = group
        self.builtins: dict[str, str] = dict(builtins or {})
        self.kind = kind
        self.key = key
        self._plugins = plugins if plugins is not None else PLUGINS
        self._classes: dict[str, T] = {}
        self._entry_points_loaded = False
        self._errors: dict[str, str] = {}

    # registering
    def register(self, cls: T) -> T:
        """Make ``cls`` available under its ``key`` attribute; usable as a class decorator."""
        name = getattr(cls, self.key, "")
        if not name:
            raise ValueError(f"{cls.__name__} needs a {self.key}")
        self._classes[str(name)] = cls
        return cls

    def unregister(self, name: str) -> None:
        """Forget ``name`` (tests register throwaway classes)."""
        self._classes.pop(name, None)

    # looking up
    def get(self, name: str) -> T:
        """The class registered as ``name``; ValueError naming the known ones and any load failures otherwise."""
        if name not in self._classes and name in self.builtins:
            self._import(self.builtins[name])
        if name not in self._classes:
            self.load()
        if name not in self._classes:
            raise ValueError(f"unknown {self.kind} {name!r}; registered: {sorted(self._classes)}{self.hint()}")
        return self._classes[name]

    def all(self) -> dict[str, T]:
        """Every class from every source, as a new {name: class} dict."""
        self.load()
        return dict(self._classes)

    def __contains__(self, name: object) -> bool:
        return name in self._classes

    def load(self) -> None:
        """Import the built-in modules, the plug-ins, and (once) the entry points."""
        for module in dict.fromkeys(self.builtins.values()):
            self._import(module)
        if self._plugins is not None:
            self._plugins.load()
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for point in entry_points(group=self.group):
            try:
                self.register(point.load())
            except Exception as exc:  # a broken plug-in must not take the device down, so it is reported instead
                self._fail(f"entry point {point.name!r}", exc)

    def errors(self) -> dict[str, str]:
        """{source: error} for every built-in module or entry point that failed to load."""
        return dict(self._errors)

    def hint(self) -> str:
        """Text for an "unknown name" error: what failed to load, or how to make a plug-in visible."""
        text = "".join(f"; {source} failed to load ({error})" for source, error in self._errors.items())
        if self._plugins is not None:
            text += self._plugins.hint()
        return text

    # internals
    def _import(self, module: str) -> None:
        label = f"built-in module {module!r}"
        if label in self._errors:
            return
        try:
            importlib.import_module(module)
        except Exception as exc:
            self._fail(label, exc)

    def _fail(self, label: str, exc: Exception) -> None:
        self._errors[label] = f"{type(exc).__name__}: {exc}"
        print(f"[pyTEMlib.analysis_server.registry] {self.group}: could not load {label}: {self._errors[label]}", file=sys.stderr)
