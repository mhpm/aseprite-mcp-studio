"""Where sprites live while the server runs: path policy, cache, history and locking.

* Paths: every path must resolve inside one of the allowed roots (ASE_STUDIO_ROOTS, separated
  by the OS path separator; default the working directory). Relative paths resolve against
  the first root. Symlinks are resolved before the check.
* Cache: a parsed sprite is kept with the file's mtime and size, so a sequence of calls on
  one file parses it once; a change on disk (the user saved in Aseprite) invalidates it.
* Edits are transactional: they run on a copy and only replace the file when every step
  succeeded. The previous bytes go on a per-file undo stack.
* One lock per file serialises concurrent calls on the same sprite.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from . import ase_format
from .document import Sprite, StudioError

UNDO_DEPTH = 50


class Store:
    def __init__(self, roots: list[str] | None = None):
        env = os.environ.get("ASE_STUDIO_ROOTS")
        raw = roots or (env.split(os.pathsep) if env else [os.getcwd()])
        self.roots = [Path(r).expanduser().resolve() for r in raw if r]
        self._cache: dict[Path, tuple[float, int, Sprite]] = {}
        self._undo: dict[Path, list[bytes | None]] = {}
        self._redo: dict[Path, list[bytes | None]] = {}
        self._locks: dict[Path, threading.Lock] = {}
        self._guard = threading.Lock()
        # called with the file's Path after every write, undo and redo (the live viewer listens)
        self.listeners: list[Callable[[Path], None]] = []

    def _notify(self, p: Path) -> None:
        for fn in list(self.listeners):
            with contextlib.suppress(Exception):  # a viewer must never break an edit
                fn(p)

    # ------------------------------------------------------------------ paths
    def resolve(self, path: str, must_exist: bool = False) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.roots[0] / p
        p = p.resolve()
        if not any(p == r or r in p.parents for r in self.roots):
            raise StudioError(
                f"{path} is outside the allowed folders {[str(r) for r in self.roots]}; "
                "set ASE_STUDIO_ROOTS to allow more"
            )
        if must_exist and not p.exists():
            raise StudioError(f"{p} does not exist")
        return p

    def lock(self, p: Path) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(p, threading.Lock())

    # ------------------------------------------------------------------ load/save
    def load(self, path: str) -> tuple[Path, Sprite]:
        p = self.resolve(path, must_exist=True)
        if p.suffix.lower() not in (".aseprite", ".ase"):
            raise StudioError(f"{p.name} is not an .aseprite/.ase file")
        st = p.stat()
        hit = self._cache.get(p)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return p, hit[2]
        spr = ase_format.read(p)
        self._cache[p] = (st.st_mtime, st.st_size, spr)
        return p, spr

    def save(self, p: Path, spr: Sprite, previous: bytes | None, record: bool = True) -> int:
        n = ase_format.write(spr, p)
        st = p.stat()
        self._cache[p] = (st.st_mtime, st.st_size, spr)
        if record:
            stack = self._undo.setdefault(p, [])
            stack.append(previous)
            del stack[:-UNDO_DEPTH]
            self._redo.pop(p, None)
        self._notify(p)
        return n

    @contextmanager
    def edit(self, path: str, create: Callable[[], Sprite] | None = None):
        """Transaction: yields (path, sprite copy); saves only if the block succeeds."""
        p = self.resolve(path)
        with self.lock(p):
            if p.exists():
                previous = p.read_bytes()
                _, spr = self.load(str(p))
                work = spr.clone()
            elif create is not None:
                previous, work = None, create()
            else:
                raise StudioError(f"{p} does not exist; create it first with sprite_new")
            yield p, work
            self.save(p, work, previous)

    def replace(self, path: str, spr: Sprite, overwrite: bool = False) -> Path:
        """Write a whole new sprite to `path` (undoable)."""
        p = self.resolve(path)
        with self.lock(p):
            if p.exists() and not overwrite:
                raise StudioError(f"{p} exists; pass overwrite=true to replace it")
            if p.suffix.lower() not in (".aseprite", ".ase"):
                raise StudioError("the output must end in .aseprite or .ase")
            previous = p.read_bytes() if p.exists() else None
            self.save(p, spr, previous)
        return p

    def undo(self, path: str, steps: int = 1) -> int:
        p = self.resolve(path)
        with self.lock(p):
            stack = self._undo.get(p, [])
            done = 0
            while done < steps and stack:
                prev = stack.pop()
                cur = p.read_bytes() if p.exists() else None
                self._redo.setdefault(p, []).append(cur)
                if prev is None:
                    p.unlink(missing_ok=True)
                    self._cache.pop(p, None)
                else:
                    p.write_bytes(prev)
                    self._cache.pop(p, None)
                done += 1
        if done:
            self._notify(p)
        return done

    def redo(self, path: str, steps: int = 1) -> int:
        p = self.resolve(path)
        with self.lock(p):
            stack = self._redo.get(p, [])
            done = 0
            while done < steps and stack:
                nxt = stack.pop()
                cur = p.read_bytes() if p.exists() else None
                self._undo.setdefault(p, []).append(cur)
                if nxt is None:
                    p.unlink(missing_ok=True)
                else:
                    p.write_bytes(nxt)
                self._cache.pop(p, None)
                done += 1
        if done:
            self._notify(p)
        return done

    def history(self, path: str) -> dict:
        p = self.resolve(path)
        return {"undo_steps": len(self._undo.get(p, [])), "redo_steps": len(self._redo.get(p, []))}
