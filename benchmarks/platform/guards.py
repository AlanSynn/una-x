"""Safety guards (HARNESS.md "Custody and reuse"; negative controls 7).

* OutputDirGuard  — one run per output directory; a completed manifest from a
  different run ID makes the directory ineligible.  All outputs, caches and
  temp state live in campaign-owned directories outside the source tree.
* SourceWriteGuard — an audit hook that fails a run before it can modify any
  git-tracked (committed) file.  Installed in children around the dispatch.
* tracked_files / assert_not_tracked — repo-scoped helpers shared by both.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

from .errors import OutputDirReuseError, SafetyViolation


def tracked_files(repo: str | Path) -> set[str]:
    """Absolute paths of all git-tracked files in *repo* (cache: per call)."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(repo),
                         capture_output=True, check=True)
    root = Path(repo).resolve()
    return {str(root / rel) for rel in out.stdout.decode().split("\0") if rel}


def assert_not_tracked(path: str | Path, repo: str | Path) -> None:
    """Raise SafetyViolation if *path* (or any ancestor) is git-tracked."""
    p = Path(path).resolve()
    tracked = tracked_files(repo)
    if str(p) in tracked:
        raise SafetyViolation(f"{p} is a committed file; runs must write "
                              "only campaign-owned out-of-tree directories")
    for t in tracked:
        if t.startswith(str(p) + os.sep):
            raise SafetyViolation(
                f"{p} contains committed files; refusing to use it as a "
                "run output directory")


class OutputDirGuard:
    """One unique run ID per output directory, stamped at prepare time."""

    MARKER = "run_manifest.json"

    def __init__(self, out_dir: str | Path, run_id: str, repo: str | Path):
        self.out_dir = Path(out_dir).resolve()
        self.run_id = run_id
        self.repo = str(Path(repo).resolve())
        self._finalized = False

    def prepare(self) -> Path:
        assert_not_tracked(self.out_dir, self.repo)
        existing = self.out_dir / self.MARKER
        if existing.is_file():
            try:
                prior = json.loads(existing.read_text(encoding="utf-8"))
            except Exception:
                prior = {}
            prior_id = prior.get("run_id")
            if prior_id and prior_id != self.run_id:
                raise OutputDirReuseError(
                    f"{self.out_dir} holds a completed manifest for run "
                    f"{prior_id!r}; refusing reuse by run {self.run_id!r} "
                    "(negative control 7)")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / self.MARKER).write_text(json.dumps({
            "run_id": self.run_id,
            "state": "prepared",
            "pid": os.getpid(),
        }, indent=2) + "\n", encoding="utf-8")
        return self.out_dir

    def finalize(self, payload: dict) -> None:
        if self._finalized:
            return
        self._finalized = True
        (self.out_dir / self.MARKER).write_text(json.dumps(
            {"run_id": self.run_id, "state": "complete", **payload},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SourceWriteGuard:
    """Audit hook: forbid writes to git-tracked paths during a run.

    Install inside the process that executes candidate work (child, or the
    test process for in-process mutants).  The guard denies *before* the
    write happens, so committed files stay byte-identical.
    """

    def __init__(self, tracked: set[str] | None = None, repo: str | None = None):
        if tracked is None:
            if repo is None:
                raise ValueError("SourceWriteGuard needs tracked or repo")
            tracked = tracked_files(repo)
        self.tracked = {str(Path(t).resolve()) for t in tracked}
        self.violations: list[str] = []
        self._installed = False

    # -- audit hook -----------------------------------------------------
    def _hook(self, event: str, args) -> None:
        if event == "open":
            path, mode, _flags = (list(args) + [None, None])[:3]
            if mode and any(m in str(mode) for m in ("w", "a", "x", "+")):
                self._check(path)
        elif event in ("os.remove", "os.rename", "os.replace", "os.rmdir",
                       "os.truncate", "shutil.copyfile"):
            src = args[0] if args else None
            self._check(src)

    def _check(self, path) -> None:
        if not path:
            return
        try:
            resolved = str(Path(os.fspath(path)).resolve())
        except (TypeError, ValueError, OSError):
            return
        if resolved in self.tracked:
            self.violations.append(resolved)
            raise SafetyViolation(
                f"attempt to modify committed file {resolved} during a run "
                "(negative control 7: evidence-write guard)")

    def __enter__(self):
        sys.addaudithook(self._hook)
        self._installed = True
        return self

    def __exit__(self, *exc):
        # Audit hooks cannot be removed; this object is per-run and cheap.
        self._installed = False
        return False
