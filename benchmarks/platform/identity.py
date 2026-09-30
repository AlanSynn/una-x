"""Installed-identity resolution (HARNESS.md "One real job", step 1).

Resolve, from a *child* interpreter with a scrubbed environment, where the
campaign package actually imports from, which distribution provides it, and
the content digest of the installed files.  A source checkout shadowing the
installed wheel must fail here (negative control 2).

The child snippet is deliberately tiny and prints one JSON object on a
dedicated marker fd so parent-side parsing never races stdout logging.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .errors import IdentityError

IDENTITY_SNIPPET = r"""
import hashlib, importlib, importlib.metadata as md, json, os, sys
pkg = sys.argv[1]
out_fd = int(sys.argv[2])
info = {"python": sys.executable, "python_version": sys.version.split()[0]}
try:
    mod = importlib.import_module(pkg)
    info["module"] = pkg
    info["module_file"] = getattr(mod, "__file__", None)
    info["module_path"] = list(getattr(mod, "__path__", []))
    try:
        info["distribution_version"] = md.version(pkg)
    except Exception:
        info["distribution_version"] = None
    root = list(getattr(mod, "__path__", []))[:1]
    digests = {}
    if root:
        import pathlib
        n = 0
        for p in sorted(pathlib.Path(root[0]).rglob("*")):
            if p.is_file() and n < 4000:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    h.update(f.read())
                digests[str(p.relative_to(root[0]))] = h.hexdigest()
                n += 1
        info["installed_files_sha256"] = digests
        agg = hashlib.sha256()
        for k in sorted(digests):
            agg.update(k.encode()); agg.update(digests[k].encode())
        info["installed_tree_sha256"] = agg.hexdigest()
    info["ok"] = True
except Exception as exc:  # reported, not raised: parent decides
    info["ok"] = False
    info["error"] = f"{type(exc).__name__}: {exc}"
os.write(out_fd, json.dumps(info).encode() + b"\n")
"""


def scrubbed_env(extra: dict | None = None) -> dict:
    """Environment for child interpreters: no PYTHONPATH, no source shadows.

    NUMBA cache roots stay caller-controlled (they are campaign-owned dirs
    and part of the cold/warm boundary definition, not an identity input).
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
                     "NUMBA_DISABLE_JIT")
    }
    env.pop("PYTHONPATH", None)
    if extra:
        env.update(extra)
    return env


def _read_marker(pipe) -> dict:
    line = pipe.readline()
    if not line:
        raise IdentityError("identity child produced no marker line")
    return json.loads(line)


def resolve_identity(python_exe: str, package: str = "urban_network_analysis",
                     extra_env: dict | None = None) -> dict:
    """Run the identity snippet in *python_exe* and return its JSON report."""
    rd, wd = os.pipe()
    try:
        proc = subprocess.Popen(
            [python_exe, "-c", IDENTITY_SNIPPET, package, str(wd)],
            env=scrubbed_env(extra_env), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, pass_fds=(wd,))
        os.close(wd)
        out, err = b"", b""
        try:
            info = _read_marker(os.fdopen(rd, "rb", closefd=False))
        finally:
            os.close(rd)
            out, err = proc.communicate(timeout=120)
        if proc.returncode != 0:
            raise IdentityError(
                f"identity probe exited {proc.returncode}: "
                f"{err.decode(errors='replace')[-800:]}")
        return info
    finally:
        try:
            os.close(rd)
        except OSError:
            pass


def assert_installed(info: dict, package: str,
                     expected_wheel_tree_sha256: str | None = None,
                     forbidden_roots: list[str] | None = None) -> dict:
    """Fail closed unless the package imports from an installed tree.

    forbidden_roots: source trees whose presence as the import root is the
    "source shadows installed wheel" defect (negative control 2).
    """
    if not info.get("ok"):
        raise IdentityError(
            f"{package} does not import in the target environment: "
            f"{info.get('error')}")
    mfile = info.get("module_file") or ""
    if "site-packages" not in mfile:
        raise IdentityError(
            f"{package} imports from {mfile!r}, not an installed "
            f"site-packages tree (source shadowing?)")
    for root in (forbidden_roots or []):
        r = str(Path(root).resolve())
        mroot = info.get("module_path") or [mfile]
        if any(str(Path(p)).startswith(r) for p in mroot):
            raise IdentityError(
                f"source tree {r} shadows the installed wheel "
                f"(module_path={info.get('module_path')})")
    if expected_wheel_tree_sha256 is not None:
        got = info.get("installed_tree_sha256")
        if got != expected_wheel_tree_sha256:
            raise IdentityError(
                f"installed tree digest {got} != frozen expectation "
                f"{expected_wheel_tree_sha256}")
    return info


def is_source_shadow(info: dict, source_root: str) -> bool:
    """True when the import root sits under *source_root* (mutant 2 state)."""
    mfile = info.get("module_file") or ""
    try:
        return Path(mfile).resolve().is_relative_to(Path(source_root).resolve())
    except (ValueError, OSError):
        return False


def source_tree_sha256(source_root: str, limit: int = 20000) -> str:
    """Deterministic digest of the candidate source tree (drift detection)."""
    root = Path(source_root)
    agg = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix == ".py" and n < limit:
            agg.update(str(p.relative_to(root)).encode())
            agg.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
            n += 1
    return agg.hexdigest()
