"""Numerical stage identity for content-addressed reuse (campaign CACHE_GRAPH,
dossier 06 "Required numerical key closure").

This module decides WHAT is numerically identical.  A stage key is a sha256
digest over a canonical encoding of the stage's COMPLETE dependency closure:

- stage name + identity schema version (bump on any closure-semantics change),
- content digests of every input file's exact bytes (never paths or mtimes —
  same bytes at a different path is the same numerical input; same path with
  different bytes is a different input),
- parent stage keys (the stage DAG: snapping depends on the topology stage,
  results depend on the snapped inputs, so mutating ANY upstream dependency
  misses everywhere downstream),
- every scalar/tuple/array parameter that reaches the stage's arithmetic
  (including -0.0 vs 0.0 and NaN payloads, encoded via IEEE hex),
- the semantic profile (una_legacy / madina_legacy / corrected_v1) and the
  environment fingerprint (interpreter + library + GEOS/LLVM versions +
  machine) for every producer whose output bits may differ across that
  environment.

Publication-only state (output folders, filenames, format flags, timestamps,
result prefixes) is deliberately UNREPRESENTABLE in these keys: changing it
can never force a numerical miss, and a numerical change can never hit
because filenames happen to match (dossier 06 "Numerical identity is not
export identity").

The encoding is strict: a dependency type it cannot canonicalize raises
TypeError instead of silently widening the key.  An under-closed key is the
one unrecoverable failure mode of a numerical cache, so this module fails
loudly rather than approximate.

Only numpy + stdlib are imported here; nothing in this package may pull the
analysis stack (CACHE_STORE package rule).
"""
from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "IDENTITY_SCHEMA",
    "geometry_env",
    "compute_env",
    "file_digest",
    "arrays_digest",
    "stage_key",
]

# Bump when any producer's closure semantics change (a new numerical input
# enters a stage, a canonicalization rule changes).  Every key embeds this.
IDENTITY_SCHEMA = 1

# Canonical-encoding tags.  Length-prefixed and typed so concatenations are
# unambiguous (no delimiter-injection from values).
_TAG_NONE = b"n;"
_TAG_TRUE = b"b1;"
_TAG_FALSE = b"b0;"
_TAG_FLOAT_NAN = b"f~nan;"
_TAG_FLOAT_INF = b"f~inf;"
_TAG_FLOAT_NINF = b"f~ninf;"


def _enc(value: Any, out: list) -> None:
    """Append the canonical encoding of ``value`` to ``out`` (byte fragments)."""
    # bool BEFORE int (bool is an int subclass)
    if value is None:
        out.append(_TAG_NONE)
    elif value is True:
        out.append(_TAG_TRUE)
    elif value is False:
        out.append(_TAG_FALSE)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        out.append(b"s%d:" % len(raw))
        out.append(raw)
        out.append(b";")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        out.append(b"d%d:" % len(raw))
        out.append(raw)
        out.append(b";")
    elif isinstance(value, float):
        # IEEE-754 hex is injective for binary64 INCLUDING -0.0 vs 0.0 and
        # NaN payloads; repr() is not guaranteed injective for payloads.
        if value != value:
            out.append(_TAG_FLOAT_NAN)
        elif value == float("inf"):
            out.append(_TAG_FLOAT_INF)
        elif value == float("-inf"):
            out.append(_TAG_FLOAT_NINF)
        else:
            out.append(b"f%s;" % float(value).hex().encode("ascii"))
    elif isinstance(value, int):
        out.append(b"i%dd;" % value)
    elif isinstance(value, (np.integer,)):
        out.append(b"i%dd;" % int(value))
    elif isinstance(value, (np.floating,)):
        _enc(float(value), out)
    elif isinstance(value, (np.bool_,)):
        out.append(_TAG_TRUE if bool(value) else _TAG_FALSE)
    elif isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        out.append(b"a%s;%s;%d;%s;" % (
            a.dtype.str.encode("ascii"),
            b",".join(b"%d" % s for s in a.shape),
            a.ndim,
            hashlib.sha256(a.tobytes()).hexdigest().encode("ascii")))
    elif isinstance(value, (list, tuple)):
        out.append(b"l%d:" % len(value))
        for item in value:
            _enc(item, out)
        out.append(b";")
    elif isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(
                f"canonical dict keys must be str, got types "
                f"{sorted({type(k).__name__ for k in value})}")
        out.append(b"m%d:" % len(value))
        for k in sorted(value):
            _enc(k, out)
            _enc(value[k], out)
        out.append(b";")
    else:
        raise TypeError(
            f"cannot canonicalize dependency of type {type(value).__name__} "
            f"(value {value!r}); numerical keys must be built from bytes, "
            f"numbers, strings, bools, None, arrays and (nested) mappings/"
            f"sequences of those — refusing to widen the key silently")


def stage_key(stage: str, deps: Mapping[str, Any],
              schema: int = IDENTITY_SCHEMA) -> str:
    """One stage key: sha256 over the canonical (stage, schema, deps) closure."""
    out: list = []
    _enc({"stage": str(stage), "schema": int(schema), "deps": dict(deps)}, out)
    return hashlib.sha256(b"".join(out)).hexdigest()


# ---------------------------------------------------------------- digests

def file_digest(data: bytes) -> str:
    """Content digest of one input file's exact bytes (the snapshot read)."""
    return hashlib.sha256(data).hexdigest()


def arrays_digest(arrays: Mapping[str, Optional[np.ndarray]]) -> str:
    """Digest over a set of named arrays (dtype.str + shape + content each).

    None entries encode as the bare name (presence/absence is part of the
    identity — a stage that produced no z array is not the same stage).
    """
    out: list = []
    for name in sorted(arrays):
        out.append(b"k%s;" % name.encode("utf-8"))
        a = arrays[name]
        if a is None:
            out.append(_TAG_NONE)
        else:
            _enc(np.ascontiguousarray(a), out)
    return hashlib.sha256(b"".join(out)).hexdigest()


# ------------------------------------------------------- environment prints

def _package_libs_geometry() -> dict:
    import geopandas
    import pyogrio
    import shapely
    try:
        geos = ".".join(str(v) for v in shapely.geos_version)
    except Exception:  # pragma: no cover - version introspection only
        geos = "unknown"
    return {
        "numpy": np.__version__,
        "shapely": shapely.__version__,
        "geos": geos,
        "geopandas": geopandas.__version__,
        "pyogrio": pyogrio.__version__,
    }


def geometry_env(profile: str, package_version: str) -> dict:
    """Environment whose variation may change PRODUCED geometry/decode bits.

    Covers every producer feeding the topology and snapping stages: file
    decoding (pyogrio/pyarrow/geopandas), length/coordinate arithmetic and
    STRtree behavior (shapely/GEOS), and array arithmetic (numpy) — plus the
    interpreter, platform, semantic profile and package version.  Included
    in every stage key: over-closure costs occasional hits after a library
    upgrade; under-closure could serve bits a different environment would
    not have produced.
    """
    import pyarrow
    libs = _package_libs_geometry()
    libs["pyarrow"] = pyarrow.__version__
    return {
        "kind": "geometry",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "profile": str(profile),
        "package_version": str(package_version),
        "libs": libs,
    }


def _host_cpu_identity() -> dict:
    """Host CPU model/ISA identity for the compute-env fingerprint (review
    MINOR-2): llvmlite's codegen targets the host CPU, so a different model
    or feature set may change JIT bit patterns even under identical library
    versions.  Module-level so tests can monkeypatch it; best-effort — an
    unreadable source degrades to a stable "unknown" marker rather than
    raising (a missing fingerprint must cost hits, never crash a run)."""
    name = "unknown"
    try:
        from llvmlite import binding as _llb

        name = _llb.get_host_cpu_name()
        if isinstance(name, bytes):
            name = name.decode("ascii", "replace")
    except Exception:
        name = "unknown"
    model = "unknown"
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        model = "unknown"
    return {"name": str(name), "model": str(model)}


def compute_env(profile: str, package_version: str) -> dict:
    """Environment for JIT/compiled numerics: geometry env + the compiler
    stack (numba/llvmlite/scipy) whose version and codegen may change bits
    (dossier 06: "compiler/math fingerprint when bits may differ") + the
    host CPU identity feeding that codegen (review MINOR-2)."""
    import llvmlite
    import numba
    env = geometry_env(profile, package_version)
    env["kind"] = "compute"
    env["cpu"] = _host_cpu_identity()
    env["libs"]["numba"] = numba.__version__
    env["libs"]["llvmlite"] = llvmlite.__version__
    try:
        import scipy
        env["libs"]["scipy"] = scipy.__version__
    except ImportError:  # scipy is optional at runtime
        env["libs"]["scipy"] = "absent"
    return env
