"""Typed MetricID v2 (P03M, dossier 04) — the identity that gates
metric-field reuse.

C31-NOTE-1 resolution (errata E8): the cohort v1 ``result_key``
(``una:cohort:o_access:v1``) coerces ``knn_weights``/``d_weights`` to
float64 while the replay fold consumes the raw arrays.  It stays a
REPORTING key.  THIS module is the only reuse identity: every field
id is derived from the actual argument bindings — raw dtype, shape,
byteorder, layout, writability, alignment class AND payload bytes for
arrays; type tag AND raw bytes for scalars.  No tuple/object address,
filename, ``.tolist()`` float coercion, NaN normalization or -0
collapse anywhere.

Dependency closure per field kind (fold source:
``AccessibilityWElevation.reach_gravity_knn_access``, lines 73-110 at
the P03M tree; verified empirically by the committed perturbation and
split-parity batteries):

* common to every kind: trace identity (content + schema), the TYPED
  cutoff, the destination-weight vector bytes, output dtypes,
  profile tag, producer code closure, compiler/math identity.
* reach (lines 75-80): the common closure only — no decay scalar is
  consumed.
* gravity_exponential (line 83): + beta bytes, plateau bytes.
* gravity_logistic (line 85): + plateau, midpoint, growth-rate bytes
  (the computed quotient the fold actually receives — NOT the two
  inputs it is derived from).
* knn_access (lines 90-108): + knn_weights, the decay branch, and —
  branch-conditionally — (beta, plateau) for "exponential",
  (plateau, midpoint, growth rate) for "logistic", nothing further
  for any other branch.
* group (the whole fold): the union of all of the above.

A field id therefore CHANGES when a consumed input changes one bit
(dtype-only, byteorder-only, layout-only, writability-only, 0/-0,
NaN payload, int-vs-float scalar) and is INVARIANT under inputs the
fold does not read (e.g. beta does not touch reach / gravity_logistic
/ knn_access on the logistic branch) — the invariance is what makes
cross-row reuse bit-safe, and it is asserted by tests, never assumed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["EncodingRefused", "METRIC_ID_SCHEMA", "PROFILE_TAG",
           "array_part", "compiler_identity", "code_closure",
           "metric_field_id", "scalar_part", "weights_digest"]

METRIC_ID_SCHEMA = "una:metric-id:v2"
PROFILE_TAG = "una_legacy-epoch-a3"

_FIELD_KINDS = ("reach", "gravity_exponential", "gravity_logistic",
                "knn_access", "group")


class EncodingRefused(TypeError):
    """A value the typed identity refuses to encode (the supplied
    typed-frame law: production enumerates the actual supported NumPy
    scalar/dtype set instead of casting to Python float as a
    convenience)."""


def scalar_part(v: Any) -> Dict[str, str]:
    """Type tag + raw IEEE/two's-complement bytes.  Python and NumPy
    bool/int/float scalars only; anything else (str, complex, decimal,
    None, array) is refused, not coerced."""
    if isinstance(v, (bool, np.bool_)):
        return {"type": "np:" + str(np.dtype(np.bool_)),
                "bytes": np.asarray(v, dtype=np.bool_).tobytes().hex()}
    if isinstance(v, (int, np.integer)):
        i = int(v)
        for dt in (np.int64,):
            if -(1 << 63) <= i < (1 << 63):
                return {"type": "np:" + str(np.dtype(dt)) +
                                ("|py:int" if isinstance(v, int) else ""),
                        "bytes": dt(i).tobytes().hex()}
        raise EncodingRefused(f"int out of int64 range: {i!r}")
    if isinstance(v, (float, np.floating)):
        if isinstance(v, np.floating) and v.dtype.itemsize > 8:
            # float64 widening is lossless ONLY up to 64-bit floats;
            # extended precision is outside the fold's scalar domain
            # and is refused rather than silently narrowed.
            raise EncodingRefused(
                f"refusing float of itemsize {v.dtype.itemsize}: "
                "outside the fold's float64 scalar domain")
        f = np.asarray(v, dtype=np.float64)
        return {"type": "np:" + str(np.dtype(type(v) if isinstance(
                    v, np.floating) else float)),
                "bytes": f.tobytes().hex()}
    raise EncodingRefused(
        f"refusing to encode scalar of type {type(v).__name__}: "
        "only bool/int/float scalars are admitted (no string, "
        "complex, object or None coercion)")


def array_part(a: np.ndarray) -> Dict[str, Any]:
    """Raw-dtype identity for a 1-D weight vector: dtype string,
    shape, layout flags, writability, 8-byte alignment class and the
    logical-order payload bytes.  Byteorder rides the dtype string;
    the bytes are the array's own native representation, so a
    big-endian view with equal values still encodes differently."""
    if not isinstance(a, np.ndarray):
        raise EncodingRefused("weight vector must be an ndarray")
    if a.ndim != 1:
        raise EncodingRefused(
            f"weight vector must be 1-D, got ndim={a.ndim}")
    return {
        "dtype": a.dtype.str,
        "shape": list(a.shape),
        "c_contiguous": bool(a.flags.c_contiguous),
        "f_contiguous": bool(a.flags.f_contiguous),
        "writeable": bool(a.flags.writeable),
        "aligned8": bool(a.ctypes.data % 8 == 0),
        "bytes": a.tobytes().hex(),
    }


def weights_digest(a: np.ndarray) -> str:
    """One canonical digest of the weight vector, computed once per
    row and reused by every field id of that row (the payload bytes
    are the same bytes ``array_part`` encodes for an ndarray)."""
    return hashlib.sha256(
        json.dumps(array_part(a), sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def code_closure() -> Dict[str, str]:
    """The producer closure a field id binds: the replay/trace module
    and the fold module byte digests (cached — module bytes cannot
    change under a running process)."""
    global _CODE_CLOSURE
    if _CODE_CLOSURE is None:
        from . import _trace_access_scratch as _trm
        from . import AccessibilityWElevation as _fold_mod
        _CODE_CLOSURE = {
            "trace_module_sha256": hashlib.sha256(
                open(_trm.__file__, "rb").read()).hexdigest(),
            "fold_module_sha256": hashlib.sha256(
                open(_fold_mod.__file__, "rb").read()).hexdigest(),
        }
    return dict(_CODE_CLOSURE)


_CODE_CLOSURE: Optional[Dict[str, str]] = None


def compiler_identity() -> Dict[str, Any]:
    """The compiler/math environment the fold's bits depend on:
    numba/numpy/package versions, the fold's own target options and
    the declared reduction plan of each admitted kernel."""
    global _COMPILER_ID
    if _COMPILER_ID is None:
        import numba
        from .. import __version__ as _pkg_version
        from .AccessibilityWElevation import reach_gravity_knn_access \
            as _fold
        _COMPILER_ID = {
            "numba": numba.__version__,
            "numpy": np.__version__,
            "package": _pkg_version,
            "fold_targetoptions": dict(_fold.targetoptions),
            "reduction_plan": {
                "reach": "whole-fold-serial-v1",
                "gravity_exponential": "whole-fold-serial-v1",
                "gravity_logistic": "whole-fold-serial-v1",
                "knn_access": "whole-fold-serial-v1",
                "ge_split": "ge-only-serial-v1",
            },
        }
    return dict(_COMPILER_ID)


_COMPILER_ID: Optional[Dict[str, Any]] = None


def metric_field_id(kind: str, *, trace_id: str, trace_schema: str,
                    cutoff: Any, weights_sha: str,
                    reach_dtype: str, metric_dtype: str,
                    beta: Any, plateau: Any, midpoint: Any,
                    growth_rate: Any, knn_decay: str,
                    knn_weights_part: Dict[str, Any]) -> str:
    """The v2 field id for one metric kind over one row's bindings.

    ``weights_sha`` is the canonical digest of the raw destination
    weight vector (one sha per row, shared by all field ids of the
    row).  Every consumed scalar/array must be pre-encoded by the
    callers through :func:`scalar_part`/:func:`array_part`-shaped
    inputs; encoding is refused rather than coerced.  The per-kind
    closure below is the dependency table documented at module level —
    each inclusion cites the fold line that consumes it.
    """
    if kind not in _FIELD_KINDS:
        raise EncodingRefused(f"unknown field kind: {kind!r}")
    parts: Dict[str, Any] = {
        "schema": METRIC_ID_SCHEMA,
        "kind": kind,
        "trace_id": str(trace_id),
        "trace_schema": str(trace_schema),
        "cutoff": scalar_part(cutoff),
        "weights_sha": str(weights_sha),
        "reach_dtype": str(reach_dtype),
        "metric_dtype": str(metric_dtype),
        "profile": PROFILE_TAG,
        "code_closure": code_closure(),
        "compiler": compiler_identity(),
    }
    if kind in ("gravity_exponential", "group"):
        parts["beta"] = scalar_part(beta)          # fold line 83
        parts["plateau"] = scalar_part(plateau)    # fold line 83
    if kind in ("gravity_logistic", "group"):
        parts["plateau"] = scalar_part(plateau)    # fold line 85
        parts["midpoint"] = scalar_part(midpoint)  # fold line 85
        parts["growth_rate"] = scalar_part(growth_rate)  # line 85
    if kind == "knn_access":
        parts["knn_weights"] = knn_weights_part    # fold line 101
        parts["knn_decay"] = str(knn_decay)        # fold lines 103-108
        if knn_decay == "exponential":
            parts["beta"] = scalar_part(beta)      # fold line 104
            parts["plateau"] = scalar_part(plateau)
        elif knn_decay == "logistic":
            parts["plateau"] = scalar_part(plateau)      # line 106
            parts["midpoint"] = scalar_part(midpoint)
            parts["growth_rate"] = scalar_part(growth_rate)
        # any other branch consumes no decay scalar (line 108)
    if kind == "group":
        parts["knn_weights"] = knn_weights_part
        parts["knn_decay"] = str(knn_decay)
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(
        (METRIC_ID_SCHEMA + "\n" + blob).encode()).hexdigest()
