"""Typed MetricID v2 (campaign una_10x post_p00, designs/04).

The C31 cohort ``result_key`` float64-coerces ``d_weights`` /
``knn_weights`` and ``float()``-coerces every scalar (E8): it is a
REPORTING key and must never be promoted into an authoritative reuse
key.  This module owns the authoritative identity instead: a versioned
typed MetricID whose fields are derived from the REAL argument
bindings the compiled fold consumes — never from names, addresses,
``.tolist()`` float coercion, NaN normalization or -0 collapse.

Identity fields (designs/04, verbatim obligations):

* metric binding schema + closure version (versioned WITHOUT touching
  the trace producer schema: the C11 trace identity is unchanged);
* the trace content+schema ID the replay consumes (the record's
  ``trace_id``) plus the record's OUTPUT dtype/cast fields — the cast
  identity the replayed arrays materialize under;
* for EVERY array the fold reads (``d_weights``, ``knn_weights``):
  the ORIGINAL dtype (including byte order), shape, logical-order
  bytes, and the meaningful layout/writability/alignment
  specialization flags;
* for EVERY scalar the fold reads: the scalar's own dtype AND its
  exact IEEE-754/integer bit pattern (-0.0 != +0.0, every 1-ULP and
  NaN-payload change visible);
* the KNN decay branch selector (the fold's ``knn_decay`` string);
* the profile;
* the fold/helper code closure: the fold module's byte digest, the
  fold's compilation flags and the compiler/math environment
  identities, plus the logical reduction plan.

Encoding is REFUSAL-BASED: the supplied typed-frame model refuses
types it cannot encode.  Production enumerates the actual NumPy
scalar/dtype support instead of casting to Python float as a
convenience — a complex, string, object or oversize integer input
raises ``TypeError``/``ValueError`` and NO identity exists for it.

The canonical frame law is the C10/C11 encoder (schema order, bit
identity), imported from the trace module — one law, no drift.
"""
import hashlib

import numba as nb
import numpy as np

from ._trace_access_scratch import _identity

__all__ = ["METRIC_ID_SCHEMA_VERSION", "METRIC_CLOSURE_VERSION",
           "METRIC_ID_REFUSAL", "MetricIdentityError",
           "typed_array_frame", "typed_scalar_frame", "fold_closure",
           "compute_metric_id", "metric_binding_fields"]

METRIC_ID_SCHEMA_VERSION = "una-10x-metricid-v2-1"
# Bumped ONLY when the binding field set or closure derivation changes
# meaning.  A bump invalidates metric IDs without invalidating an
# unchanged valid trace producer (the trace schema is untouched).
METRIC_CLOSURE_VERSION = 1

# The route-facing refusal reason (cohort counters key).
METRIC_ID_REFUSAL = "metric_id_not_encodable"


class MetricIdentityError(TypeError):
    """A binding input cannot be encoded losslessly — no identity."""


# --------------------------------------------------------------------------
# Typed frames
# --------------------------------------------------------------------------
_ARRAY_KINDS_OK = "biuf"          # bool, signed/unsigned int, float
_SCALAR_INTEGER_ITEMSIZES = (1, 2, 4, 8)
_SCALAR_FLOAT_ITEMSIZES = (2, 4, 8, 16)   # includes longdouble where real


def _dtype_label(a):
    return str(np.dtype(a.dtype))


def _check_dtype_encodable(dtype):
    if dtype.kind not in _ARRAY_KINDS_OK:
        raise MetricIdentityError(
            f"array dtype {dtype} (kind {dtype.kind!r}) is not "
            "encodable as a metric identity; refusing instead of "
            "coercing")
    if dtype.kind in "iu" and dtype.itemsize not in _SCALAR_INTEGER_ITEMSIZES:
        raise MetricIdentityError(f"integer dtype {dtype} unsupported")
    if dtype.kind == "f" and dtype.itemsize not in _SCALAR_FLOAT_ITEMSIZES:
        raise MetricIdentityError(f"float dtype {dtype} unsupported")


def typed_array_frame(pairs, name, a):
    """Append the typed closure of one array the fold reads.

    Layout law: ``bytes`` are the LOGICAL-order bytes (C-order element
    sequence, the order the fold's gathers read), while the ORIGINAL
    dtype (byte order included), shape and the
    layout/writability/alignment specialization are recorded beside
    them — so dtype-only, byte-order-only and layout-only changes each
    yield a different identity.  ``bytes`` are appended after the
    name/type/shape fields as raw unframed bytes via the shared
    canonical encoder (the caller passes them through ``pairs``).
    """
    if not isinstance(a, np.ndarray):
        raise MetricIdentityError(
            f"{name}: expected an ndarray, got {type(a)!r}")
    _check_dtype_encodable(a.dtype)
    logical = np.ascontiguousarray(a)
    pairs.append((f"{name}.dtype", _dtype_label(a)))
    pairs.append((f"{name}.shape",
                  [int(x) for x in a.shape]))
    pairs.append((f"{name}.layout",
                  [bool(a.flags.c_contiguous), bool(a.flags.f_contiguous),
                   bool(a.flags.writeable), bool(a.flags.aligned)]))
    pairs.append((f"{name}.logical_order_bits", logical.tobytes()))


def typed_scalar_frame(pairs, name, v):
    """Append the typed closure of one scalar the fold reads.

    Enumerated support (designs/04: enumerate, never convenience-cast):
    NumPy scalars of kind b/i/u/f with the supported itemsizes, and the
    Python ``bool``/``int``/``float`` builtins under their own exact
    semantics (python float IS IEEE-754 binary64; python int must fit
    signed 64 bits or the identity refuses).  Complex, string, bytes,
    ``None`` and everything else refuse.  The value enters by its own
    dtype's bytes: -0.0, NaN payloads and every 1-ULP change are
    distinct; no NaN normalization and no -0 collapse exists.
    """
    if isinstance(v, np.bool_):
        dtype, raw = "bool", np.asarray(v).tobytes()
    elif isinstance(v, np.integer):
        if v.dtype.itemsize not in _SCALAR_INTEGER_ITEMSIZES:
            raise MetricIdentityError(
                f"{name}: integer dtype {v.dtype} unsupported")
        dtype, raw = str(v.dtype), np.asarray(v).tobytes()
    elif isinstance(v, np.floating):
        if v.dtype.itemsize not in _SCALAR_FLOAT_ITEMSIZES:
            raise MetricIdentityError(
                f"{name}: float dtype {v.dtype} unsupported")
        dtype, raw = str(v.dtype), np.asarray(v).tobytes()
    elif isinstance(v, (bool, int)):
        if isinstance(v, int) and not -(1 << 63) <= v < (1 << 63):
            raise ValueError(
                f"{name}: python int outside the signed 64-bit frame "
                "range; refusing instead of coercing")
        dtype = "bool" if isinstance(v, bool) else "int64"
        raw = np.asarray(v, dtype=dtype).tobytes()
    elif isinstance(v, float):
        dtype, raw = "float64", np.asarray(v, dtype=np.float64).tobytes()
    else:
        raise MetricIdentityError(
            f"{name}: scalar type {type(v)!r} is not encodable as a "
            "metric identity; refusing instead of coercing")
    pairs.append((f"{name}.dtype", dtype))
    pairs.append((f"{name}.bits", raw))


# --------------------------------------------------------------------------
# Fold/helper code closure
# --------------------------------------------------------------------------
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_closure():
    """The compiled fold's code + math-environment closure.

    Conservative by construction: the WHOLE fold module's byte digest
    (any unrelated edit invalidates identities — fewer hits, never
    wrong hits), the fold's compilation flags, the compiler/runtime
    versions and the logical reduction plan.  Numba's own cache keys
    machine ISA; this closure binds the software identity that selects
    the math plan (fastmath, parallelism off = serial semantics).
    """
    from . import AccessibilityWElevation as _awe
    import llvmlite
    return {
        "fold_module_sha256": _sha256_file(_awe.__file__),
        "fold_qualname": "reach_gravity_knn_access",
        "fold_flags": {
            "parallel": bool(_awe.NUMBA_PARALLEL),
            "cache": bool(_awe.NUMBA_CACHE),
            "nogil": bool(_awe.NUMBA_NOGIL),
            "fastmath": bool(_awe.NUMBA_FASTMATH),
        },
        "numba_version": nb.__version__,
        "llvmlite_version": llvmlite.__version__,
        "numpy_version": np.__version__,
        "logical_reduction_plan": "canonical",
    }


# --------------------------------------------------------------------------
# The binding and the identity
# --------------------------------------------------------------------------
def metric_binding_fields(trace, d_weights, gravity_beta, gravity_plateau,
                          gravity_logistic_midpoint, gravity_growth_rate,
                          knn_decay, knn_weights, cutoff,
                          profile="una_legacy"):
    """The ordered schema fields of one replay-job binding.

    Mirrors ``replay_trace``'s parameter list EXACTLY (the real
    argument bindings the compiled fold consumes).  ``trace`` supplies
    the content+schema ID and the output dtype/cast fields.  Every
    value enters typed and uncoerced; anything lossy raises
    (``MetricIdentityError``/``ValueError``) and the caller must treat
    the job as carrying NO authoritative metric identity.
    """
    from ._trace_access_scratch import TRACE_SCHEMA_VERSION
    if getattr(trace, "schema_version", None) != TRACE_SCHEMA_VERSION:
        raise ValueError("metric identity binds complete C11 traces "
                         f"only (schema {trace.schema_version!r})")
    pairs = [
        ("metric_id_schema", METRIC_ID_SCHEMA_VERSION),
        ("closure_version", METRIC_CLOSURE_VERSION),
        ("trace_id", str(trace.trace_id)),
        ("trace_schema_version", str(trace.schema_version)),
        ("output_cast_reach_dtype", str(trace.reach_dtype)),
        ("output_cast_metric_dtype", str(trace.metric_dtype)),
        ("knn_decay_branch", str(knn_decay)),
        ("profile", str(profile)),
    ]
    typed_array_frame(pairs, "d_weights", d_weights)
    typed_array_frame(pairs, "knn_weights", knn_weights)
    typed_scalar_frame(pairs, "cutoff", cutoff)
    typed_scalar_frame(pairs, "gravity_beta", gravity_beta)
    typed_scalar_frame(pairs, "gravity_plateau", gravity_plateau)
    typed_scalar_frame(pairs, "gravity_logistic_midpoint",
                       gravity_logistic_midpoint)
    typed_scalar_frame(pairs, "gravity_growth_rate", gravity_growth_rate)
    closure = fold_closure()
    pairs.append(("fold_module_sha256", closure["fold_module_sha256"]))
    pairs.append(("fold_qualname", closure["fold_qualname"]))
    pairs.append(("fold_flags_parallel", closure["fold_flags"]["parallel"]))
    pairs.append(("fold_flags_cache", closure["fold_flags"]["cache"]))
    pairs.append(("fold_flags_nogil", closure["fold_flags"]["nogil"]))
    pairs.append(("fold_flags_fastmath", closure["fold_flags"]["fastmath"]))
    pairs.append(("numba_version", closure["numba_version"]))
    pairs.append(("llvmlite_version", closure["llvmlite_version"]))
    pairs.append(("numpy_version", closure["numpy_version"]))
    pairs.append(("logical_reduction_plan",
                  closure["logical_reduction_plan"]))
    return pairs


def compute_metric_id(trace, d_weights, gravity_beta, gravity_plateau,
                      gravity_logistic_midpoint, gravity_growth_rate,
                      knn_decay, knn_weights, cutoff,
                      profile="una_legacy"):
    """The authoritative typed MetricID of one replay-job binding.

    A sha256 over the canonical schema-ordered frame of
    :func:`metric_binding_fields`.  Refusal-based: any input the
    encoder cannot represent losslessly raises and no ID is minted.
    """
    return _identity(metric_binding_fields(
        trace, d_weights, gravity_beta, gravity_plateau,
        gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
        knn_weights, cutoff, profile=profile))
