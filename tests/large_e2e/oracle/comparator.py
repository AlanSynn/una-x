"""Bit-exact comparators for the H03 oracles.

Equality is dtype + shape + bytes (CONTRACT.md: "Exact equality is
dtype+shape+bytes for deterministic arrays"). A mismatch raises
OracleMismatch naming the label and the FIRST differing index/value —
diagnose-first-divergence, never allclose, never a tolerance fallback.
"""
from __future__ import annotations

import numpy as np


class OracleMismatch(AssertionError):
    """Raised on any oracle comparison failure (bit, dtype or shape)."""


def _bits(a) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def assert_array_bytes_equal(actual, expected, label):
    """dtype+shape+bytes equality with a first-divergence report."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.dtype != expected.dtype:
        raise OracleMismatch(
            f"{label}: dtype differs actual={actual.dtype} expected={expected.dtype}"
        )
    if actual.shape != expected.shape:
        raise OracleMismatch(
            f"{label}: shape differs actual={actual.shape} expected={expected.shape}"
        )
    a_bits, e_bits = _bits(actual), _bits(expected)
    if a_bits == e_bits:
        return
    flat_a = np.ascontiguousarray(actual).ravel()
    flat_e = np.ascontiguousarray(expected).ravel()
    # Object arrays have no bit view; fall back to elementwise !=.
    if flat_a.dtype == object:
        for i, (x, y) in enumerate(zip(flat_a, flat_e)):
            if x != y:
                raise OracleMismatch(f"{label}: first differing index {i}: {x!r} != {y!r}")
        raise OracleMismatch(f"{label}: bytes differ without a differing element")
    diff = np.flatnonzero(flat_a.view(np.uint8) != flat_e.view(np.uint8))
    if diff.size == 0:
        raise OracleMismatch(f"{label}: bytes differ without a differing byte (pad)")
    first = int(diff[0])
    item = flat_a.dtype.itemsize
    idx = first // item
    off = first % item
    raise OracleMismatch(
        f"{label}: bytes differ; first differing byte {first} "
        f"(element {idx}, byte-offset {off}, itemsize {item}); "
        f"actual element={flat_a[idx]!r} ({_hex(flat_a[idx])}) "
        f"expected element={flat_e[idx]!r} ({_hex(flat_e[idx])}); "
        f"{int(diff.size)} differing bytes total"
    )


def _hex(scalar):
    try:
        b = np.ascontiguousarray(np.atleast_1d(scalar)).tobytes()
    except Exception:  # pragma: no cover - non-arithmetic dtype
        return "?"
    return b.hex()


def assert_scalar_bits_equal(actual, expected, label):
    """Exact scalar equality on the float64/int64 bit patterns."""
    a = float(actual)
    e = float(expected)
    a_bits = np.array(a, dtype=np.float64).tobytes()
    e_bits = np.array(e, dtype=np.float64).tobytes()
    if a_bits != e_bits:
        raise OracleMismatch(
            f"{label}: scalar bits differ actual={a!r} ({a_bits.hex()}) "
            f"expected={e!r} ({e_bits.hex()})"
        )


def assert_journal_equal(actual, expected, label, path=""):
    """Deep byte-exact comparison of trace journals.

    Journals are dicts / lists / numpy arrays / scalars / strings / None.
    dict keys must match exactly (order not required); list lengths must
    match; arrays compare dtype+shape+bytes; floats compare by bits; int
    and str compare exactly. The first difference raises with its full
    key/index path.
    """
    here = label if not path else f"{label}.{path}"
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        assert_array_bytes_equal(actual, expected, here)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise OracleMismatch(f"{here}: expected dict, got {type(actual).__name__}")
        if set(actual.keys()) != set(expected.keys()):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise OracleMismatch(f"{here}: dict keys differ missing={missing} extra={extra}")
        for key in sorted(expected.keys(), key=str):
            assert_journal_equal(actual[key], expected[key], label, f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise OracleMismatch(
                f"{here}: list length/type differs actual={len(actual) if isinstance(actual, (list, tuple)) else type(actual).__name__} "
                f"expected={len(expected)}"
            )
        for i, (x, y) in enumerate(zip(actual, expected)):
            assert_journal_equal(x, y, label, f"{path}[{i}]")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if bool(actual) != bool(expected):
            raise OracleMismatch(f"{here}: bool differs actual={actual!r} expected={expected!r}")
        return
    if isinstance(expected, (int, np.integer)) and isinstance(actual, (int, np.integer)):
        if int(actual) != int(expected):
            raise OracleMismatch(f"{here}: int differs actual={actual!r} expected={expected!r}")
        return
    if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
        assert_scalar_bits_equal(actual, expected, here)
        return
    if actual != expected:
        raise OracleMismatch(f"{here}: value differs actual={actual!r} expected={expected!r}")


def require_raises(callable_obj, exc_type, label):
    """Run callable_obj() and require exactly exc_type; return the instance."""
    try:
        callable_obj()
    except exc_type as exc:  # noqa: BLE001 - narrow type enforced by caller
        return exc
    except Exception as exc:  # noqa: BLE001 - report the wrong-type case
        raise OracleMismatch(
            f"{label}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise OracleMismatch(f"{label}: expected {exc_type.__name__}, nothing raised")


def assert_disjoint_storage(a, b, label):
    """No shared memory between two arrays (output ownership/aliasing)."""
    if np.shares_memory(np.asarray(a), np.asarray(b)):
        raise OracleMismatch(f"{label}: arrays alias the same memory")


def _ulp_distance(a, b):
    """Per-element distance in float64 ULPs via the ordered-int trick.

    NaN != NaN anywhere; inf only equals +inf; -0.0 and +0.0 are 0 ULPs
    apart (their integer codes differ but both map to distance 0)."""
    ai = np.asarray(a, dtype=np.float64).view(np.int64).astype(np.int64)
    bi = np.asarray(b, dtype=np.float64).view(np.int64).astype(np.int64)
    # map sign-magnitude floats to a monotonically ordered integer scale
    order_a = np.where(ai < 0, np.int64(-9223372036854775808) - ai, ai)
    order_b = np.where(bi < 0, np.int64(-9223372036854775808) - bi, bi)
    return np.abs(order_a - order_b)


def assert_array_within_ulp(actual, expected, max_ulp, label):
    """dtype+shape equality, every finite element within max_ulp ULPs.

    Used ONLY for pure-Python trace vs compiled-kernel comparisons of
    fastmath float sums: numba's fastmath reassociates multi-term adds,
    which a strict IEEE-left-to-right Python replica cannot reproduce
    bit-for-bit on every input (observed: 1 ULP on the flow fixture).
    Candidate-vs-baseline comparisons NEVER use this — those are
    assert_array_bytes_equal (bit-exact) without exception."""
    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.dtype != b.dtype or a.shape != b.shape:
        raise OracleMismatch(
            f"{label}: dtype/shape differ: {a.dtype}/{a.shape} vs "
            f"{b.dtype}/{b.shape}")
    if a.dtype != np.float64:
        assert_array_bytes_equal(a, b, label)
        return
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        raise OracleMismatch(f"{label}: NaN placement differs")
    if not np.array_equal(np.isinf(a), np.isinf(b)):
        raise OracleMismatch(f"{label}: inf placement differs")
    dist = _ulp_distance(a[~nan_a], b[~nan_b])
    bad = np.nonzero(dist > max_ulp)[0]
    if bad.size:
        k = int(bad[0])
        raise OracleMismatch(
            f"{label}: element {k} differs by {int(dist[k])} ULPs "
            f"(>{max_ulp}); actual={a.flat[k]!r} expected={b.flat[k]!r}")


def assert_scalar_within_ulp(actual, expected, max_ulp, label):
    assert_array_within_ulp(
        np.asarray(np.float64(actual)), np.asarray(np.float64(expected)),
        max_ulp, label)
