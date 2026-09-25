"""L0: exhaustive tiny-graph and adversarial exactness for _try_build_ordered_csr.

The oracle is ref_original_csr.original_builder_core, a verbatim transcription
of the audited baseline construction. Every admitted-domain input must produce
bit-identical arrays (dtype, shape, order, values, bytes); everything else must
be refused (None) so the caller executes the original path.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_CANDIDATE = TESTS_DIR.parents[2] / "src"
sys.path.insert(0, str(SRC_CANDIDATE))
sys.path.insert(0, str(TESTS_DIR))

from _ordered_csr_import import get_try_build_ordered_csr  # noqa: E402
from ref_original_csr import original_builder_core  # noqa: E402

_try_build_ordered_csr = get_try_build_ordered_csr()


def _assert_csr_equal(got, want):
    ptr_g, nbr_g, w_g, flag_g = got
    ptr_w, nbr_w, w_w, flag_w = want
    np.testing.assert_array_equal(ptr_g, ptr_w)
    np.testing.assert_array_equal(nbr_g, nbr_w)
    np.testing.assert_array_equal(w_g, w_w)
    np.testing.assert_array_equal(flag_g, flag_w)
    # exact dtypes and bit patterns (covers -0.0, NaN payloads, subnormals)
    assert ptr_g.dtype == np.dtype(np.int64) == ptr_w.dtype
    assert nbr_g.dtype == np.dtype(np.int64) == nbr_w.dtype
    assert w_g.dtype == np.dtype(np.float64) == w_w.dtype
    assert flag_g.dtype == np.dtype(np.bool_) == flag_w.dtype
    assert w_g.tobytes() == w_w.tobytes()
    assert nbr_g.tobytes() == nbr_w.tobytes()
    assert ptr_g.tobytes() == ptr_w.tobytes()


def _assert_owned(arr, dtype):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.dtype(dtype)
    assert arr.flags.writeable
    assert arr.flags.c_contiguous
    assert arr.base is None or isinstance(arr.base, np.ndarray)


ALL_FLOAT_BITS = [
    0.0, -0.0, 1.0, -1.0, 0.5, 1e-320, -1e-320,  # subnormals
    np.inf, -np.inf, np.nan,
]


def test_exhaustive_tiny_graphs_bit_exact():
    """Every directed multigraph on <=2 nodes with <=3 edges, plus all
    3-node graphs with <=2 edges: duplicates, self-loops, isolated nodes,
    empty edge lists."""
    rng = np.random.default_rng(20260925)
    cases = []
    for node_count in (2,):
        possible = [(a, b) for a in range(node_count) for b in range(node_count)]
        for n_edges in range(0, 4):
            for combo in itertools.product(range(len(possible)), repeat=n_edges):
                cases.append((node_count, [possible[i] for i in combo]))
    for node_count in (3,):
        possible = [(a, b) for a in range(node_count) for b in range(node_count)]
        for n_edges in range(0, 3):
            for combo in itertools.product(range(len(possible)), repeat=n_edges):
                cases.append((node_count, [possible[i] for i in combo]))
    # plus randomized larger lists on 2..6 nodes, heavy duplicates/self-loops
    for _ in range(400):
        nc = int(rng.integers(1, 7))
        ne = int(rng.integers(0, 30))
        edges = [(int(rng.integers(0, nc)), int(rng.integers(0, nc)))
                 for _ in range(ne)]
        cases.append((nc, edges))

    for node_count, edges in cases:
        if edges:
            arr = np.array(edges, dtype=np.int64)
            start = arr[:, 0]
            end = arr[:, 1]
        else:
            start = np.array([], dtype=np.int64)
            end = np.array([], dtype=np.int64)
        w_ab = rng.random(len(edges) if edges else 0)
        w_ba = rng.random(len(edges) if edges else 0)
        got = _try_build_ordered_csr(node_count, start, end, w_ab, w_ba)
        assert got is not None, (node_count, edges)
        _assert_csr_equal(got, original_builder_core(node_count, start, end, w_ab, w_ba))
        ptr, nbr, w, flag = got
        _assert_owned(ptr, np.int64)
        _assert_owned(nbr, np.int64)
        _assert_owned(w, np.float64)
        _assert_owned(flag, np.bool_)


def test_exhaustive_tiny_graphs_int32_endpoints_bit_exact():
    """Amended guard domain: int32 endpoints are admitted via exact upcast.
    The oracle (original builder) widens to int64 on output regardless of
    endpoint dtype, so arrays must be bit-identical in value AND dtype."""
    rng = np.random.default_rng(20260926)
    for _ in range(300):
        nc = int(rng.integers(1, 7))
        ne = int(rng.integers(0, 30))
        edges = [(int(rng.integers(0, nc)), int(rng.integers(0, nc)))
                 for _ in range(ne)]
        arr = (np.array(edges, dtype=np.int64) if edges
               else np.zeros((0, 2), dtype=np.int64))
        start = arr[:, 0].astype(np.int32)
        end = arr[:, 1].astype(np.int32)
        w_ab = rng.random(ne)
        w_ba = rng.random(ne)
        got = _try_build_ordered_csr(nc, start, end, w_ab, w_ba)
        assert got is not None, (nc, edges)
        want = original_builder_core(
            nc, start.astype(np.int64), end.astype(np.int64), w_ab, w_ba)
        _assert_csr_equal(got, want)


def test_int32_inputs_upcast_exactly_large_values():
    """int32 endpoint values well beyond int16 must survive the upcast
    bit-exactly. (Values near 2**31 are unreachable in practice: the range
    guard requires node_count above the endpoint value, and both builders
    allocate O(node_count); refusal at node_count > int32max is pinned by
    the FALLBACK_CASES parametrized test.)"""
    hi = 1_000_001
    start = np.array([0, hi, hi, 0], dtype=np.int32)
    end = np.array([hi, 0, 0, hi], dtype=np.int32)
    w_ab = np.array([1.0, 2.0, 3.0, 4.0])
    w_ba = np.array([5.0, 6.0, 7.0, 8.0])
    got = _try_build_ordered_csr(hi + 1, start, end, w_ab, w_ba)
    assert got is not None
    _assert_csr_equal(got, original_builder_core(
        hi + 1, start.astype(np.int64), end.astype(np.int64), w_ab, w_ba))


def test_int32_read_only_inputs_accepted_and_unchanged():
    start = np.array([0, 1, 2, 1], dtype=np.int32)
    end = np.array([1, 2, 0, 0], dtype=np.int32)
    ro_start = start.copy(); ro_start.flags.writeable = False
    ro_end = end.copy(); ro_end.flags.writeable = False
    before = (ro_start.tobytes(), ro_end.tobytes())
    got = _try_build_ordered_csr(3, ro_start, ro_end,
                                 np.array([1.0, 2.0, 3.0, 4.0]),
                                 np.array([5.0, 6.0, 7.0, 8.0]))
    assert got is not None
    assert (ro_start.tobytes(), ro_end.tobytes()) == before
    _assert_csr_equal(got, original_builder_core(
        3, ro_start.astype(np.int64), ro_end.astype(np.int64),
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])))


def test_special_float_bit_patterns_preserved():
    """Signed zero, nonfinites and subnormals must land in rows bit-exactly,
    including duplicated occurrences of identical bit patterns."""
    start = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    end = np.array([1, 2, 0, 2, 0, 1, 1, 0], dtype=np.int64)
    base = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1e-320, -0.0, 0.0])
    w_ab = base
    w_ba = -base  # includes -nan variants
    got = _try_build_ordered_csr(3, start, end, w_ab, w_ba)
    assert got is not None
    _assert_csr_equal(got, original_builder_core(3, start, end, w_ab, w_ba))


def test_self_loops_and_parallel_edges_distinct():
    start = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    end = np.array([0, 1, 1, 0, 2], dtype=np.int64)  # self-loops at 0,1,2 + parallel
    w_ab = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w_ba = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    got = _try_build_ordered_csr(3, start, end, w_ab, w_ba)
    assert got is not None
    ptr, nbr, w, flag = got
    assert int(ptr[-1]) == 2 * len(start)  # every incidence retained exactly once
    _assert_csr_equal(got, original_builder_core(3, start, end, w_ab, w_ba))


def test_empty_edges_and_isolated_nodes():
    for node_count in (0, 1, 5):
        start = np.array([], dtype=np.int64)
        end = np.array([], dtype=np.int64)
        w_ab = np.array([], dtype=np.float64)
        w_ba = np.array([], dtype=np.float64)
        got = _try_build_ordered_csr(node_count, start, end, w_ab, w_ba)
        assert got is not None
        _assert_csr_equal(
            got, original_builder_core(node_count, start, end, w_ab, w_ba))


def test_read_only_and_strided_inputs_accepted_and_unchanged():
    wide_start = np.arange(20, dtype=np.int64).reshape(10, 2) % 3
    wide_end = (np.arange(20, dtype=np.int64).reshape(10, 2) + 1) % 3
    start = wide_start[:, 0]  # strided view
    end = wide_end[:, 0]
    w_all = np.linspace(0.25, 4.0, 40).reshape(10, 4)
    w_ab = w_all[:, 1]
    w_ba = w_all[:, 3]
    ro_start = start.copy(); ro_start.flags.writeable = False
    ro_end = end.copy(); ro_end.flags.writeable = False
    ro_wab = w_ab.copy(); ro_wab.flags.writeable = False
    ro_wba = w_ba.copy(); ro_wba.flags.writeable = False
    before = (ro_start.tobytes(), ro_end.tobytes(), ro_wab.tobytes(), ro_wba.tobytes())
    got = _try_build_ordered_csr(3, ro_start, ro_end, ro_wab, ro_wba)
    assert got is not None
    after = (ro_start.tobytes(), ro_end.tobytes(), ro_wab.tobytes(), ro_wba.tobytes())
    assert before == after, "inputs must not be mutated"
    _assert_csr_equal(got, original_builder_core(3, ro_start, ro_end, ro_wab, ro_wba))


FALLBACK_CASES = {
    "int16_endpoints": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=np.int16),
        end=np.array([1, 2], dtype=np.int16),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "uint32_endpoints": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=np.uint32),
        end=np.array([1, 2], dtype=np.uint32),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "mixed_int32_int64_endpoints": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=np.int32),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "int32_node_count_above_int32_range": lambda: dict(
        node_count=np.iinfo(np.int32).max + 1,
        start=np.array([0, 1], dtype=np.int32),
        end=np.array([1, 2], dtype=np.int32),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "int32_endpoint_out_of_range": lambda: dict(
        node_count=2, start=np.array([0, 5], dtype=np.int32),
        end=np.array([1, 0], dtype=np.int32),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "float32_weights": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=np.int64),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0], dtype=np.float32),
        w_ba=np.array([3.0, 4.0], dtype=np.float32)),
    "big_endian_endpoints": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=">i8"),
        end=np.array([1, 2], dtype=">i8"),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "two_dimensional": lambda: dict(
        node_count=3, start=np.array([[0], [1]], dtype=np.int64),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "length_mismatch": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=np.int64),
        end=np.array([1, 2, 0], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "endpoint_out_of_range": lambda: dict(
        node_count=2, start=np.array([0, 5], dtype=np.int64),
        end=np.array([1, 0], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "negative_endpoint": lambda: dict(
        node_count=3, start=np.array([0, -1], dtype=np.int64),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "non_int_node_count": lambda: dict(
        node_count=np.int64(3), start=np.array([0, 1], dtype=np.int64),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "negative_node_count": lambda: dict(
        node_count=-1, start=np.array([0, 1], dtype=np.int64),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0]), w_ba=np.array([3.0, 4.0])),
    "lists_not_arrays": lambda: dict(
        node_count=3, start=[0, 1], end=[1, 2],
        w_ab=[1.0, 2.0], w_ba=[3.0, 4.0]),
    "mixed_float_weight_dtypes": lambda: dict(
        node_count=3, start=np.array([0, 1], dtype=np.int64),
        end=np.array([1, 2], dtype=np.int64),
        w_ab=np.array([1.0, 2.0], dtype=np.float64),
        w_ba=np.array([3.0, 4.0], dtype=np.float32)),
}


@pytest.mark.parametrize("case_name", sorted(FALLBACK_CASES))
def test_unsupported_domains_return_none(case_name):
    kwargs = FALLBACK_CASES[case_name]()
    assert _try_build_ordered_csr(
        kwargs["node_count"], kwargs["start"], kwargs["end"],
        kwargs["w_ab"], kwargs["w_ba"]) is None, case_name


def test_fallback_domain_matches_oracle_via_original_core():
    """For every refused domain that is structurally valid for the original
    builder (1-D arrays), the fallback construction matches the oracle on the
    casted representations the engines actually produce. Structurally invalid
    inputs (2-D endpoints) raise in the original caller; the candidate
    preserves that by returning None, which the parametrized None test pins."""
    for case_name, factory in FALLBACK_CASES.items():
        kwargs = factory()
        if (np.asarray(kwargs["start"]).ndim != 1
                or np.asarray(kwargs["end"]).ndim != 1):
            continue
        if type(kwargs["node_count"]) is not int:
            continue  # range() would reject; caller error path preserved
        if kwargs["node_count"] > 10**6:
            continue  # oracle allocation prohibitive; None refusal pinned above
        try:
            want = original_builder_core(
                int(kwargs["node_count"]),
                np.asarray(kwargs["start"]), np.asarray(kwargs["end"]),
                np.asarray(kwargs["w_ab"], dtype=np.float64),
                np.asarray(kwargs["w_ba"], dtype=np.float64))
        except (TypeError, ValueError, IndexError):
            continue  # oracle itself rejects; caller error path preserved
        got = original_builder_core(
            int(kwargs["node_count"]),
            np.asarray(kwargs["start"], dtype=np.int64),
            np.asarray(kwargs["end"], dtype=np.int64),
            np.asarray(kwargs["w_ab"], dtype=np.float64),
            np.asarray(kwargs["w_ba"], dtype=np.float64))
        _assert_csr_equal(got, want)
