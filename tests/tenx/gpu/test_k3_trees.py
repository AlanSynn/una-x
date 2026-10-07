"""K3 association trees: what is locally pinnable, and what is not.

CONTEXT-DEPENDENCE FINDING (measured 2026-10-07 during G10, recorded in
gpu_implementation.json): the sum-association class numba 0.67.0 emits for
the fold's reductions is a property of the COMPILED ENGINE ARTIFACT, not of
the source lines.  A bare ``w.sum()`` njit probe AND a verbatim local
transcription of the fold body BOTH classify differently on the
big-magnitude cases than the committed engine measurements in
n00/probe_order.txt did (e.g. big_at_1: this host's local probes hit
seq/pairwise/vec8_seq while the engine fold hit
vec4_tree/vec8_tree/vec4_seq/vec16_tree).  Therefore NO local probe can
stand in for the engine when certifying a reduction tree: the per-cell
gate (--emit-host-battery + the device half at G11) classifies against the
real engine artifact, and a cell's validity is bound to that build.

These tests pin only what is context-stable here: tree definitions are
mutually distinguishable, all trees agree on well-conditioned inputs
(matching the committed small-magnitude finding), and the big-magnitude
cases genuinely split the tree set (the gate's reason to exist)."""

import numpy as np
import pytest

nb = pytest.importorskip("numba")
from gpu.kernels.host_twins import (REDUCTION_TREES,  # noqa: E402
                                    pairwise_sum, reduce_with, seq_sum,
                                    vec_sum)


@nb.njit(cache=False)
def _numba_sum(a):
    return a.sum()


_ALL_HITS = tuple(REDUCTION_TREES)


def _bits(x) -> bytes:
    return np.float64(x).tobytes()


def _tree_hits(w):
    want = _bits(_numba_sum(w))
    return tuple(t for t in REDUCTION_TREES if _bits(reduce_with(t, w)) == want)


def test_all_trees_agree_on_well_conditioned_inputs():
    # Committed small-magnitude finding (tiny_then_big / mixed: all eight
    # candidates match) — reproduced by the local probe on this host.
    TWO30 = np.float64(2.0) ** 30
    cases = {
        "tiny_then_big": np.array([2.0 ** -30] * 8 + [np.float64(2.0) ** 53]),
        "mixed": np.array([1.0, TWO30, 1.0, 1.0, TWO30, 1.0, 1.0, 1.0]),
    }
    for name, w in cases.items():
        hits = _tree_hits(w)
        assert set(hits) == set(_ALL_HITS), f"{name}: hits {hits}"


def test_big_magnitude_cases_split_the_tree_set():
    # The gate's reason to exist: on catastrophic-cancellation-adjacent
    # magnitudes the associations genuinely disagree.  (Which ones hit is
    # engine-artifact-specific — see module docstring — so only the SPLIT
    # is asserted, not a specific hit set.)
    BIG = np.float64(2.0) ** 53
    for name, w in {
        "big_at_1": np.array([1.0, BIG] + [1.0] * 6),
        "big_at_7": np.array([1.0] * 7 + [BIG]),
        "big_at_0": np.array([BIG] + [1.0] * 7),
    }.items():
        values = {t: _bits(reduce_with(t, w)) for t in REDUCTION_TREES}
        distinct = set(values.values())
        assert len(distinct) >= 2, (
            f"{name}: trees failed to split (all identical bits)")


def test_trees_are_distinguishable_on_crafted_inputs():
    BIG = np.float64(2.0) ** 53
    w = np.array([1.0, BIG] + [1.0] * 6)
    values = {t: _bits(reduce_with(t, w)) for t in REDUCTION_TREES}
    assert values["seq"] != values["vec4_tree"]
    assert values["pairwise"] != values["vec8_tree"]


def test_pairwise_matches_seq_below_unroll_bound():
    rng = np.random.default_rng(5)
    for n in (0, 1, 5, 8):
        a = rng.random(n)
        assert _bits(pairwise_sum(a)) == _bits(seq_sum(a))


def test_vec_sum_lane_combine_variants():
    a = np.arange(1, 33, dtype=np.float64)
    assert _bits(vec_sum(a, 4, "tree")) == _bits(reduce_with("vec4_tree", a))
    assert _bits(vec_sum(a, 4, "seq")) == _bits(reduce_with("vec4_seq", a))


def test_reduce_with_refuses_unknown_tree():
    with pytest.raises(ValueError, match="unknown reduction tree"):
        reduce_with("sum_everything", np.ones(3))
