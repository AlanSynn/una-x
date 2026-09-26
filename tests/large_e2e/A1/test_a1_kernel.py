"""A1 kernel-level battery (proof.md section 9): T1, T4-T10 and the
guard-decision half of T13.

Every candidate comparison is against the COMPILED B0 kernel with
comparator.assert_array_bytes_equal (dtype+shape+bytes; no tolerance).
The candidate side runs the private scratch kernel with the exact
buffers the driver branch allocates.
"""
from __future__ import annotations

import numpy as np
import pytest

import fixtures
import fixtures_a1
from comparator import OracleMismatch, assert_array_bytes_equal

import mutant_a1
from harness_a1 import b0ns, cns, run_kernel, run_scratch_kernel

CASES = fixtures.kernel_scope_cases()


def _kc(name):
    return [k for k in CASES if k.name == name][0]


def _kc_arrays(kc):
    return dict(
        adjacency_pointer=kc.adjacency_pointer,
        adjacency_vector=kc.adjacency_vector,
        adjacency_vector_weights=kc.adjacency_vector_weights,
        adjacynct_vector_network_node=kc.adjacynct_vector_network_node,
        o_terminal_idxs=kc.o_terminal_idxs,
        o_terminal_weights=kc.o_terminal_weights,
        cutoff=kc.cutoff,
        d_count=kc.d_count,
    )


def _engine_kernel_arrays(name):
    case = [c for c in fixtures.engine_cases() if c.name == name][0]
    arrays = dict(case.kernel_arrays())
    arrays.update(case.terminal_arrays())
    # single-origin kernel rows (the kernel takes one origin per call)
    arrays["o_terminal_idxs"] = arrays["o_terminal_idxs"][:1]
    arrays["o_terminal_weights"] = arrays["o_terminal_weights"][:1]
    arrays["cutoff"] = case.cutoff
    arrays["d_count"] = case.terminal_arrays()["d_count"]
    return arrays


def _assert_pair(arrays, tag):
    out_b0 = run_kernel(b0ns(), arrays)
    out_c = run_scratch_kernel(cns(), arrays)
    assert_array_bytes_equal(out_c[0], out_b0[0], f"{tag}/labels")
    assert_array_bytes_equal(out_c[1], out_b0[1], f"{tag}/pred")
    return out_b0[0], out_c[0]


# ---------------------------------------------------------------- T1
@pytest.mark.parametrize("kc", CASES, ids=[k.name for k in CASES])
def test_t1_kernel_scope_cases(kc):
    _assert_pair(_kc_arrays(kc), f"t1/{kc.name}")


def test_t1_empty_edges_kernel_arrays():
    _assert_pair(_engine_kernel_arrays("l0_empty_edges"), "t1/l0_empty_edges")


# ------------------------------------------------------- T4 float domain
@pytest.mark.parametrize(
    "name", ["flt_nextafter_cutoff", "flt_signed_zero", "flt_subnormals",
             "flt_near_overflow_finite", "flt_sentinel_two_pow_53",
             "flt_maxfloat_cutoff"])
def test_t4_float_domain_engine_cases(name):
    _assert_pair(_engine_kernel_arrays(name), f"t4/{name}")


def test_t4_true_overflow_is_admitted_and_byte_equal():
    """True +inf overflow inside a scanned row: finite inputs admitted;
    the incidence add overflows identically in both arms."""
    arrays = fixtures_a1.true_overflow()
    labels_b0, labels_c = _assert_pair(arrays, "t4/true_overflow")
    # the overflow add happened: node 1's row was scanned with the
    # popped 1.5e308 label, where edge weight 1.5e308 + 1.5e308 -> +inf
    w = arrays["adjacency_vector_weights"]
    assert np.isinf(w[1] + 1.5e308) and w[1] == 1.5e308
    assert labels_b0[1] == 1.0 and labels_b0[2] == 0.0


def test_t4_inf_incidence_weight_refuses_and_byte_equal():
    arrays = _engine_kernel_arrays("flt_inf_incidence_weight")
    # guard decision: refuse (condition 10)
    dec = _probe()(arrays["adjacency_pointer"], arrays["adjacency_vector"],
                   arrays["adjacency_vector_weights"],
                   arrays["adjacynct_vector_network_node"],
                   arrays["o_terminal_idxs"], arrays["o_terminal_weights"],
                   arrays["cutoff"])
    assert dec == (False, 0)
    _assert_pair(arrays, "t4/inf_incidence_refusal")


# ------------------------------------------------- T5 duplicate negative
def test_t5_downstream_duplicate_byte_equal():
    arrays = fixtures_a1.downstream_dup()
    labels_b0, labels_c = _assert_pair(arrays, "t5/downstream_dup")
    # snapshot semantics pinned: BOTH duplicates of node 2 are judged
    # against the pre-row label (11), so the LAST eligible value
    # (1+5=6) survives; the immediate-update mutant would keep 1+3=4
    assert labels_b0[2] == 6.0, labels_b0


def test_t5_negative_immediate_update_mutant_caught():
    """The candidate-shaped immediate-update mutant MUST move bytes of
    the downstream duplicate fixture and be caught by the comparator
    (first divergence recorded in the failure message)."""
    arrays = fixtures_a1.downstream_dup()
    compiled = np.asarray(run_kernel(b0ns(), arrays)[0])
    mutated, _ = mutant_a1.mutant_scope_immediate_update_scratch(
        arrays["o_terminal_idxs"][0], arrays["o_terminal_weights"][0],
        arrays["adjacency_pointer"], arrays["adjacency_vector"],
        arrays["adjacency_vector_weights"],
        arrays["adjacynct_vector_network_node"],
        arrays["cutoff"], arrays["d_count"],
        np.empty(2, dtype=np.int64), np.empty(2, dtype=np.float64),
    )
    mutated = np.asarray(mutated)
    assert compiled[2] == 6.0 and mutated[2] == 4.0, (compiled, mutated)
    with pytest.raises(OracleMismatch) as exc:
        assert_array_bytes_equal(mutated, compiled, "t5/negative_mutant")
    # the report names the first divergence (element 2)
    assert "first differing" in str(exc.value)
    assert "element 2" in str(exc.value)


# --------------------------------------------------------- T6/T7/T8 pins
def test_t6_stale_entry():
    _assert_pair(_kc_arrays(_kc("k_stale_queue")), "t6/k_stale_queue")


@pytest.mark.parametrize("name", ["k_degree_one_no_push", "k_flags_false_row"])
def test_t7_degree_one_and_flag_false(name):
    _assert_pair(_kc_arrays(_kc(name)), f"t7/{name}")


@pytest.mark.parametrize(
    "builder", [lambda: _kc_arrays(_kc("k_same_terminal_overwrite")),
                lambda: _kc_arrays(_kc("k_seed_equals_sentinel")),
                lambda: _engine_kernel_arrays("l0_same_terminal_unequal_costs")],
    ids=["k_same_terminal_overwrite", "k_seed_equals_sentinel",
         "l0_same_terminal_unequal_costs"])
def test_t8_terminal_quirks(builder):
    _assert_pair(builder(), "t8")


# ------------------------------------------- T9 empty edges / T10 widest
def test_t9_empty_edges_zero_scratch():
    arrays = _engine_kernel_arrays("l0_empty_edges")
    assert arrays["adjacency_vector"].shape[0] == 0
    max_degree = 0  # V=2 with no edges: PR-N4 zero-length scratch
    out_c = cns()._a1_scope_search(
        arrays["o_terminal_idxs"][0], arrays["o_terminal_weights"][0],
        arrays["adjacency_pointer"], arrays["adjacency_vector"],
        arrays["adjacency_vector_weights"],
        arrays["adjacynct_vector_network_node"],
        arrays["cutoff"], arrays["d_count"],
        np.empty(max_degree, dtype=np.int64),
        np.empty(max_degree, dtype=np.float64))
    out_b0 = run_kernel(b0ns(), arrays)
    assert_array_bytes_equal(out_c[0], out_b0[0], "t9/labels")
    assert_array_bytes_equal(out_c[1], out_b0[1], "t9/pred")


def test_t10_max_degree_row():
    arrays = fixtures_a1.max_degree_row()
    labels_b0, labels_c = _assert_pair(arrays, "t10/max_degree_row")
    # snapshot duplicate overwrites inside the widest row stick (last
    # eligible value wins: node2 -> 3.0, node3 -> 3.5); the degree-1
    # destinations are never pushed, so no later repair happens. The
    # trailing slot is the untouched destination sentinel (1+cutoff).
    assert list(labels_b0) == [0.0, 1.0, 3.0, 3.5, 2.0, 2.0, 11.0], labels_b0


# --------------------------------- guard decision probe (T13, part one)
_probe_fn = None


def _probe():
    """njit probe that resolves the candidate overload guard."""
    global _probe_fn
    if _probe_fn is None:
        import numba as nb
        admits = cns()._a1_scope_admits

        @nb.njit(cache=False)
        def _f(adjacency_pointer, adjacency_vector, adjacency_vector_weights,
               adjacynct_vector_network_node, o_terminal_idxs,
               o_terminal_weights, cutoff):
            return admits(adjacency_pointer, adjacency_vector,
                          adjacency_vector_weights,
                          adjacynct_vector_network_node, o_terminal_idxs,
                          o_terminal_weights, cutoff)

        _probe_fn = _f
    return _probe_fn


def _mut(arrays, **over):
    out = dict(arrays)
    out.update(over)
    return out


def test_guard_admits_valid_base():
    arrays = fixtures_a1.admitted_base()
    dec = _probe()(arrays["adjacency_pointer"], arrays["adjacency_vector"],
                   arrays["adjacency_vector_weights"],
                   arrays["adjacynct_vector_network_node"],
                   arrays["o_terminal_idxs"], arrays["o_terminal_weights"],
                   arrays["cutoff"])
    assert dec[0] is True or dec[0] == True  # noqa: E712 (numba bool)
    assert dec[1] == int(np.diff(arrays["adjacency_pointer"]).max())


@pytest.mark.parametrize(
    "mutate, why", [
        (lambda a: a.update(adjacency_pointer=a["adjacency_pointer"].astype(np.int32)), "ptr int32"),
        (lambda a: a.update(adjacency_vector=a["adjacency_vector"].astype(np.int32)), "nbr int32"),
        (lambda a: a.update(adjacency_vector_weights=a["adjacency_vector_weights"].astype(np.float32)), "weights float32"),
        (lambda a: a.update(adjacynct_vector_network_node=a["adjacynct_vector_network_node"].astype(np.uint8)), "flags uint8"),
        (lambda a: a.update(o_terminal_idxs=a["o_terminal_idxs"].astype(np.int32)), "oti int32"),
        (lambda a: a.update(o_terminal_weights=a["o_terminal_weights"].astype(np.float32)), "otw float32"),
        (lambda a: a.update(cutoff=np.float32(10.0)), "cutoff float32"),
        (lambda a: a.update(cutoff=float("nan")), "cutoff nan"),
        (lambda a: a.update(cutoff=float("inf")), "cutoff inf"),
        (lambda a: a.update(cutoff=-1.0), "cutoff negative"),
        (lambda a: a["adjacency_pointer"].__setitem__(1, a["adjacency_pointer"][2] + 1) or a, "pointer not monotone"),
        (lambda a: a["adjacency_pointer"].__setitem__(0, 1) or a, "pointer[0] != 0"),
        (lambda a: a.update(adjacency_vector=a["adjacency_vector"][:-1]), "pointer[V] != len(vector)"),
        (lambda a: a["adjacency_vector"].__setitem__(0, a["adjacency_pointer"].shape[0]) or a, "neighbor OOB"),
        (lambda a: a["adjacency_vector_weights"].__setitem__(0, -0.5), "negative weight"),
        (lambda a: a["adjacency_vector_weights"].__setitem__(0, float("nan")), "nan weight"),
        (lambda a: a["adjacency_vector_weights"].__setitem__(0, -float("inf")), "-inf weight"),
        (lambda a: a["o_terminal_idxs"].__setitem__((0, 0), 10**9), "terminal OOB"),
        (lambda a: a.update(adjacency_pointer=a["adjacency_pointer"][:1], adjacency_vector=a["adjacency_vector"][:0], adjacency_vector_weights=a["adjacency_vector_weights"][:0], adjacynct_vector_network_node=a["adjacynct_vector_network_node"][:0]), "empty pointer"),
    ])
def test_guard_refusal_classes(mutate, why):
    arrays = fixtures_a1.admitted_base()
    mutate(arrays)
    dec = _probe()(arrays["adjacency_pointer"], arrays["adjacency_vector"],
                   arrays["adjacency_vector_weights"],
                   arrays["adjacynct_vector_network_node"],
                   arrays["o_terminal_idxs"], arrays["o_terminal_weights"],
                   arrays["cutoff"])
    assert dec == (False, 0), why


def test_guard_refuses_scratch_cap():
    arrays = fixtures_a1.scratch_cap_refusal()
    dec = _probe()(arrays["adjacency_pointer"], arrays["adjacency_vector"],
                   arrays["adjacency_vector_weights"],
                   arrays["adjacynct_vector_network_node"],
                   arrays["o_terminal_idxs"], arrays["o_terminal_weights"],
                   arrays["cutoff"])
    assert dec == (False, 0)


def test_guard_refuses_int_cutoff_int32_endpoints_shapes():
    """Integer cutoff admits (PR-N2); 1-D/3-D terminal shapes refuse at
    the typing gate."""
    arrays = fixtures_a1.admitted_base()
    dec = _probe()(arrays["adjacency_pointer"], arrays["adjacency_vector"],
                   arrays["adjacency_vector_weights"],
                   arrays["adjacynct_vector_network_node"],
                   arrays["o_terminal_idxs"], arrays["o_terminal_weights"],
                   12)
    assert dec[0] is True or dec[0] == True  # noqa: E712
    dec = _probe()(arrays["adjacency_pointer"], arrays["adjacency_vector"],
                   arrays["adjacency_vector_weights"],
                   arrays["adjacynct_vector_network_node"],
                   arrays["o_terminal_idxs"][0],  # 1-D otw
                   arrays["o_terminal_weights"][0],
                   arrays["cutoff"])
    assert dec == (False, 0)
