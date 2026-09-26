"""Negative-mutation proof: every deliberately broken replica MUST be
caught by the bit-exact comparator on at least one crafted construction
(VALIDATION.md L0: "Add negative tests that intentionally replace
phase one/two by immediate update and confirm the duplicate fixture
fails"). Each test asserts OracleMismatch is raised — a mutant that
slips through fails the test.

Proven inert (documented, NOT counted as catches): the F2
stale-scratch driver — see test_f2_stale_scratch_is_inert_on_corpus.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

import fixtures
import mutants
import stub_topology
import trace_access
import trace_flow
import trace_gradients
from comparator import OracleMismatch, assert_array_bytes_equal
from support import b0


def _kc(name):
    return [k for k in fixtures.kernel_scope_cases() if k.name == name][0]


def _engine_case(name):
    return [c for c in fixtures.engine_cases() if c.name == name][0]


# Downstream duplicate-incidence arrays: origin at node 0, node 1's row
# holds TWO incidences to degree-0 node 2 (cheap first, expensive
# second). No stale re-scan of that row is possible (node 2 is never
# pushed), so the staged-vs-immediate distinction survives to the final
# labels — unlike the k_dup_* origin-row fixtures, where the
# double-seeded start re-scan converges both semantics to the cheap
# value (their bytes are pinned by the golden corpus instead).
_DOWNSTREAM = dict(
    pointer=np.array([0, 1, 3, 3], dtype=np.int64),
    nbrs=np.array([1, 2, 2], dtype=np.int64),
    weights=np.array([1.0, 3.0, 5.0], dtype=np.float64),
    flags=np.ones(3, dtype=np.bool_),
    o_idx=np.array([0, 0], dtype=np.int64),
    o_w=np.array([0.0, 0.0], dtype=np.float64),
    cutoff=10.0, d_count=1,
)


def test_negative_a1_immediate_update_caught_on_downstream_duplicate():
    """A1: replacing the staged snapshot with immediate update keeps the
    cheap value (4.0) where B0's staged scan keeps the LAST eligible
    value (6.0). The comparator must catch it."""
    b = b0()
    d = _DOWNSTREAM
    compiled = np.asarray(b.compact_vector_node_view_scope(
        d["o_idx"], d["o_w"], d["pointer"], d["nbrs"], d["weights"],
        d["flags"], d["cutoff"], d["d_count"])[0])
    assert compiled[2] == 6.0
    mutated, _ = mutants.mutant_scope_immediate_update(
        d["pointer"], d["nbrs"], d["weights"], d["flags"],
        d["cutoff"], d["d_count"], d["o_idx"], d["o_w"])
    assert np.asarray(mutated)[2] == 4.0
    with pytest.raises(OracleMismatch):
        assert_array_bytes_equal(mutated, compiled,
                                 "negative_a1_immediate/downstream_dup")


def test_negative_terminal_assignment_order_caught():
    """A1/A3: end-before-start terminal assignment changes the label of
    a same-start-and-end terminal with unequal weights."""
    b = b0()
    kc = _kc("k_same_terminal_overwrite")
    compiled = np.asarray(b.compact_vector_node_view_scope(
        kc.o_terminal_idxs[0], kc.o_terminal_weights[0],
        kc.adjacency_pointer, kc.adjacency_vector,
        kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
        kc.cutoff, kc.d_count)[0])
    assert compiled[0] == 0.5  # start assigned first, end overwrites
    mutated, _ = mutants.mutant_scope_terminal_order(
        kc.adjacency_pointer, kc.adjacency_vector,
        kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
        kc.cutoff, kc.d_count, kc.o_terminal_idxs[0],
        kc.o_terminal_weights[0])
    with pytest.raises(OracleMismatch):
        assert_array_bytes_equal(mutated, compiled, "negative_terminal_order")


def test_negative_a2_python_min_semantics_caught_on_signed_zero():
    """A2: Python's first-arg min keeps +0.0 where numba's min yields
    -0.0 on the (+0.0, -0.0) tie — byte comparison catches it."""
    b = b0()
    case = _engine_case("flt_signed_zero")
    topo = stub_topology.StubTopology(case)
    ge = b.Accessibility(topo).graph_engine
    term = case.terminal_arrays()
    scope = b.compact_vector_node_view_scope(
        term["o_terminal_idxs"][0], term["o_terminal_weights"][0],
        ge.adjacency_pointer, ge.adjacency_vector,
        ge.adjacency_vector_weights, ge.adjacynct_vector_network_node,
        case.cutoff, term["d_count"])[0]
    compiled = b.adjust_destination_distances(
        scope, term["d_terminal_idxs"], term["d_terminal_weights"],
        term["d_count"])
    mutated = mutants.mutant_adjust_min_python_semantics(
        np.asarray(scope), term["d_terminal_idxs"],
        term["d_terminal_weights"], term["d_count"])
    with pytest.raises(OracleMismatch):
        assert_array_bytes_equal(mutated, compiled, "negative_a2_min")


def test_negative_a2_reversed_retained_order_caught():
    b = b0()
    case = _engine_case("l0_tiny_random_dup_loop")
    csr_geo = case.kernel_arrays()
    term = case.terminal_arrays()
    scope = b.compact_vector_node_view_scope(
        term["o_terminal_idxs"][0], term["o_terminal_weights"][0],
        csr_geo["adjacency_pointer"], csr_geo["adjacency_vector"],
        csr_geo["adjacency_vector_weights"],
        csr_geo["adjacynct_vector_network_node"],
        case.cutoff, term["d_count"])[0]
    adjust = b.adjust_destination_distances(
        scope, term["d_terminal_idxs"], term["d_terminal_weights"],
        term["d_count"])
    good_dist, good_w, _ = trace_access.trace_retained_sequences(
        np.asarray(adjust), term["d_weights"], case.cutoff)
    filt = np.where(np.asarray(adjust) <= case.cutoff)[0]
    assert_array_bytes_equal(good_dist, np.asarray(adjust)[filt],
                             "faithful retained order")
    bad_dist, bad_w, _ = mutants.mutant_retained_sequence_reversed(
        np.asarray(adjust), term["d_weights"], case.cutoff)
    with pytest.raises(OracleMismatch):
        assert_array_bytes_equal(bad_dist, np.asarray(adjust)[filt],
                                 "negative_a2_reversed")
    with pytest.raises(OracleMismatch):
        assert_array_bytes_equal(bad_w, term["d_weights"][filt],
                                 "negative_a2_reversed_weights")


def _flow_engine(b, threads):
    from support import flow_engine_case
    eng, settings, _ = flow_engine_case(b, None)
    eng.num_threads = threads
    eng.Centrality(settings)
    return eng, settings


def test_negative_f1_shifted_stripe_assignment_caught():
    """F1: shifting stripe membership changes which per-stripe partial
    array absorbs each origin (the F1 observable) even though the total
    origin set is unchanged."""
    b = b0()
    eng, settings = _flow_engine(b, 2)
    faithful = trace_flow.trace_origin_loop(b, eng, settings, 2)
    shifted = trace_flow.trace_origin_loop(b, eng, settings, 2, stripe_shift=1)
    assert shifted["partials"]["stripe_members"] != \
        faithful["partials"]["stripe_members"]
    diffed = any(
        not np.array_equal(np.asarray(faithful["partials"]["local_AB"][s]),
                           np.asarray(shifted["partials"]["local_AB"][s]))
        for s in range(2))
    assert diffed, "shifted stripe membership must move per-stripe partials"


def test_negative_f1_descending_final_sum_caught():
    """F1: folding partials K-1..0 instead of 0..K-1 is a different
    float computation. With five active origins all three K=3 stripes
    carry nonzero partials, and on this fixture the BA reassociation is
    byte-visible."""
    b = b0()
    eng, settings = _flow_engine(b, 3)
    compiled = dict(
        AB=np.asarray(eng.edge_flow_AB).copy(),
        BA=np.asarray(eng.edge_flow_BA).copy(),
        node=np.asarray(eng.node_flow).copy(),
    )
    trace = trace_flow.trace_origin_loop(b, eng, settings, 3)
    assert_array_bytes_equal(trace["final_AB"], compiled["AB"],
                             "faithful fold AB")
    assert_array_bytes_equal(trace["final_BA"], compiled["BA"],
                             "faithful fold BA")
    mut_AB, mut_BA, mut_node = mutants.mutant_final_sum_descending(
        {"partials": trace["partials"]}, 3)
    candidates = (("AB", mut_AB, compiled["AB"]),
                  ("BA", mut_BA, compiled["BA"]),
                  ("node", mut_node, compiled["node"]))
    differing = [(name, mut, ref) for name, mut, ref in candidates
                 if np.asarray(mut).tobytes() != ref.tobytes()]
    assert differing, ("fixture no longer detects wrong final-sum order; "
                       "rework flow fixture weights")
    for name, mut, ref in differing:
        with pytest.raises(OracleMismatch):
            assert_array_bytes_equal(mut, ref, f"negative_f1_fold/{name}")


def test_f2_stale_scratch_is_inert_on_corpus_documented():
    """F2 NEGATIVE RESULT, recorded as required evidence (not papered
    over): not resetting dd_buf/pd_buf between ODs is byte-INERT on this
    corpus — both in the per-OD outputs and in the scratch state itself.

    Why: every destination's reverse gradient within the gradient limit
    covers the same finite node set (the tiny graph's diameter is far
    below the limit), and the next destination's scatter overwrites
    every finite column, so no stale value ever survives into a later
    OD. Geometrically, contamination needs a node X with
    d_rev(d_prev, X) < grad_limit < d_rev(d_cur, X) while
    d_fwd(o, X) + d_rev(d_prev, X) <= budget(o, d_cur); with
    budget <= 1.5 * search_radius + buffer == grad_limit this cannot
    produce a loaded arc at corpus scale.

    Consequence (recorded in oracle_manifest.json): the obligation to
    reset flow scratch between ODs (F2I) is enforced by this journal
    evidence + code review, NOT by output-byte comparison at this graph
    scale."""
    b = b0()
    eng, settings = _flow_engine(b, 1)
    faithful = trace_flow.trace_origin_loop(b, eng, settings, 1)
    stale = mutants.stale_scratch_driver(b, eng, settings, 1)
    assert_array_bytes_equal(faithful["final_AB"], stale["final_AB"],
                             "f2_stale_inert/final_AB")
    assert_array_bytes_equal(faithful["final_BA"], stale["final_BA"],
                             "f2_stale_inert/final_BA")
    for i, (f, s) in enumerate(zip(faithful["od_journals"],
                                   stale["od_journals"])):
        assert_array_bytes_equal(f["out_AB"], s["out_AB"],
                                 f"f2_stale_inert/od{i}/AB")
        # the scratch state is byte-equal too: the scatter fully
        # overwrites the previous destination's finite columns
        assert_array_bytes_equal(
            np.asarray(f["dd_buf_before_reset"]),
            np.asarray(s["dd_buf_before_reset"]),
            f"f2_stale_inert/od{i}/dd_buf")


def test_negative_f3_reversed_where_caught():
    b = b0()
    eng, settings = _flow_engine(b, 1)
    ns = eng._prepare_params(settings)
    compiled = eng._precompute_dest_gradients(ns)
    n_dest = eng._n_destinations
    dest_nodes = np.arange(eng._n_network_nodes,
                           eng._n_network_nodes + n_dest, dtype=np.int64)
    limit = float(b._cutoff_for_shortest(
        float(ns["search_radius"]), ns["mode"], float(ns["ratio"]),
        float(ns["buffer"])))
    good = trace_gradients.trace_gradient_chunks(
        scipy_dijkstra, eng._csr_rev, dest_nodes, limit, n_dest, 1)
    assert_array_bytes_equal(good["nodes"], np.asarray(compiled[1]),
                             "faithful gradient assembly")
    bad = mutants.mutant_gradient_where_reversed(
        scipy_dijkstra, eng._csr_rev, dest_nodes, limit, n_dest, 1)
    for idx in (1, 2, 3):
        with pytest.raises(OracleMismatch):
            assert_array_bytes_equal(
                bad[("indptr", "nodes", "dist", "pred")[idx]],
                np.asarray(compiled[idx]), f"negative_f3_reversed[{idx}]")


def test_negative_a3_dropped_seed_writes_caught():
    """A3: an epoch workspace that fails to track seed/terminal writes
    projects wrong labels — exactly the 'hidden direct array read'
    failure class the dossier forbids."""
    b = b0()
    case = _engine_case("l0_tiny_random_dup_loop")
    from test_l0_traces import _scope_inputs_for_engine_case
    csr, term = _scope_inputs_for_engine_case(b, case)
    csr_full = {
        "adjacency_pointer": csr["pointer"],
        "adjacency_vector": csr["vector"],
        "adjacency_vector_weights": csr["weights"],
        "adjacynct_vector_network_node": csr["flags"],
    }
    origins = [(term["o_terminal_idxs"][i], term["o_terminal_weights"][i])
               for i in range(term["o_terminal_idxs"].shape[0])]
    result = mutants.mutant_epoch_drops_seed_writes(
        csr_full, origins, case.cutoff, term["d_count"], epoch_limit=2)
    assert not result["all_projected_equal_fresh"]
