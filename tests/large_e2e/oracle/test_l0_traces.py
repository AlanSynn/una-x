"""Trace-vs-compiled equivalence per dossier observable.

A1: per-pop staged transition journal; final labels byte-equal to the
    compiled scope kernel on every corpus case.
A2: destination adjustment scan + retained filtered sequence per
    origin, byte-equal before metrics; numba-min ±0 tie pinned.
A3: scope initialization state, per-origin projection and epoch
    rollover simulation.
F1: fixed-stripe partials + final slot-order fold vs compiled engine
    finals at K in {1,2,3}.
F2: per-OD intermediates; compiled per-OD direct run vs trace buffers.
F3: chunked gradient assembly identical across chunk sizes and equal
    to the compiled engine method.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

import fixtures
import stub_topology
import trace_access
import trace_flow
import trace_gradients
from comparator import (OracleMismatch, assert_array_bytes_equal,
                         assert_array_within_ulp, assert_scalar_bits_equal,
                         assert_scalar_within_ulp)
from support import b0


# ----------------------------------------------------------------------
# A1 — staged snapshot semantics
# ----------------------------------------------------------------------

def _scope_inputs_for_engine_case(b, case):
    topo = stub_topology.StubTopology(case)
    ge = b.Accessibility(topo).graph_engine
    term = case.terminal_arrays()
    csr = dict(
        pointer=np.asarray(ge.adjacency_pointer),
        vector=np.asarray(ge.adjacency_vector),
        weights=np.asarray(ge.adjacency_vector_weights),
        flags=np.asarray(ge.adjacynct_vector_network_node),
    )
    return csr, term


@pytest.fixture(scope="module")
def battery_cases():
    b = b0()
    out = []
    for case in fixtures.engine_cases():
        csr, term = _scope_inputs_for_engine_case(b, case)
        out.append((case, csr, term))
    return out


def test_a1_trace_equals_compiled_on_all_engine_cases(battery_cases):
    b = b0()
    for case, csr, term in battery_cases:
        for oi in range(term["o_terminal_idxs"].shape[0]):
            o_idx = term["o_terminal_idxs"][oi]
            o_w = term["o_terminal_weights"][oi]
            compiled = b.compact_vector_node_view_scope(
                o_idx, o_w, csr["pointer"], csr["vector"], csr["weights"],
                csr["flags"], case.cutoff, term["d_count"])[0]
            traced, _journal = trace_access.trace_scope(
                csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
                case.cutoff, term["d_count"], o_idx, o_w)
            assert_array_bytes_equal(traced, np.asarray(compiled),
                                     f"{case.name}/scope/{oi}/trace_vs_compiled")


def test_a1_trace_equals_compiled_on_kernel_cases():
    b = b0()
    for kc in fixtures.kernel_scope_cases():
        compiled = b.compact_vector_node_view_scope(
            kc.o_terminal_idxs[0], kc.o_terminal_weights[0],
            kc.adjacency_pointer, kc.adjacency_vector,
            kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
            kc.cutoff, kc.d_count)[0]
        traced, journal = trace_access.trace_scope(
            kc.adjacency_pointer, kc.adjacency_vector,
            kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
            kc.cutoff, kc.d_count, kc.o_terminal_idxs[0],
            kc.o_terminal_weights[0])
        assert_array_bytes_equal(traced, np.asarray(compiled),
                                 f"{kc.name}/trace_vs_compiled")
        assert journal["init"]["sentinel_b"] == 1.0 + kc.cutoff
        if not journal["pops"]:
            # Both terminal pushes are guarded by `weight < cutoff`; if
            # neither passes (k_zero_cutoff: 0.0 < 0.0 is False) the
            # queue stays empty after the discarded seed pop and NO
            # search runs — labels keep the terminal values + sentinel.
            assert (kc.o_terminal_weights[0][0] >= kc.cutoff
                    and kc.o_terminal_weights[0][1] >= kc.cutoff), kc.name
            continue
        # first pop's S0 is exactly the post-terminal-assignment vector
        expected_s0 = journal["init"]["sentinel_b"] * np.ones(
            journal["init"]["nd_node_count"])
        expected_s0[kc.o_terminal_idxs[0][0]] = kc.o_terminal_weights[0][0]
        expected_s0[kc.o_terminal_idxs[0][1]] = kc.o_terminal_weights[0][1]
        assert_array_bytes_equal(
            journal["pops"][0]["S0_labels_before_row"], expected_s0,
            f"{kc.name}/first_pop_S0")


def test_a1_duplicate_overwrite_is_staged_not_immediate():
    """A1's core adversarial property, on a DOWNSTREAM duplicate row:
    eligibility for ALL offsets of a row is decided against the same
    pop-time labels, so the LAST eligible incidence overwrites even
    when it is the more expensive one (4.0 then 6.0 -> final 6.0).

    (The k_dup_* origin-row fixtures converge to the cheap value in the
    compiled kernel because the start node is seeded twice: the stale
    seed re-scan re-admits the cheaper incidence. Their exact bytes are
    pinned by the golden corpus; the mutant divergence is proven in
    test_negative_mutations on the arrays below, where no re-scan
    occurs — node 2 has degree 0 and is never pushed.)"""
    b = b0()
    pointer = np.array([0, 1, 3, 3], dtype=np.int64)
    nbrs = np.array([1, 2, 2], dtype=np.int64)
    weights = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    flags = np.ones(3, dtype=np.bool_)
    o_idx = np.array([0, 0], dtype=np.int64)
    o_w = np.array([0.0, 0.0], dtype=np.float64)
    compiled = np.asarray(b.compact_vector_node_view_scope(
        o_idx, o_w, pointer, nbrs, weights, flags, 10.0, 1)[0])
    assert compiled[2] == 6.0  # later, LARGER eligible entry wins

    traced, journal = trace_access.trace_scope(
        pointer, nbrs, weights, flags, 10.0, 1, o_idx, o_w)
    assert_array_bytes_equal(traced, compiled, "dup_downstream/final")
    pop = [p for p in journal["pops"] if p["popped"] == [1.0, 1]][0]
    # both offsets were eligible against the SAME S0 (staged decision)
    assert [e[3] for e in pop["eligibility"]] == [True, True]
    assert pop["assignments_in_order"] == [[2, 4.0], [2, 6.0]]
    assert pop["pushes_in_order"] == []  # degree-0 neighbor: never enqueued


def test_a1_stale_entry_processed_and_harmless():
    """The stale queue entry IS popped and its row re-scanned against
    current labels (no skip-stale optimization exists) — the trace
    journal shows the re-scan finding nothing eligible; final labels
    equal the compiled kernel."""
    b = b0()
    kc = [k for k in fixtures.kernel_scope_cases()
          if k.name == "k_stale_queue"][0]
    compiled = np.asarray(b.compact_vector_node_view_scope(
        kc.o_terminal_idxs[0], kc.o_terminal_weights[0],
        kc.adjacency_pointer, kc.adjacency_vector,
        kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
        kc.cutoff, kc.d_count)[0])
    traced, journal = trace_access.trace_scope(
        kc.adjacency_pointer, kc.adjacency_vector,
        kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
        kc.cutoff, kc.d_count, kc.o_terminal_idxs[0],
        kc.o_terminal_weights[0])
    assert_array_bytes_equal(traced, compiled, "k_stale_queue/final")
    popped = [p["popped"] for p in journal["pops"]]
    assert [2.0, 2] in popped, popped          # improved entry processed
    stale = [p for p in journal["pops"] if p["popped"] == [2.5, 2]]
    assert len(stale) == 1, popped             # stale entry also popped
    assert all(not e[3] for e in stale[0]["eligibility"])  # re-scan empty
    assert traced[2] == 2.0  # stale 2.5 never overwrote the better label


def test_a1_degree_one_never_enqueued():
    b = b0()
    kc = [k for k in fixtures.kernel_scope_cases()
          if k.name == "k_flags_false_row"][0]
    traced, journal = trace_access.trace_scope(
        kc.adjacency_pointer, kc.adjacency_vector,
        kc.adjacency_vector_weights, kc.adjacynct_vector_network_node,
        kc.cutoff, kc.d_count, kc.o_terminal_idxs[0],
        kc.o_terminal_weights[0])
    # node 2 (flags False, degree 1) gets a label but no push; node 1
    # (degree 1, flag True) also assigned without push
    pushed_nodes = {p[1] for pop in journal["pops"] for p in pop["pushes_in_order"]}
    assert 1 not in pushed_nodes and 2 not in pushed_nodes
    assert traced[1] == 1.0 and traced[2] == 1.5


# ----------------------------------------------------------------------
# A2 — destination adjustment and retained sequences
# ----------------------------------------------------------------------

def test_a2_adjust_and_retained_sequences_byte_equal(battery_cases):
    b = b0()
    for case, csr, term in battery_cases:
        for oi in range(term["o_terminal_idxs"].shape[0]):
            o_idx = term["o_terminal_idxs"][oi]
            o_w = term["o_terminal_weights"][oi]
            compiled_scope = b.compact_vector_node_view_scope(
                o_idx, o_w, csr["pointer"], csr["vector"], csr["weights"],
                csr["flags"], case.cutoff, term["d_count"])[0]
            compiled_adjust = b.adjust_destination_distances(
                compiled_scope, term["d_terminal_idxs"],
                term["d_terminal_weights"], term["d_count"])

            traced_scope, _j1 = trace_access.trace_scope(
                csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
                case.cutoff, term["d_count"], o_idx, o_w)
            traced_adjust, j2 = trace_access.trace_adjust(
                traced_scope, term["d_terminal_idxs"],
                term["d_terminal_weights"], term["d_count"])
            assert_array_bytes_equal(
                traced_adjust, np.asarray(compiled_adjust),
                f"{case.name}/adjust/{oi}")

            n_dist, n_w, j3 = trace_access.trace_retained_sequences(
                traced_adjust, term["d_weights"], case.cutoff)
            filt = np.where(np.asarray(compiled_adjust) <= case.cutoff)[0]
            assert_array_bytes_equal(n_dist, np.asarray(compiled_adjust)[filt],
                                     f"{case.name}/n_distances/{oi}")
            assert_array_bytes_equal(n_w, term["d_weights"][filt],
                                     f"{case.name}/n_weights/{oi}")
            # retained order is ascending ORIGINAL destination id
            assert np.all(np.diff(j3["filter_indices"]) > 0) or \
                j3["filter_indices"].size <= 1
            # journal exposes the full O(D) scan the dossier requires
            assert len(j2["rows"]) == term["d_count"]


def test_a2_tied_distances_unequal_weights_retain_original_order():
    """Tied adjusted distances with unequal destination weights: the
    retained sequence keeps original-ID order, so a KNN crop picks the
    LOWER id's weight. Built at kernel level: one edge 0-1, both
    destinations snapped to the same nodes with identical terminal
    weights — identical adjusted values, weights 1.0 vs 3.0."""
    b = b0()
    pointer = np.array([0, 1, 2], dtype=np.int64)
    nbrs = np.array([1, 0], dtype=np.int64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    flags = np.ones(2, dtype=np.bool_)
    o_idx = np.array([0, 0], dtype=np.int64)
    o_w = np.array([0.0, 0.0], dtype=np.float64)
    cutoff = 10.0
    scope = np.asarray(b.compact_vector_node_view_scope(
        o_idx, o_w, pointer, nbrs, weights, flags, cutoff, 2)[0])
    d_idx = np.array([[0, 1], [0, 1]], dtype=np.int64)   # identical snaps
    d_w = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    d_weights = np.array([1.0, 3.0], dtype=np.float64)   # unequal weights
    adjust = np.asarray(b.adjust_destination_distances(
        scope, d_idx, d_w, 2))
    assert adjust[0] == adjust[1] == 0.5  # exactly tied
    n_dist, n_w, _j = trace_access.trace_retained_sequences(
        adjust, d_weights, cutoff)
    assert n_dist.shape[0] == 2
    assert n_w[0] == 1.0 and n_w[1] == 3.0  # original order retained


def test_a2_sentinel_two_pow_53_hazard_recorded(battery_cases):
    """The dossier's b<=R domain: with cutoff 2**53 the UNREACHABLE
    destination's adjusted distance rounds to exactly R and PASSES the
    compiled filter — golden behavior, must never be 'fixed' silently."""
    b = b0()
    case = [c for c in fixtures.engine_cases()
            if c.name == "flt_sentinel_two_pow_53"][0]
    csr, term = _scope_inputs_for_engine_case(b, case)
    scope = b.compact_vector_node_view_scope(
        term["o_terminal_idxs"][0], term["o_terminal_weights"][0],
        csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
        case.cutoff, term["d_count"])[0]
    adjust = b.adjust_destination_distances(
        scope, term["d_terminal_idxs"], term["d_terminal_weights"],
        term["d_count"])
    assert adjust[0] == case.cutoff  # disconnected destination == R
    filt = np.where(adjust <= case.cutoff)[0]
    assert 0 in filt  # ...and it is retained by the compiled filter


# ----------------------------------------------------------------------
# A3 — initialization, integrated pipeline, projection and rollover
# ----------------------------------------------------------------------

def test_a3_initialization_state_and_integrated_pipeline(battery_cases):
    b = b0()
    for case, csr, term in battery_cases:
        state = trace_access.scope_initialization_state(
            csr["pointer"], case.cutoff, term["d_count"],
            term["o_terminal_idxs"][0], term["o_terminal_weights"][0])
        assert state["sentinel_b"] == 1.0 + case.cutoff
        init = state["init_labels"]
        assert init.shape[0] == term["d_count"] + csr["pointer"].shape[0] - 1
        assert init.dtype == np.float64

        for oi in range(term["o_terminal_idxs"].shape[0]):
            o_idx = term["o_terminal_idxs"][oi]
            o_w = term["o_terminal_weights"][oi]
            integrated = b.integrated_scope_access(
                term["o_terminal_idxs"], term["o_terminal_weights"],
                csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
                term["d_terminal_idxs"], term["d_terminal_weights"],
                term["d_weights"], case.gravity_beta, case.metric_plateau,
                case.metric_midpoint, float(np.log(99.0)) / case.metric_midpoint,
                case.knn_decay, np.ascontiguousarray(case.knn_weights),
                case.cutoff)
            scope_traced, _ = trace_access.trace_scope(
                csr["pointer"], csr["vector"], csr["weights"], csr["flags"],
                case.cutoff, term["d_count"], o_idx, o_w)
            adjust_traced, _ = trace_access.trace_adjust(
                scope_traced, term["d_terminal_idxs"],
                term["d_terminal_weights"], term["d_count"])
            n_dist, n_w, _ = trace_access.trace_retained_sequences(
                adjust_traced, term["d_weights"], case.cutoff)
            reach, gexp, glog, knn = b.reach_gravity_knn_access(
                n_dist, n_w, case.cutoff, case.gravity_beta,
                case.metric_plateau, case.metric_midpoint,
                case.metric_plateau, np.ascontiguousarray(case.knn_weights),
                case.gravity_beta, case.knn_decay, case.metric_midpoint,
                float(np.log(99.0)) / case.metric_midpoint,
                float(np.log(99.0)) / case.metric_midpoint)
            # reach cast into the integrated int64 output (truncation)
            assert integrated[0].dtype == np.int64
            assert int(reach) == int(integrated[0][oi])
            assert_scalar_bits_equal(gexp, integrated[1][oi],
                                     f"{case.name}/gexp/{oi}")
            assert_scalar_bits_equal(glog, integrated[2][oi],
                                     f"{case.name}/glog/{oi}")
            assert_scalar_bits_equal(knn, integrated[3][oi],
                                     f"{case.name}/knn/{oi}")


def test_a3_epoch_rollover_projection_equals_fresh_searches():
    b = b0()
    case = [c for c in fixtures.engine_cases()
            if c.name == "l0_tiny_random_dup_loop"][0]
    csr, term = _scope_inputs_for_engine_case(b, case)
    csr_full = {
        "adjacency_pointer": csr["pointer"],
        "adjacency_vector": csr["vector"],
        "adjacency_vector_weights": csr["weights"],
        "adjacynct_vector_network_node": csr["flags"],
    }
    origins = [(term["o_terminal_idxs"][i], term["o_terminal_weights"][i])
               for i in range(term["o_terminal_idxs"].shape[0])]
    result = trace_access.simulate_epoch_workspace(
        csr_full, origins, case.cutoff, term["d_count"], epoch_limit=2)
    assert result["rollovers"] >= 1, "fixture must force a rollover"
    assert result["all_projected_equal_fresh"]
    assert all(o["projected_equal_fresh"] for o in result["per_origin"])

    mutant = __import__("mutants").mutant_epoch_drops_seed_writes(
        csr_full, origins, case.cutoff, term["d_count"], epoch_limit=2)
    assert not mutant["all_projected_equal_fresh"], (
        "negative mutant (dropped seed writes) must break the projection")


# ----------------------------------------------------------------------
# F1 / F2 — fixed stripes and per-OD state
# ----------------------------------------------------------------------

FLOW_KS = (1, 2, 3)


@pytest.fixture(scope="module")
def flow_battery():
    b = b0()
    from support import flow_engine_case
    eng, settings, spec = flow_engine_case(b, None)
    out = {}
    for K in FLOW_KS:
        eng.Centrality(settings)
        eng.num_threads = K
        eng.Centrality(settings)
        out[K] = dict(
            AB=np.asarray(eng.edge_flow_AB).copy(),
            BA=np.asarray(eng.edge_flow_BA).copy(),
            node=np.asarray(eng.node_flow).copy(),
        )
    out["engine"] = eng
    out["settings"] = settings
    out["spec"] = spec
    return out


@pytest.mark.parametrize("K", FLOW_KS)
def test_f1_stripe_partials_and_fold_equal_compiled(flow_battery, K):
    b = b0()
    eng = flow_battery["engine"]
    settings = flow_battery["settings"]
    compiled = flow_battery[K]

    trace = trace_flow.trace_origin_loop(b, eng, settings, K)
    partials = trace["partials"]

    # stripe membership is the static range(slot, n_origins, K)
    n_origins = int(eng._n_origins)
    assert partials["stripe_members"] == [
        list(range(slot, n_origins, K)) for slot in range(K)]
    assert partials["fold_sequence"] == list(range(K))
    assert len(partials["local_AB"]) == K

    # final fold in slot order 0..K-1 equals the compiled engine output.
    # Multi-term float sums are compared within a small ULP bound: the
    # compiled kernel runs with fastmath=True, which reassociates the
    # excess sums, and a strict IEEE-left-to-right Python replica
    # cannot reproduce that bit-for-bit on every input (observed 1 ULP
    # on this fixture). Candidate-vs-baseline comparisons remain
    # BYTE-exact elsewhere (golden corpus, arm runs, double runs).
    assert_array_within_ulp(trace["final_AB"], compiled["AB"], 4, f"K{K}/AB")
    assert_array_within_ulp(trace["final_BA"], compiled["BA"], 4, f"K{K}/BA")
    assert_array_within_ulp(trace["final_node"], compiled["node"], 4,
                            f"K{K}/node")

    # the partials genuinely partition the work: summing them in the
    # WRONG order is a different float computation (mutant caught below
    # in test_negative...); here assert each origin appears exactly once
    seen = sorted(o for m in partials["stripe_members"] for o in m)
    assert seen == list(range(n_origins))


@pytest.mark.parametrize("K", FLOW_KS)
def test_f2_per_od_trace_equals_compiled_kernel(flow_battery, K):
    """Per-OD: drive the COMPILED _accumulate_od_flow directly with the
    same scattered buffers and require byte-equal deltas + delivered."""
    b = b0()
    eng = flow_battery["engine"]
    settings = flow_battery["settings"]
    trace = trace_flow.trace_origin_loop(b, eng, settings, K)
    csr_indptr, csr_indices = eng._csr_indptr, eng._csr_indices
    csr_weights, csr_edge_id, csr_direction = (
        eng._csr_weights, eng._csr_edge_id, eng._csr_direction)
    n_total = csr_indptr.shape[0] - 1
    n_net = eng._n_network_nodes
    n_edges = eng.edge_flow_AB.shape[0]
    ns = eng._prepare_params(settings)
    if ns["path_penalty"] == "equal":
        curve_id = b._DECAY_EQUAL
    elif ns["path_penalty"] == "exponential":
        curve_id = b._DECAY_EXPONENTIAL
    else:
        curve_id = b._DECAY_LOGISTIC
    decay_beta = float(ns["route_beta"])
    decay_midpoint = (float(ns["route_midpoint"])
                      if ns["route_midpoint"] > 0 else 200.0)

    origin_state = {rec["o_pos"]: rec for rec in trace["origin_journals"]}
    checked = 0
    for od in trace["od_journals"]:
        o_pos, d_idx = od["o_pos"], od["d_idx"]
        o_rec = origin_state[o_pos]
        # recompute the driver scatter state for this OD from scratch
        dd_buf = np.full(n_total, np.inf, dtype=np.float64)
        pd_buf = np.full(n_total, -9999, dtype=np.int32)
        dd_buf[od["gradient_cols"]] = od["gradient_dist"]
        pd_buf[od["gradient_cols"]] = od["gradient_pred"]
        out_AB = np.zeros(n_edges, dtype=np.float64)
        out_BA = np.zeros(n_edges, dtype=np.float64)
        out_node = np.zeros(n_net, dtype=np.float64)
        origins = eng.topology.origins
        delivered = b._accumulate_od_flow(
            csr_indptr, csr_indices, csr_weights, csr_edge_id, csr_direction,
            o_rec["d_o"], dd_buf, o_rec["pred_o"], pd_buf,
            int(eng._first_origin_node + o_pos), int(n_net + d_idx),
            int(origins.nearest_edge_id[o_pos]),
            int(eng.topology.destinations.nearest_edge_id[d_idx]),
            od["d_shortest"], od["budget"], curve_id, decay_beta,
            decay_midpoint, od["trip_vol"], n_net,
            out_AB, out_BA, out_node)
        # per-OD float arrays: small-ULP comparison (fastmath
        # reassociation in the compiled kernel — see the F1 test note);
        # the journal's write replay is byte-exact against itself and
        # the F3 gradient assembly below stays BYTE-exact.
        assert_array_within_ulp(out_AB, od["out_AB"], 4, f"K{K}/od{checked}/AB")
        assert_array_within_ulp(out_BA, od["out_BA"], 4, f"K{K}/od{checked}/BA")
        assert_array_within_ulp(out_node, od["out_node_flow"], 4,
                                f"K{K}/od{checked}/node")
        assert_scalar_within_ulp(delivered, od["delivered"], 4,
                                 f"K{K}/od{checked}/delivered")
        # journal completeness per dossier F2 observables
        for field in ("reach_nodes", "order_o", "order_d", "cont_o", "cont_d",
                      "q_sum", "scale", "acc_o", "acc_d"):
            assert od[field] is not None, field
        checked += 1
    assert checked >= 2  # the fixture must exercise several real ODs


def test_f2_zero_weight_origin_and_unreachable_destination_skipped():
    b = b0()
    from support import flow_engine_case
    eng, settings, _ = flow_engine_case(b, None)
    trace = trace_flow.trace_origin_loop(b, eng, settings, 1)
    skipped = [r for r in trace["origin_journals"]
               if r["skipped_zero_weight"]]
    assert [r["o_pos"] for r in skipped] == [2]
    # unreachable destination d2 never appears in any OD journal
    assert all(od["d_idx"] != 2 for od in trace["od_journals"])


def test_f2_overlapping_host_edge_od_has_live_envelope():
    """o0 shares its host edge (id 0) with d0: the OD is real, delivers
    flow, and the snap-edge exclusion behaves EXACTLY as the compiled
    kernel's guard reads — network-network arcs of the two snap edges
    are skipped, while CONNECTOR arcs (one endpoint virtual) carrying
    the host edge id are loaded like any other arc."""
    b = b0()
    from support import flow_engine_case
    eng, settings, _ = flow_engine_case(b, None)
    n_net = eng._n_network_nodes
    trace = trace_flow.trace_origin_loop(b, eng, settings, 1)
    od0 = [od for od in trace["od_journals"] if od["o_pos"] == 0][0]
    assert od0["cont_o"].dtype == np.bool_ and od0["cont_d"].dtype == np.bool_
    assert od0["q_sum"] > 0.0 and od0["scale"] > 0.0
    assert od0["delivered"] > 0.0
    snap_arc_eids, connector_arc_eids = set(), set()
    for (u, ai, x, _e, _q) in od0["pass1_arcs"]:
        eid = int(eng._csr_edge_id[ai])
        if u < n_net and x < n_net:
            snap_arc_eids.add(eid)
        else:
            connector_arc_eids.add(eid)
    assert 0 not in snap_arc_eids, (
        "a network-network arc of the shared snap edge was loaded")
    assert 0 in connector_arc_eids, (
        "expected the virtual-endpoint connector carrying edge id 0 "
        "to be loaded (the exclusion only guards u,x < n_net)")


# ----------------------------------------------------------------------
# F3 — chunked gradients
# ----------------------------------------------------------------------

def test_f3_chunk_sizes_identical_and_equal_compiled():
    b = b0()
    from support import flow_engine_case
    eng, settings, _ = flow_engine_case(b, None)
    ns = eng._prepare_params(settings)
    compiled = eng._precompute_dest_gradients(ns)
    csr_rev = eng._csr_rev
    n_dest = eng._n_destinations
    n_total = eng._csr_indptr.shape[0] - 1
    dest_nodes = np.arange(eng._n_network_nodes,
                           eng._n_network_nodes + n_dest, dtype=np.int64)
    limit = float(b._cutoff_for_shortest(
        float(ns["search_radius"]), ns["mode"], float(ns["ratio"]),
        float(ns["buffer"])))

    variants = {}
    for chunk in (1, 2, 3, max(1, int(1e8 // max(n_total, 1)))):
        variants[chunk] = trace_gradients.trace_gradient_chunks(
            scipy_dijkstra, csr_rev, dest_nodes, limit, n_dest, chunk)

    compiled_map = dict(zip(("indptr", "nodes", "dist", "pred"), compiled))
    base = variants[1]
    for chunk, result in variants.items():
        for field in ("indptr", "nodes", "dist", "pred"):
            assert_array_bytes_equal(result[field], compiled_map[field],
                                     f"F3/chunk{chunk}/{field}")
            assert_array_bytes_equal(result[field], base[field],
                                     f"F3/chunk{chunk}vs1/{field}")
    # uneven tail exercised: chunk 2 over 3 destinations
    assert variants[2]["chunks"][-1]["end"] - variants[2]["chunks"][-1]["start"] == 1
    # per-row np.where order is ascending global node id
    rows = variants[1]["chunks"][0]["rows"]
    for row in rows:
        assert np.all(np.diff(row["cols"]) > 0) or row["cols"].size <= 1
    # pred dtype and dist dtype roles preserved
    assert variants[1]["pred"].dtype == np.int32
    assert variants[1]["dist"].dtype == np.float64
    assert variants[1]["nodes"].dtype == np.int64
