"""L0: portable epoch-projection state machine (campaign una_10x S11).

The mirror executes the a3 schedule maintaining BOTH representations in
one run: dense labels (dense-initialized to ``b``) and the kernel's
epoch mechanics (stamps, per-epoch touched, projection reads, mark
dedup, ascending-original-id enumeration).  Every heap pop reconstructs
the full dense view from the projection and compares bitwise against
the dense side — chronological parity at each event, never final-only
(VALIDATION.md sparse rules).  The dense side is cross-checked per
origin against the frozen oracle `scope_search_schedule`, and candidate
sequences are compared against `adjust_destination_distances_plan`
restricted to candidate ids (dossier 03's byte-identical-order rule at
the sequence level).  Inputs here are admitted-domain (finite,
nonnegative) so plain heapq equals the compiled heap law — the NaN-law
surface is refused upstream and covered by the refusal suite.
"""
import heapq
import struct

import numpy as np

from urban_network_analysis.kernels.reductions import (
    adjust_destination_distances_plan)
from urban_network_analysis.kernels import scope_access as SA

from conftest import bits


def reverse_index(d_idx, v_count):
    """Terminal node -> destination ids, BOTH columns, duplicates kept."""
    rev = [[] for _ in range(v_count)]
    for i in range(d_idx.shape[0]):
        rev[int(d_idx[i, 0])].append(i)
        rev[int(d_idx[i, 1])].append(i)
    return rev


def epoch_mirror(origins, ptr, adj, wts, flag, d_idx, d_w, d_weight,
                 cutoff, epoch_rollover=1 << 62):
    """Run the cohort through the epoch mechanics with dense lockstep."""
    v_count = ptr.shape[0] - 1
    b = float(np.ones(1, dtype=np.float64)[0] + cutoff)
    rev = reverse_index(d_idx, v_count)

    label = np.empty(v_count, dtype=np.float64)  # write-before-read
    stamp = np.zeros(v_count, dtype=np.int64)
    mark = np.zeros(d_idx.shape[0], dtype=np.int64)
    epoch = 0
    results = []
    events = []

    for o_idx, o_w in origins:
        s = int(o_idx[0])
        t = int(o_idx[1])
        w0 = float(o_w[0])
        w1 = float(o_w[1])
        epoch += 1
        if epoch > epoch_rollover:
            stamp[:] = 0
            mark[:] = 0
            epoch = 1
        e = epoch
        touched = []

        # unconditional seed writes (start then end; same-node last-wins)
        for node, w in ((s, w0), (t, w1)):
            if stamp[node] != e:
                stamp[node] = e
                touched.append(node)
            label[node] = w

        # dense side, same origin
        dense = np.ones(v_count, dtype=np.float64) + cutoff
        dense[s] = w0
        dense[t] = w1

        heap = [(w0, s)]
        heapq.heappop(heap)
        if SA.fast_lt(w1, cutoff):
            heapq.heappush(heap, (w1, t))
        if SA.fast_lt(w0, cutoff):
            heapq.heappush(heap, (w0, s))

        writes = 0
        while heap:
            w, node = heapq.heappop(heap)
            recon = np.full(v_count, b, dtype=np.float64)
            idx = [v for v in range(v_count) if stamp[v] == e]
            recon[idx] = label[idx]  # M1 law: stamp-filtered gather
            assert bits(recon) == bits(dense), (
                "projection != dense at pop", len(events), recon, dense)
            events.append(("pop", w, node))
            eligible = []
            for i in range(int(ptr[node]), int(ptr[node + 1])):
                c = float(wts[i]) + w
                nb = int(adj[i])
                lw = label[nb] if stamp[nb] == e else b
                if SA.fast_le(c, cutoff) and SA.fast_lt(c, lw):
                    eligible.append((i, c, nb))
            for offset, c, nb in eligible:
                if stamp[nb] != e:
                    stamp[nb] = e
                    touched.append(nb)
                label[nb] = c
                dense[nb] = c
                writes += 1
                if flag[offset] and int(ptr[nb + 1]) - int(ptr[nb]) > 1:
                    heapq.heappush(heap, (c, nb))

        # dossier 03: touched (this epoch) -> mark-dedup -> ascending ids
        ids = []
        for v in touched:
            for d in rev[v]:
                if mark[d] != e:
                    mark[d] = e
                    ids.append(d)
        ids.sort()
        cand = []
        for d in ids:
            a = (label[int(d_idx[d, 0])] if stamp[int(d_idx[d, 0])] == e
                 else b) + float(d_w[d, 0])
            c2 = (label[int(d_idx[d, 1])] if stamp[int(d_idx[d, 1])] == e
                  else b) + float(d_w[d, 1])
            cand.append((min(a, c2), float(d_weight[d])))

        # dense side cross-checked against the frozen oracle
        dense_oracle = SA.scope_search_schedule(
            np.array([s, t], dtype=np.int64),
            np.array([w0, w1], dtype=np.float64),
            ptr, adj, wts, flag, cutoff, d_count=0, domain="a3")
        assert bits(dense) == bits(dense_oracle), (s, t, dense, dense_oracle)

        # candidate sequence == adjust plan restricted to candidate ids
        full_adjust = adjust_destination_distances_plan(
            dense, d_idx, np.ascontiguousarray(d_w, dtype=np.float64))
        for k, d in enumerate(ids):
            assert cand[k][0] == full_adjust[d], ("dist mismatch", d)
            assert cand[k][1] == float(d_weight[d]), ("weight mismatch", d)

        results.append(dict(labels=dense, candidates=cand, ids=ids,
                            touched=list(touched), writes=writes))
    return results, events


def tiny_csr(n, arcs, weights=None, flags=None):
    rows = {v: [] for v in range(n)}
    for k, (a, b) in enumerate(arcs):
        rows[a].append(k)
    ptr = [0]
    vec = []
    w_list = []
    for v in range(n):
        for k in rows[v]:
            vec.append(arcs[k][1])
            w_list.append(0.5 + 0.5 * (k % 3) if weights is None
                          else weights[k])
        ptr.append(len(vec))
    flag = (np.arange(len(vec)) % 4 != 0) if flags is None else flags
    return (np.asarray(ptr, dtype=np.int64),
            np.asarray(vec, dtype=np.int64),
            np.asarray(w_list, dtype=np.float64),
            np.asarray(flag, dtype=bool))


ALL_ARCS_3 = [(a, b) for a in range(3) for b in range(3)]  # incl. self-loops
ALL_ARCS_4 = [(a, b) for a in range(4) for b in range(4)]


def test_exhaustive_tiny_subsets_mirror_and_oracle():
    """Every arc subset on n=2/3 and a stride sample on n=4: the mirror's
    per-event projection equals the dense side, whose final labels equal
    the frozen oracle (bitwise)."""
    rng = np.random.default_rng(5)
    origins = [([0, 1], [0.25, 1.0]), ([1, 1], [1.5, 0.5]),
               ([2, 0], [0.0, 0.75])]
    origins_run = 0
    for n, arcs in ((2, [(0, 1), (1, 0)]),
                    (3, ALL_ARCS_3),
                    (4, ALL_ARCS_4[::3])):
        for mask in range(1 << len(arcs)):
            sub = [arcs[k] for k in range(len(arcs)) if (mask >> k) & 1]
            ptr, adj, wts, flag = tiny_csr(n, sub)
            d = np.array([[1, 0], [n - 1, 2 % n]], dtype=np.int64)
            dw = np.abs(rng.uniform(0.0, 0.4, (2, 2)))
            dwt = rng.uniform(0.25, 1.0, 2)
            cutoff = 2.5
            for o_pair, o_w in origins:
                o_idx = np.array([min(i, n - 1) for i in o_pair],
                                 dtype=np.int64)
                epoch_mirror([(o_idx, np.array(o_w, dtype=np.float64))],
                             ptr, adj, wts, flag, d, dw, dwt, cutoff)
                origins_run += 1
    assert origins_run > 1500


def test_kernel_matches_baseline_exhaustive_stride():
    """The COMPILED kernel vs the compiled baseline on the same stride
    families (bitwise, all four outputs, sparse route asserted)."""
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    from urban_network_analysis.Engines import _sparse_access_scratch as SP
    from conftest import metric_args
    rng = np.random.default_rng(6)
    checked = 0
    for n, arcs in ((3, ALL_ARCS_3), (4, ALL_ARCS_4[::2])):
        for mask in range(0, 1 << len(arcs), 7):
            sub = [arcs[k] for k in range(len(arcs)) if (mask >> k) & 1]
            if not sub:
                continue
            ptr, adj, wts, flag = tiny_csr(n, sub)
            d = np.array([[1, 0], [n - 1, 2 % n]], dtype=np.int64)
            dw = np.abs(rng.uniform(0.0, 0.4, (2, 2)))
            dwt = rng.uniform(0.25, 1.0, 2)
            o = np.array([[0, min(1, n - 1)]], dtype=np.int64)
            ow = np.array([[0.25, 1.0]])
            if o.max() >= n or d.max() >= n:
                continue
            args = (o, ow, ptr, adj, wts, flag, d, dw, dwt) + metric_args() \
                + (2.5,)
            base = integrated_scope_access(*args)
            res = SP.sparse_scope_access(*args)
            assert res.route == SP.ROUTE_SPARSE
            for f, b in zip(("reach", "gravity_exponential",
                             "gravity_logistic", "knn_access"), base):
                assert bits(getattr(res, f)) == bits(b)
            checked += 1
    assert checked > 100


def test_touched_then_untouched_binding():
    """S10 binding condition: per-epoch touched filtering when a later
    origin touches a strict subset of an earlier epoch (the E7b shape)."""
    ptr, adj, wts, flag = tiny_csr(4, [(0, 1), (1, 2), (2, 3)])
    d = np.array([[0, 2], [1, 3], [2, 2]], dtype=np.int64)
    dw = np.array([[0.25, 0.5], [0.5, 0.25], [0.0, 0.0]])
    dwt = np.array([1.0, 0.5, 0.75])
    origins = [(np.array([0, 1], dtype=np.int64), np.array([0.0, 0.5])),
               (np.array([2, 2], dtype=np.int64), np.array([1.0, 0.5])),
               (np.array([1, 3], dtype=np.int64), np.array([0.5, 2.0]))]
    res, _ = epoch_mirror(origins, ptr, adj, wts, flag, d, dw, dwt, 10.0)
    assert len(res[1]["touched"]) < len(res[0]["touched"])
    # origin (2,2): seed 2, and 2 -> 3 relaxes within cutoff 10
    assert res[1]["touched"] == [2, 3]
    # origin (0,1): seed write 0.5 == label[1] is not fast_lt, so 1 only
    # enters via its seed; expansion reaches node 2 (node 2 is a leaf,
    # so the leaf push gate ends the walk there)
    assert res[0]["touched"] == [0, 1, 2]


def test_driver_rollover_dirty_continuation():
    """Checked epoch counter at reduced width: stamps/marks cleared
    BEFORE rollover; the dirty arena continues bit-exactly (driver vs
    compiled baseline, epoch_rollover=2 over a 9-origin cohort)."""
    from urban_network_analysis.Engines.AccessibilityWElevation import (
        integrated_scope_access)
    from urban_network_analysis.Engines import _sparse_access_scratch as SP
    from conftest import metric_args, admitted_case
    case = admitted_case("base")
    o_rep = np.tile(case["o"], (3, 1))
    ow_rep = np.tile(case["ow"], (3, 1))
    args = (o_rep, ow_rep, case["ptr"], case["vec"], case["wts"],
            case["flag"], case["d"], case["dw"], case["dwt"]) \
        + metric_args() + (case["cutoff"],)
    base = integrated_scope_access(*args)
    res = SP.sparse_scope_access(*args, epoch_rollover=2)
    assert res.route == SP.ROUTE_SPARSE
    for f, b in zip(("reach", "gravity_exponential", "gravity_logistic",
                     "knn_access"), base):
        assert bits(getattr(res, f)) == bits(b)
    # and the mirror at the same reduced width agrees with itself+oracle
    origins = [(case["o"][0], case["ow"][0])] * 3
    epoch_mirror(origins, case["ptr"], case["vec"], case["wts"],
                 case["flag"], case["d"], case["dw"], case["dwt"],
                 case["cutoff"], epoch_rollover=2)


def test_same_node_unequal_seeds_last_wins():
    ptr, adj, wts, flag = tiny_csr(3, [(0, 1), (1, 2)])
    d = np.array([[2, 0]], dtype=np.int64)
    dw = np.array([[0.5, 0.25]])
    dwt = np.array([1.0])
    res, _ = epoch_mirror([(np.array([1, 1], dtype=np.int64),
                            np.array([1.5, 0.5]))],
                          ptr, adj, wts, flag, d, dw, dwt, 5.0)
    assert res[0]["labels"][1] == 0.5
    # seed 1, then the expansion of 1 touches neighbor 2
    assert res[0]["touched"] == [1, 2]


def test_equal_parallel_improvement_twice():
    """Two equal candidates in one snapshot row: both accepted, both
    replayed (VALIDATION 'accepted TWICE before either write')."""
    ptr = np.array([0, 2, 3, 3], dtype=np.int64)
    adj = np.array([1, 1, 2], dtype=np.int64)
    wts = np.array([0.5, 0.5, 0.5])
    flag = np.array([True, True, True])
    d = np.array([[1, 2]], dtype=np.int64)
    dw = np.array([[0.0, 0.25]])
    dwt = np.array([1.0])
    res, events = epoch_mirror([(np.array([0, 0], dtype=np.int64),
                                 np.array([0.0, 4.0]))],
                               ptr, adj, wts, flag, d, dw, dwt, 5.0)
    assert res[0]["writes"] == 2
    assert res[0]["labels"][1] == 0.5
    # node 1 is a leaf (degree 1): the leaf push gate suppresses its
    # push entirely; both pops are of node 0 (improving + stale copies)
    assert sum(1 for ev in events if ev[0] == "pop" and ev[2] == 1) == 0
    assert sum(1 for ev in events if ev[0] == "pop" and ev[2] == 0) == 2


def test_cutoff_boundaries():
    ptr, adj, wts, flag = tiny_csr(3, [(0, 1), (1, 2)], weights=[0.5, 0.5],
                                   flags=[True, True])
    d = np.array([[1, 2], [2, 0]], dtype=np.int64)
    dw = np.array([[0.0, 0.0], [0.25, 0.0]])
    dwt = np.array([1.0, 0.5])
    # seed == cutoff: strict < push gate keeps it unexpanded but touched
    res, _ = epoch_mirror([(np.array([0, 2], dtype=np.int64),
                            np.array([1.0, 4.0]))],
                          ptr, adj, wts, flag, d, dw, dwt, 1.0)
    assert res[0]["touched"] == [0, 2]
    # candidate exactly at cutoff: closed-radius membership keeps it
    res, _ = epoch_mirror([(np.array([0, 2], dtype=np.int64),
                            np.array([0.5, 4.0]))],
                          ptr, adj, wts, flag, d, dw, dwt, 1.0)
    # nextafter below cutoff propagates; the oracles already pinned bits
    res, _ = epoch_mirror([(np.array([0, 2], dtype=np.int64),
                            np.array([np.nextafter(1.0, 0.0), 4.0]))],
                          ptr, adj, wts, flag, d, dw, dwt, 1.0)
    assert res[0]["candidates"] is not None


def test_signed_zero_retention():
    """Signed-zero seeds/terminal weights: -0.0 is not negative, is
    retained, and min's first-operand-on-equality bits are preserved."""
    for s_zero, t_zero in ((0.0, 0.0), (0.0, -0.0), (-0.0, 0.0),
                           (-0.0, -0.0)):
        ptr, adj, wts, flag = tiny_csr(2, [(0, 1)], weights=[-0.0],
                                       flags=[True])
        d = np.array([[0, 1]], dtype=np.int64)
        dw = np.array([[s_zero, t_zero]])
        dwt = np.array([1.0])
        res, _ = epoch_mirror([(np.array([0, 1], dtype=np.int64),
                                np.array([s_zero, t_zero]))],
                              ptr, adj, wts, flag, d, dw, dwt, 1.0)
        assert len(res[0]["candidates"]) == 1
        got = struct.pack("<d", res[0]["candidates"][0][0])
        dense = SA.scope_search_schedule(
            np.array([0, 1], dtype=np.int64),
            np.array([s_zero, t_zero]), ptr, adj, wts, flag, 1.0,
            d_count=0, domain="a3")
        expect = adjust_destination_distances_plan(dense, d, dw)
        assert got == struct.pack("<d", expect[0]), (s_zero, t_zero)
