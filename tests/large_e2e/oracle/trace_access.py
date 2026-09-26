"""Explanatory Python traces of the B0 accessibility kernel family.

These replicas exist to EXPLAIN staged transitions step by step
(A1: per-pop labels snapshot S0, eligibility, assignment/push order;
A2: destination adjustment scan and the retained filtered sequence;
A3: scope initialization/projection with an epoch-rollover simulation).
The binding oracle is the compiled B0 kernel; every trace final value
is byte-compared against it by the tests.

Replication notes (empirically pinned on the frozen profile):
* Numba's binary `min` returns -0.0 on a +/-0.0 tie in EITHER argument
  order (probed through compiled adjust_destination_distances; Python's
  min would return the first argument). `numba_min2` encodes the
  observed rule.
* Heap tuples are (float64, int64); CPython heapq ordering matches
  numba's heap ordering for these.
* The kernel pops the seed tuple and DISCARDS it before the conditional
  pushes — replicated verbatim, including that quirk.
"""
from __future__ import annotations

import heapq

import numpy as np


def numba_min2(a, b):
    """Replica of numba's two-arg min under the frozen profile.

    Observed through compiled B0: min(-0.0,+0.0) == min(+0.0,-0.0) ==
    -0.0. NaN handling follows the same comparison chain (never part of
    an in-process golden fixture; hazard probes only).
    """
    fa = float(a)
    fb = float(b)
    if fa == 0.0 and fb == 0.0:
        # +/-0 tie: numba yields -0.0 regardless of argument order.
        if np.signbit(fa) or np.signbit(fb):
            return -0.0
        return 0.0
    if fa < fb:
        return fa
    return fb


# ======================================================================
# A1 — compact_vector_node_view_scope, per-pop journal
# ======================================================================

def trace_scope(pointer, adjacency_vector, adjacency_vector_weights,
                network_flags, cutoff, d_count, o_terminal_idxs,
                o_terminal_weights, immediate_update=False,
                swap_terminal_order=False):
    """Faithful replica of B0 compact_vector_node_view_scope.

    Returns (scope_weights, journal). Journal records: init state
    (nd_node_count, sentinel b, terminal assignment sequence, seed
    tuple), and one entry per pop: popped key, S0 labels snapshot
    (BEFORE the row's assignments), the row's weight sums,
    eligibility mask and eligible (offset, neighbor, c) triples,
    assignment order, push order, and the queue contents after the row.

    Mutant knobs (negative tests only):
      immediate_update    — assign during the eligibility scan instead
                            of after computing ALL p_i (breaks the
                            duplicate-destination overwrite semantics).
      swap_terminal_order — assign the end terminal before the start.
    """
    o_idx_start = int(o_terminal_idxs[0])
    o_idx_end = int(o_terminal_idxs[1])
    o_idx_start_weight = float(o_terminal_weights[0])
    o_idx_end_weight = float(o_terminal_weights[1])

    nd_node_count = d_count + pointer.shape[0] - 1
    scope = np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff
    pred = np.empty(0, dtype=o_terminal_idxs.dtype)

    init_journal = {
        "nd_node_count": nd_node_count,
        "sentinel_b": float(scope[0]),
        "terminal_assignments": [],
        "seed_tuple": [o_idx_start_weight, o_idx_start],
    }
    if swap_terminal_order:
        scope[o_idx_end] = o_idx_end_weight
        scope[o_idx_start] = o_idx_start_weight
        init_journal["terminal_assignments"] = [
            [o_idx_end, o_idx_end_weight], [o_idx_start, o_idx_start_weight]]
    else:
        scope[o_idx_start] = o_idx_start_weight
        scope[o_idx_end] = o_idx_end_weight
        init_journal["terminal_assignments"] = [
            [o_idx_start, o_idx_start_weight], [o_idx_end, o_idx_end_weight]]

    pops = []

    def snap_labels():
        return scope.copy()

    queue = [(o_idx_start_weight, o_idx_start)]
    _weight, _node = heapq.heappop(queue)  # popped and discarded, as baseline

    if o_idx_end_weight < cutoff:
        heapq.heappush(queue, (o_idx_end_weight, o_idx_end))
    if o_idx_start_weight < cutoff:
        heapq.heappush(queue, (o_idx_start_weight, o_idx_start))

    pop_index = 0
    while queue:
        weight, node = heapq.heappop(queue)
        s0 = snap_labels()
        sp = pointer[node]
        ep = pointer[node + 1]
        weights_neighbors = adjacency_vector_weights[sp:ep] + weight

        if immediate_update:
            # MUTANT: assignment interleaved with the eligibility scan.
            queue_neighbors = []
            assignments = []
            eligible = []
            local = scope  # same buffer the comparisons read
            for k in range(weights_neighbors.shape[0]):
                c = weights_neighbors[k]
                nbr = adjacency_vector[sp + k]
                eligible_flag = bool(c <= cutoff and c < local[nbr])
                if eligible_flag:
                    queue_neighbors.append(k)
                    local[nbr] = c
                    assignments.append([int(nbr), float(c)])
                eligible.append([int(k), int(nbr), float(c), eligible_flag])
        else:
            eligible_mask = (weights_neighbors <= cutoff) & (
                weights_neighbors < scope[adjacency_vector[sp:ep]])
            queue_neighbors = np.nonzero(eligible_mask)[0]
            eligible = [
                [int(k), int(adjacency_vector[sp + k]), float(weights_neighbors[k]),
                 bool(eligible_mask[k])]
                for k in range(weights_neighbors.shape[0])
            ]
            assignments = [
                [int(adjacency_vector[sp + k]), float(weights_neighbors[k])]
                for k in queue_neighbors
            ]
            for k in queue_neighbors:
                scope[adjacency_vector[sp + k]] = weights_neighbors[k]

        pushes = []
        for k in queue_neighbors:
            nbr = int(adjacency_vector[sp + k])
            if network_flags[sp + k]:
                if pointer[nbr + 1] - pointer[nbr] > 1:
                    heapq.heappush(queue, (weights_neighbors[k], nbr))
                    pushes.append([float(weights_neighbors[k]), nbr])

        pops.append({
            "pop_index": pop_index,
            "popped": [float(weight), int(node)],
            "S0_labels_before_row": s0,
            "weights_neighbors": weights_neighbors.copy(),
            "eligibility": eligible,
            "queue_neighbors": [int(k) for k in queue_neighbors],
            "assignments_in_order": assignments,
            "pushes_in_order": pushes,
            "queue_after": [[float(w), int(n)] for (w, n) in queue],
        })
        pop_index += 1

    journal = {"init": init_journal, "pops": pops,
               "final_scope": scope.copy(), "final_pred": pred}
    return scope, journal


# ======================================================================
# A2 — adjust_destination_distances and the retained metric sequences
# ======================================================================

def trace_adjust(scope, d_terminal_idxs, d_terminal_weights, d_count):
    """Faithful replica of adjust_destination_distances.

    Returns (d_distances, journal) where the journal carries the
    per-destination scan values (start/end node, terminal weights,
    dist_to_start, dist_to_end) and the min outcome, in original ID
    order.
    """
    n_count = scope.shape[0] - d_count
    d_distances = np.empty(d_count, dtype=scope.dtype)
    rows = []
    for i in range(d_count):
        start_node = d_terminal_idxs[i, 0]
        end_node = d_terminal_idxs[i, 1]
        start_weight = d_terminal_weights[i, 0]
        end_weight = d_terminal_weights[i, 1]
        dist_to_start = scope[start_node] + start_weight
        dist_to_end = scope[end_node] + end_weight
        d_distances[i] = numba_min2(dist_to_start, dist_to_end)
        rows.append({
            "i": i,
            "start_node": int(start_node), "end_node": int(end_node),
            "start_weight": float(start_weight),
            "end_weight": float(end_weight),
            "dist_to_start": float(dist_to_start),
            "dist_to_end": float(dist_to_end),
            "min": float(d_distances[i]),
        })
    journal = {"n_count": int(n_count), "rows": rows,
               "d_distances": d_distances.copy()}
    return d_distances, journal


def trace_retained_sequences(d_distance, d_weights, cutoff, reverse=False):
    """Replica of reach_gravity_knn_access's filter stage.

    n_reach_filter = np.where(d_distance <= cutoff)[0] — ascending
    original order. `reverse=True` is the A2 negative mutant (retained
    order reversed before metrics).
    """
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    if reverse:
        n_reach_filter = n_reach_filter[::-1].copy()
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    journal = {
        "filter_indices": n_reach_filter.copy(),
        "n_distances": n_distances.copy(),
        "n_weights": n_weights.copy(),
        "n_count": int(n_distances.shape[0]),
    }
    return n_distances, n_weights, journal


# ======================================================================
# A3 — scope initialization/projection and epoch-rollover simulation
# ======================================================================

def scope_initialization_state(pointer, cutoff, d_count, o_terminal_idxs,
                               o_terminal_weights):
    """The A3 observable: labels init vector and terminal assignments.

    init = np.ones(V+D) + cutoff with the two terminal assignments; the
    D tail is left holding the raw sentinel b (the value A2's sentinel
    probe reads at scope[V]).
    """
    nd = d_count + pointer.shape[0] - 1
    init = np.ones(nd, dtype=o_terminal_weights.dtype) + cutoff
    assignments = [
        (int(o_terminal_idxs[0]), float(o_terminal_weights[0])),
        (int(o_terminal_idxs[1]), float(o_terminal_weights[1])),
    ]
    state = {
        "init_labels": init,
        "sentinel_b": float(init[0]),
        "tail_value_at_V": float(init[pointer.shape[0] - 1]) if nd > pointer.shape[0] - 1 else None,
        "terminal_assignments": assignments,
    }
    after = init.copy()
    after[assignments[0][0]] = assignments[0][1]
    after[assignments[1][0]] = assignments[1][1]
    state["after_terminal_assignment"] = after
    return state


def projection_pi(workspace, marks, epoch, b):
    """A3 Option-E projection: workspace[v] when marks[v]==epoch else b."""
    return np.where(marks == epoch, workspace, np.asarray(b, dtype=workspace.dtype))


def simulate_epoch_workspace(csr, origins, cutoff, d_count, epoch_limit=2,
                             track_seed_writes=True):
    """A3 Option-E simulation against fresh-search labels.

    One shared (workspace, marks) pair serves consecutive origins with a
    monotonically increasing epoch. Writes replayed from the per-origin
    trace (assignments AND seed/terminal writes when track_seed_writes).
    After each origin, projection_pi(...) must equal that origin's fresh
    trace labels exactly. When epoch would exceed epoch_limit, the FULL
    marks array is reset at the search boundary and the epoch restarts
    at 1 (forced rollover — production dtype unchanged; the limit is the
    documented injected counter).

    csr: dict with adjacency_pointer/adjacency_vector/
    adjacency_vector_weights/adjacynct_vector_network_node arrays
    (B0's engine attribute names).
    origins: list of (o_idx_pair array, o_weight_pair array).
    """
    pointer = csr["adjacency_pointer"]
    nd = d_count + pointer.shape[0] - 1
    b = float(np.ones(1, dtype=np.float64)[0] + cutoff)  # typed sentinel value
    workspace = np.empty(nd, dtype=np.float64)
    marks = np.zeros(nd, dtype=np.int64)
    epoch = 1
    rollovers = 0
    per_origin = []
    ok_all = True
    for (o_idx, o_w) in origins:
        fresh_labels, fresh_journal = trace_scope(
            pointer, csr["adjacency_vector"], csr["adjacency_vector_weights"],
            csr["adjacynct_vector_network_node"], cutoff, d_count, o_idx, o_w)
        for entry in fresh_journal["init"]["terminal_assignments"]:
            if track_seed_writes:
                workspace[entry[0]] = entry[1]
                marks[entry[0]] = epoch
        for pop in fresh_journal["pops"]:
            for nbr, value in pop["assignments_in_order"]:
                workspace[nbr] = value
                marks[nbr] = epoch
        projected = projection_pi(workspace, marks, epoch, b)
        equal = bool(np.array_equal(projected.view(np.uint8),
                                    fresh_labels.view(np.uint8)))
        ok_all = ok_all and equal
        per_origin.append({
            "epoch": int(epoch),
            "n_writes": int(np.count_nonzero(marks == epoch)),
            "projected_equal_fresh": equal,
        })
        if epoch + 1 > epoch_limit:
            marks[:] = 0
            epoch = 1
            rollovers += 1
        else:
            epoch += 1
    return {
        "per_origin": per_origin,
        "rollovers": rollovers,
        "all_projected_equal_fresh": ok_all,
        "epoch_limit": int(epoch_limit),
    }


# ======================================================================
# A2 — sentinel probe helper (dossier-required)
# ======================================================================

def sentinel_probe(scope, v_count, d_count, cutoff):
    """Read the actual stored tail value b = scope[V] after a search with
    D>0 and evaluate the A2 admission condition (b finite AND b > R).

    The condition uses the ACTUAL stored bytes, not the mathematical
    R+1: when cutoff >= 2**53 the typed initialization rounds
    1+cutoff back to cutoff and the probe must refuse local pruning.
    """
    if d_count <= 0:
        return {"d_count": 0, "b": None, "b_finite": None, "b_gt_R": None,
                "admits_local_pruning": False,
                "reason": "D==0: unchanged empty-destination behavior"}
    if scope.shape[0] <= v_count:
        raise ValueError("scope shorter than V+D; no sentinel tail to read")
    b = float(scope[v_count])
    b_finite = bool(np.isfinite(b))
    b_gt_R = bool(b > cutoff)
    return {
        "d_count": int(d_count),
        "b": b,
        "b_bits": np.float64(b).tobytes().hex(),
        "b_finite": b_finite,
        "b_gt_R": b_gt_R,
        "admits_local_pruning": b_finite and b_gt_R,
        "cutoff": float(cutoff),
    }
