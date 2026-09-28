"""A3 first-divergence and equivalence mutants (proof.md section 7,
T2). Compiled with cache=False so no mutant cache entries land in the
campaign cache root; every other decorator flag matches the real
kernel (fastmath included: the equivalence mutant must share the real
kernel's arithmetic).

* no_cutoff_init      — init missing `+ cutoff`: must DIVERGE from the
                        real tail-free kernel (the test's discriminating
                        power against init accidents).
* tail_restored       — the tail-free kernel with the destination tail
                        put back (init over d_count + V): must be
                        BYTE-EQUAL on [0, V) for every origin — the
                        executable form of the tail-deadness claim —
                        with its tail slots stuck at the sentinel
                        1 + cutoff.
* guard_range_disabled— the A3 guard without the value-range scan:
                        returns True where the real guard refuses;
                        compared at the guard level only (never used to
                        drive a kernel).
"""
from __future__ import annotations

import numba as nb
import numpy as np
from heapq import heappush, heappop


@nb.njit(parallel=False, cache=False, nogil=True, fastmath=True)
def no_cutoff_init(
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    cutoff,
    eligible_offset,
    eligible_weight,
):
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = o_terminal_weights[0]
    o_idx_end_weight = o_terminal_weights[1]

    o_scope_weights = np.ones(adjacency_pointer.shape[0] - 1, dtype=o_terminal_weights.dtype)
    o_scope_pred = np.empty(0, dtype=o_terminal_idxs.dtype)

    o_scope_weights[o_idx_start] = o_idx_start_weight
    o_scope_weights[o_idx_end] = o_idx_end_weight

    queue = [(o_idx_start_weight, o_idx_start)]
    weight, node = heappop(queue)

    if o_idx_end_weight < cutoff:
        heappush(queue, (o_idx_end_weight, o_idx_end))

    if o_idx_start_weight < cutoff:
        heappush(queue, (o_idx_start_weight, o_idx_start))

    while queue:
        weight, node = heappop(queue)

        node_start_pointer = adjacency_pointer[node]
        node_end_pointer = adjacency_pointer[node + 1]

        n_eligible = 0
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                eligible_offset[n_eligible] = node_start_pointer + i
                eligible_weight[n_eligible] = neighbor_weight
                n_eligible += 1

        for j in range(n_eligible):
            offset = eligible_offset[j]
            neighbor_weight = eligible_weight[j]
            neighbor_node = adjacency_vector[offset]
            o_scope_weights[neighbor_node] = neighbor_weight
            if adjacynct_vector_network_node[offset]:
                if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                    heappush(queue, (neighbor_weight, neighbor_node))

    return o_scope_weights, o_scope_pred


@nb.njit(parallel=False, cache=False, nogil=True, fastmath=True)
def tail_restored(
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    cutoff,
    d_count,
    eligible_offset,
    eligible_weight,
):
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = o_terminal_weights[0]
    o_idx_end_weight = o_terminal_weights[1]

    o_scope_weights = np.ones(d_count + adjacency_pointer.shape[0] - 1, dtype=o_terminal_weights.dtype) + cutoff
    o_scope_pred = np.empty(0, dtype=o_terminal_idxs.dtype)

    o_scope_weights[o_idx_start] = o_idx_start_weight
    o_scope_weights[o_idx_end] = o_idx_end_weight

    queue = [(o_idx_start_weight, o_idx_start)]
    weight, node = heappop(queue)

    if o_idx_end_weight < cutoff:
        heappush(queue, (o_idx_end_weight, o_idx_end))

    if o_idx_start_weight < cutoff:
        heappush(queue, (o_idx_start_weight, o_idx_start))

    while queue:
        weight, node = heappop(queue)

        node_start_pointer = adjacency_pointer[node]
        node_end_pointer = adjacency_pointer[node + 1]

        n_eligible = 0
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                eligible_offset[n_eligible] = node_start_pointer + i
                eligible_weight[n_eligible] = neighbor_weight
                n_eligible += 1

        for j in range(n_eligible):
            offset = eligible_offset[j]
            neighbor_weight = eligible_weight[j]
            neighbor_node = adjacency_vector[offset]
            o_scope_weights[neighbor_node] = neighbor_weight
            if adjacynct_vector_network_node[offset]:
                if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                    heappush(queue, (neighbor_weight, neighbor_node))

    return o_scope_weights, o_scope_pred


@nb.njit
def guard_range_disabled(d_terminal_idxs, d_count, node_count):
    if d_terminal_idxs.ndim != 2:
        return False
    if d_terminal_idxs.shape[0] != d_count or d_terminal_idxs.shape[1] != 2:
        return False
    return True
