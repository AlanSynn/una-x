"""A1 negative-test mutants (test-only; never golden sources).

`mutant_scope_immediate_update_scratch` is the A1-shaped staging bug:
it keeps the two-buffer scratch structure but eliminates phase two,
updating labels and pushing inside the phase-one scan (immediate
update). On any row containing a duplicate destination the later
incidence is then tested against the ALREADY-UPDATED label instead of
the pre-row snapshot S0, which must move bytes of the final labels on
the duplicate fixtures (k_dup_dest_cheap_first/last) and be caught by
comparator.assert_array_bytes_equal.
"""
from __future__ import annotations

from heapq import heappush, heappop

import numba as nb
import numpy as np


@nb.njit(
    parallel=False,
    cache=False,  # test-only mutant: never cached, never a golden source
    nogil=True,
    fastmath=True,
)
def mutant_scope_immediate_update_scratch(
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    cutoff,
    d_count,
    eligible_offset,   # unused scratch kept in signature: candidate-shaped
    eligible_weight,   # unused scratch kept in signature: candidate-shaped
):
    """A1-shaped immediate-update mutant (staging eliminated)."""
    nd_node_count = d_count + adjacency_pointer.shape[0] - 1
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = o_terminal_weights[0]
    o_idx_end_weight = o_terminal_weights[1]

    o_scope_weights = np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff
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

        # MUTANT: single pass — labels updated and pushes issued DURING
        # the scan, so later incidences in the same row see earlier
        # writes (no pre-row snapshot for duplicate destinations).
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                o_scope_weights[neighbor_node] = neighbor_weight
                if adjacynct_vector_network_node[node_start_pointer + i]:
                    if (adjacency_pointer[neighbor_node + 1]
                            - adjacency_pointer[neighbor_node]) > 1:
                        heappush(queue, (neighbor_weight, neighbor_node))

    return o_scope_weights, o_scope_pred
