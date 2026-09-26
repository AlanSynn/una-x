"""Deliberately broken replicas for the negative-mutation tests.

Each mutant MUST be caught by the comparator on at least one crafted
fixture — the tests prove this by asserting OracleMismatch. Mutants are
confined to this module and never used as golden sources.
"""
from __future__ import annotations

import numpy as np

from trace_access import (numba_min2, trace_adjust, trace_retained_sequences,
                          trace_scope)


def mutant_scope_immediate_update(pointer, adjacency_vector,
                                  adjacency_vector_weights, network_flags,
                                  cutoff, d_count, o_terminal_idxs,
                                  o_terminal_weights):
    """A1 mutant: eligibility evaluated against labels that already
    contain this row's assignments (immediate update instead of the
    staged snapshot S0)."""
    return trace_scope(pointer, adjacency_vector, adjacency_vector_weights,
                       network_flags, cutoff, d_count, o_terminal_idxs,
                       o_terminal_weights, immediate_update=True)


def mutant_scope_terminal_order(pointer, adjacency_vector,
                                adjacency_vector_weights, network_flags,
                                cutoff, d_count, o_terminal_idxs,
                                o_terminal_weights):
    """A1/A3 mutant: terminal assignments applied end-before-start."""
    return trace_scope(pointer, adjacency_vector, adjacency_vector_weights,
                       network_flags, cutoff, d_count, o_terminal_idxs,
                       o_terminal_weights, swap_terminal_order=True)


def mutant_retained_sequence_reversed(d_distance, d_weights, cutoff):
    """A2 mutant: retained n_distances/n_weights reversed before metrics."""
    return trace_retained_sequences(d_distance, d_weights, cutoff,
                                    reverse=True)


def mutant_adjust_min_python_semantics(scope, d_terminal_idxs,
                                       d_terminal_weights, d_count):
    """A2 mutant: Python's first-arg min replaces numba's -0-on-tie min.

    Distinguishable only on a +/-0 tie between dist_to_start and
    dist_to_end (fixture flt_signed_zero)."""
    d_distances = np.empty(d_count, dtype=scope.dtype)
    for i in range(d_count):
        ds = scope[d_terminal_idxs[i, 0]] + d_terminal_weights[i, 0]
        de = scope[d_terminal_idxs[i, 1]] + d_terminal_weights[i, 1]
        d_distances[i] = min(ds, de)  # Python semantics: first arg on tie
    return d_distances


def mutant_stripe_shift(trace_result):
    """F1 mutant: stripe membership shifted by one
    (range(slot+1, n_origins, K)) — visible whenever n_origins is not a
    multiple of K."""
    partials = trace_result["partials"]
    members = partials["stripe_members"]
    n_origins = max((m[-1] + 1 for m in members if m), default=0)
    K = len(members)
    shifted = [list(range(slot + 1, n_origins, K)) for slot in range(K)]
    return {"expected": members, "mutant": shifted}


def mutant_final_sum_descending(final_arrays, K):
    """F1 mutant: final reduction folds slots K-1..0 instead of 0..K-1."""
    local_AB = final_arrays["partials"]["local_AB"]
    local_BA = final_arrays["partials"]["local_BA"]
    node = final_arrays["partials"]["local_node"]
    n_edges = local_AB[0].shape[0]
    AB = np.zeros(n_edges, dtype=np.float64)
    BA = np.zeros(n_edges, dtype=np.float64)
    nd = node[0].shape[0]
    node_sum = np.zeros(nd, dtype=np.float64)
    for slot in range(K - 1, -1, -1):
        AB += local_AB[slot]
        BA += local_BA[slot]
        if nd:
            node_sum += node[slot]
    return AB, BA, node_sum


def stale_scratch_driver(b0, engine, settings, K):
    """F2 mutant: dd_buf/pd_buf not reset between ODs — stale finite
    entries from one destination's gradient would contaminate the next
    OD. PROVEN INERT on the corpus (see
    test_f2_stale_scratch_is_inert_on_corpus_documented): the gradient
    finite-column sets coincide and the next scatter overwrites all of
    them, so outputs and even scratch states stay byte-identical."""
    from trace_flow import trace_origin_loop
    return trace_origin_loop(b0, engine, settings, K, stale_scratch=True)


def mutant_gradient_where_reversed(scipy_dijkstra, csr_rev, dest_nodes,
                                   limit, n_dest, chunk):
    """F3 mutant: per-row np.where(finite) order reversed before the
    sparse append."""
    from trace_gradients import trace_gradient_chunks
    return trace_gradient_chunks(scipy_dijkstra, csr_rev, dest_nodes,
                                 limit, n_dest, chunk, reverse_where=True)


def mutant_epoch_drops_seed_writes(csr, origins, cutoff, d_count,
                                   epoch_limit=2):
    """A3 mutant: epoch workspace tracking skips the seed/terminal
    writes, so projection_pi loses the terminal labels."""
    from trace_access import simulate_epoch_workspace
    return simulate_epoch_workspace(csr, origins, cutoff, d_count,
                                    epoch_limit=epoch_limit,
                                    track_seed_writes=False)
