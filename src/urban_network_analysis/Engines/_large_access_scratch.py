"""Private A1 helper: snapshot-preserving search scratch (campaign una_large_e2e, task A1I).

Provides a per-origin two-phase variant of the accessibility scope
search (`compact_vector_node_view_scope`) that stages each row's
eligible incidence offsets/weights in two reusable per-origin buffers
instead of materializing per-pop NumPy temporaries. Results are
byte-identical to the original kernel on the admitted domain
(campaigns/una_large_e2e/evidence/A1I/proof.md sections 3-5).

`_a1_scope_admits` is the admission guard: a compile-time typing gate
(`numba.extending.overload`) plus a runtime value scan. Signatures
outside the admitted domain resolve to a stub that refuses, so the
calling drivers fall back to their unchanged original loops for every
refusal (proof.md section 6). The guard allocates nothing, casts
nothing, warns nothing, and raises nothing; it runs once per driver
call and returns (admitted, max_degree).

`_a1_scope_search` mirrors the original kernel prologue byte-for-byte
and returns the same (o_scope_weights, o_scope_pred) pair. Scratch
buffers are allocated by the caller inside each prange iteration and
shared with no other search (proof.md section 4, decision D3).

`_a3_scope_search_tailless` (task A3I) is the tail-free private
variant: the per-origin label vector covers only the network-node
domain, dropping the destination tail [V, nd) that no admitted firing
reads or writes (campaigns/una_large_e2e/evidence/A3I/proof.md
sections 2-5). `_a3_tail_admits` is its typed guard: it validates the
destination terminal shape and value domain once per driver call, and
every refusal keeps the driver's A1 route byte-identical.
"""
from heapq import heappush, heappop

import numba as nb
import numpy as np
from numba.extending import overload

# Numba JIT compilation settings (identical to the engine modules).
NUMBA_PARALLEL = False
NUMBA_CACHE = True
NUMBA_NOGIL = True
NUMBA_FASTMATH = True

# Scratch capacity bound (proof.md decision D5): peak live scratch is
# threads * max_degree * 16 bytes; refuse the private route when that
# exceeds 256 MiB.
_A1_SCRATCH_CAP_BYTES = 268435456
_A1_SCRATCH_CAP_ELEMS = _A1_SCRATCH_CAP_BYTES // 16


@nb.njit(nogil=True)
def _a1_threads_bound():
    # numba's configured default pool size (also the set_num_threads
    # maximum), so it bounds the runtime thread count from above; read
    # through a small callee because a direct get_num_threads() call
    # marks callers as using dynamic globals and disables their
    # numba caching.
    return nb.config.NUMBA_NUM_THREADS


def _a1_scope_admits_scan(
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    o_terminal_idxs,
    o_terminal_weights,
    cutoff,
):
    """Runtime value scan (proof.md section 3, conditions 1-12).

    Refusals return (False, 0); the caller then runs its original,
    unchanged loop.  Shared verbatim by the compiled overload (numba
    compiles this body) and the pure-Python guard below (executed when
    numba JIT is disabled), so both dispatch modes make the SAME
    admission decision.  BUG-JIT-DISABLED-STUB fix: the guard used to
    have a `pass` body whose @overload only applied under compilation,
    so NUMBA_DISABLE_JIT=1 runs unpacked None and crashed.
    """
    n_pointer = adjacency_pointer.shape[0]
    if n_pointer < 1:
        return (False, 0)
    node_count = n_pointer - 1

    # Condition 8: CSR validity; maximum degree from the same pass.
    if adjacency_pointer[0] != 0:
        return (False, 0)
    max_degree = 0
    for v in range(node_count):
        degree = adjacency_pointer[v + 1] - adjacency_pointer[v]
        if degree < 0:
            return (False, 0)
        if degree > max_degree:
            max_degree = degree
    if adjacency_pointer[node_count] != adjacency_vector.shape[0]:
        return (False, 0)

    # Conditions 5/6 shape parts: terminal rows are (start, end).
    if o_terminal_idxs.shape[1] != 2:
        return (False, 0)
    if o_terminal_weights.shape[0] != o_terminal_idxs.shape[0]:
        return (False, 0)
    if o_terminal_weights.shape[1] != 2:
        return (False, 0)

    # Condition 7: cutoff finite and nonnegative.
    if not (cutoff == cutoff):
        return (False, 0)
    if not (cutoff < np.inf):
        return (False, 0)
    if not (cutoff >= 0):
        return (False, 0)

    # Conditions 9/10: endpoints in range, costs finite and >= 0.
    n_edges = adjacency_vector.shape[0]
    for j in range(n_edges):
        neighbor = adjacency_vector[j]
        if neighbor < 0 or neighbor >= node_count:
            return (False, 0)
        cost = adjacency_vector_weights[j]
        if cost != cost or cost == np.inf or cost < 0.0:
            return (False, 0)

    # Condition 11: origin terminal nodes in range.
    n_origins = o_terminal_idxs.shape[0]
    for o in range(n_origins):
        for k in range(2):
            terminal = o_terminal_idxs[o, k]
            if terminal < 0 or terminal >= node_count:
                return (False, 0)

    # Condition 12: scratch capacity (bytes = H * max_degree * 16).
    threads_bound = _a1_threads_bound()
    if threads_bound < 1:
        return (False, 0)
    if max_degree > _A1_SCRATCH_CAP_ELEMS // threads_bound:
        return (False, 0)

    return (True, max_degree)


def _a1_scope_admits(
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    o_terminal_idxs,
    o_terminal_weights,
    cutoff,
):
    # Pure-Python body: executed only when numba JIT is disabled (the
    # @overload below replaces it under compilation).  Mirrors the
    # compile-time typing gate with numpy dtypes, then runs the same
    # value scan, so interpreted diagnostic runs make the same
    # admission decision as compiled ones (BUG-JIT-DISABLED-STUB).
    admitted = (
        adjacency_pointer.ndim == 1
        and adjacency_pointer.dtype == np.int64
        and adjacency_vector.ndim == 1
        and adjacency_vector.dtype == np.int64
        and adjacency_vector_weights.ndim == 1
        and adjacency_vector_weights.dtype == np.float64
        and adjacynct_vector_network_node.ndim == 1
        and adjacynct_vector_network_node.dtype == np.bool_
        and o_terminal_idxs.ndim == 2
        and o_terminal_idxs.dtype == np.int64
        and o_terminal_weights.ndim == 2
        and o_terminal_weights.dtype == np.float64
        and isinstance(cutoff, (int, float, np.integer, np.floating))
    )
    if not admitted:
        return (False, 0)
    return _a1_scope_admits_scan(
        adjacency_pointer,
        adjacency_vector,
        adjacency_vector_weights,
        adjacynct_vector_network_node,
        o_terminal_idxs,
        o_terminal_weights,
        cutoff,
    )


@overload(_a1_scope_admits)
def _ol_a1_scope_admits(
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    o_terminal_idxs,
    o_terminal_weights,
    cutoff,
):
    # Compile-time typing gate (proof.md section 3, conditions 1-6
    # dtype/ndim parts and the cutoff type): non-admitted signatures
    # resolve to a stub that never touches the arrays, so it types for
    # every signature the original kernel call types.
    admitted = (
        adjacency_pointer.ndim == 1
        and adjacency_pointer.dtype == nb.int64
        and adjacency_vector.ndim == 1
        and adjacency_vector.dtype == nb.int64
        and adjacency_vector_weights.ndim == 1
        and adjacency_vector_weights.dtype == nb.float64
        and adjacynct_vector_network_node.ndim == 1
        and adjacynct_vector_network_node.dtype == nb.boolean
        and o_terminal_idxs.ndim == 2
        and o_terminal_idxs.dtype == nb.int64
        and o_terminal_weights.ndim == 2
        and o_terminal_weights.dtype == nb.float64
        and cutoff in (nb.float64, nb.int64)
    )
    if not admitted:
        return lambda adjacency_pointer, adjacency_vector, adjacency_vector_weights, adjacynct_vector_network_node, o_terminal_idxs, o_terminal_weights, cutoff: (False, 0)

    return _a1_scope_admits_scan


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
)
def _a1_scope_search(
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
    """Snapshot-preserving two-phase scope search (proof.md section 2).

    Identical prologue, heap sequence and return as
    compact_vector_node_view_scope; each row's eligibility pass writes
    (absolute offset, cost) pairs into the caller-owned scratch instead
    of materializing per-pop temporaries, then the assignment/push pass
    replays exactly the stored pairs in ascending row order.
    """
    nd_node_count = d_count + adjacency_pointer.shape[0] - 1
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = o_terminal_weights[0]
    o_idx_end_weight = o_terminal_weights[1]

    o_scope_weights = np.ones(nd_node_count, dtype=o_terminal_weights.dtype) + cutoff
    o_scope_pred = np.empty(0, dtype=o_terminal_idxs.dtype)

    # Add start node segment terminals to seen with their weights
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

        # Phase one: eligibility against the pre-row label snapshot;
        # no label writes in this pass.
        n_eligible = 0
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                eligible_offset[n_eligible] = node_start_pointer + i
                eligible_weight[n_eligible] = neighbor_weight
                n_eligible += 1

        # Phase two: assignments and pushes from the stored pairs, in
        # ascending row order, with no eligibility recomputation.
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
def _a3_tail_admits(d_terminal_idxs, d_count, node_count):
    """A3 tail-free route admission check (proof.md section 3.1).

    Returns True only when `d_terminal_idxs` has shape (d_count, 2) and
    every value lies in [0, node_count): the read domain that the
    shared adjust_destination_distances exercises on a tail-free label
    vector. Allocates nothing, writes nothing, raises nothing; runs
    once per driver call. A refusal sends the driver to its unchanged
    A1 route.
    """
    if d_terminal_idxs.ndim != 2:
        return False
    if d_terminal_idxs.shape[0] != d_count or d_terminal_idxs.shape[1] != 2:
        return False
    for i in range(d_count):
        for j in range(2):
            terminal_node = d_terminal_idxs[i, j]
            if terminal_node < 0 or terminal_node >= node_count:
                return False
    return True


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
)
def _a3_scope_search_tailless(
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
    """Tail-free two-phase scope search (task A3I, proof.md section 5).

    Line-for-line copy of `_a1_scope_search` with a single delta: the
    label vector is sized to the network-node domain
    `adjacency_pointer.shape[0] - 1` instead of
    `d_count + adjacency_pointer.shape[0] - 1`, dropping the
    destination tail [V, nd) that no admitted firing reads or writes
    (proof.md section 2). No other line differs: same prologue, heap
    sequence, two-phase eligibility/assignment passes, and the same
    (o_scope_weights, o_scope_pred) return shape.

    Delta inventory vs `_a1_scope_search` (review pin P3):
    - the `d_count` parameter is removed from the signature;
    - the init is `np.ones(adjacency_pointer.shape[0] - 1, ...) +
      cutoff` and the `nd_node_count` local is gone;
    - names `_a1_*` -> `_a3_*` and this docstring.

    Callers must first admit via `_a3_tail_admits` (proof.md section
    3.1); scratch buffers are caller-owned per prange iteration, as in
    A1.
    """
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = o_terminal_weights[0]
    o_idx_end_weight = o_terminal_weights[1]

    o_scope_weights = np.ones(adjacency_pointer.shape[0] - 1, dtype=o_terminal_weights.dtype) + cutoff
    o_scope_pred = np.empty(0, dtype=o_terminal_idxs.dtype)

    # Add start node segment terminals to seen with their weights
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

        # Phase one: eligibility against the pre-row label snapshot;
        # no label writes in this pass.
        n_eligible = 0
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if neighbor_weight <= cutoff and neighbor_weight < o_scope_weights[neighbor_node]:
                eligible_offset[n_eligible] = node_start_pointer + i
                eligible_weight[n_eligible] = neighbor_weight
                n_eligible += 1

        # Phase two: assignments and pushes from the stored pairs, in
        # ascending row order, with no eligibility recomputation.
        for j in range(n_eligible):
            offset = eligible_offset[j]
            neighbor_weight = eligible_weight[j]
            neighbor_node = adjacency_vector[offset]
            o_scope_weights[neighbor_node] = neighbor_weight
            if adjacynct_vector_network_node[offset]:
                if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                    heappush(queue, (neighbor_weight, neighbor_node))

    return o_scope_weights, o_scope_pred
