"""Characterization oracle for the scope-access search schedule (CPU_ALGORITHMS).

Executes SPEC.md section 3 (the characterized una_legacy scope search)
in plain Python + numpy — no numba — bit-identically to the compiled
engine kernels (`Engines/_large_access_scratch.py` A1/A3 and
`Engines/AccessibilityWElevation.py::compact_vector_node_view_scope`)
on the admitted domain.  This module is the portable schedule
definition a NATIVE_CORE port conforms against; the conformance tests
pin it bitwise against the compiled engines per platform.

Characterized fast-math laws (run cpuk-20261003T, comparison_probe/):
the una_legacy kernels compile with ``fastmath=True``, under which LLVM
canonicalizes the scalar search comparisons while leaving the fold's
vectorized ``np.where`` membership at IEEE semantics:

- ``a < b``  executes as ``not (b <= a)`` (IEEE inner compare);
- ``a <= b`` executes as ``not (b < a)``  (IEEE inner compare);
- ``a > b`` and ``a >= b`` execute at IEEE;
- ``min(a, b)`` / ``max(a, b)`` keep Python/builtin semantics.

Consequences pinned by tests and required of every port: NaN seed
weights pass the ``< cutoff`` push gates and enter the heap; NaN
candidates are eligible (``NaN <= cutoff`` true) and propagate through
labels; the heap order for NaN entries follows the fast tuple order,
reproduced here by a verbatim port of the CPython heapq sift
algorithms with the fast comparison.  ``min`` in the destination
adjust keeps first-operand-on-incomparability semantics
(``min(NaN, x) = NaN``).
"""
import os
from dataclasses import dataclass

import numpy as np

from .reductions import (
    PLAN_ADJUST,
    PLAN_FOLD,
    PLAN_SCOPE_SEARCH,
    STAGE_ADJUST,
    STAGE_FOLD,
    STAGE_SCOPE_SEARCH,
    adjust_destination_distances_plan,
    fold_reach_gravity_knn_legacy_plan,
)

__all__ = [
    "PLAN_SCOPE_SEARCH",
    "ROUTE_A3",
    "ROUTE_A1",
    "ROUTE_COMPACT",
    "ScopeAccessResult",
    "fast_lt",
    "fast_le",
    "integrated_scope_access_oracle",
    "scope_route_admits",
    "scope_search_schedule",
    "tail_admits",
]

# Route tags for attribution reporting (KernelOutputs provenance).
ROUTE_A3 = "a3_scope_search_tailless"
ROUTE_A1 = "a1_scope_search"
ROUTE_COMPACT = "compact_fallback"


def fast_lt(a, b):
    """The compiled search's ``a < b``: ``not (b <= a)`` (IEEE inner)."""
    return not (b <= a)


def fast_le(a, b):
    """The compiled search's ``a <= b``: ``not (b < a)`` (IEEE inner)."""
    return not (b < a)


def _default_threads_bound():
    """Upper bound on the numba thread pool, numba-free.

    The A1 scratch-cap condition needs the same bound the engine guard
    uses (``_large_access_scratch._a1_threads_bound`` =
    ``numba.config.NUMBA_NUM_THREADS``, whose default is the CPU count
    when the environment does not override it).  Resolves numba lazily
    so this module stays importable without numba; the conformance
    tests pin oracle and engine agreement on the host.
    """
    try:
        import numba
        return int(numba.config.NUMBA_NUM_THREADS)
    except Exception:
        env = os.environ.get("NUMBA_NUM_THREADS")
        if env:
            try:
                return int(env)
            except ValueError:
                pass
        cpu = os.cpu_count()
        return 1 if cpu is None else int(cpu)


# 256 MiB scratch-cap pattern, identical to the engine modules.
_A1_SCRATCH_CAP_BYTES = 268435456
_A1_SCRATCH_CAP_ELEMS = _A1_SCRATCH_CAP_BYTES // 16


def scope_route_admits(adjacency_pointer, adjacency_vector,
                       adjacency_vector_weights, flag,
                       o_terminal_idxs, o_terminal_weights, cutoff):
    """Pure-Python mirror of ``_large_access_scratch._a1_scope_admits``.

    Same type gate (numpy dtypes + the four admitted cutoff scalar
    types) followed by the same ordered value scan, so this returns the
    same ``(admitted, max_degree)`` decision as the engine guard's two
    dispatch modes on the same host.  Refusal is ROUTING, not error:
    the driver falls to the compact route.
    """
    admitted = (
        adjacency_pointer.ndim == 1
        and adjacency_pointer.dtype == np.int64
        and adjacency_vector.ndim == 1
        and adjacency_vector.dtype == np.int64
        and adjacency_vector_weights.ndim == 1
        and adjacency_vector_weights.dtype == np.float64
        and flag.ndim == 1
        and flag.dtype == np.bool_
        and o_terminal_idxs.ndim == 2
        and o_terminal_idxs.dtype == np.int64
        and o_terminal_weights.ndim == 2
        and o_terminal_weights.dtype == np.float64
        and (
            type(cutoff) is int
            or type(cutoff) is float
            or type(cutoff) is np.int64
            or type(cutoff) is np.float64
        )
    )
    if not admitted:
        return (False, 0)
    return _scope_route_admits_scan(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        flag, o_terminal_idxs, o_terminal_weights, cutoff)


def _scope_route_admits_scan(adjacency_pointer, adjacency_vector,
                             adjacency_vector_weights, flag,
                             o_terminal_idxs, o_terminal_weights, cutoff):
    """Ordered value scan, condition-for-condition identical to the
    engine guard's shared scan body.  The comparisons here guard
    ADMISSION (not results); they run in Python in both dispatch modes
    of the engine guard, so plain IEEE Python comparisons mirror them."""
    n_pointer = adjacency_pointer.shape[0]
    if n_pointer < 1:
        return (False, 0)
    node_count = n_pointer - 1

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

    if o_terminal_idxs.shape[1] != 2:
        return (False, 0)
    if o_terminal_weights.shape[0] != o_terminal_idxs.shape[0]:
        return (False, 0)
    if o_terminal_weights.shape[1] != 2:
        return (False, 0)

    if adjacency_vector_weights.shape[0] != adjacency_vector.shape[0]:
        return (False, 0)

    if not (cutoff == cutoff):
        return (False, 0)
    if not (cutoff < np.inf):
        return (False, 0)
    if not (cutoff >= 0):
        return (False, 0)

    n_edges = adjacency_vector.shape[0]
    for j in range(n_edges):
        neighbor = adjacency_vector[j]
        if neighbor < 0 or neighbor >= node_count:
            return (False, 0)
        cost = adjacency_vector_weights[j]
        if cost != cost or cost == np.inf or cost < 0.0:
            return (False, 0)

    n_origins = o_terminal_idxs.shape[0]
    for o in range(n_origins):
        for k in range(2):
            terminal = o_terminal_idxs[o, k]
            if terminal < 0 or terminal >= node_count:
                return (False, 0)

    threads_bound = _default_threads_bound()
    if threads_bound < 1:
        return (False, 0)
    if max_degree > _A1_SCRATCH_CAP_ELEMS // threads_bound:
        return (False, 0)

    return (True, max_degree)


def tail_admits(d_terminal_idxs, d_count, node_count):
    """Pure-Python mirror of ``_large_access_scratch._a3_tail_admits``:
    shape (d_count, 2) and every terminal in [0, node_count)."""
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


def _tuple_lt(a, b):
    """The compiled heap's item order, characterized on the engine
    (run cpuk-20261003T, comparison_probe/heap_sequences): a NaN weight
    sorts as less than everything — ``NaN`` on the left is "less" and
    on the right is "not less" — and finite-weight items compare
    lexicographically under the fast weight law with ascending node
    tie-break.  Both probed push/pop sequences reproduce bit-exactly
    under this law against numba's CPython-derived sift algorithms
    (`numba/cpython/heapq.py`), which the push/pop ports below mirror."""
    w_a = a[0]
    w_b = b[0]
    if w_a != w_a:
        return True
    if w_b != w_b:
        return False
    if w_a < w_b:
        return True
    if w_a == w_b:
        return a[1] < b[1]
    return False


def _heap_push(heap, item):
    """CPython heapq.heappush (verbatim algorithm) under _tuple_lt."""
    heap.append(item)
    pos = len(heap) - 1
    new_item = heap[pos]
    while pos > 0:
        parent_pos = (pos - 1) >> 1
        parent = heap[parent_pos]
        if _tuple_lt(new_item, parent):
            heap[pos] = parent
            pos = parent_pos
            continue
        break
    heap[pos] = new_item


def _heap_pop(heap):
    """CPython heapq.heappop (verbatim algorithm) under _tuple_lt."""
    last = heap.pop()
    if heap:
        return_item = heap[0]
        heap[0] = last
        end_pos = len(heap)
        pos = 0
        start_pos = pos
        new_item = heap[pos]
        child_pos = 2 * pos + 1
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos and not _tuple_lt(heap[child_pos],
                                                     heap[right_pos]):
                child_pos = right_pos
            child = heap[child_pos]
            if _tuple_lt(new_item, child):
                break
            heap[pos] = child
            pos = child_pos
            child_pos = 2 * pos + 1
        heap[pos] = new_item
        _heap_siftdown(heap, start_pos, pos)
        return return_item
    return last


def _heap_siftdown(heap, start_pos, pos):
    """CPython heapq._siftdown under _tuple_lt."""
    new_item = heap[pos]
    while pos > start_pos:
        parent_pos = (pos - 1) >> 1
        parent = heap[parent_pos]
        if _tuple_lt(new_item, parent):
            heap[pos] = parent
            pos = parent_pos
            continue
        break
    heap[pos] = new_item


def scope_search_schedule(o_terminal_idxs, o_terminal_weights,
                          adjacency_pointer, adjacency_vector,
                          adjacency_vector_weights, flag, cutoff,
                          d_count=0, domain="a3"):
    """One-origin scope search under ``plan.accessibility.scope_search.v1``.

    ``domain='a3'`` sizes the label vector to the network-node domain
    (tail-free route); ``domain='a1'`` sizes it to ``V + d_count``
    exactly as the A1 route does.  The destination tail is never read
    on admitted inputs (SPEC.md section 2), so the [0, V) prefixes of
    the two domains agree bit-for-bit — pinned by the conformance
    tests.  Returns the label array (the engine's empty predecessor
    array is part of the compiled signature only).
    """
    o_idx_start = o_terminal_idxs[0]
    o_idx_end = o_terminal_idxs[1]
    o_idx_start_weight = float(o_terminal_weights[0])
    o_idx_end_weight = float(o_terminal_weights[1])

    if domain == "a3":
        domain_size = adjacency_pointer.shape[0] - 1
    elif domain == "a1":
        domain_size = d_count + adjacency_pointer.shape[0] - 1
    else:
        raise ValueError(f"domain must be 'a1'|'a3', got {domain!r}")

    labels = np.ones(domain_size, dtype=o_terminal_weights.dtype) + cutoff
    labels[o_idx_start] = o_terminal_weights[0]
    labels[o_idx_end] = o_terminal_weights[1]

    queue = [(o_idx_start_weight, int(o_idx_start))]
    weight, node = _heap_pop(queue)

    # Seed push gates: STRICT ``< cutoff`` under the fast comparison
    # law (NaN weights therefore pass — characterized, module docstring).
    if fast_lt(o_idx_end_weight, cutoff):
        _heap_push(queue, (o_idx_end_weight, int(o_idx_end)))

    if fast_lt(o_idx_start_weight, cutoff):
        _heap_push(queue, (o_idx_start_weight, int(o_idx_start)))

    while queue:
        weight, node = _heap_pop(queue)

        node_start_pointer = adjacency_pointer[node]
        node_end_pointer = adjacency_pointer[node + 1]

        # Phase one: eligibility against the pre-row label snapshot,
        # collected in ascending row order; no label writes.  Both
        # tests run under the fast comparison law — NaN candidates are
        # eligible and NaN labels are improvable (characterized).
        eligible = []
        for i in range(node_end_pointer - node_start_pointer):
            neighbor_weight = float(adjacency_vector_weights[node_start_pointer + i]) + weight
            neighbor_node = adjacency_vector[node_start_pointer + i]
            if fast_le(neighbor_weight, cutoff) and fast_lt(neighbor_weight, labels[neighbor_node]):
                eligible.append((node_start_pointer + i, neighbor_weight))

        # Phase two: assignments and pushes in stored row order, gated
        # by the flag column and the leaf-node degree gate (> 1; plain
        # integer comparison).
        for j in range(len(eligible)):
            offset, neighbor_weight = eligible[j]
            neighbor_node = adjacency_vector[offset]
            labels[neighbor_node] = neighbor_weight
            if flag[offset]:
                if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                    _heap_push(queue, (neighbor_weight, int(neighbor_node)))

    return labels


@dataclass(frozen=True)
class ScopeAccessResult:
    """The four accessibility outputs plus route attribution.

    Array dtypes mirror the engine driver: ``reach`` takes
    ``o_terminal_idxs.dtype``; the three float outputs take
    ``adjacency_vector_weights.dtype`` (the engine stores the fold
    results into arrays of the edge-weight dtype — a float32 fallback
    run stores through float32, and the oracle does the same).
    """

    reach: np.ndarray
    gravity_exponential: np.ndarray
    gravity_logistic: np.ndarray
    knn_access: np.ndarray
    route: str


def integrated_scope_access_oracle(o_terminal_idxs, o_terminal_weights,
                                   adjacency_pointer, adjacency_vector,
                                   adjacency_vector_weights, flag,
                                   d_terminal_idxs, d_terminal_weights,
                                   d_weights, gravity_beta, gravity_plateau,
                                   gravity_logistic_midpoint,
                                   gravity_growth_rate, knn_decay,
                                   knn_weights, cutoff):
    """Full driver under the frozen route ladder (SPEC.md section 2).

    A3 route when the A1 guard and the tail check both admit; A1 route
    when only the A1 guard admits; otherwise the compact route (same
    schedule; the engine's fallback kernel differs in code shape only —
    the A1I/A3I proofs and the conformance tests pin schedule
    equivalence across all three routes).  Fold arithmetic per
    ``plan.accessibility.fold.reach_gravity_knn.v1`` (its membership
    stays IEEE — NaN distances are excluded — while the search's
    scalar comparisons are fast; module docstring).

    Typed refusal (SPEC.md section 7): non-float64 origin-terminal
    weights are refused before any work — no engine route executes
    that dtype domain (the guard refuses it and the fallback cannot
    compile its mixed-dtype seed heap), so there is nothing to mirror
    and a ValueError is the honest contract.
    """
    if o_terminal_weights.dtype != np.float64:
        raise ValueError(
            "origin-terminal weights must be float64 (engine-supported "
            f"dtype domain), got {o_terminal_weights.dtype}")
    o_count = o_terminal_idxs.shape[0]
    n_count = adjacency_pointer.shape[0] - 1
    d_count = d_terminal_weights.shape[0]

    reach = np.empty(o_count, dtype=o_terminal_idxs.dtype)
    gravity_exponential = np.empty(o_count, dtype=adjacency_vector_weights.dtype)
    gravity_logistic = np.empty(o_count, dtype=adjacency_vector_weights.dtype)
    knn_access = np.empty(o_count, dtype=adjacency_vector_weights.dtype)

    a1_admitted, _ = scope_route_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        flag, o_terminal_idxs, o_terminal_weights, cutoff)

    if a1_admitted and tail_admits(d_terminal_idxs, d_count, n_count):
        route = ROUTE_A3
        domain = "a3"
        d_count_tail = 0
    elif a1_admitted:
        route = ROUTE_A1
        domain = "a1"
        d_count_tail = d_count
    else:
        route = ROUTE_COMPACT
        domain = "a1"
        d_count_tail = d_count

    for o in range(o_count):
        labels = scope_search_schedule(
            o_terminal_idxs[o], o_terminal_weights[o],
            adjacency_pointer, adjacency_vector,
            adjacency_vector_weights, flag, cutoff,
            d_count=d_count_tail, domain=domain)
        d_distance = adjust_destination_distances_plan(
            labels, d_terminal_idxs, d_terminal_weights)
        # una_legacy parity: the LEGACY (njit fastmath) fold — the
        # plain-numpy fold rounds the decay aggregates differently than
        # the engine's fastmath compilation (1-ulp class, characterized
        # in reductions.py); the oracle's contract is the engine's bits.
        r, ge, gl, ka = fold_reach_gravity_knn_legacy_plan(
            d_distance, d_weights, cutoff, gravity_beta, gravity_plateau,
            gravity_logistic_midpoint, gravity_growth_rate, knn_decay,
            knn_weights)
        reach[o] = r
        gravity_exponential[o] = ge
        gravity_logistic[o] = gl
        knn_access[o] = ka

    return ScopeAccessResult(reach=reach,
                             gravity_exponential=gravity_exponential,
                             gravity_logistic=gravity_logistic,
                             knn_access=knn_access,
                             route=route)
