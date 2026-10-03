"""Private epoch-labeled sparse scope search (campaign una_10x, task S11).

Implements dossiers 02 (`campaigns/una_10x/dossiers/02_epoch_sparse_search.md`)
and 03 (`03_local_destinations_and_dispatch.md`) over the constructions
proven in S10 (`campaigns/una_10x/evidence/proofs/S10_sparse_projection/`):

- per-epoch touched filter: a logical label read is ``label[v]`` when
  ``stamp[v] == epoch``, else the original typed initializer ``b`` —
  reconstruction NEVER gathers a cumulative touched list (review finding
  M1, regression-pinned by probe E7b);
- seeds are unconditional epoch writes (start then end; same-node
  last-wins preserves the original overwrite behavior), and seed nodes
  not expanded are still touched and enumerated (dossier 03);
- no dense label array is ever materialized: the only per-origin O(V)
  cost paid once per lane is the stamp zeroing, and every label read
  goes through the projection (no accidental loads of uninitialized
  label memory — the raw-arena exposure law from S10);
- candidate destinations are deduplicated per epoch via destination
  marks over the reverse incidence CSR and enumerated in ascending
  ORIGINAL destination index (not discovery order, not distance);
- the original compiled fold (`reach_gravity_knn_access`) is called
  UNCHANGED on the candidate prefix, with the baseline a3 route's exact
  argument binding.  The omitted-evaluation proof (dossier 03) holds by
  admission: a destination with neither terminal touched has both
  logical labels equal to ``b``, so ``dist >= b > R`` and the original
  membership ``<= R`` excludes it exactly.

ADMITTED DOMAIN (narrow; every refusal falls back to the unchanged
original driver — no public ODM shrink): the verbatim engine A1 typed
gate + ordered value scan (`_a1_scope_admits`), the verbatim tail guard
(`_a3_tail_admits`, so the sparse route only serves the tail-free A3
domain), plus the dossier-02 value conditions — origin seed weights,
destination terminal weights and destination weights finite and
nonnegative (a -0.0 bit is not negative and is retained), ``b = 1.0 +
cutoff`` strictly greater than ``cutoff`` (the rounded-sentinel
refusal), arc-cost maximum plus cutoff finite (conservative
intermediate-addition bound for the search's ``cost + weight``
additions; push gates keep every heap weight below ``cutoff``), exact
``np.ndarray`` instances (no subclasses), C-contiguous layouts, and the
arena capacity cap (256 MiB, the engine scratch pattern).  The S10
engine findings F1/F2/F3 (NaN comparison laws) are out of scope here BY
REFUSAL, not by repair — NaN seeds/costs/weights refuse to the
unchanged fallback — and remain with their SPEC/engine owners.

First implementation discipline (dossier 02): arenas are per-call
(stamps and destination marks zeroed at allocation; a thrown exception
leaves nothing behind for the next caller), the epoch counter is
checked against ``epoch_rollover`` and stamps+marks are cleared BEFORE
rollover (epochs never used as float), and ``prange`` runs over LOGICAL
LANES (lane id is not a thread id), each lane owning ``label[V]``,
``stamp[V]``, ``touched[V]``, ``row_offset[max_degree]``,
``row_cost[max_degree]``, ``ids[D]``, ``mark[D]``, ``dist[D]``,
``wts[D]`` and sequentially processing its assigned origins
``lane, lane + lanes, ...``.  Counters report actual work (touched
nodes, candidates, stamps zeroed vs the dense initialization avoided)
so any later dispatch (P00) can be evidenced, not asserted.
"""
from dataclasses import dataclass

import numba as nb
import numpy as np
from heapq import heappush, heappop

from ._large_access_scratch import (
    _a1_scope_admits,
    _a1_threads_bound,
    _a3_tail_admits,
)
from .AccessibilityWElevation import (
    integrated_scope_access,
    reach_gravity_knn_access,
)

# Numba JIT compilation settings: the search/enum kernel is a parallel
# driver kernel, so parallel=True is REQUIRED for the logical-lane
# prange (the engine drivers that prange hardcode parallel=True the
# same way); cache/nogil/fastmath follow the engine contract
# (AccessibilityWElevation.py lines 57-61).  fastmath is what produces
# the una_legacy comparison law both this kernel and the baseline are
# compiled under, so it must not be turned off here.
NUMBA_PARALLEL = True
NUMBA_CACHE = True
NUMBA_NOGIL = True
NUMBA_FASTMATH = True

# Arena capacity cap, identical pattern to the engine scratch caps
# (_A1_SCRATCH_CAP_BYTES): refuse the private route before allocation.
_SP_ARENA_CAP_BYTES = 268435456

# Default epoch ceiling: int64-safe, never reached by ordinary cohorts
# (the checked-counter branch clears stamps/marks BEFORE rollover;
# tests drive it with tiny values to exhaust epochs).
DEFAULT_EPOCH_ROLLOVER = 4611686018427387904  # 2**62

ROUTE_SPARSE = "sparse_epoch_a3"
ROUTE_FALLBACK = "compact_fallback"


@dataclass(frozen=True)
class SparseCounters:
    """Actual-work counters (dossier 03: prove work reduction, not entry).

    ``dense_init_avoided`` is the per-origin O(V) dense initialization
    the original driver performs (``origins * node_count`` label slots);
    ``stamps_zeroed`` is what the sparse route pays instead (lanes *
    (node_count + d_count) once per call).  Fallback runs report zeros.
    """

    lanes: int
    origins: int
    touched_total: int
    candidates_total: int
    stamps_zeroed: int
    dense_init_avoided: int


@dataclass(frozen=True)
class SparseScopeResult:
    """The four accessibility outputs plus route attribution/counters.

    Output dtypes mirror the baseline driver exactly: ``reach`` takes
    ``o_terminal_idxs.dtype`` (the fold's float sum is stored through
    the int64 array, as the baseline does) and the three float outputs
    take ``adjacency_vector_weights.dtype`` (float64 on the admitted
    sparse domain).  ``refusal`` is "" on the sparse route and the
    guard's reason code on fallback, so guard refusal and later
    performance fallback stay distinguishable in evidence.
    """

    reach: np.ndarray
    gravity_exponential: np.ndarray
    gravity_logistic: np.ndarray
    knn_access: np.ndarray
    route: str
    refusal: str
    counters: SparseCounters


def _sp_route_admits(adjacency_pointer, adjacency_vector,
                     adjacency_vector_weights, adjacynct_vector_network_node,
                     o_terminal_idxs, o_terminal_weights,
                     d_terminal_idxs, d_terminal_weights, d_weights, cutoff):
    """Sparse-route admission: engine gates verbatim + dossier-02 values.

    Returns ``(admitted, refusal_reason, max_degree)``.  Allocates
    nothing, casts nothing, raises nothing; the caller runs the
    unchanged original driver on every refusal.
    """
    # Exact instances and layouts first (dossier 02: "Start with no
    # object/subclass/unknown alias inputs"); dtype/shape/value
    # admission follows in the verbatim engine gates.
    for arr in (adjacency_pointer, adjacency_vector,
                adjacency_vector_weights, adjacynct_vector_network_node,
                o_terminal_idxs, o_terminal_weights,
                d_terminal_idxs, d_terminal_weights, d_weights):
        if type(arr) is not np.ndarray or not arr.flags.c_contiguous:
            return (False, "layout", 0)

    # Rank closure (S12 review F1/F2/F3): the engine gates read terminal
    # shape[0]/[1] and the fold indexes the destination weight vector as
    # 1-D, so a degenerate rank either crashes the kernel while the baseline
    # runs (F1: (D,1) d_weights) or raises IndexError from THIS gate instead
    # of the baseline's TypingError (F2: 0-d d_weights, F3: 1-D d_terminal_weights).
    # Refuse the whole class here — before any shape element access, so this
    # gate itself still raises nothing — and keep every such input on the
    # unchanged fallback, which is bitwise or exception-parity exact by
    # construction.
    if (d_weights.ndim != 1
            or adjacency_pointer.ndim != 1
            or adjacency_vector.ndim != 1
            or adjacency_vector_weights.ndim != 1
            or adjacynct_vector_network_node.ndim != 1
            or o_terminal_idxs.ndim != 2
            or o_terminal_weights.ndim != 2
            or d_terminal_idxs.ndim != 2
            or d_terminal_weights.ndim != 2):
        return (False, "shape", 0)

    admitted, max_degree = _a1_scope_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        adjacynct_vector_network_node, o_terminal_idxs, o_terminal_weights,
        cutoff)
    if not admitted:
        return (False, "a1_guard", 0)

    node_count = adjacency_pointer.shape[0] - 1
    d_count = d_terminal_weights.shape[0]
    if not _a3_tail_admits(d_terminal_idxs, d_count, node_count):
        return (False, "tail", 0)

    # Shape closure the engine gates do not cover: a mismatch would
    # crash the original driver downstream, so refuse it here and keep
    # the crash behavior in the fallback (exception parity).
    if d_terminal_idxs.shape[1] != 2 or d_terminal_weights.shape[1] != 2:
        return (False, "shape", 0)
    if d_weights.shape[0] != d_count:
        return (False, "shape", 0)
    if d_weights.dtype != np.float64:
        return (False, "dtype", 0)
    if d_terminal_weights.dtype != np.float64:
        return (False, "dtype", 0)

    # Dossier-02 value domain: finite nonnegative seeds, terminal costs
    # and initial destination weights (a -0.0 bit passes >= 0.0 and is
    # retained).  NaN/Inf/negative refuse to the fallback, which is the
    # S10-characterized NaN-law domain — NOT repaired here.
    for w in (o_terminal_weights, d_terminal_weights, d_weights):
        if not bool(np.isfinite(w).all()):
            return (False, "values", 0)
        if not bool((w >= 0.0).all()):
            return (False, "values", 0)

    # Original typed initializer: b = fl(1.0 + cutoff) computed exactly
    # as the dense init does (float64 ones + the original cutoff
    # representation, int64 promoted the same way).  The
    # omitted-evaluation proof needs b > R strictly.
    b = np.ones(1, dtype=np.float64)[0] + cutoff
    if not (b > cutoff):
        return (False, "sentinel", 0)

    # Conservative intermediate-addition bound: heap weights are
    # strictly below cutoff (push gates), so cost + weight stays below
    # edge_max + cutoff; refuse when that addition is not finite.
    if adjacency_vector_weights.shape[0]:
        edge_max = adjacency_vector_weights.max()
    else:
        edge_max = np.float64(0.0)
    if not (edge_max + cutoff < np.inf):
        return (False, "addition_bound", 0)

    lanes = int(_a1_threads_bound())
    if lanes < 1:
        return (False, "threads", 0)
    per_lane_elems = 3 * node_count + 4 * d_count + 2 * max_degree
    if 8 * lanes * per_lane_elems > _SP_ARENA_CAP_BYTES:
        return (False, "capacity", 0)

    return (True, "", max_degree)


@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
)
def _sp_epoch_scope_search(
    o_terminal_idxs, o_terminal_weights,
    adjacency_pointer, adjacency_vector, adjacency_vector_weights,
    adjacynct_vector_network_node, cutoff,
    d_terminal_idxs, d_terminal_weights, d_weights,
    gravity_beta, gravity_plateau, gravity_logistic_midpoint,
    gravity_growth_rate, knn_decay, knn_weights,
    reverse_pointer, reverse_vector,
    lane_label, lane_stamp, lane_touched,
    lane_row_offset, lane_row_cost,
    lane_ids, lane_mark, lane_dist, lane_wts,
    epoch_rollover,
    reach, gravity_exponential, gravity_logistic, knn_access,
    lane_counters,
):
    """Epoch-labeled two-phase search + indexed destination enumeration.

    The search body is the `_a3_scope_search_tailless` schedule with
    exactly two deltas: label initialization is the epoch projection
    (write = record touch once, set stamp, write value) and label reads
    go through ``stamp[v] == epoch`` / else ``b``.  Heap prologue
    (dummy init/pop, end-then-start strict ``<`` push gates), the
    two-phase row scan, the flag column gate and the leaf degree > 1
    push gate are byte-identical to the baseline kernel.  Enumeration
    and fold follow dossier 03's pseudocode.
    """
    lane_count = lane_label.shape[0]
    origin_count = o_terminal_idxs.shape[0]
    node_count = adjacency_pointer.shape[0] - 1
    d_count = d_terminal_weights.shape[0]

    for lane in nb.prange(lane_count):
        label = lane_label[lane]
        stamp = lane_stamp[lane]
        touched = lane_touched[lane]
        row_offset = lane_row_offset[lane]
        row_cost = lane_row_cost[lane]
        ids = lane_ids[lane]
        mark = lane_mark[lane]
        cand_dist = lane_dist[lane]
        cand_wts = lane_wts[lane]

        # Original typed initializer value (dossier 02): element value
        # of the dense `np.ones(V, float64) + cutoff` the baseline
        # allocates per origin.
        b = 1.0 + cutoff

        epoch = np.int64(0)
        touched_total = np.int64(0)
        cand_total = np.int64(0)
        for origin in range(lane, origin_count, lane_count):
            # Checked epoch counter: clear stamps and destination marks
            # BEFORE rollover; epochs are never used as float.
            if epoch >= epoch_rollover:
                for v in range(node_count):
                    stamp[v] = 0
                for d in range(d_count):
                    mark[d] = 0
                epoch = np.int64(0)
            epoch = epoch + np.int64(1)
            e = epoch
            touched_count = 0

            o_idx_start = o_terminal_idxs[origin, 0]
            o_idx_end = o_terminal_idxs[origin, 1]
            o_idx_start_weight = o_terminal_weights[origin, 0]
            o_idx_end_weight = o_terminal_weights[origin, 1]

            # Unconditional seed writes (start then end; same-node
            # last-wins), each recording the touch exactly once in
            # this epoch: check stamp, then set stamp, then value.
            if stamp[o_idx_start] != e:
                stamp[o_idx_start] = e
                touched[touched_count] = o_idx_start
                touched_count = touched_count + 1
            label[o_idx_start] = o_idx_start_weight
            if stamp[o_idx_end] != e:
                stamp[o_idx_end] = e
                touched[touched_count] = o_idx_end
                touched_count = touched_count + 1
            label[o_idx_end] = o_idx_end_weight

            # Original dummy heap init/pop and end-then-start strict <
            # push gates (NaN weights cannot occur: seeds are admitted
            # finite nonnegative — that domain is REFUSED, not repaired).
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

                # Phase one: eligibility against the pre-row logical
                # label snapshot (projection reads; no label writes),
                # collected in ascending row order.
                n_eligible = 0
                for i in range(node_end_pointer - node_start_pointer):
                    neighbor_weight = adjacency_vector_weights[node_start_pointer + i] + weight
                    neighbor_node = adjacency_vector[node_start_pointer + i]
                    if stamp[neighbor_node] == e:
                        neighbor_label = label[neighbor_node]
                    else:
                        neighbor_label = b
                    if neighbor_weight <= cutoff and neighbor_weight < neighbor_label:
                        row_offset[n_eligible] = node_start_pointer + i
                        row_cost[n_eligible] = neighbor_weight
                        n_eligible = n_eligible + 1

                # Phase two: assignments and pushes from the stored
                # pairs, in ascending row order, gated by the flag
                # column and the leaf-node degree gate (> 1).
                for j in range(n_eligible):
                    offset = row_offset[j]
                    neighbor_weight = row_cost[j]
                    neighbor_node = adjacency_vector[offset]
                    if stamp[neighbor_node] != e:
                        stamp[neighbor_node] = e
                        touched[touched_count] = neighbor_node
                        touched_count = touched_count + 1
                    label[neighbor_node] = neighbor_weight
                    if adjacynct_vector_network_node[offset]:
                        if adjacency_pointer[neighbor_node + 1] - adjacency_pointer[neighbor_node] > 1:
                            heappush(queue, (neighbor_weight, neighbor_node))

            # Dossier 03: reverse-index candidate discovery over the
            # PER-EPOCH touched set, deduplicated by destination mark.
            cand_count = 0
            for t in range(touched_count):
                tv = touched[t]
                for k in range(reverse_pointer[tv], reverse_pointer[tv + 1]):
                    d = reverse_vector[k]
                    if mark[d] != e:
                        mark[d] = e
                        ids[cand_count] = d
                        cand_count = cand_count + 1

            # Ascending ORIGINAL destination index (contract order —
            # ids are unique, so the sort is total regardless of kind).
            if cand_count > 0:
                order = np.argsort(ids[:cand_count])
                for j in range(cand_count):
                    d = ids[order[j]]
                    t0 = d_terminal_idxs[d, 0]
                    t1 = d_terminal_idxs[d, 1]
                    if stamp[t0] == e:
                        lw0 = label[t0]
                    else:
                        lw0 = b
                    if stamp[t1] == e:
                        lw1 = label[t1]
                    else:
                        lw1 = b
                    dist_to_start = lw0 + d_terminal_weights[d, 0]
                    dist_to_end = lw1 + d_terminal_weights[d, 1]
                    cand_dist[j] = min(dist_to_start, dist_to_end)
                    cand_wts[j] = d_weights[d]

            # Original compiled fold, unchanged, on the candidate
            # prefix with the baseline a3 route's exact binding.
            reach[origin], gravity_exponential[origin], gravity_logistic[origin], knn_access[origin] = reach_gravity_knn_access(
                d_distance=cand_dist[:cand_count],
                d_weights=cand_wts[:cand_count],
                cutoff=cutoff,
                gravity_beta=gravity_beta,
                gravity_plateau=gravity_plateau,
                gravity_logistic_midpoint=gravity_logistic_midpoint,
                gravity_growth_rate=gravity_growth_rate,
                knn_gravity_plateau=gravity_plateau,
                knn_weights=knn_weights,
                knn_gravity_beta=gravity_beta,
                knn_decay=knn_decay,
                knn_gravity_logistic_midpoint=gravity_logistic_midpoint,
                knn_gravity_growth_rate=gravity_growth_rate,
            )


            touched_total = touched_total + touched_count
            cand_total = cand_total + cand_count

        lane_counters[lane, 0] = touched_total
        lane_counters[lane, 1] = cand_total


def sparse_scope_access(o_terminal_idxs, o_terminal_weights,
                        adjacency_pointer, adjacency_vector,
                        adjacency_vector_weights, adjacynct_vector_network_node,
                        d_terminal_idxs, d_terminal_weights, d_weights,
                        gravity_beta, gravity_plateau,
                        gravity_logistic_midpoint, gravity_growth_rate,
                        knn_decay, knn_weights, cutoff,
                        epoch_rollover=DEFAULT_EPOCH_ROLLOVER):
    """Private epoch-sparse integrated scope access (dossiers 02+03).

    Admitted inputs run the logical-lane epoch kernel; every refusal
    calls the UNCHANGED original driver (`integrated_scope_access`) —
    the public compact/full-array behavior is not touched.  Returns a
    `SparseScopeResult` whose four arrays are bitwise-comparable to the
    baseline driver's outputs on every input (route and counters
    attribute which path ran).
    """
    admitted, refusal, max_degree = _sp_route_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        adjacynct_vector_network_node, o_terminal_idxs, o_terminal_weights,
        d_terminal_idxs, d_terminal_weights, d_weights, cutoff)

    origin_count = o_terminal_idxs.shape[0]
    if not admitted:
        reach, gravity_exponential, gravity_logistic, knn_access = integrated_scope_access(
            o_terminal_idxs, o_terminal_weights,
            adjacency_pointer, adjacency_vector, adjacency_vector_weights,
            adjacynct_vector_network_node, d_terminal_idxs, d_terminal_weights,
            d_weights, gravity_beta, gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, cutoff)
        return SparseScopeResult(
            reach=reach, gravity_exponential=gravity_exponential,
            gravity_logistic=gravity_logistic, knn_access=knn_access,
            route=ROUTE_FALLBACK, refusal=refusal,
            counters=SparseCounters(0, 0, 0, 0, 0, 0))

    lanes = int(_a1_threads_bound())
    node_count = adjacency_pointer.shape[0] - 1
    d_count = d_terminal_weights.shape[0]

    # Reverse incidence CSR (dossier 03): terminal node -> original
    # destination index, built from BOTH terminal columns with
    # duplicates kept; checked prefix sums.  O(D log D), charged.
    if d_count:
        cols = np.concatenate((d_terminal_idxs[:, 0], d_terminal_idxs[:, 1]))
        dest = np.concatenate((np.arange(d_count, dtype=np.int64),
                               np.arange(d_count, dtype=np.int64)))
        reverse_vector = np.ascontiguousarray(dest[np.argsort(cols, kind="stable")])
        counts = np.bincount(cols, minlength=node_count)
    else:
        reverse_vector = np.empty(0, dtype=np.int64)
        counts = np.zeros(node_count, dtype=np.int64)
    reverse_pointer = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(counts, out=reverse_pointer[1:])
    if int(reverse_pointer[-1]) != 2 * d_count:
        raise ValueError("reverse index prefix sum mismatch "
                         f"({int(reverse_pointer[-1])} != {2 * d_count})")

    # Per-call logical-lane arenas: stamps and destination marks zeroed
    # here (the initialization the timing must charge); labels/touched/
    # rows/candidates are write-before-read by construction.
    lane_label = np.empty((lanes, node_count), dtype=np.float64)
    lane_stamp = np.zeros((lanes, node_count), dtype=np.int64)
    lane_touched = np.empty((lanes, node_count), dtype=np.int64)
    lane_row_offset = np.empty((lanes, max_degree), dtype=np.int64)
    lane_row_cost = np.empty((lanes, max_degree), dtype=np.float64)
    lane_ids = np.empty((lanes, d_count), dtype=np.int64)
    lane_mark = np.zeros((lanes, d_count), dtype=np.int64)
    lane_dist = np.empty((lanes, d_count), dtype=np.float64)
    lane_wts = np.empty((lanes, d_count), dtype=np.float64)

    reach = np.empty(origin_count, dtype=o_terminal_idxs.dtype)
    gravity_exponential = np.empty(origin_count, dtype=adjacency_vector_weights.dtype)
    gravity_logistic = np.empty(origin_count, dtype=adjacency_vector_weights.dtype)
    knn_access = np.empty(origin_count, dtype=adjacency_vector_weights.dtype)
    lane_counters = np.zeros((lanes, 2), dtype=np.int64)

    _sp_epoch_scope_search(
        o_terminal_idxs, o_terminal_weights,
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        adjacynct_vector_network_node, cutoff,
        d_terminal_idxs, d_terminal_weights, d_weights,
        gravity_beta, gravity_plateau, gravity_logistic_midpoint,
        gravity_growth_rate, knn_decay, knn_weights,
        reverse_pointer, reverse_vector,
        lane_label, lane_stamp, lane_touched,
        lane_row_offset, lane_row_cost,
        lane_ids, lane_mark, lane_dist, lane_wts,
        epoch_rollover,
        reach, gravity_exponential, gravity_logistic, knn_access,
        lane_counters,
    )

    counters = SparseCounters(
        lanes=lanes,
        origins=origin_count,
        touched_total=int(lane_counters[:, 0].sum()),
        candidates_total=int(lane_counters[:, 1].sum()),
        stamps_zeroed=lanes * (node_count + d_count),
        dense_init_avoided=origin_count * node_count,
    )
    return SparseScopeResult(
        reach=reach, gravity_exponential=gravity_exponential,
        gravity_logistic=gravity_logistic, knn_access=knn_access,
        route=ROUTE_SPARSE, refusal="", counters=counters)
