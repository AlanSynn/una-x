"""Private packed ordered routing trace store (campaign una_10x, C11).

Dossier 04 (`campaigns/una_10x/dossiers/04_routing_traces.md`): compute
routing ONCE for distinct metric jobs with identical prepared search
inputs, as a packed origin-major sparse trace over ORIGINAL destination
indices, with complete typed identity and honest status.

- Identity is the C10 layered spec (schema-ordered canonical frames,
  bit-exact floats): GraphID -> QueryID -> DestID -> TraceID.  Metric
  parameters and destination METRIC weights are deliberately NOT trace
  identity (MetricID owns them); one trace serves every metric
  variation of the same query.  Provenance never enters an ID.
- The search body is the S11 epoch kernel's schedule (`_sparse_access_
  scratch._sp_epoch_scope_search`, itself proven bitwise against the
  compiled baseline) with exactly two deltas: candidates are captured
  per origin into tile buffers in the SAME ascending-original-id order
  the fold consumed, and the origin loop runs over a bounded producer
  tile.  Admission is the S11 gate verbatim (imported, not copied) so
  the trace route serves exactly the S11-admitted domain.
- Trace bytes are honest: ``indptr[O+1]``, ``destination_ids[nnz]``
  int64, ``distance_bits[nnz]`` float64 of the ACTUAL evaluated
  retained distances, immutable (read-only arrays, frozen record).  The
  ONLY temporary is the tile-local candidate capture, exactly
  ``tile_origins * d_count`` lanes wide (int64 ids + float64 distances
  + int64 counts) with ``tile_origins`` clamped to
  ``max(1, min(tile_origins, origin_count))`` — so a caller may raise
  the tile width up to one tile over all origins (O*D capture), but
  the default (64) stays bounded independent of O; the capture is
  measured either way (C11-NOTE-2 law: the bound is the caller's
  tile_origins, never hidden).
- Exceeding the trace byte budget marks the record INCOMPLETE and
  selects recompute/streaming — a partial trace can never be published
  as a cache hit and ``replay_trace`` refuses it by status.
- ``replay_trace`` re-runs the engine's OWN compiled fold per origin on
  the trace's retained prefix (distances + freshly gathered metric
  weights) with the baseline a3 route's exact binding, and refuses an
  invalid metric input (NaN/negative weights, cutoff/dtype mismatch
  with the trace identity, wrong destination count) EVEN when a trace
  exists — no invalid input becomes successful because an earlier
  trace hit avoided its validation.
- A handle is not a durable cache identity and a pointer address is
  not a hash: equality of traces is bytewise, never object identity.

Public API is untouched; this module is a private facade for the C11
suite and later campaign tasks (C12 reuse qualification, C20
residency).
"""
from dataclasses import dataclass
import hashlib
import inspect

import numba as nb
import numpy as np
from heapq import heappush, heappop

from ._sparse_access_scratch import (
    DEFAULT_EPOCH_ROLLOVER,
    _a1_threads_bound,
)
import urban_network_analysis.Engines._sparse_access_scratch as _sp_mod
from .AccessibilityWElevation import reach_gravity_knn_access

# The admission gate is reused VERBATIM from the S11 module (one law,
# no drift).  Resolved by name with an arity check at import: a wrong
# name or signature fails loudly here, not silently in production.
_sp_route_admits = getattr(_sp_mod, "_sp_route_admits")
assert len(inspect.signature(_sp_route_admits).parameters) == 10, \
    "S11 admission gate signature drifted; refusing to guess"

NUMBA_PARALLEL = True
NUMBA_CACHE = True
NUMBA_NOGIL = True
NUMBA_FASTMATH = True

TRACE_SCHEMA_VERSION = "una-10x-c11-trace-1"
ROUTE_TRACE = "trace_epoch_a3"
ROUTE_BUDGET = "trace_budget_exceeded"

# Bounded producer tile (dossier 04: one bounded producer tile at a
# time; tiles are measured and charged, never silently grown).
DEFAULT_TILE_ORIGINS = 64
# Default trace budget: bytes of the PACKED trace (indptr + ids + dist
# plus fixed metadata).  Exceeding it selects recompute/streaming.
DEFAULT_TRACE_BUDGET_BYTES = 1 << 30

_SENTINEL_LAW = "sentinel b = 1.0 + cutoff, rounded-sentinel refusal"
_SEARCH_LAW = "epoch-labeled two-phase a3-tailless schedule, " \
              "reverse-index candidate enumeration, ascending " \
              "original destination id"
_EXPANSION_LAW = "end-then-start strict < push gates, flag column " \
                 "gate, leaf degree > 1 gate"
_EXTRACTION_LAW = "reverse incidence, both terminal columns, " \
                  "duplicates kept, per-epoch mark dedup"
_MIN_LAW = "dist = min(lw0 + tw0, lw1 + tw1)"
_MEMBERSHIP_LAW = "engine fold on the candidate prefix; " \
                  "retained = d_distance <= cutoff"


# --------------------------------------------------------------------------
# Canonical frame law (runtime encoder).
#
# Same law as the C10 reference encoder (tests/tenx/identity/
# _canonical.py): fields in SCHEMA ORDER (never sorted), str = u64len +
# utf8, int = 8-byte LE signed, float = 8-byte IEEE-754 bit pattern (bit
# identity: -0.0 != 0.0, every 1-ULP change visible), bytes = u64len +
# raw, bool = 1 byte, list = u64count + ordered frames, LIST ONLY (a
# tuple value is refused), field = namelen + name + value frame, ID =
# sha256(field frame).hexdigest().  The encoder never wraps silently:
# an int outside the signed 64-bit range raises struct.error and an
# unframed type raises TypeError — every encoder refusal is an
# admission refusal, never truncated or wrapped data.
# --------------------------------------------------------------------------
def _u64(n):
    import struct
    return struct.pack("<Q", n)


def _frame(v):
    import struct
    if isinstance(v, bool):
        return b"B" + bytes([1 if v else 0])
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"s" + _u64(len(b)) + b
    if isinstance(v, int):
        return b"i" + struct.pack("<q", v)
    if isinstance(v, float):
        return b"f" + struct.pack("<d", v)
    if isinstance(v, bytes):
        return b"b" + _u64(len(v)) + v
    if isinstance(v, list):
        return b"L" + _u64(len(v)) + b"".join(_frame(e) for e in v)
    raise TypeError("unframed type: %r" % (type(v),))


def _frame_fields(pairs):
    out = []
    for name, value in pairs:
        nb_b = name.encode("utf-8")
        out.append(b"F" + _u64(len(nb_b)) + nb_b + _frame(value))
    return b"".join(out)


def _identity(pairs):
    return hashlib.sha256(_frame_fields(pairs)).hexdigest()


def _module_sha256():
    h = hashlib.sha256()
    with open(__file__, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_trace_id(o_terminal_idxs, o_terminal_weights,
                     adjacency_pointer, adjacency_vector,
                     adjacency_vector_weights,
                     adjacency_vector_network_node,
                     d_terminal_idxs, d_terminal_weights, cutoff,
                     epoch_rollover, profile_id="una_legacy-epoch-a3"):
    """The C10 layered TraceID over exactly the search-defining inputs.

    Metric parameters and destination METRIC weights are excluded by
    construction (MetricID owns them): one trace serves every metric
    variation of the same query.  Provenance (paths, timestamps,
    acquisition metadata) has no field to enter.
    """
    graph_id = _identity([
        ("profile_id", str(profile_id)),
        ("csr_indptr_bits", np.ascontiguousarray(
            adjacency_pointer, dtype=np.int64).tobytes()),
        ("csr_neighbor_ids_bits", np.ascontiguousarray(
            adjacency_vector, dtype=np.int64).tobytes()),
        ("csr_cost_bits", np.ascontiguousarray(
            adjacency_vector_weights).tobytes()),
        ("csr_flags_bits", np.ascontiguousarray(
            adjacency_vector_network_node).astype(np.uint8).tobytes()),
        ("prepared_topology_semantics",
         "csr adjacency, two-terminal origins/destinations"),
        ("turn_semantics", "not-present-in-prepared-domain"),
        ("obstacle_semantics", "flag column as prepared"),
        ("elevation_semantics", "not-present-in-prepared-domain"),
        ("compiler_search_plan", _SEARCH_LAW + "; " + _EXPANSION_LAW),
        ("numerical_env", "numba " + nb.__version__ + "; trace module "
                          "sha256 " + _module_sha256()),
    ])
    query_id = _identity([
        ("graph_id", graph_id),
        ("origin_seed_ids_ordered", [int(v) for v in
                                     np.ascontiguousarray(
                                         o_terminal_idxs,
                                         dtype=np.int64).ravel()]),
        ("seed_cost_bits_ordered", [float(v) for v in
                                    np.ascontiguousarray(
                                        o_terminal_weights).ravel()]),
        ("cutoff_value", float(cutoff)),
        ("cutoff_dtype", str(np.dtype(getattr(cutoff, "dtype",
                                              type(cutoff))))),
        ("search_policy", _SEARCH_LAW),
        ("sentinel_policy", _SENTINEL_LAW + "; epoch rollover %d"
                           % int(epoch_rollover)),
        ("expansion_policy", _EXPANSION_LAW),
    ])
    dest_id = _identity([
        ("dest_terminal_ids_ordered", np.ascontiguousarray(
            d_terminal_idxs, dtype=np.int64).tobytes()),
        ("dest_cost_bits_ordered", np.ascontiguousarray(
            d_terminal_weights).tobytes()),
        ("extraction_plan", _EXTRACTION_LAW),
        ("min_plan", _MIN_LAW),
        ("membership_plan", _MEMBERSHIP_LAW),
    ])
    return _identity([
        ("query_id", query_id),
        ("dest_id", dest_id),
        ("trace_schema_version", TRACE_SCHEMA_VERSION),
    ])


# --------------------------------------------------------------------------
# Status, counters and the frozen immutable record
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TraceCounters:
    """Actual work and bytes, measured (dossier 04 charge law)."""

    origins: int
    tiles: int
    tile_origins: int
    candidates_total: int
    touched_total: int
    indptr_bytes: int
    ids_bytes: int
    dist_bytes: int
    metadata_bytes: int
    trace_bytes: int
    tile_temp_bytes: int
    stamps_zeroed: int


@dataclass(frozen=True)
class TraceRecord:
    """Packed origin-major trace: immutable bytewise record.

    Arrays are read-only; the record is frozen.  Equality of two
    records is bytewise on the packed arrays plus identity fields — a
    handle is not a durable cache identity and a pointer address is
    not a hash.  ``complete`` is False for refused/budget-exceeded
    builds; an incomplete record carries NO published trace status and
    ``replay_trace`` refuses it.
    """

    trace_id: str
    schema_version: str
    origin_count: int
    dest_count: int
    cutoff_value: float
    cutoff_dtype: str
    reach_dtype: str
    metric_dtype: str
    indptr: np.ndarray
    destination_ids: np.ndarray
    distance_bits: np.ndarray
    complete: bool
    route: str
    refusal: str
    counters: TraceCounters
    # C11-NOTE-4 (bound at build, verified at every replay gate and at
    # residency publish/load): sha256 over the packed bytes in layout
    # order, so a forged bit-flip in a retained record is refused
    # instead of silently replayed.  Empty for INCOMPLETE records.
    content_digest: str = ""


@dataclass(frozen=True)
class TraceReplayResult:
    """Fresh per-job outputs replayed from a complete trace."""

    reach: np.ndarray
    gravity_exponential: np.ndarray
    gravity_logistic: np.ndarray
    knn_access: np.ndarray
    route: str
    refusal: str
    trace_id: str
    folds_done: int


def _read_only(a):
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _reverse_incidence(d_terminal_idxs, node_count):
    """Reverse CSR terminal node -> original destination index.

    Built from BOTH terminal columns with duplicates kept, checked
    prefix sums (the S11 construction, dossier 03).
    """
    d_count = d_terminal_idxs.shape[0]
    if d_count:
        cols = np.concatenate((d_terminal_idxs[:, 0],
                               d_terminal_idxs[:, 1]))
        dest = np.concatenate((np.arange(d_count, dtype=np.int64),
                               np.arange(d_count, dtype=np.int64)))
        reverse_vector = np.ascontiguousarray(
            dest[np.argsort(cols, kind="stable")])
        counts = np.bincount(cols, minlength=node_count)
    else:
        reverse_vector = np.empty(0, dtype=np.int64)
        counts = np.zeros(node_count, dtype=np.int64)
    reverse_pointer = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(counts, out=reverse_pointer[1:])
    if int(reverse_pointer[-1]) != 2 * d_count:
        raise ValueError("reverse index prefix sum mismatch "
                         f"({int(reverse_pointer[-1])} != {2 * d_count})")
    return reverse_pointer, reverse_vector


# --------------------------------------------------------------------------
# Producer kernel: the S11 epoch schedule + per-origin capture
# --------------------------------------------------------------------------
@nb.njit(
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
    nogil=NUMBA_NOGIL,
    fastmath=NUMBA_FASTMATH,
)
def _tr_epoch_scope_search_tile(
    origin_start,
    o_terminal_idxs,
    o_terminal_weights,
    adjacency_pointer,
    adjacency_vector,
    adjacency_vector_weights,
    adjacynct_vector_network_node,
    cutoff,
    d_terminal_idxs,
    d_terminal_weights,
    d_weights,
    reverse_pointer,
    reverse_vector,
    lane_label,
    lane_stamp,
    lane_touched,
    lane_row_offset,
    lane_row_cost,
    lane_ids,
    lane_mark,
    lane_dist,
    lane_wts,
    epoch_rollover,
    lane_counters,
    tile_ids,
    tile_dist,
    tile_counts,
):
    """One bounded producer tile (C11): the S11 epoch schedule
    with per-origin capture; the metric fold is NOT run at
    build (replay owns it).  Generated from S11 bytes."""
    tile_len = tile_counts.shape[0]
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
        # Tile-start clearing: stale epoch stamps/marks from
        # earlier tiles must never alias this tile's epoch.
        for v in range(node_count):
            stamp[v] = 0
        for d in range(d_count):
            mark[d] = 0

        epoch = np.int64(0)
        touched_total = np.int64(0)
        cand_total = np.int64(0)
        for t_origin in range(lane, tile_len, lane_count):
            origin = origin_start + t_origin
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
            tile_counts[t_origin] = cand_count
            ord2 = np.argsort(ids[:cand_count])
            for j in range(cand_count):
                tile_ids[t_origin, j] = ids[ord2[j]]
                tile_dist[t_origin, j] = cand_dist[j]


            touched_total = touched_total + touched_count
            cand_total = cand_total + cand_count

        lane_counters[lane, 0] = touched_total
        lane_counters[lane, 1] = cand_total


# --------------------------------------------------------------------------
# build_trace: bounded producer tiles, honest status, measured bytes
# --------------------------------------------------------------------------
def build_trace(o_terminal_idxs, o_terminal_weights,
                adjacency_pointer, adjacency_vector,
                adjacency_vector_weights, adjacency_vector_network_node,
                d_terminal_idxs, d_terminal_weights, d_weights, cutoff,
                epoch_rollover=DEFAULT_EPOCH_ROLLOVER,
                trace_budget_bytes=DEFAULT_TRACE_BUDGET_BYTES,
                tile_origins=DEFAULT_TILE_ORIGINS,
                profile_id="una_legacy-epoch-a3"):
    """Build the packed ordered trace over admitted inputs.

    Refused inputs produce an INCOMPLETE record with the S11 gate's
    reason code and no trace arrays (route attribution tells the
    caller to use the unchanged S11/public path).  Admitted inputs run
    the bounded producer tile loop; exceeding ``trace_budget_bytes``
    stops the build with ``complete=False`` and route
    ``trace_budget_exceeded`` (recompute/streaming selection — never a
    truncated trace success).
    """
    origin_count = int(o_terminal_idxs.shape[0])
    node_count = int(adjacency_pointer.shape[0]) - 1
    d_count = int(d_terminal_weights.shape[0])
    cutoff_value = float(cutoff)
    cutoff_dtype = str(np.dtype(getattr(cutoff, "dtype", type(cutoff))))
    reach_dtype = str(np.dtype(o_terminal_idxs.dtype))
    metric_dtype = str(np.dtype(adjacency_vector_weights.dtype))
    trace_id = compute_trace_id(
        o_terminal_idxs, o_terminal_weights, adjacency_pointer,
        adjacency_vector, adjacency_vector_weights,
        adjacency_vector_network_node, d_terminal_idxs,
        d_terminal_weights, cutoff, epoch_rollover,
        profile_id=profile_id)

    def _incomplete(route, refusal, counters):
        return TraceRecord(
            trace_id=trace_id, schema_version=TRACE_SCHEMA_VERSION,
            origin_count=origin_count, dest_count=d_count,
            cutoff_value=cutoff_value, cutoff_dtype=cutoff_dtype,
            reach_dtype=reach_dtype, metric_dtype=metric_dtype,
            indptr=_read_only(np.zeros(origin_count + 1, np.int64)),
            destination_ids=_read_only(np.zeros(0, np.int64)),
            distance_bits=_read_only(np.zeros(0, np.float64)),
            complete=False, route=route, refusal=refusal,
            counters=counters, content_digest="")

    admitted, refusal, max_degree = _sp_route_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        adjacency_vector_network_node, o_terminal_idxs,
        o_terminal_weights, d_terminal_idxs, d_terminal_weights,
        d_weights, cutoff)
    if not admitted:
        return _incomplete("compact_fallback", refusal, TraceCounters(
            origins=origin_count, tiles=0, tile_origins=0,
            candidates_total=0, touched_total=0, indptr_bytes=0,
            ids_bytes=0, dist_bytes=0, metadata_bytes=0, trace_bytes=0,
            tile_temp_bytes=0, stamps_zeroed=0))

    tile_origins = max(1, min(int(tile_origins), origin_count))
    reverse_pointer, reverse_vector = _reverse_incidence(
        d_terminal_idxs, node_count)

    # Lane arenas sized ONCE for the tile (bounded producer tile);
    # stamps and destination marks are cleared per tile before use and
    # the clearing is charged (the S11 per-call discipline, hoisted out
    # of the tile loop).
    lanes = max(1, min(int(_a1_threads_bound()), tile_origins))
    lane_label = np.empty((lanes, node_count), dtype=np.float64)
    lane_stamp = np.zeros((lanes, node_count), dtype=np.int64)
    lane_touched = np.empty((lanes, node_count), dtype=np.int64)
    lane_row_offset = np.empty((lanes, max_degree), dtype=np.int64)
    lane_row_cost = np.empty((lanes, max_degree), dtype=np.float64)
    lane_ids = np.empty((lanes, d_count), dtype=np.int64)
    lane_mark = np.zeros((lanes, d_count), dtype=np.int64)
    lane_dist = np.empty((lanes, d_count), dtype=np.float64)
    lane_wts = np.empty((lanes, d_count), dtype=np.float64)

    arena_bytes = (lane_label.nbytes + lane_stamp.nbytes +
                   lane_touched.nbytes + lane_row_offset.nbytes +
                   lane_row_cost.nbytes + lane_ids.nbytes +
                   lane_mark.nbytes + lane_dist.nbytes +
                   lane_wts.nbytes)

    indptr = np.zeros(origin_count + 1, dtype=np.int64)
    ids_parts = []
    dist_parts = []
    candidates_total = 0
    touched_total = 0
    stamps_zeroed = 0
    tiles = 0
    tile_temp_peak = 0
    budget_hit = None
    # The identity string is part of the record's payload and is charged
    # honestly (dossier 04: measured byte counts).
    metadata_bytes = len(trace_id.encode("utf-8"))

    def _trace_bytes(nnz_bytes):
        return indptr.nbytes + nnz_bytes + metadata_bytes

    for start in range(0, origin_count, tile_origins):
        tile_len = min(tile_origins, origin_count - start)
        tile_ids = np.zeros((tile_len, d_count), dtype=np.int64)
        tile_dist = np.zeros((tile_len, d_count), dtype=np.float64)
        tile_counts = np.zeros(tile_len, dtype=np.int64)
        lane_counters = np.zeros((lanes, 2), dtype=np.int64)
        # C11-NOTE-1 repair: the dead per-tile fold-output buffers are
        # gone (build never runs the metric fold — replay owns it);
        # only the capture payload is allocated and charged.
        tile_temp = tile_ids.nbytes + tile_dist.nbytes + \
            tile_counts.nbytes + lane_counters.nbytes
        tile_temp_peak = max(tile_temp_peak, tile_temp + arena_bytes)

        _tr_epoch_scope_search_tile(
            origin_start=start,
            o_terminal_idxs=o_terminal_idxs,
            o_terminal_weights=o_terminal_weights,
            adjacency_pointer=adjacency_pointer,
            adjacency_vector=adjacency_vector,
            adjacency_vector_weights=adjacency_vector_weights,
            adjacynct_vector_network_node=adjacency_vector_network_node,
            cutoff=cutoff,
            d_terminal_idxs=d_terminal_idxs,
            d_terminal_weights=d_terminal_weights,
            d_weights=d_weights,
            reverse_pointer=reverse_pointer,
            reverse_vector=reverse_vector,
            lane_label=lane_label,
            lane_stamp=lane_stamp,
            lane_touched=lane_touched,
            lane_row_offset=lane_row_offset,
            lane_row_cost=lane_row_cost,
            lane_ids=lane_ids,
            lane_mark=lane_mark,
            lane_dist=lane_dist,
            lane_wts=lane_wts,
            epoch_rollover=epoch_rollover,
            lane_counters=lane_counters,
            tile_ids=tile_ids,
            tile_dist=tile_dist,
            tile_counts=tile_counts,
        )
        tiles += 1
        stamps_zeroed += lanes * (node_count + d_count)
        counts = tile_counts.astype(np.int64)
        indptr[start + 1:start + tile_len + 1] = \
            indptr[start] + np.cumsum(counts)
        for t in range(tile_len):
            c = int(counts[t])
            if c:
                ids_parts.append(tile_ids[t, :c].copy())
                dist_parts.append(tile_dist[t, :c].copy())
        candidates_total += int(counts.sum())
        touched_total += int(lane_counters[:, 0].sum())

        trace_bytes = _trace_bytes(16 * candidates_total)
        if trace_bytes > int(trace_budget_bytes):
            # Dossier 04: exceeding the budget selects
            # recompute/streaming, never truncated-trace success.
            budget_hit = start + tile_len
            break

    if budget_hit is not None:
        return _incomplete(
            ROUTE_BUDGET,
            "trace budget exceeded at origin %d" % budget_hit,
            TraceCounters(
                origins=origin_count, tiles=tiles,
                tile_origins=tile_origins,
                candidates_total=candidates_total,
                touched_total=touched_total,
                indptr_bytes=indptr.nbytes,
                ids_bytes=8 * candidates_total,
                dist_bytes=8 * candidates_total,
                metadata_bytes=metadata_bytes,
                trace_bytes=_trace_bytes(16 * candidates_total),
                tile_temp_bytes=tile_temp_peak,
                stamps_zeroed=stamps_zeroed))

    destination_ids = (np.concatenate(ids_parts)
                       if ids_parts else np.zeros(0, np.int64))
    distance_bits = (np.concatenate(dist_parts)
                     if dist_parts else np.zeros(0, np.float64))
    counters = TraceCounters(
        origins=origin_count, tiles=tiles, tile_origins=tile_origins,
        candidates_total=candidates_total, touched_total=touched_total,
        indptr_bytes=indptr.nbytes,
        ids_bytes=destination_ids.nbytes,
        dist_bytes=distance_bits.nbytes,
        metadata_bytes=metadata_bytes,
        trace_bytes=_trace_bytes(destination_ids.nbytes
                                 + distance_bits.nbytes),
        tile_temp_bytes=tile_temp_peak, stamps_zeroed=stamps_zeroed)
    return TraceRecord(
        trace_id=trace_id, schema_version=TRACE_SCHEMA_VERSION,
        origin_count=origin_count, dest_count=d_count,
        cutoff_value=cutoff_value, cutoff_dtype=cutoff_dtype,
        reach_dtype=reach_dtype, metric_dtype=metric_dtype,
        indptr=_read_only(indptr),
        destination_ids=_read_only(destination_ids),
        distance_bits=_read_only(distance_bits),
        complete=True, route=ROUTE_TRACE, refusal="", counters=counters,
        content_digest=hashlib.sha256(
            indptr.tobytes() + destination_ids.tobytes()
            + distance_bits.tobytes()).hexdigest())


# --------------------------------------------------------------------------
# replay_trace: the same full fold on the trace's retained prefix
# --------------------------------------------------------------------------
# P03C (dossier 03, compiled outer replay): the per-origin loop runs as
# the PRIVATE compiled whole-row driver below; the pre-P03C Python loop
# is preserved verbatim beside it as the bit-parity comparator.  The
# module attribute is a private test/evidence seam, not a public
# selector: nothing in the public API reads or writes it, and only
# in-process evidence/test harnesses may set it False.
_REPLAY_COMPILED = True

# Refusal codes shared by both row loops (returned to the facade,
# which maps them onto the exact pre-P03C refusal reasons).
_RC_STRUCTURE = 1   # unreachable today: the facade gates indptr first
_RC_SLICE_ORDER = 2


def _replay_rows_python(indptr, ids_all, dist_all, d_weights, cutoff,
                        gravity_beta, gravity_plateau,
                        gravity_logistic_midpoint, gravity_growth_rate,
                        knn_decay, knn_weights, reach, ge, gl, kn):
    """P03C comparator: the pre-P03C per-origin Python loop, preserved
    verbatim (the refusal reasons now ride return codes).  NOT a public
    API; used by the facade when ``_REPLAY_COMPILED`` is False and by
    the bit-parity / matched-samples evidence."""
    origin_count = indptr.shape[0] - 1
    for origin in range(origin_count):
        sl = slice(int(indptr[origin]), int(indptr[origin + 1]))
        cand_ids = ids_all[sl]
        if cand_ids.shape[0] > 1 and int(np.diff(cand_ids).min()) <= 0:
            return _RC_SLICE_ORDER
        # Writable working copies before the compiled fold: the fold
        # sees exactly the S11 argument domain (fresh C-contiguous
        # writable arrays), and the immutable trace is never aliased
        # into a callee.
        cand_dist = np.array(dist_all[sl])
        cand_wts = np.array(d_weights[cand_ids])
        reach[origin], ge[origin], gl[origin], kn[origin] = \
            reach_gravity_knn_access(
                d_distance=cand_dist,
                d_weights=cand_wts,
                cutoff=cutoff,
                gravity_beta=gravity_beta,
                gravity_plateau=gravity_plateau,
                gravity_logistic_midpoint=gravity_logistic_midpoint,
                gravity_growth_rate=gravity_growth_rate,
                knn_gravity_plateau=gravity_plateau,
                knn_weights=knn_weights,
                knn_gravity_beta=gravity_beta,
                knn_decay=knn_decay,
                knn_gravity_logistic_midpoint=(
                    gravity_logistic_midpoint),
                knn_gravity_growth_rate=gravity_growth_rate,
            )
    return 0


@nb.njit(cache=True, nogil=True, parallel=False)
def _replay_rows_compiled(indptr, ids_all, dist_all, d_weights, cutoff,
                          gravity_beta, gravity_plateau,
                          gravity_logistic_midpoint, gravity_growth_rate,
                          knn_decay, knn_weights, reach, ge, gl, kn):
    """P03C private compiled whole-row driver (dossier 03, Phase B+C).

    Serial/nogil, one dispatch per replay, no nested prange.  Phase B
    scans the WHOLE packed structure first — a corrupt later slice
    refuses before any fold runs, so no partially computed earlier
    outputs can ever be exposed.  Phase C then runs the ORIGINAL full
    compiled fold per origin with the exact original arguments and the
    same fresh per-origin working-copy allocation structure as the
    Python reference.  No float arithmetic lives at the driver level;
    the fold keeps its own specialization, flags and cache identity.

    Refusal codes: 0 success, 1 structural (unreachable — the facade
    gates it), 2 slice-order corruption.
    """
    origin_count = indptr.shape[0] - 1
    # Phase B: checked integer bounds are implied by the facade gates
    # (0 == indptr[0] == indptr[-1] - nnz, non-negative counts); the
    # ascending-slice scan is the driver's own last responsible check.
    for origin in range(origin_count):
        lo = indptr[origin]
        hi = indptr[origin + 1]
        if hi < lo:
            return 1
        for j in range(lo + 1, hi):
            if ids_all[j] <= ids_all[j - 1]:
                return 2
    # Phase C: fold per origin (empty slices included — the fold count
    # still equals all requested origins).  Positional arguments follow
    # the fold's exact signature: d_distance, d_weights, cutoff,
    # gravity_beta, gravity_plateau, gravity_logistic_midpoint,
    # knn_gravity_plateau, knn_weights, knn_gravity_beta, knn_decay,
    # knn_gravity_logistic_midpoint, gravity_growth_rate,
    # knn_gravity_growth_rate.
    for origin in range(origin_count):
        lo = indptr[origin]
        hi = indptr[origin + 1]
        cand_ids = ids_all[lo:hi]
        cand_dist = dist_all[lo:hi].copy()
        cand_wts = d_weights[cand_ids].copy()
        reach[origin], ge[origin], gl[origin], kn[origin] = \
            reach_gravity_knn_access(
                cand_dist, cand_wts, cutoff, gravity_beta,
                gravity_plateau, gravity_logistic_midpoint,
                gravity_plateau, knn_weights, gravity_beta, knn_decay,
                gravity_logistic_midpoint, gravity_growth_rate,
                gravity_growth_rate)
    return 0


def replay_trace(trace, d_weights, gravity_beta, gravity_plateau,
                 gravity_logistic_midpoint, gravity_growth_rate,
                 knn_decay, knn_weights, cutoff):
    """Replay metric jobs from a COMPLETE trace.

    Fresh per-job arrays (the C10 fresh-copy law).  Guards run BEFORE
    any reuse: an incomplete trace is never a cache hit; an invalid
    metric input is refused even though a trace exists (no invalid
    input becomes successful because an earlier trace hit avoided its
    validation); the cutoff must be typed- and bit-exact against the
    trace identity (-0.0 is not +0.0); the destination weight vector
    must match the trace's
    recorded destination count and the dossier-02 value domain.
    """
    def _refuse(reason):
        return TraceReplayResult(
            reach=np.zeros(0), gravity_exponential=np.zeros(0),
            gravity_logistic=np.zeros(0), knn_access=np.zeros(0),
            route="trace_replay_refused", refusal=reason,
            trace_id=trace.trace_id if isinstance(trace, TraceRecord)
            else "", folds_done=0)

    if not isinstance(trace, TraceRecord):
        return _refuse("not_a_trace_record")
    if trace.schema_version != TRACE_SCHEMA_VERSION:
        return _refuse("trace_schema_mismatch")
    if not trace.complete:
        return _refuse("incomplete_trace_not_a_cache_hit")

    cutoff_value = float(cutoff)
    cutoff_dtype = str(np.dtype(getattr(cutoff, "dtype", type(cutoff))))
    # C11-NOTE-3 repair: the gate is typed- AND bit-exact — IEEE-754
    # value equality conflates -0.0 with +0.0 (and distinct NaN
    # payloads), so the cutoff enters by its 8 bytes, never by ==.
    if np.float64(cutoff_value).tobytes() \
            != np.float64(trace.cutoff_value).tobytes() or \
            cutoff_dtype != trace.cutoff_dtype:
        return _refuse("cutoff_mismatch_with_trace_identity")

    if not (isinstance(d_weights, np.ndarray)
            and d_weights.ndim == 1):
        return _refuse("dest_weights_layout")
    if d_weights.shape[0] != trace.dest_count:
        return _refuse("dest_count_mismatch")
    if not np.all(np.isfinite(d_weights)) or \
            np.any(d_weights < 0.0):
        return _refuse("dest_weights_value_domain")

    origin_count = trace.origin_count
    reach = np.empty(origin_count, dtype=np.dtype(trace.reach_dtype))
    ge = np.empty(origin_count, dtype=np.dtype(trace.metric_dtype))
    gl = np.empty(origin_count, dtype=np.dtype(trace.metric_dtype))
    kn = np.empty(origin_count, dtype=np.dtype(trace.metric_dtype))
    folds_done = 0
    indptr = trace.indptr
    ids_all = trace.destination_ids
    dist_all = trace.distance_bits

    # Runtime gates on the trace ITSELF (C10-MINOR-1 behavioral
    # enforcement): a corrupt or structurally impossible record is
    # refused before any fold, never replayed as a cache hit.  The
    # packed layout must be exactly the dossier-04 law: indptr[O+1]
    # from 0, non-negative counts, nnz agreement, int64 ids in range,
    # float64 distances, and every slice strictly ascending (the
    # producer's contract order).
    if indptr.dtype != np.int64 or ids_all.dtype != np.int64 \
            or dist_all.dtype != np.float64 \
            or indptr.shape[0] != origin_count + 1 \
            or int(indptr[0]) != 0 \
            or dist_all.shape[0] != ids_all.shape[0] \
            or int(indptr[-1]) != ids_all.shape[0] \
            or int(np.diff(indptr).min(initial=0)) < 0:
        return _refuse("corrupt_trace_structure")
    if ids_all.shape[0] and (int(ids_all.min()) < 0
                             or int(ids_all.max()) >= trace.dest_count):
        return _refuse("corrupt_trace_destination_ids")
    # C11-NOTE-4: the packed bytes must still hash to the digest bound
    # at build — a forged or bit-flipped retained record is refused
    # before any fold, never replayed as a cache hit.
    if hashlib.sha256(indptr.tobytes() + ids_all.tobytes()
                      + dist_all.tobytes()).hexdigest() \
            != trace.content_digest:
        return _refuse("corrupt_trace_content_digest")

    # P03C (dossier 03): the per-origin loop runs as the private
    # compiled whole-row driver (one dispatch, serial/nogil, full
    # structural scan before any fold); the preserved Python reference
    # above serves the bit-parity and matched-samples evidence.  Both
    # loops return 0, or a refusal code for slice-layout corruption
    # discovered at this last responsible gate — outputs stay fresh
    # and are discarded with the refusal exactly as before.
    if _REPLAY_COMPILED:
        rc = _replay_rows_compiled(
            indptr, ids_all, dist_all, d_weights, cutoff, gravity_beta,
            gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, reach, ge,
            gl, kn)
    else:
        rc = _replay_rows_python(
            indptr, ids_all, dist_all, d_weights, cutoff, gravity_beta,
            gravity_plateau, gravity_logistic_midpoint,
            gravity_growth_rate, knn_decay, knn_weights, reach, ge,
            gl, kn)
    if rc:
        return _refuse("corrupt_trace_structure" if rc == _RC_STRUCTURE
                       else "corrupt_trace_slice_order")
    folds_done = origin_count
    return TraceReplayResult(
        reach=reach, gravity_exponential=ge, gravity_logistic=gl,
        knn_access=kn, route="trace_replay", refusal="",
        trace_id=trace.trace_id, folds_done=folds_done)
