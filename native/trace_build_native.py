"""N00 native trace-build route (campaign una_10x, dossier 08).

Opt-in native replacement for the producer region of
``Engines._trace_access_scratch.build_trace``: the bounded tile loop
(``_tr_epoch_scope_search_tile`` + its per-tile driver) runs in the
AOT C++17 extension; EVERYTHING else -- the S11 admission gate, the
canonical TraceID, the reverse-incidence build, lane sizing, the byte
budget law, the packed-record assembly, the content digest and the
counters -- is the trace module's own Python code, mirrored statement
for statement so a native record is indistinguishable from a numba
record (tests/native prove it bit-exact).

Engagement law: importing this module engages NOTHING.  The native
path runs only after :func:`install` swaps the cohort route's build
hook, and ``_ENGAGEMENT["dispatches"]`` counts REAL kernel dispatches
(never import-time detection).

Owner lifetime: the returned record's arrays wrap extension-owned
capsules; the extension keeps no global state, so dropping Python
references frees the buffers, and the arrays are read-only.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np

NATIVE_DIR = Path(__file__).resolve().parent
BUILD_DIR = NATIVE_DIR / "_build"

# Real-dispatch engagement counters (NOT import detection).
_ENGAGEMENT_LOCK = threading.Lock()
_ENGAGEMENT = {
    "dispatches": 0,        # native kernel dispatches (any outcome)
    "origins": 0,           # origins processed natively
    "tiles": 0,             # completed native tiles
    "budget_stops": 0,      # budget-exceeded outcomes
    "candidates": 0,        # candidates captured natively
    "admission_refusals": 0,  # S11 gate refused before native entry
    "errors": 0,            # core/binding errors surfaced
    "last_status": None,    # last core status code
}

_module = None
_module_lock = threading.Lock()


def _load_build_script():
    """Importlib-load native/build.py under a non-colliding name."""
    name = "una_n00_native_build"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        name, NATIVE_DIR / "build.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_extension():
    """Importlib-load the built extension (build first if stale)."""
    global _module
    with _module_lock:
        if _module is not None:
            return _module
        so = _load_build_script().ensure_built()
        name = "una_n00_trace_producer"
        loader = importlib.machinery.ExtensionFileLoader(name, str(so))
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _module = mod
        return mod


def build_info() -> dict:
    """The extension's own build identity + this wrapper's counters."""
    mod = _load_extension()
    info = dict(mod.build_info())
    info["build_manifest"] = (BUILD_DIR / "build_manifest.json").read_text()
    info["engagement"] = engagement()
    return info


def engagement() -> dict:
    with _ENGAGEMENT_LOCK:
        return dict(_ENGAGEMENT)


def _eng(key, amount=1):
    with _ENGAGEMENT_LOCK:
        _ENGAGEMENT[key] += amount


# --------------------------------------------------------------------------
# The native build_trace twin: same signature, same refusal law, same
# record.  Imported lazily so this module imports without the engine
# (tests control both sides explicitly).
# --------------------------------------------------------------------------
def native_build_trace(o_terminal_idxs, o_terminal_weights,
                       adjacency_pointer, adjacency_vector,
                       adjacency_vector_weights,
                       adjacency_vector_network_node,
                       d_terminal_idxs, d_terminal_weights, d_weights,
                       cutoff, epoch_rollover=None,
                       trace_budget_bytes=None, tile_origins=None,
                       profile_id="una_legacy-epoch-a3"):
    """build_trace with the tile loop running in the native extension.

    Refusal law is VERBATIM build_trace: the S11 admission gate runs
    first (numba/Python, unchanged); a refused input produces the same
    INCOMPLETE record with the same reason code, so the cohort route's
    sparse gate cannot tell a native refusal from a numba one.
    """
    from urban_network_analysis.Engines import _trace_access_scratch as trm
    from urban_network_analysis.Engines._sparse_access_scratch import (
        DEFAULT_EPOCH_ROLLOVER,
        _a1_threads_bound,
    )
    if epoch_rollover is None:
        epoch_rollover = DEFAULT_EPOCH_ROLLOVER
    if trace_budget_bytes is None:
        trace_budget_bytes = trm.DEFAULT_TRACE_BUDGET_BYTES
    if tile_origins is None:
        tile_origins = trm.DEFAULT_TILE_ORIGINS

    origin_count = int(o_terminal_idxs.shape[0])
    node_count = int(adjacency_pointer.shape[0]) - 1
    d_count = int(d_terminal_weights.shape[0])
    cutoff_value = float(cutoff)
    cutoff_dtype = str(np.dtype(getattr(cutoff, "dtype", type(cutoff))))
    reach_dtype = str(np.dtype(o_terminal_idxs.dtype))
    metric_dtype = str(np.dtype(adjacency_vector_weights.dtype))
    trace_id = trm.compute_trace_id(
        o_terminal_idxs, o_terminal_weights, adjacency_pointer,
        adjacency_vector, adjacency_vector_weights,
        adjacency_vector_network_node, d_terminal_idxs,
        d_terminal_weights, cutoff, epoch_rollover,
        profile_id=profile_id)

    def _incomplete(route, refusal, counters):
        return trm.TraceRecord(
            trace_id=trace_id, schema_version=trm.TRACE_SCHEMA_VERSION,
            origin_count=origin_count, dest_count=d_count,
            cutoff_value=cutoff_value, cutoff_dtype=cutoff_dtype,
            reach_dtype=reach_dtype, metric_dtype=metric_dtype,
            indptr=trm._read_only(np.zeros(origin_count + 1, np.int64)),
            destination_ids=trm._read_only(np.zeros(0, np.int64)),
            distance_bits=trm._read_only(np.zeros(0, np.float64)),
            complete=False, route=route, refusal=refusal,
            counters=counters, content_digest="")

    admitted, refusal, max_degree = trm._sp_route_admits(
        adjacency_pointer, adjacency_vector, adjacency_vector_weights,
        adjacency_vector_network_node, o_terminal_idxs,
        o_terminal_weights, d_terminal_idxs, d_terminal_weights,
        d_weights, cutoff)
    if not admitted:
        # The native core never ran; count the refusal honestly.
        _eng("admission_refusals")
        return _incomplete("compact_fallback", refusal, trm.TraceCounters(
            origins=origin_count, tiles=0, tile_origins=0,
            candidates_total=0, touched_total=0, indptr_bytes=0,
            ids_bytes=0, dist_bytes=0, metadata_bytes=0, trace_bytes=0,
            tile_temp_bytes=0, stamps_zeroed=0))

    tile_origins = max(1, min(int(tile_origins), origin_count))
    reverse_pointer, reverse_vector = trm._reverse_incidence(
        d_terminal_idxs, node_count)

    # Same lane sizing law as build_trace (numba's configured pool).
    lanes = max(1, min(int(_a1_threads_bound()), tile_origins))
    metadata_bytes = len(trace_id.encode("utf-8"))
    indptr_bytes = (origin_count + 1) * 8
    # Exact integer identity with build_trace's per-tile budget check:
    #   indptr_bytes + 16*cand_total + metadata > budget
    #   <=>  16*cand_total > budget - metadata - indptr_bytes
    budget_cand_bytes = int(trace_budget_bytes) - metadata_bytes \
        - indptr_bytes

    # The cohort route times this call as build_s exactly as it times
    # the numba build_trace; no local timer is kept here.
    mod = _load_extension()
    res = mod.run_stream(
        o_terminal_idxs=o_terminal_idxs,
        o_terminal_weights=o_terminal_weights,
        adjacency_pointer=adjacency_pointer,
        adjacency_vector=adjacency_vector,
        adjacency_vector_weights=adjacency_vector_weights,
        adjacency_flag_column=adjacency_vector_network_node,
        d_terminal_idxs=d_terminal_idxs,
        d_terminal_weights=d_terminal_weights,
        reverse_pointer=reverse_pointer,
        reverse_vector=reverse_vector,
        cutoff=cutoff,
        epoch_rollover=int(epoch_rollover),
        tile_origins=tile_origins,
        lanes=lanes,
        budget_cand_bytes=budget_cand_bytes,
        max_degree=int(max_degree),
    )

    counts = np.asarray(res["counts"], dtype=np.int64)
    tiles = int(res["tiles_done"])
    _eng("dispatches")
    _eng("origins", origin_count)
    _eng("tiles", tiles)
    _eng("candidates", int(res["cand_total"]))
    with _ENGAGEMENT_LOCK:
        _ENGAGEMENT["last_status"] = int(res["status"])

    trace_bytes_stat = indptr_bytes + \
        16 * int(res["cand_total"]) + metadata_bytes
    if int(res["budget_hit"]) >= 0:
        _eng("budget_stops")
        return _incomplete(
            trm.ROUTE_BUDGET,
            "trace budget exceeded at origin %d" % int(res["budget_hit"]),
            trm.TraceCounters(
                origins=origin_count, tiles=tiles,
                tile_origins=tile_origins,
                candidates_total=int(res["cand_total"]),
                touched_total=int(res["touched_total"]),
                indptr_bytes=indptr_bytes,
                ids_bytes=8 * int(res["cand_total"]),
                dist_bytes=8 * int(res["cand_total"]),
                metadata_bytes=metadata_bytes,
                trace_bytes=trace_bytes_stat,
                tile_temp_bytes=int(res["scratch_peak_bytes"]),
                stamps_zeroed=tiles * lanes * (node_count + d_count)))

    # build_trace's assembly, mirrored: indptr from cumsum(counts),
    # packed parts verbatim (origin-major ascending ids), digest over
    # the packed bytes in layout order.
    indptr = np.zeros(origin_count + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    destination_ids = trm._read_only(np.asarray(res["ids"],
                                                dtype=np.int64))
    distance_bits = trm._read_only(np.asarray(res["dist"],
                                              dtype=np.float64))
    counters = trm.TraceCounters(
        origins=origin_count, tiles=tiles, tile_origins=tile_origins,
        candidates_total=int(res["cand_total"]),
        touched_total=int(res["touched_total"]),
        indptr_bytes=indptr_bytes,
        ids_bytes=destination_ids.nbytes,
        dist_bytes=distance_bits.nbytes,
        metadata_bytes=metadata_bytes,
        trace_bytes=indptr_bytes + destination_ids.nbytes
        + distance_bits.nbytes + metadata_bytes,
        # Measured native scratch high-water (the extension's own
        # bound, declared in native/trace_producer.h) -- the numba
        # route charges its tile capture; the routes measure
        # different scratch by design and both honestly.
        tile_temp_bytes=int(res["scratch_peak_bytes"]),
        stamps_zeroed=tiles * lanes * (node_count + d_count))
    return trm.TraceRecord(
        trace_id=trace_id, schema_version=trm.TRACE_SCHEMA_VERSION,
        origin_count=origin_count, dest_count=d_count,
        cutoff_value=cutoff_value, cutoff_dtype=cutoff_dtype,
        reach_dtype=reach_dtype, metric_dtype=metric_dtype,
        indptr=trm._read_only(indptr),
        destination_ids=destination_ids,
        distance_bits=distance_bits,
        complete=True, route=trm.ROUTE_TRACE, refusal="",
        counters=counters,
        content_digest=hashlib.sha256(
            np.asarray(indptr).tobytes()
            + np.asarray(destination_ids).tobytes()
            + np.asarray(distance_bits).tobytes()).hexdigest())


# --------------------------------------------------------------------------
# Opt-in install into the cohort route (the ONLY production wiring).
# --------------------------------------------------------------------------
def install() -> None:
    """Point the cohort route's build hook at the native twin.

    Purely opt-in: until this runs, the cohort route calls the numba
    build_trace exactly as before.  Idempotent.
    """
    from urban_network_analysis.batch import cohort as cohort_mod
    _load_extension()
    cohort_mod.install_native_build_hook(native_build_trace)


def uninstall() -> None:
    from urban_network_analysis.batch import cohort as cohort_mod
    cohort_mod.install_native_build_hook(None)
