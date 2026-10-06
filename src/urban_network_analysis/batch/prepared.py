"""Prepared-layer reuse — una_10x P02 increments B+C (private).

Contract: campaigns/una_10x/tenx-20261005T2331Z/p02d/prepared_contract.json
(increment B: worker-local reuse; increment C: run-wide producer per
immutable dependency identity — single-flight).

Scope: ONE process reuses ITS OWN prepared layers across repeated Add*
calls (batch rows re-loading identical inputs), and — increment C — all
workers of one batch run coordinate through a run-scoped disk store so one
decode/build per identity happens per RUN, not per worker.  There is no
public API here — this module is reached only from
Topology.AddNetwork/BuildAccessPoints/Add* internals and the batch
worker's attach/detach.

Laws implemented (contract identity_and_keys / toctou_law / mutability_law
/ chronology):

- Identity closes over the admitted snapshot's exact BYTES (sha256) + the
  producer-read settings + profile + environment — never the path or the
  mtime.  Key compositions mirror the stage-cache keys field-for-field
  (``una:topology:v1`` / ``una:snap:v1``) under separate ``una:prepared:*``
  table namespaces, and every snap identity chains its parent network
  stage key, so dependency invalidation is exact per layer (PP04).
- TOCTOU: bytes are snapshotted once and both hashed AND decoded from the
  same snapshot (dossier 06 rule; PP01 fixtures prove the reader routes
  equivalent).
- Refuse-different-bytes at lease time: a served bundle's recorded digest
  must equal the request digest (store-side truth guard).
- Chronology on hits: geometry validation, CRS handling and every
  content-derived log line fire in the produced order.  The validation
  helpers here ARE the produced path's helpers (single source — the
  produced call sites were moved verbatim into this module).
- Immutable leases: payloads are private copies, isolated from the
  producing row's objects at publish and from every consumer at restore;
  consumers receive fresh copies of each array/DataFrame surface the
  produced path would have freshly built (fresh mutable outputs).  A
  bundle held for restore carries a lease credit (``in_use``); eviction
  never drops a leased entry (PP09), and because consumers never hold
  payload references, an entry popped underneath a restore (the
  refuse-different-bytes path) still restores correctly — no
  use-after-release is possible.
- Engagement is .geojson-first (contract admission_domain, designs/02):
  other formats return MISS and fall through to the produced path
  unchanged.  A produced failure never publishes (payloads are
  post-validation content); empty layers are refused.
- Kill switch: UNA_PREPARED=0 disables admission entirely (paired-control
  duty); the process-global table is cleared via reset().
- Run-wide tier (increment C): the attached run store is a disk
  ContentStore in the run's ``.una-batch-prepared`` directory (beside the
  first dispatched row's public folder — the C30 cohort-root convention:
  stable across resumes, safe to delete, identity-verified at every
  load).  Transport reuses the stage serializers' exact payload format
  (WKB + ``.npy`` typed arrays, ``allow_pickle=False`` — a payload that
  cannot be represented without pickle is refused at the run tier and
  keeps only its worker-local bundle).  Coordination is
  ``ContentStore.singleflight`` (cohort budget: 600 s), so one worker
  builds per identity while its losers are served from the committed
  generation; an owner that dies mid-build leaves a stale lock that is
  stolen on pid liveness, and a loser whose owner committed nothing is
  re-elected owner by the store.  The registry of run stores is one
  process entry per root (``_PROCESS_RUN_STORES``); workers attach once
  per row and detach before anything is transported (the run route is
  runtime-only, never observable serial state).
"""
from __future__ import annotations

import itertools
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import __version__ as _PACKAGE_VERSION
from ..cache import identity as _ckid

_ENV_FLAG = "UNA_PREPARED"
_ENV_BUDGET = "UNA_PREPARED_BUDGET_BYTES"
_SCHEMA_TOPOLOGY = "una:prepared:topology:v1"
_SCHEMA_SNAP = "una:prepared:snap:v1"
_SINGLEFLIGHT_WAIT_S = 600.0          # cohort budget: matches P00 config

# Residency guard for the worker-local table (PP09).  This is the
# REGISTRY accounting law — exact ndarray bytes + pandas deep memory +
# real WKB bulk — not a claim about physical RSS: reserving against the
# row budget before admission is the batch runtime's admission duty
# (designs/02; the projection seam for that is the publish accounting
# itself).  Override in tests via UNA_PREPARED_BUDGET_BYTES.
_DEFAULT_TABLE_BUDGET_BYTES = 256 * 1024 * 1024

_lock = threading.Lock()
_table: Dict[str, "_Bundle"] = {}
_table_bytes = 0
_seq = itertools.count()
_stats: Dict[str, Dict[str, int]] = {
    "topology": {"requests": 0, "hits": 0, "misses": 0, "publishes": 0,
                 "refusals": 0},
    "snap": {"requests": 0, "hits": 0, "misses": 0, "publishes": 0,
             "refusals": 0},
}
# Run-tier counters (designs/02: cold build / verified hit / contention
# / corruption rebuild / refusal are DIFFERENT counters).  Kept separate
# from the per-layer counters the increment-B fixtures pin.
_run_counters: Dict[str, int] = {
    "tier2_hits": 0,          # verified hits served from the run store
    "tier2_publishes": 0,
    "tier2_refusals": 0,      # payload not representable / store refused
    "cold_builds": 0,         # this caller entered the flight as owner
    "contention_waits": 0,    # loser exits: another holder had committed
    "contention_timeouts": 0, # bounded wait expired; built unserialized
    "corruption_rebuilds": 0, # run-store payload damaged; quarantined
    "evictions": 0,           # LRU evictions from the worker-local table
    "budget_refusals": 0,     # publish refused: table budget not met
    "promote_skips": 0,       # tier-2 hit served WITHOUT retaining: budget
}

# One process-level store pair per run root (the C30 cohort discipline:
# built once, reused across this worker's rows; the worker process dies
# with the run, so the pair's lifetime is the run's).
_PROCESS_RUN_STORES: Dict[str, "_RunStore"] = {}
_RUN: Optional["_RunStore"] = None


class _RunStore:
    """The attached run-wide tier: disk store + stage-format facade."""

    def __init__(self, root: str) -> None:
        from ..Execution import CacheOptions
        from ..cache.store import ContentStore
        from ..cache.stages import StageCache
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.store = ContentStore(
            CacheOptions(mode="disk", directory=root))
        self.stage = StageCache(self.store)


def attach_run_store(root: str) -> None:
    """Attach the run-wide tier for this worker (no-op content when the
    kill switch is on: callers gate on enabled(), and even a stray attach
    only builds an empty store — nothing is admitted)."""
    global _RUN
    with _lock:
        rs = _PROCESS_RUN_STORES.get(root)
        if rs is None:
            rs = _RunStore(root)
            _PROCESS_RUN_STORES[root] = rs
        _RUN = rs


def detach_run_store() -> None:
    """Detach before anything observes or transports this row (the run
    route is runtime-only).  The process store pair stays for this
    worker's remaining rows; publishes after detach keep only their
    worker-local bundle — the run store is never touched post-detach."""
    global _RUN
    with _lock:
        _RUN = None


def enabled() -> bool:
    """Kill switch: UNA_PREPARED=0 turns prepared reuse off entirely."""
    return os.environ.get(_ENV_FLAG, "1") != "0"


def _table_budget() -> int:
    raw = os.environ.get(_ENV_BUDGET)
    return int(raw) if raw else _DEFAULT_TABLE_BUDGET_BYTES


def reset() -> None:
    """Clear the worker-local table and counters (test isolation).  The
    attached run store is managed explicitly (attach/detach) and is NOT
    touched here."""
    global _table_bytes
    with _lock:
        _table.clear()
        _table_bytes = 0
        for counters in _stats.values():
            for k in counters:
                counters[k] = 0
        for k in _run_counters:
            _run_counters[k] = 0


def stats() -> Dict[str, Dict[str, int]]:
    with _lock:
        out = {layer: dict(counters) for layer, counters in _stats.items()}
        out["run"] = dict(_run_counters)
        out["table"] = {"entries": len(_table), "bytes": _table_bytes}
        return out


def mark() -> Dict[str, Dict[str, int]]:
    """Row-start snapshot for the worker's per-row delta (cohort pattern)."""
    return stats()


def delta(before: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """This row's slice of every counter (transported on the RowResult;
    runtime-only reporting, never observable serial state)."""
    now = stats()
    out: Dict[str, Dict[str, int]] = {}
    for group, counters in now.items():
        b = before.get(group) or {}
        out[group] = {k: int(v) - int(b.get(k, 0))
                      for k, v in counters.items()}
    return out


@dataclass(frozen=True)
class PreparedRequest:
    """One admitted Add* input: pinned bytes plus both identity keys."""

    layer: str            # "topology" | "snap"
    key: str              # una:prepared:* worker-local table key
    stage_key: str        # una:topology:v1 / una:snap:v1 identity
    file_sha256: str
    snapshot: bytes
    source_file: str


class _Bundle:
    """Immutable-by-discipline lease payload (private to this module).

    ``in_use`` is the live-lease credit (acquire/promote +1, restore
    finally −1); eviction (PP09) never drops a leased entry.  ``nbytes``
    is the registry accounting (bulk-exact: arrays, pandas deep memory,
    real WKB bulk).  ``source`` names the serving tier for the restore
    marker: ``worker`` or ``run``."""

    __slots__ = ("key", "file_sha256", "payload", "source", "in_use",
                 "nbytes", "seq")

    def __init__(self, key: str, file_sha256: str, payload: dict, *,
                 source: str = "worker") -> None:
        self.key = key
        self.file_sha256 = file_sha256
        self.payload = payload
        self.source = source
        self.in_use = 0
        self.nbytes = _account_bytes(payload)
        self.seq = next(_seq)


def _admitted(source_file: str) -> bool:
    """.geojson-first engagement (contract admission_domain)."""
    return enabled() and source_file.lower().endswith(".geojson")


def _snapshot(source_file: str) -> bytes:
    # Deferred: _read_snapshot is Topology-module-level and Topology
    # imports this package at module load (one-way at import time).
    from urban_network_analysis.Topology import _read_snapshot
    return _read_snapshot(source_file)


# ── identity composition (mirrors the stage-cache closures exactly) ─────

def topology_request(source_file: str, settings) -> Optional[PreparedRequest]:
    """Admit one AddNetwork input and compose its identity, or None.

    The closure mirrors Topology.AddNetwork's ``una:topology:v1`` key
    field-for-field, so a prepared hit leaves ``network._stage_key``
    identical to what the produced+cached path records."""
    if not _admitted(source_file):
        return None
    snapshot = _snapshot(source_file)
    digest = _ckid.file_digest(snapshot)
    closure = {
        "file_sha256": digest,
        "params": {
            "network_weight_column": str(settings.network_weight_column),
            "network_precision": int(settings.network_precision),
            "network_load_nodes": bool(settings.network_load_nodes),
        },
        "profile": str(settings.execution.semantic_profile),
        "env": _ckid.geometry_env(
            str(settings.execution.semantic_profile), _PACKAGE_VERSION),
    }
    with _lock:
        _stats["topology"]["requests"] += 1
    return PreparedRequest(
        layer="topology",
        key=_ckid.stage_key(_SCHEMA_TOPOLOGY, closure),
        stage_key=_ckid.stage_key("una:topology:v1", closure),
        file_sha256=digest,
        snapshot=snapshot,
        source_file=source_file,
    )


def snap_request(topology, source_file: str, cost_attribute,
                 default_cost, uid_attribute) -> Optional[PreparedRequest]:
    """Admit one AddOrigins/AddDestinations input, or None.

    The snap identity chains the parent network stage key — a network
    change invalidates every dependent layer (exact dependency closure).
    A network without a stage identity (produced outside prepared/stage
    tracking) means no admission: the produced path runs unchanged."""
    if not _admitted(source_file):
        return None
    parent = getattr(topology.network, "_stage_key", None)
    if not parent:
        return None
    snapshot = _snapshot(source_file)
    digest = _ckid.file_digest(snapshot)
    closure = {
        "file_sha256": digest,
        "parent_network_stage": parent,
        "params": {
            "cost_attribute": str(cost_attribute),
            "default_cost": default_cost,
            "uid_attribute": str(uid_attribute or ""),
        },
        "profile": topology._stage_profile,
        "env": _ckid.geometry_env(
            topology._stage_profile, _PACKAGE_VERSION),
    }
    with _lock:
        _stats["snap"]["requests"] += 1
    return PreparedRequest(
        layer="snap",
        key=_ckid.stage_key(_SCHEMA_SNAP, closure),
        stage_key=_ckid.stage_key("una:snap:v1", closure),
        file_sha256=digest,
        snapshot=snapshot,
        source_file=source_file,
    )


def acquire(req: PreparedRequest) -> Optional[_Bundle]:
    """Worker-local lease lookup with the refuse-different-bytes guard.

    The key already embeds the request digest, so a changed file normally
    misses by construction; the recorded-digest comparison is the
    contract's store-side truth guard and pops any bundle that fails it.
    A hit takes one lease credit, released by the restore's finally —
    eviction never targets a leased entry."""
    with _lock:
        bundle = _table.get(req.key)
        if bundle is None:
            _stats[req.layer]["misses"] += 1
            return None
        if bundle.file_sha256 != req.file_sha256:
            _table.pop(req.key, None)
            _table_bytes -= bundle.nbytes
            _stats[req.layer]["refusals"] += 1
            return None
        bundle.in_use += 1
        bundle.seq = next(_seq)  # LRU refresh on use
        _stats[req.layer]["hits"] += 1
        return bundle


# ── validation (single source: the produced call sites moved verbatim) ──

def validate_network_gdf(gdf) -> None:
    """Network geometry validation (verbatim from Topology.AddNetwork)."""
    invalid_types = gdf.geometry.geom_type[
        gdf.geometry.geom_type != 'LineString'].unique().tolist()
    if invalid_types:
        # geom_type is NaN (a float) for features with missing/null
        # geometry — cast to str so the message itself never crashes,
        # and name the null-geometry case explicitly since it is the
        # most common cause.
        type_names = ', '.join(
            'missing/null geometry' if not isinstance(x, str) else x
            for x in invalid_types)
        n_null = int(gdf.geometry.isna().sum()
                     + gdf.geometry.is_empty.fillna(False).sum())
        raise ValueError(
            f"Network file must contain LineString (line/polyline) geometries, "
            f"but the file contains: {type_names}"
            + (f" ({n_null} feature(s) with missing/empty geometry — drop them "
               f"before loading)" if n_null else "")
            + ". Please provide a file with line or polyline geometries."
        )


def validate_point_gdf(gdf, label: str, network_crs) -> None:
    """Point-layer validation (verbatim from Topology.BuildAccessPoints,
    with the network CRS parametrized)."""
    invalid_types = gdf.geometry.geom_type[
        gdf.geometry.geom_type != 'Point'].unique().tolist()
    if invalid_types:
        raise ValueError(
            f"{label} file must contain Point geometries, "
            f"but the file contains: {', '.join(invalid_types)}. "
            f"Please provide a file with point geometries."
        )
    if network_crs is not None:
        if gdf.crs is None:
            raise ValueError(
                f"{label} file has no CRS defined, but network CRS is {network_crs}. "
                f"Please assign CRS {network_crs} to the file before loading."
            )
        elif not gdf.crs.equals(network_crs):
            raise ValueError(
                f"{label} file CRS ({gdf.crs}) does not match network CRS ({network_crs}). "
                f"Please reproject to {network_crs} before loading."
            )


def replay_log_extras(logger, extras) -> None:
    """Replay content-derived warning lines ((event, details, v) triples)
    recorded at preparation time — the same discipline as the stage-cache
    ``log_replay`` metadata."""
    for event, details, v in (extras or []):
        logger.log(event, details, v=int(v))


# ── restore (produced field set + chronology, per-consumer copies) ──────

def restore_topology(topology, req: PreparedRequest, bundle: _Bundle) -> None:
    """Populate ``topology.network`` from an immutable lease: the exact
    produced-path state and log chronology, with fresh copies of every
    array (the lease itself is never handed out)."""
    p = bundle.payload
    try:
        _restore_topology_into(topology, req, bundle, p)
    finally:
        with _lock:
            bundle.in_use -= 1


def _restore_topology_into(topology, req: PreparedRequest,
                           bundle: _Bundle, p: dict) -> None:

    # Chronology 1: geometry validation before any CRS handling, over the
    # stored input frame when carried (or the final geometry — both are
    # post-validation content by construction; a corrupted payload fails
    # closed here).
    surface = p["input_gdf"]
    if surface is None:
        surface = gpd.GeoDataFrame(geometry=p["geometry"])
    validate_network_gdf(surface)

    def _copy(x):
        return None if x is None else x.copy()

    crs = p["crs"]
    net = topology.network
    # Chronology 2-5: CRS handling, the added-line, weight warning,
    # empty-geometry removal — produced order (Topology.AddNetwork).
    topology.crs = crs
    if crs is None:
        topology.logger.log(
            "Network added", "Warning: Network file has no CRS defined.",
            v=1)
    else:
        topology.logger.log("Network added", f"CRS set to {crs}.", v=2)
    topology.logger.log(
        "Network added",
        f"Network added from source {req.source_file}, "
        f"with {p['n_input_features']} edges.", v=1)
    if p["weight_warning"]:
        topology.logger.log("Network added", p["weight_warning"], v=1)
    if p["n_empty_removed"]:
        topology.logger.log(
            "BuildTopology",
            f"Removing {p['n_empty_removed']} empty geometries from "
            f"network before topology build.", v=1)
    # State: fresh copies (the produced path builds each array per row).
    net.geometry = p["geometry"].copy()
    net.lengths = p["lengths"].copy()
    net.weights = p["weights"].copy()
    net.node_points = p["node_points"].copy()
    net.start_nodes = _copy(p["start_nodes"])
    net.end_nodes = _copy(p["end_nodes"])
    net.degrees = p["degrees"].copy()
    net.z = _copy(p["z"])
    net.start_degrees = p["start_degrees"].copy()
    net.end_degrees = p["end_degrees"].copy()
    net.min_degrees = p["min_degrees"].copy()
    net.max_degrees = p["max_degrees"].copy()
    # Derived state: rebuilt from the restored geometries (same order,
    # same bits) — never stored, never transported.
    net.edge_sindex = net.geometry.sindex
    net._stage_key = p["stage_key"]
    # Chronology 6-8: BuildTopology/close lines + the restore marker
    # (mirrors the stage-cache restore's replay discipline).
    topology.logger.log(
        "BuildTopology",
        f"Topology built with {p['n_nodes']} nodes and "
        f"{p['n_edges']} edges.", v=2)
    topology.logger.log('create topology', "edge_list_dict created", v=2)
    marker = ("worker-local" if bundle.source == "worker" else "run store")
    topology.logger.log(
        "Network added",
        f"Topology restored from prepared layer {req.key[:12]}… "
        f"({marker}).", v=2)


def restore_access_points(topology, req: PreparedRequest, bundle: _Bundle,
                          label: str):
    """Rebuild one AccessPoints object from an immutable lease (produced
    field set + chronology; per-consumer copies throughout)."""
    from urban_network_analysis.Topology import AccessPoints
    p = bundle.payload
    try:
        return _restore_access_points_into(topology, req, bundle, p,
                                           AccessPoints, label)
    finally:
        with _lock:
            bundle.in_use -= 1


def _restore_access_points_into(topology, req: PreparedRequest,
                                bundle: _Bundle, p: dict,
                                AccessPoints, label: str) -> "object":
    gdf = (p["input_gdf"].copy(deep=True)
           if p["input_gdf"] is not None else None)
    if gdf is None:
        gdf = gpd.GeoDataFrame(geometry=p["geometry"].copy())
    # Chronology 1-2: validation (geometry types, then CRS-vs-network),
    # then reset_index — produced order.
    validate_point_gdf(gdf, label, topology.crs)
    gdf = gdf.reset_index(drop=True)
    # Chronology 3-4: the built-line, then content-derived warnings.
    topology.logger.log(
        "Build AccessPoints",
        f"Access points are built from source {req.source_file}, "
        f"with {len(gdf)} points.", v=1)
    replay_log_extras(topology.logger, p["log_extras"])
    pt = AccessPoints(topology.logger)
    if p["input_gdf"] is not None:
        # Produced aliasing: raw shares the frame geometry comes from.
        pt.raw = gdf
    else:
        pt.raw = None
    pt.geometry = gdf.geometry
    pt.node_weight = p["node_weight"].copy()
    uid = p["uid"]
    if uid is None:
        pt.uid = None
    elif isinstance(uid, pd.Series):
        pt.uid = uid.copy()
    else:
        pt.uid = np.asarray(uid).copy()
    if p["snapped"]:
        pt.nearest_edge_point = gpd.array.GeometryArray(
            np.asarray(p["nearest_edge_point"]).copy())
        pt.nearest_edge_id = p["nearest_edge_id"].copy()
        pt.edge_start_node = p["edge_start_node"].copy()
        pt.edge_end_node = p["edge_end_node"].copy()
        pt.weight_to_start = p["weight_to_start"].copy()
        pt.weight_to_end = p["weight_to_end"].copy()
    pt._stage_key = p["stage_key"]
    marker = ("worker-local" if bundle.source == "worker" else "run store")
    topology.logger.log(
        "Build AccessPoints",
        f"Snapped layer restored from prepared layer {req.key[:12]}… "
        f"({marker}).", v=2)
    return pt


# ── publish (post-success only; payloads are isolated copies) ───────────

def _account_bytes(payload: dict) -> int:
    """Registry accounting for one bundle (PP09 budget law): exact for
    the bulk fields that scale with the layer — ndarray ``nbytes``,
    pandas deep memory, real WKB bulk for geometry — and a flat 256-byte
    allowance per scalar/list field.  This is the TABLE's residency
    accounting, not a physical-RSS claim (see the module docstring)."""
    total = 0
    for v in payload.values():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            total += int(v.nbytes)
        elif isinstance(v, (pd.Series, pd.DataFrame)):
            total += int(v.memory_usage(deep=True).sum()
                         if isinstance(v, pd.DataFrame)
                         else v.memory_usage(deep=True))
        elif isinstance(v, (gpd.GeoSeries, gpd.array.GeometryArray)):
            import shapely
            wkbs = shapely.to_wkb(np.asarray(v))
            total += int(sum(len(b) for b in wkbs)) + 64 * len(wkbs)
        elif isinstance(v, (list, tuple)):
            total += 256 * max(1, len(v))
        else:
            total += 256
    return total


def _admit_locked(req: PreparedRequest, payload: dict, layer: str) -> bool:
    """Insert one bundle under the table budget (PP09): evict LRU among
    UNLEASED entries while over budget; refuse (and stay under the cap
    floor) when even that cannot fit — an oversized payload is never
    partially admitted.  Caller holds _lock.  Returns True when inserted."""
    global _table_bytes
    bundle = _Bundle(req.key, req.file_sha256, payload)
    old = _table.get(req.key)
    if old is not None:
        _table_bytes -= old.nbytes  # replaced (digest-refusal republish)
    while _table_bytes + bundle.nbytes > _table_budget():
        candidates = [(b.seq, k) for k, b in _table.items()
                      if b.in_use == 0 and k != req.key]
        if not candidates:
            _run_counters["budget_refusals"] += 1
            if old is not None:
                _table[req.key] = old
                _table_bytes += old.nbytes
            return False
        _, victim = min(candidates)
        _table_bytes -= _table.pop(victim).nbytes
        _run_counters["evictions"] += 1
    _table[req.key] = bundle
    _table_bytes += bundle.nbytes
    _stats[layer]["publishes"] += 1
    return True


def _publish_to_run_store(req: PreparedRequest, builder) -> None:
    """Commit the produced state to the attached run store as a
    stage-format generation (WKB + typed arrays, no pickle).  A payload
    the format cannot represent is refused with a counter — the layer
    keeps only its worker-local bundle and other workers recompute it
    (correctness never depends on the run tier)."""
    with _lock:
        run = _RUN
    if run is None:
        return
    built = builder()
    if built is None:
        with _lock:
            _run_counters["tier2_refusals"] += 1
        return
    arrays, metadata = built
    metadata = {**metadata, "file_sha256": req.file_sha256,
                "prepared": "run:v1"}
    ok = run.stage.put_arrays(req.key, arrays, metadata)
    with _lock:
        _run_counters["tier2_publishes" if ok else "tier2_refusals"] += 1


def publish_topology(req: PreparedRequest, net, gdf, *, crs,
                     weight_warning, n_empty_removed, n_nodes, n_edges,
                     n_input_features) -> None:
    """Publish one topology bundle into the worker-local table (budget
    admitted) and the attached run store.  Called only after the produced
    path fully succeeded (payloads are post-validation content).  Also
    records ``net._stage_key`` so dependent snap identities can chain the
    parent — the pure produced path leaves it unset.  Everything is
    copied at publish: the producing row keeps mutating its own objects
    (engine fold) and must never reach lease state."""
    payload = {
        "input_gdf": None if gdf is None else gdf.copy(deep=True),
        "geometry": net.geometry.copy(),
        "lengths": net.lengths.copy(),
        "weights": net.weights.copy(),
        "node_points": net.node_points.copy(),
        "start_nodes": (None if net.start_nodes is None
                        else net.start_nodes.copy()),
        "end_nodes": (None if net.end_nodes is None
                      else net.end_nodes.copy()),
        "degrees": net.degrees.copy(),
        "z": None if net.z is None else net.z.copy(),
        "start_degrees": net.start_degrees.copy(),
        "end_degrees": net.end_degrees.copy(),
        "min_degrees": net.min_degrees.copy(),
        "max_degrees": net.max_degrees.copy(),
        "crs": crs,
        "stage_key": req.stage_key,
        "weight_warning": weight_warning,
        "n_empty_removed": int(n_empty_removed),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_input_features": int(n_input_features),
    }
    with _lock:
        # Conservative admission: empty layers stay unadmitted.
        if n_edges == 0 or len(net.geometry) == 0:
            _stats["topology"]["refusals"] += 1
            net._stage_key = req.stage_key
            return
        _admit_locked(req, payload, "topology")
        net._stage_key = req.stage_key
    from urban_network_analysis.Topology import _network_stage_payload
    _publish_to_run_store(
        req, lambda: _network_stage_payload(
            net, gdf, weight_warning=weight_warning,
            no_crs=(crs is None), crs=crs))


def publish_access_points(req: PreparedRequest, pt, *, log_extras,
                          n_points) -> None:
    """Publish one snapped point layer (post-success only; isolated
    copies, as publish_topology).  Identity is recorded on the object in
    every path (including refusals) — the same store philosophy as the
    stage cache: downstream engine keys (o_access chains
    ``origins/destinations._stage_key``) stay complete regardless of
    whether storage happened."""
    pt._stage_key = req.stage_key
    uid = pt.uid
    payload = {
        "input_gdf": (None if getattr(pt, "raw", None) is None
                      else pt.raw.copy(deep=True)),
        "geometry": pt.geometry.copy(),
        "node_weight": pt.node_weight.copy(),
        "uid": (uid.copy() if isinstance(uid, pd.Series)
                else None if uid is None else np.asarray(uid).copy()),
        "snapped": pt.nearest_edge_id is not None,
        "stage_key": req.stage_key,
        "n_points": int(n_points),
        "log_extras": [tuple(x) for x in (log_extras or [])],
    }
    with _lock:
        if int(n_points) == 0:
            _stats["snap"]["refusals"] += 1
            return
        if payload["snapped"]:
            payload["nearest_edge_point"] = gpd.array.GeometryArray(
                np.asarray(pt.nearest_edge_point).copy())
            payload["nearest_edge_id"] = pt.nearest_edge_id.copy()
            payload["edge_start_node"] = pt.edge_start_node.copy()
            payload["edge_end_node"] = pt.edge_end_node.copy()
            payload["weight_to_start"] = pt.weight_to_start.copy()
            payload["weight_to_end"] = pt.weight_to_end.copy()
        _admit_locked(req, payload, "snap")
    from urban_network_analysis.Topology import _access_points_stage_payload
    _publish_to_run_store(
        req, lambda: _access_points_stage_payload(
            pt, n_points=n_points, log_extras=log_extras))


# ── run-wide serving (increment C): lookup, decode, single-flight ──────

def _decode_topology_bundle(req: PreparedRequest, arrays: dict,
                            metadata: dict) -> _Bundle:
    """Run-store topology generation → lease payload (the B-payload shape
    the restore path consumes).  Arrays arrive fresh from the store's
    verified bytes; ``input_gdf`` is not carried by the stage format —
    the restore validates the stored final geometry (both are
    post-validation content by construction)."""
    from pyproj import CRS
    from urban_network_analysis.Topology import _geoms_from_wkb_payload
    crs = (CRS.from_json(metadata["crs_projjson"])
           if metadata.get("crs_projjson") else None)
    payload = {
        "input_gdf": None,
        "geometry": gpd.GeoSeries(
            _geoms_from_wkb_payload(arrays["geometry_wkb"],
                                    arrays["geometry_wkb_lens"]),
            name="geometry", crs=crs),
        "lengths": arrays["lengths"],
        "weights": arrays["weights"],
        "node_points": arrays["node_points"],
        "start_nodes": arrays.get("start_nodes"),
        "end_nodes": arrays.get("end_nodes"),
        "degrees": arrays["degrees"],
        "z": arrays.get("z"),
        "start_degrees": arrays["start_degrees"],
        "end_degrees": arrays["end_degrees"],
        "min_degrees": arrays["min_degrees"],
        "max_degrees": arrays["max_degrees"],
        "crs": crs,
        "stage_key": req.stage_key,
        "weight_warning": metadata.get("weight_warning"),
        "n_empty_removed": int(metadata.get("n_empty_removed") or 0),
        "n_nodes": int(metadata["n_nodes"]),
        "n_edges": int(metadata["n_edges"]),
        "n_input_features": int(metadata["n_input_features"]),
    }
    return _Bundle(req.key, req.file_sha256, payload, source="run")


def _decode_snap_bundle(req: PreparedRequest, arrays: dict,
                        metadata: dict) -> _Bundle:
    """Run-store snap generation → lease payload (same restore contract)."""
    from pyproj import CRS
    from urban_network_analysis.Topology import _geoms_from_wkb_payload
    crs = (CRS.from_json(metadata["crs_projjson"])
           if metadata.get("crs_projjson") else None)
    uid_array = arrays.get("uid")
    uid = (None if uid_array is None
           else (pd.Series(uid_array) if metadata.get("uid_is_series")
                 else uid_array))
    payload = {
        "input_gdf": None,
        "geometry": gpd.GeoSeries(
            _geoms_from_wkb_payload(arrays["geometry_wkb"],
                                    arrays["geometry_wkb_lens"]),
            name="geometry", crs=crs),
        "node_weight": arrays["node_weight"],
        "uid": uid,
        "snapped": bool(metadata.get("snapped")),
        "stage_key": req.stage_key,
        "n_points": int(metadata["n_points"]),
        "log_extras": [tuple(x) for x in metadata.get("log_extras", [])],
    }
    if payload["snapped"]:
        payload["nearest_edge_point"] = gpd.array.GeometryArray(
            _geoms_from_wkb_payload(arrays["nearest_edge_point_wkb"],
                                    arrays["nearest_edge_point_wkb_lens"]))
        payload["nearest_edge_id"] = arrays["nearest_edge_id"]
        payload["edge_start_node"] = arrays["edge_start_node"]
        payload["edge_end_node"] = arrays["edge_end_node"]
        payload["weight_to_start"] = arrays["weight_to_start"]
        payload["weight_to_end"] = arrays["weight_to_end"]
    return _Bundle(req.key, req.file_sha256, payload, source="run")


def _lookup(req: PreparedRequest) -> Optional[_Bundle]:
    """Two-tier lookup: worker-local lease first, then the attached run
    store (verified read; corruption quarantines and rebuilds).  A run
    hit is decoded and PROMOTED into the worker-local table so this
    worker's remaining rows skip the store read (the promotion takes the
    lease credit the caller's restore will release).  Retention is
    budgeted like publish admission: evict unleased entries to make
    room; when nothing fits, serve WITHOUT retaining — the design's
    NO_FIT law refuses admission, never serving (the produced path
    would have allocated this decoded layer anyway)."""
    global _table_bytes
    bundle = acquire(req)
    if bundle is not None:
        return bundle
    with _lock:
        run = _RUN
    if run is None:
        return None
    from ..cache.store import CacheCorruption
    try:
        store_hit = run.store.require(req.key)  # verified; raises on damage
    except KeyError:
        return None                             # absent: normal miss
    except CacheCorruption:
        with _lock:
            _run_counters["corruption_rebuilds"] += 1
        run.store.get(req.key)                  # quarantine the generation
        return None
    if store_hit.metadata.get("file_sha256") != req.file_sha256:
        # Belt over the content-addressed key: recorded digest must match
        # the request (the same refuse-different-bytes law as tier 1).
        # Nothing is UNREADABLE, so there is nothing to quarantine — the
        # rebuild's republish supersedes the bogus generation in place.
        with _lock:
            _run_counters["corruption_rebuilds"] += 1
        return None
    from ..cache.stages import decode_arrays
    arrays = decode_arrays(store_hit.payload)
    decode = (_decode_topology_bundle if req.layer == "topology"
              else _decode_snap_bundle)
    bundle = decode(req, arrays, dict(store_hit.metadata))
    with _lock:
        old = _table.get(req.key)
        if old is not None:
            _table_bytes -= old.nbytes
        while _table_bytes + bundle.nbytes > _table_budget():
            candidates = [(b.seq, k) for k, b in _table.items()
                          if b.in_use == 0 and k != req.key]
            if not candidates:
                # Serve unretained: pop the key (its old bytes were
                # already released) and hand the lease straight to the
                # caller's restore, which releases it as usual.
                _table.pop(req.key, None)
                _run_counters["promote_skips"] += 1
                _run_counters["tier2_hits"] += 1
                bundle.in_use = 1
                return bundle
            _, victim = min(candidates)
            _table_bytes -= _table.pop(victim).nbytes
            _run_counters["evictions"] += 1
        _table[req.key] = bundle
        _table_bytes += bundle.nbytes
        bundle.in_use = 1  # handed straight to this caller's restore
        _run_counters["tier2_hits"] += 1
    return bundle


def served_topology(topology, req: PreparedRequest) -> bool:
    """Serve one AddNetwork input from either tier; False = build."""
    bundle = _lookup(req)
    if bundle is None:
        return False
    restore_topology(topology, req, bundle)
    return True


def served_access_points(topology, req: PreparedRequest, *, label: str):
    """Serve one Add* point input from either tier; None = build."""
    bundle = _lookup(req)
    if bundle is None:
        return None
    return restore_access_points(topology, req, bundle, label)


@contextmanager
def flight(req: Optional[PreparedRequest]) -> Iterator[bool]:
    """Run-wide compute-once around the produced build (designs/02: the
    lazy gate — first-consumer single-flight AFTER the original
    validation, which ``*_request`` already performed).

    Yields True when the caller builds (owner — or no run tier attached,
    which degrades to the plain increment-B behavior with ZERO extra
    lookups); False when another holder committed this identity while we
    waited — the caller re-serves and returns.  ``req=None`` (unadmitted
    row) yields True without touching anything.  A waiter whose owner
    committed nothing (refusal / unrepresentable payload) is re-elected
    owner by the store.  The wait is bounded by the cohort budget
    (600 s); on timeout the caller builds unserialized rather than
    spinning forever (countered, never silent)."""
    if req is None:
        yield True
        return
    with _lock:
        run = _RUN
    if run is None:
        yield True
        return
    try:
        with run.store.singleflight(req.key,
                                    wait_timeout_s=_SINGLEFLIGHT_WAIT_S) \
                as owned:
            with _lock:
                if owned:
                    _run_counters["cold_builds"] += 1
                else:
                    _run_counters["contention_waits"] += 1
            yield bool(owned)
    except TimeoutError:
        # __enter__ raised before yielding, so the success path did not
        # count this caller: it STILL builds from scratch unserialized —
        # count the cold build too, or the counters undercount work.
        with _lock:
            _run_counters["contention_timeouts"] += 1
            _run_counters["cold_builds"] += 1
        yield True
