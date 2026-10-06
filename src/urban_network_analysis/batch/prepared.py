"""Worker-local prepared-layer reuse — una_10x P02 increment B (private).

Contract: campaigns/una_10x/tenx-20261005T2331Z/p02d/prepared_contract.json
(increment B: worker-local reuse; increment C adds the run-wide producer).

Scope: ONE process reuses ITS OWN prepared layers across repeated Add*
calls (batch rows re-loading identical inputs).  There is no cross-process
coordination, no persistent store and no public API here — this module is
reached only from Topology.AddNetwork/BuildAccessPoints/Add* internals.

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
  produced path would have freshly built (fresh mutable outputs).
- Engagement is .geojson-first (contract admission_domain, designs/02):
  other formats return MISS and fall through to the produced path
  unchanged.  A produced failure never publishes (payloads are
  post-validation content); empty layers are refused.
- Kill switch: UNA_PREPARED=0 disables admission entirely (paired-control
  duty); the process-global table is cleared via reset().

Budget note (prepared_memory.json): the worker-local table is unbounded in
this increment by design — batch cohorts are bounded row sets and payload
size is the decoded layer itself; growth policy/eviction under live leases
is increment C's duty (PP09).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import __version__ as _PACKAGE_VERSION
from ..cache import identity as _ckid

_ENV_FLAG = "UNA_PREPARED"
_SCHEMA_TOPOLOGY = "una:prepared:topology:v1"
_SCHEMA_SNAP = "una:prepared:snap:v1"

_lock = threading.Lock()
_table: Dict[str, "_Bundle"] = {}
_stats: Dict[str, Dict[str, int]] = {
    "topology": {"requests": 0, "hits": 0, "misses": 0, "publishes": 0,
                 "refusals": 0},
    "snap": {"requests": 0, "hits": 0, "misses": 0, "publishes": 0,
             "refusals": 0},
}


def enabled() -> bool:
    """Kill switch: UNA_PREPARED=0 turns prepared reuse off entirely."""
    return os.environ.get(_ENV_FLAG, "1") != "0"


def reset() -> None:
    """Clear the worker-local table and counters (test isolation)."""
    with _lock:
        _table.clear()
        for counters in _stats.values():
            for k in counters:
                counters[k] = 0


def stats() -> Dict[str, Dict[str, int]]:
    with _lock:
        return {layer: dict(counters) for layer, counters in _stats.items()}


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
    """Immutable-by-discipline lease payload (private to this module)."""

    __slots__ = ("key", "file_sha256", "payload")

    def __init__(self, key: str, file_sha256: str, payload: dict) -> None:
        self.key = key
        self.file_sha256 = file_sha256
        self.payload = payload


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
    contract's store-side truth guard and pops any bundle that fails it."""
    with _lock:
        bundle = _table.get(req.key)
        if bundle is None:
            _stats[req.layer]["misses"] += 1
            return None
        if bundle.file_sha256 != req.file_sha256:
            _table.pop(req.key, None)
            _stats[req.layer]["refusals"] += 1
            return None
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
    topology.logger.log(
        "Network added",
        f"Topology restored from prepared layer {req.key[:12]}… "
        f"(worker-local).", v=2)


def restore_access_points(topology, req: PreparedRequest, bundle: _Bundle,
                          label: str):
    """Rebuild one AccessPoints object from an immutable lease (produced
    field set + chronology; per-consumer copies throughout)."""
    from urban_network_analysis.Topology import AccessPoints
    p = bundle.payload
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
    topology.logger.log(
        "Build AccessPoints",
        f"Snapped layer restored from prepared layer {req.key[:12]}… "
        f"(worker-local).", v=2)
    return pt


# ── publish (post-success only; payloads are isolated copies) ───────────

def publish_topology(req: PreparedRequest, net, gdf, *, crs,
                     weight_warning, n_empty_removed, n_nodes, n_edges,
                     n_input_features) -> None:
    """Publish one topology bundle into the worker-local table.  Called
    only after the produced path fully succeeded (payloads are
    post-validation content).  Also records ``net._stage_key`` so
    dependent snap identities can chain the parent — the pure produced
    path leaves it unset.  Everything is copied at publish: the producing
    row keeps mutating its own objects (engine fold) and must never reach
    lease state."""
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
        _table[req.key] = _Bundle(req.key, req.file_sha256, payload)
        _stats["topology"]["publishes"] += 1
        net._stage_key = req.stage_key


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
        _table[req.key] = _Bundle(req.key, req.file_sha256, payload)
        _stats["snap"]["publishes"] += 1
