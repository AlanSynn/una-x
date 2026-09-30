"""Immutable model adapters over the Madina-parity Zonal state.

The facade's tables and graphs are mutable pinned-state (Madina mutates
Zonal layers and network tables; that mutability IS part of the parity
surface).  Downstream typed consumers (the MADINA_* engines and the
native/GPU backends) must not alias that mutable state: this module
hands out immutable snapshots.  Snapshots are value objects — equal
inputs produce equal snapshots, so they are also the unit of comparison
and (later) of cache/fingerprint identity.

Nothing here changes facade behavior; it only reads it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


def _readonly(values) -> np.ndarray:
    arr = np.asarray(values)
    ro = arr.view()
    ro.flags.writeable = False
    return ro


def _f64_bits(values) -> tuple:
    """Bit-exact representation of float64 values (no tolerance)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return tuple(struct.pack("<d", v) for v in arr)


@dataclass(frozen=True, eq=False)
class NetworkState:
    """Immutable typed view of a prepared Zonal network.

    ids/index preserve the pinned order; arrays are read-only views over
    private copies, so later facade mutation cannot change a snapshot.
    Equality/hash compare array CONTENTS bitwise (the dataclass-generated
    __eq__ cannot: ndarray == ndarray is elementwise).
    """
    node_ids: tuple
    node_xy: np.ndarray          # (n, 2) float64, read-only
    node_type: tuple
    edge_ids: tuple
    edge_start: np.ndarray       # int64, read-only
    edge_end: np.ndarray         # int64, read-only
    edge_weight: np.ndarray      # float64, read-only
    weight_attribute: str | None
    turn_threshold_degree: float | None
    turn_penalty_amount: float | None

    def weight_bits(self) -> tuple:
        return _f64_bits(self.edge_weight)

    def _key(self) -> tuple:
        return (self.node_ids, self.node_type, self.edge_ids,
                self.node_xy.shape, self.node_xy.tobytes(),
                self.edge_start.shape, self.edge_start.tobytes(),
                self.edge_end.shape, self.edge_end.tobytes(),
                self.edge_weight.shape, self.edge_weight.tobytes(),
                self.weight_attribute, self.turn_threshold_degree,
                self.turn_penalty_amount)

    def __eq__(self, other):
        if not isinstance(other, NetworkState):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


@dataclass(frozen=True)
class GraphState:
    """Immutable view of one networkx graph variant (light/d/od)."""
    directed: bool
    edges: tuple                 # sorted ((u, v, weight, id), ...) with weight as float bits
    added_nodes: tuple           # pinned graph.graph["added_nodes"] order

    @classmethod
    def from_nx(cls, graph) -> "GraphState":
        edges = sorted(
            (int(u), int(v), _f64_bits([d["weight"]])[0],
             int(d.get("id", -1)))
            for u, v, d in graph.edges(data=True)
        )
        return cls(
            directed=graph.is_directed(),
            edges=tuple(edges),
            added_nodes=tuple(graph.graph.get("added_nodes", ())),
        )


def network_state(zonal) -> NetworkState:
    """Snapshot the facade network's typed core data (read-only)."""
    nodes = zonal.network.nodes
    edges = zonal.network.edges
    xy = np.column_stack([
        nodes.geometry.x.to_numpy(dtype=np.float64),
        nodes.geometry.y.to_numpy(dtype=np.float64),
    ])
    return NetworkState(
        node_ids=tuple(int(i) for i in nodes.index),
        node_xy=_readonly(xy.copy()),
        node_type=tuple(str(t) for t in nodes["type"]),
        edge_ids=tuple(int(i) for i in edges.index),
        edge_start=_readonly(edges["start"].to_numpy(dtype=np.int64).copy()),
        edge_end=_readonly(edges["end"].to_numpy(dtype=np.int64).copy()),
        edge_weight=_readonly(edges["weight"].to_numpy(dtype=np.float64).copy()),
        weight_attribute=zonal.network.weight_attribute,
        turn_threshold_degree=zonal.network.turn_threshold_degree,
        turn_penalty_amount=zonal.network.turn_penalty_amount,
    )
