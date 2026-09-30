"""Corrected_v1 per-origin betweenness statistics reference (SCIENCE).

Corrected rules for BUG-MADINA-BTN-STATS-NAMEERROR and FACT-BTN-POISON
(registry; FAILURES fai-20260930T1710Z reproduction):

1. The per-origin stats derive from the SAME eligible-destination set
   the routing branch produced, so every branch (closest-destination,
   Huff) binds the distances the stats block needs.  The pinned defect:
   ``eligible_destinations_shortest_distance`` is bound only in the
   Huff branch (betweenness.py:612) and dereferenced unconditionally
   (:858), so closest_destination=True raised UnboundLocalError for
   every origin.

2. Working-graph mutation is scoped: the origin is inserted on enter
   and removed in a guaranteed cleanup path, so an exception anywhere
   inside per-origin processing cannot leave the origin inserted in the
   shared d_graph.  The pinned defect: the bare except around the stats
   block ``continue``d, skipping remove_node_to_graph (:889), so
   processed origins leaked into subsequent origins' routing graphs
   (FAILURES observed 14 -> 16 nodes).

Arithmetic is corrected_v1: explicit binary64, declared orders; the
gravity term uses np.exp (the pinned ``pow(np.e, ...)`` form is legacy
bits, retained there; the reviewed corrected form is exp).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "per_origin_destination_stats",
    "WorkingGraphScope",
    "ScopeInsertionError",
]


class ScopeInsertionError(RuntimeError):
    """The working-graph scope could not restore its inserted node."""


def per_origin_destination_stats(
    eligible_destinations,
    destination_weights,
    beta,
):
    """Per-origin destination diagnostics (corrected, branch-agnostic).

    Parameters
    ----------
    eligible_destinations : dict[node_id, float]
        The eligible destination set with its shortest-path distances,
        exactly as the routing branch produced it (one entry under the
        closest-destination rule, the eligible set under Huff).
    destination_weights : dict[node_id, float]
        Destination weight per node id (float64).
    beta : float
        Gravity decay coefficient.

    Returns a dict with the stats columns the legacy stats block
    writes, plus the per-destination arrays in ascending node-id order
    (declared corrected order):

    - destination_ids (sorted list)
    - destination_distances (np.float64 array, same order)
    - destination_gravities (np.float64 array, same order;
      w / exp(beta * d), explicit binary64 per term)
    - closest_destination_distance (min over the eligible distances)
    - furthest_destination_distance (max)
    - eligible_destinations (count)

    An empty eligible set raises ValueError: the caller's routing stage
    decides what an origin with no eligible destinations means; the
    stats stage refuses to invent values (legacy silently skipped such
    origins inside its bare except -- a safety delta recorded for
    legacy, not corrected behavior).
    """
    if not eligible_destinations:
        raise ValueError(
            "per_origin_destination_stats: empty eligible destination set")
    ids = sorted(eligible_destinations)          # declared order
    distances = np.array(
        [float(eligible_destinations[i]) for i in ids], dtype=np.float64)
    weights = np.array(
        [float(destination_weights[i]) for i in ids], dtype=np.float64)
    beta = float(beta)

    n_ids = len(ids)
    gravities = np.empty(n_ids, dtype=np.float64)
    for j in range(n_ids):
        gravities[j] = weights[j] / float(np.exp(beta * distances[j]))

    # Ordered min/max scans in the declared (ascending id) order.
    closest = float(distances[0])
    furthest = float(distances[0])
    for j in range(1, n_ids):
        d = float(distances[j])
        if d < closest:
            closest = d
        if d > furthest:
            furthest = d

    return {
        "destination_ids": ids,
        "destination_distances": distances,
        "destination_gravities": gravities,
        "closest_destination_distance": closest,
        "furthest_destination_distance": furthest,
        "eligible_destinations": n_ids,
    }


class WorkingGraphScope:
    """Guaranteed insert/remove discipline for per-origin processing.

    Mirrors the pinned call pair (``network.add_node_to_graph`` /
    ``network.remove_node_to_graph`` on a shared working graph) but
    guarantees removal on ANY exit path, including exceptions raised by
    the per-origin diagnostics -- the corrected rule that removes the
    FACT-BTN-POISON leak (an origin left inserted in the shared
    ``d_graph`` distorts every later origin's routing).

    ``graph_api`` must provide ``add_node_to_graph(graph, node)`` and
    ``remove_node_to_graph(graph, node)``.  Re- raising is deliberate:
    the scope fixes the mutation leak, it does not swallow errors.
    """

    def __init__(self, network, working_graph, node):
        self._network = network
        self._graph = working_graph
        self._node = node
        self._inserted = False

    def __enter__(self):
        self._network.add_node_to_graph(self._graph, self._node)
        self._inserted = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._inserted:
            try:
                self._network.remove_node_to_graph(self._graph, self._node)
            except Exception as exc2:            # noqa: BLE001 - never mask
                raise ScopeInsertionError(
                    f"working-graph cleanup failed for node "
                    f"{self._node!r}: {exc2!r}") from exc2
            self._inserted = False
        return False    # never suppress the caller's exception
