"""Independent oracle: verbatim transcription of the ORIGINAL baseline CSR
construction from the audited upstream builders (blob
246fe439...Accessibility.py / 69061684...AccessibilityWElevation.py).

This is the golden reference for L0 tests. It must remain byte-for-byte the
same algorithm as upstream; candidate code must never be imported here.
"""
from __future__ import annotations

import numpy as np


def original_builder_core(node_count, start_nodes, end_nodes, weights_ab, weights_ba):
    """Original masked-scan CSR construction (count pass + fill pass).

    Returns (adjacency_pointer, adjacency_vector, adjacency_vector_weights,
    adjacynct_vector_network_node) exactly as the baseline np.array(...) calls
    produce them.
    """
    start_nodes = np.asarray(start_nodes)
    end_nodes = np.asarray(end_nodes)
    weights_ab = np.asarray(weights_ab)
    weights_ba = np.asarray(weights_ba)

    adjacency_pointer = np.zeros(node_count + 1, dtype=np.int64)
    adjacency_vector = []
    adjacency_vector_weights = []
    adjacynct_vector_network_node = []

    # Count neighbors per node
    for node in range(node_count):
        mask_start = start_nodes == node
        mask_end = end_nodes == node
        count = np.sum(mask_start) + np.sum(mask_end)
        adjacency_pointer[node + 1] = adjacency_pointer[node] + count

    # Fill adjacency vectors (forward incidences then reverse incidences)
    for node in range(node_count):
        mask_start = start_nodes == node
        end_neighbors = end_nodes[mask_start]
        neighbor_weights = weights_ab[mask_start]
        is_network = np.ones(len(end_neighbors), dtype=np.bool_)

        adjacency_vector.extend(end_neighbors)
        adjacency_vector_weights.extend(neighbor_weights)
        adjacynct_vector_network_node.extend(is_network)

        mask_end = end_nodes == node
        start_neighbors = start_nodes[mask_end]
        neighbor_weights = weights_ba[mask_end]
        is_network = np.ones(len(start_neighbors), dtype=np.bool_)

        adjacency_vector.extend(start_neighbors)
        adjacency_vector_weights.extend(neighbor_weights)
        adjacynct_vector_network_node.extend(is_network)

    return (
        adjacency_pointer,
        np.array(adjacency_vector, dtype=np.int64),
        np.array(adjacency_vector_weights, dtype=np.float64),
        np.array(adjacynct_vector_network_node, dtype=np.bool_),
    )
