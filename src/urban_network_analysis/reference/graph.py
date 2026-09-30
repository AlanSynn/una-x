"""Validated CSR graph for the corrected_v1 reference profile.

corrected_v1 searches run only on finite nonnegative cost graphs
(NUMERICS.md validation fixtures: "nonfinite rejection"; dossier 02 #4:
"Prove termination on admitted finite nonnegative graphs").  Every
reference search entry point funnels through ``validate_csr`` so the
admitted domain is enforced in one reviewed place, in pure Python --
outside any fastmath context by construction.
"""
from __future__ import annotations

import math

import numpy as np

from .errors import (
    MalformedGraphError,
    NegativeCostError,
    NonFiniteCostError,
    OutOfRangeTerminalError,
)


def validate_csr(
    adjacency_pointer,
    adjacency_vector,
    adjacency_weights,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Validate CSR structure and cost domain; return (node_count, ...) views.

    Checks (each a typed rejection, first failure wins):
    - pointer is 1-D int64, vector/weights are 1-D int64/float64;
    - pointer[0] == 0, nondecreasing, pointer[-1] == len(vector);
    - every neighbor index lies in [0, node_count);
    - every cost is finite (NaN/inf -> NonFiniteCostError) and
      nonnegative (NegativeCostError).

    Self-loops and parallel edges (same neighbor repeated) are ADMITTED
    -- they are part of the NUMERICS fixture list, and the search rules
    handle them (self-loops never improve labels; parallel arcs compete
    by strict improvement).
    """
    pointer = np.asarray(adjacency_pointer)
    vector = np.asarray(adjacency_vector)
    weights = np.asarray(adjacency_weights)

    if pointer.ndim != 1 or pointer.dtype != np.int64:
        raise MalformedGraphError(
            f"adjacency_pointer must be 1-D int64, got {pointer.dtype} "
            f"ndim={pointer.ndim}")
    if vector.ndim != 1 or vector.dtype != np.int64:
        raise MalformedGraphError(
            f"adjacency_vector must be 1-D int64, got {vector.dtype} "
            f"ndim={vector.ndim}")
    if weights.ndim != 1 or weights.dtype != np.float64:
        raise MalformedGraphError(
            f"adjacency_weights must be 1-D float64, got {weights.dtype} "
            f"ndim={weights.ndim}")
    if pointer.shape[0] < 1:
        raise MalformedGraphError("adjacency_pointer needs at least one slot")
    node_count = pointer.shape[0] - 1

    if pointer[0] != 0:
        raise MalformedGraphError(f"pointer[0] must be 0, got {pointer[0]}")
    prev = 0
    for v in range(node_count):
        cur = int(pointer[v + 1])
        if cur < prev:
            raise MalformedGraphError(
                f"pointer not nondecreasing at node {v}: {prev} -> {cur}")
        prev = cur
    if int(pointer[node_count]) != vector.shape[0]:
        raise MalformedGraphError(
            f"pointer[-1]={int(pointer[node_count])} != "
            f"len(vector)={vector.shape[0]}")

    for j in range(vector.shape[0]):
        neighbor = int(vector[j])
        if neighbor < 0 or neighbor >= node_count:
            raise MalformedGraphError(
                f"adjacency_vector[{j}]={neighbor} outside [0, {node_count})")

    for j in range(weights.shape[0]):
        c = float(weights[j])
        if math.isnan(c):
            raise NonFiniteCostError(f"adjacency_weights[{j}] is NaN")
        if math.isinf(c):
            raise NonFiniteCostError(f"adjacency_weights[{j}] is infinite")
        if c < 0.0:
            raise NegativeCostError(f"adjacency_weights[{j}]={c!r} < 0")

    return node_count, pointer, vector, weights


def validate_cutoff(cutoff) -> float:
    """Radius must be finite and nonnegative."""
    c = float(cutoff)
    if math.isnan(c) or math.isinf(c):
        raise NonFiniteCostError(f"cutoff is not finite: {cutoff!r}")
    if c < 0.0:
        raise NegativeCostError(f"cutoff={c!r} < 0")
    return c


def validate_terminals(
    terminal_idxs,
    terminal_weights,
    node_count: int,
    *,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (n, 2) terminal index array and (n, 2) split weights.

    Splits must be finite and nonnegative (they are distances along the
    host edge); endpoint indices must lie in [0, node_count).
    """
    idxs = np.asarray(terminal_idxs)
    wts = np.asarray(terminal_weights)
    if idxs.ndim != 2 or idxs.shape[1] != 2 or idxs.dtype != np.int64:
        raise OutOfRangeTerminalError(
            f"{what} terminal_idxs must be (n, 2) int64, got "
            f"{idxs.dtype} shape={idxs.shape}")
    if (idxs < 0).any() or (idxs >= node_count).any():
        bad = int(np.argmin((idxs >= 0) & (idxs < node_count)))
        raise OutOfRangeTerminalError(
            f"{what} terminal endpoint out of [0, {node_count}) at row {bad}")
    if wts.shape != idxs.shape or wts.dtype != np.float64:
        raise NonFiniteCostError(
            f"{what} terminal_weights must be (n, 2) float64 matching idxs, "
            f"got {wts.dtype} shape={wts.shape}")
    for i in range(wts.shape[0]):
        for k in range(2):
            w = float(wts[i, k])
            if math.isnan(w) or math.isinf(w):
                raise NonFiniteCostError(
                    f"{what} terminal_weights[{i},{k}] not finite: {w!r}")
            if w < 0.0:
                raise NegativeCostError(
                    f"{what} terminal_weights[{i},{k}]={w!r} < 0")
    return idxs, wts


__all__ = [
    "validate_csr",
    "validate_cutoff",
    "validate_terminals",
]
