"""F2 precondition + route-selection tests (ruling condition (c) and
the dense-overlap crossover (f)).

_gradient_slices_strictly_ascending is the dossier step-1 gate: any
non-strictly-ascending slice disables the fast route for the WHOLE run
(unchanged fallback). The vectorized implementation is differential-
tested against a brute-force per-slice check on adversarial boundary
cases — including the empty-slice masking trap (a violation in the last
pair of a slice followed by an empty slice must STILL be caught).
"""
from __future__ import annotations

import numpy as np

import pytest as _pytest
from _legacy_paths import legacy_layout_present as _llp
if not _llp():
    _pytest.skip('historical large_e2e layout not present '
                 '(set UNA_LEGACY_WORKSPACE)', allow_module_level=True)

from urban_network_analysis.Engines import AggregateFlow as AF
from harness_f2 import record


def brute_force_ascending(g_indptr, g_nodes):
    """O(N) per-slice reference: every slice strictly ascending."""
    n_dest = g_indptr.shape[0] - 1
    for d in range(n_dest):
        s = g_nodes[g_indptr[d]:g_indptr[d + 1]]
        if s.shape[0] > 1 and not np.all(np.diff(s) > 0):
            return False
    return True


def test_check_matches_bruteforce_adversarial():
    cases = {
        "all_ascending": (
            [0, 2, 5], [10, 11, 20, 21, 22]),
        "within_slice_descending": (
            [0, 2, 5], [10, 11, 22, 21, 20]),
        "within_slice_duplicate": (
            [0, 2, 5], [10, 11, 20, 20, 21]),
        "cross_boundary_unordered_ok": (
            [0, 2, 5], [10, 30, 5, 6, 7]),
        "cross_boundary_duplicate_ok": (
            [0, 2, 5], [10, 30, 30, 31, 32]),
        "empty_slice_middle": (
            [0, 2, 2, 5], [10, 11, 20, 21, 22]),
        # False-positive trap: an empty slice between two non-empty
        # ones must not leave the legal cross-boundary pair checked.
        "empty_slice_legal_cross": (
            [0, 1, 1, 3], [10, 3, 4]),
        # The masking trap: violation in the LAST pair of a slice whose
        # SUCCESSOR slice is empty — the boundary exemption must not
        # swallow this within-slice violation.
        "violation_before_empty_slice": (
            [0, 3, 3, 5], [10, 11, 9, 20, 21]),
        "violation_after_empty_slice": (
            [0, 1, 1, 4], [10, 20, 19, 18]),
        "single_slice": ([0, 3], [7, 3, 5]),
        "empty_all": ([0, 0, 0], []),
        "no_slices": ([0], []),
    }
    for name, (indptr, nodes) in cases.items():
        gi = np.asarray(indptr, dtype=np.int64)
        gn = np.asarray(nodes, dtype=np.int64)
        got = AF._gradient_slices_strictly_ascending(gi, gn)
        want = brute_force_ascending(gi, gn)
        record(f"precondition/{name}", got=bool(got), want=bool(want))
        assert got == want, \
            f"{name}: vectorized check {got} != brute force {want}"


def test_check_matches_bruteforce_random():
    rng = np.random.default_rng(20260927)
    for trial in range(200):
        n_nodes = int(rng.integers(0, 15))
        n_cuts = int(rng.integers(0, 6))
        cuts = np.sort(rng.integers(0, n_nodes + 1, size=n_cuts))
        g_indptr = np.concatenate(([0], cuts, [n_nodes])).astype(np.int64)
        g_nodes = rng.integers(0, 8, size=n_nodes).astype(np.int64)
        got = AF._gradient_slices_strictly_ascending(g_indptr, g_nodes)
        want = brute_force_ascending(g_indptr, g_nodes)
        assert got == want, \
            (f"random trial {trial}: vectorized {got} != brute force "
             f"{want} (indptr={g_indptr.tolist()}, "
             f"nodes={g_nodes.tolist()})")
    record("precondition/random_trials", trials=200)


def test_use_local_route_crossover():
    """(f) deterministic dense-overlap crossover: fast route iff the
    precondition holds AND slice <= n_total // _F2_LOCAL_SLICE_FRAC."""
    frac = AF._F2_LOCAL_SLICE_FRAC
    assert frac >= 2
    # Precondition false -> never.
    assert AF._use_local_route(False, 1, 10**6) is False
    # Tiny slice of a big CSR -> fast.
    assert AF._use_local_route(True, 1, 10**6) is True
    # Exactly at the threshold -> fast (<=).
    threshold = max(1, 10**6 // frac)
    assert AF._use_local_route(True, threshold, 10**6) is True
    # One past the threshold -> baseline.
    assert AF._use_local_route(True, threshold + 1, 10**6) is False
    # Degenerate tiny CSR: threshold floors at 1.
    assert AF._use_local_route(True, 1, 1) is True
    assert AF._use_local_route(True, 2, 1) is False
    record("route/crossover", frac=int(frac))
