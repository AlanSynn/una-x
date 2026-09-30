"""Registry-hypothesis characterization assertions (SCIENCE unit S1).

Each test executes its probe in a bounded subprocess (conftest.run_probe)
and asserts the RECORDED characterization of the baseline engines / pinned
madina stack.  A probe capsule records behavior; these assertions pin the
promotions and refutations that flow into the task evidence disposition
table (never silently — every registry reconciliation is also recorded in
the SCIENCE receipt).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.science

from conftest import (  # noqa: E402  (path set up by conftest)
    BASELINE_PROBES,
    MADINA_PROBES,
    baseline_runner,
    madina_runner,  # noqa: F401  (fixture)
    run_probe,
)


class TestFracWeightTrunc:
    """BUG-FRAC-WEIGHT-TRUNC promotion to reproduced.

    reach_gravity_knn_access allocates the reach vector with
    o_terminal_idxs.dtype (int64) and the njit kernel stores the
    binary64 weighted sum into it, so fractional reach silently
    truncates toward zero.
    """

    def test_fractional_reach_truncated_to_int(self, baseline_runner):
        cap = run_probe("frac_weight_trunc", baseline_runner)
        assert cap["ok"], cap
        obs = cap["observations"]
        assert obs["exact_weighted_sum_binary64"] == 0.75
        assert obs["reach_dtype"] == "int64"
        assert obs["reach_value"] == 0
        assert obs["truncated"] is True


class TestSameEdgeOd:
    """BUG-SAME-EDGE-OD promotion to reproduced.

    Origin at 10 m and destination at 90 m on one 100 m edge: the direct
    partial impedance is 80 m, but the kernel assembles min(endpoint
    routes) = 100 m (seed init routes through both split endpoints).
    """

    def test_endpoint_route_beats_direct_partial(self, baseline_runner):
        cap = run_probe("same_edge_od", baseline_runner)
        assert cap["ok"], cap
        obs = cap["observations"]
        assert obs["od_distance"] == 100.0
        assert obs["direct_partial_impedance"] == 80.0
        assert obs["matches_endpoint_routes_not_direct"] is True


class TestCoincidentSeeds:
    """BUG-COINCIDENT-SEEDS promotion to reproduced.

    Seed init assigns the start weight then overwrites with the end
    weight (no min), so a coincident-terminal origin keeps the end split
    weight; the neighbor label then comes from the stale queue entry of
    the discarded start weight.
    """

    def test_seed_overwrites_rather_than_mins(self, baseline_runner):
        cap = run_probe("coincident_seeds", baseline_runner)
        assert cap["ok"], cap
        obs = cap["observations"]
        assert obs["coincident_node_label"] == obs["overwrite_rule_label"]
        assert obs["matches_overwrite_not_min"] is True
        # mechanism cross-check: the neighbor relaxes from the stale
        # (discarded) start-seed entry, not from the kept end weight
        assert obs["neighbor_label"] == obs["neighbor_via_stale_start_entry"]


class TestNonfiniteValidation:
    """BUG-NONFINITE-NOVALIDATION — partially confirmed.

    Confirmed: NaN cost silently unreachable; negative costs accepted and
    propagated (invalid negative labels, no rejection); zero-cost cycle
    terminates on the pinned 2-node shape.  NOT observed: sentinel
    overflow at extreme radius (the ones(...) + cutoff sentinel stays
    finite because nextafter(max, 0) + 1 does not round up to inf here —
    recorded as a non-observation, not a pass).
    """

    def test_recorded_kernel_behavior_under_invalid_costs(self, baseline_runner):
        cap = run_probe("nonfinite_validation", baseline_runner)
        assert cap["ok"], cap
        obs = cap["observations"]
        # NaN: no validation error, neighbor silently beyond reach
        assert obs["nan_edge_silently_unreachable"] is True
        assert obs["nan_edge_neighbor_label"] > 100.0
        # zero-cost cycle: terminates (termination is NOT the defect on
        # this shape; acceptance of invalid cost graphs is)
        assert obs["zero_cost_cycle_terminated"] is True
        # negative costs accepted: labels go negative, kernel returns
        assert obs["negative_cycle"] == "terminated"
        assert any(lab < 0.0 for lab in obs["negative_cycle_labels"])
        # sentinel overflow NOT observed on this stack — recorded honestly
        assert obs["sentinel_overflow_label_init"] is False
        assert obs["extreme_cutoff_reachable_label"] < float("inf")


class TestPrepZeroCoord:
    """BUG-PREP-ZERO-COORD-DROP — REFUTED on the pinned reference stack.

    The hypothesis: pinned prepare_geometry strips Z with
    tuple(filter(None, [x, y])), and filter(None) drops 0.0, so planar
    coordinates equal to zero are eaten once the layer carries Z.  The
    executed probe shows they are PRESERVED on shapely 2.1.2: transform's
    vectorized path calls _to_2d per AXIS (xs, ys, zs tuples), and a
    non-empty axis tuple is truthy — filter(None) can only drop whole
    (never-occurring) empty axes.  The registry entry is refuted as
    stated for single-part LineStrings on the pinned environment; the
    per-coordinate fallback path (shapely listcomp branch) is unreachable
    for the geometry classes prepare_geometry receives (multi-part
    inputs crash earlier: BUG-MLS-MIXED-CRASH).
    """

    def test_planar_zeros_preserved_on_pinned_stack(self, madina_runner):
        cap = run_probe("prep_zero_coord", madina_runner)
        assert cap["ok"], cap
        obs = cap["observations"]
        # control row unaffected
        assert obs["control_row_first_vertex"] == [50.0, 50.0]
        # the refutation itself: (0,0) survives the pinned Z-strip
        assert obs["planar_zeros_dropped"] is False
        assert obs["zero_row_first_vertex"] == [0.0, 0.0]
