"""Old-fails / corrected-passes / legacy-retains triples (dossier 02).

For each confirmed registry defect:
  - OLD-FAILS: the legacy engine violates the intended invariant on the
    registered failing input (executed live against the real kernels);
  - CORRECTED-PASSES: the corrected_v1 reference satisfies it;
  - LEGACY-RETAINS: the legacy behavior is pinned AS RECORDED -- legacy
    outputs are never silently overwritten by corrected expectations
    (the defect is the legacy profile's valid-input behavior until a
    profile switch, and safety deltas are declared, not erased).

Legacy kernel calls here use small, terminating fixtures only (the
dangerous legacy domains live in the supervised probe suite).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.science

REPO = Path(__file__).resolve().parents[2]

# Legacy kernels are imported at RUN time, not module import: pytest
# imports every test module during collection, and
# Engines/Accessibility.py writes os.environ['USE_PYGEOS']='0' at import
# (its geopandas stack pin).  A module-level import here would pollute
# the environment before tests/platform_geometry's facade-import
# cleanliness test spawns its subprocess (directory order runs
# platform_geometry before science).  Reference imports are clean.
from urban_network_analysis.reference import errors            # noqa: E402
from urban_network_analysis.reference.accessibility import (   # noqa: E402
    corrected_reach_gravity_knn,
)
from urban_network_analysis.reference.betweenness_stats import (  # noqa: E402
    WorkingGraphScope,
    per_origin_destination_stats,
)
from urban_network_analysis.reference.search import (          # noqa: E402
    corrected_od_distances,
    corrected_scope_labels,
)


def bits(x) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


@pytest.fixture(scope="module")
def legacy_A():
    """Legacy engine module, imported at run time (see import note)."""
    from urban_network_analysis.Engines import Accessibility as A
    return A


def csr(edges, node_count):
    edges = sorted(edges)
    pointer = np.zeros(node_count + 1, dtype=np.int64)
    for u, v, _ in edges:
        pointer[u + 1] += 1
        pointer[v + 1] += 1
    pointer = np.cumsum(pointer)
    vector = np.zeros(int(pointer[-1]), dtype=np.int64)
    weights = np.zeros(int(pointer[-1]), dtype=np.float64)
    fill = pointer[:-1].copy()
    for u, v, w in edges:
        vector[fill[u]] = v
        weights[fill[u]] = w
        fill[u] += 1
        vector[fill[v]] = u
        weights[fill[v]] = w
        fill[v] += 1
    return pointer, vector, weights


# --------------------------------------------------------------------------
# BUG-FRAC-WEIGHT-TRUNC
# --------------------------------------------------------------------------

class TestFracWeightTrunc:
    def test_triple(self, legacy_A):
        pointer, vector, weights = csr([(0, 1, 10.0)], 2)
        o_idxs = np.array([[0, 0]], dtype=np.int64)
        o_wts = np.array([[0.0, 0.0]], dtype=np.float64)
        d_idxs = np.array([[0, 0], [1, 1]], dtype=np.int64)
        d_term = np.zeros((2, 2), dtype=np.float64)
        d_weights = np.array([0.25, 0.50], dtype=np.float64)

        # OLD-FAILS: legacy reach truncates 0.75 into an int64
        legacy = legacy_A.integrated_scope_access(
            o_idxs, o_wts, pointer, vector, weights,
            np.array([True, True]), d_idxs, d_term, d_weights,
            0.0, 0.0, 1.0, 1.0, "none", np.array([1.0, 0.5]), 100.0)
        legacy_reach = legacy[0]
        assert legacy_reach.dtype == np.int64
        assert int(legacy_reach[0]) != 0.75        # the defect, live

        # CORRECTED-PASSES: binary64 reach, exact value
        reach, *_ = corrected_reach_gravity_knn(
            np.array([10.0, 10.0]), d_weights, 100.0,
            0.0, 0.0, 0.0, 0.0, np.array([1.0]), "none")
        assert isinstance(reach, float)
        assert bits(reach) == bits(0.75)

        # LEGACY-RETAINS: legacy output pinned as recorded (int64, 0)
        assert bits(float(legacy_reach[0])) == bits(0.0)


# --------------------------------------------------------------------------
# BUG-SAME-EDGE-OD
# --------------------------------------------------------------------------

class TestSameEdgeOd:
    def test_triple(self, legacy_A):
        pointer, vector, weights = csr([(0, 1, 100.0)], 2)
        o_idxs = np.array([[0, 1]], dtype=np.int64)
        o_wts = np.array([[10.0, 90.0]], dtype=np.float64)
        d_idxs = np.array([[0, 1]], dtype=np.int64)
        d_wts = np.array([[90.0, 10.0]], dtype=np.float64)

        # OLD-FAILS: legacy assembles endpoint routes only -> 100, not
        # the direct partial 80
        legacy_od = legacy_A.od_compact_vector_node_view_scope(
            o_idxs, o_wts, pointer, vector, weights,
            np.array([True, True]), 1000.0, 1, d_idxs, d_wts)
        assert bits(legacy_od[0, 0]) == bits(100.0)
        assert bits(legacy_od[0, 0]) != bits(80.0)

        # CORRECTED-PASSES: direct along-edge arc admitted -> 80
        od = corrected_od_distances(
            o_idxs, o_wts, ["e0"], pointer, vector, weights, 1000.0,
            ["e0"], np.array([[0, 1]], dtype=np.int64), d_idxs, d_wts)
        assert bits(od[0, 0]) == bits(80.0)

        # LEGACY-RETAINS: the legacy value stays exactly as recorded
        assert bits(legacy_od[0, 0]) == bits(100.0)


# --------------------------------------------------------------------------
# BUG-COINCIDENT-SEEDS
# --------------------------------------------------------------------------

class TestCoincidentSeeds:
    def test_triple(self, legacy_A):
        pointer, vector, weights = csr([(0, 1, 10.0)], 2)
        o_idxs = np.array([[0, 0]], dtype=np.int64)
        o_wts = np.array([[0.0, 5.0]], dtype=np.float64)
        net = np.array([True, True])

        # OLD-FAILS: legacy seed overwrite keeps the end split 5.0
        legacy_scope, _ = legacy_A.compact_vector_node_view_scope(
            o_idxs[0], o_wts[0], pointer, vector, weights, net, 100.0, 0)
        assert bits(legacy_scope[0]) == bits(5.0)

        # CORRECTED-PASSES: minimum rule -> 0.0
        label = corrected_scope_labels(
            o_idxs, o_wts, pointer, vector, weights, 100.0)
        assert bits(label[0]) == bits(0.0)
        # and the neighbor route is consistent with the kept seed
        assert bits(label[1]) == bits(10.0)

        # LEGACY-RETAINS: legacy overwrite bit pinned as recorded
        assert bits(legacy_scope[0]) == bits(5.0)


# --------------------------------------------------------------------------
# BUG-NONFINITE-NOVALIDATION
# --------------------------------------------------------------------------

class TestNonfiniteValidation:
    def test_triple(self, legacy_A):
        pointer, vector, weights = csr([(0, 1, 10.0)], 2)
        o_idxs = np.array([[0, 0]], dtype=np.int64)
        o_wts = np.array([[0.0, 0.0]], dtype=np.float64)
        net = np.array([True, True])

        # OLD-FAILS: legacy accepts a NaN cost silently (neighbor stays
        # sentinel: no error, no marker) ...
        nan_w = weights.copy()
        nan_w[0] = np.nan          # directed arc 0 -> 1
        legacy_nan, _ = legacy_A.compact_vector_node_view_scope(
            o_idxs[0], o_wts[0], pointer, vector, nan_w, net, 100.0, 0)
        assert legacy_nan[1] > 100.0

        # ... and accepts negative costs, returning invalid negative
        # labels instead of a rejection
        neg_w = weights.copy()
        neg_w[:] = -5.0
        legacy_neg, _ = legacy_A.compact_vector_node_view_scope(
            o_idxs[0], o_wts[0], pointer, vector, neg_w, net, 100.0, 0)
        assert legacy_neg[1] < 0.0

        # CORRECTED-PASSES: typed rejections before any search
        with pytest.raises(errors.NonFiniteCostError):
            corrected_scope_labels(
                o_idxs, o_wts, pointer, vector, nan_w, 100.0)
        with pytest.raises(errors.NegativeCostError):
            corrected_scope_labels(
                o_idxs, o_wts, pointer, vector, neg_w, 100.0)

        # LEGACY-RETAINS: pinned silent acceptance reproduced above and
        # retained in the supervised probe capsule (nonfinite_validation
        # in the S1 evidence); no legacy output is "fixed" in place.


# --------------------------------------------------------------------------
# BUG-SVC-GEOMCOLL is carried by the FAILURES INC03 reproduction
# (geometry-domain defect; its corrected geometry rule is exercised by
# the MADINA_ACCESS service-area task against this reference's
# discipline: typed rejection / component preservation, no fabricated
# hull area).  See evidence disposition table.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# BUG-MADINA-BTN-STATS-NAMEERROR / FACT-BTN-POISON
# --------------------------------------------------------------------------

class _FakeNetwork:
    """Minimal stand-in exposing the pinned add/remove call pair."""

    def __init__(self):
        self.added = []
        self.removed = []
        self.fail_on_remove = False

    def add_node_to_graph(self, graph, node):
        self.added.append(int(node))
        graph.add_node(int(node))

    def remove_node_to_graph(self, graph, node):
        if self.fail_on_remove:
            raise RuntimeError("injected removal failure")
        self.removed.append(int(node))
        graph.remove_node(int(node))


class TestBtnStats:
    def test_corrected_stats_bind_on_closest_destination_branch(self):
        """CORRECTED-PASSES: the stats derive from whatever eligible set
        the branch produced -- a single-destination (closest) set is as
        valid as a Huff set; no unbound diagnostics variable."""
        stats = per_origin_destination_stats(
            {7: 42.0}, {7: 3.0}, 0.05)
        assert stats["eligible_destinations"] == 1
        assert bits(stats["closest_destination_distance"]) == bits(42.0)
        assert bits(stats["furthest_destination_distance"]) == bits(42.0)
        assert bits(stats["destination_gravities"][0]) == bits(
            3.0 / float(np.exp(0.05 * 42.0)))

    def test_corrected_stats_reject_empty_eligible_set(self):
        with pytest.raises(ValueError):
            per_origin_destination_stats({}, {}, 0.05)

    def test_working_graph_scope_guarantees_removal(self):
        """CORRECTED-PASSES (FACT-BTN-POISON rule): an exception inside
        per-origin processing cannot leave the origin inserted."""
        import networkx as nx

        net = _FakeNetwork()
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        with pytest.raises(NameError):
            with WorkingGraphScope(net, graph, 1):
                assert 1 in graph.nodes
                # the pinned defect's raising diagnostics, reproduced in
                # miniature: a diagnostics variable that the branch never
                # bound (pinned betweenness.py raises UnboundLocalError;
                # here the same class of failure is a plain NameError)
                eligible_destinations_shortest_distance  # noqa: B018
        assert 1 not in graph.nodes            # removed on the error path
        assert net.added == [1] and net.removed == [1]

    def test_working_graph_scope_cleanup_failure_is_masked_never(self):
        """Cleanup failures surface as ScopeInsertionError chained to the
        original error -- never silently swallowed, never suppressing the
        caller's exception."""
        import networkx as nx

        net = _FakeNetwork()
        net.fail_on_remove = True
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        from urban_network_analysis.reference.betweenness_stats import (
            ScopeInsertionError,
        )
        with pytest.raises(ScopeInsertionError):
            with WorkingGraphScope(net, graph, 1):
                raise ValueError("origin diagnostics failed")

    def test_legacy_retains_pinned_defect_pattern(self):
        """LEGACY-RETAINS: the pinned madina source keeps the defective
        pattern (stats dereference outside the closest-destination
        branch; bare-except continue skipping the removal).  Live
        behavior reproduction lives in the FAILURES incident capsules
        (evidence/incidents/fai-20260930T1710Z, FACT-BTN-POISON)."""
        pinned = (REPO / ".refs" / "madina" / "src" / "madina" / "una"
                  / "betweenness.py").read_text()
        assert "if closest_destination:" in pinned
        assert "eligible_destinations_shortest_distance.min()" in pinned
        # the binding happens only under the Huff else-branch (line ~612)
        bind_at = pinned.index("eligible_destinations_shortest_distance = np.array")
        closest_at = pinned.index("if closest_destination:")
        stats_at = pinned.index("eligible_destinations_shortest_distance.min()")
        assert closest_at < bind_at < stats_at
