"""Immutable model adapter tests (TOPOLOGY objective).

Snapshots must be frozen value objects: read-only arrays, stable bits,
and immunity to later facade mutation.  They are the typed hand-off for
the MADINA_* engines and later cache/fingerprint identity.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.platform_geometry


def _build_grid():
    from urban_network_analysis.compat.madina.zonal import Zonal
    from _scenario import write_fixtures
    import tempfile
    root = tempfile.mkdtemp(prefix="adapter_")
    paths = write_fixtures(root, "grid")
    random.seed(20260930)
    z = Zonal()
    z.load_layer("streets", paths["streets"])
    z.load_layer("origins", paths["origins"])
    z.load_layer("destinations", paths["destinations"])
    z.create_street_network(source_layer="streets",
                            node_snapping_tolerance=0.0)
    z.insert_node(layer_name="origins", label="origin")
    z.insert_node(layer_name="destinations", label="destination")
    z.create_graph()
    return z


def test_snapshot_is_immutable_and_stable():
    from urban_network_analysis.compat.madina.zonal.adapter import (
        GraphState, network_state)
    z = _build_grid()
    snap = network_state(z)
    gsnap = GraphState.from_nx(z.network.d_graph)

    for name in ("node_xy", "edge_start", "edge_end", "edge_weight"):
        arr = getattr(snap, name)
        with pytest.raises(ValueError):
            arr[0] = arr[0]
    with pytest.raises(Exception):
        snap.node_ids = ()
    with pytest.raises(Exception):
        gsnap.edges = ()

    # equal inputs -> equal snapshots; re-snapshot is bit-stable
    assert network_state(z) == snap
    assert GraphState.from_nx(z.network.d_graph) == gsnap
    assert snap.weight_bits() == snap.weight_bits()


def test_snapshot_survives_facade_mutation():
    from urban_network_analysis.compat.madina.zonal.adapter import (
        network_state)
    z = _build_grid()
    snap = network_state(z)

    # mutate the live facade after snapshotting (clear_nodes rebinds
    # network.nodes; set_node_value edits a cell in place)
    z.network.set_node_value(z.network.nodes.index[0], "degree", 99)
    z.clear_nodes()
    assert network_state(z) != snap          # live state moved...
    assert len(snap.node_ids) > len(network_state(z).node_ids)
    # ...while the snapshot kept the pre-mutation values
    assert snap.node_type.count("origin") == 2
    assert snap.node_type.count("destination") == 2
    assert snap.turn_threshold_degree == 45.0


def test_weight_bits_are_exact_not_tolerance():
    from urban_network_analysis.compat.madina.zonal.adapter import _f64_bits
    a = [0.1 + 0.2]
    b = [0.3]
    assert a[0] != b[0]                       # sanity: distinct doubles
    assert _f64_bits(a) != _f64_bits(b)       # bit-level comparator agrees
    assert _f64_bits([1.0]) == _f64_bits([1.0])
