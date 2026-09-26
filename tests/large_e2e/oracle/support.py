"""Shared lazy singletons for the H03 oracle tests (single in-process
B0 load per pytest session; no candidate code is ever imported)."""
from __future__ import annotations

import json
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = ORACLE_DIR / "golden"

_B0 = None


def b0():
    global _B0
    if _B0 is None:
        from b0_import import load_arm
        _B0 = load_arm()  # defaults to arm root A (B0)
    return _B0


def golden_hashes():
    with open(GOLDEN_DIR / "golden_b0.hashes.json") as fh:
        return json.load(fh)


def golden_meta():
    with open(GOLDEN_DIR / "golden_b0.json") as fh:
        return json.load(fh)


def flow_engine_case(b0, settings, node_flow=True):
    """Build the flow engine + settings pair used by F1/F2/F3 tests."""
    import fixtures
    import stub_topology

    spec = fixtures.flow_network_spec()
    overrides = fixtures.flow_settings_overrides(node_flow=node_flow)
    eff_settings = settings
    if settings is None:
        eff_settings = stub_topology.make_settings(
            b0.Settings, accessibility=False, **overrides)
    eng = b0.AggregateFlow(stub_topology.StubFlowTopology(spec))
    eng.num_threads = 1
    eng.Centrality(eff_settings)  # build CSR/digraph; final arrays replaced below
    return eng, eff_settings, spec
