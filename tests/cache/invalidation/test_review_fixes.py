"""Regression tests for the CACHE_GRAPH review round on e616326.

Each test pins one reviewer finding so its mutant is SELECTED by the suite:

- MAJOR-1  the direct ``graph_engine.o_access`` fallback ran IN ADDITION to
  the cache's own producer on a cold cached run (off=1/cold=2/warm=0 direct
  calls; cold ~1.75x slower than off).  Contract now: ``arrays_once`` hands
  back valid arrays whether served or produced-fresh (StageProduced), and
  the caller recomputes ONLY when the facade could not participate at all
  (disabled / unrepresentable key).  Pin: off=1, cold=1, warm=0.
- MINOR-2  compute_env carries host CPU identity (llvmlite codegen target);
  geometry_env does not.
- MINOR-3  AddObstacles snapshots the obstacle file exactly once per run
  (the hashed bytes and the decoded bytes cannot diverge).
- MINOR-4  a policy-less Topology (``cache_options is None``) runs
  BuildAccessPoints through the shared disabled facade instead of raising
  AttributeError.
- NOTE-6   ``arrays_once`` degrades to a local produce on a per-key wait
  timeout instead of crashing the caller (a second writer can only publish
  an identical generation).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.cache.invalidation.conftest import (
    make_settings, run_access, hash_out, stats_snapshot, stats_delta)

pytestmark = pytest.mark.cache_graph


# ------------------------------------------------------- MAJOR-1 (AWE engine)

@pytest.mark.cache_graph
def test_awe_direct_o_access_off1_cold1_warm0(tmp_path, workload, cache_dir,
                                              monkeypatch):
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE

    calls = {"direct": 0}
    orig = AWE.AccessibilityWElevation.CompactNodeView.o_access

    def counting(self, o_idx, settings, d_weights):
        calls["direct"] += 1
        return orig(self, o_idx, settings, d_weights)

    monkeypatch.setattr(AWE.AccessibilityWElevation.CompactNodeView,
                        "o_access", counting)

    s_off = make_settings(workload, tmp_path / "out_off")
    run_access(s_off)
    assert calls["direct"] == 1
    calls["direct"] = 0

    s_cold = make_settings(workload, tmp_path / "out_cold",
                           cache_mode="disk", cache_dir=cache_dir)
    run_access(s_cold)
    # THE regression: e616326 computed the cold run twice (owner's produce
    # AND the unguarded fallback) -> cold was 2 here and ~1.75x SLOWER than
    # off.
    assert calls["direct"] == 1
    calls["direct"] = 0

    s_warm = make_settings(workload, tmp_path / "out_warm",
                           cache_mode="disk", cache_dir=cache_dir)
    run_access(s_warm)
    assert calls["direct"] == 0

    assert hash_out(s_off) == hash_out(s_cold) == hash_out(s_warm)


# ------------------------------------------------ MAJOR-1 (plain engine)

@pytest.mark.cache_graph
def test_plain_engine_direct_o_access_off1_cold1_warm0(tmp_path, workload,
                                                       cache_dir, monkeypatch):
    """The plain engine is API-constructed (RunAccessibility always routes
    to AWE), so the same off=1/cold=1/warm=0 contract is pinned on direct
    Centrality calls; the cache rides on ``topology.cache_options``."""
    from urban_network_analysis.Engines import Accessibility as ACC

    calls = {"direct": 0}
    orig = ACC.Accessibility.CompactNodeView.o_access

    def counting(self, o_idx, settings, d_weights):
        calls["direct"] += 1
        return orig(self, o_idx, settings, d_weights)

    monkeypatch.setattr(ACC.Accessibility.CompactNodeView,
                        "o_access", counting)

    # Build the topology under DISK mode so topology/snap `_stage_key`s
    # exist (an off-built topology has no stage keys -> the plain engine's
    # result key is None and the cache cannot engage at all).  The runner
    # also consumes/normalizes the settings in place — the plain engine
    # feeds settings.knn_weights straight into numba, so that normalization
    # is part of its input contract.
    s_run = make_settings(workload, tmp_path / "out_run", elevation=False,
                          cache_mode="disk", cache_dir=cache_dir)
    una = run_access(s_run)
    eng = ACC.Accessibility(una.topology)

    # off: cache_options None -> shared disabled facade -> exactly one
    # direct compute, and the store sees nothing.
    una.topology.cache_options = None
    before = stats_snapshot(s_run)
    eng.Centrality(s_run)
    assert calls["direct"] == 1
    assert stats_delta(s_run, before) == (0, 0)
    calls["direct"] = 0

    # cold: with the store engaged the owner's produce IS the compute —
    # the regression (e616326) computed a SECOND time through the
    # unguarded fallback (cold was 2 here, ~1.75x slower than off).
    una.topology.cache_options = s_run.execution.cache
    before = stats_snapshot(s_run)
    eng.Centrality(s_run)
    assert calls["direct"] == 1
    assert stats_delta(s_run, before) == (0, 1)
    calls["direct"] = 0

    # warm: served from the store — zero direct computes.
    before = stats_snapshot(s_run)
    eng.Centrality(s_run)
    assert calls["direct"] == 0
    assert stats_delta(s_run, before) == (1, 0)


# ---------------------------------------------------------------- MINOR-2

@pytest.mark.cache_graph
def test_compute_env_carries_cpu_identity():
    from urban_network_analysis.cache import identity as ckid

    env = ckid.compute_env("una_legacy", "test")
    genv = ckid.geometry_env("una_legacy", "test")
    # compute (JIT/codegen) env pins the host CPU; the geometry env must NOT
    # gain it (geometry bytes do not depend on the codegen target).
    assert isinstance(env["cpu"], dict) and env["cpu"]
    assert all(isinstance(v, str) and v for v in env["cpu"].values())
    assert "cpu" not in genv

    # ...and the identity is key-material: a different host must key
    # differently (monkeypatched the same way the unit tests do).
    orig = ckid._host_cpu_identity
    try:
        ckid._host_cpu_identity = lambda: {"name": "OTHERCORE",
                                           "model": "OTHER"}
        env2 = ckid.compute_env("una_legacy", "test")
    finally:
        ckid._host_cpu_identity = orig
    assert env2["cpu"] == {"name": "OTHERCORE", "model": "OTHER"}
    assert env2 != env


# ---------------------------------------------------------------- MINOR-3

@pytest.mark.cache_graph
def test_obstacle_file_is_snapshotted_exactly_once_per_run(
        tmp_path, workload, cache_dir, monkeypatch):
    """The hashed snapshot and the decoded gdf must come from the SAME read
    (TOCTOU): before the fix AddObstacles read the file raw for
    penalties/directions AND BuildAccessPoints read it again for snapping."""
    from urban_network_analysis import Topology as T

    reads = {"obstacles": 0}
    orig = T._read_snapshot

    def counting(source_file):
        if "obstacles" in str(source_file):
            reads["obstacles"] += 1
        return orig(source_file)

    monkeypatch.setattr(T, "_read_snapshot", counting)

    s_off = make_settings(workload, tmp_path / "out_off")
    una_off = run_access(s_off)
    assert reads["obstacles"] == 1
    pen_off = una_off.topology.obstacles.penalty
    dir_off = np.asarray(una_off.topology.obstacles.direction).copy()

    reads["obstacles"] = 0
    s_cold = make_settings(workload, tmp_path / "out_cold",
                           cache_mode="disk", cache_dir=cache_dir)
    una_cold = run_access(s_cold)
    assert reads["obstacles"] == 1
    # penalties/directions survive the unified snapshot (same bits reach the
    # engine key and the arrays).
    assert una_cold.topology.obstacles.penalty.tobytes() == pen_off.tobytes()
    assert (np.asarray(una_cold.topology.obstacles.direction).tolist()
            == dir_off.tolist())
    assert hash_out(s_off) == hash_out(s_cold)


# ---------------------------------------------------------------- MINOR-4

@pytest.mark.cache_graph
def test_stage_cache_for_none_is_shared_disabled_facade():
    from urban_network_analysis.cache.stages import (
        stage_cache_for, StageProduced)

    c1 = stage_cache_for(None)
    c2 = stage_cache_for(None)
    assert c1 is c2 and c1.enabled is False
    assert c1.stats()["mode"] == "off"
    # and it still runs producers (the disabled facade degenerates to
    # compute-through; nothing is stored).
    arrays, meta, hit = c1.arrays_once(
        "k", lambda: ({"a": np.arange(3)}, None))
    assert isinstance(hit, StageProduced)
    assert (arrays["a"] == np.arange(3)).all()


@pytest.mark.cache_graph
def test_policyless_topology_build_access_points_still_works(
        workload, tmp_path, monkeypatch):
    """A topology whose cache policy was never applied (API-constructed;
    ``cache_options is None``) runs the ORIGINAL uncached snap path —
    e616326 raised AttributeError here (review MINOR-4)."""
    s = make_settings(workload, tmp_path / "out")
    una = run_access(s)
    t = una.topology
    t.cache_options = None  # simulate the policy-less public construction

    ap = t.BuildAccessPoints(str(workload / "origins.geojson"),
                             cost_attribute="weight", uid_attribute="uid",
                             label="Origins")
    assert ap is not None
    assert ap.node_weight.shape == una.topology.origins.node_weight.shape
    assert (ap.node_weight == una.topology.origins.node_weight).all()

    # ...and the snapshot passthrough kwarg (MINOR-3 plumbing) is accepted
    # on the same policy-less object.
    from urban_network_analysis.Topology import _read_snapshot
    snap = _read_snapshot(str(workload / "origins.geojson"))
    ap2 = t.BuildAccessPoints(str(workload / "origins.geojson"),
                              cost_attribute="weight", uid_attribute="uid",
                              label="Origins", snapshot=snap)
    assert (ap2.node_weight == ap.node_weight).all()


# ----------------------------------------------------------------- NOTE-6

@pytest.mark.cache_graph
def test_arrays_once_falls_back_to_produce_on_wait_timeout():
    """A per-key wait that outlives ``wait_timeout_s`` (store contract: a
    holder slower than the timeout) degrades to a local compute instead of
    crashing the caller.  A second writer can only publish an IDENTICAL
    generation — the key closure pins every numerical input — so the
    fallback can never serve different bits."""
    from urban_network_analysis.cache.stages import StageCache, StageProduced

    class _TimeoutStore:
        def get(self, key):
            return None

        def singleflight(self, key):
            raise TimeoutError("simulated stale holder")

    cache = StageCache(_TimeoutStore())
    arrays, meta, hit = cache.arrays_once(
        "k", lambda: ({"a": np.arange(4)}, {"n": 4}))
    assert isinstance(hit, StageProduced) and hit.source == "produced"
    assert (arrays["a"] == np.arange(4)).all()
    assert meta == {"n": 4}
