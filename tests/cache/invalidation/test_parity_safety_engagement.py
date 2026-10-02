"""Warm/cold/off bit parity, restore safety, and producer engagement.

- disabled (mode='off') is fully inert and byte-identical to the historical
  path;
- warm artifacts are bitwise-equal to cold artifacts and to the off path,
  including exported geojson/feather/csv bytes and the restored CRS;
- restored payloads are fresh copies (mutating them cannot poison the
  store), checksummed bytes are verified on read (a corrupted blob cannot
  come back as a hit), file mtimes and paths are not part of identity;
- a warm run demonstrably SKIPS the producers (decode, BuildTopology,
  o_access compute) instead of just reading faster.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.cache.invalidation.conftest import (
    make_settings, run_access, hash_out, stats_snapshot, stats_delta)

pytestmark = pytest.mark.cache_graph


# ------------------------------------------------------------- inert off

@pytest.mark.cache_graph
def test_off_mode_creates_no_directory(tmp_path, workload):
    s = make_settings(workload, tmp_path / "out")
    run_access(s)
    assert not Path(".una-cache").exists()
    assert not (tmp_path / "una-cache").exists()


# --------------------------------------------------------------- parity

@pytest.mark.cache_graph
def test_warm_cold_off_artifacts_bitwise_equal(tmp_path, workload, cache_dir):
    s_off = make_settings(workload, tmp_path / "out_off")
    run_access(s_off)
    s_cold = make_settings(workload, tmp_path / "out_cold", cache_mode="disk",
                           cache_dir=cache_dir)
    run_access(s_cold)
    s_warm = make_settings(workload, tmp_path / "out_warm", cache_mode="disk",
                           cache_dir=cache_dir)
    run_access(s_warm)
    h_off = hash_out(s_off)
    assert h_off == hash_out(s_cold)
    assert h_off == hash_out(s_warm)
    assert len(h_off) >= 3  # geojson + feather + csv all present


@pytest.mark.cache_graph
def test_restored_crs_matches_uncached_crs(tmp_path, workload, cache_dir):
    s_off = make_settings(workload, tmp_path / "out_off")
    una_off = run_access(s_off)
    s_warm = make_settings(workload, tmp_path / "out_warm", cache_mode="disk",
                           cache_dir=cache_dir)
    run_access(s_warm)
    s_warm2 = make_settings(workload, tmp_path / "out_warm2",
                            cache_mode="disk", cache_dir=cache_dir)
    una_warm = run_access(s_warm2)
    assert una_off.topology.crs is not None
    assert (una_warm.topology.crs.to_json()
            == una_off.topology.crs.to_json())
    assert (una_warm.topology.origins.geometry.crs.to_json()
            == una_off.topology.origins.geometry.crs.to_json())


# --------------------------------------------------------------- safety

@pytest.mark.cache_graph
def test_mutating_restored_payloads_cannot_poison_later_runs(
        tmp_path, workload, cache_dir):
    s1 = make_settings(workload, tmp_path / "out1", cache_mode="disk",
                       cache_dir=cache_dir)
    una1 = run_access(s1)
    h1 = hash_out(s1)
    # Clobber every restored numerical surface on the live objects.
    una1.accessibility.reach[:] = -999.0
    una1.topology.network.weights[:] = 0.0
    una1.topology.origins.node_weight[:] = -1.0
    s2 = make_settings(workload, tmp_path / "out2", cache_mode="disk",
                       cache_dir=cache_dir)
    run_access(s2)
    assert hash_out(s2) == h1


@pytest.mark.cache_graph
def test_corrupted_payload_cannot_serve_a_hit(tmp_path, workload, cache_dir):
    s1 = make_settings(workload, tmp_path / "out1", cache_mode="disk",
                       cache_dir=cache_dir)
    run_access(s1)
    h1 = hash_out(s1)
    # Corrupt the largest stored blob (the network geometry payload).
    blobs = sorted(Path(cache_dir).rglob("*"), key=lambda p: p.stat().st_size
                   if p.is_file() else 0)
    target = [p for p in blobs if p.is_file()][-1]
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    target.write_bytes(bytes(raw))
    s2 = make_settings(workload, tmp_path / "out2", cache_mode="disk",
                       cache_dir=cache_dir)
    una2 = run_access(s2)  # must not raise and must not serve wrong bits
    assert una2.topology.network.weights.shape[0] > 0
    assert hash_out(s2) == h1


@pytest.mark.cache_graph
def test_same_bytes_new_mtime_still_hits(tmp_path, workload, cache_dir):
    s1 = make_settings(workload, tmp_path / "out1", cache_mode="disk",
                       cache_dir=cache_dir)
    run_access(s1)
    net = Path(workload) / "network.geojson"
    stale = net.stat().st_mtime + 5000.0
    import os
    os.utime(net, (stale, stale))
    s2 = make_settings(workload, tmp_path / "out2", cache_mode="disk",
                       cache_dir=cache_dir)
    before = stats_snapshot(s2)
    run_access(s2)
    assert stats_delta(s2, before) == (5, 0)


@pytest.mark.cache_graph
def test_same_bytes_different_path_still_hits(tmp_path, workload, cache_dir):
    s1 = make_settings(workload, tmp_path / "out1", cache_mode="disk",
                       cache_dir=cache_dir)
    run_access(s1)
    data2 = tmp_path / "data_elsewhere"
    shutil.copytree(workload, data2)
    s2 = make_settings(data2, tmp_path / "out2", cache_mode="disk",
                       cache_dir=cache_dir)
    before = stats_snapshot(s2)
    run_access(s2)
    assert stats_delta(s2, before) == (5, 0)


@pytest.mark.cache_graph
def test_no_obstacle_row_still_caches_and_matches(tmp_path, workload,
                                                  cache_dir):
    s_off = make_settings(workload, tmp_path / "out_off",
                          obstacle_points_file="")
    run_access(s_off)
    s_cold = make_settings(workload, tmp_path / "out_cold", cache_mode="disk",
                           cache_dir=cache_dir, obstacle_points_file="")
    run_access(s_cold)
    s_warm = make_settings(workload, tmp_path / "out_warm",
                           cache_mode="disk", cache_dir=cache_dir,
                           obstacle_points_file="")
    before = stats_snapshot(s_warm)
    run_access(s_warm)
    assert stats_delta(s_warm, before) == (4, 0)  # 4 stages: no obstacle snap
    assert hash_out(s_warm) == hash_out(s_off)


# ---------------------------------------------------- producer engagement

@pytest.mark.cache_graph
def test_warm_run_skips_the_producers(tmp_path, workload, cache_dir,
                                      monkeypatch):
    from urban_network_analysis import Topology as T
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE

    calls = {"snapshot": 0, "decode": 0, "topology": 0, "o_access": 0}

    orig_read = T._read_snapshot

    def counting_read(source_file):
        calls["snapshot"] += 1
        return orig_read(source_file)

    orig_decode = T._decode_snapshot_gdf

    def counting_decode(*a, **k):
        calls["decode"] += 1
        return orig_decode(*a, **k)

    orig_bt = T.Network.BuildTopology  # BuildTopology is a Network method

    def counting_bt(net_self, *a, **k):
        calls["topology"] += 1
        return orig_bt(net_self, *a, **k)

    orig_produce = AWE.AccessibilityWElevation._compute_o_access

    def counting_produce(self, settings):
        calls["o_access"] += 1
        return orig_produce(self, settings)

    monkeypatch.setattr(T, "_read_snapshot", counting_read)
    monkeypatch.setattr(T, "_decode_snapshot_gdf", counting_decode)
    monkeypatch.setattr(T.Network, "BuildTopology", counting_bt)
    monkeypatch.setattr(AWE.AccessibilityWElevation, "_compute_o_access",
                        counting_produce)

    s_cold = make_settings(workload, tmp_path / "out_cold",
                           cache_mode="disk", cache_dir=cache_dir)
    run_access(s_cold)
    cold = dict(calls)

    calls.update({"snapshot": 0, "decode": 0, "topology": 0, "o_access": 0})
    s_warm = make_settings(workload, tmp_path / "out_warm",
                           cache_mode="disk", cache_dir=cache_dir)
    run_access(s_warm)
    warm = dict(calls)

    # Cold: 4 layer snapshots hashed (network + 3 point layers), 4 decodes,
    # one topology build, one o_access compute.  Warm: the producers never
    # run.  The snapshot READ still happens on warm — a content-addressed
    # key cannot be computed without reading the bytes it digests; what is
    # avoided is the decode + topology build + snapping + search work.
    assert cold["snapshot"] == 4 and warm["snapshot"] == 4
    assert cold["decode"] == 4 and warm["decode"] == 0
    assert cold["topology"] == 1 and warm["topology"] == 0
    assert cold["o_access"] == 1 and warm["o_access"] == 0


# ------------------------------------------------ plain Accessibility engine

@pytest.mark.cache_graph
def test_plain_accessibility_engine_result_cache(tmp_path, workload, cache_dir):
    """The plain (symmetric-weight) engine is constructed directly by API
    users; its result stage chains the same cached topology/snaps and must
    reproduce the AWE no-elevation bits bitwise."""
    from urban_network_analysis.Engines.Accessibility import Accessibility

    s_run = make_settings(workload, tmp_path / "out_run", cache_mode="disk",
                          cache_dir=cache_dir, elevation=False)
    una = run_access(s_run)

    # Fully uncached reference: same layers, plain engine, no stage cache.
    s_nc = make_settings(workload, tmp_path / "out_nc", elevation=False)
    una_nc = run_access(s_nc)
    eng_nc = Accessibility(una_nc.topology)
    eng_nc.Centrality(s_nc)

    eng_cold = Accessibility(una.topology)
    eng_warm = Accessibility(una.topology)

    before = stats_snapshot(s_run)
    eng_cold.Centrality(s_run)
    mid = stats_snapshot(s_run)
    eng_warm.Centrality(s_run)
    hits, misses = stats_delta(s_run, before)

    assert (mid["hits"] - before["hits"],
            mid["misses"] - before["misses"]) == (0, 1)  # cold miss
    assert (hits, misses) == (1, 1)  # warm hit the committed generation
    for attr in ("reach", "gravity_exponential", "gravity_logistic",
                 "knn_access"):
        a = getattr(eng_cold, attr)
        b = getattr(eng_warm, attr)
        assert a.tobytes() == b.tobytes()
        # uncached-anchor: cached topology/snaps/result reproduce the plain
        # engine's own bits on an uncached topology (obstacles are ignored
        # by THIS engine by design, so the anchor is engine-to-engine).
        assert a.tobytes() == getattr(eng_nc, attr).tobytes()
