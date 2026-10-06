"""P02 increment C: run-wide prepared serving + single-flight.

Fixtures (design 02 increment C acceptance; counter law: cold build /
verified hit / contention wait / corruption rebuild / refusal are
DIFFERENT counters):
- PP05  admission + refusal boundaries at the run tier: changed bytes
        never stale-hit; row-level validation identical attached vs
        detached; non-representable payloads refused at tier 2 but kept
        at tier 1; no pickle in the store; no hidden source snapshot;
        stray sidecar files are irrelevant
- PP06  corrupted store generation quarantines and rebuilds (counted);
        a dead owner's lock is stolen on pid liveness (poll-bounded,
        never a timeout)
- PP07  user mutation of returned objects never reaches the leases or
        the store (fresh mutable outputs at every tier)
- PP08  mixed rows: distinct producer-read params are distinct
        identities; row-level errors identical prepared on/off with the
        isfile gate preceding admission
- PP09  budget law: publish admission refuses (cap floor, never partial
        admission) while the run store and serving continue; LRU
        eviction among UNLEASED entries only; a promotion that cannot
        fit serves without retaining; attach/detach cycles reuse one
        process entry and a detached publish touches no store
plus the deterministic same-process contention signature and the
cross-process RunBatch ``singleflight_not_lost`` claim (6 rows over one
identity set -> 3 builds, not 18).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

import urban_network_analysis
import urban_network_analysis.batch.prepared as prepared
from urban_network_analysis.Topology import Topology

SRC = os.path.dirname(  # .../src
    os.path.dirname(urban_network_analysis.__file__))
REPO = os.path.dirname(SRC)
SMOKE = os.path.join(REPO, "tests", "execution", "fixtures")


@pytest.fixture(autouse=True)
def _isolated_prepared():
    """The prepared table/counters are process-global: isolate every
    test.  The attached run store is managed by the run_store fixture."""
    prepared.reset()
    yield
    prepared.reset()


@pytest.fixture
def run_store(tmp_path, monkeypatch):
    """Attach one run store for the test; fresh root per test, detached
    and reset afterwards.  The default table budget applies unless a
    test overrides it."""
    monkeypatch.delenv("UNA_PREPARED", raising=False)
    monkeypatch.delenv("UNA_PREPARED_BUDGET_BYTES", raising=False)
    root = str(tmp_path / ".una-batch-prepared")
    prepared.attach_run_store(root)
    yield root
    prepared.detach_run_store()
    prepared.reset()


def _arr_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


NETWORK_ARRAYS = (
    "lengths", "weights", "node_points", "start_nodes", "end_nodes",
    "degrees", "z", "start_degrees", "end_degrees", "min_degrees",
    "max_degrees",
)


def _network_surface(topo):
    net = topo.network
    # Snapshots, not references: a later in-place mutation of the row's
    # objects must not move an already-captured baseline.
    surf = {k: np.asarray(getattr(net, k)).copy()
            for k in NETWORK_ARRAYS}
    surf["geometry_wkb"] = [g.wkb for g in net.geometry]
    surf["crs_projjson"] = (topo.crs.to_json()
                            if topo.crs is not None else None)
    surf["stage_key"] = getattr(net, "_stage_key", None)
    return surf


def _ap_surface(pt):
    return {
        "geometry_wkb": [g.wkb for g in pt.geometry],
        "node_weight": np.asarray(pt.node_weight).copy(),
        "uid_values": (list(pt.uid) if pt.uid is not None else None),
        "nearest_edge_id": (None if getattr(pt, "nearest_edge_id", None)
                            is None
                            else np.asarray(pt.nearest_edge_id).copy()),
        "stage_key": getattr(pt, "_stage_key", None),
    }


def _assert_surface_eq(a, b):
    for k, v in a.items():
        if k in ("geometry_wkb", "crs_projjson", "stage_key", "uid_values"):
            assert v == b[k], k
        else:
            assert _arr_eq(v, b[k]), k


def _run_row(settings):
    t = Topology()
    t.AddNetwork(settings)
    t.AddOrigins(settings)
    t.AddDestinations(settings)
    return t


def _run_counters():
    return prepared.stats()["run"]


def _delta_run(before):
    now = _run_counters()
    return {k: now[k] - before[k] for k in now}


def _store_files(root):
    out = {}
    for base, _dirs, names in os.walk(root):
        for n in names:
            p = os.path.join(base, n)
            with open(p, "rb") as f:
                out[p] = f.read()
    return out


def _gen_dir(root, key):
    return os.path.join(root, "store", key[:2], key)


# ── PP05: admission and refusal boundaries at the run tier ──────────────


def test_pp05_bytes_change_never_stale_hit_at_tier2(run_store,
                                                    smoke_fixture, tmp_path,
                                                    make_prep_settings):
    """Rewrite the network bytes under a preserved mtime while a run
    store is attached: the identity must move, tier 2 must never serve
    the old generation, and the new generation must be published."""
    d = tmp_path / "data"
    shutil.copytree(smoke_fixture, d)
    net_file = d / "network.geojson"
    st = net_file.stat()

    s1 = make_prep_settings(str(d))
    t1 = _run_row(s1)
    r1 = _run_counters()
    assert r1["cold_builds"] == 3 and r1["tier2_publishes"] == 3
    assert r1["tier2_hits"] == 0

    gdf = gpd.read_file(net_file, engine="pyogrio", use_arrow=True)
    gdf.geometry = gdf.geometry.translate(xoff=0.001)
    gdf.to_file(net_file, driver="GeoJSON")
    os.utime(net_file, (st.st_atime, st.st_mtime))
    assert net_file.stat().st_mtime == st.st_mtime

    s2 = make_prep_settings(str(d))
    t2 = _run_row(s2)
    r2 = _run_counters()
    assert r2["tier2_hits"] == 0, "changed bytes must never serve tier 2"
    assert r2["cold_builds"] == 6 and r2["tier2_publishes"] == 6, \
        "new identities build and publish; nothing stale-served"
    assert ([g.wkb for g in t1.network.geometry]
            != [g.wkb for g in t2.network.geometry])
    assert t1.network._stage_key != t2.network._stage_key
    # Both generations exist in the store; only the new one is reachable.
    new_bundle = next(b for b in prepared._table.values()
                      if b.payload.get("stage_key") == t2.network._stage_key)
    assert new_bundle.file_sha256 == hashlib.sha256(
        open(net_file, "rb").read()).hexdigest()
    assert os.path.isdir(_gen_dir(run_store, new_bundle.key))


def test_pp05_polygon_error_identical_attached_vs_detached(
        run_store, smoke_fixture, tmp_path, make_prep_settings):
    """Row-level validation is a produced-path law: with a run store
    attached, a Polygon network raises the same error the detached path
    raises, and nothing is published for the failed row."""
    d = tmp_path / "poly"
    d.mkdir()
    poly = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:32616")
    poly.to_file(d / "network.geojson", driver="GeoJSON")
    shutil.copy(os.path.join(str(smoke_fixture), "origins.geojson"),
                d / "origins.geojson")
    shutil.copy(os.path.join(str(smoke_fixture), "destinations.geojson"),
                d / "destinations.geojson")

    s = make_prep_settings(str(d))
    before = _run_counters()
    with pytest.raises(ValueError) as attached:
        Topology().AddNetwork(s)
    assert _delta_run(before)["tier2_publishes"] == 0, \
        "a produced failure never publishes"

    prepared.detach_run_store()
    prepared.reset()
    with pytest.raises(ValueError) as detached:
        Topology().AddNetwork(s)
    assert str(attached.value) == str(detached.value)


def test_pp05_object_uid_refused_at_tier2_kept_at_tier1(
        run_store, smoke_fixture, tmp_path, make_prep_settings):
    """An object-dtype uid column cannot cross the store without pickle:
    the layer keeps its worker-local bundle (tier-1 serving) while the
    run tier records a refusal — and the store contains no pickle bytes
    and no hidden snapshot of the source files."""
    d = tmp_path / "uid"
    shutil.copytree(smoke_fixture, d)
    ogdf = gpd.read_file(d / "origins.geojson",
                         engine="pyogrio", use_arrow=True)
    ogdf["UID_STR"] = ["a", "b"][: len(ogdf)]
    ogdf.to_file(d / "origins.geojson", driver="GeoJSON")
    inputs = {
        n: hashlib.sha256(open(d / n, "rb").read()).hexdigest()
        for n in ("network.geojson", "origins.geojson",
                  "destinations.geojson")
    }

    s = make_prep_settings(str(d), origin_uid_column="UID_STR")
    t1 = _run_row(s)
    r = _run_counters()
    assert r["tier2_refusals"] == 1, "origins uid refused at tier 2"
    assert r["tier2_publishes"] == 2, "topology + destinations published"
    assert prepared.stats()["snap"]["publishes"] == 2, "tier 1 kept both"
    assert list(t1.origins.uid) == ["a", "b"]

    files = _store_files(run_store)
    for path, blob in files.items():
        assert b"\x80" not in blob, f"pickle protocol bytes in {path}"
        assert hashlib.sha256(blob).hexdigest() not in inputs, \
            f"source snapshot hidden in store at {path}"

    # The refused layer still serves — from tier 1 in this process.
    t2 = _run_row(s)
    assert prepared.stats()["snap"]["hits"] == 2
    assert list(t2.origins.uid) == ["a", "b"]
    _assert_surface_eq(_ap_surface(t1.origins), _ap_surface(t2.origins))


def test_pp05_digest_belt_and_sidecar_irrelevance(run_store, smoke_fixture,
                                                  make_prep_settings):
    """The recorded ``file_sha256`` in store metadata is a
    refuse-different-bytes belt: tampered metadata refuses serving and
    the rebuild's republish supersedes the bogus generation (counted),
    and a stray file in the run-store root never affects serving."""
    s1 = make_prep_settings(smoke_fixture)
    t1 = _run_row(s1)
    surf1 = _network_surface(t1)

    bundle = next(b for b in prepared._table.values()
                  if "n_nodes" in b.payload)
    manifest = os.path.join(_gen_dir(run_store, bundle.key),
                            "manifest.json")
    with open(manifest, "r", encoding="utf-8") as f:
        doc = json.load(f)
    doc["metadata"]["file_sha256"] = "0" * 64
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(doc, f)

    prepared.reset()  # force the store read
    before = _run_counters()
    t2 = _run_row(s1)
    dr = _delta_run(before)
    # Two refusals compose: the pre-flight lookup refuses, the loser
    # re-serve refuses again (nothing was unreadable, so the bogus
    # generation stood until the rebuild's republish superseded it).
    assert dr["corruption_rebuilds"] == 2
    assert dr["tier2_publishes"] == 1, "rebuilt generation republished"
    _assert_surface_eq(surf1, _network_surface(t2))

    # Sidecar irrelevance.
    with open(os.path.join(run_store, "junk.bin"), "wb") as f:
        f.write(b"not-a-generation")
    t3 = _run_row(s1)
    assert prepared.stats()["topology"]["hits"] == 1
    assert prepared.stats()["snap"]["hits"] == 2
    _assert_surface_eq(surf1, _network_surface(t3))


# ── PP06: corruption and owner death ─────────────────────────────────────


def test_pp06_store_corruption_quarantines_and_rebuilds(
        run_store, smoke_fixture, make_prep_settings):
    """A damaged payload file fails the store's verified read: require
    raises, prepared quarantines and rebuilds (exactly one counted
    corruption rebuild), and the served surface stays bitwise equal."""
    s = make_prep_settings(smoke_fixture)
    t1 = _run_row(s)
    surf1 = _network_surface(t1)

    bundle = next(b for b in prepared._table.values()
                  if "n_nodes" in b.payload)
    payload_dir = os.path.join(_gen_dir(run_store, bundle.key), "payload")
    name = sorted(os.listdir(payload_dir))[0]
    path = os.path.join(payload_dir, name)
    with open(path, "rb") as f:
        blob = f.read()
    with open(path, "wb") as f:
        f.write(blob[:-8] if len(blob) > 8 else blob + b"x")

    prepared.reset()  # force the store read
    before = _run_counters()
    t2 = _run_row(s)
    dr = _delta_run(before)
    assert dr["corruption_rebuilds"] == 1
    assert dr["tier2_publishes"] == 1
    _assert_surface_eq(surf1, _network_surface(t2))

    # The rebuilt generation serves again from a clean store.
    prepared.reset()
    before = _run_counters()
    t3 = _run_row(s)
    assert _delta_run(before)["tier2_hits"] == 3
    _assert_surface_eq(surf1, _network_surface(t3))


def test_pp06_owner_death_stale_lock_stolen(run_store, smoke_fixture,
                                            tmp_path, make_prep_settings):
    """An owner that dies holding the single-flight lock leaves a stale
    lock: the next waiter steals it on pid liveness (poll-bounded — the
    600 s cohort budget is never reached) and becomes the owner."""
    script = tmp_path / "holder.py"
    ready = tmp_path / "ready"
    script.write_text(
        "import os, sys\n"
        "from urban_network_analysis.Execution import CacheOptions\n"
        "from urban_network_analysis.cache.store import ContentStore\n"
        "root, key, ready = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "store = ContentStore(CacheOptions(mode='disk', directory=root))\n"
        "with store.singleflight(key, wait_timeout_s=30.0):\n"
        "    open(ready, 'w').close()\n"
        "    os._exit(1)\n",  # die INSIDE the with: no release runs
        encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=SRC)

    s = make_prep_settings(smoke_fixture)
    req = prepared.topology_request(
        os.path.join(str(smoke_fixture), "network.geojson"), s)
    assert req is not None

    proc = subprocess.Popen(
        [sys.executable, str(script), run_store, req.key,
         str(ready)], env=env)
    deadline = time.monotonic() + 60.0
    while not ready.exists() and proc.poll() is None:
        assert time.monotonic() < deadline, "holder never signalled ready"
        time.sleep(0.01)
    assert ready.exists(), "holder died before signalling ready"
    assert proc.wait(timeout=30) == 1

    before = _run_counters()
    t0 = time.monotonic()
    with prepared.flight(req) as owned:
        assert owned is True, "stale lock must be stolen, not waited out"
    assert time.monotonic() - t0 < 30.0, "steal is poll-bounded"
    dr = _delta_run(before)
    assert dr["cold_builds"] == 1
    assert dr["contention_timeouts"] == 0, "no timeout was involved"
    assert dr["contention_waits"] == 0


# ── PP07: user mutation isolation across tiers ───────────────────────────


def test_pp07_user_mutation_isolated(run_store, smoke_fixture,
                                     make_prep_settings):
    """Mutating the returned objects in place never reaches the leases
    (row 2 served unchanged) nor the store payload (after a reset, a
    fresh attach re-serves the ORIGINAL generation from tier 2)."""
    s = make_prep_settings(smoke_fixture)
    t1 = _run_row(s)
    surf1 = (_network_surface(t1), _ap_surface(t1.origins))

    t1.network.weights[:] = 0.0
    line = t1.network.geometry.iloc[0]
    t1.network.geometry.iloc[0] = LineString(
        [(x + 5.0, y) for x, y in line.coords])
    t1.origins.node_weight[:] = 0.0

    t2 = _run_row(s)
    _assert_surface_eq(surf1[0], _network_surface(t2))
    _assert_surface_eq(surf1[1], _ap_surface(t2.origins))

    prepared.reset()
    prepared.detach_run_store()
    prepared.attach_run_store(run_store)  # same root: read what was stored
    t3 = _run_row(s)
    assert _run_counters()["tier2_hits"] == 3, "row 3 must come from the store"
    _assert_surface_eq(surf1[0], _network_surface(t3))
    _assert_surface_eq(surf1[1], _ap_surface(t3.origins))


# ── PP08: mixed rows and error parity ────────────────────────────────────


def test_pp08_mixed_identities_distinct_keys(run_store, smoke_fixture,
                                             tmp_path,
                                             make_prep_settings):
    """Mixed rows over one data dir: distinct producer-read params are
    distinct identities (each builds + publishes); a row repeating an
    earlier param set rides the SAME identity (tier-1 served) — output
    folder is not a producer-read param."""
    s_a = make_prep_settings(smoke_fixture,
                             output_folder=str(tmp_path / "out_a"))
    t_a = _run_row(s_a)
    r1 = _run_counters()
    assert r1["cold_builds"] == 3 and r1["tier2_publishes"] == 3

    s_b = make_prep_settings(smoke_fixture, network_precision=7,
                             output_folder=str(tmp_path / "out_b"))
    t_b = _run_row(s_b)
    r2 = _run_counters()
    assert r2["cold_builds"] == 6 and r2["tier2_publishes"] == 6

    s_c = make_prep_settings(smoke_fixture,
                             output_folder=str(tmp_path / "out_c"))
    t_c = _run_row(s_c)
    r3 = _run_counters()
    assert r3["cold_builds"] == 6, "row C must not rebuild"
    assert r3["tier2_hits"] == 0
    assert prepared.stats()["topology"]["hits"] == 1
    assert prepared.stats()["snap"]["hits"] == 2
    _assert_surface_eq(_network_surface(t_a), _network_surface(t_c))
    assert t_a.network._stage_key != t_b.network._stage_key


def test_pp08_missing_origins_error_parity_and_no_publish(
        run_store, smoke_fixture, tmp_path, make_prep_settings,
        monkeypatch):
    """A missing origins file fails identically with the run tier
    attached and with prepared off — the isfile gate precedes admission,
    and nothing is published for the failed row."""
    d = tmp_path / "partial"
    d.mkdir()
    shutil.copy(os.path.join(str(smoke_fixture), "network.geojson"),
                d / "network.geojson")
    s = make_prep_settings(str(d))

    # The network itself must be present so the failure lands on origins.
    Topology().AddNetwork(s)  # succeeds and publishes
    files_before = set(_store_files(run_store))

    with pytest.raises(FileNotFoundError) as attached:
        Topology().AddOrigins(s)
    assert set(_store_files(run_store)) == files_before, \
        "a failed row must not touch the store"

    monkeypatch.setenv("UNA_PREPARED", "0")
    with pytest.raises(FileNotFoundError) as off:
        Topology().AddOrigins(s)
    assert str(attached.value) == str(off.value)
    assert str(attached.value).startswith("Origins file not found")


# ── PP09: budget, leases, eviction, attach/detach ────────────────────────


def test_pp09_tiny_budget_refuses_admission_not_serving(
        run_store, smoke_fixture, make_prep_settings, monkeypatch):
    """A 1-byte table budget: publishes refuse (cap floor, table stays
    EMPTY — never partial admission) while the run store still receives
    generations and rows are served from tier 2 WITHOUT retaining
    (promote_skips) — serving is never refused."""
    monkeypatch.setenv("UNA_PREPARED_BUDGET_BYTES", "1")
    s = make_prep_settings(smoke_fixture)
    t1 = _run_row(s)
    r1 = _run_counters()
    assert r1["budget_refusals"] == 3
    assert r1["tier2_publishes"] == 3
    assert prepared.stats()["table"] == {"entries": 0, "bytes": 0}

    surf1 = (_network_surface(t1), _ap_surface(t1.origins))
    before = dict(_run_counters())
    t2 = _run_row(s)
    dr = {k: _run_counters()[k] - before[k] for k in before}
    assert dr["tier2_hits"] == 3
    assert dr["promote_skips"] == 3, "served without retaining"
    assert prepared.stats()["table"] == {"entries": 0, "bytes": 0}
    _assert_surface_eq(surf1[0], _network_surface(t2))
    _assert_surface_eq(surf1[1], _ap_surface(t2.origins))


def test_pp09_lease_never_evicted_then_lru(run_store, smoke_fixture,
                                           make_prep_settings, monkeypatch):
    """Size the budget to exactly row 1's table: while every entry is
    leased, a new identity's publish cannot evict and refuses; after the
    leases release, publishing another identity evicts LRU (the row-1
    topology goes first) and the new identity is admitted."""
    s_a = make_prep_settings(smoke_fixture)
    _run_row(s_a)
    budget = prepared.stats()["table"]["bytes"]
    assert prepared.stats()["table"]["entries"] == 3
    monkeypatch.setenv("UNA_PREPARED_BUDGET_BYTES", str(budget))

    keys_a = set(prepared._table)
    with prepared._lock:
        for b in prepared._table.values():
            b.in_use += 1  # simulate live leases across the next row

    s_b = make_prep_settings(smoke_fixture, network_precision=7)
    before = _run_counters()
    _run_row(s_b)
    dr = _delta_run(before)
    assert dr["budget_refusals"] == 3, "nothing evictable: refuse"
    assert dr["evictions"] == 0, "a leased entry is never evicted"
    assert set(prepared._table) == keys_a, "leased entries intact"
    assert dr["tier2_publishes"] == 3, "the run store is unaffected"

    with prepared._lock:
        for b in prepared._table.values():
            b.in_use -= 1  # release

    s_c = make_prep_settings(smoke_fixture, network_precision=13)
    before = _run_counters()
    _run_row(s_c)
    dr = _delta_run(before)
    assert dr["evictions"] >= 1, "unleased entries evict LRU now"
    assert dr["budget_refusals"] == 0
    assert not (keys_a & set(prepared._table)), \
        "row-1 entries were evicted to make room"


def test_pp09_attach_detach_cycles_and_post_detach(run_store,
                                                   smoke_fixture,
                                                   make_prep_settings):
    """Attach/detach cycles reuse ONE process entry per root; a publish
    while detached touches no store; re-attaching picks up the stored
    generations for a tier-2 served row."""
    s = make_prep_settings(smoke_fixture)
    t1 = _run_row(s)
    surf1 = (_network_surface(t1), _ap_surface(t1.origins))
    assert _run_counters()["tier2_publishes"] == 3
    n_roots = len(prepared._PROCESS_RUN_STORES)
    pair = prepared._PROCESS_RUN_STORES[run_store]

    prepared.detach_run_store()
    prepared.reset()
    files_before = set(_store_files(run_store))

    t2 = _run_row(s)  # produced; publishes must not touch the store
    dr = _delta_run(_run_counters())
    assert dr["tier2_publishes"] == 0
    assert set(_store_files(run_store)) == files_before
    _assert_surface_eq(surf1[0], _network_surface(t2))

    prepared.attach_run_store(run_store)
    assert prepared._PROCESS_RUN_STORES[run_store] is pair, \
        "attach/detach cycles reuse one process entry per root"
    assert len(prepared._PROCESS_RUN_STORES) == n_roots
    prepared.reset()  # row 2's bundles would tier-1-serve row 3

    before = dict(_run_counters())
    t3 = _run_row(s)
    dr = {k: _run_counters()[k] - before[k] for k in before}
    assert dr["tier2_hits"] == 3
    _assert_surface_eq(surf1[0], _network_surface(t3))
    _assert_surface_eq(surf1[1], _ap_surface(t3.origins))


# ── Contention signature (deterministic, same process) ──────────────────


def test_contention_waits_signature(run_store, smoke_fixture,
                                    make_prep_settings):
    """Thread A owns the flight and commits a generation; thread B (a
    late waiter) exits as a counted contention LOSER (owned=False) —
    never a timeout, never a silent rebuild."""
    s = make_prep_settings(smoke_fixture)
    req = prepared.topology_request(
        os.path.join(str(smoke_fixture), "network.geojson"), s)
    assert req is not None
    rs = prepared._PROCESS_RUN_STORES[run_store]

    started = threading.Event()
    results = {}

    def owner():
        with prepared.flight(req) as owned:
            results["a"] = owned
            started.set()
            time.sleep(0.4)  # hold the flight while B arrives
            rs.stage.put_arrays(
                req.key, {"x": np.zeros(3, dtype="float64")},
                {"file_sha256": req.file_sha256, "prepared": "run:v1"})

    def waiter():
        started.wait(timeout=30)
        time.sleep(0.05)
        with prepared.flight(req) as owned:
            results["b"] = owned

    ta = threading.Thread(target=owner)
    tb = threading.Thread(target=waiter)
    before = _run_counters()
    ta.start()
    tb.start()
    ta.join(timeout=30)
    tb.join(timeout=30)
    dr = _delta_run(before)
    assert results == {"a": True, "b": False}
    assert dr["cold_builds"] == 1 and dr["contention_waits"] == 1
    assert dr["contention_timeouts"] == 0


# ── singleflight_not_lost: the cross-process RunBatch claim ─────────────


def test_runbatch_singleflight_not_lost(tmp_path, monkeypatch):
    """P00 measured singleflight_lost=31: six parallel rows over one
    identity set rebuilt the same prepared layers per worker.  With the
    run-wide tier engaged, 6 rows -> exactly 3 builds and 3 committed
    generations (one per identity), with a per-row counter slice and the
    store root reported in the batch report; the paired control
    (UNA_PREPARED=0) reports no prepared surface at all."""
    monkeypatch.delenv("UNA_PREPARED", raising=False)

    from urban_network_analysis import UNA

    def project(out_root):
        p = UNA(verbosity=0)
        p.settings.data_folder = str(tmp_path)
        shutil.copytree(SMOKE, tmp_path, dirs_exist_ok=True)
        p.settings.network_file = "network.geojson"
        p.settings.origins_file = "origins.geojson"
        p.settings.destinations_file = "destinations.geojson"
        p.settings.output_folder = str(out_root)
        p.settings.output_wStamp = False
        for i in range(6):
            p.settings.name = f"row{i}"
            p.settings.search_radius = 90 + 10 * i  # not producer-read
            p.SaveSettingsToProject()
        return p

    p = project(tmp_path / "out")
    p.RunBatch("accessibility", parallel=True, workers=2)

    report = p.batch_report
    assert all(r.status == "ran" for r in report.rows)
    prep = report.extra["prepared"]
    run = prep["run"]
    assert run["cold_builds"] == 3, \
        "6 rows over 3 identities -> 3 builds, not 18"
    assert run["tier2_publishes"] == 3
    assert run["contention_timeouts"] == 0
    assert run["corruption_rebuilds"] == 0
    assert sorted(prep["per_row"]) == [str(i) for i in range(6)]
    assert prep["store_root"].endswith(".una-batch-prepared")
    assert os.path.isdir(prep["store_root"])
    assert any("prepared reuse: 3 cold build(s)" in n
               for n in report.notes_runtime)

    # Paired control: kill switch restores the byte-shaped report.
    monkeypatch.setenv("UNA_PREPARED", "0")
    p0 = project(tmp_path / "out_off")
    p0.RunBatch("accessibility", parallel=True, workers=2)
    assert all(r.status == "ran" for r in p0.batch_report.rows)
    assert "prepared" not in p0.batch_report.extra
