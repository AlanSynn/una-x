"""P02 prepared-input route (una_10x, prepared_contract.json @ f139dd3).

Every test drives the PUBLIC surfaces: the batch route through RunBatch
(prepared armed by the runtime beside the cohort root), the topology
seam through AddNetwork/AddOrigins with an attached PrepSession, and
the Flight/arena units behind them.  Laws under test: bitwise parity
with the prepared route armed (E1), content-derived log replay (E2,
NF-1 CRS line), snap store keys carry the role component (E3/BF-1),
engagement gate (E4/NF-7), byte-identity invalidation (E5), owner
death (E6), refusal taxonomy (E7), NO_FIT serves without retention
(E8), detach-before-transport (E9), served-copy mutation isolation
(E10), num_threads bit-equality (E11).  E12 (snapshot re-read cost)
is a P02J measurement, not a gate.
"""
from __future__ import annotations

import hashlib
import os
import signal
import time

import numpy as np
import pytest

from conftest import tree_bytes

pytestmark = pytest.mark.tenx_post_p00

PREPARED_DIRNAME = ".una-batch-prepared"


def _key(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _prep_session(root, cap=1 << 26, state=None):
    from urban_network_analysis.batch.preparation import PrepSession
    return PrepSession(root, cap, state)


def _bare_settings(data_folder, network="network.geojson",
                   origins="origins.geojson",
                   destinations="destinations.geojson"):
    """A real Settings instance over the smoke grid (defaults otherwise)."""
    from urban_network_analysis import UNA
    p = UNA(verbosity=0)
    s = p.settings
    s.data_folder = str(data_folder)
    s.network_file = network
    s.origins_file = origins
    s.destinations_file = destinations
    return s


# ---------------------------------------------------------------------------
# E1 + E4: public batch route — bitwise parity, engagement gate, warm repeat
# ---------------------------------------------------------------------------

def test_prepared_bitwise_engagement_and_warm(make_batch, tmp_path):
    ser = make_batch(3, out_folder=tmp_path / "ser")
    ser.RunBatch("accessibility")

    out = tmp_path / "par"
    par = make_batch(3, out_folder=out)
    par.RunBatch("accessibility", parallel=True, workers=2)

    extra = getattr(par.batch_report, "extra", None) or {}
    prep = extra.get("preparation")
    assert prep is not None, "prepared route did not report"
    assert prep["rows"] == 3 and prep["stages"] == 9
    assert prep["refusals"] == {}, prep["refusals"]
    served = (prep["cold_build"] + prep["verified_hit"]
              + prep["contention_wait"] + prep["fallback_compute"])
    assert served == prep["stages"]
    # NF-7 engagement gate: 3 distinct identities (1 network + 1 origins
    # + 1 destinations); equality asserted only with zero fallbacks.
    assert prep["fallback_compute"] == 0
    assert prep["cold_build"] == 3, prep
    # the store root persists beside the row outputs (warm resumes)
    assert os.path.isdir(os.path.join(str(out), PREPARED_DIRNAME))

    cold = tree_bytes(out, skip_hidden=True)
    assert cold == tree_bytes(tmp_path / "ser", skip_hidden=True)

    # warm repeat: the persisted store serves every stage; nothing builds
    par2 = make_batch(3, out_folder=out)
    par2.RunBatch("accessibility", parallel=True, workers=2)
    prep2 = (getattr(par2.batch_report, "extra", None) or {}).get(
        "preparation")
    assert prep2 is not None and prep2["stages"] == 9
    assert prep2["cold_build"] == 0, prep2
    assert prep2["fallback_compute"] == 0
    assert prep2["verified_hit"] + prep2["contention_wait"] == 9
    assert tree_bytes(out, skip_hidden=True) == cold


def test_serial_and_single_job_never_arm_the_prepared_route(
        make_batch, tmp_path):
    """The serial route and a plain RunAccessibility never create the
    prepared store root nor report a preparation block (same negative
    law as the cohort route)."""
    out = tmp_path / "out"
    p = make_batch(3, out_folder=out)
    p.RunBatch("accessibility")
    assert (getattr(p.batch_report, "extra", None) or {}).get(
        "preparation") is None
    assert not os.path.exists(os.path.join(str(out), PREPARED_DIRNAME))

    q = make_batch(1, out_folder=tmp_path / "single")
    q.RunAccessibility()
    assert (getattr(q.batch_report, "extra", None) or {}).get(
        "preparation") is None
    assert not os.path.exists(os.path.join(str(tmp_path / "single"),
                                           PREPARED_DIRNAME))
    assert getattr(q.topology, "_prepared_session", None) is None


# ---------------------------------------------------------------------------
# E3 / BF-1: the snap STORE key carries the role component
# ---------------------------------------------------------------------------

def test_cross_role_snap_collision_distinct_entries(make_batch, tmp_path):
    """origins_file == destinations_file is LEGAL input; the two roles
    must occupy two distinct store entries (BF-1) and the run stays
    bitwise vs serial."""
    def same_roles(i, s):
        s.destinations_file = "origins.geojson"

    ser = make_batch(2, out_folder=tmp_path / "ser", extra=same_roles)
    ser.RunBatch("accessibility")

    out = tmp_path / "par"
    par = make_batch(2, out_folder=out, extra=same_roles)
    par.RunBatch("accessibility", parallel=True, workers=2)

    prep = (getattr(par.batch_report, "extra", None) or {}).get(
        "preparation")
    assert prep is not None and prep["rows"] == 2
    assert prep["refusals"] == {}, prep["refusals"]
    # 1 network + 2 role-distinct snap entries (NOT 2 — a shared entry
    # would replay the first role's log_extras into the second)
    from urban_network_analysis.Execution import CacheOptions
    from urban_network_analysis.cache.store import ContentStore
    store = ContentStore(CacheOptions(
        mode="disk", directory=os.path.join(str(out), PREPARED_DIRNAME),
        verification="on_read"))
    assert store.stats()["entries"] == 3, store.stats()

    assert tree_bytes(out, skip_hidden=True) == \
        tree_bytes(tmp_path / "ser", skip_hidden=True)


# ---------------------------------------------------------------------------
# E7 + E8: refusal taxonomy and NO_FIT
# ---------------------------------------------------------------------------

def test_session_refusal_taxonomy(tmp_path):
    root = str(tmp_path / "refusals")
    for state, reason in (
            ({"turns": True}, "turns_row"),
            ({"obstacle_points_file": "obs.geojson"}, "obstacle_row"),
            ({"network_file": "n.gpkg", "origins_file": "o.geojson",
              "destinations_file": "d.geojson"},
             "format_not_admitted:network"),
            ({"origins_file": "o.parquet"}, "format_not_admitted:origins"),
            ({"destinations_file": "d.feather"},
             "format_not_admitted:destinations"),):
        s = _prep_session(root, state=state)
        assert s.admitted is False
        assert list(s.counters["refusals"]) == [reason], s.counters
        assert s.stage_cache("una:topology:v1") is None
        assert s.stage_cache("una:snap:v1") is None
    s = _prep_session(root)
    assert s.admitted is True and s.counters["refusals"] == {}
    assert s.stage_cache("una:topology:v1") is not None
    # non-admitted stage: counted, None
    assert s.stage_cache("una:awe:o_access:v1") is None
    assert list(s.counters["refusals"]) == \
        ["stage_not_admitted:una:awe:o_access:v1"]


def test_no_fit_serves_without_retention(tmp_path):
    """E8: a cap too small for the payload never governs correctness —
    arrays are served, arena_nofit is counted, nothing is retained."""
    from urban_network_analysis.batch import preparation as P
    root = str(tmp_path / "nofit")
    s = _prep_session(root, cap=1)          # 1 byte: nothing ever fits
    fl = s.stage_cache("una:topology:v1")
    arrays = {"w": np.arange(4, dtype=np.float64)}
    a1, m1, mk1 = fl.arrays_once(_key("nofit"), lambda: (arrays, {"s": 1}))
    assert np.array_equal(a1["w"], arrays["w"]) and m1 == {"s": 1}
    assert isinstance(mk1, P.StageProduced)
    assert s.counters["arena_nofit"] >= 1
    assert len(s.pair().arena) == 0          # nothing retained
    a2, _, mk2 = fl.arrays_once(_key("nofit"), lambda: (arrays, {"s": 1}))
    assert np.array_equal(a2["w"], arrays["w"])
    assert isinstance(mk2, P.StageHit)       # store served (verified)
    assert len(s.pair().arena) == 0


# ---------------------------------------------------------------------------
# E9 + E10: detach law and mutation isolation
# ---------------------------------------------------------------------------

def test_detach_law_pickle_refusal(tmp_path):
    import pickle
    s = _prep_session(str(tmp_path / "prep"))
    with pytest.raises(TypeError, match="detach"):
        pickle.dumps(s)


def test_served_copy_mutation_isolation(tmp_path):
    from urban_network_analysis.batch import preparation as P
    s = _prep_session(str(tmp_path / "e10"))
    fl = s.stage_cache("una:snap:v1")
    k = _key("e10")
    fl.arrays_once(k, lambda: ({"w": np.arange(3, dtype=np.float64)}, {}))
    got = fl.arrays_once(k, lambda: ({}, None))[0]
    assert isinstance(got, dict) and "w" in got
    got["w"][0] = 999.0                      # mutate the SERVED copy
    again = fl.arrays_once(k, lambda: ({}, None))[0]
    assert again["w"][0] == 0.0              # next serve unaffected
    assert s.counters["verified_hit"] == 2


# ---------------------------------------------------------------------------
# NR-4/NR-5 (P02R review): the empty-payload OWNER law, committed as a
# regression test (its evidence previously lived only in the dev smoke)
# ---------------------------------------------------------------------------

def _store_manifests(root):
    out = []
    for dirpath, _dirs, names in os.walk(os.path.join(str(root), "store")):
        out.extend(os.path.join(dirpath, n) for n in names
                   if n == "manifest.json")
    return out


def test_empty_payload_owner_law(tmp_path):
    """The owner branch commits NOTHING and retains NOTHING for an
    unrepresentable payload — the ({}, None) sentinel, an all-None
    encode (NR-5: encode-once guard, no ValueError out of put()), or a
    TypeError dtype — and a real producer on the same key then builds,
    commits and serves normally (no poisoned empty entry)."""
    root = tmp_path / "prep"
    s = _prep_session(str(root))
    fl = s.stage_cache("una:topology:v1")
    k = _key("empty-payload")

    arr, _meta, marker = fl.arrays_once(k, lambda: ({}, None))
    assert marker.__class__.__name__ == "StageProduced"
    assert arr == {}
    assert len(_store_manifests(root)) == 0          # nothing committed
    assert len(s.pair().arena) == 0                  # nothing retained
    assert s.counters["cold_build"] == 1

    # all-None dict: the encode drops every entry — stays uncached
    # (NR-5) instead of raising ValueError out of store.put
    arr2, _m2, marker2 = fl.arrays_once(
        _key("empty-payload-2"), lambda: ({"w": None}, {}))
    assert marker2.__class__.__name__ == "StageProduced"
    assert len(_store_manifests(root)) == 0
    assert s.counters["cold_build"] == 2

    # TypeError dtype: stays uncached, never fails the row
    arr3, _m3, marker3 = fl.arrays_once(
        _key("empty-payload-3"),
        lambda: ({"w": np.array([1, 2], dtype=object)}, {}))
    assert marker3.__class__.__name__ == "StageProduced"
    assert len(_store_manifests(root)) == 0
    assert s.counters["cold_build"] == 3

    # a real producer on the FIRST key builds and commits normally
    good = {"w": np.arange(3, dtype=np.float64)}
    arr4, _m4, marker4 = fl.arrays_once(k, lambda: (good, {"stage": "t"}))
    assert marker4.__class__.__name__ == "StageProduced"
    assert np.array_equal(arr4["w"], good["w"])
    assert len(_store_manifests(root)) == 1
    assert len(s.pair().arena) == 1
    assert s.counters["cold_build"] == 4
    assert s.counters["fallback_compute"] == 0


# ---------------------------------------------------------------------------
# E6: owner death — waiters complete
# ---------------------------------------------------------------------------

def _hold_flight(root, key, q):
    # module-level (picklable target); the holder sleeps inside produce
    import numpy as np
    from urban_network_analysis.batch import preparation as P
    s = P.PrepSession(root, 1 << 26, {})
    fl = s.stage_cache("una:topology:v1")

    def produce():
        q.put("held")
        time.sleep(60)
        return {"w": np.arange(3, dtype=np.int64)}, {"stage": "held"}

    fl.arrays_once(key, produce)


def test_owner_death_waiter_completes(tmp_path):
    """kill -9 the flight owner: the lock is stolen via pid liveness and
    the waiter completes with correct bits (never a poisoned entry)."""
    import multiprocessing as mp
    root = str(tmp_path / "prep")
    key = _key("owner-death")
    q = mp.Queue()
    p = mp.Process(target=_hold_flight, args=(root, key, q))
    p.start()
    try:
        assert q.get(timeout=120) == "held"
        time.sleep(0.3)          # the owner is inside its commit window
        os.kill(p.pid, signal.SIGKILL)
    finally:
        p.join(timeout=60)
    assert not p.is_alive()

    s = _prep_session(root)
    fl = s.stage_cache("una:topology:v1")
    arr, _meta, marker = fl.arrays_once(
        key, lambda: ({"w": np.arange(3, dtype=np.int64)}, {}))
    assert np.array_equal(arr["w"], np.arange(3, dtype=np.int64))
    assert marker.__class__.__name__ in ("StageProduced", "StageHit")
    assert s.counters["cold_build"] == 1
    assert s.counters["fallback_compute"] == 0


# ---------------------------------------------------------------------------
# E11: snap derivation bit-equality across num_threads
# ---------------------------------------------------------------------------

def test_snap_num_threads_bit_equality(smoke_fixture, tmp_path):
    """E11 (P02D review): the snap payload is pinned bit-equal across
    thread counts BY TEST (it is keyed on the closure, never on
    threads)."""
    tops = []
    for nt in (1, 3):
        t = _run_prepared_topology(smoke_fixture, tmp_path / f"nt{nt}",
                                   num_threads=nt)
        tops.append(t)
    a, b = tops
    oa, ob = a.origins, b.origins
    for attr in ("node_weight", "weight_to_start", "weight_to_end",
                 "nearest_edge_id", "edge_start_node", "edge_end_node"):
        va, vb = getattr(oa, attr), getattr(ob, attr)
        assert np.array_equal(np.asarray(va), np.asarray(vb)), attr
    assert list(g.wkb_hex for g in oa.geometry) == \
        list(g.wkb_hex for g in ob.geometry)


def _run_prepared_topology(data_folder, root, num_threads=1):
    """AddNetwork + AddOrigins through an attached PrepSession; the
    resulting Topology builds its own store root (cold build)."""
    from urban_network_analysis.Topology import Topology
    s = _bare_settings(data_folder)
    t = Topology(verbosity=0, num_threads=num_threads)
    t._prepared_session = _prep_session(str(root))
    t.AddNetwork(s)
    t.AddOrigins(s)
    return t


# ---------------------------------------------------------------------------
# E2: content-derived log replay (NF-1 CRS line included)
# ---------------------------------------------------------------------------

def test_prepared_hit_replays_log_stream(smoke_fixture, tmp_path):
    """Owner and hit log streams agree on every content-derived line:
    'CRS set to ...' (NF-1), 'Network added from source ...' — the hit
    adds only its own v=2 announcement."""
    from urban_network_analysis.Topology import Topology
    s = _bare_settings(smoke_fixture)
    root = str(tmp_path / "prep")
    t1 = Topology(verbosity=0)
    t1._prepared_session = _prep_session(root)
    t1.AddNetwork(s)
    owner = [(e["event"], e["details"]) for e in t1.logger.log_list]

    # the UNCACHED route's stream (no session at all): the prepared
    # owner must be byte-identical to it (NR-1 — including the
    # unconditional tail line that sits below the seam's early return)
    t0 = Topology(verbosity=0)
    t0.AddNetwork(s)
    original = [(e["event"], e["details"]) for e in t0.logger.log_list]
    assert owner, "owner logged nothing"
    assert owner == original, (owner, original)

    t2 = Topology(verbosity=0)
    t2._prepared_session = _prep_session(root)
    t2.AddNetwork(s)
    hit = [(e["event"], e["details"]) for e in t2.logger.log_list]

    # the hit stream adds exactly its own v=2 announcement; every other
    # content-derived line must be byte-identical — the FULL produce
    # stream replays (NR-2): the sindex line inside BuildTopology and
    # the NF-1 CRS line included
    hit_owner = [x for x in hit
                 if not x[1].startswith("Topology restored from cache")]
    assert owner == hit_owner, (owner, hit_owner)
    assert len(hit_owner) == len(hit) - 1
    assert any(d == "edge_sindex created" for _e, d in hit), hit
    assert any(d.startswith("CRS set to ") for _e, d in hit), hit
    assert any(d.startswith("Network added from source ")
               for _e, d in hit)
    # counters: t1's session built; t2's session — SAME process, so the
    # same registry Stores pair — served an arena hit (cold_build == 0)
    ses = t2._prepared_session
    assert ses.counters["cold_build"] == 0, ses.counters
    assert ses.counters["verified_hit"] == 1


# ---------------------------------------------------------------------------
# E5: byte-identity invalidation (changed bytes, unchanged size/mtime)
# ---------------------------------------------------------------------------

def test_invalidation_on_same_size_mtime_change(smoke_fixture, tmp_path):
    """Swapping two digits in the network file (same size, restored
    mtime) must invalidate: a fresh build with DIFFERENT geometry, not
    a stale hit."""
    import shutil
    work = tmp_path / "data"
    shutil.copytree(smoke_fixture, work)
    net = work / "network.geojson"
    t1 = _run_prepared_topology(work, tmp_path / "prep1")
    st = os.stat(net)
    raw = net.read_text()
    # swap the first two digit characters INSIDE the coordinate block —
    # same byte length, restored mtime: identity is the BYTE DIGEST,
    # never size or mtime (the grid's property fields also carry digits)
    ci = raw.index('"coordinates"')
    idx = next(i for i in range(ci, len(raw))
               if raw[i].isdigit() and raw[i] != '0')
    j = next(i for i in range(idx + 1, len(raw))
             if raw[i].isdigit() and raw[i] != raw[idx])
    raw2 = raw[:idx] + raw[j] + raw[idx + 1:j] + raw[idx] + raw[j + 1:]
    assert len(raw2) == len(raw) and raw2 != raw
    net.write_text(raw2)
    os.utime(net, (st.st_atime, st.st_mtime))

    from urban_network_analysis.Topology import Topology
    s = _bare_settings(work)
    t2 = Topology(verbosity=0)
    ses = _prep_session(str(tmp_path / "prep1"))
    t2._prepared_session = ses
    t2.AddNetwork(s)
    g1 = [g.wkb_hex for g in t1.network.geometry]
    g2 = [g.wkb_hex for g in t2.network.geometry]
    assert g1 != g2, "stale hit served after byte change"
    # the changed bytes forced a fresh build under a different key (a
    # NEW session — counters are its own: one build, zero hits)
    assert ses.counters["cold_build"] == 1, ses.counters
    assert ses.counters["verified_hit"] == 0


# ---------------------------------------------------------------------------
# corruption + file-disappearance chronology
# ---------------------------------------------------------------------------

def test_corrupted_entry_rebuilds(tmp_path):
    """A corrupted store entry never serves: verification fails on read,
    the flight rebuilds, and the served bits are the producer's."""
    from urban_network_analysis.batch import preparation as P
    root = tmp_path / "prep"
    s = _prep_session(str(root))
    fl = s.stage_cache("una:topology:v1")
    k = _key("corrupt")
    good = {"w": np.arange(5, dtype=np.float64)}
    fl.arrays_once(k, lambda: (good, {"stage": "t"}))

    # corrupt ONE committed payload byte on disk
    store_dir = root / "store"
    target = None
    for dirpath, _dirs, names in os.walk(store_dir):
        for name in names:
            p = os.path.join(dirpath, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0 \
                    and not name.endswith(".lock"):
                if target is None or os.path.getsize(p) > \
                        os.path.getsize(target):
                    target = p
    assert target is not None
    blob = bytearray(open(target, "rb").read())
    blob[-1] ^= 0xFF
    open(target, "wb").write(bytes(blob))

    # release THIS process's arenas: otherwise the fresh session would be
    # served from the arena and never observe the corrupted disk bytes
    from urban_network_analysis.batch import preparation as P
    P.close_registry()
    P.reopen_registry()
    fl2 = _prep_session(str(root)).stage_cache("una:topology:v1")
    arr, _meta, marker = fl2.arrays_once(
        k, lambda: (good, {"stage": "t"}))
    assert np.array_equal(arr["w"], good["w"])
    assert marker.__class__.__name__ == "StageProduced"


def test_file_disappearance_gates_at_seam(smoke_fixture, tmp_path):
    """The per-row snapshot gate runs BEFORE the flight consult: a
    missing source file raises the uncached path's FileNotFoundError
    even when the arena already holds the entry (chronology law)."""
    import shutil
    work = tmp_path / "data"
    shutil.copytree(smoke_fixture, work)
    t1 = _run_prepared_topology(work, tmp_path / "prep")
    assert t1.network is not None

    (work / "network.geojson").unlink()
    from urban_network_analysis.Topology import Topology
    s = _bare_settings(work)
    t2 = Topology(verbosity=0)
    t2._prepared_session = _prep_session(str(tmp_path / "prep"))
    with pytest.raises(FileNotFoundError):
        t2.AddNetwork(s)


# ---------------------------------------------------------------------------
# R5 (una_10x-next): decoded-stage lease reuse — the lease tier laws
# ---------------------------------------------------------------------------

def _prepared_topology(root, data_folder, num_threads=1):
    """Fresh Topology with a fresh PrepSession on ``root``.  Same process
    across calls: the registry Stores pair — and with it the decoded
    residency — persists between rows, exactly as inside one worker."""
    from urban_network_analysis.Topology import Topology
    s = _bare_settings(data_folder)
    t = Topology(verbosity=0, num_threads=num_threads)
    t._prepared_session = _prep_session(str(root))
    return t, s


def test_r5_lease_shares_decoded_state_with_fresh_numerics(
        smoke_fixture, tmp_path):
    """Rows 2+ of one worker lease the decoded resident: shared
    GeometryArray + shared cached STRtree (the R5 point), fresh wrappers
    and FRESH numeric copies per row (E10 unchanged), byte-equal
    geometry and CRS, and mutation isolation — a row that mutates what
    it received never corrupts the next serve."""
    root = tmp_path / "prep"
    t1, s = _prepared_topology(root, smoke_fixture)
    t1.AddNetwork(s)
    c1 = t1._prepared_session.counters
    assert c1["cold_build"] == 1 and c1["lease_hits"] == 0
    # the owner row registers nothing (its objects are its own)
    assert len(t1._prepared_session.pair().decoded) == 0

    t2, _ = _prepared_topology(root, smoke_fixture)
    t2.AddNetwork(s)  # arena hit -> decodes once and registers
    c2 = t2._prepared_session.counters
    assert c2["verified_hit"] == 1 and c2["lease_hits"] == 0
    assert c2["decoded_nofit"] == 0
    assert len(t2._prepared_session.pair().decoded) == 1

    t3, _ = _prepared_topology(root, smoke_fixture)
    t3.AddNetwork(s)  # lease: the per-row decode is skipped
    c3 = t3._prepared_session.counters
    assert c3["lease_hits"] == 1 and c3["verified_hit"] == 1
    assert len(t3._prepared_session.pair().decoded) == 1

    g2, g3 = t2.network, t3.network
    # the lease law: shared decoded array + shared tree, fresh wrapper
    assert g3.edge_sindex is g2.edge_sindex
    assert g3.geometry.values is g2.geometry.values
    assert g3.geometry is not g2.geometry
    # E10 unchanged: fresh numeric copies, byte-equal
    assert g3.lengths is not g2.lengths
    assert np.array_equal(g3.lengths, g2.lengths)
    assert np.array_equal(g3.weights, g2.weights)
    assert np.array_equal(g3.node_points, g2.node_points)
    assert g3.geometry.crs == g2.geometry.crs
    assert ([g.wkb_hex for g in g3.geometry]
            == [g.wkb_hex for g in t1.network.geometry])

    # mutation isolation: a row's writes die with the row
    g3.lengths[0] = -999.0
    g3.weights[0] = -999.0
    t4, _ = _prepared_topology(root, smoke_fixture)
    t4.AddNetwork(s)
    assert t4.network.lengths[0] == g2.lengths[0]
    assert t4.network.weights[0] == g2.weights[0]
    assert t4.network.edge_sindex is g2.edge_sindex
    assert t4.network.geometry.values is g2.geometry.values


def test_r5_lease_stream_is_row_invariant(smoke_fixture, tmp_path):
    """A lease row's log stream is byte-identical to the registering
    row's stream (E2 across the lease tier) — for arena-hit registration
    (source 'memory') and, after wiping the in-process residency, for
    disk-hit registration (recorded generation/source replayed)."""
    from urban_network_analysis.batch import preparation as P
    root = tmp_path / "prep"
    t1, s = _prepared_topology(root, smoke_fixture)
    t1.AddNetwork(s)
    t2, _ = _prepared_topology(root, smoke_fixture)
    t2.AddNetwork(s)  # arena hit -> registers
    t3, _ = _prepared_topology(root, smoke_fixture)
    t3.AddNetwork(s)  # lease
    hit = [(e["event"], e["details"]) for e in t2.logger.log_list]
    lease = [(e["event"], e["details"]) for e in t3.logger.log_list]
    assert hit, "hit row logged nothing"
    assert lease == hit, (lease, hit)
    assert any(d.startswith("Topology restored from cache")
               for _e, d in lease)

    # disk variant: wipe arena+residency; the registering row re-decodes
    # from the verified store and the lease replays ITS record
    P.close_registry()
    P.reopen_registry()
    t4, _ = _prepared_topology(root, smoke_fixture)
    t4.AddNetwork(s)  # store hit (disk)
    t5, _ = _prepared_topology(root, smoke_fixture)
    t5.AddNetwork(s)  # lease of the disk-registered resident
    hit_d = [(e["event"], e["details"]) for e in t4.logger.log_list]
    lease_d = [(e["event"], e["details"]) for e in t5.logger.log_list]
    assert lease_d == hit_d, (lease_d, hit_d)
    assert any(d.startswith("Topology restored from cache") and "disk" in d
               for _e, d in lease_d), lease_d


def test_r5_snap_lease_and_role_separation(smoke_fixture, tmp_path):
    """Snap stages lease under the ROLE-CARRYING store key (BF-1):
    origins and destinations register separate residents and lease
    independently; served snap objects are fresh per row."""
    root = tmp_path / "prep"
    ts = []
    for _i in range(3):
        t, s = _prepared_topology(root, smoke_fixture)
        t.AddNetwork(s)
        t.AddOrigins(s)
        t.AddDestinations(s)
        ts.append(t)
    c2 = ts[1]._prepared_session.counters
    assert c2["lease_hits"] == 0 and c2["verified_hit"] == 3
    # three separate residents: network + origins + destinations (roles
    # never collide even though the observed _stage_keys stay role-free)
    assert len(ts[1]._prepared_session.pair().decoded) == 3
    c3 = ts[2]._prepared_session.counters
    assert c3["lease_hits"] == 3 and c3["verified_hit"] == 3
    p2, p3 = ts[1].origins, ts[2].origins
    assert p3.nearest_edge_id is not p2.nearest_edge_id
    assert np.array_equal(p3.nearest_edge_id, p2.nearest_edge_id)
    assert p3.node_weight is not p2.node_weight
    assert np.array_equal(p3.node_weight, p2.node_weight)
    assert ([g.wkb_hex for g in p3.geometry]
            == [g.wkb_hex for g in p2.geometry])
    assert ([g.wkb_hex for g in ts[2].destinations.geometry]
            == [g.wkb_hex for g in ts[1].destinations.geometry])


def test_r5_flight_lease_tier_and_nofit(tmp_path):
    """Flight unit laws: a registered resident is served WITHOUT calling
    produce (lease tier, counters carry the truth); an oversized resident
    is refused (decoded_nofit) and the row falls through to the payload
    tiers; mark() exposes the new counters."""
    from urban_network_analysis.batch.preparation import _StageLease
    root = tmp_path / "prep"
    s1 = _prep_session(str(root))
    fl = s1.stage_cache("una:topology:v1")
    k = _key("lease-unit")
    res = {"geometry_array": None, "tree": None, "crs": None,
           "arrays": {"w": np.arange(4, dtype=np.float64)}}
    assert fl.remember_decoded(k, res, 128, {"stage": "t"}, 7, "disk")
    called = []

    def _prod():
        called.append(1)
        return {}, None

    arr, meta, marker = fl.arrays_once(k, _prod)
    assert arr is None and meta == {"stage": "t"}
    assert isinstance(marker, _StageLease)
    assert marker.resident is res
    assert marker.key == k and marker.generation == 7
    assert marker.source == "disk"
    assert called == []  # produce never ran
    c = s1.counters
    assert c["lease_hits"] == 1 and c["verified_hit"] == 1
    assert c["cold_build"] == 0
    m = s1.mark()
    assert "lease_hits" in m and "decoded_nofit" in m

    # NO_FIT: an oversized resident is refused and counted; the same key
    # then falls through to the payload tiers (no lease)
    root2 = tmp_path / "prep2"
    s2 = _prep_session(str(root2), cap=64)
    fl2 = s2.stage_cache("una:topology:v1")
    k2 = _key("nofit-lease")
    good = {"w": np.arange(3, dtype=np.float64)}
    fl2.arrays_once(k2, lambda: (good, {"stage": "t"}))
    big = {"arrays": {"w": np.zeros(256, dtype=np.float64)}}
    assert not fl2.remember_decoded(k2, big, 4096, {}, None, "disk")
    assert s2.counters["decoded_nofit"] == 1
    _arr, _meta, marker2 = fl2.arrays_once(
        k2, lambda: (good, {"stage": "t"}))
    assert not isinstance(marker2, _StageLease)
    assert s2.counters["decoded_nofit"] == 1
    assert s2.counters["lease_hits"] == 0


def test_r5_decoded_residency_lru_and_nofit():
    """DecodedResidency unit laws: LRU eviction under the cap, LRU
    refresh on get, NO_FIT above the whole cap (ledger unchanged),
    drop/len."""
    from urban_network_analysis.batch.preparation import DecodedResidency
    d = DecodedResidency(100)
    assert d.put("a", {"n": 1}, 40, {}, None, "disk")
    assert d.put("b", {"n": 2}, 40, {}, None, "disk")
    assert d.get("a") is not None  # refresh: 'a' becomes newest
    assert d.put("c", {"n": 3}, 40, {}, None, "disk")
    assert d.get("b") is None  # 'b' evicted, 'a' retained
    assert d.get("a") is not None and d.get("c") is not None
    assert not d.put("d", {"n": 4}, 200, {}, None, "disk")  # NO_FIT
    assert d.get("a") is not None and d.get("c") is not None
    assert len(d) == 2
    d.drop("a")
    assert len(d) == 1 and d.get("a") is None


def test_r5_invalidation_never_leases_stale(smoke_fixture, tmp_path):
    """A byte change (same size, restored mtime) yields a DIFFERENT key:
    the old resident is never consulted and the fresh build serves the
    new bytes (E5 extended to the lease tier)."""
    import shutil
    work = tmp_path / "data"
    shutil.copytree(smoke_fixture, work)
    net = work / "network.geojson"
    t1, s = _prepared_topology(tmp_path / "prep", work)
    t1.AddNetwork(s)
    t2, _ = _prepared_topology(tmp_path / "prep", work)
    t2.AddNetwork(s)  # arena hit -> registers
    assert t2._prepared_session.counters["lease_hits"] == 0
    t2b, _ = _prepared_topology(tmp_path / "prep", work)
    t2b.AddNetwork(s)
    assert t2b._prepared_session.counters["lease_hits"] == 1

    st = os.stat(net)
    raw = net.read_text()
    ci = raw.index('"coordinates"')
    idx = next(i for i in range(ci, len(raw))
               if raw[i].isdigit() and raw[i] != '0')
    j = next(i for i in range(idx + 1, len(raw))
             if raw[i].isdigit() and raw[i] != raw[idx])
    raw2 = raw[:idx] + raw[j] + raw[idx + 1:j] + raw[idx] + raw[j + 1:]
    assert len(raw2) == len(raw) and raw2 != raw
    net.write_text(raw2)
    os.utime(net, (st.st_atime, st.st_mtime))

    t3, _ = _prepared_topology(tmp_path / "prep", work)
    t3.AddNetwork(s)
    c3 = t3._prepared_session.counters
    assert c3["lease_hits"] == 0 and c3["cold_build"] == 1, c3
    assert ([g.wkb_hex for g in t3.network.geometry]
            != [g.wkb_hex for g in t1.network.geometry])


def test_r5_batch_warm_leases_bitwise(smoke_fixture, tmp_path, make_batch):
    """Public route: an 8-row parallel run over one input identity
    leases the decoded residents in every worker that repeats a stage
    key, and the exports stay bitwise vs the serial route — cold AND
    warm repeats."""
    ser = make_batch(8, out_folder=tmp_path / "ser")
    ser.RunBatch("accessibility")

    out = tmp_path / "par"
    par = make_batch(8, out_folder=out)
    par.RunBatch("accessibility", parallel=True, workers=2)
    prep = (getattr(par.batch_report, "extra", None) or {}).get(
        "preparation")
    assert prep is not None and prep["rows"] == 8
    assert prep["fallback_compute"] == 0 and prep["refusals"] == {}
    assert prep["lease_hits"] >= 1, prep  # the tier engaged
    cold = tree_bytes(out, skip_hidden=True)
    assert cold == tree_bytes(tmp_path / "ser", skip_hidden=True)

    par2 = make_batch(8, out_folder=out)
    par2.RunBatch("accessibility", parallel=True, workers=2)
    prep2 = (getattr(par2.batch_report, "extra", None) or {}).get(
        "preparation")
    assert prep2 is not None and prep2["stages"] == 24
    assert prep2["cold_build"] == 0 and prep2["fallback_compute"] == 0
    assert prep2["lease_hits"] >= 1, prep2
    assert tree_bytes(out, skip_hidden=True) == cold


def test_r5_optional_fields_survive_residency(tmp_path):
    """Field-completeness law: every optional payload field the restore
    path binds — 'z' foremost (elevation networks) — must survive the
    resident round-trip; a lease row missing z would be a silent parity
    break."""
    import geopandas as gpd
    import shapely
    from urban_network_analysis.Topology import Topology, Network
    from urban_network_analysis.cache.stages import StageHit

    geoms = shapely.LineString([(0, 0), (1, 1)]).wkb
    a = {
        "geometry_wkb": np.frombuffer(geoms, dtype=np.uint8),
        "geometry_wkb_lens": np.array([len(geoms)], dtype=np.int64),
        "lengths": np.array([2.0], dtype=np.float64),
        "weights": np.array([2.0], dtype=np.float64),
        "node_points": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "start_nodes": np.array([0]),
        "end_nodes": np.array([1]),
        "degrees": np.array([1, 1], dtype=np.int64),
        "z": np.array([7.5, 9.0]),  # the elevation field
        "start_degrees": np.array([1]),
        "end_degrees": np.array([1]),
        "min_degrees": np.array([1]),
        "max_degrees": np.array([1]),
    }
    m = {"stage": "una:topology:v1", "n_input_features": 1,
         "n_nodes": 2, "n_edges": 1, "crs_projjson": None}
    hit = StageHit(key=_key("z"), generation=3, arrays=a, metadata=m,
                   source="disk")

    t = Topology(verbosity=0)
    t.network = Network(logger=t.logger)
    resident = t._network_resident_from_hit(hit)
    assert "z" in resident["arrays"]
    t._bind_network_from_resident("src.geojson", hit, resident)
    assert t.network.z is not None
    assert np.array_equal(t.network.z, a["z"])
    # the lease row binds a fresh copy, not the resident's array
    assert t.network.z is not a["z"]
    assert len(t.network.geometry) == 1
