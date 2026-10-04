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
    owner = [(e["event"], e["details"]) for e in t1.logger.log_list
             if e["event"] == "Network added"]

    t2 = Topology(verbosity=0)
    t2._prepared_session = _prep_session(root)
    t2.AddNetwork(s)
    hit = [(e["event"], e["details"]) for e in t2.logger.log_list
           if e["event"] == "Network added"]

    assert owner, "owner logged nothing"
    # the hit stream adds exactly its own v=2 announcement; every other
    # content-derived line must be byte-identical
    hit_owner = [x for x in hit
                 if not x[1].startswith("Topology restored from cache")]
    assert owner == hit_owner, (owner, hit_owner)
    assert len(hit_owner) == len(hit) - 1
    assert any(d.startswith("CRS set to ") for _e, d in hit), hit
    assert any(d.startswith("Network added from source ") for _e, d in hit)
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
