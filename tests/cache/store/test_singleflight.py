"""Single-flight: compute-once, cancellation/exception release, stale and
live locks, token-matched release."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from urban_network_analysis.cache import ContentStore
from urban_network_analysis.Execution import CacheOptions

REPO = Path(__file__).resolve().parents[3]

K = "a1b2c3d4" + "0" * 56  # 64 hex chars
PAYLOAD = b"winner-committed-this"


def _lock_path(store, key=K):
    return os.path.join(str(store.options.directory), "locks", f"{key}.lock")


def _read_lock(store, key=K):
    with open(_lock_path(store, key), "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.cache_store
def test_owner_yields_true_and_release_removes_lock(store):
    with store.singleflight(K) as owner:
        assert owner is True
        assert os.path.exists(_lock_path(store))
        info = _read_lock(store)
        assert info["pid"] == os.getpid()
        assert set(info) == {"pid", "token", "host"}
    assert not os.path.exists(_lock_path(store))  # clean release


@pytest.mark.cache_store
def test_loser_regets_instead_of_recomputing(store):
    """The dossier-06 protocol: whoever acquires the lock after the winner
    committed sees False and re-gets the winner's generation."""
    with store.singleflight(K) as winner:
        assert winner is True
        store.put(K, {"out": PAYLOAD})
    with store.singleflight(K) as loser:
        assert loser is False
    assert store.get(K).payload["out"] == PAYLOAD


@pytest.mark.cache_store
def test_no_commit_means_next_caller_owns(store):
    with store.singleflight(K):
        pass  # holder crashed before committing — nothing published
    with store.singleflight(K) as owner:
        assert owner is True


@pytest.mark.cache_store
@pytest.mark.parametrize("exc", [ValueError("worker blew up"),
                                 KeyboardInterrupt("user cancelled"),
                                 TimeoutError("compute deadline")])
def test_exception_and_cancellation_release_the_lock(store, exc):
    with pytest.raises(type(exc)):
        with store.singleflight(K):
            raise exc
    assert not os.path.exists(_lock_path(store))
    with store.singleflight(K) as owner:   # the lock is freely reacquirable
        assert owner is True


def _guaranteed_dead_pid():
    """A pid above pid_max can never be assigned — deterministically dead,
    with no subprocess and no pid-reuse race (reuse on a busy shared node
    made 'spawn a child and use its pid' flaky)."""
    try:
        with open("/proc/sys/kernel/pid_max", "r", encoding="utf-8") as f:
            pid_max = int(f.read().strip())
    except (OSError, ValueError):
        pid_max = 4194304  # Linux default ceiling
    return pid_max + 12345


@pytest.mark.cache_store
def test_stale_lock_from_dead_holder_is_stolen(store):
    dead = _guaranteed_dead_pid()
    os.makedirs(os.path.dirname(_lock_path(store)), exist_ok=True)
    with open(_lock_path(store), "w", encoding="utf-8") as f:
        json.dump({"pid": dead, "token": "crashed-holder", "host": "x"}, f)
    with store.singleflight(K) as owner:
        assert owner is True                    # dead holder's lock stolen
        assert _read_lock(store)["token"] != "crashed-holder"
    assert not os.path.exists(_lock_path(store))


@pytest.mark.cache_store
def test_dead_holder_lock_is_stolen_without_commit_by_waiter(store):
    """Crashed-holder recovery end to end: the waiter that steals the lock
    owns the compute (nothing was committed), and afterwards the key is
    served from its own commit."""
    os.makedirs(os.path.dirname(_lock_path(store)), exist_ok=True)
    with open(_lock_path(store), "w", encoding="utf-8") as f:
        json.dump({"pid": _guaranteed_dead_pid(), "token": "crashed",
                   "host": "x"}, f)
    assert store.get(K) is None
    with store.singleflight(K) as owner:
        assert owner is True
        store.put(K, {"out": PAYLOAD})
    assert store.get(K).payload["out"] == PAYLOAD
    assert not os.path.exists(_lock_path(store))


@pytest.mark.cache_store
@pytest.mark.parametrize("content", ["{not json at all", '{"no": "pid"}',
                                     ""])
def test_unreadable_lock_is_treated_as_stale(store, content):
    os.makedirs(os.path.dirname(_lock_path(store)), exist_ok=True)
    with open(_lock_path(store), "w", encoding="utf-8") as f:
        f.write(content)
    with store.singleflight(K) as owner:
        assert owner is True


@pytest.mark.cache_store
def test_live_foreign_holder_times_out_without_destroying_lock(store):
    """A live holder (its pid exists) is never stolen from — the waiter
    times out and the holder's lock file survives untouched."""
    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3)"])
    try:
        os.makedirs(os.path.dirname(_lock_path(store)), exist_ok=True)
        with open(_lock_path(store), "w", encoding="utf-8") as f:
            json.dump({"pid": holder.pid, "token": "live-holder",
                       "host": "x"}, f)
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="single-flight"):
            with store.singleflight(K, wait_timeout_s=0.3, poll_s=0.01):
                pass
        assert time.monotonic() - t0 >= 0.3
        assert _read_lock(store)["token"] == "live-holder"  # not destroyed
    finally:
        holder.terminate()
        holder.wait()


@pytest.mark.cache_store
def test_release_never_unlinks_a_stolen_and_reissued_lock(store):
    """Token matching: if our lock was stolen and reissued to someone else,
    our exit must not unlink THEIR lock."""
    with store.singleflight(K):
        # simulate theft + reissue while we still hold the token
        with open(_lock_path(store), "w", encoding="utf-8") as f:
            json.dump({"pid": 999999999, "token": "reissued-to-other",
                       "host": "x"}, f)
    # context exited; the reissued lock must still be there
    assert _read_lock(store)["token"] == "reissued-to-other"


@pytest.mark.cache_store
def test_compute_once_under_thread_race(make_store):
    """N threads all miss, all enter single-flight: exactly one computes;
    everyone ends on the committed generation (dossier 06 inflight test)."""
    store = make_store()
    computes = []
    barrier = threading.Barrier(4)
    errors = []

    def worker():
        try:
            barrier.wait()
            hit = store.get(K)
            if hit is None:
                with store.singleflight(K, wait_timeout_s=30) as owner:
                    if owner:
                        computes.append(threading.get_ident())
                        time.sleep(0.05)  # widen the race window
                        store.put(K, {"out": PAYLOAD})
        except Exception as e:  # pragma: no cover - reported via errors
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    assert len(computes) == 1
    assert store.get(K).payload["out"] == PAYLOAD
    assert store.stats()["hits"] >= 1


@pytest.mark.cache_store
def test_serialized_threads_use_single_flight_correctly(make_store):
    """The same key from many threads, sequential timing: second arriver
    must see False (commit already visible), not recompute."""
    store = make_store()
    n_compute = 0
    for _ in range(3):
        if store.get(K) is None:
            with store.singleflight(K) as owner:
                if owner:
                    n_compute += 1
                    store.put(K, {"out": PAYLOAD})
                # owner False: fall through to the re-get below
        assert store.get(K).payload["out"] == PAYLOAD
    assert n_compute == 1


@pytest.mark.cache_store
def test_singleflight_on_disabled_store(tmp_path):
    store = ContentStore(CacheOptions(
        mode="off", directory=str(tmp_path / "unused")))
    with store.singleflight(K) as owner:
        assert owner is True


_SF_WORKER = """
import sys
sys.path.insert(0, sys.argv[3])
from urban_network_analysis.Execution import CacheOptions
from urban_network_analysis.cache import ContentStore

d, key = sys.argv[1], sys.argv[2]
store = ContentStore(CacheOptions(mode="disk", directory=d))
if store.get(key) is not None:
    print("LOSER")
else:
    with store.singleflight(key, wait_timeout_s=30, poll_s=0.005) as owner:
        if owner:
            store.put(key, {"out": b"committed-by-the-single-owner"})
            print("OWNER")
        else:
            print("LOSER")
"""


@pytest.mark.cache_store
def test_multiprocess_singleflight_computes_once(store, tmp_path):
    """REAL contending processes (the regime the original implementation
    got wrong: an O_EXCL create followed by a buffered content write left a
    window in which a waiter read an empty lock, declared it stale, and
    stole a live holder's flight — reproduced 1-in-12 by the independent
    review).  The lock is now published atomically WITH its content via
    write-to-tmp + os.link, so across N processes exactly one computes."""
    src = str(REPO / "src")
    procs = [subprocess.Popen(
        [sys.executable, "-c", _SF_WORKER, str(store.options.directory), K,
         src], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True) for _ in range(6)]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, f"worker failed: {err}"
        outs.append(out.strip())
    assert sorted(outs) == ["LOSER"] * 5 + ["OWNER"]
    hit = store.get(K)
    assert hit.payload["out"] == b"committed-by-the-single-owner"
    assert not os.path.exists(_lock_path(store))  # no orphaned locks


@pytest.mark.cache_store
def test_lock_file_is_never_observed_empty_or_partial(store):
    """Invariant behind the atomic publish: at any instant a lock file
    exists, its content is complete ({pid, token, host} JSON).  Threads
    poll the lock while a holder holds it — any empty/partial observation
    would be the MAJOR-1 steal window."""
    stop = threading.Event()
    bad = []

    def watcher():
        while not stop.is_set():
            try:
                with open(_lock_path(store), "r", encoding="utf-8") as f:
                    info = json.load(f)  # raises on empty/partial JSON
                if set(info) != {"pid", "token", "host"}:
                    bad.append(info)
            except FileNotFoundError:
                pass  # no lock right now: fine
            except Exception as e:
                bad.append(repr(e))  # observed a partial/empty lock!

    watchers = [threading.Thread(target=watcher) for _ in range(3)]
    for t in watchers:
        t.start()
    try:
        for _ in range(5):
            with store.singleflight(K):
                time.sleep(0.05)
    finally:
        stop.set()
        for t in watchers:
            t.join(timeout=10)
    assert bad == []
