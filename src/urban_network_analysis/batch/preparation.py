"""Run-scoped prepared input preparation (una_10x P02; dossier 02).

Runtime-only session beside CohortSession: worker rows of one
accessibility batch share a content-addressed preparation store under the
run root, so the network topology and the origin/destination snap layers
are decoded, validated and built ONCE per immutable input identity
(single-flight BEFORE decode), then restored per row from verified bytes.

Laws (prepared_contract.json @ f139dd3, tree 050b316):
- per-row snapshot TOCTOU: the row's own isfile gate and snapshot read
  always run at the Topology seam; reuse keys on verified byte identity.
- BF-1: snap STORE keys carry the role component; the OBSERVED _stage_key
  stays the role-free CACHE_GRAPH value.  Roles are computed at the seam;
  this module is key-opaque.
- memory: retained decoded payloads bounded by the C31-MINOR-1 law
  max(64 MiB, allowance // workers) — the same per_worker_residency_cap
  the cohort route uses.  Retained payloads and producer arrays are
  never handed to a row (copies only, E10).  Store verification is
  pinned 'on_read' (NF-2).  NO_FIT counts arena_nofit and continues
  WITHOUT retention — the cap governs retained bytes, never correctness.
  The per-process registry is BOUNDED (design 02 line 121; NF-8: the
  cohort _PROCESS_STORES unbounded pattern is not copied).
- state machine (design 02): ABSENT -> BUILDING(owner) -> COMPLETE ->
  LEASED (this process's bounded arena).  Owner failure leaves no
  commit; waiters re-elect or compute locally and never suppress their
  own validation/errors.  No unlink in v1 (run-scoped root lifecycle).
- transport (E9): pickling a PrepSession is a loud TypeError — the
  worker must detach before RowResult/state-bundle packing.
- fallback law: every refusal returns None and the row takes the CURRENT
  path with zero observable difference; refusals are counted, never
  silent.

Counters (design 02: different counters, never collapsed): cold_build,
verified_hit, contention_wait, fallback_compute (spin-exhaust/wait-timeout
local compute — NF-7's engagement gate), arena_nofit, refusals.
corruption_rebuild is store-level (store stats at report time).
Engagement law (NF-7): builds <= distinct identities; equality asserted
only when fallback_compute == 0.
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from ..Execution import CacheOptions
from ..cache.stages import (StageHit, StageProduced, decode_arrays,
                            encode_arrays)
from ..cache.store import ContentStore

__all__ = ["PrepSession", "per_worker_residency_cap"]

# One sizing law for both sessions (C31-MINOR-1): the law stays defined
# exactly once, in cohort.py; re-exported for runtime.py.
from .cohort import per_worker_residency_cap  # noqa: E402,F401

# Admitted preparation stages (v1 contract).  Anything else is a counted
# refusal at the seam, and the row takes the current path.
ADMITTED_STAGES = ("una:topology:v1", "una:snap:v1")


def _abytes(arrays: Mapping[str, Any]) -> int:
    """Byte size of an arrays payload (arrays plus raw blobs)."""
    total = 0
    for v in arrays.values():
        if v is not None and hasattr(v, "nbytes"):
            total += int(v.nbytes)
        elif isinstance(v, (bytes, bytearray)):
            total += len(v)
    return total


def _acopy(arrays: Mapping[str, Any]) -> Dict[str, Any]:
    """Fresh copies of every array (E10: nothing retained ever aliases
    into a row; nothing produced is ever retained)."""
    out: Dict[str, Any] = {}
    for k, v in arrays.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = bytes(v)
        elif hasattr(v, "copy"):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


class Arena:
    """Bounded decoded-payload ledger for one process (the LEASED state;
    TraceResidency precedent, review NF-5).  LRU eviction; a claim larger
    than the whole cap is NO_FIT (False, ledger unchanged)."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self._ent: "OrderedDict[str, tuple]" = OrderedDict()
        self._n = 0

    def put(self, key: str, arrays: Mapping[str, Any],
            meta: Mapping[str, object], gen: Optional[int]) -> bool:
        """Retain private copies under ``key``.  True when retained."""
        payload = _acopy(arrays)
        n = _abytes(payload)
        if n > self.cap:
            return False
        self.drop(key)
        while self._n + n > self.cap:
            _, old = self._ent.popitem(last=False)
            self._n -= _abytes(old[0])
        self._ent[key] = (payload, dict(meta), gen)
        self._n += n
        return True

    def get(self, key: str) -> Optional[tuple]:
        """LRU-refresh; returns the RETAINED (arrays, meta, gen) — copy
        before handing anything to a row."""
        e = self._ent.get(key)
        if e is None:
            return None
        self._ent.move_to_end(key)
        return e

    def drop(self, key: str) -> None:
        e = self._ent.pop(key, None)
        if e is not None:
            self._n -= _abytes(e[0])

    def __len__(self) -> int:
        return len(self._ent)


class Stores:
    """Per-process (store, arena) pair for one prepared root."""

    def __init__(self, root: str, cap: int):
        os.makedirs(root, exist_ok=True)
        # NF-2: verification pinned 'on_read' by construction.
        self.store = ContentStore(CacheOptions(mode="disk",
                                               directory=root,
                                               verification="on_read"))
        self.arena = Arena(cap)

    def close(self) -> None:
        # Explicit lease release (design 02 line 120); store bytes stay
        # on disk for the run root's own lifecycle (no unlink in v1).
        self.arena = Arena(self.arena.cap)


_REG_MAX = 4  # design 02 line 121: bounded, not per-distinct-root
_REG: "OrderedDict[str, Stores]" = OrderedDict()
_REG_LOCK = threading.Lock()
_REG_CLOSED = False


def _stores_for(root: str, cap: int) -> Stores:
    """Process-wide pair for ``root``; BOUNDED registry (NF-8): the
    oldest root is closed and evicted when the bound is exceeded."""
    global _REG_CLOSED
    with _REG_LOCK:
        if _REG_CLOSED:
            raise RuntimeError("prepared registry closed (shutdown)")
        p = _REG.get(root)
        if p is None:
            p = Stores(root, cap)
            _REG[root] = p
            while len(_REG) > _REG_MAX:
                _, old = _REG.popitem(last=False)
                old.close()
        else:
            p.arena.cap = int(cap)
        _REG.move_to_end(root)
        return p


def close_registry() -> None:
    """Close every arena and refuse new leases (design 02: refuse new
    leases during shutdown).  Store bytes on disk are untouched."""
    global _REG_CLOSED
    with _REG_LOCK:
        _REG_CLOSED = True
        while _REG:
            _, p = _REG.popitem()
            p.close()


def reopen_registry() -> None:
    """Undo close_registry (fresh serial/test reuse in one process)."""
    global _REG_CLOSED
    with _REG_LOCK:
        _REG_CLOSED = False


class PrepSession:
    """One row's handle on the run's prepared store (per-row instance,
    shared process-wide stores — the CohortSession pattern).

    ``admitted`` is decided at construction from the row's bound settings
    snapshot: no turns (C31-NOTE-3 defense in depth), no obstacles
    (outside the v1 domain), geojson sources only (v1 formats).  A
    refused session counts its refusal and serves nothing.
    """

    def __init__(self, root: str, cap: int,
                 settings_state: Optional[Mapping[str, Any]] = None):
        self.root = root
        self.cap = int(cap)
        self.counters: Dict[str, Any] = {
            "stages": 0,
            "cold_build": 0,
            "verified_hit": 0,
            "contention_wait": 0,
            "fallback_compute": 0,
            "arena_nofit": 0,
            "refusals": {},
            "build_s": 0.0,
        }
        self.admitted = True
        self._st: Optional[Stores] = None
        st = dict(settings_state or {})
        if not root:
            self.refuse("no_root")
        elif st.get("turns"):
            self.refuse("turns_row")
        elif str(st.get("obstacle_points_file") or "").strip():
            self.refuse("obstacle_row")
        else:
            for role, key in (("network", "network_file"),
                              ("origins", "origins_file"),
                              ("destinations", "destinations_file")):
                name = str(st.get(key) or "").strip()
                if name and not name.lower().endswith(".geojson"):
                    self.refuse("format_not_admitted:" + role)

    def __getstate__(self) -> None:
        raise TypeError(
            "PrepSession is runtime-only state; detach before "
            "RowResult/state-bundle transport (E9 detach law)")

    def refuse(self, reason: str) -> None:
        self.admitted = False
        self.counters["refusals"][reason] = \
            self.counters["refusals"].get(reason, 0) + 1

    def pair(self) -> Stores:
        if self._st is None:
            self._st = _stores_for(self.root, self.cap)
        return self._st

    @property
    def store(self) -> ContentStore:
        return self.pair().store

    def close(self) -> None:
        self._st = None  # lease release for THIS handle only

    # -- per-row attribution (CohortSession mark/delta pattern) -----------

    def mark(self) -> dict:
        c = self.counters
        return {"stages": c["stages"], "cold_build": c["cold_build"],
                "verified_hit": c["verified_hit"],
                "contention_wait": c["contention_wait"],
                "fallback_compute": c["fallback_compute"],
                "arena_nofit": c["arena_nofit"],
                "refusals": dict(c["refusals"]),
                "build_s": c["build_s"]}

    def delta(self, before: dict) -> dict:
        c = self.counters
        rb = before["refusals"]
        ref = {k: v - rb.get(k, 0) for k, v in c["refusals"].items()
               if v - rb.get(k, 0) > 0}
        return {"root": self.root,
                "stages": c["stages"] - before["stages"],
                "cold_build": c["cold_build"] - before["cold_build"],
                "verified_hit": c["verified_hit"] - before["verified_hit"],
                "contention_wait":
                    c["contention_wait"] - before["contention_wait"],
                "fallback_compute":
                    c["fallback_compute"] - before["fallback_compute"],
                "arena_nofit": c["arena_nofit"] - before["arena_nofit"],
                "refusals": ref,
                "build_s": c["build_s"] - before["build_s"]}

    # -- the seam surface --------------------------------------------------

    def stage_cache(self, stage: str) -> Optional["Flight"]:
        """The stage's single-flight, or None when this row/session
        cannot use prepared preparation (counted refusal; the row takes
        the current path)."""
        if not self.admitted:
            return None
        if stage not in ADMITTED_STAGES:
            self.refuse("stage_not_admitted:" + stage)
            return None
        return Flight(self)


class Flight:
    """The prepared single-flight for one stage of one row.

    Mirrors StageCache.arrays_once exactly (same spin threshold, same
    timeout fallback, same commit law, same StageProduced marker
    semantics) with session counters observed per outcome, plus the
    bounded arena in front of the store.  Outcomes: verified_hit (arena
    or store, content verified on read), cold_build (owner: produce +
    commit), contention_wait (loser re-got the owner's commit),
    fallback_compute (spin-exhaust or wait-timeout: local compute WITHOUT
    commit).  An unrepresentable producer payload — the empty ``({}, None)``
    sentinel the Topology builders return for a stage that cannot be
    stored, or a TypeError dtype — commits NOTHING and retains NOTHING
    (encode_arrays({}) is a valid empty commit: letting it through would
    poison every waiter's restore); waiters spin out to their own
    local compute.  Arena retention on every successful serve is
    best-effort (NO_FIT counts arena_nofit and continues without
    retention — the cap governs retained bytes, never correctness).
    """

    SPINS = 3  # stages.arrays_once's bounded-loser law, kept equal

    def __init__(self, session: PrepSession):
        self.s = session

    def arrays_once(self, key: str,
                    produce: Callable[[], Tuple[Dict[str, Any],
                                                Optional[Mapping[str, object]]]],
                    ) -> Tuple[Dict[str, Any], Optional[Mapping[str, object]],
                               Union[StageHit, StageProduced]]:
        c = self.s.counters
        p = self.s.pair()
        store, arena = p.store, p.arena
        c["stages"] += 1

        # Arena hit: fresh copies, never the retained objects (E10).
        got = arena.get(key)
        if got is not None:
            arrays, meta, gen = got
            fresh = _acopy(arrays)
            c["verified_hit"] += 1
            return (fresh, dict(meta),
                    StageHit(key=key, generation=gen, arrays=fresh,
                             metadata=dict(meta), source="memory"))

        spins = 0
        while True:
            # Store hit: content verified on read (store law); decoded
            # fresh; retained when the arena admits it.
            hit = store.get(key)
            if hit is not None:
                arrays = decode_arrays(hit.payload)
                if not arena.put(key, arrays, hit.metadata, hit.generation):
                    c["arena_nofit"] += 1
                c["verified_hit"] += 1
                return (arrays, dict(hit.metadata),
                        StageHit(key=key, generation=hit.generation,
                                 arrays=arrays, metadata=dict(hit.metadata),
                                 source=hit.hit))
            try:
                with store.singleflight(key) as owned:
                    if owned:
                        t0 = time.perf_counter()
                        arrays, meta = produce()
                        c["build_s"] += time.perf_counter() - t0
                        if arrays:
                            # Empty payload = the producer's "unrepresentable"
                            # sentinel (Topology returns ({}, None) when the
                            # stage cannot be stored).  encode_arrays({}) is a
                            # valid empty commit, so it must be skipped
                            # EXPLICITLY: a poisoned empty entry would crash
                            # every waiter's restore.  Waiters spin out to
                            # their own local compute (fallback_compute).
                            try:
                                store.put(key, encode_arrays(arrays),
                                          metadata=dict(meta or {}))
                            except TypeError:
                                pass  # unrepresentable dtype: stays uncached
                            if not arena.put(key, arrays, dict(meta or {}),
                                             None):
                                c["arena_nofit"] += 1
                        c["cold_build"] += 1
                        return arrays, meta, StageProduced(key=key)
                # Loser: the owner finished; its commit should be visible.
                hit = store.get(key)
                if hit is not None:
                    arrays = decode_arrays(hit.payload)
                    if not arena.put(key, arrays, hit.metadata,
                                     hit.generation):
                        c["arena_nofit"] += 1
                    c["contention_wait"] += 1
                    return (arrays, dict(hit.metadata),
                            StageHit(key=key, generation=hit.generation,
                                     arrays=arrays,
                                     metadata=dict(hit.metadata),
                                     source=hit.hit))
                spins += 1
                if spins >= self.SPINS:
                    # Owner(s) could not commit; fall back to local
                    # compute instead of spinning.
                    t0 = time.perf_counter()
                    arrays, meta = produce()
                    c["build_s"] += time.perf_counter() - t0
                    c["fallback_compute"] += 1
                    return arrays, meta, StageProduced(key=key)
            except TimeoutError:
                t0 = time.perf_counter()
                arrays, meta = produce()
                c["build_s"] += time.perf_counter() - t0
                c["fallback_compute"] += 1
                return arrays, meta, StageProduced(key=key)
