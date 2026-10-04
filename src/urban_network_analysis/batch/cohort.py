"""Cohort route (dossier 06, campaign una_10x C30).

Integrator-owned share route for the PARALLEL batch's worker rows: rows
proven independent by the planner (``worker_admissible``) whose analysis
is accessibility run their dominant ``o_access`` stage against a shared,
content-addressed routing trace instead of one search per row.

Laws this module keeps (each one traceable to a reviewed gate):

* **Engine files untouched.**  The wrapper (:class:`CohortGraphEngine`)
  duck-types the compact graph view's ``o_scope``/``o_access`` surface
  and is injected at ``accessibility.graph_engine`` by the UNA
  integrator dispatch — never by the engine package.  ``o_scope``
  delegates unchanged; only ``o_access`` may take the cohort route.
* **Sparse gate.**  Every refusal — keys not canonicalizable, trace
  build refused or budget-exceeded, residency corruption, replay
  refusal, ANY unexpected cohort-route exception — falls back to the
  inner ``o_access`` unchanged.  The scientific result is always the
  engine's own; the cohort route can only save work, never change a
  number (the fallback recomputes from scratch).  Refusals are counted,
  not swallowed silently.
* **Bitwise parity.**  The route replays through the engine's own
  reviewed fold path (``replay_trace`` → ``reach_gravity_knn_access``)
  with the EXACT argument threading ``o_access`` uses — including the
  raw ``knn_weights`` array (never re-typed) and the identical
  ``gravity_growth_rate`` division.  C11/C12's characterization suites
  proved replay ≡ fold bitwise across lengths, ties, boundaries,
  extremes, branches, specializations and layouts.
* **Produce-once (C20-NOTE-2).**  Build → publish is wrapped in
  ``ContentStore.singleflight(key)``: one worker builds per trace
  identity per host; the rest fault in from the store's decoded-arena
  residency (full identity verification at every load).
* **Identity.**  The trace identity is the C11 ``trace_id`` (canonical
  frame hash over the graph arrays, cutoff, epoch rollover and producer
  profile) — destination weights and metric parameters are replay
  INPUTS, not identity, so one trace serves every row sharing the
  engine key and cutoff (identical network/origins/destinations stage
  keys, obstacles, elevation and search radius).  Grouping is
  runtime-resolved: workers derive identities after the graph build and
  report them; the coordinator aggregates — it never builds topologies
  to pre-group.
* **Runtime-only transport.**  The cohort store root and the per-worker
  residency cap ride :class:`~urban_network_analysis.batch.worker.RowJob`
  as runtime state (like ``staging_dir``); they are never Settings and
  never observable serial state.  The store root is a hidden,
  content-addressed directory beside the row outputs (like the staging
  root, it persists — warm resumes are a feature; it is safe to delete).

Residency sizing (C20-MAJOR-1 owner task, from RESOURCES.md): the host
working budget is ``0.60 * available RAM`` at admission; the initial
trace resident allowance is at most 25% of that budget, split across
the effective workers so the sum of per-worker decoded arenas stays
inside the allowance.  The parent derives the per-worker cap once and
passes it down; workers never re-derive it.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..cache import identity as _ckid
from ..Engines import _trace_access_scratch as trm
from ..Execution import CacheOptions

__all__ = ["CohortSession", "CohortGraphEngine",
           "per_worker_residency_cap", "record_meta"]

_SCHEMA_VERSION = trm.TRACE_SCHEMA_VERSION

_PRODUCER_SHA256 = hashlib.sha256(
    open(trm.__file__, "rb").read()).hexdigest()


def _producer_sha256() -> str:
    """The trace module's own byte digest — the producer identity the
    residency binds into every key and load-time check (cached: the
    module bytes cannot change under a running process)."""
    return _PRODUCER_SHA256


def per_worker_residency_cap(workers_effective: int) -> int:
    """RESOURCES.md sizing: 25% of the host working budget
    (0.60 * available RAM at admission), split across the effective
    workers.  Floor of 64 MiB total so tiny hosts still cohort."""
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        available = 0
    if available <= 0:
        return 1 << 26
    allowance = int(0.25 * 0.60 * available)
    return max(1 << 26, allowance // max(1, int(workers_effective)))


def record_meta(rec: trm.TraceRecord) -> dict:
    """The record's OWN identity fields, verbatim (the rv6 mapping)."""
    return {"trace_id": rec.trace_id,
            "trace_schema_version": rec.schema_version,
            "dest_count": int(rec.dest_count),
            "cutoff_bits": np.float64(rec.cutoff_value).tobytes().hex(),
            "cutoff_dtype": rec.cutoff_dtype,
            "reach_dtype": rec.reach_dtype,
            "metric_dtype": rec.metric_dtype}


class _CohortStores:
    """Per-process (store, residency) pair for one cohort root.

    A worker process runs many rows; the pair is built once per root and
    kept for the process lifetime, so within-worker repeats hit the
    decoded residency directly and cross-worker sharing goes through the
    disk store's single flight.
    """

    def __init__(self, root: str, residency_cap_bytes: int):
        from ..cache.store import ContentStore
        from ..cache.trace_residency import TraceResidency
        os.makedirs(root, exist_ok=True)
        self.store = ContentStore(CacheOptions(mode="disk",
                                               directory=root))
        self.residency = TraceResidency(
            self.store,
            max_resident_bytes=int(residency_cap_bytes),
            producer_sha256=_producer_sha256())

    def close(self) -> None:
        try:
            self.residency.close()
        except Exception:
            pass


_PROCESS_STORES: Dict[str, _CohortStores] = {}
_PROCESS_STORES_LOCK = threading.Lock()


def _stores_for(root: str, residency_cap_bytes: int) -> _CohortStores:
    with _PROCESS_STORES_LOCK:
        pair = _PROCESS_STORES.get(root)
        if pair is None:
            pair = _CohortStores(root, residency_cap_bytes)
            _PROCESS_STORES[root] = pair
        return pair


class CohortSession:
    """Run-scoped cohort share state (one per worker process per run).

    ``counters`` accumulates over rows; :meth:`mark`/:meth:`delta`
    attribute a per-row slice for the RowResult transport.
    """

    def __init__(self, root: str, residency_cap_bytes: int):
        self.root = root
        self.residency_cap_bytes = int(residency_cap_bytes)
        self.counters: Dict[str, Any] = {
            "rows_attempted": 0,
            "builds": 0,             # this process ran the flight owner's build
            "faultins": 0,           # decoded packed bytes from the store
            "resident_hits": 0,      # served from this process's arenas
            "singleflight_lost": 0,  # waited for another worker's flight
            "replays": 0,
            "folds_done": 0,
            "refusals": {},          # reason -> count
            "build_s": 0.0,
            "faultin_s": 0.0,
            "replay_s": 0.0,
        }
        self.last_trace_id: Optional[str] = None
        self.last_result_key: Optional[str] = None

    # -- per-row attribution -------------------------------------------------

    def mark(self) -> dict:
        """Snapshot the accumulators before a row (cheap deep-enough)."""
        c = self.counters
        return {"rows": c["rows_attempted"], "builds": c["builds"],
                "faultins": c["faultins"],
                "resident_hits": c["resident_hits"],
                "singleflight_lost": c["singleflight_lost"],
                "replays": c["replays"], "folds_done": c["folds_done"],
                "refusals": dict(c["refusals"]),
                "build_s": c["build_s"], "faultin_s": c["faultin_s"],
                "replay_s": c["replay_s"]}

    def delta(self, before: dict) -> dict:
        """The counters slice this row produced."""
        c = self.counters
        ref_before = before["refusals"]
        refusals = {k: v - ref_before.get(k, 0)
                    for k, v in c["refusals"].items()
                    if v - ref_before.get(k, 0) > 0}
        return {
            "root": self.root,
            "rows": c["rows_attempted"] - before["rows"],
            "builds": c["builds"] - before["builds"],
            "faultins": c["faultins"] - before["faultins"],
            "resident_hits": c["resident_hits"] - before["resident_hits"],
            "singleflight_lost":
                c["singleflight_lost"] - before["singleflight_lost"],
            "replays": c["replays"] - before["replays"],
            "folds_done": c["folds_done"] - before["folds_done"],
            "refusals": refusals,
            "build_s": c["build_s"] - before["build_s"],
            "faultin_s": c["faultin_s"] - before["faultin_s"],
            "replay_s": c["replay_s"] - before["replay_s"],
            "trace_id": self.last_trace_id,
            "result_key": self.last_result_key,
        }

    def _refuse(self, reason: str) -> None:
        self.counters["refusals"][reason] = \
            self.counters["refusals"].get(reason, 0) + 1

    # -- the route -----------------------------------------------------------

    def o_access(self, view, engine, settings,
                 d_weights) -> Optional[Tuple[Any, Any, Any, Any]]:
        """The cohort route for one ``o_access`` call, or None to run the
        inner engine unchanged.  NEVER raises."""
        self.counters["rows_attempted"] += 1
        try:
            return self._o_access(view, engine, settings, d_weights)
        except Exception as exc:                     # sparse gate: any
            self._refuse(f"exception:{type(exc).__name__}")
            self.last_trace_id = None
            self.last_result_key = None
            return None

    def _o_access(self, view, engine, settings, d_weights):
        cutoff = settings.search_radius
        # The producer's admission gate demands exact C-contiguous
        # ndarray layouts; the engine's own compact view legitimately
        # holds non-contiguous terminal slices on real networks (the
        # smoke grid's (2,2) origin terminals come out non-contiguous).
        # ``ascontiguousarray`` is value-preserving — identical bytes in
        # identical logical order — so the trace identity, the built
        # record and the replay fold see exactly the engine's values,
        # and it is a no-op (zero copy) for every already-contiguous
        # array.  The FALLBACK path keeps the engine's own originals.
        o_terminal_idxs = np.ascontiguousarray(view.o_terminal_idxs)
        o_terminal_weights = np.ascontiguousarray(view.o_terminal_weights)
        adjacency_pointer = np.ascontiguousarray(view.adjacency_pointer)
        adjacency_vector = np.ascontiguousarray(view.adjacency_vector)
        adjacency_vector_weights = np.ascontiguousarray(
            view.adjacency_vector_weights)
        network_node = np.ascontiguousarray(
            view.adjacynct_vector_network_node)
        d_terminal_idxs = np.ascontiguousarray(view.d_terminal_idxs)
        d_terminal_weights = np.ascontiguousarray(view.d_terminal_weights)
        d_weights = np.ascontiguousarray(d_weights)
        trace_id = trm.compute_trace_id(
            o_terminal_idxs, o_terminal_weights,
            adjacency_pointer, adjacency_vector,
            adjacency_vector_weights, network_node,
            d_terminal_idxs, d_terminal_weights,
            cutoff, trm.DEFAULT_EPOCH_ROLLOVER,
            profile_id="una_legacy-epoch-a3")
        self.last_trace_id = trace_id

        # The result identity closes over exactly what the replay fold
        # consumes (metric params + this row's destination weights),
        # canonicalized exactly like the engine's own o_access result
        # stage.  The compact-view trace_id is the engine identity: it
        # hashes the post-elevation/obstacle arrays themselves, so it is
        # complete WITHOUT the stage-cache keys (which only exist when
        # the topology stage cache is enabled — mode 'off' is the
        # default, and the route must engage on default runs).
        try:
            result_key = _ckid.stage_key("una:cohort:o_access:v1", {
                "trace_id": trace_id,
                "gravity_beta": float(settings.gravity_beta),
                "gravity_plateau": float(settings.gravity_plateau),
                "gravity_logistic_midpoint":
                    float(settings.gravity_logistic_midpoint),
                "gravity_decay_constant":
                    float(settings.gravity_decay_constant),
                "knn_decay": str(settings.knn_decay),
                "knn_weights": np.asarray(settings.knn_weights,
                                          dtype=np.float64),
                "search_radius": float(cutoff),
                "d_weights": np.asarray(d_weights, dtype=np.float64),
            })
        except (TypeError, ValueError):
            self._refuse("keys_not_canonicalizable")
            return None
        self.last_result_key = result_key

        pair = _stores_for(self.root, self.residency_cap_bytes)
        residency = pair.residency
        key = residency.key_for(trace_id, _SCHEMA_VERSION,
                                _producer_sha256())

        rec = None
        # Produce-once (C20-NOTE-2): the build->publish path runs inside
        # the store's per-key single flight; losers wait for the owner's
        # commit and fault the packed bytes in instead of rebuilding.
        with pair.store.singleflight(key, wait_timeout_s=600.0) as owned:
            if not owned:
                self.counters["singleflight_lost"] += 1
            lease = None
            try:
                was_resident = key in residency._entries
                lease = residency.acquire(key)
                if lease is not None:
                    t1 = time.perf_counter()
                    rebuilt = self._rebuild(residency, key, lease)
                    lease.release()
                    lease = None
                    self.counters["faultin_s"] += \
                        time.perf_counter() - t1
                    if rebuilt is None:
                        self._refuse("resident_load_refused")
                        return None
                    if was_resident:
                        self.counters["resident_hits"] += 1
                    else:
                        self.counters["faultins"] += 1
                    rec = rebuilt
                if rec is None:
                    # Flight owner (or a loser whose owner failed to
                    # commit): build through the engine's own spliced
                    # producer kernel.
                    t1 = time.perf_counter()
                    built = trm.build_trace(
                        o_terminal_idxs, o_terminal_weights,
                        adjacency_pointer, adjacency_vector,
                        adjacency_vector_weights, network_node,
                        d_terminal_idxs, d_terminal_weights,
                        d_weights, cutoff)
                    self.counters["build_s"] += \
                        time.perf_counter() - t1
                    if not built.complete:
                        self._refuse(
                            f"trace_build:{built.route}:{built.refusal}")
                        return None
                    residency.publish(
                        np.asarray(built.indptr),
                        np.asarray(built.destination_ids),
                        np.asarray(built.distance_bits),
                        record_meta(built),
                        bound_digest=built.content_digest)
                    self.counters["builds"] += 1
                    rec = built
            finally:
                if lease is not None:
                    lease.release()

        # Replay through the engine's own reviewed fold with o_access's
        # EXACT argument threading (raw knn_weights, same division).
        t2 = time.perf_counter()
        res = trm.replay_trace(
            rec, d_weights,
            settings.gravity_beta,
            settings.gravity_plateau,
            settings.gravity_logistic_midpoint,
            settings.gravity_decay_constant /
            settings.gravity_logistic_midpoint,
            settings.knn_decay,
            settings.knn_weights,
            cutoff)
        self.counters["replay_s"] += time.perf_counter() - t2
        if res.route != "trace_replay":
            self._refuse(f"replay:{res.refusal}")
            return None
        self.counters["replays"] += 1
        self.counters["folds_done"] += int(res.folds_done)
        return (res.reach, res.gravity_exponential,
                res.gravity_logistic, res.knn_access)

    @staticmethod
    def _rebuild(residency, key, lease) -> Optional[trm.TraceRecord]:
        """A live TraceRecord over a faulted-in lease (identity fields
        from the residency's verified meta)."""
        meta = None
        try:
            meta = dict(residency._entries[key]["meta"])
        except Exception:
            return None
        ip, ids, dist = lease.arrays()
        try:
            # The recorded cutoff is TYPED (C11: the identity carries the
            # cutoff's own dtype — an int ``search_radius`` publishes
            # cutoff_dtype "int64"): ``record_meta`` stores the record's
            # float64 ``cutoff_value`` bits, and the declared dtype passes
            # through verbatim so the rebuilt record carries EXACTLY the
            # fields the built record carried — the same record the
            # builder's own replay already accepted.
            if not isinstance(meta.get("cutoff_bits"), str):
                return None
            cutoff_value = float(np.frombuffer(
                bytes.fromhex(meta["cutoff_bits"]),
                dtype=np.float64)[0])
            counters = trm.TraceCounters(
                origins=int(meta["origin_count"]), tiles=0,
                tile_origins=0, candidates_total=0, touched_total=0,
                indptr_bytes=0, ids_bytes=0, dist_bytes=0,
                metadata_bytes=0, trace_bytes=0, tile_temp_bytes=0,
                stamps_zeroed=0)
            return trm.TraceRecord(
                trace_id=str(meta["trace_id"]),
                schema_version=str(meta["trace_schema_version"]),
                origin_count=int(meta["origin_count"]),
                dest_count=int(meta["dest_count"]),
                cutoff_value=cutoff_value,
                cutoff_dtype=str(meta["cutoff_dtype"]),
                reach_dtype=str(meta["reach_dtype"]),
                metric_dtype=str(meta["metric_dtype"]),
                indptr=ip, destination_ids=ids, distance_bits=dist,
                complete=True, route="cohort_residency", refusal="",
                counters=counters,
                content_digest=str(meta["content_digest"]))
        except Exception:
            return None


class CohortGraphEngine:
    """Duck-type shim injected at ``accessibility.graph_engine``.

    ``o_scope`` delegates unchanged; ``o_access`` offers the cohort
    route first and falls back to the inner view for every refusal.
    All other attribute access (the compact arrays the producer kernel
    reads) goes straight through to the inner view.
    """

    def __init__(self, inner, engine, session: CohortSession):
        self._inner = inner
        self._engine = engine
        self._session = session

    def o_scope(self, o_idx, search_radius):
        return self._inner.o_scope(o_idx, search_radius)

    def o_access(self, o_idx, settings, d_weights):
        served = self._session.o_access(self._inner, self._engine,
                                        settings, d_weights)
        if served is not None:
            return served
        return self._inner.o_access(o_idx, settings, d_weights)

    def __getattr__(self, name):
        return getattr(self._inner, name)
