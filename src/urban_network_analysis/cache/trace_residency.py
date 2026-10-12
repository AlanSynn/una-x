"""Bounded immutable resident trace leases (campaign una_10x, C20).

Dossier 06 laws implemented here:

- The resident layer stores IMMUTABLE DECODED ARRAYS ONLY, under
  bounded byte accounting: its own allowance bounds its decoded
  arenas; the encoded cache tier is bounded by the STORE's own
  options; and ``charged_bytes()`` reports the two TOGETHER — no
  double claim of the same trace's bytes and no hidden second quota
  (dossier 06 "bounded byte accounting"; RESOURCES.md "cache memory +
  decoded arenas cannot each claim the same quota").  The deployment
  sizes the residency allowance to include the share the encoded
  tier occupies (resident_memory.json records the accounting).
- Public access is always FRESH: leases hand out new writable copies;
  resident arrays are read-only and never exposed, so callers cannot
  mutate shared state (the "caller edits an exposed array"
  invalidation hole cannot exist).
- Entries pinned by live leases are NEVER evicted; unpinned entries
  evict least-recently-used.  Dead owners are reaped by weakref
  finalizers (explicit ownership: every lease releases exactly once).
- Never execute pickle from disk cache: payloads are plain named byte
  segments decoded with ``np.frombuffer``.
- Load verifies identity, not presence: residency schema, producer
  module digest, trace id, record schema version, dtype plan and the
  packed content digest (the C11-NOTE-4 digest bound at build) — a
  forged, partial, stale-code or bit-flipped entry raises
  :class:`ResidencyError`; it is never a silent hit.
- Cache disabling keeps the original route: a disabled store cannot
  back a residency (no identity or serialization work happens).

This module stays inside the cache package boundary: it imports
nothing from the analysis stack and never rebuilds engine objects;
record reconstruction is the caller's contract (C30).

Deferred to the C30 cohort owner (C20-NOTE-2): produce-once for a
missing trace is the BUILD->publish path's single-flight duty —
two processes faulting in the same missing trace both decode+verify
here (store-level correctness is safe; the duplicate work is
bounded and benign).  C30 wraps the build->publish in
``store.singleflight``.
"""
import hashlib
import json
import weakref
from typing import Dict, Optional, Tuple

import numpy as np

from .store import ContentStore

RESIDENT_SCHEMA = "una-10x-c20-resident-1"

_IDENTITY_FIELDS = ("trace_id", "trace_schema_version", "dest_count",
                    "cutoff_bits", "cutoff_dtype", "reach_dtype",
                    "metric_dtype")


class ResidencyError(RuntimeError):
    """A resident or backing entry failed identity/digest checks."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packed_digest(indptr_bytes: bytes, ids_bytes: bytes,
                  dist_bytes: bytes) -> str:
    """sha256 over the packed bytes in layout order (the C11-NOTE-4
    digest law; build binds the same function of the same order)."""
    return _sha(indptr_bytes + ids_bytes + dist_bytes)


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _slices_ascending(ip: np.ndarray, di: np.ndarray) -> bool:
    """True when destination ids ascend WITHIN every indptr slice
    (the C10/C11 packed-order law; empty slices are vacuously fine)."""
    for i in range(int(ip.shape[0]) - 1):
        a, b = int(ip[i]), int(ip[i + 1])
        if b > a and di.shape[0] \
                and not np.all(np.diff(di[a:b]) > 0):
            return False
    return True


def _is_count(v) -> bool:
    """A true non-negative integer (bool excluded): json round-trips
    the write path's int() outputs exactly."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


class ResidentLease:
    """A pinned handle onto one resident trace.

    ``arrays()`` returns FRESH writable copies of (indptr,
    destination_ids, distance_bits) — the resident arrays themselves
    are never aliased into a caller.  Release exactly once:
    explicitly, or automatically at collection via the weakref
    finalizer (owner death / reap).
    """

    def __init__(self, residency: "TraceResidency", key: str):
        self.key = key
        self.released = False
        self._residency = residency
        self._finalizer = weakref.finalize(
            self, TraceResidency._reap, residency, key)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.released:
            raise ResidencyError("lease already released")
        return self._residency._fresh(self.key)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self._finalizer()  # invokes _reap exactly once
        self._residency = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class _StreamLease(ResidentLease):
    """An oversized entry handed back from the store WITHOUT
    residency: fresh writable copies of the verified decode."""

    def __init__(self, residency, key, ip, di, db):
        super().__init__(residency, key)
        self._copies = (np.array(ip), np.array(di), np.array(db))

    def arrays(self):
        if self.released:
            raise ResidencyError("lease already released")
        return tuple(np.array(a) for a in self._copies)


class TraceResidency:
    """LRU residency of immutable decoded trace arrays over one
    ContentStore.  ``max_resident_bytes`` bounds THIS RESIDENCY's
    decoded arenas only (C20-MAJOR-1 repair: the earlier draft
    claimed a combined bound that the code never enforced).  The
    encoded tier is bounded by the STORE's own options and is
    reported together with the arenas by ``charged_bytes()`` — a
    report, not an enforcement.  A deployment that needs a combined
    cap must size the store's ``max_memory_bytes`` plus this
    allowance explicitly against the host budget (C30 owner task:
    the cohort route derives its share from RESOURCES.md, never from
    the wording here).

    ``producer_sha256`` is the digest of the producer module's bytes:
    entries recorded under a different module digest refuse to load
    (schema/profile/code mismatch cannot alias into a hit).
    """

    def __init__(self, store: ContentStore, max_resident_bytes: int,
                 producer_sha256: str):
        if not isinstance(store, ContentStore) or store.disabled:
            raise ValueError(
                "TraceResidency requires an enabled ContentStore "
                "(cache disabled keeps the original route: no "
                "identity or serialization work)")
        if not isinstance(max_resident_bytes, int) or \
                max_resident_bytes <= 0:
            raise ValueError("max_resident_bytes must be a positive int")
        if not isinstance(producer_sha256, str) or \
                len(producer_sha256) != 64:
            raise ValueError("producer_sha256 must be a sha256 hexdigest")
        self.store = store
        self.max_resident_bytes = int(max_resident_bytes)
        self.producer_sha256 = producer_sha256
        self._entries: Dict[str, dict] = {}
        self._resident_bytes = 0
        self._lru_clock = 0
        # documented test hook (corruption tests); never set in
        # production code
        self.fault: Optional[str] = None
        self.counters = {
            "publishes": 0, "publish_rejections": 0,
            "resident_hits": 0, "fault_ins": 0, "store_misses": 0,
            "evictions": 0, "capacity_writes": 0,
            "digest_verifications": 0, "decode_bytes": 0,
            "leases_ended": 0,
        }

    # ---- identity -----------------------------------------------------
    def key_for(self, trace_id: str, trace_schema_version,
                producer_sha256: str) -> str:
        """Content address over the FULL identity closure: the C11
        typed trace id, the record schema version (its TRUE value —
        the engine's ``TraceRecord.schema_version`` is the STRING
        ``una-10x-c11-trace-1``, never coerced: C20-MAJOR-2 repair)
        and the producer module digest."""
        frame = json.dumps(
            ["una-10x-c20-resident-key-1", RESIDENT_SCHEMA,
             str(trace_id), str(trace_schema_version),
             str(producer_sha256)],
            sort_keys=True).encode("utf-8")
        return _sha(frame)

    # ---- publish ------------------------------------------------------
    def publish(self, indptr: np.ndarray, ids: np.ndarray,
                dist: np.ndarray, meta: dict,
                bound_digest: str = "") -> str:
        """Bind, store and make resident one COMPLETE trace's packed
        arrays.  ``meta`` must carry the identity fields; the record's
        bound content digest is verified against the packed bytes
        here (a disagreeing record is rejected, never published).

        Encoded payload goes through the ContentStore (checksummed,
        single-flight, crash-safe); decoded arrays enter the bounded
        LRU.  Returns the content key.  A residency too small for the
        entry degrades to a store-only publish (fault-in on acquire
        streams the verified payload back without residency).
        """
        ip = np.ascontiguousarray(indptr, dtype=np.int64)
        di = np.ascontiguousarray(ids, dtype=np.int64)
        db = np.ascontiguousarray(dist, dtype=np.float64)
        # the packed layout law (dossier 04): indptr[0] == 0,
        # NON-DECREASING slice bounds, ids ASCENDING WITHIN every
        # slice, nnz agreement — a non-conforming triple is rejected
        # here, never stored to fail later at decode (C20-MINOR-1
        # repair: the monotone/ascending clauses are enforced, not
        # just claimed)
        if ip.shape[0] < 1 or int(ip[0]) != 0 \
                or int(ip[-1]) != int(di.shape[0]) \
                or db.shape[0] != di.shape[0] \
                or (di.shape[0] and int(ip.min()) < 0):
            self.counters["publish_rejections"] += 1
            raise ResidencyError("packed layout violation at publish")
        if ip.shape[0] >= 2 and not np.all(np.diff(ip) >= 0):
            self.counters["publish_rejections"] += 1
            raise ResidencyError(
                "packed layout violation at publish: indptr is not "
                "non-decreasing")
        if not _slices_ascending(ip, di):
            self.counters["publish_rejections"] += 1
            raise ResidencyError(
                "packed layout violation at publish: destination ids "
                "are not ascending within every slice")
        missing = [k for k in _IDENTITY_FIELDS if k not in meta]
        if missing:
            raise ResidencyError(
                "publish meta incomplete: " + ",".join(missing))
        digest = packed_digest(ip.tobytes(), di.tobytes(), db.tobytes())
        key = self.key_for(str(meta["trace_id"]),
                           meta["trace_schema_version"],
                           self.producer_sha256)
        existing = self._entries.get(key)
        if existing is not None \
                and existing["meta"]["content_digest"] != digest:
            # C20-MINOR-3 repair: a content address IS the identity —
            # republishing DIFFERENT packed bytes under a live
            # resident key diverged the two tiers silently (the store
            # served the new generation while acquire() served the
            # old resident arrays).  Refuse; never replace a resident
            # entry in place.
            self.counters["publish_rejections"] += 1
            raise ResidencyError(
                "republish under an existing key with different "
                "packed bytes refuses (identity violation)")
        prior = self.store.get(key)
        if prior is not None:
            p = prior.payload
            prior_digest = packed_digest(p["indptr.bin"],
                                         p["destination_ids.bin"],
                                         p["distance_bits.bin"])
            if prior_digest != digest:
                # the same law against the encoded tier: an existing
                # content address with DIFFERENT stored bytes is an
                # identity violation even with no resident copy
                self.counters["publish_rejections"] += 1
                raise ResidencyError(
                    "republish under an existing key with different "
                    "packed bytes refuses (identity violation)")
        if bound_digest and bound_digest != digest:
            self.counters["publish_rejections"] += 1
            raise ResidencyError(
                "record content digest does not match packed bytes")
        full_meta = dict(meta)
        full_meta.update({
            "residency_schema": RESIDENT_SCHEMA,
            "producer_sha256": self.producer_sha256,
            "content_digest": digest,
            "origin_count": int(ip.shape[0] - 1),
            "nnz": int(di.shape[0]),
        })
        payload = {
            "indptr.bin": ip.tobytes(),
            "destination_ids.bin": di.tobytes(),
            "distance_bits.bin": db.tobytes(),
            "meta.json": json.dumps(
                full_meta, sort_keys=True,
                separators=(",", ":")).encode("utf-8"),
        }
        self.store.put(key, payload)
        if not self._admit(key, ip, di, db, full_meta):
            self.counters["capacity_writes"] += 1
        self.counters["publishes"] += 1
        return key

    # ---- acquire ------------------------------------------------------
    def acquire(self, key: str) -> Optional[ResidentLease]:
        """Pin one resident entry; a residency miss faults in through
        the store (verified decode) or returns None (the caller's
        recompute route)."""
        entry = self._entries.get(key)
        if entry is None:
            hit = self.store.get(key)
            if hit is None:
                self.counters["store_misses"] += 1
                return None
            ip, di, db, meta = self._decode(key, hit.payload)
            if not self._admit(key, ip, di, db, meta):
                self.counters["capacity_writes"] += 1
                return _StreamLease(self, key, ip, di, db)
            self.counters["fault_ins"] += 1
        else:
            self.counters["resident_hits"] += 1
        entry = self._entries[key]
        entry["pinned"] += 1
        self._lru_clock += 1
        entry["lru"] = self._lru_clock
        return ResidentLease(self, key)

    def verify_resident(self, key: str) -> None:
        """Recompute the resident arrays' digest against the bound
        meta (fault-injection point for the corruption tests)."""
        entry = self._entries.get(key)
        if entry is None:
            raise ResidencyError("key is not resident")
        ip, di, db = entry["arrays"]
        self.counters["digest_verifications"] += 1
        digest = packed_digest(ip.tobytes(), di.tobytes(), db.tobytes())
        if self.fault == "verify":
            digest += "00"  # simulate a failing verification
        if digest != entry["meta"]["content_digest"]:
            raise ResidencyError("resident arrays fail content digest")

    def resident_bytes(self) -> int:
        return self._resident_bytes

    def charged_bytes(self) -> int:
        """The shared-quota charge: decoded arenas + the encoded
        store's own memory tier, TOGETHER."""
        st = self.store.stats()
        return self._resident_bytes + int(st.get("memory_bytes", 0))

    def close(self) -> int:
        """Drop unpinned entries (pinned ones drop at reaping).
        Returns the number of entries dropped."""
        drop = [k for k, e in self._entries.items() if not e["pinned"]]
        for k in drop:
            self._drop(k)
        return len(drop)

    # ---- internals ----------------------------------------------------
    def _fresh(self, key: str):
        entry = self._entries.get(key)
        if entry is None:
            raise ResidencyError("key is not resident")
        ip, di, db = entry["arrays"]
        return (np.array(ip), np.array(di), np.array(db))

    def _admit(self, key, ip, di, db, meta) -> bool:
        if key in self._entries:
            return True
        nbytes = int(ip.nbytes + di.nbytes + db.nbytes)
        self._evict_under_budget(nbytes)
        # the allowance bounds the RESIDENCY's decoded arenas; the
        # encoded tier is the store's own contract (charged_bytes
        # reports both together)
        if self._resident_bytes + nbytes > self.max_resident_bytes:
            return False
        self._entries[key] = {
            "arrays": (_readonly(np.array(ip)), _readonly(np.array(di)),
                       _readonly(np.array(db))),
            "meta": meta, "bytes": nbytes, "pinned": 0, "lru": 0,
        }
        self._resident_bytes += nbytes
        self._lru_clock += 1
        self._entries[key]["lru"] = self._lru_clock
        return True

    def _decode(self, key: str, payload):
        """Plain-segment decode (never pickle), then full identity +
        digest verification against the bound meta.  EVERY malformed
        meta raises :class:`ResidencyError` (C20-MINOR-2 repair: a
        caller following the documented discipline — catch
        ResidencyError, fall back to recompute — must never see a
        foreign KeyError/ValueError), and the entry's identity is
        value-checked against the requested key, not merely
        presence-checked (C20-MINOR-1 repair)."""
        try:
            meta = json.loads(payload["meta.json"].decode("utf-8"))
            ip = np.frombuffer(payload["indptr.bin"], dtype=np.int64)
            di = np.frombuffer(payload["destination_ids.bin"],
                               dtype=np.int64)
            db = np.frombuffer(payload["distance_bits.bin"],
                               dtype=np.float64)
            if meta.get("residency_schema") != RESIDENT_SCHEMA:
                raise ResidencyError("residency schema mismatch")
            missing = [k for k in _IDENTITY_FIELDS if k not in meta]
            if missing:
                raise ResidencyError(
                    "incomplete identity: " + ",".join(missing))
            if meta.get("producer_sha256") != self.producer_sha256:
                raise ResidencyError("producer code mismatch")
            # the accounting fields are validated with types INSIDE
            # this guard: a missing content_digest or a non-int count
            # is a residency refusal, never a KeyError/ValueError
            content_digest = meta["content_digest"]
            if not isinstance(content_digest, str) \
                    or len(content_digest) != 64:
                raise ResidencyError("bound content digest malformed")
            origin_count = meta["origin_count"]
            nnz = meta["nnz"]
            if not (_is_count(origin_count) and _is_count(nnz)):
                raise ResidencyError("count fields must be integers")
            if int(ip.shape[0]) != origin_count + 1 \
                    or int(di.shape[0]) != nnz \
                    or int(db.shape[0]) != nnz:
                raise ResidencyError("packed shape mismatch")
            # key<->meta binding: the requested content address must
            # be re-derivable from the entry's own identity — a meta
            # swapped under another trace's key refuses here
            derived = self.key_for(str(meta["trace_id"]),
                                   meta["trace_schema_version"],
                                   str(meta["producer_sha256"]))
            if derived != key:
                raise ResidencyError(
                    "key does not bind the entry's identity")
        except ResidencyError:
            raise
        except Exception as exc:
            raise ResidencyError("resident payload undecodable") from exc
        self.counters["decode_bytes"] += int(
            ip.nbytes + di.nbytes + db.nbytes)
        digest = packed_digest(ip.tobytes(), di.tobytes(), db.tobytes())
        if self.fault == "decode":
            digest += "00"  # simulate corruption seen at decode
        if digest != content_digest:
            raise ResidencyError("packed bytes fail content digest")
        self.counters["digest_verifications"] += 1
        # the packed layout law holds at LOAD too, AFTER the digest:
        # a digest-consistent structural forgery (recomputed digest
        # over non-monotone/unsorted bytes) refuses — never a silent
        # hit (C20-MINOR-1 repair).  Digest first so a plain bit-flip
        # keeps reporting the digest gate.
        if ip.shape[0] < 1 or int(ip[0]) != 0 \
                or (ip.shape[0] >= 2 and not np.all(np.diff(ip) >= 0)) \
                or (di.shape[0] and int(ip.min()) < 0) \
                or not _slices_ascending(ip, di):
            raise ResidencyError("packed layout violation at load")
        return np.array(ip), np.array(di), np.array(db), meta

    def _evict_under_budget(self, incoming: int) -> None:
        need = self._resident_bytes + incoming \
            - self.max_resident_bytes
        if need <= 0:
            return
        candidates = sorted(
            (e["lru"], k) for k, e in self._entries.items()
            if not e["pinned"])
        for _, k in candidates:
            if need <= 0:
                break
            need -= self._entries[k]["bytes"]
            self._drop(k)
            self.counters["evictions"] += 1

    def _drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._resident_bytes -= entry["bytes"]

    @staticmethod
    def _reap(residency: "TraceResidency", key: str) -> None:
        """Weakref-finalizer reap: owner death releases the pin.

        Reaping runs at arbitrary GC/interpreter-shutdown times, so it
        must never touch the filesystem: releasing the pin is all it
        does; the entry becomes evictable at the NEXT admission (the
        law is only that leased entries are never evicted — unpinned
        eviction timing is an admission-time decision)."""
        entry = residency._entries.get(key)
        if entry is not None and entry["pinned"] > 0:
            entry["pinned"] -= 1
            residency.counters["leases_ended"] += 1

    # counter law (C20-NOTE-3 repair): "leases_ended" counts EXPLICIT
    # releases AND owner-death reaps — both end a lease through this
    # same exactly-once finalizer; it is not owner-death evidence on
    # its own.  (The committed resident_memory.json from the C20
    # implementation run records the pre-repair key "reaped_leases"
    # with the same both-paths semantics; history is preserved.)
