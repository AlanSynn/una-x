"""Analysis-stage cache facade over the ContentStore primitive (CACHE_GRAPH).

Bridges the analysis stack (Topology / Engines) to the CACHE_STORE storage
primitive:

- :class:`StageCache` stores stage payloads as named ``.npy`` arrays
  (``numpy.lib.format`` with ``allow_pickle=False`` — non-executable by
  construction) plus a JSON metadata dict for scalar provenance (counts,
  replayed log details, CRS WKT, coordinate dimension).
- ``mode='off'`` (the default ``CacheOptions``) yields a DISABLED facade:
  every get misses, every put is a no-op, single-flight trivially yields the
  owner — callers collapse to their original compute path with zero key
  computation, zero snapshot reads and no directory created.
- ``stage_cache_for(options)`` memoizes one ContentStore per distinct
  CacheOptions tuple so serial batch rows, threads and repeated calls share
  counters and single-flight (compute-once per process).

Aliasing discipline (dossier 06): every array returned by
:meth:`StageCache.get_arrays` is freshly materialized from bytes; mutating
it cannot touch the store, and stored bytes are written out at put time (no
retained references).  Arrays with dtypes ``.npy`` cannot represent without
pickle (object/record/void) are refused at put — the stage simply does not
cache (a future miss recomputes; never a wrong hit).

Only numpy + this package's store/Execution are imported at module level;
the analysis stack stays importable without this module.
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from ..Execution import CacheOptions
from .store import ContentStore, StoreHit

__all__ = ["StageCache", "StageHit", "StageProduced", "stage_cache_for"]


def encode_arrays(arrays: Mapping[str, Optional[np.ndarray]]) -> Dict[str, bytes]:
    """Named arrays → named ``.npy`` byte blobs (``allow_pickle=False``).

    ``None`` entries are skipped (absence is recorded in metadata/keys by
    the caller).  Raises TypeError for dtypes ``.npy`` can only carry via
    pickle — the caller treats that as "not cacheable", not as an error.
    """
    payload: Dict[str, bytes] = {}
    for name, arr in arrays.items():
        if arr is None:
            continue
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"cache payload '{name}' must be a numpy array, got "
                f"{type(arr).__name__}")
        if arr.dtype.hasobject or arr.dtype.kind in ("V", "O"):
            raise TypeError(
                f"cache payload '{name}' has non-representable dtype "
                f"{arr.dtype} (non-executable store: no pickle)")
        buf = io.BytesIO()
        np.lib.format.write_array(buf, np.ascontiguousarray(arr),
                                  allow_pickle=False)
        payload[name] = buf.getvalue()
    return payload


def decode_arrays(payload: Mapping[str, bytes]) -> Dict[str, np.ndarray]:
    """Named ``.npy`` blobs → fresh arrays (``allow_pickle=False``)."""
    arrays: Dict[str, np.ndarray] = {}
    for name, blob in payload.items():
        arrays[name] = np.lib.format.read_array(io.BytesIO(blob),
                                                allow_pickle=False)
    return arrays


@dataclass(frozen=True)
class StageHit:
    """One stage payload read: fresh arrays + the producer's metadata."""

    key: str
    generation: int
    arrays: Dict[str, np.ndarray]
    metadata: Mapping[str, object]
    source: str  # "memory" | "disk"


@dataclass(frozen=True)
class StageProduced:
    """Marker returned by ``arrays_once`` when the arrays were computed in
    THIS call (single-flight owner, or a bounded fallback) instead of being
    served from the store.  The arrays are valid either way; callers use the
    marker only to keep "restored from cache" log lines truthful and to skip
    a redundant recompute (review MAJOR-1)."""

    key: str
    generation: None = None
    source: str = "produced"


class StageCache:
    """Facade the analysis stack talks to.  ``enabled`` is False iff the
    configured mode is 'off' — then every method degenerates to the
    uncached behavior (miss / no-op / trivial owner)."""

    def __init__(self, store: Optional[ContentStore]):
        self._store = store

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def get_arrays(self, key: str) -> Optional[StageHit]:
        if self._store is None:
            return None
        hit: Optional[StoreHit] = self._store.get(key)
        if hit is None:
            return None
        return StageHit(key=key, generation=hit.generation,
                        arrays=decode_arrays(hit.payload),
                        metadata=dict(hit.metadata), source=hit.hit)

    def put_arrays(self, key: str, arrays: Mapping[str, Optional[np.ndarray]],
                   metadata: Optional[Mapping[str, object]] = None) -> bool:
        """Commit one stage generation.  Returns False when disabled OR when
        the payload is not representable (nothing stored — a future run
        recomputes; never a silent wrong hit)."""
        if self._store is None:
            return False
        try:
            payload = encode_arrays(arrays)
        except TypeError:
            return False
        if not payload:
            return False
        self._store.put(key, payload, metadata=dict(metadata or {}))
        return True

    def arrays_once(self, key: str,
                    produce: Callable[[], Tuple[Dict[str, Optional[np.ndarray]],
                                                Optional[Mapping[str, object]]]]
                    ) -> Tuple[Dict[str, np.ndarray], Optional[Mapping[str, object]],
                               Union[StageHit, StageProduced]]:
        """``(arrays, metadata, hit)`` for ``key`` with compute-once semantics.

        A cache hit returns the stored arrays with ``hit`` a :StageHit:.
        On a miss the single-flight owner runs ``produce()`` exactly once and
        commits; losers block on the store's per-key lock and re-get the
        committed generation.  The returned ``hit`` is a :StageProduced:
        marker when the arrays were computed in THIS call — served or
        produced, the arrays are valid and the caller must not recompute
        (review MAJOR-1: a cold cached run is the owner's compute, not an
        uncached fallback).  If owners cannot commit (payload not
        representable), a bounded loser falls back to its own compute the
        same way — the stage simply stays uncached.

        A per-key wait timeout (store contract: a holder slower than
        ``wait_timeout_s``) also falls back to a local compute rather than
        crashing the caller; a second writer can only publish an identical
        generation (the key closure pins every numerical input), never
        wrong bits.
        """
        if not self.enabled:
            arrays, metadata = produce()
            return arrays, metadata, StageProduced(key=key)
        spins = 0
        while True:
            hit = self.get_arrays(key)
            if hit is not None:
                return hit.arrays, hit.metadata, hit
            try:
                with self._store.singleflight(key) as owned:
                    if owned:
                        arrays, metadata = produce()
                        self.put_arrays(key, arrays, metadata)
                        return arrays, metadata, StageProduced(key=key)
                # Loser: the owner finished — its commit should now be visible.
                spins += 1
                if spins >= 3:
                    # Owner(s) could not commit an unrepresentable payload;
                    # fall back to uncached compute instead of spinning.
                    arrays, metadata = produce()
                    return arrays, metadata, StageProduced(key=key)
            except TimeoutError:
                arrays, metadata = produce()
                return arrays, metadata, StageProduced(key=key)

    @property
    def store(self) -> Optional[ContentStore]:
        """The underlying store (single-flight, stats, inspect, clear)."""
        return self._store

    def stats(self) -> dict:
        if self._store is None:
            return {"mode": "off", "hits": 0, "misses": 0,
                    "hit_reasons": {"memory": 0, "disk": 0, "miss": 0},
                    "entries": 0}
        return self._store.stats()


# --------------------------------------------------------------- registry

_REGISTRY: Dict[tuple, StageCache] = {}
_REGISTRY_LOCK = threading.Lock()


_DISABLED_CACHE = StageCache(None)


def stage_cache_for(options: Optional[CacheOptions]) -> StageCache:
    """One shared StageCache per distinct CacheOptions tuple (per process).

    Sharing matters: single-flight (identical requests compute once) and the
    hit/miss counters are per-ContentStore.  ``mode='off'`` — or a ``None``
    options object (a Topology constructed without ever going through an
    Add*, hence no applied cache policy; review MINOR-4) — returns a shared
    disabled facade.
    """
    if options is None:
        return _DISABLED_CACHE
    key = (options.mode, options.directory, options.max_memory_bytes,
           options.max_disk_bytes, options.verification,
           options.schema_version)
    with _REGISTRY_LOCK:
        cache = _REGISTRY.get(key)
        if cache is None:
            store = None if options.mode == "off" else ContentStore(options)
            cache = StageCache(store)
            _REGISTRY[key] = cache
        return cache
