"""Typed metric field store (P03M, dossier 04) — the run-scoped,
worker-local cache behind the admitted gravity_exponential split.

Laws this module keeps:

* **Immutable inside, fresh outside.  Stored arrays are read-only
  copies; every hit serves a NEW writable array with independent
  write ownership (the served copy can never poison the store, and
  no two rows ever share a mutable output).**
* **Bounded.**  The store is LRU-bounded by total stored bytes
  against the SAME per-worker cap law as the trace residency (one
  sizing rule, defined once in ``batch/cohort.py``).  An entry larger
  than the whole cap is never stored (counted ``nofit``) — the cap
  governs residency, never correctness: a nofit field simply forces
  the caller back to the whole-row replay.
* **Never a correctness surface.  The store holds only outputs of a
  successful whole-row replay of the SAME typed identity
  (``una:metric-id:v2``); any consumer falls back to the whole-row
  replay on a miss, and the caller never fabricates a field.**
* **Runtime-only transport.  ``__getstate__`` refuses — the store
  never rides a RowResult or a state bundle (the P02 detach law).**
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

__all__ = ["MetricFieldStore"]


class MetricFieldStore:
    """LRU byte-bounded, worker-local store of immutable metric
    field arrays keyed by typed ``una:metric-id:v2`` field ids."""

    def __init__(self, cap_bytes: int):
        self.cap_bytes = max(0, int(cap_bytes))
        self._entries: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self.counters: Dict[str, int] = {
            "hits": 0, "misses": 0, "produced": 0,
            "copied_bytes": 0, "evictions": 0, "nofit": 0,
        }

    def get(self, field_id: str) -> Optional[np.ndarray]:
        """A fresh writable copy of the stored field, or None."""
        arr = self._entries.get(field_id)
        if arr is None:
            self.counters["misses"] += 1
            return None
        self._entries.move_to_end(field_id)
        self.counters["hits"] += 1
        out = np.array(arr, dtype=arr.dtype, copy=True, order="C")
        self.counters["copied_bytes"] += int(out.nbytes)
        return out

    def put(self, field_id: str, arr: np.ndarray) -> bool:
        """Store a read-only copy of ``arr`` (LRU).  Returns True when
        resident.  An entry larger than the whole cap is not stored
        (``nofit``); the caller must never depend on a put."""
        payload = np.array(arr, dtype=arr.dtype, copy=True, order="C")
        payload.setflags(write=False)
        n = int(payload.nbytes)
        if n > self.cap_bytes:
            self.counters["nofit"] += 1
            return False
        old = self._entries.pop(field_id, None)
        if old is not None:
            self._bytes -= int(old.nbytes)
        while self._entries and self._bytes + n > self.cap_bytes:
            _k, ev = self._entries.popitem(last=False)
            self._bytes -= int(ev.nbytes)
            self.counters["evictions"] += 1
        self._entries[field_id] = payload
        self._bytes += n
        self.counters["produced"] += 1
        return True

    def __getstate__(self):
        raise TypeError(
            "MetricFieldStore is runtime-only session state and is "
            "never transported (P02 detach law)")

    def __len__(self):
        return len(self._entries)
