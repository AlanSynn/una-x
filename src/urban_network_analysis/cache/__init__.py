"""Content-addressed cache primitives (campaign CACHE_STORE, dossier 06).

Only the safe store primitive lives here: bounded, non-executable,
checksum-verified, single-flight, crash-safe.  Numerical key closure and
engine/topology integration are later DAG tasks (CACHE_GRAPH) — nothing in
this package imports the analysis stack, and nothing here decides WHAT is
numerically identical; under the checksummed policies (the default
``on_read``, plus ``on_write``/``always``) whatever bytes were committed
under a key come back bitwise, or the entry is a miss.  (``never`` is the
caller's explicit no-checksum choice; sizes are still enforced.)
"""

from .store import (  # noqa: F401
    CacheCorruption,
    ContentStore,
    StoreHit,
)

__all__ = ["CacheCorruption", "ContentStore", "StoreHit"]
