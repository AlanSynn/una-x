"""Bounded content-addressed store: checksums, single-flight, atomic commits.

Campaign task CACHE_STORE per dossier 06 ("Store protocol") and contract §6
``CacheOptions``.  This module is the STORAGE PRIMITIVE only:

- Keys are 64-char lowercase sha256 hex digests computed by CALLERS from the
  numerical closure (CACHE_GRAPH owns what goes into the closure).  The store
  never derives identity from paths, mtimes, or object ids — a path+mtime-only
  key is unrepresentable here by construction.
- Values are raw byte payloads + a JSON manifest.  Nothing executable is ever
  deserialized: no pickle anywhere, payloads are plain bytes, the manifest is
  ``json.loads`` of our own written schema.  Array wrappers (npy with
  ``allow_pickle=False`` etc.) belong to the layers above.
- Writes stage into ``tmp/``, fsync, then publish by atomic directory rename;
  a crash mid-write leaves no readable generation (readers see a miss, never
  partial data).  ``verification='on_write'|'always'`` reads the staged bytes
  back and compares them against the caller's bytes before publishing.
- Reads verify sizes (+ sha256 checksums per ``verification`` policy) BEFORE
  use; corruption is quarantined and reported as a miss — never silently
  accepted, never surfaced as partially-finite data.  The bitwise-or-miss
  guarantee holds under the checksummed policies (default ``on_read``, plus
  ``on_write``/``always``); ``verification='never'`` is the caller's
  explicit choice to skip checksums, and same-length damage then comes back
  as a hit (sizes are still enforced).
- Per-key single-flight locks with dead-holder recovery; cancellation releases
  the lock on context exit; a stale lock from a crashed holder is stolen.
- Bounded memory LRU (requires a positive ``max_memory_bytes`` — the memory
  tier is INERT without an explicit bound: no unbounded resident data); disk
  quota evicts least-recently-used generations.
- Modes (contract §6 ``mode``): ``off`` fully inert (no directory created);
  ``memory`` memory-only — no disk residue, and inert without a positive
  bound; ``disk`` the disk-backed store, with the memory tier as an optional
  accelerator layer.
- Everything returned is fresh bytes; the store never hands out live buffers
  of retained data (Python ``bytes`` are immutable; layers above that publish
  mutable arrays must copy on insert AND on publish — dossier 06).

Subprocess/stdlib only (plus ``urban_network_analysis.Execution.CacheOptions``):
importing this package must not pull the analysis stack.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

from ..Execution import CacheOptions

__all__ = ["CacheCorruption", "ContentStore", "StoreHit"]

MANIFEST_NAME = "manifest.json"
_HEX = set("0123456789abcdef")
_STALE_SWEEP_S = 6 * 3600  # tmp//quarantine debris older than this ages out


def _require_key(key: str) -> None:
    """Keys are content digests — this is also the path-traversal guard."""
    if (not isinstance(key, str) or len(key) != 64
            or any(c not in _HEX for c in key)):
        raise ValueError(
            f"cache key must be a 64-char lowercase sha256 hex digest "
            f"(content-addressed; path-derived or mtime-derived keys are "
            f"unrepresentable by design), got {key!r}")


def sha256_hex(data: bytes) -> str:
    """The one digest form used for keys and payload checksums."""
    return hashlib.sha256(data).hexdigest()


class CacheCorruption(RuntimeError):
    """Raised by :meth:`ContentStore.require` when an entry fails
    verification.  :meth:`ContentStore.get` never raises this — it treats
    corruption as a miss (quarantine + miss), per dossier 06."""


@dataclass(frozen=True)
class StoreHit:
    """One verified read.  ``payload`` maps name → fresh bytes (copies —
    mutating them cannot touch the store); ``hit`` is 'memory' or 'disk'."""

    key: str
    generation: int
    payload: Dict[str, bytes]
    metadata: Mapping[str, object]
    hit: str  # "memory" | "disk"


class _KeyLock:
    """Process-level (threading) guard complementing the on-disk lock file:
    threads inside one process serialize here before touching the file lock,
    so concurrent threads don't steal each other's O_EXCL locks."""

    def __init__(self):
        self._cv = threading.Condition()
        self._held: set = set()

    @contextmanager
    def hold(self, key: str):
        with self._cv:
            while key in self._held:
                self._cv.wait()
            self._held.add(key)
        try:
            yield
        finally:
            with self._cv:
                self._held.discard(key)
                self._cv.notify_all()


class ContentStore:
    """Disk + optional memory content store honoring contract CacheOptions.

    Public surface (dossier 06): configuration, get/put with single-flight,
    clear/inspect/stats, bytes, generation and hit reasons.  A disabled store
    (``mode='off'``) is inert: ``get`` misses, ``put`` is a no-op, compute
    always runs.
    """

    def __init__(self, options: CacheOptions):
        if not isinstance(options, CacheOptions):
            raise TypeError(
                f"options must be CacheOptions, got "
                f"{type(options).__name__}")
        if (not isinstance(options.schema_version, int)
                or isinstance(options.schema_version, bool)
                or options.schema_version < 1):
            raise ValueError(
                f"cache.schema_version must be a positive integer, got "
                f"{options.schema_version!r}")
        self.options = options
        self._hits = 0
        self._misses = 0
        self._quarantined = 0
        self._evicted_disk = 0
        self._evicted_memory = 0
        self._hit_reasons = {"memory": 0, "disk": 0, "miss": 0}
        self._ctr_lock = threading.Lock()  # diagnostic counters are shared
        # Memory tier policy: INERT unless a positive bound is configured —
        # an unbounded resident layer is exactly what dossier 06 forbids.
        self._memory_enabled = (
            options.mode in ("memory", "disk")
            and options.max_memory_bytes is not None
            and options.max_memory_bytes > 0)
        # mode semantics: 'memory' caches in memory ONLY (no disk residue —
        # so without a positive bound it is fully inert); 'disk' is the
        # disk-backed store with the memory tier as an optional accelerator.
        self._disk_enabled = options.mode == "disk"
        self._mem: "OrderedDict[str, Tuple[int, dict, dict, dict]]" = \
            OrderedDict()
        self._mem_bytes = 0
        self._mem_lock = threading.Lock()
        self._mem_gen: Dict[str, int] = {}
        self._key_lock = _KeyLock()
        if options.mode == "off":
            self._root = None
            return
        if not isinstance(options.directory, str) or not options.directory.strip():
            raise ValueError(
                f"cache directory must be a non-empty string, "
                f"got {options.directory!r}")
        self._root = options.directory
        if self._disk_enabled:
            for sub in ("store", "locks", "tmp", "quarantine"):
                os.makedirs(os.path.join(self._root, sub), exist_ok=True)

    # ------------------------------------------------------------------ dirs

    @property
    def disabled(self) -> bool:
        return self._root is None

    def _gen_dir(self, key: str) -> str:
        return os.path.join(self._root, "store", key[:2], key)

    def _staging_dir(self, key: str) -> str:
        return os.path.join(self._root, "tmp", f"{key}-{os.getpid()}-{next(_UNIQ)}")

    def _lock_path(self, key: str) -> str:
        return os.path.join(self._root, "locks", f"{key}.lock")

    # ------------------------------------------------------------------ put

    def put(self, key: str, payload: Mapping[str, bytes],
            metadata: Optional[Mapping[str, object]] = None,
            verification: Optional[str] = None) -> int:
        """Commit one generation atomically; returns the generation number.

        Staging → fsync files → fsync manifest → atomic dir rename → fsync
        parent.  A crash anywhere before the rename leaves tmp/ only — no
        readable generation, no hit.  Re-putting a key publishes a new
        generation number; the retire-republish window is not reader-atomic
        (a reader racing exactly inside it observes a miss, never partial
        data, and never quarantines — quarantine happens only when the
        damage is still current; first publication, the common case, is a
        pure rename).  A crash inside the retire window can lose the entry
        entirely; its tmp/ debris is swept once older than the stale-tmp
        grace, and the next put for the key restarts generation numbering.
        """
        _require_key(key)
        if self.disabled:
            return 0  # inert store: caller's compute stands, nothing stored
        verify = verification or self.options.verification
        if verify not in ("never", "on_write", "on_read", "always"):
            raise ValueError(f"unknown verification mode {verify!r}")
        if not payload:
            raise ValueError("payload must contain at least one named blob")

        if not self._disk_enabled:  # mode='memory': bounded resident only
            generation = self._next_mem_generation(key, payload)
            if generation == 0:
                return 0  # memory mode without a bound, or entry larger
                          # than the whole tier: never resident, stored
                          # nothing — 0 says so (review NOTE-2)
            self._mem_put(key, payload, {
                "generation": generation,
                "metadata": dict(metadata or {})})
            return generation

        self._sweep_stale()  # crashed writers' tmp/ debris ages out here
        staging = self._staging_dir(key)
        payloads_dir = os.path.join(staging, "payload")
        os.makedirs(payloads_dir)
        try:
            entries = []
            total = 0
            for name, blob in payload.items():
                if not name or "/" in name or name in (".", "..") or "\\" in name:
                    raise ValueError(f"invalid payload name {name!r}")
                if not isinstance(blob, (bytes, bytearray, memoryview)):
                    raise TypeError(
                        f"payload '{name}' must be raw bytes, got "
                        f"{type(blob).__name__} (non-executable store: no "
                        f"pickled objects)")
                blob = bytes(blob)
                digest = sha256_hex(blob)
                path = os.path.join(payloads_dir, name)
                with open(path, "wb") as f:
                    f.write(blob)
                    f.flush()
                    os.fsync(f.fileno())
                if verify in ("on_write", "always"):
                    # Read the staged bytes back and compare against the
                    # caller's bytes — a real check of the write path (a
                    # digest compared with itself would verify nothing).
                    with open(path, "rb") as f:
                        written = f.read()
                    if len(written) != len(blob) \
                            or sha256_hex(written) != digest:
                        raise CacheCorruption(
                            f"payload '{name}' failed write verification "
                            f"(staged bytes differ from the caller's)")
                entries.append({"name": name, "sha256": digest,
                                "bytes": len(blob)})
                total += len(blob)
            manifest = {
                "schema_version": self.options.schema_version,
                "key": key,
                "generation": self._next_generation(key),
                "payload": entries,
                "metadata": dict(metadata or {}),
            }
            mpath = os.path.join(staging, MANIFEST_NAME)
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(manifest, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            final = self._gen_dir(key)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            # Atomic publish: directory rename.  If a concurrent writer won,
            # our staged tree is discarded (its bytes are identical by
            # construction of content addressing).
            retired = None
            if os.path.isdir(final):
                retired = self._staging_dir(key)
                os.rename(final, retired)
            try:
                os.rename(staging, final)
            except OSError:
                if retired is not None and not os.path.isdir(final):
                    os.rename(retired, final)
                raise
            finally:
                if retired is not None:
                    shutil.rmtree(retired, ignore_errors=True)
            self._fsync_dir(os.path.dirname(final))
            generation = manifest["generation"]
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        self._mem_invalidate(key)
        self._enforce_disk_quota()
        self._mem_put(key, payload, manifest)
        return generation

    def _next_generation(self, key: str) -> int:
        current = self._read_manifest(key)
        if current is None:
            return 1
        try:
            return int(current.get("generation", 0)) + 1
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _fsync_dir(path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # best-effort durability on filesystems without dir fsync

    # ------------------------------------------------------------------ get

    def _record(self, reason: str) -> None:
        """One counter transition ('memory' | 'disk' | 'miss'), thread-safe."""
        with self._ctr_lock:
            if reason == "miss":
                self._misses += 1
            else:
                self._hits += 1
            self._hit_reasons[reason] += 1

    def get(self, key: str) -> Optional[StoreHit]:
        """Verified read: returns a StoreHit or None (miss — including any
        corrupted/incomplete entry, which is quarantined)."""
        _require_key(key)
        if self.disabled:
            self._record("miss")
            return None
        hit = self._mem_get(key)
        if hit is not None:
            self._record("memory")
            return hit
        if not self._disk_enabled:
            self._record("miss")
            return None
        hit = self._disk_get(key)
        if hit is None:
            self._record("miss")
            return None
        self._record("disk")
        self._touch(key)  # LRU: generation dir mtime drives disk eviction
        self._mem_put(key, hit.payload, {
            "generation": hit.generation, "metadata": dict(hit.metadata)})
        return hit

    def require(self, key: str) -> StoreHit:
        """Like :meth:`get`, but corruption RAISES instead of miss-ing —
        for callers that must distinguish 'absent' from 'damaged'.  Damaged
        entries are left in place (not quarantined) so evidence survives."""
        _require_key(key)
        if self.disabled:
            raise CacheCorruption(
                "require() on a disabled store (mode='off'): no entry can "
                "exist, let alone verify")
        hit = self._mem_get(key)
        if hit is not None:
            return hit
        if not self._disk_enabled:
            # memory-only mode: absent from the tier = absent, period
            raise KeyError(f"cache entry {key[:12]}… is not present")
        if self._read_manifest(key) is None \
                and not os.path.isdir(self._gen_dir(key)):
            raise KeyError(f"cache entry {key[:12]}… is not present")
        hit = self._disk_get(key, quarantine=False)
        if hit is None:
            raise CacheCorruption(
                f"cache entry {key[:12]}… is damaged (evidence retained "
                f"in place)")
        self._touch(key)
        return hit

    def _disk_get(self, key: str, quarantine: bool = True) -> Optional[StoreHit]:
        gen_dir = self._gen_dir(key)
        manifest = self._read_manifest(key)
        if manifest is None:
            # No manifest = never published (crash mid-write) or unreadable:
            # either way it is a miss, and a damaged dir is quarantined.
            if os.path.isdir(gen_dir):
                self._fail_read(key, None, gen_dir, quarantine)
            return None
        generation = manifest["generation"]  # validated by _read_manifest
        verify = self.options.verification
        payload = {}
        for entry in manifest["payload"]:
            path = os.path.join(gen_dir, "payload", entry["name"])
            try:
                with open(path, "rb") as f:
                    blob = f.read()
            except OSError:
                self._fail_read(key, generation, gen_dir, quarantine)
                return None
            if len(blob) != entry["bytes"]:
                self._fail_read(key, generation, gen_dir, quarantine)
                return None
            if verify in ("on_read", "always") and \
                    sha256_hex(blob) != entry["sha256"]:
                self._fail_read(key, generation, gen_dir, quarantine)
                return None
            payload[entry["name"]] = blob
        return StoreHit(key=key, generation=generation, payload=payload,
                        metadata=manifest["metadata"], hit="disk")

    def _read_manifest(self, key: str) -> Optional[dict]:
        """Read AND validate a manifest.  Returns None for anything that is
        not exactly our schema — every field the read path later trusts is
        checked here, so a structurally corrupt manifest is ordinary
        corruption (miss/quarantine), never a raised TypeError/KeyError,
        and manifest-declared payload names pass the same guard put-side
        names do (review MINOR-1)."""
        path = os.path.join(self._gen_dir(key), MANIFEST_NAME)
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(manifest, dict) or \
                manifest.get("schema_version") != self.options.schema_version \
                or manifest.get("key") != key:
            return None
        generation = manifest.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool) \
                or generation < 1:
            return None
        payload = manifest.get("payload")
        if not isinstance(payload, list) or not payload:
            return None
        for entry in payload:
            if not isinstance(entry, dict):
                return None
            name = entry.get("name")
            if not isinstance(name, str) or not name or "/" in name \
                    or "\\" in name or name in (".", ".."):
                return None
            nbytes = entry.get("bytes")
            if not isinstance(nbytes, int) or isinstance(nbytes, bool) \
                    or nbytes < 0:
                return None
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 \
                    or any(c not in _HEX for c in digest):
                return None
        if not isinstance(manifest.get("metadata"), dict):
            return None
        return manifest

    def _quarantine(self, key: str, gen_dir: Optional[str]) -> None:
        """Move a damaged generation aside (never silently accepted, and
        kept in quarantine/ for diagnosis — best effort: if even the
        quarantine rename fails, the unusable tree is removed rather than
        left to be re-read as recurring corruption).  In-flight readers are
        unaffected — they hold fresh copies."""
        self._mem_invalidate(key)
        if gen_dir is None or not os.path.isdir(gen_dir):
            return
        dest = os.path.join(
            self._root, "quarantine",
            f"{key[:16]}-{os.getpid()}-{next(_UNIQ)}")
        try:
            os.rename(gen_dir, dest)
            self._fsync_dir(os.path.join(self._root, "quarantine"))
        except OSError:
            shutil.rmtree(gen_dir, ignore_errors=True)  # unusable anyway
        with self._ctr_lock:
            self._quarantined += 1

    def _fail_read(self, key: str, expected_generation: Optional[int],
                   gen_dir: Optional[str], quarantine: bool) -> None:
        """A verified read failed.  Quarantine ONLY if the damage is still
        current: a writer that superseded the entry mid-read left our
        evidence stale, and moving the NEW generation aside would destroy
        good data (review MINOR-2).  ``expected_generation=None`` means the
        manifest itself was unreadable — stand down only if a readable
        generation has since appeared."""
        self._mem_invalidate(key)
        if not quarantine:
            return
        current = self._read_manifest(key)
        if expected_generation is None:
            if current is not None:
                return  # superseded by a readable generation: stand down
        elif current is None \
                or current.get("generation") != expected_generation:
            return  # superseded mid-read: plain miss, touch nothing
        self._quarantine(key, gen_dir)

    # ---------------------------------------------------------- memory tier

    def _next_mem_generation(self, key: str,
                             payload: Mapping[str, bytes]) -> int:
        """Monotonic per-key generation for memory-only mode (no disk
        manifest to read the counter from).  Returns 0 when nothing can be
        stored — tier inert (no positive bound) or the entry alone is
        larger than the whole tier — and does NOT consume a number then."""
        if not self._memory_enabled:
            return 0
        size = sum(len(b) for b in payload.values())
        if size > self.options.max_memory_bytes:
            return 0
        n = self._mem_gen.get(key, 0) + 1
        self._mem_gen[key] = n
        return n

    def _sweep_stale(self) -> None:
        """Age out abandoned tmp/ staging trees (crashed writers) and
        quarantine/ entries: nothing live stages (or waits for inspection)
        for hours, and unbounded crash debris would break the store's own
        bounded-data discipline at the store/ boundary (review MINOR-2)."""
        if self.disabled or not self._disk_enabled:
            return
        now = time.time()
        for sub in ("tmp", "quarantine"):
            root = os.path.join(self._root, sub)
            try:
                names = os.listdir(root)
            except OSError:
                continue
            for name in names:
                path = os.path.join(root, name)
                try:
                    if now - os.lstat(path).st_mtime > _STALE_SWEEP_S:
                        shutil.rmtree(path, ignore_errors=True)
                except OSError:
                    continue

    def _mem_put(self, key: str, payload: Mapping[str, bytes],
                 manifest: Mapping[str, object]) -> None:
        if not self._memory_enabled:
            return
        blob = {name: bytes(b) for name, b in payload.items()}
        # Insert-time digests: the anchor for verification='always' checks
        # of resident copies (a digest of the same bytes compared with
        # itself would verify nothing).
        checksums = {name: sha256_hex(b) for name, b in blob.items()}
        size = sum(len(b) for b in blob.values())
        if size > self.options.max_memory_bytes:
            return  # single entry exceeds the whole tier: never resident
        with self._mem_lock:
            old = self._mem.pop(key, None)
            if old is not None:
                self._mem_bytes -= old[0]
            self._mem[key] = (size, blob, dict(manifest), checksums)
            self._mem_bytes += size
            while self._mem_bytes > self.options.max_memory_bytes:
                _old_key, (old_size, _blob, _manifest, _csums) = \
                    self._mem.popitem(last=False)
                self._mem_bytes -= old_size
                self._evicted_memory += 1

    def _mem_get(self, key: str) -> Optional[StoreHit]:
        if not self._memory_enabled:
            return None
        with self._mem_lock:
            item = self._mem.get(key)
            if item is None:
                return None
            size, blob, manifest, checksums = item
            self._mem.move_to_end(key)
        if self.options.verification == "always":
            for name, b in blob.items():
                if name not in checksums or sha256_hex(b) != checksums[name]:
                    self._mem_invalidate(key)
                    with self._ctr_lock:
                        self._quarantined += 1
                    return None
        payload = {name: bytes(b) for name, b in blob.items()}  # fresh copies
        try:
            generation = int(manifest["generation"])
        except (KeyError, TypeError, ValueError):
            return None
        return StoreHit(key=key, generation=generation, payload=payload,
                        metadata=manifest.get("metadata", {}), hit="memory")

    def _mem_invalidate(self, key: str) -> None:
        with self._mem_lock:
            old = self._mem.pop(key, None)
            if old is not None:
                self._mem_bytes -= old[0]

    # ------------------------------------------------------------- eviction

    def _enforce_disk_quota(self) -> None:
        """Evict least-recently-used generations (generation dir mtime is
        touched on every verified read) until under ``max_disk_bytes``.
        Eviction never touches in-flight readers: they hold fresh copies."""
        if (self.disabled or not self._disk_enabled
                or self.options.max_disk_bytes is None):
            return
        entries = []
        total = 0
        store_root = os.path.join(self._root, "store")
        for prefix in os.listdir(store_root):
            for key in os.listdir(os.path.join(store_root, prefix)):
                gen = os.path.join(store_root, prefix, key)
                try:
                    usage = os.stat(gen)
                    size = self._tree_size(gen)
                    entries.append((usage.st_atime_ns, gen, size))
                    total += size
                except OSError:
                    continue
        if total <= self.options.max_disk_bytes:
            return
        for atime, gen, size in sorted(entries):
            if total <= self.options.max_disk_bytes:
                break
            key = os.path.basename(gen)
            self._mem_invalidate(key)
            shutil.rmtree(gen, ignore_errors=True)
            total -= size
            with self._ctr_lock:
                self._evicted_disk += 1

    @staticmethod
    def _tree_size(path: str) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total

    def _touch(self, key: str) -> None:
        """LRU touch of the generation dir after a verified disk read."""
        try:
            os.utime(self._gen_dir(key), None)
        except OSError:
            pass

    # -------------------------------------------------------- single-flight

    @contextmanager
    def singleflight(self, key: str, wait_timeout_s: float = 60.0,
                     poll_s: float = 0.02) -> Iterator[bool]:
        """Per-key single flight (dossier 06).  Yields True when THIS caller
        owns the compute (lock acquired and no generation committed yet),
        False when the lock holder that preceded us already committed this
        key — the caller should then re-run ``get`` and use the winner's
        committed generation instead of recomputing.

        The lock is a host-local file published atomically WITH its content
        (write to tmp/, fsync, ``os.link`` — O_EXCL semantics, so no waiter
        can ever observe an empty or half-written lock and steal a live
        holder's flight); the content records owner pid + a unique token.
        Release unlinks only on token match, so a stolen-then-reacquired
        lock is never unlinked by its former holder.  Dead holders (crashed
        process) are detected via pid liveness and the lock stolen.
        Cancellation/exceptions release the lock via the context manager.

        Scope notes: cross-process flight exists in disk mode only —
        memory mode has no on-disk lock by its no-disk-residue rule, so its
        flight is in-process only.  ``wait_timeout_s`` bounds the FILE-lock
        wait; the in-process serialization wait is unbounded by design (a
        same-process loser blocks until the owner's context exits, which is
        exactly the compute-once it asked for).  Not reentrant within a
        thread; single-flight is for compute-once on a miss — refresh flows
        that want a newer generation decide that explicitly.
        """
        _require_key(key)
        if self.disabled:
            yield True
            return
        with self._key_lock.hold(key):
            token = None
            if self._disk_enabled:
                token = f"{os.getpid()}-{next(_UNIQ)}"
                acquired = self._acquire_file_lock(
                    key, token, wait_timeout_s, poll_s)
            else:
                acquired = True  # in-process serialization is the flight
            # A loser ALSO acquired the file lock — it must release it on
            # exit, or its lock file orphans the key and stalls every later
            # waiter (release is token-matched, so this is always safe).
            owned = acquired and not self._committed(key)
            try:
                yield owned
            finally:
                if acquired and token is not None:
                    self._release_file_lock(key, token)

    def _committed(self, key: str) -> bool:
        """Has any generation for this key been committed (memory tier or
        disk)?  The single-flight loser check — deliberately NOT a verified
        read: it must be cheap and must not quarantine."""
        if self._memory_enabled:
            with self._mem_lock:
                if key in self._mem:
                    return True
        if self._disk_enabled and self._read_manifest(key) is not None:
            return True
        return False

    def _acquire_file_lock(self, key: str, token: str,
                           wait_timeout_s: float, poll_s: float) -> bool:
        """Acquire the per-key lock file.  The {pid, token, host} content is
        published ATOMICALLY with creation — written to a private tmp/ file,
        fsynced, then ``os.link``-ed into place (link fails with EEXIST if
        the lock exists).  An O_CREAT|O_EXCL create followed by a buffered
        content write would leave a stealable create→flush window in which
        a waiter reads an empty lock, declares it stale, and steals a LIVE
        holder's flight (review MAJOR-1, reproduced across processes)."""
        path = self._lock_path(key)
        deadline = time.monotonic() + wait_timeout_s
        while True:
            tmp = os.path.join(self._root, "tmp",
                               f"lock-{os.getpid()}-{next(_UNIQ)}")
            published = False
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"pid": os.getpid(), "token": token,
                               "host": os.uname().nodename}, f)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.link(tmp, path)  # atomic: appears WITH content
                    published = True
                except FileExistsError:
                    pass
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            if published:
                return True
            if self._lock_is_stale(path):
                try:
                    os.unlink(path)
                    continue  # immediate retry — the lock is gone
                except OSError:
                    pass  # someone else stole it first; loop and retry
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"single-flight wait timed out for key {key[:12]}… "
                    f"(holder may be alive but slow)")
            time.sleep(poll_s)

    @staticmethod
    def _lock_is_stale(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            pid = int(info["pid"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return True  # unreadable lock: treat as stale
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True   # holder is gone: crashed holder's lock is stolen
        except PermissionError:
            return False  # pid exists (another user): assume alive
        except OSError:
            return False
        return False

    def _release_file_lock(self, key: str, token: str) -> None:
        path = self._lock_path(key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            if info.get("token") != token:
                return  # our lock was stolen and reissued: never unlink theirs
        except (OSError, json.JSONDecodeError):
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    # ------------------------------------------------------------ diagnostics

    def inspect(self, key: str) -> Optional[dict]:
        """Raw view of a generation WITHOUT verifying or quarantining —
        for diagnostics and tests; never a substitute for :meth:`get`."""
        _require_key(key)
        if self.disabled:
            return None
        if self._memory_enabled:
            with self._mem_lock:
                item = self._mem.get(key)
            if item is not None:
                _size, blob, manifest, _csums = item
                return {"key": key,
                        "generation": manifest.get("generation"),
                        "metadata": dict(manifest.get("metadata", {})),
                        "files": {n: len(b) for n, b in blob.items()},
                        "on_disk_bytes": 0}
        if not self._disk_enabled:
            return None
        manifest = self._read_manifest(key)
        if manifest is None:
            return None
        files = {}
        for entry in manifest["payload"]:
            path = os.path.join(self._gen_dir(key), "payload", entry["name"])
            try:
                files[entry["name"]] = os.lstat(path).st_size
            except OSError:
                files[entry["name"]] = None
        return {"key": key, "generation": manifest.get("generation"),
                "metadata": manifest.get("metadata", {}), "files": files,
                "on_disk_bytes": self._tree_size(self._gen_dir(key))}

    def stats(self) -> dict:
        """Counters + bytes.  ``entries`` counts committed generations —
        disk manifests in disk mode, resident entries in memory mode; a
        hit counter alone is not engagement (dossier 06): callers benchmark
        avoided producer work above this layer."""
        entries = 0
        disk_bytes = 0
        if not self.disabled and self._disk_enabled:
            store_root = os.path.join(self._root, "store")
            for prefix in os.listdir(store_root):
                for key in os.listdir(os.path.join(store_root, prefix)):
                    gen = os.path.join(store_root, prefix, key)
                    if os.path.isfile(os.path.join(gen, MANIFEST_NAME)):
                        entries += 1
                        disk_bytes += self._tree_size(gen)
        with self._mem_lock:
            memory_entries = len(self._mem)
            memory_bytes = self._mem_bytes
            if not self._disk_enabled:
                entries = memory_entries  # the tier IS the store
            return {
                "mode": self.options.mode,
                "hits": self._hits,
                "misses": self._misses,
                "hit_reasons": dict(self._hit_reasons),
                "quarantined": self._quarantined,
                "evicted_disk": self._evicted_disk,
                "evicted_memory": self._evicted_memory,
                "memory_entries": memory_entries,
                "memory_bytes": memory_bytes,
                "memory_enabled": self._memory_enabled,
                "entries": entries,
                "disk_bytes": disk_bytes,
            }

    def clear(self) -> None:
        """Remove every cached byte, lock, staging and quarantine tree under
        the cache directory (the directory is dedicated to the cache).
        In-flight readers keep their copies — eviction under use is safe.
        clear() is NOT coordinated with concurrent writers or single-flight
        holders: a put racing clear fails LOUDLY (OSError), and held lock
        files are removed (cross-process flight is briefly unguarded) —
        quiesce writers/flights before clearing (review NOTE-6)."""
        if self.disabled:
            return
        with self._mem_lock:
            self._mem.clear()
            self._mem_bytes = 0
        if not self._disk_enabled:
            return
        for sub in ("store", "locks", "tmp", "quarantine"):
            shutil.rmtree(os.path.join(self._root, sub), ignore_errors=True)
            os.makedirs(os.path.join(self._root, sub), exist_ok=True)


class _Uniq:
    """Process-unique suffix without wall-clock in identities."""

    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()

    def __next__(self) -> int:
        with self._lock:
            self._n += 1
            return self._n


_UNIQ = _Uniq()
