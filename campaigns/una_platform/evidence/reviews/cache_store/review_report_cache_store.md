# CACHE_STORE — Independent Review (dossier 06 store protocol)

Reviewer: independent (fresh context, not implementation owner)
Worktree: `/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/cache-store` (branch `worktree-cache-store`)
Scope under review (uncommitted): `src/urban_network_analysis/cache/__init__.py`,
`src/urban_network_analysis/cache/store.py`, `tests/cache/store/` (conftest + 5 modules, 85 tests).
Interpreter: `/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python`, `PYTHONHASHSEED=0`.
`src/` and `tests/` untouched by the reviewer except in-place mutant application, each restored
and verified byte-identical (sha256 `9156b6f8966f35d4c0c3099c28d8f817a2f3ed97b506e428cad388ea99bd876a`
before and after; clean 85/85 re-run after final restore). Probes live in `/tmp/cache_review/`.

## Verdict: ACCEPT-WITH-FINDINGS

1 MAJOR (cross-process single-flight steal race — must fix before merge), 4 MINOR, 8 NOTE.
The storage primitive is genuinely sound where it claims to be sound: 85/85 green, all six
claimed sabotage mutants selected, no pickle anywhere, atomic publish verified down to a real
`os._exit(9)` between the two renames, corruption quarantined, fresh-copy discipline real,
import hygiene real. The MAJOR is one narrow but real race in the lock protocol that the
delivered suite structurally cannot see (it tests threads and a synthetic pid, not contending
processes), plus minor contract/documentation honesty items.

---

## Findings

### MAJOR

**MAJOR-1 — Cross-process single-flight is stealable while the holder is alive (lock content
is not atomic with lock creation).**
`store.py:604-609` (`_acquire_file_lock`): the lock file is created with `O_CREAT|O_EXCL` and
its `{pid, token, host}` content is written afterwards through a *buffered* text file, flushed
only at close. `store.py:624-631` (`_lock_is_stale`): any unreadable lock — empty **or
partially written JSON** — is declared stale, and `store.py:612-617` unlinks it immediately and
retries. A waiter arriving inside the create→flush window therefore steals a **live** holder's
lock; both sides proceed as owner and both compute.

Evidence (all mine, commands in the run log below):
- Real 6-process contention (`/tmp/cache_review/probe1_multiproc_sf.py` + `worker_sf.py`):
  run 2 observed **two processes printing `COMPUTED owner=True`** for the same key (p1, p2);
  the 5 losers and the double-owners all ended on identical committed bytes. Output was
  produced by the children directly on inherited stdout, so this is genuine child behavior,
  not a harness artifact.
- Deterministic mechanism proof (`/tmp/cache_review/probeD_lock_steal.py`): a holder holds the
  O_EXCL-created empty lock open for 0.5 s (exactly the code window between `os.open` and the
  content write); a real `singleflight()` from an independent instance declares it stale,
  unlinks it, acquires, and yields `owner=True` while the first holder is alive and about to
  write its content. Verdict printed: `VIOLATION`.
- Frequency: `probe1b_repeat.py`, 10 further runs of 6 processes → 10/10 single owner. The
  race is intermittent (arrival clustering dependent), which is exactly why the delivered
  suite misses it.
- Release discipline is NOT the problem: after a steal, the former holder's release reads the
  new token and does not unlink (`store.py:642-654`), and probe F2 found **no orphaned locks**
  after a 6-process run (`locks/` empty).

Consequence: duplicate producer work for the same key (results stay correct — content
addressing + atomic rename + identical bytes), i.e. the dossier-06 "identical requests compute
once" guarantee is violated under exactly the multi-process concurrency the cache exists for.
Suggested fix (small, local): publish the lock content atomically with creation — write
`{pid, token, host}` to `locks/.tmp-<uniq>` and `os.link(tmp, lockpath)` (O_EXCL semantics with
content), or refuse to steal an unreadable lock until its mtime exceeds a small grace period.
Also add a real multi-process single-flight test (see NOTE-8).

### MINOR

**MINOR-1 — Structurally-corrupt manifests make `get()`/`require()` raise instead of
miss/quarantine; read-side payload names bypass the put-side traversal guard.**
`_read_manifest` (`store.py:401-413`) validates only top-level shape (dict / schema_version /
key / payload-is-list); `_disk_get` (`store.py:377-392`) then trusts each element's structure
(`entry["name"]`, `entry["bytes"]`, `entry["sha256"]`) and uses `entry["name"]` in
`os.path.join` with no traversal validation — the put-side name guard (`store.py:221`) does not
apply on the read path. Probe B (six hand-corrupted manifests over a committed entry):
`payload-entry-is-string` → `TypeError: string indices must be integers`;
`payload-entry-missing-bytes` → `KeyError: 'bytes'`; `payload-entry-missing-sha256` →
`KeyError: 'sha256'`; `payload-name-is-int` → `TypeError` from `os.path.join`;
`metadata-not-a-mapping` → `TypeError`; only `payload-name-traversal` was safely a miss (caught
by size/checksum, not by name validation). This contradicts the module docstring
(`store.py:18-20`: "corruption is quarantined and reported as a miss") and the dossier's
"Readers verify lengths/checksums/schema before use". `require()` also breaks its typed
contract: callers catching `CacheCorruption` for "damaged" get raw `TypeError`/`KeyError`
instead. The traversal read is gated by attacker-chosen manifest `bytes`/`sha256` fields
matching the target file, so it is not a privilege escalation over "can write the cache dir" —
but the schema check is incomplete. Fix: validate each payload entry (dict, str name without
separators/`..`, int bytes, hex sha256) in `_read_manifest` and return None otherwise.

**MINOR-2 — The retire–republish window quarantines the *good* new generation; crash-in-window
leaks a complete generation into `tmp/` that is never swept; generation counter regresses.**
`put()`'s docstring (`store.py:191-195`) says a reader racing the retire window "observes a
miss, never partial data". Probe C (deterministic: reader descheduled after `_read_manifest`,
writer republishes a different-length payload) shows the reader does more than miss: it
verifies OLD checksums against NEW files and **quarantines the freshly published generation**
(gen 2, payload `b'BBBBBB'` ended up only in `quarantine/`, `store/<key>` gone, every later
reader misses until recompute). The miss claim holds; the side effect is undocumented entry
loss. Separately, probe F1 (child process `os._exit(9)` between the retire rename
`store.py:269` and the publish rename `store.py:271`): reader sees a MISS (claim 3 holds), the
store recovers on the next put — but the retired generation, **including its valid
manifest.json**, remains in `tmp/` indefinitely (still present after the recovery put), is
invisible to `_enforce_disk_quota` (walks `store/` only, `store.py:505`) and to `stats()`, so
repeated crashes accumulate unbounded `tmp/` (and `quarantine/` similarly never self-sweeps) —
bounded-resident-data discipline stops at the store/ boundary. The recovery put also returned
generation 1 again (the pre-crash generation was also 1): generation numbers are not stable
across crashes, which nothing documents. Fix directions: sweep stale `tmp/<key>-<pid>-<n>`
trees on startup/quota walk (they are parseable and ownable by pid), and either re-verify
before quarantining or restrict quarantine to entries whose damage is confirmed against the
manifest that names them.

**MINOR-3 — `CacheOptions.schema_version` is silently ignored by the store.**
Contract §6 / `Execution.py:114` define `schema_version: int = 1` as a CacheOptions field; the
store hardcodes `_SCHEMA_VERSION = 1` for both write (`store.py:250`) and read validation
(`store.py:409`). Probe E1: with `CacheOptions(..., schema_version=2)` the manifest still
records `"schema_version": 1`, the write is accepted, and a manifest that *does* say 2 is
treated as corrupt (miss) — the option has no effect and raises no error. Either honor the
field or reject values != 1 loudly.

**MINOR-4 — Required task outputs are absent.**
TASKS_CLAUDE.yaml CACHE_STORE lists outputs
`evidence/tasks/CACHE_STORE/<RUN_ID>/receipt.json` and `evidence.json`; no
`evidence/tasks/CACHE_STORE/` exists anywhere in the worktree (verified by find over the whole
tree). As of this review the deliverable is code+tests only.

### NOTE

**NOTE-1 — `_quarantine` docstring "never destroyed" is falsified by its own fallback.**
`store.py:416-418` promises quarantine "never destroyed: quarantine/ keeps it for diagnosis";
`store.py:428-429` `shutil.rmtree(gen_dir, ignore_errors=True)` when the quarantine rename
fails. Probe E5 (rename denied): the damaged entry is silently destroyed, `quarantine/` empty.
Rare path; the docstring should say "best effort".

**NOTE-2 — Oversized put in memory mode returns a nonzero generation for an entry that was
never stored.** `store.py:205-212` + `store.py:451-452`: `put()` returns generation 1 while
`_mem_put` dropped the oversized entry (probe E2: `gen=1, get()=None, entries=0`). Also
`stats()["entries"]` is always 0 in memory mode (it counts only disk manifests). Callers using
`put(...) != 0` as "stored" are misled in exactly the bounded-tier case the bounds exist for.

**NOTE-3 — `singleflight` docstring omits three operational facts.** (a) cross-process flight
exists only in disk mode — memory mode's flight is in-process only (`store.py:571-576`), which
is forced by memory mode's no-disk-residue rule but never stated; (b) the in-process
`_KeyLock` wait is unbounded and `wait_timeout_s` does not apply to it (probe E7: a second
thread was still blocked 2 s into a 0.3 s timeout, released only when the owner released);
(c) unreadable-lock-equals-stale also races live holders (see MAJOR-1).

**NOTE-4 — Diagnostics counters are unlocked.** `_hits/_misses/_quarantined/...`
(`store.py:133-138`) are plain ints incremented from multiple threads; increments can be lost
under contention. Diagnostics only, but "hit reasons" are the dossier's engagement surface.

**NOTE-5 — Package docstring states the bitwise-or-miss guarantee unconditionally.**
`__init__.py:8` and `store.py:30-32` promise "whatever bytes were committed under a key come
back bitwise, or the entry is a miss"; under `verification='never'` a same-length corruption is
returned as a hit (the suite itself documents this as an explicit caller choice,
`test_verification_never_documents_its_trust`). The guarantee should be scoped to
checksummed policies (default `on_read` makes it true by default).

**NOTE-6 — Concurrent `clear()` aborts in-flight puts loudly; undocumented.** Probe F3 (2
putter threads + a clear/stats thread, 4 s): 3469 errors, all `put` →
`FileNotFoundError` (clear rmtree'd `tmp/`/`store/` mid-put); no corruption, no silent
acceptance. `clear()` also unlinks currently-held single-flight lock files (it rmtree's
`locks/`), temporarily breaking cross-process flight; neither effect is in the docstring
(`store.py:720-733`).

**NOTE-7 — Coverage gap: oversized-entry collateral eviction is unobservable.** Mutating the
size guard `store.py:451` to `if False:` (my first M6 attempt) leaves the suite **85/85 green**
— the while-drain (`store.py:459-463`) produces the same end-state counters by evicting the
*entire* tier to admit one oversized entry. No test distinguishes "oversized put never resident,
nothing else evicted" from "oversized put drains everything".

**NOTE-8 — Coverage gaps against the completion condition.** Corruption / partial write /
crash / stale lock / eviction / inflight / no-pickle are each covered (see audit below), but:
(a) **no multi-process single-flight test** — the suite's concurrency is threads within one
process plus a synthetic live pid (`test_singleflight.py:127-146`); the MAJOR-1 regime
(contending processes) is untested; (b) no structurally-corrupt-manifest variants (probe B's
five raise paths); (c) the import-hygiene claim (`store.py:34-35`, `__init__.py:5-6`) has no
test in `tests/cache/store/` (unlike tests/execution's import-hygiene probes) — I verified it
myself (probe E6: no numpy/pandas/numba/... and only `Execution` + cache siblings loaded).

---

## Independent run log (commands + observed outcomes)

```
# A. Suite
$ cd .../worktrees/cache-store && PYTHONHASHSEED=0 <alan python> -m pytest tests/cache/store/ -q
85 passed in 1.02s        # re-run after final restore: 85 passed in 0.88s

# Probe 1/1b — multi-process single-flight (6 procs, 1 key, 0.15s owner compute)
run 1: p0 COMPUTED, 5 losers (single-flight held)          [harness bug: no PIPE; child output direct]
run 2: p1 COMPUTED **and** p2 COMPUTED, 4 losers           -> double ownership observed
probe1b (PIPE capture, 10 runs): 10/10 runs computed=1     -> intermittent, mechanism confirmed by probe D

# Probe D — deterministic mid-write steal
waiter state while holder's lock was still EMPTY: ['waiter']; lock file unlinked
VERDICT: VIOLATION — live holder's not-yet-written lock declared stale and stolen

# Probe B — structurally corrupt manifests (6 variants)
5/6 RAISED TypeError/KeyError out of get(); 1/6 (traversal name) safe miss+quarantine

# Probe C — reader racing re-put (different payload), deterministic deschedule
writer gen 1 -> 2; racing reader: miss; reader quarantine counter: 1
store/<key> present: False; quarantine/ holds gen 2 payload b'BBBBBB' (good data quarantined)

# Probe E — edges
E1 schema_version=2 requested: manifest=1, get(v2-manifest)=False -> option IGNORED
E2 memory-mode oversized put: gen=1, get=None, entries=0    -> misleading nonzero gen
E3 bytearray/memoryview mutated after put (disk+mem tier): reads back originals -> SAFE (no aliasing)
E4 os.fsync raising: put raises, tmp leftovers=[], entries=0 -> clean abort, no residue
E5 quarantine rename denied: entry rmtree'd, quarantine/ empty -> evidence DESTROYED (vs docstring)
E6 import urban_network_analysis.cache: heavy=[] siblings=[Execution, cache, cache.store] -> hygiene HOLDS
E7 thread waiter with wait_timeout_s=0.3: still blocked at 2s; released when owner released
E8 retire-rename failure in-process: original generation intact (restore path works)

# Probe F — real crash + hygiene
F1 child os._exit(9) between retire and publish: exit=9; store/<key> gone; reader MISS;
   tmp/ holds the FULL retired generation incl. manifest.json (4 files, manifest=True);
   recovery put -> gen 1 (counter regressed); tmp residue still present after recovery
F2 6-process run after fix of harness: computed=1/1; locks/ after run: [] (no orphaned locks)
F3 2 putters + clear/stats loop, 4s: 3469 errors, all put->FileNotFoundError (loud, no corruption)

# C. Mutant re-runs (each: sed in place -> pytest -> restore -> sha256 == 9156b6f8...; final clean run 85/85)
M1 remove _touch after disk get      -> 1 failed:  test_recent_read_is_protected_from_disk_eviction   CONFIRMED
M2 skip on_read checksum             -> 2 failed:  test_bitflip_detected_on_read,
                                                   test_verification_never_documents_its_trust        CONFIRMED
M3 token-mismatch release unlinks    -> 1 failed:  test_release_never_unlinks_a_stolen_and_reissued_lock CONFIRMED
M4b quarantine rename disabled       -> 2 failed:  test_truncated_payload_is_quarantined_miss,
                                                   test_clear_removes_everything_including_quarantine  CONFIRMED
M5 owned = acquired (loser recomputes)-> 2 failed: test_loser_regets_instead_of_recomputing,
                                                   test_compute_once_under_thread_race                 CONFIRMED
M6 while-drain disabled (while False)-> 4 failed:  memory_lru_eviction_order_pure_tier,
                                                   disk_tier_survives_memory_eviction,
                                                   memory_never_exceeds_bound,
                                                   memory_mode_also_bounded_and_lru                    CONFIRMED
M6' size-guard -> if False (my variant, not the claimed form)
                                     -> 0 failed: 85/85 green — survives; see NOTE-7
```

## Test-coverage audit vs completion condition

| Completion-condition behavior | Covered by | Gap |
|---|---|---|
| Corruption | truncated payload, bitflip, 5 manifest damage variants, memory rot under 'always' | structural manifest damage (MINOR-1), quarantine rename failure (NOTE-1) |
| Partial write | staging leftovers, crash-before-publish, crash-during-payload-write | crash between retire/publish leaks tmp (MINOR-2, only probed) |
| Crash | in-process monkeypatched rename/open; `singleflight` no-commit path | no real multi-process crash; no tmp sweep (behavior absent) |
| Stale lock | dead-holder steal (deterministic dead pid), unreadable lock, live-holder timeout | mid-write live holder (MAJOR-1) |
| Eviction | disk LRU order, read-protection, under-use safety, clear | oversized-entry collateral eviction (NOTE-7) |
| Inflight | fresh-copy after clear, disk eviction under use, thread race compute-once | multi-process race (MAJOR-1, NOTE-8a) |
| No untrusted pickle | non-bytes rejected at the door; store tree contains only manifest+payload | import-hygiene probe absent (NOTE-8c) |
| No unbounded resident data | memory tier inert without positive bound; oversize never resident; quota eviction | tmp/ and quarantine/ unbounded across crashes (MINOR-2) |

## Dossier-06 compliance table

| Dossier requirement (Store protocol / Tests) | Status |
|---|---|
| Read-only immutable values, copy-on-use for mutable outputs | PASS (fresh bytes every read; probe E3, `test_hit_payloads_are_fresh_copies`) |
| Per-key single flight | PARTIAL — holds in-process and in 11/12 observed 6-process runs; stealable live-lock race (MAJOR-1); memory-mode flight is in-process only (NOTE-3a) |
| Bounded memory LRU + bounded disk quota | PASS (M6 mutants selected; quota walk ignores tmp//quarantine — MINOR-2 residue) |
| No unbounded retained data | PARTIAL — resident data bounded; on-crash tmp/ residue unbounded (MINOR-2) |
| Writers: schema/dtypes/shape/payload checksums in temp generation, fsync, atomic manifest publish | PASS for bytes/checksum schema (payload-level manifest verified, probe B baseline + `test_disk_layout_and_manifest_checksums`); fsync failure aborts clean (E4); dtype/shape are upper layers' responsibility (raw-bytes store, documented) |
| Crash midwrite never creates a hit | PASS (probe F1 real `os._exit(9)`: reader MISS, recovery works) |
| Readers verify lengths/checksums/schema before use | PARTIAL — lengths/checksums yes; schema only top-level (MINOR-1) |
| Corruption: treat as miss/quarantine, never partially-finite | PARTIAL — recognized damage miss+quarantine; structural damage raises (MINOR-1); quarantine can destroy evidence on rename failure (NOTE-1); retire window can quarantine good data (MINOR-2) |
| Non-executable formats, allow_pickle=False, path traversal checks | PASS for store (no pickle anywhere — grep + type gate + only-manifest-and-payload files); payload-name traversal checked at put only, manifest-declared names unvalidated on read (MINOR-1) |
| Ejecting data must not invalidate in-flight readers | PASS (fresh copies; tests + probe F1/F2) |
| Warm/cold/off bit parity | PASS (`test_cold_warm_off_bitwise_equivalence`, promotion test) |
| Identical requests compute once | PARTIAL — see MAJOR-1 |
| Cancellation releases lock | PASS (exception/KeyboardInterrupt param test; probe D shows release stays token-safe even after steal) |
| Corrupt entry and stale lock recover | PASS |
| Returned arrays do not alias cache | PASS |
| Hit counter not mistaken for engagement | PASS (stats() docstring delegates producer-work benchmarking upward; no engagement claim made at this layer) |
| Contract §6 CacheOptions honored | PARTIAL — 5 of 6 fields honored; `schema_version` silently ignored (MINOR-3) |

## Claim-by-claim audit of the implementer's 11 claims

1. 64-hex keys / traversal guard — TRUE (`_require_key`, 10 bad-key forms, all entry points).
2. Non-executable store — TRUE (no pickle/eval/marshal; type gate at put; manifest is own-schema json).
3. Atomic publish — TRUE (staging→fsync→rename; real-crash probe F1; retire-window caveat MINOR-2).
4. Size always + sha256 per policy; quarantine-as-miss; require() distinguishes — TRUE for recognized damage (MINOR-1 exceptions; require() evidence-in-place verified).
5. on_write/always genuine readback — TRUE (readback compared to caller bytes; tampering test selects; 'always' re-checksums memory against insert-time digests, test selects).
6. Single-flight semantics (both release; token-matched unlink; True/False; timeout; dead-holder recovery) — TRUE in-process; FALSE across processes under the create/flush race (MAJOR-1); wait_timeout_s does not bound in-process waits (NOTE-3b).
7. Memory tier inert without positive bound; bounded LRU fresh-copy; oversize never resident; memory mode disk-free; disk mode + optional tier — TRUE (M6 selected; E3; probe E2 caveat NOTE-2).
8. Disk quota LRU via touched atime; eviction invalidates memory; in-flight safe — TRUE (M1/M6' selected; `test_disk_eviction_under_use_is_safe`).
9. Bitwise round-trip or miss — TRUE under default on_read; unconditional docstring overstates (NOTE-5).
10. mode='off' fully inert, no directory — TRUE (`test_contract_defaults_are_off_and_create_nothing`).
11. No analysis-stack imports — TRUE (probe E6), but untested in the suite (NOTE-8c).

## Bottom line

Fix MAJOR-1 (atomic lock-content publish or grace-period before stealing unreadable locks) and
add one multi-process single-flight test; decide and enforce `schema_version` (MINOR-3); align
the retire-window/quarantine/evidence docstrings with probed reality (MINOR-2, NOTE-1, NOTE-3,
NOTE-5); validate manifest payload-element shape (MINOR-1); produce the required receipt/
evidence outputs (MINOR-4). None of these disturb the store's data-correctness guarantees;
MAJOR-1 disturbs its work-saving guarantee under contention.
