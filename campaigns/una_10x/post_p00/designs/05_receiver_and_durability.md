# P04: bounded drain and ordered durable publication

## Conditional, not automatic
Current workers already export inside RunAccessibility; do not count this as newly parallel work.
writer_concurrency was unused in P00. Admit a receiver change only if post-P02/P03 telemetry shows
material result-put/commit backpressure. Keep the original serial and synchronous fallback.
Changing a durability guarantee to increase throughput is prohibited.

## First schedule: one receiver, one commit owner
Retain synchronous completed-result transport. A SINGLE receiver drains it into an explicitly
bounded ready map while the coordinator commits the next logical row. No second consumer races
with the receiver, including worker-death drain and cancellation. Receiver cannot modify UNA,
Settings, composite state, public paths, journal or cursor. It only transfers ownership of a
complete receipt and records errors. The coordinator remains sole publication/state owner.

```
worker: compute -> original export -> close -> staged hashes -> send complete RowResult
receiver: accept complete result -> checked byte reservation -> ready[row_id] -> wake commit
coordinator: wait for cursor receipt -> verify -> fsync -> rename -> dir sync -> apply state
             -> composite -> journal -> release ready credit -> advance cursor
```

A bounded ready map can deadlock if later results fill it while the earliest result is blocked
in the common pipe. Reserve complete-response memory/descriptor credit BEFORE dispatch. Every
in-flight row must have admission credit that permits reception independent of arrival order.
Never dispatch more unreserved results than can be accepted. Reserve an error/control lane too.
Unknown response bounds must use a measured conservative preflight bound or the original route,
not a guessed unlimited exception. Spill large owned bundles/receipts to private files with
checksums when justified; account staging bytes and verify before unpickling trusted same-run
state. Do not put untrusted cache pickle into this path. Staging URLs are not proof of ownership.

The last engine/topology bundle can be much bigger than other receipts. Count serialization,
probe-pickle, pipe, receiver decode and final rehydration separately. Do not send a huge last-row
bundle through the only channel while its unknown size blocks earlier required receipts. A
small descriptor + owned checked state spool is an OPTIONAL separately reviewed solution; keep
public state complete, never discard it or rerun the scientific job as an uncharged shortcut.

## Crash and cancellation ownership
Receiver has a stop event and a bounded shutdown protocol; it must not wait for a committed row
while commit waits for it. Coordinator observes receiver failure, stops dispatch and follows the
existing earliest-ordered-failure/prefix contract. Test worker death during put, full pipes,
partial/truncated pickle, slow consumer, cancellation during receive, and retained sibling results.
Poll saying bytes exist does not prove a whole serialized frame is available. A background thread
blocked forever in a receive is not fixed by a timed join. If framing can block indefinitely,
use bounded-framed/per-worker channels or owned descriptor spooling after a separate protocol
review; otherwise do not promote the new receiver. Keep supervisor bounds in fault tests.
Never regress to asynchronous Queue feeder semantics that can lose completed results on death.
Do not silently weaken the guarantee because the baseline also has an untested edge case.

## Publication order is unchanged
Existing order includes worker close/hash, parent staged digest verification, file fsync before
name flip, os.replace for each file, destination-directory fsync, original state/composite fold,
and journal completion. A row is not multi-file atomic simply because individual renames are.
A crash midway leaves no COMPLETE row journal; resume revalidates every expected artifact.
The new pipeline must retain that rule and original ordering, including same-name conflicts.
Retain the existing best-effort policy when directory fsync is unsupported; do not claim stronger
filesystem durability than was actually obtained or convert it into a new unexpected API error.

First implementation changes ONLY reception/ownership, leaving _verify_staged/_publish_staged
unchanged. If file fsync is later a measured bottleneck, pre-sync privately closed staged files
before cursor is eligible ONLY with proof that inode/content cannot change after sync. Keep the
parent verification, error class and per-row publication check. A new synced-token is valid only
for that owned inode+generation+digest, never a path-only boolean. Directory sync/journal still
follow rename and state. Integrity rechecking is not deleted to claim a win.

## Writer limits and memory
If writer_concurrency is implemented, define it as the cap on simultaneous encode/write stages,
not a new serialized global scientific loop. Threads/processes consumed by writers share C=4.
Changing the previously ignored option's default can regress throughput; benchmark and document
its semantics before promotion. Do not introduce a fifth compute owner to beat a four-core arm.
Every staged output remains a real independent file; no shared hard-link aliases or omitted
copies. Queue depth limits jobs AND admitted bytes. Release descriptors, logs, buffers and owned
spools only after corresponding consumers and final state complete.

## Required tests
Reverse completion order with cursor delayed; ready-map cap reached; oversized last bundle;
worker death before/within/after put; receiver exception; early-row validation failure while later
rows finish; corruption after worker hash; fsync/rename/journal failures; interruption after one
of a row's two files; resume with mixed valid/invalid prefix; same output names; two batches using
same output root; no leaked threads/processes/leases. Validate artifacts and live state at every
commit. Only performance runs with all required jobs completed and verified enter selection.
