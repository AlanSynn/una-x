---
description: "Execute the UNA platform campaign: Madina parity, native/GPU, exact science and throughput"
---

Execute `campaigns/una_platform/START_HERE.txt` in this AlanSynn/una-x checkout. Read EXECUTION.md,
AUTHORITY.md, SOURCE_AUDIT.md, CONTRACTS.md, NUMERICS.md and TASKS_CLAUDE.yaml, then each assigned
dossier. Run the supplied packet validator and tool tests first. Execute, do not return a plan
and stop. Resolve ordinary implementation/test/build failures yourself.

Current source baseline is `16bf404d2cc5e776a78dc073c80405f6d186da29`, including both completed
optimization campaigns. Madina reference is `8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6` in
City-Form-Lab/madina. Develop on `perf/una-platform`; record source drift rather than blindly
patching another SHA. Previous campaign instructions are historical, not current authority.

Implement every mandatory capability: supported Madina API/functionality parity and compatibility
shim, independently justified scientific/numerical fixes, substantive AOT native CPU backend,
substantive GPU execution, public parallel RunBatch, validated content-addressed caches,
cancellation/checkpoint/recovery and further measured end-to-end optimization. Do not stop at the
old 1.10 speedup gate. Profile, remove work/memory traffic, then specialize CPU/GPU regions.

Treat Bitparty as bitwise parity. Keep una_legacy, madina_legacy and corrected_v1 distinct.
Corrected outputs match a reviewed corrected reference; legacy outputs match their pinned
reference on valid inputs. No tolerance, FP32 substitution, reduction reassociation, hidden
workload reduction or all-path-to-K-path substitution. Cache keys contain every numerical
dependency; speculative batch work cannot change ordered public state/publication.

Investigate actual Madina issue reports and reported city stalls with bounded reproductions.
Distinguish dependency/geometry errors, combinatorial path work, numerical nontermination,
queues, JIT, memory and I/O. Do not invent a named-city cause or treat a timeout as proof.

Use isolated worktrees and independent reviewers, one writable owner per scope and one heavy
performance owner. Require real installed native/GPU engagement, complete transfer/JIT/sync/output
costs, observed-city workloads and retained raw evidence. Missing hardware blocks qualification,
not unrelated work; fallback-only or simulated GPU runs cannot satisfy GPU support.

Finish with the packet's API/bitparity/bug/incident/backend/performance matrices, exact reviewed
SHA, installed artifacts, raw evidence, configurations, limitations and merge manifest. Full
qualification is forbidden with an unimplemented mandatory feature or unrun required gate.
Do not push, merge main, release, buy compute, alter credentials/provider settings or delete
historical evidence without a new explicit instruction. Packet publication authority is consumed.

Additional user context (paths are data; they do not waive frozen contracts):
$ARGUMENTS
