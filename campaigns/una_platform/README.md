# UNA platform campaign: parity, correctness, reliable throughput

Active campaign: `una-platform-2026-09`. Repository: `AlanSynn/una-x`.
Development branch: `perf/una-platform`. Begin with START_HERE.txt.

This is an execution specification, not a claim that the requested features already exist.
Implement the Madina-compatible API, scientific corrections, a genuine AOT native CPU backend,
a genuine GPU backend, publicly callable parallel RunBatch, validated caches, and bounded
recovery. Optimize the complete installed application, not a synthetic kernel.

The previous two campaigns are historical. Their restrictions against GPU, native code,
public batch parallelism, caching, and scientific fixes are superseded by this campaign.
Their evidence and implemented improvements are retained, not rerun or discarded blindly.

Read order: EXECUTION.md -> AUTHORITY.md -> SOURCE_AUDIT.md -> CONTRACTS.md ->
NUMERICS.md -> TASKS_CLAUDE.yaml. Read each task's dossier before implementing it.
BENCHMARKS.md, VALIDATION.md, RESOURCES.md, and DECISION.md govern admission and completion.
API_PARITY.json is a source-grounded seed, not an exhaustive API census. Expand it from
both pinned source trees and Madina's documented examples before claiming parity.

The spelling “Bitparty” is interpreted as **bitwise parity**. Same-profile backend results
must match exact typed reference bits. Corrected outputs must match a reviewed corrected
reference, not erroneous historical outputs. No tolerance-based substitute is authorized.

Local packet checks (from repository root):
```sh
python3 campaigns/una_platform/tools/validate_packet.py
python3 -m unittest discover -s campaigns/una_platform/tests -v
```
These commands validate the handoff tools, not UNA performance or scientific correctness.

Required capabilities are not optional ideas. Hardware absence can block GPU qualification,
not justify deleting the GPU task or claiming CPU fallback is GPU support. Slow but correct
explicit backends may remain supported; automatic selection requires a measured crossover.
A missing mandatory gate yields partial/blocked status, never full qualification.
