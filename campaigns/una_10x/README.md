# UNA-X 10x: sparse queries and dependency-aware execution

Status: executable campaign specification, not an achieved application speedup.
Active goal: at least 10x throughput over the strongest equally resourced current-main
implementation, through installed public APIs with every required output and bitwise parity.

Start at `START_HERE.txt`. The execution DAG is `TASKS_CLAUDE.yaml` (JSON-compatible YAML).
The routing entrypoint is the repository's `/goal`; development branch: `perf/una-10x`.

The two principal hypotheses are (S) epoch-tagged labels plus local destination enumeration,
and (C) routing traces plus per-metric reuse across DISTINCT batch jobs. The attached research
packet is preserved byte-for-byte under `prior/una_10x_investigation/`. It reports prepared-array
component results, not a checkout/public-API/wheel win. Treat it as evidence to bridge, not code
to copy into the package blindly. Its small and search-dominated regressions are retained.

The baseline is `395cdc5f683894b6f2ba460f7dbcefee99981ba3`, including prior CSR/search/flow,
cache, public batch, and kernel-contract work. Never restart at pre-CSR source to inflate a ratio.

Read `AUTHORITY.md`, `CONTRACT.md`, `SOURCE_AUDIT.md`, `MODEL.md`, `WORKLOADS.md`,
`BENCHMARKS.md`, `VALIDATION.md`, `RESOURCES.md`, `PLATFORM_CONTINUITY.md`, `DECISION.md`,
and the dossier for your task. Operational ownership and recovery are in `EXECUTION.md`.

This campaign reprioritizes the unfinished platform program; it does not declare it completed
or waive Madina parity, scientific fixes, substantive native/GPU support, public RunBatch,
cache integrity, cancellation, or recovery. See `PLATFORM_CONTINUITY.md` and the unchanged
`campaigns/una_platform/` packet. A scoped 10x result is not full platform qualification.

Packet-only verification, no heavy application work:
```bash
python3 campaigns/una_10x/tools/validate_packet.py
python3 -m unittest discover -s campaigns/una_10x/tests -v
```
Run the source audit and direct prototype bridge only in a clean, isolated checkout and a
budgeted environment. Commands are in `START_HERE.txt` and dossier 01.

Final artifacts include a reproducible paired-ratio result, raw windows, exact wheel/source
identities, bit/state/artifact checks, sparse/reuse engagement counts, memory lifetimes, capability
coverage, inherited obligations, and an honest target disposition. No result file starts as passed.
