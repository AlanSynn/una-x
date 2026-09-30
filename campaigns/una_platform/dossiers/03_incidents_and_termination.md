# D03: Reported city failures, hangs, and termination

## Evidence discipline
incidents.json distinguishes published reports, source hypotheses and locally reproduced facts.
No reviewed report yet establishes a named city's universal failure or a city-specific stall
root cause. Obtain the reported dataset/settings/environment when available; otherwise retain
that gap and construct minimal mechanism tests. Never label a synthetic failure as the actual report.

Published leads: Madina #12/PR13 pandas fastpath and NumPy GeoDataFrame splitting; #8/PR10
GeometryArray.data network creation; #7 service-area clipping with GeometryCollection; #15
missing dependency declaration; #14 package distribution. PRs are not assumed merged fixes.
The source explicitly warns of large detour all-path memory/performance blowup.

## Reproduction matrix
Run unchanged Madina, current UNA, dependency-bridged Madina and corrected candidate in separate
pinned environments. Record versions, OS/start method, dataset hash/CRS, network size/component
census, origins/destinations/radius/detour, stage entered, output obligations and exit status.
Each attempt has an independent watchdog and process-tree memory/disk limits. Repeated executions
of a known runaway input are forbidden unless the mechanism or stopping bound changed.

## Distinguish failure classes
- Import/ABI/dependency failure before computation.
- Geometry/type/index/CRS error or malformed topology during preparation.
- Finite but combinatorial all-path work; record paths generated and lower-bound output bytes.
- Algorithmic nontermination: repeated state with no admissible progress (negative cycles,
  zero-cost walk enumeration, stale mutation). Prove with a minimized state trace.
- Queue deadlock or orphaned worker: retain parent/child stacks, queue ownership, last event.
- Memory pressure, swapping, disk exhaustion, JIT compilation or slow I/O masquerading as a hang.
A timeout alone is NOT a root cause and NOT a finite successful execution time.

## Diagnostic protocol
Parent emits heartbeat from its own loop even if a worker holds the GIL. Worker emits phase,
completed units, queue frontier, high-water memory and last successful commit at bounded cadence.
On timeout: request stack dump where safe, set cooperative cancellation, wait a bounded grace,
terminate only owned workers, join/reap, preserve repro/log/state capsule and mark result incomplete.
Do not publish final-looking scientific artifacts from incomplete jobs. Resume tokens carry
input/profile/code identity and exact completed logical unit sequence.

## Mechanism minimization
Preserve original source IDs while reducing edges/points/settings. Reduce one factor at a time:
geometry types, duplicate/self-loop structure, components, cost values, radius, detour, process
start mode and output filesystem. Test causal intervention: a single correction should remove
the failure and a deliberate restoration should reproduce it within the safe bound.
GeometryArray workaround that drops non-LineStrings is not an equivalent topology fix.

## Exact all-path scaling
An API returning every path has an output-size lower bound and may have exponentially many
valid results. Optimize pruning with exact stored lower bounds, stream/spill intermediate
paths, and preserve enumeration/order. A finite resource/deadline limit returns an explicit
resource exception or documented incomplete stream, never K paths labeled ALL. Streaming is
an additive API; compatibility materialization remains available when it fits resources.

## Completion
Incident report: status, source URL, pinned reproduction, cause confidence, failed-stage trace,
minimal fixture, fix commit, regression, residual limitations. Unavailable original city data
stays unavailable. Core mechanisms and robust cancellation must still be implemented and tested.
