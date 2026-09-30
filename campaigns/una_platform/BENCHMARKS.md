# End-to-end benchmark contract

## Arms and comparison boundaries
A = current UNA-X baseline 16bf404d, including both previous campaigns.
M = pinned Madina, in a separately audited working dependency environment.
R = reviewed corrected CPU reference, for corrected_v1 only.
B = exact improved CPU algorithm/layout; N = same algorithm in native backend;
G = same semantics with the best validated GPU execution region;
P/C = public parallel and cache configurations of the same numerical profile.

Use M only for behaviorally equivalent Madina workflows. Do not compare aggregate flow to
all-path betweenness as one speedup. Compare bug-fixed optimizations to R under identical work,
and separately report semantic changes relative to A/M. Report failed/stalled references as
censored failures with limits and partial progress, never an infinite speedup.

Measure nested leaf/adapter/preparation/region/public-call/job/batch boundaries. The primary
boundary includes read, validation, topology/snapping, graph, computation, materialization
and all user-required exports. Batch ends after required ordered publication and live-state
commit, not last kernel launch. Record GPU preparation, H2D, launches, compute, D2H,
synchronization, conversions and remaining CPU time. Cold JIT/import/build-cache and warm
resident execution are separate; do not amortize startup over an invented production lifetime.

Profile with counters outside final uninstrumented measurements. Never sum a diagnostic
repeat constructor into application time. Cache hits/misses, OS page-cache conditions and
input hash cost are charged. Hashing for post-run verification is reported separately;
validated-success throughput and application throughput boundaries are both explicit.

## Equal resources and sample selection
Freeze workload content, complete settings, exact route model, outputs, hardware budget and
numerical partition before candidate measurements. First compare the same W/H/queue/writer
settings. Then compare each arm's best validated schedule under the same resource budget.
GPU hardware differences are reported, not hidden behind “equal resources”. Tune at medium
scale, not by mutating the final workload. No workstation background builds during timing.

Screen at most three variants of one algorithm per round. Cheap L0/L1 first; genuine L2,
then L3. Select finalists before held-out data. Final comparison: at least five independent
paired windows per primary family, AB/BA randomized using a recorded seed, each warm window
aiming at >=90 s when resource budget permits. Fixed job counts are chosen from baseline-only
pilot so arms do identical work. Very large individual jobs may constitute a window. Failure
to afford this window is an evidence limitation, not permission to inflate n with per-job
samples from one process. Startup diagnostics use fresh processes and isolated JIT caches.

Use paired log-throughput ratios, report individual ratios, median and a seeded paired
bootstrap interval; preserve exact reduction/quantile method. A five-pair interval is limited
precision, not proof of broad statistical universality. Run baseline-vs-baseline drift checks.
Retain failed/cancelled runs with reason. No post-hoc dropping of inconvenient slow samples.

## Selection and ambitious performance objective
There is no user-supplied absolute throughput target or production hardware. Record that as
unavailable. The engineering objective is continued Pareto improvement over A/R and M where
semantics match, until the measured candidate frontier is exhausted. Do not stop at the old
1.10 gate or a kernel win. A default performance change needs a positive paired improvement
with lower interval bound >1 plus meaningful coverage; preregister a 1.05 median target for
noisy new automatic selectors, with <=1.03 median protected-cell regression. Smaller stable
wins may be retained as separately reviewed micro-improvements but not called primary target
qualification. Mandatory capability implementation does not depend on winning every cell.

Compare cold/warm/no-cache, one job and multi-job, first/sustained GPU runs, small/medium/large,
node/turn state, real paths and edge outputs, and memory-pressure cases. Reprofile composed
candidates; do not multiply overlapping gains. Do not optimize solely to validation data.

## Required raw records
JSONL/CSV numerical samples include full source and wheel hashes, profile/model, input and
settings hashes, output manifest, successful validated job count, failures/cancellations,
wall and component times, process-tree/device memory samples, requested/effective pools,
compile flags, hardware/OS/power state, cache state, timings' start/end definitions, and
sample status. Record exact commands with credentials redacted. Keep the raw numerical
samples and minimized failures in the repository or a user-approved durable artifact store;
checksums alone are not a replacement for bytes. See EVIDENCE.md.
