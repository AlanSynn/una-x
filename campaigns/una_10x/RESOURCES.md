# Shared resource and lifetime policy

Discover the actual execution host, not the author's container or the old laptop: CPU model,
ISA, physical/logical cores, scheduler/cgroup/affinity quota, available/total RAM, swap, disk,
filesystem, native/GPU runtime, device memory, interconnect and power/thermal state when relevant.
Do not claim P-core pinning because W*H equals the P-core count. Effective affinity is evidence.

Initial admitted CPU C = usable scheduler entitlement; for a nonexclusive interactive host start
with min(8, usable entitlement) until the performance owner explicitly records a smaller/larger
safe allocation. All candidates and baselines receive equal C. Charge Numba, BLAS/OpenMP,
AggregateFlow ThreadPoolExecutor, I/O/hash workers and process orchestration separately. Prevent
nested pools from multiplying a nominal budget. Set env before imports. Use spawn; do not fork
an initialized JIT/thread/device runtime. A logical scratch lane is not a hardware-thread ID.

Host working budget policy: min(explicit scheduler/user hard cap, 0.60*available RAM at admission),
with a process-tree supervisor and safety headroom. Abort heavy launches when available RAM is
below max(1 GiB, 10% total), or disk headroom is inadequate for bounded outputs/checkpoints.
Record limits and measured peaks in bytes. A historical estimate is not a safe launch permit.
Never solve an OOM by shrinking one arm's scientific problem. Record capacity failure for that
frozen case. On laptop pressure, checkpoint and continue small independent correctness work.

Before W>1, measure W=1 peak across input/trace/output phases. Calculate W only from overlapping
lifetimes; add shared resident arrays once and private arenas per worker. Cap outstanding jobs,
trace tiles, result descriptors and writer concurrency separately. Leased data cannot be evicted
while readers use it; failed workers release leases or are reaped using explicit ownership.

Initial trace resident allowance: at most 25% of admitted host working budget and included in,
not added to, that budget. Cache memory plus decoded arenas cannot each independently consume the
same quota. Queue descriptors have byte and count caps. Whole-project state rehydration is timed
and charged. Avoid unbounded Python lists of per-origin arrays; pack bounded contiguous segments.

GPU: charge static data + state + scratch + queues + runtime + staging. Keep declared device
margin (initial policy 20%). Auto selection requires measured full-region crossover and exact
capability. Missing FP64/profile support refuses, never substitutes FP32. Synchronize before
results/leases are released. No paid compute or remote hardware provisioning without permission.

Heavy tasks: one owner, aggregate initial 28,800 s host wall cap for this campaign, at least
7,200 s reserved for verification/confirmation/review support. Profiling is outside final timing.
Two screening pairs per admitted candidate/config, at most four configs in first sweep, at most
three final candidates including baseline. At most two structural variants per principal track.
Extensions are explicit recorded continuation, not unbounded CI. Multi-agent tokens do not create
extra physical CPU/RAM/I/O. Use representative L0-L3 gates before any final actual-target L4.
