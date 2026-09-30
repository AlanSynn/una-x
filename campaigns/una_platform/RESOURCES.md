# Machine, safety and experiment budgets

Before any heavy execution discover OS/ISA, physical/logical CPUs, affinity/cgroup limits,
available/total RAM and swap, disk free space, accelerator model/driver/memory, build tools,
and existing Python/native pools. Do not copy the previous M1 configuration onto another
host. Heterogeneous efficiency/performance cores are not a portable CPU-affinity guarantee.
Record actual scheduler behavior and effective worker/thread counts.

Default local campaign envelope until explicitly amended: one heavy owner, CPU budget bounded
by discovered limits with at least one logical core left for system responsiveness when possible;
RAM admission <= min(0.65*physical RAM, 0.75*available-at-admission); disk reserve >= max(5 GiB,
0.10*filesystem capacity). Host/device allocations include queues and runtime. GPU workspace
<=0.70*currently free device memory initially, refined only by recorded measurements. These
are conservative policy choices, not measured hardware capacities. Do not allocate by assuming
all nominal RAM/device memory is available. One worker must be measured before adding others.

Default heavy-wall budget is 21,600 s for the first campaign round, including compilation,
profiling and screens; time is charged once per exclusive machine lease. No paid cloud or
new remote hardware may be provisioned without explicit permission. Implementation and cheap
unit tests can continue when a performance budget is exhausted. Stop with a resumable blocked
qualification record rather than declaring unavailable mandatory tests passed. Budget extension
is a resource decision, never a numerical/semantic gate relaxation.

A central admission table accounts for Python processes, Numba/OpenMP/BLAS/native/flow pools,
writers, cache producers and devices. Sum of concurrently active CPU work must fit C;
logical stripe count is numerical and must NOT be reduced to fit physical threads.
Use bounded queues and scratch by lifetime. Share immutable data through owned snapshots,
not mutable globals. Avoid fork after initializing threaded/JIT/device runtimes; use spawn.

Supervise risky reference/path/nonfinite tests in disposable subprocesses with wall, progress,
output-byte and memory limits. Log visited/popped/emitted counts so slow combinatorics differs
from a deadlock. On breach cancel admitted work, join owned descendants, keep failed status
and resumable/checkpoint metadata. Never SIGKILL unrelated user processes or delete user data.
Watch pressure and disk as well as process RSS. Backend/device faults invalidate uncommitted
results and poison the affected execution context until reset.

Keep numerical raw evidence compact; don't regenerate identical export files thousands of
times in tracked Git. Preserve sufficient raw samples, representative complete artifacts and
minimized failures plus replay manifests. Do not repeat the previous purge of decisive raw
session bytes. Cleanup requires a manifest proving durable retention and explicit authority.
