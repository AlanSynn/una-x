# Total work, movement and 10x feasibility

Let V/E be network nodes/arcs, O origins, D destinations, H independent scratch lanes, K jobs,
S_o original heap/arc work, I_o reverse-incidence work, M_o candidate destinations and R_o kept
destinations. These are descriptors, not measured values until instrumented.

Current access service roughly contains O*(V+D) initialization/scan work + sum S_o + ordered
metrics/sorts. Proposed sparse work contains one structural validation/index build, H private
initializations, and sum(S_o + I_o + M_o log M_o + original metrics on R_o). It retains the full
network, destinations and output extent. Dense or tiny cases can lose; dispatch is essential.

General lower bound:
  T >= max(W/C, critical_path_span, Q_DRAM/B_DRAM, Q_IO/B_IO).
Logical array bytes are not DRAM traffic. Separately record memory hierarchy/cache miss counters
only when measured. Do not infer effective bandwidth from object sizes or OS RSS.

Independent/cohort stage model:
  T_batch ~= sum_j ceil(K_j/N_j)*t_j(N_j,H_j) + T_serial + T_overhead.
Include decoding, fingerprinting, index builds, searches, trace construction, cache validation,
serialization, IPC, worker start/JIT, device copies, metric work, ordered commit and all exports.
Worker count times kernel speedup is not a product of independent gains on the same CPU quota.

Amdahl gate:
  S_total = 1/(1-f + f/s + delta).
For target G=10, the needed stage speed is s >= f/(1/G-1+f-delta), only for positive denominator.
For several disjoint changed regions sum their residual contributions; never count overlap twice.
All modeled gains are hypothetical sensitivity analysis, even when f comes from measurements.

Historical component ratio 48.0138296 implies f>=.9191433 at delta=0 for application 10x. The
reported absolute pair 1.715870517/0.035737006 seconds leaves unchanged service <=.150944495 s
for 10x. These are derived from preserved component measurements, NOT observed application f.
Use new H01/W00 stage observations instead of treating those numbers as a production budget.

For identical routing across K distinct jobs:
  T_repeat = K*(P+S+M+X)
  T_cohort = P_shared + S_shared + M_invariant + sum_k M_changed_k
             + sum_k X_k + T_keys + T_traces + T_leases + T_commit.
Only proven identical work enters shared terms. Exports X_k still exist K times. Whole final
answer caching already exists in the baseline and must be enabled for exact-repeat controls.
If outputs occupy >10% of original wall and remain unchanged, accelerating only compute cannot
reach 10x. Inspect preparation/serialization/publication then; never delete output obligations.

Scratch payload estimate for the prototype, not a memory guarantee:
  per lane ~= 24V + 16D + 16*max_degree bytes plus sort/filter/heap temporaries;
  reverse index ~= 8*(V+1) + 16D plus construction scratch;
  output ~= O*(reach_itemsize + 3*8);
  trace ~= 8*(O+1) + 16*sum R_o plus optional metadata, before packing overhead.
M_live = M_persistent + max_s M_workspace,s + M_queues + M_native + M_device_host_staging.
Charge concurrent producers and consumers, allocator retention, cache bytes AND decoded arrays.
No dense O*D or K*O*D tensor to implement trace reuse. Bound sparse trace size, tile lifetimes,
and outstanding output/state descriptors. mmap does not remove residency/scratch costs.

For GPU include prepare + H2D + launch + kernel + D2H + sync + conversion + remaining service;
separate cold setup J from per-cohort recurrence. CPU+GPU uses the same resource policy and
publication semantics. A kernel-only ratio is never the application goal.
