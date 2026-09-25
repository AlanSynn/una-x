# Claude Code Execution Packet Authoring Guide for Astra

## Purpose

Use this guide when Astra is responsible for high-level reasoning, architecture, optimization strategy, mathematical equivalence analysis, and execution planning, while Claude Code is responsible for implementation, testing, debugging, benchmarking, independent review, integration, and producing the final candidate.

The packet must be detailed enough that Claude Code can complete the work without repeatedly returning to Astra for ordinary implementation decisions.

The objective is not to create another planning document that Claude Code then has to reinterpret.

The objective is to transfer enough design authority, proof obligations, execution rules, benchmark discipline, and failure policy into a structured packet that Claude Code can execute autonomously.

A good packet converts high-level reasoning into a bounded engineering program.

---

# 1. Division of Responsibility

Astra owns high-level reasoning.

Claude Code owns execution.

Astra should determine:

| Astra responsibilityRequired output |                                                         |
| ----------------------------------- | ------------------------------------------------------- |
| Problem definition                  | Exact objective and non-objectives                      |
| Semantic contract                   | What must remain unchanged                              |
| Architecture                        | Execution model and subsystem boundaries                |
| Mathematical reasoning              | Equivalence arguments and optimization candidates       |
| Performance theory                  | Work, span, memory, I/O, throughput models              |
| Optimization portfolio              | Ranked experiments and rejection conditions             |
| Resource policy                     | CPU, RAM, accelerator, worker limits                    |
| Validation strategy                 | What must be proven at each scale                       |
| Promotion policy                    | What evidence permits a default change                  |
| Packaging/DX                        | What users may or may not be required to do             |
| Termination policy                  | When to integrate, reject, or close the research track  |
| Merge target                        | What source state will eventually be reviewed for merge |

Claude Code should determine ordinary implementation details that remain inside those boundaries.

Examples include local helper names, buffer implementation details, narrow refactors, fixture construction, debugging of ordinary failures, compiler invocation details, and repairing tests caused by its own patch.

Claude Code should not ask Astra for approval after each step.

Escalation is appropriate only when implementation reveals that a high-level assumption is false or that completing the task requires changing a frozen semantic, scientific, numerical, user-facing, safety, or resource contract.

---

# 2. First Rule: Audit the Current Repository Before Writing the Packet

Never author a performance or implementation packet solely from an old handover.

Astra must inspect the actual target branch first.

Record:

| ItemExample                              |                               |
| ---------------------------------------- | ----------------------------- |
| Repository                               | `owner/project`               |
| Integration branch                       | `perf/optimization`           |
| Current branch SHA                       | immutable commit              |
| Main/base SHA                            | immutable commit              |
| Merge base                               | immutable commit              |
| Dirty state policy                       | preserve user changes         |
| Existing optimization tracks             | accepted, rejected, inactive  |
| Existing benchmarks                      | measured scope                |
| Existing scientific/numerical exceptions | frozen                        |
| Existing packaging behavior              | wheel/source/install behavior |
| Existing CI state                        | required vs optional checks   |

Then distinguish three kinds of information:

**Source facts:** directly visible in current code.

**Measured evidence:** produced by an actual execution under recorded conditions.

**Hypotheses:** plausible but not yet tested.

Never convert a hypothesis into a source fact merely because it appeared in a previous handover.

Never convert a microbenchmark into an end-to-end claim.

Never convert a synthetic workload into the actual target workload.

---

# 3. Freeze the Semantic Contract Before Optimizing

The packet must state what optimization is not allowed to change.

At minimum capture:

| Contract classQuestions |                                                                           |
| ----------------------- | ------------------------------------------------------------------------- |
| API                     | Which public functions, CLI commands, parameters and defaults are frozen? |
| Inputs                  | File schemas, layouts, metadata, coordinate systems, masks                |
| Outputs                 | Files, bands, dtypes, names, metadata, ordering                           |
| Numerical               | dtype promotion, operation order, rounding, NaN/Inf, signed zero          |
| State                   | chronological dependencies, carried state, mutations                      |
| Geometry                | resolution, ray support, neighborhood context, edge semantics             |
| Physics/domain          | equations, components, patch count, timesteps                             |
| Errors                  | required failure and warning behavior                                     |
| Persistence             | checkpoint, durability, atomicity, recovery                               |
| Cache                   | identity, invalidation, provenance                                        |
| Resources               | memory limit, CPU budget, accelerator policy                              |
| DX                      | install command, import path, CLI use, optional dependencies              |

Explicitly separate:

```math
\text{mathematical equivalence}
```

from

```math
\text{floating-point execution equivalence}
```

from

```math
\text{domain/scientific equivalence}
```

from

```math
\text{artifact/API equivalence}.
```

An algebraically equivalent expression is not automatically numerically equivalent.

---

# 4. Performance Analysis Must Precede Framework Selection

Do not begin with:

> Use CUDA.

> Rewrite with MLX.

> Port to ISPC.

> Fuse all kernels.

Begin by modeling total service time.

A useful lower-bound model is:

```math
T \ge \max \left( \frac{W}{C}, S, \frac{Q_{\mathrm{memory}}}{B_{\mathrm{memory}}}, \frac{Q_{\mathrm{IO}}}{B_{\mathrm{IO}}} \right)
```

where:

`W` is computational work.

`C` is effective compute capacity.

`S` is the critical-path span.

`Q` is actual data movement.

`B` is effective bandwidth.

For multiple independent jobs:

```math
T_{\mathrm{batch}} \approx \sum_j \left\lceil \frac{K_j}{N_j} \right\rceil t_j(N_j,H_j) + T_{\mathrm{serial}} + T_{\mathrm{overhead}}.
```

Do not multiply worker speedup, SIMD speedup, kernel speedup, and GPU speedup unless they consume independent resources and remove disjoint costs.

---

# 5. Search for Less Work Before Faster Work

The packet should instruct Claude Code to examine optimization opportunities in roughly this order.

### Dead computation

Construct a backward dependency graph from observable outputs, public returns, future state, neighbor reads, required side effects, errors and mutations.

Anything unreachable may be removable.

### Duplicate computation

Find identical computations repeated across pixels, patches, directions, timesteps, tiles, stages, cache identities, export paths, or diagnostic calculations.

Memoization requires complete input identity.

### State reduction

A rich recurrence may only expose a smaller state.

Given:

```math
s_{n+1}=F(s_n,x_n)
```

find a projection `\pi` such that:

```math
\pi(F(s,x)) = F'(\pi(s),x).
```

Then the smaller state may replace the original if all future observations depend only on `\pi(s)`.

### Absorbing states

Find states satisfying:

```math
F(s^\*,x)=s^\*
```

for every valid future input.

If outputs are also invariant, remaining work may terminate exactly.

### Conservative bounds

Use upper/lower bounds to prove that the remainder of a search, ray, traversal, or iterative process cannot change the result.

### Exact finite-state precomputation

If an expensive expression depends on a small exact state space:

```math
y=F(c,s),\qquad |S|\ll N,
```

compute exact values once:

```math
L[c,s]=F(c,s).
```

Retain original accumulation order.

### Dataflow elimination

Remove intermediate arrays when their producer and consumer can operate in one bounded region without changing semantics.

---

# 6. Memory Movement Is an Algorithmic Cost

The packet should require byte-level reasoning.

A transformation such as:

```text
decode
→ write N×P temporary
→ read N×P temporary
→ classify
→ write another temporary
→ reduction

```

should be compared against:

```text
decode bounded tile
→ classify
→ accumulate
→ discard

```

when exactness permits.

For every significant buffer record:

| PropertyRequired information |                                          |
| ---------------------------- | ---------------------------------------- |
| Producer                     | Which stage allocates/fills it           |
| Consumer                     | Who reads it                             |
| Lifetime                     | first write to last read                 |
| Size                         | actual bytes                             |
| Ownership                    | thread/process/device                    |
| Reuse                        | number of reads                          |
| Layout                       | consumer-friendly or conversion required |
| Persistence                  | transient/cache/checkpoint               |
| Alternative                  | fuse, recompute, pack, stream            |

Memory should be modeled by overlapping lifetimes:

```math
M_{\mathrm{live}} = M_{\mathrm{persistent}} + \max_s M_{\mathrm{workspace},s} + M_{\mathrm{queues}} + M_{\mathrm{runtime}}.
```

Do not sum peak buffers from stages that never coexist.

---

# 7. Apply FlashAttention-Like Thinking Beyond ML

Use I/O-aware optimization whenever memory traffic dominates.

The transferable principle is:

> Cheap computation can be preferable to expensive data movement.

Consider:

```math
C_{\mathrm{recompute}}
```

versus

```math
C_{\mathrm{store}} + C_{\mathrm{reload}} + C_{\mathrm{capacity\ pressure}}.
```

A cheap value may be recomputed if this eliminates large temporary storage.

Preferred structure:

```text
slow memory
→ bounded tile
→ registers/cache/shared memory
→ several operations
→ final accumulation
→ write required result

```

Avoid giant materializations when downstream operations immediately consume them.

---

# 8. CPU and GPU Must Be Treated as Alternative Schedules

The packet should not assume CPU-only or GPU-first.

For each stage classify:

| CriterionQuestion    |                                                    |
| -------------------- | -------------------------------------------------- |
| Element parallelism  | How many independent outputs?                      |
| Recurrence           | Does each output require ordered inner operations? |
| Arithmetic intensity | FLOPs per byte                                     |
| Reuse                | Are coefficients/state reused heavily?             |
| Divergence           | Do lanes execute similar paths?                    |
| Residency            | Can state remain on device?                        |
| Transfer             | How much crosses the host/device boundary?         |
| Precision            | Are FP64 or exact math semantics required?         |
| Output volume        | How much returns to host?                          |

For GPU evaluation use:

```math
T_{\mathrm{GPU}} = T_{\mathrm{prepare}} + T_{\mathrm{H2D}} + T_{\mathrm{launch}} + T_{\mathrm{kernel}} + T_{\mathrm{D2H}} + T_{\mathrm{sync}} + T_{\mathrm{remaining}}.
```

A GPU candidate is useful only if the full region wins.

Prefer GPU residency regions:

```text
upload
→ stage A
→ stage B
→ timestep 1
→ timestep 2
→ ...
→ final required download

```

rather than repeated CPU/GPU round trips.

---

# 9. Preserve Ordered Numerical Recurrences

For floating-point recurrence:

```math
a_{p+1} = RN(a_p+c_p),
```

do not automatically parallelize the `p` reduction axis.

Instead parallelize independent outputs:

```text
for p in original order:
    parallel/SIMD over x:
        a[x] = original_update(a[x], contribution[x,p])

```

This pattern is especially useful for CPU SIMD, ISPC, GPU work-items, CUDA, Metal, OpenCL, SYCL, Dr.Jit, or other execution backends.

If a tree reduction or reassociation is desired, it must be treated as a distinct numerical profile rather than an invisible optimization.

---

# 10. Framework Evaluation Must Separate Algorithmic Gains

Always construct three comparisons.

| ArmMeaning |                                                  |
| ---------- | ------------------------------------------------ |
| A          | Current accepted implementation                  |
| B          | Improved algorithm/layout using existing backend |
| C          | Same B algorithm/layout using new backend        |

If:

```math
T_B \le T_C,
```

the new framework may not be worth adopting.

This prevents attributing a layout or fusion improvement to CUDA, MLX, ISPC, OpenCL, Dr.Jit, Triton, or another framework.

---

# 11. Backend Selection Guidance

The packet may nominate backends, but should not mandate every one.

| Backend classSuitable workloads |                                                  |
| ------------------------------- | ------------------------------------------------ |
| Numba/LLVM                      | existing typed CPU loops                         |
| ISPC                            | CPU SIMD across independent outputs              |
| Highway / portable SIMD         | architecture-specific CPU control                |
| Dr.Jit LLVM                     | traced loops and divergent element-wise programs |
| Halide                          | images, stencil, regular pipelines               |
| MLX                             | regular array graphs, Apple-oriented execution   |
| OpenCL                          | portable accelerator kernels                     |
| SYCL                            | cross-vendor C++ heterogeneous execution         |
| CUDA/Triton                     | NVIDIA-dominant deployment                       |
| Metal                           | Apple GPU execution                              |
| C/C++ extension                 | stable native kernel interface                   |
| Pythran/Cython                  | narrow Python numerical regions                  |

Do not keep testing frameworks merely because they are available.

Each experimental backend must have a termination criterion.

---

# 12. Parallel Claude Code Execution

Parallel inference can be broad.

Parallel modification of the same state cannot.

Separate model concurrency from machine concurrency.

### Agent concurrency

Claude Code may use many independent agents when useful.

Typical roles:

| RoleResponsibility    |                                 |
| --------------------- | ------------------------------- |
| Coordinator           | task DAG and completion         |
| Source auditor        | current code and evidence       |
| Numerical proof agent | equivalence and counterexamples |
| Implementation worker | bounded code scope              |
| Fixture worker        | tests and adversarial cases     |
| Runtime worker        | scheduling/memory/concurrency   |
| Packaging worker      | wheel/install/DX                |
| Performance owner     | benchmark execution             |
| Independent reviewer  | patch and evidence review       |
| Integrator            | shared source and final branch  |

The strongest reasoning model should be used for difficult proof, architecture, numerical debugging, cross-cutting review, and final audit.

Use cheaper workers for bounded mechanical tasks where appropriate.

### File ownership

At any moment, one agent owns a writable code scope.

Shared API, dispatch, package schema, central pipeline, and merge state should have a single integrator.

Alternative implementations belong in isolated worktrees.

### Worktrees

Workers should start from an immutable verified SHA.

Preferred structure:

```text
integration branch
    ├── detached worker worktree A
    ├── detached worker worktree B
    ├── detached reviewer worktree
    └── benchmark owner worktree

```

Do not allow workers to accidentally start from an outdated default branch.

### No polling

Do not repeatedly ask workers for status.

A worker should execute:

```text
implement
→ test
→ debug
→ repair
→ retest
→ produce evidence
→ completion report

```

and report only completion or a genuine blocker.

### No recursive explosion

Workers normally should not recursively create uncontrolled worker trees.

The coordinator owns task decomposition.

---

# 13. Machine Resource Admission

Unlimited model tokens do not imply unlimited local execution.

The packet must define resource ownership.

| ResourceRule     |                                                         |
| ---------------- | ------------------------------------------------------- |
| Inference agents | broad parallelism allowed                               |
| CPU-heavy tests  | bounded globally                                        |
| RAM              | reserve before spawning                                 |
| Compiler builds  | bounded                                                 |
| Disk-heavy tests | bounded                                                 |
| Benchmark host   | exclusive performance owner                             |
| GPU              | one benchmark owner unless concurrency itself is tested |

Do not benchmark while other workers are compiling or profiling on the same machine.

If measuring CPU budget `C`:

```math
\sum_i (\mathrm{workers}_i \times \mathrm{threads}_i) \le C
```

unless oversubscription is itself the experiment.

---

# 14. Benchmark Discipline

The packet must define the benchmark boundary before results exist.

Record:

| FieldRequired      |                                    |
| ------------------ | ---------------------------------- |
| Commit             | immutable SHA                      |
| Workload           | exact manifest                     |
| Resource budget    | CPU/RAM/GPU                        |
| Worker count       | requested and admitted             |
| Threads            | requested and effective            |
| Block/tile size    | exact                              |
| JIT state          | cold/warm                          |
| Cache state        | cold/warm                          |
| Compiler version   | exact                              |
| Backend            | exact                              |
| Start boundary     | explicit                           |
| End boundary       | explicit                           |
| Synchronization    | included                           |
| Conversion/packing | included                           |
| Output/checkpoint  | included if production requires it |
| Memory             | process tree, not only parent      |
| Raw timings        | retained                           |

Lazy or asynchronous frameworks must synchronize before stopping the timer.

Do not use profiler time as the only runtime measurement.

---

# 15. Measure the Real Execution Boundary

A common failure pattern is:

```text
fast kernel
+ expensive producer
+ expensive packing
+ expensive guard
= slower application

```

Therefore measure nested boundaries independently:

```text
kernel leaf
adapter
producer + adapter
stage
small pipeline
full workflow

```

A kernel win is useful diagnostic evidence.

It is not an end-to-end speedup claim.

---

# 16. Pre-Register Selection Gates

Before candidate performance results exist, define the selection logic.

Example:

```math
S_{\mathrm{total}} = \frac{1} {1-f+f/s+\delta}.
```

Use this to reject kernels whose maximum possible impact is too small.

A promotion policy should specify:

| GateExample |                                         |
| ----------- | --------------------------------------- |
| Numerical   | exact required fields                   |
| Local speed | candidate > A and strongest B           |
| End-to-end  | minimum pipeline improvement            |
| Regression  | maximum allowed protected-cell loss     |
| Coverage    | actual optimized execution fraction     |
| Memory      | bounded peak                            |
| First use   | no unacceptable cold regression         |
| DX          | no new user action                      |
| Packaging   | installed wheel actually uses candidate |
| Platform    | explicit OS/ISA scope                   |

Do not loosen gates after candidate failure.

If measurement environment constraints require changing a measurement window, record the amendment before candidate results are observed.

---

# 17. Validation Tiers

Use progressively expensive tests.

| TierTypical use |                                               |
| --------------- | --------------------------------------------- |
| L0              | algebra, state machines, bit patterns, guards |
| L1              | direct baseline vs candidate kernel           |
| L2              | small genuine end-to-end chronology           |
| L3              | medium performance and resource selection     |
| L4              | final target workload only                    |

Small development arrays are allowed.

Reduced final scientific workloads are not.

Preserve patch counts, chronological semantics, physical equations and boundary rules in final qualification.

Do not run 1024-scale or production-scale workloads on every patch.

---

# 18. Evidence Reuse

A previous passing result may be reused only if its relevant dependency closure is unchanged.

Check:

```text
source
fixture
reference
compiler/math profile
runtime policy
thread policy
backend
test harness

```

If all relevant components are unchanged, rerunning is unnecessary.

If a shared helper changes, invalidate affected descendants.

Docs-only changes do not invalidate numerical evidence.

---

# 19. Independent Review

Implementation and review must be separate.

A reviewer should inspect:

```text
source delta
proof obligation
tests
raw evidence
claimed result
failure paths
fallback
performance boundary

```

The reviewer does not need to rerun every successful command.

Repeat execution only when evidence is missing, suspicious, environment-dependent, or central to the claim.

Required corrections return to the implementation owner.

---

# 20. Evidence Ledger

Every optimization track should produce machine-readable and human-readable records.

Recommended evidence classes:

```text
inventory/
contract/
probes/
trials/
reviews/
selection/
integration/
installed/
freeze/
handover/
final/

```

Performance records should state whether a number is:

**measured**

**derived from measured data**

**hypothetical**

**unavailable**

Never mix them.

Rejected experiments should remain documented so future agents do not rediscover them.

---

# 21. Packaging and Developer Experience Are Part of Performance Engineering

A fast backend is not production-ready if users must install a compiler manually when the previous package did not require one.

Freeze the DX contract.

Record:

| User-visible propertyRequirement |                                          |
| -------------------------------- | ---------------------------------------- |
| Package name                     | unchanged unless explicitly intended     |
| Import name                      | unchanged                                |
| CLI                              | unchanged                                |
| Function signatures              | unchanged                                |
| Defaults                         | unchanged                                |
| Input formats                    | unchanged                                |
| Output formats                   | unchanged                                |
| Additional compiler              | not required for ordinary binary install |
| Runtime download                 | not hidden                               |
| Backend env variable             | optional expert override, not required   |
| Unsupported systems              | trusted fallback                         |

For native binary delivery, use correct platform wheel tags.

Source installations may fall back to the portable backend when binaries are unavailable.

Do not count fallback execution as proof that native deployment works.

---

# 22. Automatic Backend Selection Must Fail Closed

A default selector should not simply use:

```text
native available → native

```

Use a qualified registry or equivalent policy.

Conceptually:

```text
capability
+ admitted input profile
+ qualified OS/ISA
+ qualified artifact
+ performance-certified workload domain
→ optimized path

otherwise
→ trusted fallback

```

Missing or malformed performance qualification should not silently activate experimental code.

Failures after an admitted optimized execution begins should normally fail loudly rather than hide a broken backend behind fallback, unless the contract explicitly defines otherwise.

---

# 23. User Overrides Must Remain Separate From Automatic Policy

It is often useful to retain an expert override such as:

```text
BACKEND=native
BACKEND=numba

```

but this is not the same as automatic default selection.

Automatic selection requires promotion evidence.

Expert routes may intentionally execute unqualified code for development and diagnostics.

Never use an expert-route success to claim default qualification.

---

# 24. GPU Promotion Has Additional Requirements

For GPU candidates, record:

```text
H2D bytes
D2H bytes
device-resident bytes
global-memory traffic
kernel launches
sync points
device setup
compile time
warm execution

```

The GPU should win over the strongest relevant CPU baseline at the complete execution region.

A device-resident multi-stage region may be worthwhile even when one individual GPU kernel is slower.

Do not compare GPU kernel time against CPU full-stage time.

---

# 25. Branch Policy

A typical packet should state one named integration branch.

Workers may use detached worktrees.

Only the integrator writes reviewed changes to the named integration branch.

Unless explicitly authorized, do not:

```text
push
create PR
merge main
release
rewrite history
rebase user work
force-clean dirty worktrees
modify credentials
change global provider configuration

```

Preserve the user's work.

---

# 26. CI Policy

Optimization development should primarily use targeted local validation.

Do not trigger broad hosted CI after every experiment.

Use:

```text
relevant unit/kernel tests
→ family tests
→ small integration
→ installed-wheel gate when packaging settles
→ required CI near merge/release

```

Optional CI may use concurrency cancellation.

Never skip required checks and report them as passed.

A skipped or unavailable check remains skipped or unavailable.

---

# 27. Final Decision Must Be Finite

Every optimization campaign requires a termination policy.

Useful states include:

| StateMeaning       |                                                               |
| ------------------ | ------------------------------------------------------------- |
| `qualified`        | candidate meets promotion gates                               |
| `cpu_only`         | accelerator/native candidate rejected, CPU candidate retained |
| `improvement_only` | some optimizations retained but main objective not met        |
| `blocked`          | external requirement unavailable                              |
| `no_safe_merge`    | unresolved correctness/DX issue                               |
| `merge_ready`      | exact reviewed SHA can be considered for main                 |

Avoid perpetual states such as:

> There are more ideas to try.

A final attempt should specify exactly which remaining hypotheses may be tested.

If those fail, close the research track for that release.

---

# 28. Merge Preparation

The execution packet should end by producing a merge manifest, not merely a handover paragraph.

A merge manifest should identify:

```text
source branch
final source SHA
target main SHA
merge base
active default path
optional paths
files included
research-only files excluded
exact test commands
installed-wheel result
performance evidence
known limitations
unverified claims
actual-target status
review status
CI status
recommended merge disposition

```

Test the proposed merge in a disposable worktree before declaring it merge-ready.

The packet should not automatically merge unless the user explicitly authorizes the merge action.

---

# 29. Cleanup Research Scaffolding Before Main

Optimization branches often accumulate:

```text
experimental dispatchers
empty qualification registries
inactive backends
duplicate source copies
benchmark-only adapters