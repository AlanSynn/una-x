# A3: private scope initialization, not public state shrinkage

## Admission first
The old ~1% one-origin probe is not universal. Reprofile multiple actual origins over larger V and D while recording local expansion work, with the SAME obstacles/elevation/cutoff as the full workload. Admit only if private allocation/init is at least 5% of complete-job service or causes measured capacity pressure. Otherwise stop not_admitted with the measured upper bound.

This task has TWO increasing-risk options but at most ONE design may be selected after proof review. Prefer the smaller change that covers the measured cost. Do not implement both and combine unreviewed savings.

Option T (preferred): remove the unobserved D tail ONLY in a new private integration routine. The public compact_vector_node_view_scope, o_scope, OD_Matrix and all exposed graph/results remain unchanged. Every reader of the private vector must be enumerated. A2's scope[V] sentinel read is an explicit dependency: either retain that tail for the A2 composition or replace it with an independently verified typed sentinel production before combining. Do not blindly change shape and leave that read out of bounds. Unused local n_count calculations do not alone prove all consumers are absent.

Option E (only if reset dominates): reusable label workspace and generation/touched metadata per physical owner. No concurrent search may share a mutable workspace. A private read of unvisited node returns the exact original typed sentinel b, never mathematical R+1 or np.inf. Writes mark validity and append reset targets. Terminal assignments and every relaxation use the same accessor semantics; hidden direct array reads invalidate the proof.

Projection: pi(workspace,marks,epoch)[v] = workspace[v] when marks[v]==epoch, otherwise b. Prove pi(candidate_initial_state)=baseline_initial_state; each baseline label read equals projected read; each assignment updates the same v/value; heap operations and eligibility snapshots match; every final consumer observes only pi. For snapshot rows the marks/labels must not change during phase one. A materialized public array requires full initialization and may erase benefit; do not claim private saving for that public path.

Epoch rollover: use an explicit integer dtype/maximum, check before increment, reset the full marks array at a search boundary, then restart epoch at a nonzero value. Tests force rollover using a small injected counter limit; production dtype remains unchanged. Partial failure resets/invalidates the workspace before reuse. A thrown exception must not lead to a later origin seeing residual values. No memset race while another search is active.

Ownership can be per-origin for option T or per-fixed sequential chunk of origins for option E. If changing scheduling chunks, preserve independent-origin mapping and original output indexing; all results are required. Never share one workspace by thread ID without proving the runtime's task/thread lifecycle, recursion and nested calls. No new package-level global cache.

Tests: all A1/A2 states, projected vector at every pop, origins that share terminals, disjoint consecutive regions, all-unvisited/empty cases, capacity fallback, early return/error then another origin, rollover, read-only input arrays, and concurrent owners. Actual compiled bit equality and LLVM/math flags checked. Refuse nonfinite/negative unsupported inputs rather than treating inf sentinel as equivalent.

Benchmark scratch lifetime plus complete installed calls. Clearing marks or constructing reverse-index state O(V) can replace the very work being removed: count it. Reject if branch/indirection, final materialization, rollover overhead or per-owner memory negates the gain. Do not touch global graph topology, physical domain, public output extent or chronological semantics.
