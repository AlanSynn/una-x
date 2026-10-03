# 02: epoch-tagged private search labels

Scope: initial una_legacy A3 finite/nonnegative input domain. Same original heap/row arithmetic.
Do not replace public compact_vector_node_view_scope's full-array return. Optimize only a private
consumer boundary proven not to expose the dense buffer; materialize required public arrays.

## Admission before allocation/unsafe compilation
Validate array exact type/dtype/shape and lengths: int64 pointers/neighbors/(O,2)/(D,2) terminals,
float64 prepared costs/seeds/terminal costs/weights, bool flags, native byte order, known layouts.
CSR starts at zero, monotonic, final pointer equals arc lengths; every index in range. Check O,D,V,
2D, per-lane allocations and all byte arithmetic for overflow before allocating. Start with no
object/subclass/unknown alias inputs. Empty O/D/V legal only as baseline handles them. Derive
max_degree once, not per origin. Value checks run without unsafe fast-math. Require finite R>=0,
finite nonnegative effective costs, seeds, terminal costs and initial destination weights. A
negative-zero bit is not negative; retain it. For a narrow first implementation also bound
intermediate additions conservatively if the selected compiler's overflow behavior is unproved.
Accept checked int64 or float64 cutoff representations only with their ORIGINAL initializer
promotion/comparison behavior; do not cast an integer cutoff merely to match a prototype.
Compute b using the original typed initializer; require b>R. Use original fallback on refusal.

## State projection and schedule
Each exclusive logical lane owns label[V], stamp[V], touched[V], row_offset[max_degree],
row_cost[max_degree]. Stamp zero initially. Epoch e nonzero. Define logical L[v]=label[v] when
stamp[v]==e, else b. All label READS go through this projection, without accidental loads of
uninitialized label memory. A write records v as touched exactly once in epoch e, sets stamp then
writes the original value; the order of scientific writes is unchanged.

Pseudocode:
  begin_origin(e)
  write(seed_start, w_start); write(seed_end, w_end)  # overwrite even if same node
  reproduce original dummy heap init/pop and end-then-start strict-< push gates
  while original heap nonempty:
    (weight,node) = original_pop()
    eligible_count = 0
    for off in ORIGINAL row offset order:
      c = original_add(cost[off], weight)
      if original_le(c,R) and original_lt(c,read(neighbor[off])):
        record(off,c)                              # NO label writes yet
    for (off,c) in recorded sequence:
      write(neighbor[off],c)
      if original flag and original degree>1: original_push(c,neighbor)
  return private view plus touched sequence to the direct consumer

Inductive proof: initialization projection equals original dense b; seeds match in order; heap
state matches; each row reads identical pre-row labels, produces identical eligible sequence and
replays identical writes/pushes. Both equal parallel candidates can pass snapshot eligibility.
No settled flag, decrease-key rewrite, stale-pop skip, changed tie policy or canceled duplicate
push. Original odd seed-overwrite/leaf behavior is preserved, not 'fixed' in this optimization.

## Ownership and rollover
Use prange over LOGICAL LANES, each sequentially processes its assigned origin indices and writes
only those outputs. Lane ID is not a physical thread ID. No global arena shared across concurrent
calls. First implementation resets stamps per call and uses checked epoch counter. Persistent
arenas require exclusive leases, no use-after-cancel, and reset stamps/destination marks BEFORE
counter rollover, with no active reader; test reduced-width counters to exhaust epochs. Epochs
never used as float. A thrown exception invalidates/releases the lease; next caller sees pristine
state. Charge initialization H*V and lifetime overhead in timing and memory.

Tests: VALIDATION.md plus two concurrent calls, interrupted origin, high-degree scratch boundary,
untouched dense-view reconstruction, same-node unequal seeds, equal parallel improvements, and
all corner cases around cutoff/b. Compare full labels at each chronological event, not only the
last arrays. First implementation uses original compiled fold unchanged. Promote only with
public-path coverage and conservative performance dispatch, not a helper microbenchmark.
