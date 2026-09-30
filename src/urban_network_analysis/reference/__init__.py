"""corrected_v1 reference profile (campaign una-platform-2026-09, SCIENCE).

An independent, reviewed scalar/graph reference for the declared corrected
semantics (NUMERICS.md; dossier 02).  This package is deliberately NOT a
fork of the engines: it re-derives the intended quantities from the model
rules with explicit binary64 arithmetic, so corrected outputs have a
reference that is reviewable line-by-line and independent of the compiled
kernels it may later be compared against.

Profile rules implemented here (NUMERICS.md "Floating-point implementation
is part of the reference"):

- binary64 throughout; every elementary operation is one IEEE-754
  round-to-nearest step in a documented order.
- Scalar accumulations run left-to-right in a declared index order
  (destination index, adjacency offset, or stripe ID).  ``np.sum`` /
  ``np.dot`` / BLAS pairwise or tree reductions are never used where the
  order is part of the reference; loops use plain float adds.
- No unsafe fast-math, no FMA contraction, no reassociation: pure-Python
  float ops cannot contract, and nothing here enables fastmath.
- Transcendentals are the platform libm implementations via numpy scalars
  (``np.exp`` on binary64), retained and documented per NUMERICS ("start
  by retaining trusted CPU math-sensitive coefficients/decays").  Any
  future portable replacement requires independent bit-level validation
  and versions this profile.
- Invalid cost domains are rejected with typed errors before any search
  (errors.py): NaN/inf costs, negative costs, malformed CSR, out-of-range
  terminals.  Legacy profiles keep their pinned behavior on such inputs;
  corrected_v1 refuses them.
- Multi-origin aggregation uses the frozen canonical logical partition
  (partition.py): min(32, number_of_origins) stripes, origin i -> i mod L,
  each stripe processed in increasing origin order, final fold in
  increasing stripe ID -- the scalar reference implements the same
  partition with one physical worker.  PARTITION_VERSION versions this
  contract; any change invalidates numerical caches.

Legacy profiles are untouched: una_legacy and madina_legacy keep their
pinned bits on valid inputs (including their recorded defects, retained
as legacy behavior).  Deltas are declared in the task evidence and, at
reconciliation, in the packet bug_deltas.json.
"""

# Profile identity (EXECUTION/backends plumb this string; it is part of
# every numerical cache key that stores corrected_v1 outputs).
PROFILE_NAME = "corrected_v1"

# Declared arithmetic profile (see module docstring).
ARITHMETIC = "binary64-explicit-order"
TRANSCENDENTALS = "platform-libm-via-numpy"   # retained, not replaced
ACCUMULATION = "ordered-left-to-right"

# Frozen canonical logical partition.  Bumping this invalidates every
# numerical cache derived from a corrected_v1 aggregation.
PARTITION_VERSION = "canonical-min32-imodL-v1"

__all__ = [
    "PROFILE_NAME",
    "ARITHMETIC",
    "TRANSCENDENTALS",
    "ACCUMULATION",
    "PARTITION_VERSION",
]
