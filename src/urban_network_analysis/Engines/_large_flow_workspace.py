"""Private byte-budgeted workspace policy for AggregateFlow's
destination-gradient precompute (F3I, spec C1/C2).

ONE pure chunk-selection helper plus the frozen policy constants.  The
module is imported module-privately by Engines.AggregateFlow only; it
is exported through no package __init__ and performs no I/O.

Purity contract (spec C2, dossier line 13): ``select_chunk`` takes
only integer arguments and returns an integer or the NO_FIT constant.
It performs no environment query of any kind (no free-RAM, loadavg,
cgroup, psutil, clock or randomness), no attribute writes, no I/O and
no import; identical integers always yield the identical result.

Selector shape (proof section 3, ruling e0827fc0): the caller
re-invokes ``select_chunk`` at EVERY slice with the ACTUAL retained
parts bytes (model (b), per-chunk re-invocation).  ``budget_k`` is
signed integer arithmetic and CAN go negative; the NO_FIT test
dominates the ``c_k`` computation: whenever
``budget_k < per_source`` -- explicitly including every negative
``budget_k`` -- NO_FIT is returned and no ``c_k`` value (in
particular no negative floor result and never 0) escapes the
selector.

The literal ``8`` in the dossier's per_source pseudocode
("V'*(8+pred_itemsize+mask_itemsize)") is the float64 distance
itemsize of the pinned call; this implementation charges the ACTUAL
returned ``dist_itemsize`` per the spec C2 binding terms ("the three
itemsizes as RETURNED by SciPy"), which equals 8 at the pinned
float64 CSR.  The caller verifies the returned itemsizes after each
SciPy call and reverts to the original formula (the ruled fallback)
on any deviation, so the sizing contract is never trusted blindly.
"""

# NO_FIT: explicit refusal signal (spec C2: never 0, never a silent
# clamp).  Distinct from every legal slice width c >= 1.
NO_FIT = -1

# Fixed 64 MiB safety margin charged at every selection (proof section
# 3(d): value 67,108,864 B; covers the per-row temporary surge, SciPy's
# internal workspace by declaration -- measured at L0 against the
# margin/2 gate -- and allocator granularity/fragmentation).
MARGIN_BYTES = 64 * 1024 * 1024

# Internal policy caps (dossier line 36: "at most three preselected
# workspace caps (policy choices 64/128/256 MiB)"; PD1 default 256 MiB).
# Not a Settings field; no cap change after freeze.
CAP_64 = 64 * 1024 * 1024
CAP_128 = 128 * 1024 * 1024
CAP_256 = 256 * 1024 * 1024
ADMITTED_CAPS_BYTES = (CAP_64, CAP_128, CAP_256)
DEFAULT_CAP_BYTES = CAP_256


def select_chunk(v_prime, remaining, dist_itemsize, pred_itemsize,
                 mask_itemsize, fixed_live_base, retained_parts_bytes,
                 row_temporaries, source_index_cost, margin_bytes,
                 cap_bytes):
    """Return the next slice width c (1 <= c <= remaining), or NO_FIT.

    Pure integer arithmetic (proof section 3, exact form::

        per_source = V'*(dist_itemsize + pred_itemsize + mask_itemsize)
                     + row_temporaries + source_index_cost
        budget_k   = cap_bytes - fixed_live_base - retained_parts_bytes
                     - margin_bytes
        c_k        = min(remaining, budget_k // per_source)
        NO_FIT     when budget_k < per_source   (signed test; negative
                   budgets included -- the NO_FIT branch dominates the
                   c_k arithmetic, so no negative floor result, no 0,
                   and no silent clamp can escape)

    The integer floor makes ``retained_parts_bytes + c*per_source +
    margin_bytes <= cap_bytes`` hold by construction at every slice
    (proof section 3, max-transient theorem).  All inputs are integers
    measured by the caller from actual array nbytes / returned dtypes;
    nothing is modeled here.
    """
    if remaining < 1:
        return NO_FIT
    per_source = (v_prime * (dist_itemsize + pred_itemsize + mask_itemsize)
                  + row_temporaries + source_index_cost)
    budget_k = cap_bytes - fixed_live_base - retained_parts_bytes - margin_bytes
    if budget_k < per_source:
        return NO_FIT
    c = budget_k // per_source
    if c > remaining:
        c = remaining
    return c
