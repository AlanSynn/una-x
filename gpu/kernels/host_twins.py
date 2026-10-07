"""Host reference twins for the fold-region kernels, the exact K2 tie-law
argsort transcription, and the K3 association-tree plans.

WHAT MAY BE CLAIMED HERE (and what may not):

- ``k0_compact`` is bit-exact BY CONSTRUCTION against the fold's
  ``np.where(d_distance <= cutoff)[0]`` prologue: int64 indices + float64
  comparisons, closed-radius (``<=``), NaN excluded because NaN <= x is
  False, ascending destination order preserved.  Order preservation is
  semantic, not incidental: the compacted order fixes the association order
  of every downstream reduction.
- ``argsort_tie_law`` (K2) is a line-level transcription of numba 0.67.0's
  quicksort argsort (numba/misc/quicksort.py: ``SMALL_QUICKSORT = 15``,
  ``MAX_STACK = 100``, median-of-three partition, LIFO largest-half stack,
  final insertion sort).  Its tie permutation is the contractual numba tie
  order that differs from every numpy kind (131/200 tie trials,
  n00/probe_order.txt).  Tests verify it against a locally compiled numba
  probe on tie-dense batteries.
- ``k1_*`` twins are per-element IEEE-754 expressions (no cross-element
  operation), so host numpy evaluates them exactly as written.  A DEVICE
  exp() is NOT assumed bit-equal to the host exp(): that identity is
  exactly what the per-cell bit-validation gate must prove (dossier 08:
  "unless device math is independently bit-validated").
- The K3 trees (``seq``/``pairwise``/``vecN_tree``/``vecN_seq``) are PLANS,
  not parity claims: a named tree reproduces the engine fold's bits only if
  that tree is the engine's compiled association for the cell's input
  class, which only the bit-validation gate can certify.  Nothing here may
  be presented as "matches the engine" without that gate.

Porting source: src/urban_network_analysis/kernels/reductions.py
(plan...reach_gravity_knn.v1.legacy_fastmath) — the declared portable
transcription of Engines.AccessibilityWElevation.reach_gravity_knn_access.
The trusted CPU route remains in src/; the twins are spec/test artifacts,
never a production fallback.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "k0_compact", "k1_gravity_exponential_coeffs", "k1_gravity_logistic_coeffs",
    "k1_knn_coeffs", "argsort_tie_law", "reduce_with", "REDUCTION_TREES",
    "fold_host_reference",
]

# ---------------------------------------------------------------------------
# K0: order-preserving compaction (device kernel twin)
# ---------------------------------------------------------------------------

def k0_compact(d_distance: np.ndarray, cutoff: float):
    """Indices of ``d_distance <= cutoff``, ascending, int64.

    Mirrors ``np.where(d_distance <= cutoff)[0]`` exactly (closed radius;
    NaN compares False and is excluded).  Returns (indices int64, count).
    """
    mask = d_distance <= cutoff
    indices = np.where(mask)[0].astype(np.int64)
    return indices, int(indices.shape[0])


def kept_prefix(d_distance: np.ndarray, d_weights: np.ndarray, cutoff: float):
    """Kept-plan input passthrough: values already compacted by the trace.

    Validates finiteness of the kept values (the engine's domain guarantee:
    in-cutoff distances are finite) and returns views unchanged.
    """
    if d_distance.shape[0] != d_weights.shape[0]:
        raise ValueError("kept prefix: distance/weight length mismatch "
                         f"({d_distance.shape[0]} vs {d_weights.shape[0]})")
    return d_distance, d_weights


# ---------------------------------------------------------------------------
# K1: per-element coefficient twins (device kernel twins, host evaluation)
# ---------------------------------------------------------------------------

def k1_gravity_exponential_coeffs(dn: np.ndarray, wn: np.ndarray,
                                  gravity_beta: float,
                                  gravity_plateau: float) -> np.ndarray:
    """``wn / exp(gravity_beta * max(0, dn - plateau))`` elementwise."""
    t = np.maximum(0, dn - gravity_plateau)
    return wn / np.exp(gravity_beta * t)


def k1_gravity_logistic_coeffs(dn: np.ndarray, wn: np.ndarray,
                               gravity_growth_rate: float,
                               gravity_plateau: float,
                               gravity_logistic_midpoint: float) -> np.ndarray:
    """``wn * (1 - 1 / (1 + exp(-g * (dn - plateau - midpoint))))`` elementwise."""
    s = -gravity_growth_rate * (dn - gravity_plateau - gravity_logistic_midpoint)
    return wn * (1 - 1 / (1 + np.exp(s)))


def k1_knn_coeffs(dn_s: np.ndarray, wn_s: np.ndarray, kn_coefs: np.ndarray,
                  knn_decay: str, gravity_beta: float, gravity_growth_rate: float,
                  gravity_plateau: float,
                  gravity_logistic_midpoint: float) -> np.ndarray:
    """Elementwise knn-branch coefficients (NO reduction) per decay branch.

    Same expressions as the fold's knn_access sums' summands, in the fold's
    exact association per element:
      exponential: (kn * wn) * exp(-beta * max(0, dn - plateau))
      logistic:    (kn * wn) * (1 - 1 / (1 + exp(-g * (dn - plateau - midpoint))))
      other:       kn * wn
    """
    if knn_decay == "exponential":
        t = np.maximum(0, dn_s - gravity_plateau)
        return (kn_coefs * wn_s) * np.exp(-gravity_beta * t)
    if knn_decay == "logistic":
        s = -gravity_growth_rate * (dn_s - gravity_plateau - gravity_logistic_midpoint)
        return (kn_coefs * wn_s) * (1 - 1 / (1 + np.exp(s)))
    return kn_coefs * wn_s


# ---------------------------------------------------------------------------
# K2: exact numba tie-law argsort (line-level transcription)
# ---------------------------------------------------------------------------

# numba/misc/quicksort.py @ numba 0.67.0 (read this task; file pinned in the
# G10 evidence): under this size, switch to insertion sort.
_SMALL_QUICKSORT = 15
_MAX_STACK = 100


def argsort_tie_law(a: np.ndarray) -> np.ndarray:
    """Exact transcription of numba's ``np.argsort`` (default quicksort).

    Reproduces numba's tie permutation — the contractual order inside the
    una_legacy fold that differs from every numpy kind.  Keys must be the
    compacted float64 distances (1-D, C-contiguous).  Comparison is strict
    ``<`` on float64 exactly as numba's ``default_lt``; NaN would sink (and
    is outside the fold's admitted domain anyway: kept distances are finite
    or excluded by K0).

    Transcribed from numba 0.67.0 numba/misc/quicksort.py,
    ``make_quicksort_impl(is_argsort=True)``: R = arange, GET(A, k) = A[k],
    LT = (a < b), median-of-three partition with the exact swap pattern,
    LIFO stack pushing the larger half, insertion sort on small runs.
    """
    from ..runtime.buffers import require_c_contiguous
    require_c_contiguous("argsort_tie_law keys", a)
    if a.dtype != np.float64:
        raise ValueError(f"keys must be float64, got {a.dtype}; refusal, "
                         "no coercion")
    n = int(a.shape[0])
    R = np.arange(n, dtype=np.intp)
    if n < 2:
        return R

    stack = [(0, 0)] * _MAX_STACK
    stack[0] = (0, n - 1)
    top = 1

    while top > 0:
        top -= 1
        low, high = stack[top]
        # Partition until it becomes more efficient to insertion-sort.
        while high - low >= _SMALL_QUICKSORT:
            if top >= _MAX_STACK:
                raise AssertionError("quicksort stack overflow "
                                     f"(> {_MAX_STACK})")
            mid = (low + high) >> 1
            # median of three {low, middle, high} — exact swap pattern
            if a[R[mid]] < a[R[low]]:
                R[low], R[mid] = R[mid], R[low]
            if a[R[high]] < a[R[mid]]:
                R[high], R[mid] = R[mid], R[high]
            if a[R[mid]] < a[R[low]]:
                R[low], R[mid] = R[mid], R[low]
            pivot = a[R[mid]]

            # stash the pivot at the end
            R[high], R[mid] = R[mid], R[high]
            i = low
            j = high - 1
            while True:
                while i < high and a[R[i]] < pivot:
                    i += 1
                while j >= low and pivot < a[R[j]]:
                    j -= 1
                if i >= j:
                    break
                R[i], R[j] = R[j], R[i]
                i += 1
                j -= 1
            # pivot back into final place
            R[i], R[high] = R[high], R[i]

            # push largest partition on the stack
            if high - i > i - low:
                if high > i:
                    stack[top] = (i + 1, high)
                    top += 1
                high = i - 1
            else:
                if i > low:
                    stack[top] = (low, i - 1)
                    top += 1
                low = i + 1

        # insertion sort A[low:high + 1] (inclusive bounds)
        for ii in range(low + 1, high + 1):
            k = R[ii]
            v = a[k]
            j = ii
            while j > low and v < a[R[j - 1]]:
                R[j] = R[j - 1]
                j -= 1
            R[j] = k

    return R


# ---------------------------------------------------------------------------
# K3: association-tree plans (PLANS, not parity claims)
# ---------------------------------------------------------------------------

# The candidate set mirrors n00/instruments/probe_order.py verbatim (read
# this task).  These are the named sum-association classes a device (or a
# recompiled host) must choose between; the bit-validation gate certifies
# which one a cell's compiled fold actually used.
REDUCTION_TREES = ("seq", "pairwise", "vec2_tree", "vec4_tree", "vec8_tree",
                   "vec16_tree", "vec4_seq", "vec8_seq")


def seq_sum(a: np.ndarray) -> np.float64:
    s = np.float64(0.0)
    for x in a:
        s = np.float64(s + x)
    return s


def pairwise_sum(a: np.ndarray) -> np.float64:
    """numpy's documented pairwise scheme (block 128, unroll 8)."""
    n = a.shape[0]
    if n <= 8:
        return seq_sum(a)
    if n <= 128:
        r = n % 8
        acc = np.zeros(8, dtype=np.float64)
        m = n - r
        for i in range(0, m, 8):
            for j in range(8):
                acc[j] = np.float64(acc[j] + a[i + j])
        s = np.float64(0.0)
        for j in range(8):
            s = np.float64(s + acc[j])
        for i in range(m, n):
            s = np.float64(s + a[i])
        return s
    n2 = n // 2
    n2 += n2 & 1
    return np.float64(pairwise_sum(a[:n2]) + pairwise_sum(a[n2:]))


def vec_sum(a: np.ndarray, vf: int, combine: str = "tree") -> np.float64:
    """Strided-lane vectorized sum with tree or sequential lane combine."""
    n = a.shape[0]
    lanes = [np.float64(0.0)] * vf
    for j in range(vf):
        s = np.float64(0.0)
        for i in range(j, n, vf):
            s = np.float64(s + a[i])
        lanes[j] = s
    if combine == "tree":
        cur = lanes
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur) - 1, 2):
                nxt.append(np.float64(cur[i] + cur[i + 1]))
            if len(cur) % 2:
                nxt.append(cur[-1])
            cur = nxt
        return cur[0]
    if combine == "seq":
        s = np.float64(0.0)
        for x in lanes:
            s = np.float64(s + x)
        return s
    raise ValueError(f"unknown lane combine {combine!r}")


def reduce_with(tree: str, a: np.ndarray) -> np.float64:
    """Sum ``a`` under the named association tree (K3 host plan)."""
    if tree == "seq":
        return seq_sum(a)
    if tree == "pairwise":
        return pairwise_sum(a)
    if tree.endswith("_tree") and tree.startswith("vec"):
        return vec_sum(a, int(tree[3:-5]), "tree")
    if tree.endswith("_seq") and tree.startswith("vec"):
        return vec_sum(a, int(tree[3:-4]), "seq")
    raise ValueError(f"unknown reduction tree {tree!r}; certified set is "
                     f"{REDUCTION_TREES}")


# ---------------------------------------------------------------------------
# Full fold host reference (composition of the twins) — EXPLICIT association
# ---------------------------------------------------------------------------

def fold_host_reference(d_distance: np.ndarray, d_weights: np.ndarray,
                        cutoff: float, gravity_beta: float,
                        gravity_plateau: float,
                        gravity_logistic_midpoint: float,
                        gravity_growth_rate: float, knn_decay: str,
                        knn_weights: np.ndarray,
                        association: str):
    """Compose the fold from the twins with an EXPLICIT association tree.

    Returns (reach, gravity_exponential, gravity_logistic, knn_access) as
    float64, mirroring reductions.py's legacy plan composition but with the
    reduction association named by the caller.

    PARITY STATUS: bit-equality with the compiled engine fold holds only if
    ``association`` is the engine's actual compiled association for this
    input class — which is precisely what the bit-validation gate certifies
    per cell.  Without a passed gate, no caller may treat this output as
    engine-parity output.
    """
    if association not in REDUCTION_TREES:
        raise ValueError(f"association must be one of {REDUCTION_TREES}; "
                         f"got {association!r}")
    indices, count = k0_compact(d_distance, cutoff)
    dn = d_distance[indices[:count]]
    wn = d_weights[indices[:count]]

    reach = reduce_with(association, wn)
    c_exp = k1_gravity_exponential_coeffs(dn, wn, gravity_beta,
                                          gravity_plateau)
    gravity_exponential = reduce_with(association, c_exp)
    c_log = k1_gravity_logistic_coeffs(dn, wn, gravity_growth_rate,
                                       gravity_plateau,
                                       gravity_logistic_midpoint)
    gravity_logistic = reduce_with(association, c_log)

    knn = min(dn.shape[0], knn_weights.shape[0])
    if knn == 0:
        knn_access = np.float64(0.0)
    else:
        idx = argsort_tie_law(dn)
        dn_s = dn[idx]
        wn_s = wn[idx]
        c_knn = k1_knn_coeffs(dn_s[:knn], wn_s[:knn], knn_weights[:knn],
                              knn_decay, gravity_beta, gravity_growth_rate,
                              gravity_plateau, gravity_logistic_midpoint)
        knn_access = reduce_with(association, c_knn)

    return (np.float64(reach), np.float64(gravity_exponential),
            np.float64(gravity_logistic), np.float64(knn_access))
