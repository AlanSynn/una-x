import os, sys
import numpy as np
import numba as nb

SRC = "/Users/alansynn/orca/workspaces/una-x/main/src"
sys.path.insert(0, SRC)

# Probe 1: what reduction order does the ACTUAL fold use?
from urban_network_analysis.Engines.AccessibilityWElevation import reach_gravity_knn_access as fold

def seq_sum(a):
    s = np.float64(0.0)
    for x in a:
        s = np.float64(s + x)
    return s

def pairwise_sum(a):  # numpy's documented pairwise (block 128, unroll 8)
    n = a.shape[0]
    if n <= 8:
        return seq_sum(a)
    if n <= 128:
        # numpy: unrolled-by-8 loop then remainder
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

def vec_sum(a, vf, combine="tree"):
    # strided lanes + combine; try both tree and sequential combine
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
    else:
        s = np.float64(0.0)
        for x in lanes:
            s = np.float64(s + x)
        return s

BIG = np.float64(2.0) ** 53
TWO30 = np.float64(2.0) ** 30

cases = {}
# w must be finite nonneg (admitted domain)
cases["big_at_1"] = np.array([1.0, BIG] + [1.0] * 6)
cases["big_at_7"] = np.array([1.0] * 7 + [BIG])
cases["big_at_0"] = np.array([BIG] + [1.0] * 7)
cases["tiny_then_big"] = np.array([2.0 ** -30] * 8 + [BIG])
cases["mixed"] = np.array([1.0, TWO30, 1.0, 1.0, TWO30, 1.0, 1.0, 1.0])

for name, w in cases.items():
    d = np.zeros(w.shape[0], dtype=np.float64)  # all distances 0 <= cutoff
    cutoff = np.float64(10.0)
    # beta=0 -> exp(0)=1, gravity sum == plain sum of w/1
    reach, ge, gl, kn = fold(d, w, cutoff, 0.0, 0.0, 1.0, 0.0,
                             np.ones(4, dtype=np.float64), 0.0, "linear",
                             1.0, 1.0, 1.0)
    cands = {
        "seq": seq_sum(w),
        "pairwise": pairwise_sum(w),
        "vec2_tree": vec_sum(w, 2), "vec4_tree": vec_sum(w, 4),
        "vec8_tree": vec_sum(w, 8), "vec4_seq": vec_sum(w, 4, "seq"),
        "vec8_seq": vec_sum(w, 8, "seq"), "vec16_tree": vec_sum(w, 16),
    }
    hits = [k for k, v in cands.items() if np.float64(v).tobytes() == np.float64(reach).tobytes()]
    print(f"{name}: numba_reach={reach!r} matches={hits}")

# Probe 2: argsort tie permutation inside numba vs numpy kinds
@nb.njit(cache=False)
def _argsort_probe(a):
    return np.argsort(a)

rng = np.random.default_rng(7)
tie_conflicts = 0
total = 0
for trial in range(200):
    n = int(rng.integers(2, 60))
    a = np.round(rng.random(n) * 3, 1)  # many ties
    idx_nb = _argsort_probe(a)
    kinds = {}
    for kind in ("quicksort", "stable", "heapsort"):
        kinds[kind] = np.argsort(a, kind=kind)
    match = [k for k, v in kinds.items() if np.array_equal(idx_nb, v)]
    total += 1
    if not match:
        tie_conflicts += 1
        if tie_conflicts <= 3:
            print(f"TIE-PERM mismatch trial {trial}: numba={idx_nb.tolist()} np_quicksort={kinds['quicksort'].tolist()} a={a.tolist()}")
print(f"argsort trials={total} numba!=all-numpy-kinds count={tie_conflicts}")
