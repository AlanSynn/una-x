"""INDEPENDENT adversarial review driver for S10 (review_indep).

Imports the probe module from its real path (no edits to it) and runs
NEW hostile cases through the probe's own case() function, plus a
dirty-arena continuation test the probe's E7 does not cover.

Hard acceptance: projection bitwise == frozen oracle on every case
(any hit in the law/a1 findings buckets is recorded, not a failure).
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, '..', 'probe_sparse_projection_tiny.py')

spec = importlib.util.spec_from_file_location('s10_probe', PROBE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

import urban_network_analysis.kernels.scope_access as sa
import urban_network_analysis.Engines.AccessibilityWElevation as aw
print('oracle from:', sa.__file__)
print('compact from:', aw.__file__)
assert 'cpu-kernels' in sa.__file__, 'oracle not loaded from audited worktree'
assert 'cpu-kernels' in aw.__file__, 'compact not loaded from audited worktree'

results = []


def run(tag, arcs, weights, o, w, cutoff, n, d_count=0, domain='a3'):
    ptr, adj, wts = P.csr(n, arcs, weights)
    ok = P.case(tag, *P.ot(o[0], o[1], w[0], w[1]), ptr, adj, wts, cutoff,
                d_count=d_count, domain=domain)
    fails = [f for f in P.CASES['failed'] if f['tag'] == tag]
    div = tag in P.CASES.get('law_divergent_tags', [])
    a1d = tag in P.CASES.get('a1_nan_divergent_tags', [])
    results.append({'tag': tag, 'case_ok': bool(ok),
                    'law_divergent': div, 'a1_nan_divergent': a1d,
                    'hard_fails': [f['detail'] for f in fails]})
    print(f"{'OK ' if ok else 'FAIL'} {tag} law_div={div} a1_div={a1d}"
          + (f" hard={results[-1]['hard_fails']}" if fails else ''))
    return ok


NAN = float('nan')
INF = float('inf')

# --- H1: dead-end node (out-degree 0) mixed with a cycle; seed AT the dead end
run('h1/deadend_seed',
    [(0, 1), (1, 2), (2, 0), (1, 3)],
    [0.5, 0.5, 0.25, 0.75], (0, 3), (0.2, 0.4), np.float64(1.5), 4)

# --- H1b: dead end reached through the cycle, NaN candidate from the cycle
run('h1b/deadend_nan_cycle',
    [(0, 1), (1, 2), (2, 0), (2, 3)],
    [0.5, 0.5, 1.0, 0.5], (0, 3), (0.1, NAN), np.float64(1.5), 4)

# --- H2: NaN cutoff (admission refuses it; the unbounded oracle PUMPS
#     forever here, so case() cannot be used -- cap-bounded comparison
#     instead, E5-style). DISCOVERY: fast-law eligibility gates are
#     ALWAYS TRUE under NaN cutoff (fast_le(c,NaN) and fast_lt(c,NaN)
#     are both True for every finite c), so the heap pumps without end.
def cap_case(tag, arcs, weights, o, w, cutoff, n, cap=2000):
    ptr, adj, wts = P.csr(n, arcs, weights)
    fl = P.flags_all(adj.shape[0])
    _, ev_m, cap_m = P.trace_schedule(P.ot(o[0], o[1], w[0], w[1])[0],
                                      P.ot(o[0], o[1], w[0], w[1])[1],
                                      ptr, adj, wts, fl, cutoff,
                                      pop_cap=cap)
    _, ev_k, cap_k = P.Kp_schedule(P.ot(o[0], o[1], w[0], w[1])[0],
                                   P.ot(o[0], o[1], w[0], w[1])[1],
                                   ptr, adj, wts, fl, cutoff,
                                   arenas=P.fresh_arenas(n), pop_cap=cap)
    same = P.canon_ev(ev_m) == P.canon_ev(ev_k)
    both_capped = bool(cap_m and cap_k)
    print(f"{'OK ' if same else 'FAIL'} {tag} capped=({cap_m},{cap_k}) "
          f"events_equal={same}")
    results.append({'tag': tag, 'case_ok': bool(same),
                    'law_divergent': False, 'a1_nan_divergent': False,
                    'hard_fails': [] if same else ['capped event mismatch'],
                    'both_capped': both_capped,
                    'note': 'cap-bounded comparison; both_capped=True means '
                            'the frozen oracle itself nonterminates and the '
                            'projection pumps identically'})
    return same


cap_case('h2/cutoff_nan_capped', [(0, 1), (1, 0)], [0.5, 0.5], (0, 1),
         (0.1, 0.9), np.float64(NAN), 2)
cap_case('h2b/cutoff_nan_nan_seeds_capped', [(0, 1), (1, 0)], [0.5, 0.5],
         (0, 1), (NAN, NAN), np.float64(NAN), 2)

# --- H3: negative cutoffs
run('h3/cutoff_neg', [(0, 1), (1, 0)], [0.5, 0.5], (0, 1), (0.1, 0.9),
    np.float64(-0.5), 2)
run('h3b/cutoff_neg_negseeds', [(0, 1), (1, 2), (2, 0)],
    [0.5, 0.5, 0.25], (0, 2), (-0.9, -0.75), np.float64(-0.5), 3)

# --- H4: int64 cutoff 2^63-1 with NaN seeds
run('h4/cutoff_int64max_nanseeds', [(0, 1), (1, 2), (2, 0), (0, 2)],
    [0.5, 0.5, 1.0, 0.25], (0, 2), (NAN, 0.5),
    np.int64(2 ** 63 - 1), 3)
run('h4b/cutoff_int64min', [(0, 1), (1, 0)], [0.5, 0.5], (0, 1),
    (0.1, 0.9), np.int64(-2 ** 63), 2)

# --- H5: duplicated arcs with NaN weights (NaN ARC weights: never in probe)
run('h5/dup_nan_arcs', [(0, 1), (0, 1), (1, 2), (1, 2)],
    [NAN, 0.5, 0.5, NAN], (0, 2), (0.1, 0.2), np.float64(1.5), 3)
# h5b: NaN arc weight + NaN seed => the frozen oracle itself NONTERMINATES
# (measured: >20 s no return; a1 too; compact and IEEE mirrors return).
# case() cannot be used (unbounded oracle); cap-bounded comparison only.
cap_case('h5b/dup_nan_arcs_nan_seeds_capped', [(0, 1), (0, 1), (1, 0), (1, 0)],
         [NAN, 0.5, NAN, 0.5], (0, 1), (NAN, 0.5), np.float64(1.5), 2)

# --- H6: 1-node graph, self-loop, same-node NaN seeds
run('h6/single_selfloop', [(0, 0)], [0.5], (0, 0), (NAN, 0.5),
    np.float64(1.5), 1)
run('h6b/single_selfloop_finite', [(0, 0)], [0.25], (0, 0), (0.1, 0.2),
    np.float64(1.5), 1)

# --- H7: seeds exactly AT cutoff (strict push gate, closed-radius writes)
run('h7/seeds_at_cutoff', [(0, 1), (1, 2), (2, 0)], [0.5, 0.5, 0.25],
    (0, 2), (1.5, 1.5), np.float64(1.5), 3)
run('h7b/seeds_above_cutoff', [(0, 1), (1, 0)], [0.5, 0.5], (0, 1),
    (1.5, 2.0), np.float64(1.5), 2)

# --- H8: inf seeds and inf arc weights mixed
run('h8/inf_seeds', [(0, 1), (1, 2)], [0.5, 0.5], (0, 2), (INF, 0.5),
    np.float64(1.5), 3)
run('h8b/inf_arc', [(0, 1), (1, 2)], [INF, 0.5], (0, 2), (0.1, 0.2),
    np.float64(1.5), 3)
run('h8c/neginf_arc', [(0, 1), (1, 2)], [-INF, 0.5], (0, 2), (0.1, 0.2),
    np.float64(1.5), 3)

# --- H9: A1 domain with destination tail (d_count > 0) -- the probe's
#     case matrix NEVER exercises this; drive it here.
dt = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64)
assert sa.tail_admits(dt, 4, 4), 'tail_admits refused the H9 fixture'
ptr9, adj9, w9 = P.csr(4, [(0, 1), (1, 2), (2, 0), (1, 3)],
                       [0.5, 0.5, 0.25, 0.75])
lab_oracle = sa.scope_search_schedule(P.ot(0, 3, 0.2, 0.4)[0],
                                      P.ot(0, 3, 0.2, 0.4)[1],
                                      ptr9, adj9, w9,
                                      P.flags_all(adj9.shape[0]),
                                      np.float64(1.5), d_count=4,
                                      domain='a1')
deg = 2
off = np.empty(deg + 1, dtype=np.int64)
wt = np.empty(deg + 1, dtype=np.float64)
import urban_network_analysis.Engines._large_access_scratch as las
assert 'cpu-kernels' in las.__file__, 'a1 not loaded from audited worktree'
lab_a1 = las._a1_scope_search(
    P.ot(0, 3, 0.2, 0.4)[0], P.ot(0, 3, 0.2, 0.4)[1], ptr9, adj9, w9,
    P.flags_all(adj9.shape[0]), np.float64(1.5), 4, off, wt)[0]
arenas9 = P.fresh_arenas(4 + 4)
lab_kp, ev_kp, _ = P.Kp_schedule(P.ot(0, 3, 0.2, 0.4)[0],
                                 P.ot(0, 3, 0.2, 0.4)[1], ptr9, adj9, w9,
                                 P.flags_all(adj9.shape[0]),
                                 np.float64(1.5), d_count=4, domain='a1',
                                 arenas=arenas9)
h9_oracle = P.bit_eq(lab_kp, lab_oracle)
h9_a1 = P.bit_eq(lab_kp, lab_a1)
tail_sentinel = (np.ones(1, dtype=np.float64) + np.float64(1.5))[0]
tail_ok = all(lab_kp[4 + i] == tail_sentinel for i in range(4))
print(f"H9 a1-domain: kp==oracle {h9_oracle}, kp==a1 {h9_a1}, "
      f"tail all sentinel {tail_ok}, tail bytes "
      f"{[hex(v) for v in lab_kp.view(np.uint64)[4:]]}")
assert h9_oracle, 'H9 HARD FAILURE: projection != oracle on a1 domain'
results.append({'tag': 'h9/a1_tail_domain', 'case_ok': bool(
    h9_oracle and h9_a1 and tail_ok), 'law_divergent': False,
    'a1_nan_divergent': False,
    'hard_fails': [] if h9_oracle else ['kp != oracle']})

# --- H10: shared arenas across origins WITH a genuinely mid-origin
#     injected exception, then CONTINUE on the same dirty arenas
#     (E7 raises only after the search returns, on fresh arenas).


class Trap(np.ndarray):
    """Weights array that raises on the k-th element read."""
    budget = [2]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            Trap.budget[0] -= 1
            if Trap.budget[0] < 0:
                raise RuntimeError('injected MID-ORIGIN interruption')
        return super().__getitem__(idx)


ptrA, adjA, wA = P.csr(4, [(0, 1), (1, 2), (2, 0), (0, 2), (1, 3)],
                       [0.5, 0.25, 0.75, 1.25, 0.1])
flA = P.flags_all(adjA.shape[0])
arenas = P.fresh_arenas(4)
bad = []
origins = [(0, 1, 0.05, 0.9), (1, 2, 0.15, 0.8), (2, 3, 0.25, 0.7),
           (3, 0, 0.35, 0.6), (0, 2, 0.45, 0.5)]
# corrupt the arenas first (simulating leftovers), then run epochs on top
arenas[1][:] = 97
arenas[0][:] = -1.0
for k, (s, e, w0, w1) in enumerate(origins):
    if k == 2:
        tw = wA.view(Trap)
        Trap.budget[0] = 2
        try:
            P.Kp_schedule(P.ot(s, e, w0, w1)[0], P.ot(s, e, w0, w1)[1],
                          ptrA, adjA, tw, flA, np.float64(1.5),
                          arenas=arenas, epoch=k + 1)
            bad.append(f'e{k}: no interrupt raised')
        except RuntimeError:
            pass
        # CONTINUE on the same dirty arenas with the next epoch
        ep = k + 2
    else:
        ep = k + 1
    ref = sa.scope_search_schedule(P.ot(s, e, w0, w1)[0],
                                   P.ot(s, e, w0, w1)[1], ptrA, adjA, wA,
                                   flA, np.float64(1.5))
    got, _, _ = P.Kp_schedule(P.ot(s, e, w0, w1)[0], P.ot(s, e, w0, w1)[1],
                              ptrA, adjA, wA, flA, np.float64(1.5),
                              arenas=arenas, epoch=ep)
    if not P.bit_eq(got, ref):
        bad.append(f'e{ep}: dirty-arena continuation diverged')
if bad:
    print('H10 FAILED:', bad)
    results.append({'tag': 'h10/dirty_arena_continuation',
                    'case_ok': False, 'law_divergent': False,
                    'a1_nan_divergent': False, 'hard_fails': bad})
else:
    print('H10 OK: 4 clean origins + mid-origin interrupt + continued on '
          'same arenas, all bitwise vs oracle')
    results.append({'tag': 'h10/dirty_arena_continuation', 'case_ok': True,
                    'law_divergent': False, 'a1_nan_divergent': False,
                    'hard_fails': []})

# --- H11: same-node double-NaN seeds (F2 territory, projection vs oracle)
run('h11/same00_nan_nan', [(0, 1), (1, 0)], [0.5, 0.5], (0, 0),
    (NAN, NAN), np.float64(1.5), 2)

hard = [r for r in results if not r['case_ok']]
summary = {
    'hostile_cases': results,
    'hard_failures': hard,
    'law_divergent': [r['tag'] for r in results if r['law_divergent']],
    'a1_nan_divergent': [r['tag'] for r in results if r['a1_nan_divergent']],
    'probe_cases_failed_total': len(P.CASES['failed']),
}
with open(os.path.join(HERE, 'hostile_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)
    f.write('\n')
print('\nHARD FAILURES:', len(hard))
print(json.dumps(summary['law_divergent'], indent=1))
print(json.dumps(summary['a1_nan_divergent'], indent=1))
sys.exit(1 if hard else 0)
