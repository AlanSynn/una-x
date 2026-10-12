"""S12 pump attribution: NaN seed weight on a small cyclic graph (v3).

Bounded (timeout-guarded) subprocess runs of:
  A) the ORIGINAL driver integrated_scope_access,
  B) the engine's staged a3 kernel _a3_scope_search_tailless, invoked the
     way the engine invokes it (inside an @njit wrapper, per-origin row
     args) — v2 showed direct interpreter calls die in numba's heapq
     tuple typing regardless of NaN, so the engine call shape is used,
  C) the engine's compact kernel compact_vector_node_view_scope (same
     wrapper treatment),
  D) the sparse module's gate on the same NaN-seed input.
Plus a finite control.  Reviewer observation only — attribution of a
nonterminal class in the ENGINE, not the sparse module.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SRC = "/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/s12-review-r1/src"

PTR = [0, 1, 2, 4]
VEC = [1, 2, 0, 0]
WTS = [0.5, 0.5, 0.5, 0.7]
FLAG = [True, True, True, True]
D = [[1, 2], [2, 0]]
DW = [[0.25, 0.0], [0.0, 0.25]]
DWT = [1.0, 0.5]
MET = "[1.5, 2.0, 3.0, 0.8, 'exponential', np.array([1.0, 0.5])]"

HEADER = (
    "import sys\n"
    "sys.path.insert(0, {src!r})\n"
    "import numpy as np\n"
    "import numba as nb\n"
    "import signal\n"
    "def onalarm(s, f):\n"
    "    raise TimeoutError('inner 10s alarm')\n"
    "signal.signal(signal.SIGALRM, onalarm)\n"
    "signal.alarm(10)\n").format(src=SRC)

ARRS = (
    "PTR = np.array({ptr!r}, dtype=np.int64)\n"
    "VEC = np.array({vec!r}, dtype=np.int64)\n"
    "WTS = np.array({wts!r}, dtype=np.float64)\n"
    "FLAG = np.array({flag!r}, dtype=bool)\n"
    "D = np.array({d!r}, dtype=np.int64)\n"
    "DW = np.array({dw!r}, dtype=np.float64)\n"
    "DWT = np.array({dwt!r}, dtype=np.float64)\n"
    "MET = {met}\n"
    "O = np.array([[0, 1]], dtype=np.int64)\n"
    "OW = np.array([[0.0, 0.1]])\n"
    "OWN = np.array([[0.0, np.nan]])\n"
).format(ptr=PTR, vec=VEC, wts=WTS, flag=FLAG, d=D, dw=DW, dwt=DWT,
         met=MET)

CASE_A = HEADER + ARRS + (
    "from urban_network_analysis.Engines.AccessibilityWElevation"
    " import integrated_scope_access\n"
    "r = integrated_scope_access(O, OWN, PTR, VEC, WTS, FLAG, D, DW, DWT,"
    " *MET, 1.2)\n"
    "signal.alarm(0)\n"
    "print('DONE-driver', [float(x) for x in r[0][:3]])\n")

CASE_B = HEADER + ARRS + (
    "from urban_network_analysis.Engines._large_access_scratch import"
    " _a3_scope_search_tailless\n"
    "@nb.njit(cache=False)\n"
    "def wrap(o_row, ow_row, ptr, vec, wts, flag, cutoff):\n"
    "    eo = np.empty(4, dtype=np.int64)\n"
    "    ew = np.empty(4, dtype=np.float64)\n"
    "    return _a3_scope_search_tailless(o_row, ow_row, ptr, vec, wts,"
    " flag, cutoff, eo, ew)\n"
    "r = wrap(np.array([0, 1], dtype=np.int64), OWN[0], PTR, VEC, WTS,"
    " FLAG, 1.2)\n"
    "signal.alarm(0)\n"
    "print('DONE-a3kernel', [float(x) for x in r[0][[0, 1, 2]]])\n")

CASE_C = HEADER + ARRS + (
    "from urban_network_analysis.Engines.AccessibilityWElevation"
    " import compact_vector_node_view_scope\n"
    "@nb.njit(cache=False)\n"
    "def wrapc(o_row, ow_row, ptr, vec, wts, flag, cutoff, d_count):\n"
    "    return compact_vector_node_view_scope(o_row, ow_row, ptr, vec,"
    " wts, flag, cutoff, d_count)\n"
    "r = wrapc(np.array([0, 1], dtype=np.int64), OWN[0], PTR, VEC, WTS,"
    " FLAG, 1.2, 2)\n"
    "signal.alarm(0)\n"
    "print('DONE-compact', [float(x) for x in r[0][[0, 1, 2]]])\n")

CASE_D = HEADER + ARRS + (
    "from urban_network_analysis.Engines import _sparse_access_scratch"
    " as SP\n"
    "r = SP.sparse_scope_access(O, OWN, PTR, VEC, WTS, FLAG, D, DW, DWT,"
    " *MET, 1.2)\n"
    "signal.alarm(0)\n"
    "print('DONE-sparse-gate', r.route, r.refusal)\n")

CONTROL_B = CASE_B.replace(
    "wrap(np.array([0, 1], dtype=np.int64), OWN[0],",
    "wrap(np.array([0, 1], dtype=np.int64), OW[0],")

# 30-node lattice-v2 graph: the case on which the ORIGINAL driver was
# observed nonterminating with unbounded RSS during the lattice probe.
ARRS30 = (
    "rng = np.random.default_rng(77)\n"
    "n = 30\n"
    "src = np.concatenate([np.arange(n, dtype=np.int64),"
    " rng.integers(0, n, 40).astype(np.int64)])\n"
    "dst = np.concatenate([((np.arange(n) + 1) % n).astype(np.int64),"
    " rng.integers(0, n, 40).astype(np.int64)])\n"
    "order = np.argsort(src, kind='stable')\n"
    "src, dst = src[order], dst[order]\n"
    "PTR30 = np.zeros(n + 1, dtype=np.int64)\n"
    "PTR30[1:] = np.cumsum(np.bincount(src, minlength=n))\n"
    "VEC30 = np.ascontiguousarray(dst)\n"
    "WTS30 = np.ascontiguousarray(rng.uniform(0.1, 0.8, dst.shape[0]))\n"
    "FLAG30 = np.ones(dst.shape[0], dtype=bool)\n"
    "O30 = np.array([[3, 4]], dtype=np.int64)\n"
    "OW30 = np.array([[0.05, np.nan]])\n")

CASE_E = HEADER + ARRS + ARRS30 + (
    "from urban_network_analysis.Engines.AccessibilityWElevation"
    " import integrated_scope_access\n"
    "r = integrated_scope_access(O30, OW30, PTR30, VEC30, WTS30, FLAG30,"
    " D, DW, DWT, *MET, 1.2)\n"
    "signal.alarm(0)\n"
    "print('DONE-driver30', [float(x) for x in r[0][:3]])\n")

CASE_F = HEADER + ARRS30 + (
    "from urban_network_analysis.Engines._large_access_scratch import"
    " _a3_scope_search_tailless\n"
    "@nb.njit(cache=False)\n"
    "def wrap(o_row, ow_row, ptr, vec, wts, flag, cutoff):\n"
    "    eo = np.empty(8, dtype=np.int64)\n"
    "    ew = np.empty(8, dtype=np.float64)\n"
    "    return _a3_scope_search_tailless(o_row, ow_row, ptr, vec, wts,"
    " flag, cutoff, eo, ew)\n"
    "r = wrap(O30[0], OW30[0], PTR30, VEC30, WTS30, FLAG30, 1.2)\n"
    "signal.alarm(0)\n"
    "print('DONE-a3kernel30', [float(x) for x in r[0][[0, 1, 2]]])\n")

CASE_G = HEADER + ARRS30 + (
    "from urban_network_analysis.Engines.AccessibilityWElevation"
    " import compact_vector_node_view_scope\n"
    "@nb.njit(cache=False)\n"
    "def wrapc(o_row, ow_row, ptr, vec, wts, flag, cutoff, d_count):\n"
    "    return compact_vector_node_view_scope(o_row, ow_row, ptr, vec,"
    " wts, flag, cutoff, d_count)\n"
    "r = wrapc(O30[0], OW30[0], PTR30, VEC30, WTS30, FLAG30, 1.2, 2)\n"
    "signal.alarm(0)\n"
    "print('DONE-compact30', [float(x) for x in r[0][[0, 1, 2]]])\n")

CASE_H = HEADER + ARRS30 + ARRS + (
    "from urban_network_analysis.Engines import _sparse_access_scratch"
    " as SP\n"
    "r = SP.sparse_scope_access(O30, OW30, PTR30, VEC30, WTS30, FLAG30,"
    " D, DW, DWT, *MET, 1.2)\n"
    "signal.alarm(0)\n"
    "print('DONE-sparse-gate30', r.route, r.refusal)\n")

# gate ONLY (no fallback): proves the sparse gate refuses `values`
# before any sparse work on the same 30-node NaN-seed input.
CASE_I = HEADER + ARRS + ARRS30 + (
    "from urban_network_analysis.Engines import _sparse_access_scratch"
    " as SP\n"
    "print('GATE30', SP._sp_route_admits(PTR30, VEC30, WTS30, FLAG30,"
    " O30, OW30, D, DW, DWT, 1.2))\n")

results = []
tmpd = tempfile.mkdtemp(prefix="s12_pump_")
for tag, body, timeout in (
        ("driver-nan-seed", CASE_A, 30),
        ("a3kernel-nan-seed", CASE_B, 30),
        ("compact-nan-seed", CASE_C, 30),
        ("sparse-gate-nan-seed", CASE_D, 30),
        ("a3kernel-control-finite", CONTROL_B, 30),
        ("driver30-nan-seed", CASE_E, 60),
        ("a3kernel30-nan-seed", CASE_F, 60),
        ("compact30-nan-seed", CASE_G, 60),
        ("sparse-gate30-nan-seed", CASE_H, 60),
        ("sparse-gate30-ONLY-no-fallback", CASE_I, 30)):
    path = os.path.join(tmpd, f"{tag}.py")
    with open(path, "w") as f:
        f.write(body)
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = ("/storage/scratch1/1/dsynn6/campaign_data/"
                              "numba_cache_h01")
    try:
        pr = subprocess.run([sys.executable, path], capture_output=True,
                            text=True, timeout=timeout, env=env)
        if pr.returncode == 0:
            status = pr.stdout.strip().splitlines()[-1]
        else:
            err = [l for l in pr.stderr.splitlines() if l.strip()]
            status = f"exit={pr.returncode}: {err[-1][:180] if err else ''}"
    except subprocess.TimeoutExpired as e:
        status = f"TIMEOUT-{timeout}s (no DONE printed; inner 10s alarm " \
                 f"did not fire => noninterruptible native loop)"
    row = dict(probe=tag, status=status)
    results.append(row)
    print(json.dumps(row), flush=True)

with open("/storage/scratch1/1/dsynn6/una-x/.claude/worktrees/"
          "s12-review-r1/evidence/tasks/S12/raw/pump_attribution.json",
          "w") as f:
    json.dump(dict(
        description="NaN seed weights: (a) tiny ring+chord graph (ptr "
                    "[0,1,2,4], vec [1,2,0,0], wts [.5,.5,.5,.7], flags "
                    "all True, cutoff 1.2, ow [[0.0, nan]]) — original "
                    "driver, staged a3 kernel, compact kernel, sparse "
                    "gate, finite control; (b) the 30-node lattice-v2 "
                    "graph (default_rng(77) ring + 40 random arcs, wts "
                    "U(0.1,0.8), o [[3,4]], ow [[0.05, nan]], cutoff 1.2) "
                    "on which the driver was observed nonterminating with "
                    "unbounded RSS during the lattice probe.  Kernels "
                    "invoked inside @njit wrappers with per-origin ROW "
                    "args (engine call shape; passing the 2D block makes "
                    "o_terminal_weights[0] an array and heapq tuple "
                    "typing dies regardless of NaN).",
        graph=dict(ptr=PTR, vec=VEC, wts=WTS, flag=FLAG, cutoff=1.2,
                   o=[[0, 1]], ow_nan="[[0.0, nan]]"),
        graph30="default_rng(77): ring 0..29 + 40 random arcs, "
                "wts U(0.1,0.8), o [[3,4]], ow [[0.05, nan]], cutoff 1.2",
        results=results), f, indent=1)
print("WROTE pump_attribution.json")
