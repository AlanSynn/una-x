"""A1I admission recheck: re-derive the H05 A1 IR allocation-callsite
counts against THIS source tree, using the SAME evidence method as
evidence/H05/h05_probe_kernels.py (NRT alloc callsites parsed from
inspect_llvm; numba emits dynamic-size NRT_MemInfo_alloc_aligned
callsites, so "callsite count" = count of non-literal-size NRT alloc
calls, summed over compiled signatures).

Compile-only probe: tiny mini-arrays of the OBSERVED dtypes (int64
pointer/vector/terminals, float64 weights, bool flags) produce the same
jit signatures as the observed topology, so the LLVM (and therefore the
callsite count) is the observed specialization's. No campaign data is
loaded; wall-bounded; cache goes to a private temp dir.

Expected (H05 admissions.json, probe A1_002140.json at this src hash):
  integrated_scope_access  dynamic callsites == 30
  compact_vector_node_view_scope == 12
  reach_gravity_knn_access == 13
  adjust_destination_distances == 0
"""
import json
import os
import sys
import tempfile
import time

os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="a1i_ir_recheck_")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

RE = r"@NRT_[A-Za-z0-9_]*[Aa]lloc[A-Za-z0-9_]*\(\s*(?:i64\s+)?([^\s,)]+)"

EXPECTED = {
    "integrated_scope_access": 30,
    "compact_vector_node_view_scope": 12,
    "reach_gravity_knn_access": 13,
    "adjust_destination_distances": 0,
}


def nrt_callsites(disp):
    import re
    out = {"n_signatures": len(disp.signatures), "signatures": []}
    for i, sig in enumerate(disp.signatures):
        llvm = disp.inspect_llvm(sig)
        sizes, dynamic = [], 0
        for m in re.finditer(RE, llvm):
            tok = m.group(1)
            if tok.isdigit():
                sizes.append(int(tok))
            else:
                dynamic += 1
        out["signatures"].append({
            "index": i, "str(sig)": str(sig),
            "n_static_callsites": len(sizes),
            "n_dynamic_callsites": dynamic,
        })
    out["total_dynamic_callsites"] = sum(
        s["n_dynamic_callsites"] for s in out["signatures"])
    return out


def mini(dt, shape):
    import numpy as np
    a = np.zeros(shape, dtype=dt)
    return a


def main():
    import numpy as np
    sys.path.insert(0, "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/src")
    from urban_network_analysis.Engines import AccessibilityWElevation as AWE

    deadline = time.monotonic() + 300.0
    rec = {
        "probe": "A1I_admission_recheck",
        "method": "identical to evidence/H05/h05_probe_kernels.py:nrt_callsites "
                  "(dynamic NRT alloc callsites from inspect_llvm, summed over "
                  "compiled signatures)",
        "src_tree_expected": "a5883dddcef3afb8debdb4438a6338da886add6f",
        "kernels": {},
    }
    args = {
        "compact_vector_node_view_scope": lambda: (
            mini(np.int64, (2,)), mini(np.float64, (2,)),
            mini(np.int64, (4,)), mini(np.int64, (3,)),
            mini(np.float64, (3,)), mini(np.bool_, (3,)),
            1.0, 1),
        "adjust_destination_distances": lambda: (
            mini(np.float64, (5,)), mini(np.int64, (2, 2)),
            mini(np.float64, (2, 2)), 2),
        "reach_gravity_knn_access": lambda: (
            mini(np.float64, (4,)), mini(np.float64, (4,)), 1.0,
            0.001, 0.0, 500.0, 0.0,
            np.array([1.0, 1.0, 0.5]), 0.001, "logistic", 500.0, 1.0, 1.0),
        "integrated_scope_access": lambda: (
            mini(np.int64, (2, 2)), mini(np.float64, (2, 2)),
            mini(np.int64, (4,)), mini(np.int64, (3,)),
            mini(np.float64, (3,)), mini(np.bool_, (3,)),
            mini(np.int64, (1, 2)), mini(np.float64, (1, 2)),
            mini(np.float64, (2,)), 0.001, 0.0, 500.0, 1.0, "logistic",
            np.array([1.0, 1.0, 0.5]), 1.0),
    }
    for name, mk in args.items():
        disp = getattr(AWE, name)
        t0 = time.monotonic()
        disp(*mk())
        rec["kernels"][name] = nrt_callsites(disp)
        rec["kernels"][name]["compile_s"] = round(time.monotonic() - t0, 2)
        rec["kernels"][name]["expected_from_H05"] = EXPECTED[name]
        rec["kernels"][name]["match"] = (
            rec["kernels"][name]["total_dynamic_callsites"] == EXPECTED[name])
        if time.monotonic() > deadline:
            rec["truncated"] = True
            break
    rec["all_match"] = all(k.get("match") for k in rec["kernels"].values())
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ir_callsites_recheck.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: {"got": v["total_dynamic_callsites"],
                          "expected": v["expected_from_H05"],
                          "match": v["match"]}
                      for k, v in rec["kernels"].items()}, indent=1))
    print("wrote", out, "all_match:", rec["all_match"])


if __name__ == "__main__":
    main()
