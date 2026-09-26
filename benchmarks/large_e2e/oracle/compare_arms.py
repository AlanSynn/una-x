"""Two-import-root arm comparison driver (H03 oracle; reusable as-is by
later candidate tasks — no modification needed).

Spawns tests/large_e2e/oracle/runner_arm.py once per arm, each in a
fresh subprocess with EXACTLY ONE import root bound (UNA_ORACLE_ARM_A /
UNA_ORACLE_ARM_B), a per-arm unique NUMBA_CACHE_DIR under the workdir
(no cache reuse across arms or runs), L1_REUSE_DIR removed, and a
neutral cwd. Then compares the two NPZs array-by-array:
dtype + shape + bytes. Any difference names the first differing key.

Defaults: BOTH roots are the immutable B0 tree — the H03 B0-vs-B0
determinism gate. For candidate comparisons pass --arm-a/--arm-b.

Usage:
  python compare_arms.py --workdir /tmp/h03_compare [--arm-a ROOT]
      [--arm-b ROOT] --report /path/report.json
Exit code 0 iff both runners succeed AND all arrays are bit-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ORACLE_DIR = Path("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
                  "tests/large_e2e/oracle")
B0_ROOT = "/Users/alansynn/orca/workspaces/una-x/wt-b0/src"
CAMPAIGN_PYTHON = ("/Users/alansynn/orca/workspaces/una-x/venvs/campaign/"
                   "bin/python")


def _key_digest(arr) -> str:
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def run_arm(arm_env_var, root, npz_path, json_path, cache_root, extra_env):
    env = dict(os.environ)
    env[arm_env_var] = root
    env["PYTHONPATH"] = root
    env["NUMBA_CACHE_DIR"] = cache_root
    env.pop("L1_REUSE_DIR", None)
    env.update(extra_env)
    proc = subprocess.run(
        [CAMPAIGN_PYTHON, str(ORACLE_DIR / "runner_arm.py"),
         "--out-npz", str(npz_path), "--out-json", str(json_path),
         "--runs", "2"],
        cwd="/tmp",  # neutral cwd — no repository shadowing
        env=env,
        capture_output=True, text=True, timeout=1800,
    )
    return proc


def compare_npz(path_a, path_b):
    import numpy as np
    za = np.load(path_a)
    zb = np.load(path_b)
    keys_a = set(za.files)
    keys_b = set(zb.files)
    if keys_a != keys_b:
        return False, {
            "error": "key sets differ",
            "only_in_a": sorted(keys_a - keys_b),
            "only_in_b": sorted(keys_b - keys_a),
        }
    mismatches = []
    digests = {}
    for key in sorted(keys_a):
        da = _key_digest(za[key])
        db = _key_digest(zb[key])
        digests[key] = {"a": da, "b": db, "equal": da == db}
        if da != db:
            mismatches.append(key)
    return (len(mismatches) == 0), {
        "n_keys": len(keys_a),
        "mismatched_keys": mismatches,
        "digests": digests,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--arm-a", default=B0_ROOT)
    ap.add_argument("--arm-b", default=B0_ROOT)
    ap.add_argument("--report", default=None)
    ap.add_argument("--label", default="b0_vs_b0")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "a_npz": workdir / "arm_a.npz", "a_json": workdir / "arm_a.json",
        "b_npz": workdir / "arm_b.npz", "b_json": workdir / "arm_b.json",
    }
    procs = {
        "a": run_arm("UNA_ORACLE_ARM_A", args.arm_a, paths["a_npz"],
                     paths["a_json"], workdir / "cache_a", {}),
        "b": run_arm("UNA_ORACLE_ARM_B", args.arm_b, paths["b_npz"],
                     paths["b_json"], workdir / "cache_b", {}),
    }
    report = {
        "label": args.label,
        "arm_a_root": args.arm_a,
        "arm_b_root": args.arm_b,
        "runners": {
            side: {
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            } for side, proc in procs.items()
        },
    }
    if any(p.returncode != 0 for p in procs.values()):
        report["verdict"] = "RUNNER_FAILURE"
        ok = False
    else:
        ok, comparison = compare_npz(paths["a_npz"], paths["b_npz"])
        report["comparison"] = comparison
        report["verdict"] = "BIT_IDENTICAL" if ok else "MISMATCH"
        with open(paths["a_json"]) as fh:
            report["arm_a_meta"] = json.load(fh)
        with open(paths["b_json"]) as fh:
            report["arm_b_meta"] = json.load(fh)

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    print(json.dumps({
        "verdict": report.get("verdict"),
        "mismatched_keys": report.get("comparison", {}).get("mismatched_keys"),
        "n_keys": report.get("comparison", {}).get("n_keys"),
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
